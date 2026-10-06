"""Client-observed API timing and paired comparisons (not server inference time)."""
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack

import httpx


def make_client(timeout=30, connection_mode='reuse'):
    # HTTP/1.1 for every provider, one in-flight call, no automatic retries.
    # The default HTTPX transport has retries=0. Keep standard proxy/TLS env support.
    return httpx.Client(
        timeout=httpx.Timeout(timeout), follow_redirects=False, http2=False,
        limits=httpx.Limits(max_connections=1,
                           max_keepalive_connections=1 if connection_mode == 'reuse' else 0,
                           keepalive_expiry=60),
    )


def post_json(url, key, payload, timeout, idempotency_key=None, *, client=None, metrics=None, clock=None):
    """Time full HTTP exchange separately from preparation and JSON decoding.

    Metrics are populated even on failure. A timeout has elapsed time, but no
    completed API latency. No body text, credentials or arbitrary headers logged.
    """
    metrics = metrics if metrics is not None else {}
    clock = clock or time.perf_counter_ns
    metrics.update(api_latency_ns=None, http_elapsed_ns=None, time_to_headers_ns=None,
                   json_decode_ns=None, http_status=None, error_kind=None)
    with ExitStack() as stack:
        client = client or stack.enter_context(make_client(timeout))
        prep_start = clock()
        headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        if idempotency_key:
            headers['Idempotency-Key'] = idempotency_key
        request = client.build_request('POST', url, headers=headers,
                                       content=json.dumps(payload).encode(), timeout=timeout)
        metrics['request_prepare_ns'] = clock() - prep_start
        start = clock()
        response = None
        try:
            response = client.send(request, stream=True)
            headers_at = clock()
            metrics['time_to_headers_ns'] = headers_at - start
            metrics['http_status'] = response.status_code
            raw = response.read()
            # Capture immediately after full body read, before metadata/JSON work.
            finished_at = clock()
            metrics['api_latency_ns'] = finished_at - start
            metrics['http_elapsed_ns'] = finished_at - start
            metrics['http_version'] = response.http_version
            metrics['http_request_id'] = response.headers.get('x-request-id')
            metrics['response_bytes'] = len(raw)
            if not 200 <= response.status_code < 300:
                metrics['error_kind'] = 'http_status'
                raise RuntimeError(f'HTTP {response.status_code}; no automatic retry')
        except httpx.TimeoutException:
            metrics['error_kind'] = 'timeout'
            raise RuntimeError('Network timeout; outcome unknown; no automatic retry') from None
        except httpx.RequestError:
            metrics['error_kind'] = 'transport'
            raise RuntimeError('Network failure; outcome unknown; no automatic retry') from None
        finally:
            if metrics['http_elapsed_ns'] is None:
                metrics['http_elapsed_ns'] = clock() - start
            if response is not None:
                response.close()
        decode_start = clock()
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            metrics['error_kind'] = 'json_decode'
            raise RuntimeError('Response JSON decoding failed') from None
        finally:
            metrics['json_decode_ns'] = clock() - decode_start


def distribution_ms(values_ns):
    """Nearest-rank empirical percentiles. Raw integers stay in result rows."""
    values = sorted(v / 1_000_000 for v in values_ns if v is not None)
    n = len(values)
    result = {'n': n, 'mean': None, 'stdev': None, 'min': None, 'max': None,
              'p50': None, 'p90': None, 'p95': None, 'p99': None}
    if n:
        result.update(mean=statistics.fmean(values), min=values[0], max=values[-1],
                      stdev=statistics.stdev(values) if n > 1 else None)
        for name, q in [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99)]:
            result[name] = values[max(0, math.ceil(n * q) - 1)]
    return result


def timing_summary(rows):
    valid = [r for r in rows if r.get('route') is not None and not r.get('error')]
    errors = [r for r in rows if r.get('error')]
    summary = {
        'api_ms_valid': distribution_ms([r.get('api_latency_ns') for r in valid]),
        'api_ms_http_2xx': distribution_ms([r.get('api_latency_ns') for r in rows
                                          if 200 <= (r.get('http_status') or 0) < 300]),
        'failed_attempt_elapsed_ms': distribution_ms([r.get('http_elapsed_ns') for r in errors]),
        'decision_ms_valid': distribution_ms([r.get('decision_latency_ns') for r in valid]),
        'headers_ms_valid': distribution_ms([r.get('time_to_headers_ns') for r in valid]),
        'json_decode_ms_valid': distribution_ms([r.get('json_decode_ns') for r in valid]),
        'timeouts': sum(r.get('error_kind') == 'timeout' for r in rows),
    }
    summary['tail_sample_warning'] = summary['api_ms_valid']['n'] < 100
    return summary


def paired_timing(pairs):
    """A-B differences on matched valid case/repetition pairs only.

    Bootstrap resamples cases, not repetitions. This avoids treating repeats of
    one prompt as independent workload examples. CI describes sampled cases only.
    """
    pairs = [(a, b) for a, b in pairs if a.get('api_latency_ns') is not None
             and b.get('api_latency_ns') is not None]
    deltas = [a['api_latency_ns'] - b['api_latency_ns'] for a, b in pairs]
    by_case = defaultdict(list)
    for (a, b), delta in zip(pairs, deltas):
        by_case[a['id']].append(delta / 1_000_000)
    case_means = [statistics.fmean(v) for v in by_case.values()]
    interval = None
    if len(case_means) >= 2:
        rng = random.Random(42)
        estimates = sorted(statistics.fmean(rng.choices(case_means, k=len(case_means))) for _ in range(2000))
        interval = [estimates[49], estimates[1949]]
    ratios = [a['api_latency_ns'] / b['api_latency_ns'] for a, b in pairs if b['api_latency_ns'] > 0]
    return {
        'matched_pairs': len(pairs), 'distinct_cases': len(by_case),
        'delta_first_minus_second_ms': distribution_ms(deltas),
        'mean_case_delta_ms': statistics.fmean(case_means) if case_means else None,
        'case_bootstrap_95pct_ci_ms': interval,
        'median_first_over_second_ratio': statistics.median(ratios) if ratios else None,
        'second_faster_pairs': sum(d > 0 for d in deltas),
        'first_faster_pairs': sum(d < 0 for d in deltas), 'ties': sum(d == 0 for d in deltas),
    }


def schedule(cases, providers, repeat, warmup, seed):
    # Separate RNGs keep measured order stable when warm-up count changes.
    for phase, rng in [('warmup', random.Random(seed + 1)), ('measured', random.Random(seed))]:
        for repetition in range(warmup if phase == 'warmup' else repeat):
            order = [cases[repetition % len(cases)]] if phase == 'warmup' else cases.copy()
            rng.shuffle(order)
            for case in order:
                provider_order = providers.copy()
                rng.shuffle(provider_order)
                for provider in provider_order:
                    yield phase, repetition, case, provider


def write_latency_csv(path, rows):
    fields = ['provider', 'id', 'repetition', 'phase', 'sequence', 'started_at_utc',
              'model_returned', 'http_status', 'http_version', 'route', 'expected', 'error_kind',
              'api_latency_ns', 'http_elapsed_ns', 'time_to_headers_ns',
              'request_prepare_ns', 'json_decode_ns', 'validation_ns', 'decision_latency_ns']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def comparison_markdown(report):
    lines = ['# API latency comparison', '',
             'Client-observed full-response latency in milliseconds; measured valid calls only.',
             'Includes network and service time. Excludes payload preparation, JSON decoding and validation.', '',
             '| Provider | Valid timed calls | Errors / attempts | p50 ms | p95 ms | Mean ms |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return f'{value:.2f}' if value is not None else '—'
    for provider, m in report['providers'].items():
        timing = m['timing']['api_ms_valid']
        lines.append(f"| {provider} | {timing['n']} | {m['errors']} / {m['attempts']} | {fmt(timing['p50'])} | {fmt(timing['p95'])} | {fmt(timing['mean'])} |")
    lines.extend(['', 'Paired differences use the same case and repetition. Positive A−B means B was faster.', ''])
    for name, m in report['paired'].items():
        p = m['api_latency_comparison']
        lines.append(f"- {name}: {p['matched_pairs']} matched pairs, {p['distinct_cases']} distinct cases; mean case A−B = {fmt(p['mean_case_delta_ms'])} ms.")
    for provider, counts in report.get('warmup', {}).items():
        if counts['attempts']:
            lines.append(f"- {provider} warm-ups: {counts['attempts']} calls, {counts['errors']} errors (excluded above).")
    lines.extend(['', 'Warm-up calls are excluded above but retained in results.jsonl and latency.csv.',
                  'With fewer than 100 timed samples, tail percentiles are especially unstable.',
                  'Compare accuracy and errors too; inspect summary.json for failed-call timings and paired confidence intervals.', ''])
    return '\n'.join(lines)
