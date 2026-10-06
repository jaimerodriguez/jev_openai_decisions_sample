"""Offline timing tests: controlled clocks, streaming transports, no provider calls."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval as app
import latency
from test_eval import decisions_response, jev_response


class Clock:
    def __init__(self):
        self.value = 10 ** 20  # Large base confirms integer subtraction, not float timestamps.

    def __call__(self):
        return self.value

    def advance(self, ns):
        self.value += ns


class BodyStream(latency.httpx.SyncByteStream):
    def __init__(self, clock, data=b'{"ok":true}', fail=False):
        self.clock, self.data, self.fail = clock, data, fail

    def __iter__(self):
        self.clock.advance(30_000_000)
        if self.fail:
            raise latency.httpx.ReadTimeout('PRIVATE ERROR BODY')
        yield self.data


class Timing(unittest.TestCase):
    def test_warmup_failure_remains_visible_but_does_not_pollute_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, out = root / 'cases.jsonl', root / 'run'
            cases.write_text(json.dumps({'id': 'one', 'prompt': 'Is it ready?', 'expected': 'rx_status'}))
            with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture'}, clear=True), \
                 patch.object(app, 'post_json', side_effect=[RuntimeError('HTTP 429'), decisions_response()]), \
                 patch.object(app.sys, 'argv', ['eval.py', '--cases', str(cases), '--providers', 'openai-decisions',
                                              '--warmup', '1', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 1)
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['warmup']['openai-decisions'], {'attempts': 1, 'errors': 1})
            self.assertEqual(report['providers']['openai-decisions']['errors'], 0)
            self.assertEqual(report['providers']['openai-decisions']['accuracy_all_attempts'], 1)
            self.assertIn('1 calls, 1 errors', (out / 'latency.md').read_text())

    def test_network_timer_excludes_preparation_and_decoding(self):
        clock, metrics = Clock(), {}
        def handler(request):
            clock.advance(20_000_000)
            return latency.httpx.Response(200, stream=BodyStream(clock), headers={'x-request-id': 'fixture'})
        with latency.httpx.Client(transport=latency.httpx.MockTransport(handler)) as client:
            build, decode = client.build_request, json.loads
            def slow_build(*args, **kwargs):
                clock.advance(5_000_000)
                return build(*args, **kwargs)
            def slow_decode(*args, **kwargs):
                clock.advance(7_000_000)
                return decode(*args, **kwargs)
            with patch.object(client, 'build_request', side_effect=slow_build), \
                 patch.object(latency.json, 'loads', side_effect=slow_decode):
                result = latency.post_json('https://example.test/api', 'fixture', {}, 30,
                                           client=client, metrics=metrics, clock=clock)
        self.assertEqual(result, {'ok': True})
        self.assertEqual(metrics['request_prepare_ns'], 5_000_000)
        self.assertEqual(metrics['time_to_headers_ns'], 20_000_000)
        self.assertEqual(metrics['api_latency_ns'], 50_000_000)
        self.assertEqual(metrics['http_elapsed_ns'], 50_000_000)
        self.assertEqual(metrics['json_decode_ns'], 7_000_000)
        self.assertEqual(metrics['http_request_id'], 'fixture')

    def test_timeout_preserves_elapsed_but_not_completed_latency(self):
        clock, metrics, calls = Clock(), {}, []
        def handler(request):
            calls.append(request)
            clock.advance(20_000_000)
            return latency.httpx.Response(200, stream=BodyStream(clock, fail=True))
        with latency.httpx.Client(transport=latency.httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, 'timeout') as error:
                latency.post_json('https://example.test/api', 'fixture', {}, 30,
                                  client=client, metrics=metrics, clock=clock)
        self.assertNotIn('PRIVATE', str(error.exception))
        self.assertEqual(len(calls), 1)
        self.assertEqual(metrics['error_kind'], 'timeout')
        self.assertIsNone(metrics['api_latency_ns'])
        self.assertEqual(metrics['http_elapsed_ns'], 50_000_000)

    def test_http_errors_redirects_and_bad_json_are_not_successes(self):
        for status, content, kind in [(429, b'private body', 'http_status'),
                                      (302, b'', 'http_status'), (200, b'not json', 'json_decode')]:
            clock, metrics, calls = Clock(), {}, []
            def handler(request):
                calls.append(request)
                clock.advance(10_000_000)
                return latency.httpx.Response(status, content=content, headers={'location': 'https://elsewhere.test'})
            with latency.httpx.Client(transport=latency.httpx.MockTransport(handler), follow_redirects=False) as client:
                with self.assertRaises(RuntimeError):
                    latency.post_json('https://example.test/api', 'fixture', {}, 30,
                                      client=client, metrics=metrics, clock=clock)
            self.assertEqual(len(calls), 1)
            self.assertEqual(metrics['error_kind'], kind)
            self.assertEqual(metrics['api_latency_ns'], 10_000_000)
            metrics.update(error='failed', route=None)
            report = latency.timing_summary([metrics])
            self.assertEqual(report['api_ms_valid']['n'], 0)
            self.assertEqual(report['failed_attempt_elapsed_ms']['n'], 1)

    def test_connection_policy(self):
        for mode, kept in [('reuse', 1), ('fresh', 0)]:
            with patch.object(latency.httpx, 'Client') as client:
                latency.make_client(12, mode)
            config = client.call_args.kwargs
            self.assertEqual(config['limits'].max_keepalive_connections, kept)
            self.assertEqual(config['limits'].max_connections, 1)
            self.assertFalse(config['follow_redirects'])
            self.assertFalse(config['http2'])
            self.assertEqual(config['timeout'].read, 12)

    def test_nearest_rank_and_empty_distributions(self):
        result = latency.distribution_ms([i * 1_000_000 for i in range(1, 101)])
        self.assertEqual(result['p50'], 50)
        self.assertEqual(result['p95'], 95)
        self.assertEqual(result['p99'], 99)
        self.assertEqual(result['mean'], 50.5)
        self.assertIsNone(latency.distribution_ms([])['p95'])
        self.assertIsNone(latency.distribution_ms([1])['stdev'])

    def test_paired_differences_and_case_bootstrap(self):
        pairs = [({'id': str(i), 'api_latency_ns': 30_000_000},
                  {'id': str(i), 'api_latency_ns': 10_000_000}) for i in range(3)]
        pairs.append(({'id': 'timeout', 'api_latency_ns': None}, {'api_latency_ns': 5}))
        report = latency.paired_timing(pairs)
        self.assertEqual(report['matched_pairs'], 3)
        self.assertEqual(report['mean_case_delta_ms'], 20)
        self.assertEqual(report['median_first_over_second_ratio'], 3)
        self.assertEqual(report['case_bootstrap_95pct_ci_ms'], [20, 20])
        repeated = latency.paired_timing([pairs[0]] * 10)
        self.assertEqual(repeated['distinct_cases'], 1)
        self.assertIsNone(repeated['case_bootstrap_95pct_ci_ms'])

    def test_warmup_preserves_measured_order(self):
        cases = [{'id': str(i)} for i in range(5)]
        a = list(latency.schedule(cases, ['a', 'b'], 2, 0, 42))
        b = list(latency.schedule(cases, ['a', 'b'], 2, 3, 42))
        self.assertEqual(a, [job for job in b if job[0] == 'measured'])
        self.assertEqual(sum(job[0] == 'warmup' for job in b), 6)

    def test_runner_logs_warmups_but_excludes_them_from_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, out = root / 'cases.jsonl', root / 'run'
            cases.write_text(json.dumps({'id': 'one', 'prompt': 'Is it ready?', 'expected': 'rx_status'}))
            calls = []
            def fake_post(url, *args, metrics, client):
                calls.append((url, client))
                ns = 100_000_000 if 'typesafe' in url else 200_000_000
                metrics.update(api_latency_ns=ns, http_elapsed_ns=ns, http_status=200)
                return jev_response() if 'typesafe' in url else decisions_response()
            with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture', 'TYPESAFE_API_KEY': 'fixture'}, clear=True), \
                 patch.object(app, 'post_json', side_effect=fake_post), \
                 patch.object(app.sys, 'argv', ['eval.py', '--cases', str(cases), '--repeat', '2',
                                              '--warmup', '1', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 0)
            self.assertEqual(len(calls), 6)
            # Each provider has one persistent client; the two providers never share it.
            self.assertEqual(len({id(client) for url, client in calls}), 2)
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['warmup']['jev']['attempts'], 1)
            self.assertEqual(report['providers']['jev']['attempts'], 2)
            self.assertEqual(report['providers']['jev']['timing']['api_ms_valid']['mean'], 100)
            paired = report['paired']['jev_vs_openai-decisions']['api_latency_comparison']
            self.assertEqual(paired['matched_pairs'], 2)
            self.assertEqual(paired['mean_case_delta_ms'], -100)
            with (out / 'latency.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(sum(r['phase'] == 'warmup' for r in rows), 2)
            self.assertIn('| jev | 2 | 0 / 2 | 100.00 |', (out / 'latency.md').read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['timing_schema_version'], 2)


if __name__ == '__main__':
    unittest.main()
