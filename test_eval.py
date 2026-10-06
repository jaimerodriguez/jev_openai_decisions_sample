"""Offline contract fixtures and failure-path tests, not model evaluation."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import eval as app


def jev_response(route='rx_status'):
    return {'model': 'fixture-jev', 'answers': {'route': {
        'type': 'choice', 'choice': route, 'confidence': 1,
        'probabilities': {r: float(r == route) for r in app.ROUTES},
    }}, 'usage': {'input_tokens': 100, 'output_tokens': 5}}


def openai_response(route='rx_status'):
    return {'id': 'fixture-only', 'model': 'fixture-openai', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'route': route})}]}]}


def expected_fixed_accuracy():
    cases = app.load_cases(app.ROOT / 'cases.jsonl')
    return sum(c['expected'] == 'rx_status' for c in cases) / len(cases)


def decisions_response(route='rx_status'):
    # Shape from the official Decisions guide. No id/status is required.
    return {'model': 'gpt-6-luna', 'answers': [{
        'type': 'choice', 'name': 'route', 'choice': route, 'confidence': 1,
        'probabilities': [{'value': r, 'probability': float(r == route)} for r in app.ROUTES],
    }], 'usage': {'input_tokens': 100, 'output_tokens': 0, 'total_tokens': 100}}


class Contracts(unittest.TestCase):
    def test_curl_script_builds_real_payload_and_checks_key(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_curl = Path(directory) / 'curl'
            fake_curl.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake_curl.chmod(0o755)
            env = dict(app.os.environ, PATH=directory + app.os.pathsep + str(Path(app.sys.executable).parent) + app.os.pathsep + app.os.environ['PATH'],
                       OPENAI_API_KEY='fixture-only', PYTHONDONTWRITEBYTECODE='1')
            transcript = 'Is my refill ready? "Do not reorder."'
            result = subprocess.run(['bash', str(app.ROOT / 'openai_endpoint.sh'), transcript],
                                    env=env, capture_output=True, text=True, check=True)
            args = json.loads(result.stdout)
            self.assertIn('https://api.openai.com/v1/decisions', args)
            payload = json.loads(args[args.index('--data-binary') + 1])
            self.assertEqual(payload, app.payload_for('openai-decisions', 'gpt-6-luna', transcript))
            for key in (None, ''):
                if key is None:
                    env.pop('OPENAI_API_KEY', None)
                else:
                    env['OPENAI_API_KEY'] = key
                result = subprocess.run(['bash', str(app.ROOT / 'openai_endpoint.sh')],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn('OPENAI_API_KEY', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_decisions_request_contract_and_shared_policy(self):
        endpoint, key, model = app.PROVIDERS['openai-decisions']
        self.assertEqual((endpoint, key, model),
                         ('https://api.openai.com/v1/decisions', 'OPENAI_API_KEY', 'gpt-6-luna'))
        payload = app.payload_for('openai-decisions', model, 'caller transcript')
        self.assertEqual(set(payload), {'model', 'input', 'questions'})
        self.assertEqual(json.loads(payload['input']), {'transcript': 'caller transcript'})
        self.assertEqual(len(payload['questions']), 1)
        question = payload['questions'][0]
        self.assertEqual(question['name'], 'route')
        self.assertEqual(question['type'], 'choice')
        self.assertEqual(question['instructions'], app.POLICY)
        self.assertEqual({c['value']: c['description'] for c in question['choices']}, app.ROUTES)

    def test_decisions_normalizes_probabilities_by_value(self):
        body = decisions_response()
        body['answers'][0]['probabilities'].reverse()
        parsed = app.parse_result('openai-decisions', body)
        self.assertEqual(parsed['route'], 'rx_status')
        self.assertEqual(parsed['probabilities'], {r: float(r == 'rx_status') for r in app.ROUTES})
        self.assertEqual(parsed['model_returned'], 'gpt-6-luna')
        self.assertEqual(parsed['usage']['input_tokens'], 100)
        self.assertIsNone(parsed['request_id'])

    def test_decisions_rejects_mismatched_and_malformed_answers(self):
        bodies = [{'answers': []}, {'answers': {}},
                  {'answers': [decisions_response()['answers'][0]] * 2}]
        for field, value in [('name', 'wrong'), ('type', 'predicate'), ('choice', 'unknown'),
                             ('confidence', float('nan')), ('confidence', True), ('probabilities', {})]:
            body = decisions_response()
            body['answers'][0][field] = value
            bodies.append(body)
        for entries in ([{'value': 'rx_status', 'probability': 1}] * 2,
                        [{'value': 'rx_status', 'probability': 1}],
                        [{'value': r, 'probability': 0} for r in app.ROUTES]):
            body = decisions_response()
            body['answers'][0]['probabilities'] = entries
            bodies.append(body)
        for bad in (float('nan'), -1, 2, True):
            body = decisions_response()
            body['answers'][0]['probabilities'][0]['probability'] = bad
            bodies.append(body)
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                app.parse_result('openai-decisions', body)

    def test_decisions_refusal_counts_as_failure_and_human_fallback(self):
        refusal = {'answers': [{'type': 'refusal', 'name': 'route'}]}
        with self.assertRaisesRegex(RuntimeError, 'refused'):
            app.parse_result('openai-decisions', refusal)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture'}, clear=True), \
                 patch.object(app, 'post_json', return_value=refusal), \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'openai-decisions', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 1)
            report = json.loads((out / 'summary.json').read_text())['providers']['openai-decisions']
            n = len(app.load_cases(app.ROOT / 'cases.jsonl'))
            self.assertEqual(report['errors'], n)
            self.assertEqual(report['accuracy_all_attempts'], 0)
            self.assertEqual(report['disposition_counts'], {'human_queue_on_error': n})

    def test_decisions_default_runner_and_separate_model_flags(self):
        def fake_post(url, key, payload, *args, **kwargs):
            if url.endswith('/decisions'):
                self.assertEqual(payload['model'], 'gpt-6-luna')
                self.assertEqual(key, 'fixture-openai')
                return decisions_response()
            self.assertEqual(url, app.PROVIDERS['jev'][0])
            return jev_response()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture-openai', 'TYPESAFE_API_KEY': 'fixture-jev'}, clear=True), \
                 patch.object(app, 'post_json', side_effect=fake_post) as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--openai-model', 'responses-only', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 0)
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(set(report['providers']), {'jev', 'openai-decisions'})
            self.assertEqual(post.call_count, 2 * len(app.load_cases(app.ROOT / 'cases.jsonl')))
            for summary in report['providers'].values():
                self.assertEqual(summary['accuracy_all_attempts'], expected_fixed_accuracy())
                self.assertGreater(summary['brier_sample_count'], 0)

    def test_decisions_model_override_and_dry_run_without_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.dict(app.os.environ, {}, clear=True), \
                 patch.object(app, 'post_json') as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--dry-run', '--providers', 'openai-decisions',
                                              '--decisions-model', 'fixture-model', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 0)
                post.assert_not_called()
            requests = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['payload']['model'] == 'fixture-model' for r in requests))

    def test_decisions_missing_or_empty_key_fails_before_network(self):
        for environment in ({}, {'OPENAI_API_KEY': ''}):
            with patch.dict(app.os.environ, environment, clear=True), \
                 patch.object(app, 'post_json') as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'openai-decisions']), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    app.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn('OPENAI_API_KEY', stderr.getvalue())
                post.assert_not_called()

    def test_expanded_dataset_rejects_duplicate_ids(self):
        cases = app.load_cases(app.ROOT / 'cases.jsonl')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.jsonl'
            path.write_text('\n'.join(json.dumps(c) for c in [*cases, cases[0]]))
            with self.assertRaisesRegex(ValueError, 'unique ID'):
                app.load_cases(path)

    def test_openrouter_contract_and_metadata(self):
        endpoint, key_name, model = app.PROVIDERS['jev-openrouter']
        self.assertEqual(endpoint, 'https://openrouter.ai/api/alpha/decisions')
        self.assertEqual(key_name, 'OPENROUTER_API_KEY')
        self.assertEqual(model, 'typesafe/jev-1.13')
        self.assertEqual(app.payload_for('jev-openrouter', model, 'transcript'),
                         app.payload_for('jev', model, 'transcript'))
        response = jev_response()
        response.update(id='fixture-openrouter', provider='TypeSafe',
                        model='typesafe/jev-1.13-20260917')
        response['usage']['cost'] = 0.000019992
        parsed = app.parse_result('jev-openrouter', response)
        self.assertEqual(parsed['route'], 'rx_status')
        self.assertEqual(parsed['request_id'], 'fixture-openrouter')
        self.assertEqual(parsed['provider_returned'], 'TypeSafe')
        self.assertEqual(parsed['model_returned'], response['model'])
        self.assertEqual(parsed['usage']['cost'], 0.000019992)
        self.assertIsNone(parsed['credits_used'])

    def test_openrouter_rejects_invalid_answers_and_error_envelopes(self):
        invalid = jev_response()
        invalid['answers']['route']['probabilities']['rx_status'] = float('nan')
        for response in (invalid, {'error': {'code': 402, 'message': 'fixture'}}):
            with self.assertRaises((ValueError, KeyError)):
                app.parse_result('jev-openrouter', response)

    def test_openrouter_missing_or_empty_key_fails_before_network(self):
        for environment in ({}, {'OPENROUTER_API_KEY': ''}):
            with patch.dict(app.os.environ, environment, clear=True), \
                 patch.object(app, 'post_json') as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'jev-openrouter']), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    app.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn('OPENROUTER_API_KEY', stderr.getvalue())
                post.assert_not_called()

    def test_openrouter_runner_uses_own_key_and_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.dict(app.os.environ, {'OPENROUTER_API_KEY': 'fixture-or'}, clear=True), \
                 patch.object(app, 'post_json', side_effect=lambda *args, **kwargs: jev_response()) as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'jev-openrouter',
                                              '--openrouter-model', 'fixture-model', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 0)
            self.assertEqual(post.call_count, len(app.load_cases(app.ROOT / 'cases.jsonl')))
            for call in post.call_args_list:
                self.assertEqual(call.args[0], 'https://openrouter.ai/api/alpha/decisions')
                self.assertEqual(call.args[1], 'fixture-or')
                self.assertEqual(call.args[2]['model'], 'fixture-model')
                self.assertIsNone(call.args[4])
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['models_requested']['jev-openrouter'], 'fixture-model')
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['providers']['jev-openrouter']['accuracy_all_attempts'], expected_fixed_accuracy())

    def test_dataset_accepts_variable_size_and_partial_route_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.jsonl'
            for size in (1, 7, 1001):
                cases = [{'id': str(i), 'prompt': 'Is my prescription ready?', 'expected': 'rx_status'}
                         for i in range(size)]
                path.write_text('\n'.join(json.dumps(c) for c in cases))
                with self.subTest(size=size):
                    self.assertEqual(app.load_cases(path), cases)

    def test_dataset_rejects_empty_or_blank_only_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.jsonl'
            for content in ('', '\n  \n'):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, 'at least one case'):
                    app.load_cases(path)

    def test_custom_single_case_runner_uses_actual_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path, out = Path(directory) / 'cases.jsonl', Path(directory) / 'run'
            path.write_text(json.dumps({'id': 'single', 'prompt': 'Is it ready?', 'expected': 'rx_status'}))
            with patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture'}, clear=True), \
                 patch.object(app, 'post_json', return_value=decisions_response()) as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'openai-decisions',
                                              '--cases', str(path), '--repeat', '2', '--out', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 0)
            self.assertEqual(post.call_count, 2)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['dataset_case_count'], 1)
            report = json.loads((out / 'summary.json').read_text())['providers']['openai-decisions']
            self.assertEqual(report['attempts'], 2)
            self.assertEqual(report['accuracy_all_attempts'], 1)
            self.assertEqual(report['per_class']['emergency']['support'], 0)

    def test_no_labels_leak_into_requests(self):
        case = {'prompt': 'TRANSCRIPT', 'expected': 'SECRET_LABEL', 'why': 'SECRET_RATIONALE'}
        for provider in app.PROVIDERS:
            body = app.payload_for(provider, 'fixture', case['prompt'])
            serialized = json.dumps(body)
            self.assertIn('TRANSCRIPT', serialized)
            self.assertNotIn('SECRET_', serialized)

    def test_same_policy_and_options(self):
        jev = app.payload_for('jev', 'fixture', 'test')
        oa = app.payload_for('openai-responses', 'fixture', 'test')
        self.assertEqual(jev['questions']['route']['instructions'], app.POLICY)
        self.assertTrue(oa['input'][0]['content'].startswith(app.POLICY))
        self.assertEqual(set(oa['text']['format']['schema']['properties']['route']['enum']), set(jev['questions']['route']['criteria']))

    def test_direct_and_gateway_wrappers(self):
        direct = jev_response()
        gateway = {'code': 0, 'data': {'result': direct, 'creditsUsed': 1}}
        self.assertEqual(app.parse_result('jev', direct)['route'], 'rx_status')
        self.assertEqual(app.parse_result('jev-gateway', gateway)['credits_used'], 1)
        with self.assertRaises(KeyError):
            app.parse_result('jev', gateway)
        with self.assertRaises(ValueError):
            app.parse_result('jev-gateway', {'code': 2})

    def test_bad_distributions_and_choice_are_rejected(self):
        for bad_value in (float('nan'), float('inf'), -0.1, 2, True):
            response = jev_response()
            response['answers']['route']['probabilities']['rx_status'] = bad_value
            with self.assertRaises(ValueError):
                app.parse_result('jev', response)
        response = jev_response()
        response['answers']['route']['choice'] = 'refill_request'
        with self.assertRaises(ValueError):
            app.parse_result('jev', response)

    def test_openai_completion_refusal_and_truncation(self):
        self.assertEqual(app.parse_result('openai-responses', openai_response())['route'], 'rx_status')
        incomplete = openai_response()
        incomplete['status'] = 'incomplete'
        refusal = openai_response()
        refusal['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'fixture'}]
        malformed = openai_response()
        malformed['output'][0]['content'][0]['text'] = '{'
        for response in (incomplete, refusal, malformed, openai_response('invented_route')):
            with self.assertRaises(ValueError):
                app.parse_result('openai-responses', response)

    def test_errors_cannot_inflate_accuracy(self):
        rows = [
            {'expected': 'rx_status', 'route': 'rx_status', 'latency_ms': 10, 'disposition': app.handoff('rx_status')},
            {'expected': 'representative', 'route': 'refill_request', 'latency_ms': 20, 'disposition': app.handoff('refill_request')},
            {'expected': 'emergency', 'route': None, 'error': 'timeout', 'latency_ms': 30, 'disposition': app.handoff(None)},
        ]
        result = app.summarize(rows)
        self.assertAlmostEqual(result['accuracy_all_attempts'], 1 / 3)
        self.assertEqual(result['accuracy_valid_only'], .5)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['protected_cases_sent_to_automation'], 1)
        self.assertEqual(result['emergency_route_misses_including_errors'], 1)
        self.assertEqual(result['per_class']['emergency']['recall'], 0)
        self.assertEqual(result['confusion_matrix']['emergency'], {'__error__': 1})

    def test_disposition_boundary(self):
        self.assertEqual(app.handoff('provider_new_rx'), 'human_queue')
        self.assertEqual(app.handoff('emergency'), 'emergency_protocol')
        self.assertEqual(app.handoff(None), 'human_queue_on_error')

    def test_full_runner_uses_provider_responses_not_expected_labels(self):
        # Deliberately return one fixed label for every input. Perfect accuracy
        # here would reveal label leakage or scoring the expected output itself.
        def fake_post(url, *args, **kwargs):
            return openai_response() if 'openai.com' in url else copy.deepcopy(jev_response())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch.dict(app.os.environ, {'TYPESAFE_API_KEY': 'fixture', 'OPENAI_API_KEY': 'fixture'}), \
                 patch.object(app, 'post_json', side_effect=fake_post) as post, \
                 patch.object(app.sys, 'argv', ['eval.py', '--providers', 'jev', 'openai-responses', '--out', str(out)]):
                self.assertEqual(app.main(), 0)
            self.assertEqual(post.call_count, 2 * len(app.load_cases(app.ROOT / 'cases.jsonl')))
            result = json.loads((out / 'summary.json').read_text())
            for provider in ('jev', 'openai-responses'):
                self.assertEqual(result['providers'][provider]['accuracy_all_attempts'], expected_fixed_accuracy())
            self.assertEqual(len((out / 'results.jsonl').read_text().splitlines()), post.call_count)


if __name__ == '__main__':
    unittest.main()
