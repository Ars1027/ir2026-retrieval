"""Protocol regression tests; all documents and answers here are synthetic."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import test_pipeline
from ir2026 import (
    BudgetExceeded, CorpusTools, DataError, POLICY, PROTOCOL_VERSION,
    count_model_tool_calls, read_jsonl, validate_results, validate_run_info, write_jsonl,
)

PROJECT = Path(__file__).resolve().parents[1]


class MethodProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_pipeline.ProjectTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.config = {'release': 'IR2026-BCP-200-v1', 'limits': dict(POLICY)}
        self.info = {
            **self.config, 'protocol_version': PROTOCOL_VERSION, 'method': 'bm25-v1',
            'method_type': 'retrieval', 'model': None, 'mode': 'coursework', 'split': 'dev',
        }

    def tearDown(self):
        self.fixture.tearDown()

    def test_non_model_submission_and_embedding_pipeline(self):
        self.assertEqual(validate_run_info(self.info, self.config, 'dev', True), [])
        self.info['model'] = 'embedding-model/version'
        self.assertEqual(validate_run_info(self.info, self.config, 'dev', True), [])

    def test_agent_requires_actual_model_identifier(self):
        self.info['method_type'] = 'agent'
        self.assertTrue(validate_run_info(self.info, self.config, 'dev', True))
        self.info['model'] = 'model/version'
        self.assertEqual(validate_run_info(self.info, self.config, 'dev', True), [])

    def test_demo_cannot_be_submitted_by_changing_only_mode(self):
        for field, value in [('mode', 'demo'), ('agent_module', 'examples.keyword_agent'),
                             ('model', 'offline-keyword-demo')]:
            info = {**self.info, field: value}
            self.assertTrue(validate_run_info(info, self.config, 'dev', True))

    def test_release_limits_split_and_protocol_are_checked(self):
        for field, value in [('release', 'wrong'), ('split', 'test'), ('limits', {}),
                             ('protocol_version', 'unknown'), ('method', ' '),
                             ('method_type', 'unknown'), ('model', {}), ('mode', 'other')]:
            with self.subTest(field=field):
                self.assertTrue(validate_run_info({**self.info, field: value}, self.config, 'dev'))
        self.assertTrue(validate_run_info([], self.config, 'dev'))

    def test_legacy_demo_metadata_still_readable(self):
        info = {**self.config, 'mode': 'demo', 'split': 'dev',
                'agent_module': 'examples.keyword_agent', 'model': 'offline-keyword-demo'}
        self.assertEqual(validate_run_info(info, self.config, 'dev'), [])
        self.assertTrue(validate_run_info(info, self.config, 'dev', True))

    def retrieval_record(self, count=1, output=None):
        tools = CorpusTools(self.fixture.corpus, 'q1', method_type='retrieval',
                            trace_path=self.root / 'traces/q1.jsonl')
        for _ in range(count):
            tools.record_retrieval('rank', {'query': 'Orion'}, output or {'doc_ids': ['d1']}, ['d1'], 1)
        r = copy.deepcopy(self.fixture.good)
        r['doc_ids'] = ['d1']
        r['usage'].update(tool_calls=0, trace_events=count)
        return r, tools

    def errors(self, result, method_type='retrieval'):
        return validate_results(self.fixture.qs[:1], [result], self.fixture.corpus,
                                trace_dir=self.root / 'traces', method_type=method_type)

    def test_more_than_32_retrieval_stages_are_valid(self):
        r, tools = self.retrieval_record(40)
        self.assertEqual(count_model_tool_calls(tools.trace), 0)
        self.assertEqual(self.errors(r), [])

    def test_observation_cap_only_applies_to_model_tools(self):
        r, tools = self.retrieval_record(output={'text': 'x' * 12001})
        self.assertEqual(self.errors(r), [])
        tools.trace[0]['event_type'] = 'model_tool'
        write_jsonl(self.root / 'traces/q1.jsonl', tools.trace)
        r['usage']['tool_calls'] = 1
        self.assertTrue(any('oversized' in e for e in self.errors(r, 'agent')))

    def test_model_tool_budget_still_rejects_33_calls(self):
        r, tools = self.retrieval_record(33)
        for event in tools.trace:
            event['event_type'] = 'model_tool'
        write_jsonl(self.root / 'traces/q1.jsonl', tools.trace)
        r['usage']['tool_calls'] = 33
        self.assertTrue(any('tool-call count' in e for e in self.errors(r, 'agent')))

    def test_typed_logs_cannot_silently_change_counting(self):
        r, tools = self.retrieval_record()
        self.assertTrue(any('current protocol metadata' in e for e in self.errors(r, None)))
        del tools.trace[0]['event_type']
        write_jsonl(self.root / 'traces/q1.jsonl', tools.trace)
        self.assertTrue(any('event_type' in e for e in self.errors(r)))
        tools.trace[0]['event_type'] = 'model_tool'
        write_jsonl(self.root / 'traces/q1.jsonl', tools.trace)
        self.assertTrue(any('require method_type=agent' in e for e in self.errors(r)))

    def test_trace_count_and_document_provenance_remain_required(self):
        r, _ = self.retrieval_record()
        del r['usage']['trace_events']
        self.assertTrue(any('trace_events is required' in e for e in self.errors(r)))
        r['usage']['trace_events'] = 2
        self.assertTrue(any('event count' in e for e in self.errors(r)))
        r['usage']['trace_events'] = 1
        r['doc_ids'].append('d2')
        self.assertTrue(any('not surfaced' in e for e in self.errors(r)))

    def test_retrieval_time_budget_remains_enforced(self):
        r, tools = self.retrieval_record()
        r['usage']['elapsed_seconds'] = 606
        self.assertTrue(any('wall-clock' in e for e in self.errors(r)))
        tools.started -= 601
        with self.assertRaises(BudgetExceeded):
            tools.read('d1')

    def test_agent_cannot_use_retrieval_only_logging_helper(self):
        tools = CorpusTools(self.fixture.corpus, 'q1', method_type='agent')
        with self.assertRaises(DataError):
            tools.record_retrieval('rank', {}, {}, ['d1'], 1)

    def test_unknown_tokens_require_an_explanation(self):
        r, _ = self.retrieval_record()
        r['usage'].update(model_input_tokens=None, token_notes=' ')
        self.assertTrue(any('token_notes' in e for e in self.errors(r)))
        r['usage']['token_notes'] = 'Provider does not expose token usage.'
        self.assertEqual(self.errors(r), [])
        r['usage']['tool_calls_complete'] = 'yes'
        self.assertTrue(any('must be a boolean' in e for e in self.errors(r)))

    def test_failed_log_write_preserves_last_complete_file(self):
        path = self.root / 'atomic.jsonl'
        write_jsonl(path, [{'id': 'good'}])
        with self.assertRaises(ValueError):
            write_jsonl(path, [{'id': 'bad', 'value': float('nan')}])
        self.assertEqual(read_jsonl(path), [{'id': 'good'}])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_pipeline.ProjectTests()
        self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / 'project'
        self.project.mkdir()
        for name in ['ir2026.py', 'run.py', 'check_results.py', 'evaluate.py']:
            shutil.copy2(PROJECT / name, self.project / name)
        shutil.copytree(PROJECT / 'configs', self.project / 'configs')
        shutil.copytree(self.fixture.root, self.project / 'corpus')
        data = self.project / 'datas'
        data.mkdir()
        for split in ['dev', 'test']:
            write_jsonl(data / f'{split}.jsonl', self.fixture.qs)
        write_jsonl(data / 'dev_labels.jsonl', self.fixture.labels)
        (self.project / 'fixture_method.py').write_text('''
def solve(question, tools, config):
    doc_id = 'd1' if question['id'] == 'q1' else 'd2'
    text = tools.read(doc_id)['text']
    if config['method_type'] == 'retrieval':
        for _ in range(35):
            tools.record_retrieval('fixture_rank', {}, {'doc_ids': [doc_id]}, [doc_id], 0)
    return {'answer': 'Alpha University' if doc_id == 'd1' else 'Beta University',
            'doc_ids': [doc_id], 'evidence': [{'doc_id': doc_id, 'text': text}], 'status': 'ok'}
''', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()
        self.fixture.tearDown()

    def command(self, *args, expected=0):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=self.project,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def run_method(self, method_type, split='dev'):
        run_dir = self.project / f'{method_type}_{split}'
        args = ['run.py', '--split', split, '--agent', 'fixture_method',
                '--method-label', 'synthetic-protocol-test', '--method-type', method_type,
                '--output', run_dir]
        if method_type == 'agent':
            args += ['--model-label', 'synthetic-model-fixture']
        self.command(*args)
        return run_dir

    def test_retrieval_runner_submission_check_and_evaluation(self):
        run_dir = self.run_method('retrieval')
        self.command('check_results.py', '--split', 'dev', '--run-dir', run_dir, '--submission')
        self.command('evaluate.py', '--split', 'dev', '--run-dir', run_dir)
        rows = read_jsonl(run_dir / 'results.jsonl')
        self.assertEqual(rows[0]['usage']['tool_calls'], 0)
        self.assertEqual(rows[0]['usage']['trace_events'], 36)
        self.assertEqual(rows[0]['usage']['model_input_tokens'], 0)
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        self.assertEqual(metrics['gold_hit_at_10'], 1)
        self.assertEqual(metrics['mean_trace_events'], 36)

    def test_agent_runner_uses_model_tool_events(self):
        run_dir = self.run_method('agent')
        self.command('check_results.py', '--run-dir', run_dir, '--submission')
        self.command('evaluate.py', '--run-dir', run_dir)
        rows = read_jsonl(run_dir / 'results.jsonl')
        self.assertEqual(rows[0]['usage']['tool_calls'], 1)
        self.assertIsNone(rows[0]['usage']['model_input_tokens'])

    def test_missing_model_for_agent_is_a_cli_error(self):
        self.command('run.py', '--method-type', 'agent', expected=2)

    def test_partial_submission_still_rejected(self):
        run_dir = self.project / 'partial'
        self.command('run.py', '--agent', 'fixture_method', '--limit', '1', '--output', run_dir)
        result = self.command('check_results.py', '--run-dir', run_dir, '--submission', expected=1)
        self.assertIn('missing=1', result.stdout)


if __name__ == '__main__':
    unittest.main()
