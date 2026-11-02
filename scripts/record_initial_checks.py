"""Run upstream checks and a 3-query demo; publish aggregate metadata only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    now = datetime.now(timezone(timedelta(hours=8)))
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    local = ROOT / 'output' / f'initial_checks_{stamp}'
    local.mkdir(parents=True)
    demo = local / 'demo_dev3'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    stages = []

    def run(name: str, *arguments: str) -> subprocess.CompletedProcess:
        command = [sys.executable, *arguments]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding='utf-8', timeout=1900)
        (local / f'{name}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        stages.append({'stage': name, 'exit_code': result.returncode})
        print(f'{name}: exit={result.returncode}', flush=True)
        result.check_returncode()
        return result

    verified = run('verify_data', 'verify_data.py', '--full')
    tests = run('upstream_tests', '-m', 'unittest', 'discover', '-s', 'tests', '-v')
    run('demo_run', 'run.py', '--split', 'dev', '--agent', 'examples.keyword_agent',
        '--method-type', 'retrieval', '--method-label', 'upstream-keyword-demo',
        '--limit', '3', '--output', str(demo))
    run('demo_check', 'check_results.py', '--split', 'dev', '--run-dir', str(demo),
        '--questions', str(demo / 'questions.jsonl'))
    run('demo_evaluate', 'evaluate.py', '--split', 'dev', '--run-dir', str(demo), '--allow-subset')
    metrics = json.loads((demo / 'metrics.json').read_text(encoding='utf-8'))
    allowed = ['queries', 'full_split_queries', 'evaluation_scope', 'gold_hit_at_1',
               'gold_hit_at_5', 'gold_hit_at_10', 'evidence_recall_at_10', 'answer_em',
               'completion_rate', 'mean_elapsed_seconds', 'mean_tool_calls', 'mean_trace_events']
    count = re.search(r'Ran (\d+) tests?', tests.stdout + tests.stderr)
    rows = [json.loads(line) for line in (demo / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    status_counts = {}
    for row in rows:
        status_counts[row['status']] = status_counts.get(row['status'], 0) + 1
    source_names = ['run.py', 'ir2026.py', 'check_results.py', 'evaluate.py', 'examples/keyword_agent.py']
    record = {
        'record_kind': 'upstream_environment_and_partial_demo_only',
        'recorded_at': now.isoformat(),
        'student_implementation_ready': False,
        'formal_baseline_metrics': None,
        'note': 'Upstream keyword demo on 3 dev queries; not a student baseline, full-dev result, or submission.',
        'python': platform.python_version(), 'platform': platform.platform(),
        'data_verification': json.loads(verified.stdout),
        'upstream_tests': {'passed': True, 'count': int(count.group(1)) if count else None},
        'stages': stages, 'demo_metrics': {k: metrics[k] for k in allowed if k in metrics},
        'demo_status_counts': status_counts,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
        'local_artifacts': str(local.relative_to(ROOT)).replace('\\', '/'),
        'publication': 'Only aggregate metadata here; QA, labels, quotes, corpus and traces stay local.'
    }
    destination = ROOT / 'records' / f'initial_checks_{stamp}.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Aggregate record: {destination.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
