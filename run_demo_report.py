#!/usr/bin/env python3
"""Run and report the IR2026 repository's unmodified offline demo pipeline."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unicodedata


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def log(message: str) -> None:
    print(f'{datetime.now():%H:%M:%S}  INFO  {message}', flush=True)


def run_stage(project: Path, command: list[str], name: str, log_dir: Path,
              live: bool = False) -> str:
    log(f'{name}: {shlex.join(["python3", *command])}')
    lines = []
    with (log_dir / f'{name}.log').open('w', encoding='utf-8') as saved:
        proc = subprocess.Popen([sys.executable, '-u', *command], cwd=project,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8')
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                saved.write(line)
                saved.flush()
                lines.append(line)
                if live and line.strip():
                    log(f'{name}: {line.rstrip()}')
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f'{name} exited with code {code}\n' + ''.join(lines[-12:]))
    return ''.join(lines)


def fit(text: str, width: int) -> str:
    text = ' '.join(text.split())
    sizes = [0 if unicodedata.combining(c) else
             2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text]
    total = sum(sizes)
    if total <= width:
        return text + ' ' * (width - total)
    result, used = '', 0
    for char, size in zip(text, sizes):
        if used + size > width - 3:
            break
        result += char
        used += size
    return result + '...' + ' ' * (width - 3 - used)


def title_for(project: Path, meta: dict) -> str:
    with (project / 'corpus' / meta['path']).open(encoding='utf-8') as stream:
        for _ in range(12):
            line = stream.readline()
            if not line:
                break
            if line.startswith('title:'):
                return line.split(':', 1)[1].strip()
    return meta.get('url', meta['doc_id'])


def report(project: Path, run_dir: Path) -> None:
    info = json.loads((run_dir / 'run_info.json').read_text())
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    results = load_jsonl(run_dir / 'results.jsonl')
    traces = {r['id']: load_jsonl(run_dir / 'traces' / f'{r["id"]}.jsonl') for r in results}
    # Labels are loaded only after the repository runner and evaluator have finished.
    labels = {r['id']: r for r in load_jsonl(project / 'datas' / f'{info["split"]}_labels.jsonl')}
    docs = {r['doc_id']: r for r in load_jsonl(project / 'corpus/documents.jsonl')}
    with (run_dir / 'per_query.csv').open(newline='', encoding='utf-8') as stream:
        scores = {r['id']: r for r in csv.DictReader(stream)}
    first = results[0]
    events = traces[first['id']]
    search = next((e for e in events if e['tool'] == 'search'), None)
    reads = [e for e in events if e['tool'] == 'read']
    matches = {m['doc_id']: m for m in search['output'].get('matches', [])} if search else {}
    gold = set(labels[first['id']]['gold_doc_ids'])
    selected = scores[first['id']]
    usage = first['usage']

    print('\n' + '=' * 88)
    print('IR2026  |  RETRIEVAL RUN REPORT')
    print('=' * 88)
    print(f'Dataset : {info["release"]}  |  split={info["split"]}  |  '
          f'queries={metrics["queries"]}/{metrics["full_split_queries"]}  |  corpus={len(docs):,}')
    print(f'Method  : {info.get("method", info.get("agent_module", "external"))}  |  scope={metrics["evaluation_scope"]}')
    print(f'Query   : {first["id"]}  |  status={first["status"]}  |  '
          f'stages={usage.get("trace_events", len(events))}  |  elapsed={usage["elapsed_seconds"]:.3f}s')
    if search:
        print('Terms   : ' + ', '.join(search['arguments']['terms']))
        print(f'Search  : matches={search["output"]["total_matches"]:,}  |  '
              f'returned={len(first["doc_ids"])}  |  elapsed={search["elapsed_ms"] / 1000:.3f}s')
    for event in reads:
        print(f'Read    : doc_id={event["arguments"]["doc_id"]}  |  '
              f'start={event["arguments"]["start"]}  |  chars={len(event["output"].get("text", ""))}')
    print('-' * 88)
    print(f'{"RANK":>4}  {"DOC_ID":<8} {"TERMS":>5}  {"GOLD":<4}  TITLE / SOURCE')
    print('-' * 88)
    for rank, doc_id in enumerate(first['doc_ids'], 1):
        terms = matches.get(doc_id, {}).get('score', '-')
        mark = 'HIT' if doc_id in gold else '-'
        title = fit(title_for(project, docs[doc_id]), 57).rstrip()
        print(f'{rank:>4}  {doc_id:<8} {terms:>5}  {mark:<4}  {title}')
    print('-' * 88)
    print(f'Query {first["id"]}:  Hit@1={int(selected["gold_hit_at_1"])}  '
          f'Hit@5={int(selected["gold_hit_at_5"])}  '
          f'Hit@10={int(selected["gold_hit_at_10"])}  '
          f'EvidenceRecall@10={float(selected["evidence_recall_at_10"]):.3f}')
    if len(results) > 1:
        print(f'Run mean:  Hit@10={metrics["gold_hit_at_10"]:.3f}  '
              f'EvidenceRecall@10={metrics["evidence_recall_at_10"]:.3f}')
    print('TERMS = literal term coverage; GOLD = post-run label check.')
    print(f'Artifacts: {run_dir.relative_to(project) if run_dir.is_relative_to(project) else run_dir}')
    print('           results.jsonl  traces/  run_info.json  metrics.json  per_query.csv')
    print('=' * 88, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent,
                        help='IR_2026_Project root (default: directory containing this script)')
    parser.add_argument('--split', choices=['train', 'dev'], default='dev')
    parser.add_argument('--limit', type=int, default=1, help='Number of questions (default: 1)')
    parser.add_argument('--output', type=Path, help='New or empty run directory')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    project = args.project.resolve()
    if not (project / 'run.py').is_file():
        parser.error(f'Project root not found: {project}; use --project PATH')
    if args.output:
        run_dir = args.output if args.output.is_absolute() else project / args.output
        run_dir = run_dir.resolve()
    else:
        run_dir = project / 'output' / datetime.now().strftime('demo_%Y%m%d_%H%M%S_%f')
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f'Output directory must be empty: {run_dir}')
    log_dir = run_dir.parent / (run_dir.name + '_logs')
    log_dir.mkdir(parents=True, exist_ok=False)
    run_arg = str(run_dir.relative_to(project)) if run_dir.is_relative_to(project) else str(run_dir)
    try:
        verified = json.loads(run_stage(project, ['verify_data.py', '--full'], 'verify', log_dir))
        if not verified.get('valid'):
            raise RuntimeError('Repository data verification failed')
        log(f'verify: PASS | documents={verified["documents"]:,} | full_hash_check=true')
        run_stage(project, ['run.py', '--split', args.split, '--limit', str(args.limit),
                            '--agent', 'examples.keyword_agent', '--output', run_arg],
                  'run', log_dir, live=True)
        checked = json.loads(run_stage(project, ['check_results.py', '--split', args.split,
                              '--run-dir', run_arg, '--questions', run_arg + '/questions.jsonl'],
                              'check', log_dir))
        (run_dir / 'check_results.json').write_text(json.dumps(checked, indent=2) + '\n')
        if not checked.get('valid'):
            raise RuntimeError(f'Repository result validation failed: {checked.get("errors")}')
        log(f'check: PASS | queries={checked["queries"]} | errors=0')
        run_stage(project, ['evaluate.py', '--split', args.split, '--run-dir', run_arg,
                            '--allow-subset'], 'evaluate', log_dir)
        log('evaluate: DONE | metrics.json + per_query.csv written')
        report(project, run_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: {exc}\nStage logs: {log_dir}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
