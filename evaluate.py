#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from ir2026 import Corpus,DataError,evaluate,keyed,load_policy,load_questions,read_jsonl,validate_results,validate_run_info

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Compute GoldHit@1/5/10, EvidenceRecall@10 and auxiliary AnswerEM.')
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,help='Teacher-only path for test labels')
    p.add_argument('--allow-subset',action='store_true',help='Explicit partial demo evaluation; not accepted for final grading')
    args=p.parse_args()
    try:
        config=load_policy(ROOT/'configs/default.json')
        info=json.loads((args.run_dir/'run_info.json').read_text(encoding='utf-8'))
        metadata_errors=validate_run_info(info,config,args.split)
        if metadata_errors:raise DataError('\n'.join(metadata_errors))
        questions=load_questions(ROOT/'datas'/f'{args.split}.jsonl')
        full_count=len(questions);results=read_jsonl(args.run_dir/'results.jsonl')
        if args.allow_subset:
            ids=set(keyed(results,'results'))
            if not ids<=set(keyed(questions,'questions')):raise DataError('Unknown question IDs in subset')
            questions=[q for q in questions if q['id'] in ids]
        labels_path=args.labels or ROOT/'datas'/f'{args.split}_labels.jsonl'
        if not labels_path.exists():raise DataError('Test labels are teacher-only. Submit the test run for instructor evaluation.')
        errors=validate_results(questions,results,Corpus(ROOT/'corpus'),config['limits'],args.run_dir/'traces',method_type=info.get('method_type'))
        if errors:raise DataError('\n'.join(errors[:30]))
        metrics,per_query=evaluate(questions,results,read_jsonl(labels_path))
        metrics.update({'split':args.split,'release':config['release'],'full_split_queries':full_count,
                        'method':info.get('method',info.get('agent_module')),'method_type':info.get('method_type','legacy'),
                        'protocol_version':info.get('protocol_version','legacy'),
                        'evaluation_scope':'partial_demo' if args.allow_subset else 'full_split',
                        'citation_span_match_rate':1.0 if any(r['evidence'] for r in results) else None,
                        'citation_note':'Only literal source-span validity was checked; this is not semantic evidence correctness.'})
        (args.run_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with (args.run_dir/'per_query.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_query[0]));writer.writeheader();writer.writerows(per_query)
        print(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False));return 0
    except (DataError,OSError,ValueError,TypeError) as exc:
        print(f'Evaluation failed: {exc}');return 2

if __name__=='__main__':raise SystemExit(main())
