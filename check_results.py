#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from ir2026 import Corpus,DataError,load_policy,load_questions,read_jsonl,validate_results,validate_run_info

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Validate query coverage, document IDs, quotes, budgets and traces.')
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--questions',type=Path,help='Demo only: explicit query file. Forbidden with --submission.')
    p.add_argument('--submission',action='store_true')
    args=p.parse_args()
    if args.submission and args.questions:p.error('Submission validation always uses the full official split')
    try:
        config=load_policy(ROOT/'configs/default.json')
        questions=load_questions(args.questions or ROOT/'datas'/f'{args.split}.jsonl')
        info=json.loads((args.run_dir/'run_info.json').read_text())
        errors=validate_run_info(info,config,args.split,submission=args.submission)
        if not errors:
            errors=validate_results(questions,read_jsonl(args.run_dir/'results.jsonl'),Corpus(ROOT/'corpus'),config['limits'],args.run_dir/'traces',method_type=info.get('method_type'))
        print(json.dumps({'valid':not errors,'queries':len(questions),'errors':errors},ensure_ascii=False,indent=2))
        return 1 if errors else 0
    except (DataError,OSError,ValueError,TypeError) as exc:
        print(json.dumps({'valid':False,'errors':[str(exc)]},ensure_ascii=False));return 2

if __name__=='__main__':raise SystemExit(main())
