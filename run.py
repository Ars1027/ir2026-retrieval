#!/usr/bin/env python3
"""Run a student solve() adapter or the clearly labeled offline format demo."""
from __future__ import annotations
import argparse
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from ir2026 import BudgetExceeded,Corpus,CorpusTools,DataError,PROTOCOL_VERSION,count_model_tool_calls,load_policy,load_questions,read_jsonl,write_jsonl

ROOT=Path(__file__).resolve().parent

def worker(connection,agent_name,question,corpus_root,config,trace_path):
    if hasattr(os,'setsid'):
        try:os.setsid()
        except OSError:pass
    started=time.monotonic();tools=None
    try:
        tools=CorpusTools(Corpus(corpus_root),question['id'],config['limits'],trace_path=trace_path,method_type=config['method_type'])
        module=importlib.import_module(agent_name)
        result=module.solve(dict(question),tools,config)
        if not isinstance(result,dict):raise DataError('solve() must return a dictionary')
    except BudgetExceeded as exc:
        result={'answer':'','doc_ids':[],'evidence':[],'status':'budget_exceeded','error':str(exc)}
    except Exception as exc:
        result={'answer':'','doc_ids':[],'evidence':[],'status':'error','error':f'{type(exc).__name__}: {exc}'}
    result['id']=question['id']
    usage=result.get('usage',{})
    if not isinstance(usage,dict):usage={}
    usage.update({'tool_calls':count_model_tool_calls(tools.trace) if tools else 0,
                  'trace_events':len(tools.trace) if tools else 0,'elapsed_seconds':round(time.monotonic()-started,4)})
    usage.setdefault('tool_calls_complete',True)
    default_tokens=None if config['model'] else 0
    usage.setdefault('model_input_tokens',default_tokens);usage.setdefault('model_output_tokens',default_tokens)
    usage.setdefault('token_notes','Model usage unavailable; explain the reporting limitation.' if config['model'] else 'No model used.')
    result['usage']=usage
    try:connection.send((result,tools.trace if tools else []))
    finally:connection.close()

def stop_process(p):
    if p.is_alive():
        try:
            if os.name=='posix' and os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
            else:p.terminate()
        except ProcessLookupError:pass
        p.join(2)
    if p.is_alive():p.kill();p.join()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--agent',default='examples.keyword_agent',help='Python module exposing solve(question, tools, config)')
    p.add_argument('--method-label',help='Method name/version; defaults to the adapter module name')
    p.add_argument('--method-type',choices=['retrieval','agent'],default='retrieval',help='Fixed retrieval pipeline (default) or model-directed tool interaction')
    p.add_argument('--model-label',help='Actual model identifier, if used; required for --method-type agent')
    p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'output'/'demo')
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.method_label is not None and not args.method_label.strip():p.error('--method-label cannot be blank')
    if args.model_label is not None and not args.model_label.strip():p.error('--model-label cannot be blank')
    if args.method_type=='agent' and not args.model_label:p.error('--method-type agent requires --model-label')
    if args.model_label=='offline-keyword-demo':p.error('Omit --model-label when no model is used')
    config=load_policy(ROOT/'configs/default.json')
    config.update({'method_type':args.method_type,'model':args.model_label})
    questions=load_questions(ROOT/'datas'/f'{args.split}.jsonl')
    if args.limit is not None:questions=questions[:args.limit]
    if args.output.exists() and any(args.output.iterdir()):p.error('Use a new or empty output directory')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'traces').mkdir()
    write_jsonl(args.output/'questions.jsonl',questions)
    info={'release':config['release'],'protocol_version':PROTOCOL_VERSION,
          'method':args.method_label or args.agent,'method_type':args.method_type,
          'agent_module':args.agent,'model':args.model_label,
          'mode':'demo' if args.agent.startswith('examples.') else 'coursework',
          'split':args.split,'queries':len(questions),'limits':config['limits'],
          'python':sys.version.split()[0]}
    (args.output/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    ctx=mp.get_context('spawn');results=[]
    for i,q in enumerate(questions,1):
        receiving,sending=ctx.Pipe(duplex=False)
        trace_path=args.output/'traces'/f'{q["id"]}.jsonl'
        proc=ctx.Process(target=worker,args=(sending,args.agent,q,ROOT/'corpus',config,trace_path))
        proc.start();sending.close()
        try:
            if receiving.poll(config['limits']['max_seconds']):
                try:result,trace=receiving.recv()
                except EOFError:result,trace=None,[]
            else:result,trace=None,[]
            if result is None:
                timed_out=proc.is_alive()
                stop_process(proc)
                if trace_path.exists():
                    try:trace=read_jsonl(trace_path)
                    except (DataError,OSError):trace=[]
                result={'id':q['id'],'answer':'','doc_ids':[],'evidence':[],
                        'status':'timeout' if timed_out else 'error',
                        'error':'Worker timed out or exited before returning a record.',
                        'usage':{'tool_calls':count_model_tool_calls(trace),'trace_events':len(trace),'tool_calls_complete':False,'elapsed_seconds':config['limits']['max_seconds'] if timed_out else 0,
                                 'model_input_tokens':None if args.model_label else 0,'model_output_tokens':None if args.model_label else 0,
                                 'token_notes':'Worker did not return model usage. Tool count covers persisted events and may miss an in-flight call.'}}
        finally:
            receiving.close();proc.join(1);stop_process(proc)
        results.append(result)
        write_jsonl(args.output/'traces'/f'{q["id"]}.jsonl',trace)
        write_jsonl(args.output/'results.jsonl',results)
        print(f'{i}/{len(questions)} id={q["id"]} status={result["status"]}',flush=True)
    print(f'Results: {args.output / "results.jsonl"}')

if __name__=='__main__':main()
