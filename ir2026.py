"""Shared data, tool, validation and evaluation utilities for IR2026.

Only the Python 3.10+ standard library is required. These path checks protect
this helper's file reads; they are not an OS sandbox for student code.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import tempfile
import unicodedata

STATUSES={'ok','no_answer','timeout','error','budget_exceeded'}
PROTOCOL_VERSION='ir2026.retrieval.v2'
METHOD_TYPES=('retrieval','agent')
EVENT_TYPES=('retrieval','model_tool')
POLICY={'max_tool_calls':32,'max_tool_output_chars':12000,'max_seconds':600,
        'max_doc_ids':10,'max_evidence_quotes':5,'min_quote_chars':20,'max_quote_chars':1200}

class DataError(ValueError): pass
class BudgetExceeded(RuntimeError): pass

def _reject_constant(value):
    raise DataError(f'Non-finite JSON constant: {value}')

def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: row=json.loads(line,parse_constant=_reject_constant)
            except (ValueError,TypeError) as exc: raise DataError(f'{Path(path).name}:{n}: invalid JSON') from exc
            if not isinstance(row,dict):raise DataError(f'{Path(path).name}:{n}: expected an object')
            rows.append(row)
    return rows

def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Keep the last complete log if a worker is terminated during persistence.
    temp_path=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
            temp_path=Path(f.name)
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        os.replace(temp_path,path)
    finally:
        if temp_path is not None:temp_path.unlink(missing_ok=True)

def keyed(rows,kind):
    out={}
    for row in rows:
        qid=row.get('id')
        if not isinstance(qid,str) or not qid:raise DataError(f'{kind}: ID must be a nonempty string')
        if qid in out:raise DataError(f'{kind}: duplicate ID {qid}')
        out[qid]=row
    return out

def load_questions(path):
    rows=read_jsonl(path);keyed(rows,'questions')
    if not rows:raise DataError('Question set is empty')
    for row in rows:
        if not isinstance(row.get('question'),str) or not row['question'].strip():raise DataError('Invalid question text')
    return rows

def load_policy(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=_reject_constant)
    if data.get('protocol_version',PROTOCOL_VERSION)!=PROTOCOL_VERSION:raise DataError('Unsupported project protocol_version')
    actual=data.get('limits',{})
    for k,v in POLICY.items():
        if actual.get(k)!=v:raise DataError(f'Course policy mismatch: {k}')
    return data

def is_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def validate_run_info(info,config,split,submission=False):
    """Shared metadata checks for student validation and instructor grading.

    Historical runs without a protocol version retain their original tool-call
    semantics. New exports explicitly identify their method and logging mode.
    """
    if not isinstance(info,dict):return ['run_info.json must be an object']
    errors=[]
    if info.get('release')!=config['release']:errors.append('Wrong release in run_info.json')
    if info.get('limits')!=config['limits']:errors.append('Wrong limits in run_info.json')
    if info.get('split')!=split:errors.append('Split differs from run_info.json')
    if info.get('mode') not in ('demo','coursework'):errors.append('Invalid run mode')
    modern='protocol_version' in info or 'method_type' in info
    if modern:
        if info.get('protocol_version')!=PROTOCOL_VERSION:errors.append('Unsupported run protocol_version')
        if info.get('method_type') not in METHOD_TYPES:errors.append('method_type must be retrieval or agent')
        if not isinstance(info.get('method'),str) or not info['method'].strip():errors.append('Identify the retrieval method in run_info.json')
        if 'model' not in info:errors.append('Specify model; use null when no model is used')
    model=info.get('model')
    if model is not None and (not isinstance(model,str) or not model.strip()):errors.append('model must be null or a nonempty model identifier')
    if (info.get('method_type')=='agent' or (not modern and submission)) and not model:
        errors.append('Agent runs require a model identifier; use method_type=retrieval and model=null for non-model methods')
    module=info.get('agent_module')
    if module is not None and (not isinstance(module,str) or not module.strip()):errors.append('Invalid agent_module')
    if submission and (info.get('mode')!='coursework' or
                       (isinstance(module,str) and module.startswith('examples.')) or model=='offline-keyword-demo'):
        errors.append('Demonstration runs are not course submissions; submit your implemented method and full results')
    return errors

def count_model_tool_calls(events):
    """Legacy records have no event_type and represent model tool calls."""
    return sum(event.get('event_type','model_tool')=='model_tool' for event in events)

class Corpus:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.docs={}
        for row in read_jsonl(self.root/'documents.jsonl'):
            doc_id=row.get('doc_id')
            if not isinstance(doc_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',doc_id):raise DataError('Invalid corpus ID')
            if doc_id in self.docs:raise DataError('Duplicate corpus ID')
            target=(self.root/row['path']).resolve()
            if not target.is_relative_to(self.root) or not target.is_file():raise DataError('Invalid corpus path')
            self.docs[doc_id]={**row,'resolved_path':target}
        if not self.docs:raise DataError('Corpus is empty')

    def text(self,doc_id):
        if not isinstance(doc_id,str) or doc_id not in self.docs:raise DataError(f'Unknown document ID: {doc_id}')
        return self.docs[doc_id]['resolved_path'].read_text(encoding='utf-8')

    def verify(self):
        errors=[]
        for doc_id,row in self.docs.items():
            data=row['resolved_path'].read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:errors.append(doc_id)
        return errors

class CorpusTools:
    """Optional corpus helpers; the runner always supplies an explicit method type.

    Direct legacy callers retain the original agent budget by default.
    """
    def __init__(self,corpus,qid,limits=None,trace_path=None,method_type='agent'):
        if method_type not in METHOD_TYPES:raise DataError('Invalid method_type')
        self.corpus=corpus;self.qid=qid;self.limits=limits or dict(POLICY)
        self.trace=[];self.started=time.monotonic();self.trace_path=trace_path
        self.method_type=method_type
        self.event_type='model_tool' if method_type=='agent' else 'retrieval'

    def _begin(self):
        if self.method_type=='agent' and count_model_tool_calls(self.trace)>=self.limits['max_tool_calls']:raise BudgetExceeded('tool_calls')
        if time.monotonic()-self.started>self.limits['max_seconds']:raise BudgetExceeded('time')
        return time.monotonic()

    def _record(self,tool,args,output,doc_ids,started,error=None):
        encoded=json.dumps(output,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        if self.method_type=='agent' and len(encoded)>self.limits['max_tool_output_chars']:raise DataError('Tool output exceeds course character limit')
        self.trace.append({'id':self.qid,'seq':len(self.trace)+1,'event_type':self.event_type,'tool':tool,'arguments':args,
                           'output':output,'doc_ids':doc_ids,'error':error,
                           'elapsed_ms':round((time.monotonic()-started)*1000,3),'output_chars':len(encoded)})
        if self.trace_path is not None:write_jsonl(self.trace_path,self.trace)
        return output

    def record_retrieval(self,stage,arguments,output,doc_ids,elapsed_ms):
        """Record an already executed custom retrieval/reranking stage.

        Only for a fixed retrieval pipeline. Agent observations must go through
        actual model-tool logs, not be relabeled as internal retrieval stages.
        """
        if self.method_type!='retrieval':raise DataError('record_retrieval requires method_type=retrieval')
        self._begin()
        if not isinstance(stage,str) or not stage.strip() or not isinstance(arguments,dict):raise DataError('Invalid retrieval stage')
        if not is_number(elapsed_ms) or elapsed_ms<0:raise DataError('Invalid stage elapsed time')
        if not isinstance(doc_ids,list) or any(not isinstance(d,str) or d not in self.corpus.docs for d in doc_ids):raise DataError('Invalid stage document IDs')
        return self._record(stage,arguments,output,doc_ids,time.monotonic()-elapsed_ms/1000)

    def search(self,terms,match_all=False,limit=10):
        """Scan raw files; score is literal term coverage, not a learned ranking."""
        started=self._begin()
        args={'terms':terms,'match_all':match_all,'limit':limit}
        try:
            if not isinstance(terms,list) or not 1<=len(terms)<=8:raise DataError('Use 1 to 8 literal terms')
            if any(not isinstance(t,str) or not 2<=len(t)<=160 for t in terms):raise DataError('Invalid search term')
            if type(match_all) is not bool or type(limit) is not int or not 1<=limit<=20:raise DataError('Invalid search options')
            terms=list(dict.fromkeys(t.casefold() for t in terms))
            matches=[]
            for index,(doc_id,meta) in enumerate(self.corpus.docs.items()):
                if index%128==0 and time.monotonic()-self.started>self.limits['max_seconds']:
                    raise BudgetExceeded('time')
                text=meta['resolved_path'].read_text(encoding='utf-8');folded=text.casefold()
                found=[t for t in terms if t in folded]
                if not found or (match_all and len(found)!=len(terms)):continue
                # Preserve original text when choosing the displayed preview.
                position=next((m.start() for t in found if (m:=re.search(re.escape(t),text,re.IGNORECASE))),0)
                start=max(0,position-80)
                matches.append({'doc_id':doc_id,'score':len(found),'start':start,'text':text[start:start+260]})
            matches.sort(key=lambda r:(-r['score'],r['doc_id']))
            visible=matches[:limit]
            output={'matches':visible,'total_matches':len(matches),'truncated':len(matches)>limit}
            while len(json.dumps(output,ensure_ascii=False,separators=(',',':')))>self.limits['max_tool_output_chars'] and visible:
                visible.pop();output['truncated']=True
            return self._record('search',args,output,[x['doc_id'] for x in visible],started)
        except (DataError,BudgetExceeded,OSError) as exc:
            self._record('search',args,{'error':str(exc)},[],started,error=str(exc))
            raise

    def read(self,doc_id,start=0,max_chars=2000):
        """Offsets refer to Unicode characters, not encoded byte offsets."""
        started=self._begin();args={'doc_id':doc_id,'start':start,'max_chars':max_chars}
        try:
            if type(start) is not int or start<0:raise DataError('Invalid start offset')
            if type(max_chars) is not int or not 1<=max_chars<=10000:raise DataError('Invalid read length')
            text=self.corpus.text(doc_id)
            if start>len(text):raise DataError('Start offset exceeds document length')
            result={'doc_id':doc_id,'start':start,'end':min(start+max_chars,len(text)),
                    'total_chars':len(text),'text':text[start:start+max_chars],
                    'truncated':start+max_chars<len(text)}
            while len(json.dumps(result,ensure_ascii=False,separators=(',',':')))>self.limits['max_tool_output_chars']:
                result['text']=result['text'][:max(0,len(result['text'])-500)]
                result['end']=start+len(result['text']);result['truncated']=True
            return self._record('read',args,result,[doc_id],started)
        except (DataError,OSError) as exc:
            self._record('read',args,{'error':str(exc)},[],started,error=str(exc));raise

def validate_results(questions,results,corpus,limits=None,trace_dir=None,method_type=None):
    limits=limits or POLICY
    if method_type is not None and method_type not in METHOD_TYPES:return ['Invalid method_type']
    qmap=keyed(questions,'questions');rmap=keyed(results,'results')
    errors=[]
    if set(rmap)!=set(qmap):
        errors.append(f'Question IDs differ: missing={len(set(qmap)-set(rmap))}, extra={len(set(rmap)-set(qmap))}')
    for qid,r in rmap.items():
        prefix=f'{qid}: '
        status=r.get('status');ids=r.get('doc_ids');quotes=r.get('evidence');usage=r.get('usage')
        if not isinstance(status,str) or status not in STATUSES:errors.append(prefix+'invalid status')
        if not isinstance(r.get('answer'),str):errors.append(prefix+'answer must be a string')
        elif status=='ok' and not r['answer'].strip():errors.append(prefix+'ok requires a nonempty answer')
        valid_ids=isinstance(ids,list) and all(isinstance(x,str) for x in ids)
        if not valid_ids:errors.append(prefix+'doc_ids must be a string list');ids=[]
        if len(ids)>limits['max_doc_ids'] or len(ids)!=len(set(ids)):errors.append(prefix+'too many or duplicate document IDs')
        if any(x not in corpus.docs for x in ids):errors.append(prefix+'unknown document ID')
        if not isinstance(quotes,list):errors.append(prefix+'evidence must be a list');quotes=[]
        if len(quotes)>limits['max_evidence_quotes']:errors.append(prefix+'too many evidence quotes')
        if ids and not quotes:errors.append(prefix+'retrieved documents require at least one evidence quote')
        for quote in quotes:
            if not isinstance(quote,dict):errors.append(prefix+'invalid evidence object');continue
            doc_id=quote.get('doc_id');text=quote.get('text')
            if doc_id not in ids or doc_id not in corpus.docs:errors.append(prefix+'evidence document is not in final doc_ids');continue
            if not isinstance(text,str) or not limits['min_quote_chars']<=len(text)<=limits['max_quote_chars']:
                errors.append(prefix+'invalid evidence length');continue
            if text not in corpus.text(doc_id):errors.append(prefix+'evidence quote does not match source text')
        if not isinstance(usage,dict):errors.append(prefix+'usage is required');continue
        calls=usage.get('tool_calls');elapsed=usage.get('elapsed_seconds')
        if type(calls) is not int or not 0<=calls<=limits['max_tool_calls']:errors.append(prefix+'invalid tool-call count')
        if method_type=='retrieval' and calls!=0:errors.append(prefix+'retrieval pipelines have zero model tool calls')
        trace_events=usage.get('trace_events')
        if method_type is not None and 'trace_events' not in usage:errors.append(prefix+'trace_events is required by the current protocol')
        if 'trace_events' in usage and (type(trace_events) is not int or trace_events<0):errors.append(prefix+'invalid trace event count')
        if not is_number(elapsed) or elapsed<0:errors.append(prefix+'invalid elapsed time')
        elif elapsed>limits['max_seconds']+5:errors.append(prefix+'wall-clock budget exceeded')
        for key in ['model_input_tokens','model_output_tokens']:
            value=usage.get(key)
            if value is not None and (type(value) is not int or value<0):errors.append(prefix+f'invalid {key}')
        if any(usage.get(k) is None for k in ['model_input_tokens','model_output_tokens']) and (not isinstance(usage.get('token_notes'),str) or not usage['token_notes'].strip()):
            errors.append(prefix+'unavailable token counts require token_notes')
        if 'tool_calls_complete' in usage and type(usage['tool_calls_complete']) is not bool:errors.append(prefix+'tool_calls_complete must be a boolean')
        if trace_dir is not None:
            if not re.fullmatch(r'[A-Za-z0-9_-]+',qid):errors.append(prefix+'unsafe trace ID');continue
            trace_path=Path(trace_dir)/(qid+'.jsonl')
            if not trace_path.is_file():errors.append(prefix+'missing trace');continue
            try:events=read_jsonl(trace_path)
            except (DataError,OSError):errors.append(prefix+'invalid trace file');continue
            if calls!=count_model_tool_calls(events):errors.append(prefix+'tool count differs from trace')
            if 'trace_events' in usage and trace_events!=len(events):errors.append(prefix+'event count differs from trace')
            seen=set()
            for i,event in enumerate(events,1):
                event_type=event.get('event_type','model_tool' if method_type is None else None)
                if event_type not in EVENT_TYPES:errors.append(prefix+'invalid or missing event_type')
                if method_type is None and event_type=='retrieval':errors.append(prefix+'retrieval events require current protocol metadata')
                if method_type=='retrieval' and event_type=='model_tool':errors.append(prefix+'model tool events require method_type=agent')
                if event.get('id')!=qid or type(event.get('seq')) is not int or event.get('seq')!=i:errors.append(prefix+'trace sequence mismatch')
                if not isinstance(event.get('tool'),str) or not event['tool'].strip() or not isinstance(event.get('arguments'),dict):errors.append(prefix+'invalid trace call')
                if not is_number(event.get('elapsed_ms')) or event['elapsed_ms']<0:errors.append(prefix+'invalid trace elapsed time')
                output=event.get('output')
                output_chars=len(json.dumps(output,ensure_ascii=False,allow_nan=False,separators=(',',':')))
                if type(event.get('output_chars')) is not int or event.get('output_chars')!=output_chars or (event_type=='model_tool' and output_chars>limits['max_tool_output_chars']):
                    errors.append(prefix+'invalid or oversized tool output')
                observed=event.get('doc_ids',[])
                if not isinstance(observed,list) or any(not isinstance(d,str) or d not in corpus.docs for d in observed):
                    errors.append(prefix+'invalid trace document IDs')
                else:seen.update(observed)
            if not set(ids)<=seen:errors.append(prefix+'final documents were not surfaced in trace')
    return errors

def normalize_answer(text):
    # Deliberately conservative: preserve numeric signs, decimal points and units.
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split()).strip('"“”')

def evaluate(questions,results,labels):
    qmap=keyed(questions,'questions');rmap=keyed(results,'results');lmap=keyed(labels,'labels')
    if not set(qmap)<=set(lmap):raise DataError('Labels are missing for selected questions')
    rows=[]
    for qid in qmap:
        r=rmap.get(qid,{'doc_ids':[],'answer':'','status':'error','usage':{}});gold=lmap[qid]
        gids=set(gold.get('gold_doc_ids',[]));eids=set(gold.get('evidence_doc_ids',[]))
        if not gids or not eids:raise DataError('Each question needs nonempty gold and evidence labels')
        ids=r.get('doc_ids',[]);answer=r.get('answer','')
        row={'id':qid,'status':r.get('status'),'answer_em':int(bool(answer.strip()) and normalize_answer(answer)==normalize_answer(gold['answer'])),
             'evidence_recall_at_10':len(set(ids[:10])&eids)/len(eids),
             'tool_calls':r.get('usage',{}).get('tool_calls',0),
             'trace_events':r.get('usage',{}).get('trace_events',r.get('usage',{}).get('tool_calls',0)),
             'elapsed_seconds':r.get('usage',{}).get('elapsed_seconds',0)}
        for k in [1,5,10]:row[f'gold_hit_at_{k}']=int(bool(set(ids[:k])&gids))
        rows.append(row)
    n=len(rows)
    if not n:raise DataError('Cannot evaluate an empty set')
    metrics={'queries':n,'primary_metric':'gold_hit_at_10'}
    for key in ['gold_hit_at_1','gold_hit_at_5','gold_hit_at_10','evidence_recall_at_10','answer_em','tool_calls','trace_events','elapsed_seconds']:
        metrics[('mean_'+key) if key in ['tool_calls','trace_events','elapsed_seconds'] else key]=sum(r[key] for r in rows)/n
    metrics['completion_rate']=sum(r['status']=='ok' for r in rows)/n
    metrics['incomplete_tool_usage_queries']=sum(r.get('usage',{}).get('tool_calls_complete') is False for r in results)
    metrics['tool_usage_note']='tool_calls counts model_tool events, not fixed retrieval stages; legacy untyped events count as model tools. Timed-out workers may miss an in-flight event.'
    metrics['effect_points_out_of_6']=round(6*metrics['gold_hit_at_10'],4)
    metrics['answer_em_note']='Conservative normalized exact match, NOT official semantic/LLM-judge accuracy.'
    metrics['evidence_note']='Recall over the released evidence-document IDs, not exhaustive semantic relevance.'
    return metrics,rows
