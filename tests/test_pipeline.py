import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ir2026 import BudgetExceeded,Corpus,CorpusTools,DataError,POLICY,evaluate,keyed,normalize_answer,read_jsonl,validate_results,write_jsonl

class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'docs').mkdir()
        self.texts={'d1':'Alpha University founded the Orion research project in 1984. This passage supplies the required evidence.',
                    'd2':'Beta University studied marine animals in 2001. This document is a separate source with different facts.'}
        rows=[]
        for doc_id,text in self.texts.items():
            raw=text.encode();(self.root/'docs'/f'{doc_id}.txt').write_bytes(raw)
            rows.append({'doc_id':doc_id,'path':f'docs/{doc_id}.txt','sha256':hashlib.sha256(raw).hexdigest()})
        write_jsonl(self.root/'documents.jsonl',rows)
        self.corpus=Corpus(self.root)
        self.qs=[{'id':'q1','question':'Who founded Orion?'},{'id':'q2','question':'Which university studied marine animals?'}]
        self.labels=[{'id':'q1','answer':'Alpha University','gold_doc_ids':['d1'],'evidence_doc_ids':['d1','d2']},
                     {'id':'q2','answer':'Beta University','gold_doc_ids':['d2'],'evidence_doc_ids':['d2']}]
        self.good={'id':'q1','answer':'Alpha University','doc_ids':['d2','d1'],
                   'evidence':[{'doc_id':'d1','text':self.texts['d1']}],'status':'ok',
                   'usage':{'tool_calls':2,'elapsed_seconds':1.0,'model_input_tokens':10,'model_output_tokens':3}}
        self.failed={'id':'q2','answer':'','doc_ids':[],'evidence':[],'status':'error',
                     'usage':{'tool_calls':0,'elapsed_seconds':1.0,'model_input_tokens':0,'model_output_tokens':0}}
    def tearDown(self):self.tmp.cleanup()
    def errors(self,result):return validate_results(self.qs,[result,self.failed],self.corpus)

    def test_rank_cutoffs_and_failure_denominator(self):
        self.assertEqual(self.errors(self.good),[])
        m,rows=evaluate(self.qs,[self.good,self.failed],self.labels)
        self.assertEqual(m['gold_hit_at_1'],0)
        self.assertEqual(m['gold_hit_at_5'],.5)
        self.assertEqual(m['gold_hit_at_10'],.5)
        self.assertEqual(m['evidence_recall_at_10'],.5)
        self.assertEqual(m['answer_em'],.5)
        self.assertEqual(m['effect_points_out_of_6'],3)

    def test_missing_ids_do_not_silently_shrink_evaluation(self):
        errors=validate_results(self.qs,[self.good],self.corpus)
        self.assertTrue(any('missing=1' in e for e in errors))
        metrics,_=evaluate(self.qs,[self.good],self.labels)
        self.assertEqual(metrics['queries'],2)
        self.assertEqual(metrics['gold_hit_at_10'],.5)

    def test_duplicate_result_id_rejected(self):
        with self.assertRaises(DataError):keyed([self.good,self.good],'results')

    def test_duplicate_and_unknown_documents_rejected(self):
        r=copy.deepcopy(self.good);r['doc_ids']=['d1','d1']
        self.assertTrue(any('duplicate' in e for e in self.errors(r)))
        r['doc_ids']=['not-in-corpus'];r['evidence']=[]
        self.assertTrue(any('unknown document' in e for e in self.errors(r)))

    def test_fabricated_quote_rejected(self):
        r=copy.deepcopy(self.good);r['evidence'][0]['text']='An invented claim that does not appear in the actual document.'
        self.assertTrue(any('does not match' in e for e in self.errors(r)))

    def test_quote_must_be_in_ranked_documents(self):
        r=copy.deepcopy(self.good);r['doc_ids']=['d2']
        self.assertTrue(any('not in final' in e for e in self.errors(r)))

    def test_budget_and_bad_usage_types(self):
        for value in [33,-1,True,1.5]:
            r=copy.deepcopy(self.good);r['usage']['tool_calls']=value
            self.assertTrue(any('tool-call' in e for e in self.errors(r)))
        r=copy.deepcopy(self.good);r['usage']['elapsed_seconds']=700
        self.assertTrue(any('wall-clock' in e for e in self.errors(r)))

    def test_malformed_status_does_not_crash_validator(self):
        r=copy.deepcopy(self.good);r['status']=['ok']
        self.assertTrue(any('invalid status' in e for e in self.errors(r)))

    def test_path_traversal_rejected(self):
        with self.assertRaises(DataError):self.corpus.text('../secret')
        with self.assertRaises(DataError):self.corpus.text(['d1'])

    def test_source_hash_verification(self):
        self.assertEqual(self.corpus.verify(),[])
        (self.root/'docs/d1.txt').write_text('changed')
        self.assertEqual(self.corpus.verify(),['d1'])

    def test_search_read_and_persisted_trace(self):
        trace=self.root/'traces/q1.jsonl';tools=CorpusTools(self.corpus,'q1',trace_path=trace)
        result=tools.search(['Orion'],limit=10)
        self.assertEqual(result['matches'][0]['doc_id'],'d1')
        self.assertEqual(tools.read('d1',0,50)['text'],self.texts['d1'][:50])
        self.assertEqual(len(read_jsonl(trace)),2)
        r=copy.deepcopy(self.good);r['doc_ids']=['d1']
        write_jsonl(self.root/'traces/q2.jsonl',[])
        self.assertEqual(validate_results(self.qs,[r,self.failed],self.corpus,trace_dir=self.root/'traces'),[])

    def test_unsurfaced_final_document_rejected(self):
        tools=CorpusTools(self.corpus,'q1');tools.search(['Orion'])
        r=copy.deepcopy(self.good);r['usage']['tool_calls']=1
        write_jsonl(self.root/'traces/q1.jsonl',tools.trace);write_jsonl(self.root/'traces/q2.jsonl',[])
        self.assertTrue(any('not surfaced' in e for e in validate_results(self.qs,[r,self.failed],self.corpus,trace_dir=self.root/'traces')))

    def test_tool_limit(self):
        tools=CorpusTools(self.corpus,'q1',limits={**POLICY,'max_tool_calls':1})
        tools.read('d1')
        with self.assertRaises(BudgetExceeded):tools.read('d2')

    def test_answer_normalization_preserves_numeric_meaning(self):
        self.assertEqual(normalize_answer('  ALPHA   University  '),'alpha university')
        self.assertNotEqual(normalize_answer('2.5'),normalize_answer('25'))
        self.assertNotEqual(normalize_answer('-5'),normalize_answer('5'))

    def test_nonfinite_json_rejected(self):
        p=self.root/'bad.jsonl';p.write_text('{"id":"q1","value":NaN}\n')
        with self.assertRaises(DataError):read_jsonl(p)

if __name__=='__main__':unittest.main()
