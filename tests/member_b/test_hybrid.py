import sys
import types
import unittest
from unittest.mock import patch
from pipeline.hybrid import retrieve


class PipelineTests(unittest.TestCase):
    def test_real_stage_order_and_dense_failure(self):
        hit = dict(chunk_id='c', doc_id='d', start=0, end=20, rank=1, score=1, source='fixture')
        class Retriever:
            def __init__(self, *args): pass
            def search(self, *args): return [hit]
        class Tools:
            def __init__(self): self.stages=[]
            def record_retrieval(self, stage, *args): self.stages.append(stage)
        modules={}
        for name, attr, value in [('retrieval.dense','DenseRetriever',Retriever),('retrieval.bm25','BM25Retriever',Retriever),('retrieval.document_rank','aggregate_documents',lambda hits,k:[dict(doc_id='d',rank=1,score=1,hits=hits)])]:
            modules[name]=types.ModuleType(name);setattr(modules[name],attr,value)
        settings=dict(bm25_index_dir='unused',dense_index_dir='unused',candidate_k=20,rrf_k=60)
        with patch.dict(sys.modules,modules):
            tools=Tools()
            docs=retrieve('fictional query',tools,settings)
            self.assertEqual(tools.stages,['bm25_search','bm25_aggregate','dense_search','dense_aggregate','rrf'])
            self.assertEqual(docs[0]['source_ranks'],dict(bm25=1,dense=1))
            with patch.object(Retriever,'search',side_effect=RuntimeError('encoding failed')):
                with self.assertRaisesRegex(RuntimeError,'encoding failed'):
                    retrieve('fictional query',Tools(),settings)
