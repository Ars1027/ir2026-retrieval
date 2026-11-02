import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from retrieval.dense import build_index, DenseRetriever

class FakeEncoder:
    last_tokens = 1
    def encode_documents(self, texts):
        return np.array([[1,0] if t == 'cat' else [0,1] for t in texts], dtype='float32')
    def encode_query(self, query):
        return np.array([[1,0]], dtype='float32')

class DenseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='\u4e2d\u6587')
        self.path = Path(self.tmp.name)/'index'
        self.settings = {'embedding': {'dimension':2,'batch_size':1,'revision':'fixture'}}
        self.chunks = [dict(chunk_id='original-'+str(i),doc_id='doc-'+str(i), start=0,end=3,text=t) for i,t in enumerate(['cat','dog'])]
    def tearDown(self):
        self.tmp.cleanup()
    def build(self, chunks=None):
        build_index(self.chunks if chunks is None else chunks, {'version':1},self.path,self.settings,FakeEncoder())
        return DenseRetriever(self.path,self.settings,FakeEncoder())
    def test_reload_mapping_and_scores(self):
        a=self.build().search('feline',2)
        b=DenseRetriever(self.path,self.settings,FakeEncoder()).search('feline',2)
        self.assertEqual(a,b)
        self.assertEqual(a[0]['chunk_id'],'original-0')
        self.assertEqual(a[0]['score'],1)
        self.assertEqual(a[0]['start'],0)
    def test_empty(self):
        self.assertEqual(self.build([]).search('query',10),[])
    def test_empty_query(self):
        self.assertEqual(self.build().search('',10),[])
    def test_version_mismatch(self):
        self.build()
        self.settings['embedding']['revision']='different'
        with self.assertRaisesRegex(ValueError,'mismatch'):
            DenseRetriever(self.path,self.settings,FakeEncoder())
    def test_dimension_mismatch(self):
        self.settings['embedding']['dimension']=3
        with self.assertRaisesRegex(ValueError,'dimension'):
            self.build()
    def test_no_overwrite(self):
        self.build()
        with self.assertRaises(FileExistsError):self.build()
    def test_source_metadata_mismatch(self):
        self.build()
        self.settings['chunks_dir']=self.tmp.name
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):
            DenseRetriever(self.path,self.settings,FakeEncoder())

if __name__ == '__main__':unittest.main()
