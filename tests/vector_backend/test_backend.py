import base64
from http.client import RemoteDisconnected, IncompleteRead
import json
import io
import os
import faiss
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import numpy as np
from scripts.build_vector import build,encode_retry
from vector_backend.ada import Retriever,Fusion,normalize,Encoder

class Fake:
    last_tokens=5
    calls=0
    def encode(self,texts):
        self.calls+=1
        if "cat" in texts[0]:time.sleep(.02)
        return np.asarray([[1.,0.] if "cat" in t else [0.,1.] for t in texts],dtype="float32")

class Tests(unittest.TestCase):
    def test_build_resume_mapping_and_source(self):
        with tempfile.TemporaryDirectory(prefix="中文") as folder:
            root=Path(folder);chunks=root/"chunks";chunks.mkdir()
            rows=[dict(chunk_id="original-"+str(i),doc_id="d"+str(i),start=0,end=3,text=t) for i,t in enumerate(["cat","dog"])]
            (chunks/"chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (chunks/"meta.json").write_text(json.dumps(dict(chunks=2)))
            settings=dict(chunks_dir=str(chunks),embedding=dict(dimension=2,batch_size=1),vector=dict(index_dir=str(root/"index")))
            fake=Fake();result=build(settings,fake)
            self.assertEqual(result["input_tokens"],10)
            mapping=[json.loads(line) for line in (root/"index/mapping.jsonl").read_text().splitlines()]
            self.assertEqual([row["chunk_id"] for row in mapping],["original-0","original-1"])
            retriever=Retriever(settings,fake)
            hits=retriever.search("cat",2)
            self.assertEqual(hits[0]["hits"][0]["chunk_id"],"original-0")
            self.assertEqual(retriever.last_query_tokens,5)
            metadata=root/"index/metadata.json";old=json.loads(metadata.read_text());old["complete"]=False;metadata.write_text(json.dumps(old))
            calls=fake.calls;build(settings,fake);self.assertEqual(calls,fake.calls)
            self.assertEqual(retriever.search("",2),[])
    def test_rrf(self):
        a=dict(doc_id="a",rank=1,score=1,hits=[dict(chunk_id="a")])
        b=dict(doc_id="b",rank=1,score=1,hits=[dict(chunk_id="b")])
        docs=Fusion(dict(vector={})).fuse([a,a],[b],10)
        self.assertEqual([d["doc_id"] for d in docs],["a","b"])
        self.assertAlmostEqual(docs[0]["score"],1/61)

    def test_invalid_embeddings(self):
        with self.assertRaisesRegex(ValueError,"dimension"):
            normalize([[1,2,3]],1,2)
        with self.assertRaisesRegex(ValueError,"Zero"):
            normalize([[0,0]],1,2)
        with self.assertRaisesRegex(ValueError,"values"):
            normalize([[float("nan"),1]],1,2)

    def test_retry_only_transient_http_errors(self):
        client=Fake()
        with patch.object(client,"encode",side_effect=[HTTPError("url",429,"busy",{"Retry-After":"2"},None),[[1.,0.]]]) as call, patch("scripts.build_vector.time.sleep") as sleep:
            self.assertEqual(encode_retry(client,["cat"]),[[1.,0.]])
            self.assertEqual(call.call_count,2)
            sleep.assert_called_once_with(2.)
        with patch.object(client,"encode",side_effect=HTTPError("url",401,"denied",{},None)) as call:
            with self.assertRaises(HTTPError):encode_retry(client,["cat"])
            self.assertEqual(call.call_count,1)
        with patch.object(client,"encode",side_effect=HTTPError("url",503,"busy",{},None)) as call, patch("scripts.build_vector.time.sleep"):
            with self.assertRaises(HTTPError):encode_retry(client,["cat"])
            self.assertEqual(call.call_count,4)

    def test_chunk_rank_survives_document_deduplication(self):
        retriever=Retriever.__new__(Retriever)
        retriever.encoder=Fake()
        retriever.chunks=[dict(chunk_id=str(i),doc_id=doc,start=0,end=3) for i,doc in enumerate(["a","a","b"])]
        retriever.index=faiss.IndexFlatIP(2)
        retriever.index.add(np.array([[1.,0.],[.8,.6],[0.,1.]],dtype="float32"))
        docs=retriever.search("cat",2)
        self.assertEqual([doc["rank"] for doc in docs],[1,2])
        self.assertEqual([doc["hits"][0]["rank"] for doc in docs],[1,3])

    def test_api_usage_missing_stays_unknown(self):
        encoder=Encoder(dict(model="fixture",base_url="https://example.invalid/v1",dimension=2))
        response={"data":[{"index":0,"embedding":[1.,0.]}]}
        with patch.dict(os.environ,{"POLOAI_API_KEY":"synthetic-test-key"}), patch("vector_backend.ada.urllib.request.urlopen",return_value=io.BytesIO(json.dumps(response).encode())):
            encoder.encode(["fictional cat"])
        self.assertIsNone(encoder.last_tokens)

    def test_base64_response_restores_input_order_and_dimension(self):
        def encoded(values):
            return base64.b64encode(np.asarray(values,dtype="<f4").tobytes()).decode("ascii")
        encoder=Encoder(dict(model="fixture",base_url="https://example.invalid/v1",dimension=2))
        response={"data":[{"index":1,"embedding":encoded([0.,5.])},{"index":0,"embedding":encoded([3.,0.])}],"usage":{"prompt_tokens":11}}
        with patch.dict(os.environ,{"POLOAI_API_KEY":"synthetic-test-key"}), patch("vector_backend.ada.urllib.request.urlopen",return_value=io.BytesIO(json.dumps(response).encode())) as request:
            values=encoder.encode(["fictional cat","fictional dog"])
        np.testing.assert_allclose(values,[[1.,0.],[0.,1.]])
        self.assertEqual(encoder.last_tokens,11)
        self.assertEqual(json.loads(request.call_args.args[0].data)["encoding_format"],"base64")
        response={"data":[{"index":0,"embedding":encoded([1.,2.,3.])}]}
        with patch.dict(os.environ,{"POLOAI_API_KEY":"synthetic-test-key"}), patch("vector_backend.ada.urllib.request.urlopen",return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError,"dimension"):
                encoder.encode(["fictional cat"])

    def test_network_retries_have_fixed_limit(self):
        client=Fake()
        errors=[URLError("connection failed"),RemoteDisconnected("closed"),IncompleteRead(b"partial",8),TimeoutError("timed out")]
        for error in errors:
            with self.subTest(kind=type(error).__name__):
                with patch.object(client,"encode",side_effect=error) as call, patch("scripts.build_vector.time.sleep") as sleep:
                    with self.assertRaises(type(error)):encode_retry(client,["cat"])
                    self.assertEqual(call.call_count,4)
                    self.assertEqual([c.args[0] for c in sleep.call_args_list],[1,2,4])
                with patch.object(client,"encode",side_effect=[error,[[1.,0.]]]) as call, patch("scripts.build_vector.time.sleep"):
                    self.assertEqual(encode_retry(client,["cat"]),[[1.,0.]])
                    self.assertEqual(call.call_count,2)

    def test_bge_config_and_index_do_not_accept_ada_model(self):
        project=Path(__file__).resolve().parents[2]
        bge=json.loads((project/"configs/vector.bge.json").read_text(encoding="utf-8"))
        ada=json.loads((project/"configs/vector.ada.json").read_text(encoding="utf-8"))
        self.assertEqual(bge["embedding"]["model"],"baai/bge-m3")
        self.assertEqual(bge["embedding"]["dimension"],1024)
        self.assertEqual(bge["embedding"]["api_key_env"],"OPENROUTER_API_KEY")
        self.assertNotEqual(bge["vector"]["index_dir"],ada["vector"]["index_dir"])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);chunks=root/"chunks";chunks.mkdir()
            (chunks/"meta.json").write_text('{"chunks": 1}')
            row=dict(chunk_id="original",doc_id="doc",start=0,end=3,text="cat")
            (chunks/"chunks.jsonl").write_text(json.dumps(row))
            settings=dict(chunks_dir=str(chunks),embedding=bge["embedding"],vector=dict(index_dir=str(root/"index")))
            class Fixture1024:
                last_tokens=1
                def encode(self,texts):
                    values=np.zeros((len(texts),1024),dtype="float32");values[:,0]=1
                    return values
            fixture=Fixture1024();build(settings,fixture)
            self.assertEqual(Retriever(settings,fixture).index.d,1024)
            changed=dict(settings,embedding=ada["embedding"])
            with self.assertRaisesRegex(ValueError,"incompatible"):
                Retriever(changed,fixture)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                build(changed,fixture)

    def test_summary_reads_build_wall_seconds(self):
        from scripts.summarize_runs import _index_costs
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);index=root/"vector";index.mkdir()
            (index/"metadata.json").write_text(json.dumps(dict(build_wall_seconds=12.5,encoding_seconds=60.)))
            (root/"method_config.json").write_text(json.dumps(dict(vector=dict(enabled=True,index_dir=str(index)))))
            costs=_index_costs(root)["vector"]
            self.assertEqual(costs["build_wall_seconds"],12.5)
            self.assertNotIn("wall_seconds",costs)

    def test_overloaded_large_batch_splits_without_reordering(self):
        class Overloaded:
            last_tokens = None
            def __init__(self, unknown=False):
                self.calls=[]
                self.unknown=unknown
            def encode(self, texts):
                self.calls.append(list(texts))
                if len(texts)>8:raise HTTPError("url",429,"engine overloaded",{},None)
                self.last_tokens=None if self.unknown and texts[0]=="8" else len(texts)*2
                return np.asarray([[int(text),1.] for text in texts],dtype="float32")
        texts=[str(i) for i in range(19)]
        for unknown in (False,True):
            client=Overloaded(unknown)
            with patch("scripts.build_vector.time.sleep") as sleep:
                vectors=encode_retry(client,texts)
            self.assertEqual(vectors[:,0].tolist(),list(range(19)))
            self.assertEqual([len(batch) for batch in client.calls],[19,8,8,3])
            self.assertEqual(client.last_tokens,None if unknown else 38)
            sleep.assert_not_called()

    def test_small_overloaded_batch_retains_bounded_backoff(self):
        client=Fake()
        with patch.object(client,"encode",side_effect=HTTPError("url",429,"overloaded",{},None)) as call, patch("scripts.build_vector.time.sleep") as sleep:
            with self.assertRaises(HTTPError):encode_retry(client,["cat"]*8)
            self.assertEqual(call.call_count,4)
            self.assertEqual([c.args[0] for c in sleep.call_args_list],[60,120,180])
