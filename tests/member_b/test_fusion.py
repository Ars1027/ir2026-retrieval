import unittest
from retrieval.fusion import fuse_rrf


def doc(name, rank):
    return {"doc_id": name, "rank": rank, "score": 100, "hits": [{"chunk_id": name, "doc_id": name, "start": 0, "end": 20}]}


class FusionTests(unittest.TestCase):
    def test_hand_calculation_duplicates_and_spans(self):
        result = fuse_rrf([doc("a", 1), doc("a", 2), doc("b", 3)], [doc("b", 1), doc("a", 2)], 60, 10)
        self.assertEqual(result[0]["doc_id"], "a")
        self.assertAlmostEqual(result[0]["score"], 1/61+1/62)
        self.assertEqual(result[0]["source_ranks"], {"bm25": 1, "dense": 2})
        self.assertEqual(len(result[0]["hits"]), 2)
        self.assertEqual(result[0]["hits"][0]["end"], 20)

    def test_empty_ties_and_limit(self):
        self.assertEqual(fuse_rrf([], [], 60, 10), [])
        self.assertEqual([d["doc_id"] for d in fuse_rrf([doc("z", 1)], [doc("a", 1)], 60, 10)], ["a", "z"])
        self.assertEqual(len(fuse_rrf([doc(str(i), i+1) for i in range(20)], [], 60, 30)), 10)
        self.assertEqual(fuse_rrf([doc("a", 1)], [], 60, 0), [])
