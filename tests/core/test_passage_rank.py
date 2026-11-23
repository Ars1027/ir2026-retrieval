"""Passage ranking and ablation tests (core)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import passage_stage, settings  # noqa: E402
from retrieval import contracts  # noqa: E402
from retrieval.passage_rank import rerank_documents  # noqa: E402


class PassageRankTests(unittest.TestCase):
    def setUp(self):
        # 建立原候选排名，并用短原文模拟局部证据集中在第二篇。
        self.texts = {
            "d1": "General historical notes from the regional archive.",
            "d2": "The scarlet observatory houses a rare telescope.",
        }
        self.docs = []
        for rank, (doc_id, body) in enumerate(self.texts.items(), 1):
            chunk = contracts.make_chunk(doc_id, 0, 0, len(body), body)
            hit = contracts.make_hit(chunk, 3.0 - rank, rank, "bm25")
            doc = contracts.make_ranked_doc(doc_id, rank, hit["score"], [hit])
            self.docs.append(doc)

        path = ROOT / "configs" / "bonus1" / "passage.json"
        config = settings.load_settings(path)
        self.options = dict(config["passage_rerank"], use_stopwords=True)

    def test_passage_only_promotes_match_and_preserves_evidence(self):
        # 检查名次被更新，原始 hit 与可追溯字符跨度都保留。
        ranked, rows = rerank_documents(
            "scarlet observatory", self.docs, self.texts.__getitem__, self.options,
        )
        self.assertEqual([doc["doc_id"] for doc in ranked], ["d2", "d1"])
        self.assertEqual(contracts.validate_ranked_docs(ranked), [])
        self.assertEqual(ranked[0]["hits"], self.docs[1]["hits"])
        self.assertEqual(self.docs[1]["rank"], 2)
        start, end = rows[0]["passage_span"]
        self.assertEqual(self.texts["d2"][start:end], self.texts["d2"])
        self.assertEqual(rows[0]["base_rank"], 2)

    def test_fusion_tie_prefers_the_base_ranking(self):
        # 两组名次刚好对调，等权融合得分相等，应稳定沿用基础名次。
        options = dict(self.options, mode="fusion")
        ranked, rows = rerank_documents(
            "scarlet observatory", self.docs, self.texts.__getitem__, options,
        )
        self.assertEqual([doc["doc_id"] for doc in ranked], ["d1", "d2"])
        self.assertEqual([row["passage_rank"] for row in rows], [2, 1])
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])

    def test_disabled_stage_preserves_base_results_without_tools(self):
        # 无重排配置的老调用方无需语料读取或额外日志。
        depth = passage_stage.candidate_depth({}, 1)
        self.assertEqual(depth, 1)
        ranked = passage_stage.rerank("scarlet", self.docs, None, {}, 1)
        self.assertEqual(ranked, self.docs[:1])


if __name__ == "__main__":
    unittest.main()
