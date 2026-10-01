"""分数探针：确认「哪条路径能拿到可用的分数」。

## 背景

P0-8 提议「用相似度门槛过滤噪声」，但落地前要确认**哪种分数可用**。
源码已确认三个死路（langchain_milvus/vectorstores/milvus.py）：

    1902  if self._is_multi_vector: -> _collection_hybrid_search()
          （vector_field=['dense','sparse'] 必走 hybrid，无开关）
    2136  _select_relevance_score_fn 在多向量时 raise ValueError
          （=> as_retriever(search_type="similarity_score_threshold") 不可用）
    2484  默认 WeightedRanker（非 RRF）

本探针直接把三条路径的**原始返回**打出来对比：

    路径 A  client.search(anns_field="dense")            -> IP 分数
    路径 B  client.hybrid_search(RRFRanker)              -> RRF 融合分
    路径 C  client.hybrid_search(WeightedRanker)         -> 加权归一化分

判断标准：**golden 块的分数能否与非 golden 块分开**（有分界 = 门槛可行）。

**本脚本不吞异常** —— 上一版把 MilvusException 吞成空列表，直接导致
recall 从 0.7492 假跌到 0.0333 却「看起来跑通了」。错误必须立刻可见。
"""

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker  # noqa: E402

from rag_kb.llm_models.embeddings_model import bge_embedding  # noqa: E402
from rag_kb.utils.env_utils import MILVUS_URI  # noqa: E402

E = ROOT / "eval"


def fp(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


rows = [json.loads(l) for l in (E / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
picked = [r for r in rows if r.get("source_blocks") and r.get("source") == "manuals"][:3]
print("探针题目: %s" % [r["id"] for r in picked])

client = MilvusClient(uri=MILVUS_URI)

# 先看新语料的 category 分布（决定 filter 该怎么写）
COLL = "kb_manuals_bge_large_1024"
try:
    allrows = client.query(collection_name=COLL, filter="", output_fields=["category"], limit=2000)
    from collections import Counter

    print("category 分布: %s" % dict(Counter(r.get("category") for r in allrows)))
except Exception as e:  # noqa: BLE001
    print("category 查询失败: %s: %s" % (type(e).__name__, e))

OUT = ["text", "filename", "category"]


def show(tag, hits, gold):
    print()
    print("  --- %s ---" % tag)
    for i, h in enumerate(hits, 1):
        ent = h.get("entity", {}) or {}
        txt = ent.get("text") or ""
        key = fp(txt)
        mark = "  <== GOLDEN" if key in gold else ""
        print(
            "    %2d. score=%-10.5f %-34s %s%s"
            % (i, h.get("distance"), str(ent.get("filename"))[:34], txt[:46].replace("\n", " "), mark)
        )


for case in picked:
    q, gold = case["q"], set(case["source_blocks"])
    print()
    print("=" * 104)
    print("[%s] %s" % (case["id"], q[:96]))
    print("  golden blocks: %s" % sorted(gold))
    qv = bge_embedding.embed_query(q)

    # 路径 A：纯 dense（IP）
    try:
        hits = client.search(
            collection_name=COLL,
            data=[qv],
            anns_field="dense",
            search_params={"metric_type": "IP", "params": {}},
            limit=10,
            output_fields=OUT,
        )
        show("A. dense only (IP)", hits[0] if hits else [], gold)
    except Exception as e:  # noqa: BLE001
        print("  路径 A 失败: %s: %s" % (type(e).__name__, e))

    # 路径 B：hybrid + RRF（与生产一致）
    try:
        reqs = [
            AnnSearchRequest(data=[qv], anns_field="dense", param={"metric_type": "IP"}, limit=12),
            AnnSearchRequest(data=[q], anns_field="sparse", param={"metric_type": "BM25"}, limit=12),
        ]
        hits = client.hybrid_search(collection_name=COLL, reqs=reqs, ranker=RRFRanker(100), limit=10, output_fields=OUT)
        show("B. hybrid + RRFRanker(100)", hits[0] if hits else [], gold)
    except Exception as e:  # noqa: BLE001
        print("  路径 B 失败: %s: %s" % (type(e).__name__, e))

    # 路径 C：hybrid + WeightedRanker(1,1)
    try:
        hits = client.hybrid_search(
            collection_name=COLL, reqs=reqs, ranker=WeightedRanker(1.0, 1.0), limit=10, output_fields=OUT
        )
        show("C. hybrid + WeightedRanker(1,1)", hits[0] if hits else [], gold)
    except Exception as e:  # noqa: BLE001
        print("  路径 C 失败: %s: %s" % (type(e).__name__, e))

print()
print("=" * 104)
print("探针结束。判断：哪种分数的 golden / 非 golden 分界最清晰？")
