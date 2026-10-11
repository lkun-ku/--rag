"""可行性验证 2：dense 相似度绝对值能否作为「分级判分」的分流信号。

背景：验证 1 已证明**排名**不可用（top-5 的相关率仅 40%）。RRF 分数是题内相对值，
      且当前检索器未把分数写进 metadata。本脚本直接从 Milvus 取 distance，
      检验「相似度绝对值 → 相关性」是否存在单调关系。若有，则改造成本
      （暴露 score 到 metadata）值得；若没有，分级判分应放弃。

方法：同 5 题 → Milvus dense 检索 top-12（带 distance）→ 逐块 LLM 判分 →
     按 distance 分桶统计 yes 率，并对比「按 distance 排序」与「按 RRF 排名」的纯度。

用法：python _probe_score.py [题目数] [topk]
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COLLECTION = "kb_auto_eec008a3_bge_large_1024"
BUCKETS = [(0.70, 1.01), (0.60, 0.70), (0.50, 0.60), (0.40, 0.50), (0.0, 0.40)]


def main() -> int:
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    gold = ROOT / "eval" / "golden_auto.jsonl"
    all_cases = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    pool = [c for c in all_cases if "eec008a3" in (c.get("source") or "")]
    cases = pool[:n_cases]

    from pymilvus import MilvusClient

    from rag_kb.graph_crag.grade_documents_node import _grade_one
    from rag_kb.llm_models.embeddings_model import bge_embedding
    from rag_kb.utils.env_utils import MILVUS_URI
    from langchain_core.documents import Document

    client = MilvusClient(uri=MILVUS_URI)
    stat = {b: {"yes": 0, "n": 0} for b in BUCKETS}
    rows_all = []

    for i, case in enumerate(cases, 1):
        q = case["q"]
        vec = bge_embedding.embed_query(q)
        res = client.search(
            collection_name=COLLECTION, data=[vec], anns_field="dense", limit=topk,
            output_fields=["text", "filename"], search_params={"metric_type": "IP", "params": {"ef": 128}},
        )[0]
        hits = []
        for r in res:
            d = r["distance"]
            doc = Document(page_content=r["entity"]["text"], metadata={"filename": r["entity"].get("filename")})
            ok = _grade_one(q, doc)[1]
            hits.append((d, ok))
            for lo, hi in BUCKETS:
                if lo <= d < hi:
                    stat[(lo, hi)]["n"] += 1
                    stat[(lo, hi)]["yes"] += 1 if ok else 0
                    break
        rows_all.extend(hits)
        print("  [%d] distance %.3f~%.3f，yes %d/%d | %s" % (
            i, res[0]["distance"], res[-1]["distance"], sum(1 for _, o in hits if o), len(hits), q[:36]))

    print()
    print("=" * 76)
    print("按 dense 相似度（IP）分桶的相关率")
    print("=" * 76)
    print("%-14s %8s %8s %8s" % ("distance 段", "块数", "判yes", "yes率"))
    for b in BUCKETS:
        v = stat[b]
        rate = v["yes"] / v["n"] * 100 if v["n"] else 0
        print("%-14s %8d %8d %7.1f%%" % ("[%.2f, %.2f)" % b, v["n"], v["yes"], rate))

    total = len(rows_all)
    yes = sum(1 for _, o in rows_all if o)
    print()
    print("整体相关率: %d/%d = %.1f%%" % (yes, total, yes / total * 100))

    # 若在 0.60 以上放行、0.40 以下丢弃
    hi = [o for d, o in rows_all if d >= 0.60]
    lo = [o for d, o in rows_all if d < 0.40]
    mid = [o for d, o in rows_all if 0.40 <= d < 0.60]
    print("若「>=0.60 放行 / <0.40 丢弃 / 中间判分」：")
    print("  放行 %d 块，其中无关 %d 块（%.0f%%）" % (len(hi), len(hi) - sum(hi), (len(hi) - sum(hi)) / len(hi) * 100 if hi else 0))
    print("  丢弃 %d 块，其中相关 %d 块（漏掉 %.0f%%）" % (len(lo), sum(lo), sum(lo) / len(lo) * 100 if lo else 0))
    print("  判分调用 %d -> %d，节省 %.0f%%" % (total, len(mid), (total - len(mid)) / total * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
