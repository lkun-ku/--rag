"""① golden 粒度复盘：标注的「块级」口径是否让指标失真？

## 背景（上一轮实测发现，不是推测）

`n027`（「工程实践中 RAG 系统有哪些常见的失败点？」）的 golden 有 **18 个块**，
**全部来自同一篇论文**（`Seven Failure Points When Engineering a RAG System`），
都讲同一主题；检索只召回 5 个 -> `recall=0.273`。

**这可能不是「检索差」，而是「标注粒度」与「检索粒度」不匹配**：
把同一篇文档的 18 个块都记为「应当全部召回」，等于要求检索覆盖**整篇论文**。

这条如果成立，会影响**所有以 recall 为门禁的判断**（本会话已被「测量口径」
坑过三次，见 4.10 / 4.12 / 4.13），所以先量清楚。

## 本脚本只做测量，不改任何代码与数据

1. 逐题统计 golden 的**块数**与**来源分布**（几个 filename、同源最大块数）；
2. 对比两种口径：
   * **块级 recall**（现状）：命中 golden 块数 / golden 块总数
   * **组级 recall**（候选）：命中「来源组」数 / 来源组总数
     （同一 `filename` 的 golden 块算 1 组，命中任一即该组命中）
3. 找出 `golden 同源块数 >= 3` 的题 —— 这些题的块级 recall 天然偏低。

**不预设哪个口径「对」**：两者各适合一类问题 ——
「问一篇论文讲了什么」适合组级，「问某表某行的值」适合块级。
本脚本只把**差异量化**，判断留给数据。
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
               "kb_manuals_bge_large_1024"]

# ---------------------------------------------------------------------------
# 建索引：块 ID -> (filename, 源集合)  —— 与 golden 用同一套指纹
# ---------------------------------------------------------------------------
client = MilvusClient(uri=MILVUS_URI)
by_id: dict = {}
for coll in COLLECTIONS:
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "text", "title"], limit=4000):
        by_id[metrics.chunk_id(r.get("text") or "")] = {
            "filename": r.get("filename", "?"), "coll": coll,
            "title": r.get("title", ""), "text": r.get("text") or ""}

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

print("=" * 104)
print("① golden 粒度复盘（纯测量：不预设哪种口径对，只把差异量化）")
print("=" * 104)
print("  库内块 %d（去重后 %d）| 题目 %d 道（含 golden 的 %d 道）" % (
    sum(1 for _ in by_id), len(by_id), len(cases),
    sum(1 for c in cases if c.get("source_blocks"))))

# ---------------------------------------------------------------------------
# 第一节：逐题的 golden 来源分布
# ---------------------------------------------------------------------------
rows = []
for c in cases:
    gids = c.get("source_blocks") or []
    if not gids:
        continue
    srcs = collections.Counter()
    unknown = []
    for g in gids:
        info = by_id.get(g)
        if info:
            srcs[info["filename"]] += 1
        else:
            unknown.append(g)
    top_fn, top_n = (srcs.most_common(1)[0] if srcs else ("?", 0))
    rows.append({"id": c.get("id"), "source": c.get("source", "?"), "q": c["q"],
                 "n_golden": len(gids), "n_files": len(srcs),
                 "top_file": top_fn, "top_n": top_n,
                 "same_ratio": round(top_n / max(len(gids), 1), 3),
                 "unknown": unknown})

rows.sort(key=lambda r: (-r["top_n"], -r["n_golden"]))
print()
print("  第一节：golden 来源分布（按「同源最大块数」降序，只列前 14 题）")
print("  %-6s %-8s %-9s %-8s %-30s %s" % ("题", "golden", "来源文件数", "同源最大", "同源文件", "占比"))
print("  " + "-" * 100)
for r in rows[:14]:
    print("  %-6s %-8d %-9d %-8d %-30s %.0f%%" % (
        r["id"], r["n_golden"], r["n_files"], r["top_n"], r["top_file"][:30], r["same_ratio"] * 100))
print()
n_multi = sum(1 for r in rows if r["top_n"] >= 3)
print("  golden 同源块数 >= 3 的题: **%d / %d**" % (n_multi, len(rows)))

# ---------------------------------------------------------------------------
# 第二节：两种口径的指标对比（跑真实检索）
# ---------------------------------------------------------------------------
from tools.retriever_tools import retriever  # noqa: E402

print()
print("=" * 104)
print("第二节：块级口径 vs 组级口径（跑真实检索，final_k=24）")
print("=" * 104)
per = []
for c in cases:
    gids = c.get("source_blocks") or []
    if not gids:
        continue
    try:
        docs = retriever.invoke(c["q"])
    except Exception as e:  # noqa: BLE001
        print("  [%s] 检索失败: %s" % (c.get("id"), type(e).__name__))
        continue
    rids = [metrics.chunk_id(d.page_content) for d in docs]
    pos = set(rids)

    block_rec = len(set(gids) & pos) / len(gids)

    # 组级：按 filename 分组（未知来源归到自身 ID，避免虚高）
    def grp(g):
        info = by_id.get(g)
        return info["filename"] if info else ("?" + g)

    gold_groups = {grp(g) for g in gids}
    hit_groups = {grp(g) for g in gids if g in pos}
    group_rec = len(hit_groups) / max(len(gold_groups), 1)

    per.append({"id": c.get("id"), "q": c["q"], "n_golden": len(gids),
                "n_groups": len(gold_groups), "block_recall": block_rec,
                "group_recall": group_rec,
                "delta": group_rec - block_rec})

n = len(per)
avg_b = sum(r["block_recall"] for r in per) / n
avg_g = sum(r["group_recall"] for r in per) / n
print("  题数 %d" % n)
print()
print("  %-14s %-12s %-12s %s" % ("口径", "平均 recall", "hit_rate", "说明"))
print("  " + "-" * 92)
for label, key in (("块级（现状）", "block_recall"), ("组级（候选）", "group_recall")):
    avg = sum(r[key] for r in per) / n
    hit = sum(1 for r in per if r[key] > 0) / n
    print("  %-14s %-12.4f %-12.4f %s" % (
        label, avg, hit,
        "命中 golden 块数 / golden 块总数" if key == "block_recall"
        else "命中的来源文件数 / 来源文件总数（同源算 1 组）"))
print()
print("  **折算到全 32 题**（run_eval 口径 = 只算有 golden 的题，其余记 0）：")
print("     块级 %.4f  |  组级 %.4f" % (avg_b * n / len(cases), avg_g * n / len(cases)))
print("     （run_eval 基线为 0.7993，块级折算 %.4f 应与之一致 —— 用来验证本脚本口径正确）"
      % (avg_b * n / len(cases)))

# ---------------------------------------------------------------------------
# 第三节：差异最大的题（组级 > 块级 说明标注跨了多块同源）
# ---------------------------------------------------------------------------
print()
print("=" * 104)
print("第三节：两种口径差异最大的题（组级 - 块级 > 0.15）")
print("=" * 104)
print("  %-6s %-8s %-8s %-10s %-10s %-8s %s" % (
    "题", "golden", "组数", "块级", "组级", "Δ", "来源"))
print("  " + "-" * 100)
big = [r for r in per if r["delta"] > 0.15]
big.sort(key=lambda r: -r["delta"])
for r in big:
    print("  %-6s %-8d %-8d %-10.3f %-10.3f %+-8.3f %s" % (
        r["id"], r["n_golden"], r["n_groups"], r["block_recall"],
        r["group_recall"], r["delta"],
        next((x["top_file"] for x in rows if x["id"] == r["id"]), "")[:34]))
print()
print("  差异题共 %d 道；其中 golden 同源块数 >= 3 的 %d 道" % (
    len(big), sum(1 for r in big if any(x["id"] == r["id"] and x["top_n"] >= 3 for x in rows))))

(HERE / "audit_golden_granularity.json").write_text(json.dumps(
    {"n_cases": len(cases), "rows": rows, "per_case": per,
     "avg_block_recall": round(avg_b, 4), "avg_group_recall": round(avg_g, 4)},
    ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/audit_golden_granularity.json")
