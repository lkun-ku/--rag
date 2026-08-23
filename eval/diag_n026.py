"""n026 深挖：为什么 8 个 golden 块只召回 1 个？（重建前 0.462 -> 现在 0.12）

问题：「Milvus 支持哪些相似度度量类型？」

## 要分清两种可能（修法完全不同）

    A. **检索层**：golden 块没进 final_k=24 -> 可能要调 retrieval，或 golden 本身不完备；
    B. **golden 层**：那 7 个「漏召回」的块其实**不在任何一个集合里**
       （重建后块 ID 变了，而 4.18 的修订脚本只处理「失效」的 ID，
       **没有处理「块还在但内容被重新切分、ID 变了」的情况**）。

先查 B（便宜且常见）：把 7 个漏召回的 ID 拿去备份里查，
看它们**重建前**属于哪个文件、是什么形态 —— 若全是扁平块，说明它们**本来就该消失**。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
import metrics  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402

QID = "n026"
MISSED = ["1b3292b1589a", "259df3338d91", "4d43889ea877", "4fddcc128c87",
          "5045647d914c", "6f8ea5911e19", "fb6eb0bb850d"]

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
case = next(c for c in cases if c.get("id") == QID)
GOLD = set(case["source_blocks"])

print("=" * 104)
print("n026 深挖")
print("=" * 104)
print("  问题: %s" % case["q"])
print("  golden（%d 块）: %s" % (len(GOLD), sorted(GOLD)))
print()

# --- 第一步：这 7 个块在「重建前」是什么形态？ ---
bk = sorted((HERE / "_backup_collections").glob("backup_*.json"))[-1]
old = json.loads(bk.read_text(encoding="utf-8"))
old_by_id = {}
for coll, v in old["collections"].items():
    for it in v.get("items", []):
        old_by_id[it["chunk_id"]] = it

print("=" * 104)
print("第一步：7 个「漏召回」的 golden 块，重建前是什么形态？（关键）")
print("=" * 104)
print("  %-14s %-16s %-30s %s" % ("ID", "形态", "文件", "块首行"))
print("  " + "-" * 100)
n_flat = n_struct = n_absent = 0
for g in MISSED:
    it = old_by_id.get(g)
    if not it:
        n_absent += 1
        print("  %-14s %-16s %-30s %s" % (g, "**备份里没有**", "", ""))
        continue
    t = it["text"]
    head = t.lstrip()[:40]
    if head.startswith("表格："):
        shape, n_struct = "结构化(表格：)", n_struct + 1
    elif "列名：" in t[:500]:
        shape, n_struct = "结构化(列名：)", n_struct + 1
    else:
        shape, n_flat = "扁平", n_flat + 1
    print("  %-14s %-16s %-30s %s" % (g, shape, (it.get("filename") or "")[:30],
                                      t[:52].replace("\n", " ")))
print()
print("  统计: 扁平 %d / 结构化 %d / 备份里没有 %d" % (n_flat, n_struct, n_absent))

# --- 第二步：现在检索到底召回了什么 ---
print()
print("=" * 104)
print("第二步：当前检索结果（final_k=24）里有什么")
print("=" * 104)
from tools.retriever_tools import retriever  # noqa: E402

docs = retriever.invoke(case["q"])
ids = [metrics.chunk_id(d.page_content) for d in docs]
print("  检索到 %d 块；golden 命中 %d/%d" % (
    len(docs), len(GOLD & set(ids)), len(GOLD)))
print()
print("  %-4s %-6s %-30s %s" % ("#", "golden", "文件", "块首行"))
print("  " + "-" * 100)
for i, d in enumerate(docs, 1):
    cid = metrics.chunk_id(d.page_content)
    print("  %-4d %-6s %-30s %s" % (
        i, "GOLD" if cid in GOLD else "", (d.metadata.get("filename") or "")[:30],
        d.page_content.replace("\n", " ")[:56]))

# --- 第三步：命中的那个 golden 块 + 一个漏召回的块，直接搜一下 ---
print()
print("=" * 104)
print("第三步：把「漏召回的 golden 块正文」直接当查询（看检索器能否找到它）")
print("=" * 104)
client = MilvusClient(uri=MILVUS_URI)
new_text_by_id = {}
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
             "kb_manuals_bge_large_1024"):
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "text"], limit=8000):
        t = r.get("text") or ""
        new_text_by_id[metrics.chunk_id(t)] = t
print("  新库块数: %d" % len(new_text_by_id))
for g in sorted(GOLD):
    where = "在库里" if g in new_text_by_id else "**不在库里**"
    print("     golden %s  %s" % (g, where))
