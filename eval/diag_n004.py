"""n004 逐块诊断：判分是「误杀」还是「golden 补错了」？

## 背景

③ 起点实测（4.19）发现 n004「Pinecone-p2.x8-1node 在流式写入下的 QPS 是多少？」
的 golden 块 `408208b67503` **被判分筛掉**（检索后 golden 命中 1 -> 判分后 0）。

这可能有两种完全不同的原因，**修法也完全不同**：

  A. **判分误杀**：块里**确实有**那行数据，但 LLM 判了 no -> 要改判分环节；
  B. **golden 补错**：重建时用内容匹配补进来的块里**没有**那行数据
     -> 要改 golden，与判分无关。

本脚本把两者分开：先看块里有没有 Pinecone/流式写入/QPS 的那一行，
再逐块打印判分结果与块的头部，让判定依据可见。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402
from graph2.grader_chain import retrieval_grader_chain  # noqa: E402

QID = "n004"
cases = {
    json.loads(l).get("id"): json.loads(l)
    for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}
case = cases[QID]
Q = case["q"]
GOLD = set(case["source_blocks"])

print("=" * 104)
print("n004 逐块诊断")
print("=" * 104)
print("  问题: %s" % Q)
print("  golden: %s" % sorted(GOLD))
print()

docs = retriever.invoke(Q)
print("  检索到 %d 块" % len(docs))
print()

# --- 第一步：golden 块里到底有没有那行数据？ ---
hit = None
for d in docs:
    cid = metrics.chunk_id(d.page_content)
    if cid in GOLD:
        hit = d
        break
if hit is None:
    print("  [!] golden 块**不在本次检索结果里**（那是另一类问题：检索没召回）")
else:
    t = hit.page_content
    print("=" * 104)
    print("  第一步：golden 块 `%s` 里有没有「Pinecone 流式写入」那一行？" % metrics.chunk_id(t))
    print("=" * 104)
    lines = t.splitlines()
    found = [ln for ln in lines if "Pinecone" in ln or "pinecone" in ln]
    print("  该块共 %d 行；含 'Pinecone' 的行 %d 条:" % (len(lines), len(found)))
    for ln in found[:8]:
        print("     %s" % ln[:150])
    stream = [ln for ln in lines if "流式写入" in ln]
    print()
    print("  含 '流式写入' 的行 %d 条:" % len(stream))
    for ln in stream[:8]:
        print("     %s" % ln[:150])
    print()
    both = [ln for ln in lines if "Pinecone" in ln and "流式写入" in ln]
    print("  **同时含两者（即答案行）: %d 条**" % len(both))
    for ln in both[:4]:
        print("     %s" % ln[:170])
    print()

# --- 第二步：逐块判分，看 golden 块被判成什么 ---
print("=" * 104)
print("  第二步：逐块判分（真实判分链），看 golden 块与它的「邻居」")
print("=" * 104)
print("  %-6s %-8s %-10s %s" % ("序号", "golden", "判定", "块首行"))
print("  " + "-" * 100)
for i, d in enumerate(docs, 1):
    cid = metrics.chunk_id(d.page_content)
    try:
        sc = retrieval_grader_chain.invoke({"question": Q, "document": d.page_content})
        v = str(getattr(sc, "binary_score", "")).strip().lower()
    except Exception as e:  # noqa: BLE001
        v = "ERR:%s" % type(e).__name__
    mark = "GOLD" if cid in GOLD else "    "
    head = d.page_content.replace("\n", " ")[:60]
    print("  %-6d %-8s %-10s %s" % (i, mark, v, head))
