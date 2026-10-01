"""决定性实验：n005 拒答的根因是「表格碎片」还是「模型能力」？

## 设计

两组上下文，各跑 3 次（`temperature=0`，但服务端仍可能有细微差异）：

| 组 | 上下文 | 字符数 |
|---|---|---|
| **A. 生产** | 检索+判分后的 10 块（2 个是表格片段） | ~17,000 |
| **B. 完整表** | 该表**全部 4 个切片拼接**成一张完整表 | ~8,600 |

## 判读

* A 抖动 / B 稳定答对 -> 根因是**表格碎片**，应做「同表切片合并」
* A、B 都稳定拒答      -> 根因是**模型不会跨行聚合**，要靠 prompt 或结构化查询
* A、B 都稳定答对      -> 之前的拒答是**偶发抖动**，无需改造

**每组跑 3 次**，是为了区分「稳定」与「抖动」——
之前踩过「单轮对比不可靠」的坑（见 `generate_node2.py` 注释里的教训）。
"""

from __future__ import annotations

import json
import os
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.utils.citation import is_refusal  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from rag_kb.graph2.generate_node2 import generate, format_docs  # noqa: E402
from rag_kb.graph2.graph_2 import run_with_trace  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402
from rag_kb.utils.env_utils import MILVUS_URI  # noqa: E402

QID = "n005"
TARGET_FILE = "vector_db_benchmark.xlsx.md"
TARGET_TITLE = "检索性能实测"
RUNS = 3

cases = {}
for line in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        cases[d.get("id")] = d
C = cases[QID]
Q, GOLD = C["q"], C.get("answer")

print("=" * 100)
print("n005 表格碎片 vs 完整表（每组 %d 次）" % RUNS)
print("=" * 100)
print("  问题  : %s" % Q)
print("  golden: %s" % GOLD)
print()


def ask(ctx_docs, label):
    """对给定 documents 跑 generate，返回 (是否拒答, 是否命中, 摘要)。"""
    st = {"question": Q, "documents": ctx_docs}
    out = generate(dict(st))
    g = (out or {}).get("generation") or ""
    refused = is_refusal(g)
    hit = ("61.35" in g) and ("turbopuffer" in g.lower())
    return refused, hit, g[:150].replace("\n", " ")


# ---- A. 生产上下文 ----------------------------------------------------------
print("=" * 100)
print("A. 生产上下文（跑一次全图拿到真实的 10 块）")
print("=" * 100)
out = run_with_trace(Q)
prod_docs = out.get("documents") or []
if not isinstance(prod_docs, list):
    prod_docs = [prod_docs]
prod_ctx = format_docs(prod_docs)
print("  块数 %d / %d 字符 / 含 61.35: %s" % (len(prod_docs), len(prod_ctx), "61.35" in prod_ctx))

a_res = []
for i in range(RUNS):
    r, h, txt = ask(prod_docs, "A")
    a_res.append((r, h))
    print("  第%d次: 拒答=%-3s 命中=%-3s | %s" % (i + 1, "是" if r else "否", "是" if h else "否", txt[:92]))

# ---- B. 完整表 -------------------------------------------------------------
print()
print("=" * 100)
print("B. 完整表（该表全部切片拼接）")
print("=" * 100)
client = MilvusClient(uri=MILVUS_URI)
parts = []
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024", "kb_manuals_bge_large_1024"):
    try:
        rows = client.query(collection_name=coll, filter="", output_fields=["filename", "title", "text"], limit=8000)
    except Exception:  # noqa: BLE001
        continue
    for r in rows:
        if r.get("filename") == TARGET_FILE and (r.get("title") or "") == TARGET_TITLE:
            parts.append(r.get("text") or "")


def _start_line(t: str) -> int:
    m = re.search(r"第\s*(\d+)\s*-\s*(\d+)\s*行", t)
    return int(m.group(1)) if m else 10**9


parts.sort(key=_start_line)
full_lines, seen = [], set()
for p in parts:
    for l in p.splitlines():
        if l.startswith("表格：") or l.startswith("列名："):
            continue
        if l.startswith("|") and set(l.strip()) <= set("|-: "):
            if "sep" in seen:
                continue
            seen.add("sep")
        if l.startswith("|") and ("场景" in l or "产品与规格" in l):
            if "hdr" in seen:
                continue
            seen.add("hdr")
        full_lines.append(l)
full_text = "\n".join(full_lines)
print(
    "  切片 %d 个 -> 拼接 %d 行 / %d 字符 / 含 61.35: %s"
    % (len(parts), len(full_lines), len(full_text), "61.35" in full_text)
)

b_docs = [Document(page_content=full_text, metadata={"filename": TARGET_FILE, "title": TARGET_TITLE})]
b_res = []
for i in range(RUNS):
    r, h, txt = ask(b_docs, "B")
    b_res.append((r, h))
    print("  第%d次: 拒答=%-3s 命中=%-3s | %s" % (i + 1, "是" if r else "否", "是" if h else "否", txt[:92]))

# ---- 判读 ------------------------------------------------------------------
print()
print("=" * 100)
print("判读")
print("=" * 100)


def summary(res):
    n_ref = sum(1 for r, _ in res if r)
    n_hit = sum(1 for _, h in res if h)
    kind = "稳定拒答" if n_ref == RUNS else ("稳定答对" if n_hit == RUNS else "抖动")
    return "%d/%d 拒答 · %d/%d 命中 -> **%s**" % (n_ref, RUNS, n_hit, RUNS, kind)


print("  A 生产上下文: %s" % summary(a_res))
print("  B 完整表    : %s" % summary(b_res))
print()
a_kind = "稳定拒答" if all(r for r, _ in a_res) else ("稳定答对" if all(h for _, h in a_res) else "抖动")
b_kind = "稳定拒答" if all(r for r, _ in b_res) else ("稳定答对" if all(h for _, h in b_res) else "抖动")

if a_kind == "抖动":
    print("  -> A 是**抖动**：这是不稳定样本，单轮评测的结论不可靠。")
elif b_kind == "稳定答对" and a_kind == "稳定拒答":
    print("  -> **根因是表格碎片**：完整表能稳定答对，碎片上下文稳定拒答。")
    print("     => 应做「同表切片合并」（检索后把同一张表的切片拼回完整表）")
elif a_kind == "稳定拒答" and b_kind == "稳定拒答":
    print("  -> **根因不是碎片**：完整表也拒答，说明模型不会跨行聚合。")
    print("     => 要靠 prompt 引导（明确「表格中找最小值」）或结构化查询")
elif a_kind == "稳定答对":
    print("  -> A 已能稳定答对：之前的拒答属偶发，**无需改造**。")
