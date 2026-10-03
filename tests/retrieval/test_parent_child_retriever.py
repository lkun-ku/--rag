"""Parent-Child 检索器 —— 契约测试（ROADMAP 4.52）。

**全部用假 client / 假 retriever**，不需要真实 Milvus —— 检索流程与排序公式
与语料无关，可完全离线验证。

## 契约

| # | 契约 |
|---|---|
| R1 | 子块检索的 expr **必须**含 `is_parent == false`（否则父块会被当子块检索）|
| R2 | 父块按 **命中子块数** 降序 |
| R3 | 命中数相同 -> 按 **最高子块相似度** 降序 |
| R4 | 父块回查 **一次批量**（`client.query` 只调 1 次，不做 N+1）|
| R5 | `top_n_parents` 截断生效 |
| R6 | 无命中 -> `[]`（正常的「没找到」，不是降级）|
| R7 | 父块 metadata 带 `matched_children`（命中的子块原文，供生成时标注）|
| R8 | **开关关闭 -> 返回 `None`**（与「无命中返回 []」严格区分）|
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT)]

from langchain_core.documents import Document  # noqa: E402

from rag_kb.tools import parent_child_retriever as pcr  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if cond:
        print("  [OK ] %s" % name)
    else:
        print("  [FAIL] %s  %s" % (name, detail))
        FAILED.append(name)


class FakeChildRetriever:
    def __init__(self, kids):
        self.kids = kids

    def invoke(self, question):
        return list(self.kids)


class FakeClient:
    """记录 query 调用次数与 filter，用于验证「批量 vs N+1」。"""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, collection_name, filter="", output_fields=None, limit=None):
        self.calls.append(filter)
        return list(self.rows)


def kid(pid, sim, text="子块内容"):
    return Document(page_content=text, metadata={"parent_id": pid, "score": sim})


def parent_row(pid, text="父块正文"):
    return {"parent_id": pid, "text": text, "heading_path": "第一章", "filename": "a.md"}


def run(kids, rows, top_n=6):
    """跑一次，临时把开关打开。"""
    _orig = pcr.PARENT_CHILD_ENABLED
    pcr.PARENT_CHILD_ENABLED = True
    try:
        return pcr.retrieve_with_parents(
            "问题", FakeChildRetriever(kids), FakeClient(rows), "coll",
            top_k_children=24, top_n_parents=top_n,
        ), pcr.CHILD_FILTER
    finally:
        pcr.PARENT_CHILD_ENABLED = _orig


print("=" * 100)
print("R1/R4 expr 与批量回查")
print("=" * 100)
check("R1 CHILD_FILTER 含 is_parent == false",
      "is_parent == false" in pcr.CHILD_FILTER, pcr.CHILD_FILTER)
expr = pcr.build_parent_expr(["p1", "p2"])
check("R1 父块 expr 含 is_parent == true", "is_parent == true" in expr, expr)
check("R1 父块 expr 含 in [...]", "parent_id in [" in expr, expr)

clients = []
_kids = [kid("p1", 0.9), kid("p1", 0.8), kid("p2", 0.7)]
_rows = [parent_row("p1"), parent_row("p2")]
_orig = pcr.PARENT_CHILD_ENABLED
pcr.PARENT_CHILD_ENABLED = True
try:
    c = FakeClient(_rows)
    res = pcr.retrieve_with_parents("q", FakeChildRetriever(_kids), c, "coll")
finally:
    pcr.PARENT_CHILD_ENABLED = _orig
check("R4 client.query 只调 1 次（批量，非 N+1）", len(c.calls) == 1, len(c.calls))
check("R4 回查 expr 含两个 parent_id",
      '"p1"' in c.calls[0] and '"p2"' in c.calls[0], c.calls[0][:80])

print()
print("=" * 100)
print("R2/R3 排序")
print("=" * 100)
kids = [
    kid("pA", 0.95),                                    # A: 1 命中，相似度最高
    kid("pB", 0.60), kid("pB", 0.62), kid("pB", 0.64),  # B: 3 命中
    kid("pC", 0.80), kid("pC", 0.82),                   # C: 2 命中
]
rows = [parent_row("pA"), parent_row("pB"), parent_row("pC")]
res, _ = run(kids, rows)
order = [d.metadata["parent_id"] for d in res]
check("R2 命中多的排前面（B=3 > C=2 > A=1）", order == ["pB", "pC", "pA"], order)

# 命中数相同 -> 看 max_sim
kids2 = [kid("pX", 0.5), kid("pY", 0.9)]
rows2 = [parent_row("pX"), parent_row("pY")]
res2, _ = run(kids2, rows2)
check("R3 命中数相同 -> 相似度高的排前",
      [d.metadata["parent_id"] for d in res2] == ["pY", "pX"],
      [d.metadata["parent_id"] for d in res2])

print()
print("=" * 100)
print("R5/R6/R7 截断、无命中、matched_children")
print("=" * 100)
many_kids = [kid("p%d" % i, 0.5 + i * 0.01) for i in range(10)]
many_rows = [parent_row("p%d" % i) for i in range(10)]
res5, _ = run(many_kids, many_rows, top_n=3)
check("R5 top_n_parents=3 截断", len(res5) == 3, len(res5))

res6, _ = run([], [], top_n=3)
check("R6 无命中 -> []", res6 == [], repr(res6))

res7, _ = run([kid("p1", 0.9, text="这是命中的子块原文")], [parent_row("p1")])
check("R7 父块带 matched_children",
      res7 and "这是命中的子块原文" in res7[0].metadata.get("matched_children", []),
      res7[0].metadata.get("matched_children") if res7 else None)
check("R7 父块带 hit_count / max_child_sim",
      res7 and res7[0].metadata.get("hit_count") == 1
      and abs(res7[0].metadata.get("max_child_sim", 0) - 0.9) < 1e-6)

print()
print("=" * 100)
print("R8 开关关闭 -> None（与「无命中 []」严格区分）")
print("=" * 100)
_orig = pcr.PARENT_CHILD_ENABLED
pcr.PARENT_CHILD_ENABLED = False
try:
    r = pcr.retrieve_with_parents("q", FakeChildRetriever([]), FakeClient([]), "coll")
finally:
    pcr.PARENT_CHILD_ENABLED = _orig
check("R8 关闭时返回 None 而非 []", r is None, repr(r))

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
