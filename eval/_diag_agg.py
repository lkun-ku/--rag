"""诊断：聚合类问题（「召回率最低」）是抖动还是稳定失败。

## 要区分三种可能

    ① 检索没覆盖到 61.35 那块      -> docs 里不含 "61.35"
    ② 含 61.35 但表格不完整（没合并）-> 含 61.35，但最大块行数远小于 91
    ③ 表格完整但模型没算对          -> 含 61.35 且最大行数 ~91，仍答错

同时打印 citations 的**真实键名**（build_citation_meta 返回的是 `n` 不是 `idx`）。
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from graph2.graph_2 import run_with_trace  # noqa: E402
from tools import reranker_tools as rt  # noqa: E402

Q = "参与评测的产品里，召回率最低的是哪个？召回率是多少？"
print("=" * 96)
print("聚合类问题 3 次诊断")
print("=" * 96)
print("  问题: %s" % Q)
print("  聚合类识别: %s" % rt.is_aggregation_query(Q))
print()

for i in range(1, 4):
    out = run_with_trace(Q, save=False)
    gen = (out.get("generation") or "").strip()
    docs = out.get("documents") or []
    has_6135 = any("61.35" in (d.page_content or "") for d in docs)
    max_lines = max((len((d.page_content or "").splitlines()) for d in docs), default=0)
    n_with_6135 = sum(1 for d in docs if "61.35" in (d.page_content or ""))
    hit = "61.35" in gen
    verdict = ("① 检索未覆盖" if not has_6135
               else ("③ 表格完整但答错" if max_lines >= 80 else "② 表格未合并完整"))
    print("  第%d次: 块数=%2d 含61.35块=%d 最大行数=%3d 答出61.35=%-3s | %s"
          % (i, len(docs), n_with_6135, max_lines, "是" if hit else "否", verdict))
    print("         答案: %s" % gen[:110].replace("\n", " "))

print()
print("=" * 96)
print("citations 键名核对")
print("=" * 96)
out = run_with_trace("TurboPuffer 在 98% 过滤下的 QPS 是多少？", save=False)
cs = out.get("citations") or []
print("  引用条数: %d" % len(cs))
if cs:
    print("  实际键: %s" % sorted(cs[0].keys()))
    print("  内容: n=%r filename=%r chars=%r" %
          (cs[0].get("n"), str(cs[0].get("filename"))[:38], cs[0].get("chars")))
    print("  **注意：键是 `n` 不是 `idx`** —— 调用方若用 idx 会拿到 None")
