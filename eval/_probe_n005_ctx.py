"""探针：n005 最终进入生成的上下文是什么（修正版）。

## 上一版的两个错误

1. `patch generate_node2.generate` **无效** —— `graph_2.py` 里是
   `from .generate_node2 import generate` 之后再包 `traced(...)`，
   **已绑定原引用**，改模块属性不影响图里用的函数。
2. `metrics.is_refusal` 不存在 —— 实际在 `utils.citation`。

## 本版做法

不 patch 任何东西，直接看 `run_with_trace` 返回的**最终状态**：
`state["documents"]` 就是 generate 收到的那批块（generate 不改 documents）。
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
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.citation import is_refusal  # noqa: E402
from graph2.generate_node2 import format_docs  # noqa: E402
from graph2.graph_2 import run_with_trace  # noqa: E402

QID = "n005"
cases = {}
for line in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        cases[d.get("id")] = d
C = cases[QID]
Q, GOLD = C["q"], C.get("answer")
GOLD_BLOCKS = set(C.get("source_blocks") or [])

print("=" * 100)
print("n005 上下文探针（修正版）")
print("=" * 100)
print("  问题  : %s" % Q)
print("  golden: %s" % GOLD)
print("  golden 块: %s" % sorted(GOLD_BLOCKS))
print()

out = run_with_trace(Q)
docs = out.get("documents") or []
if not isinstance(docs, list):
    docs = [docs]
gen = (out.get("generation") or "").strip()

ctx = format_docs(docs) if docs else ""
ids = [metrics.chunk_id(d.page_content) for d in docs]

print("=" * 100)
print("generate 收到的上下文")
print("=" * 100)
print("  块数            : %d" % len(docs))
print("  上下文字符数    : %d" % len(ctx))
print("  含 61.35        : %s" % ("**是**" if "61.35" in ctx else "**否**"))
print("  含 TurboPuffer  : %s" % ("是" if "TurboPuffer" in ctx else "否"))
print()
print("  块的 chunk_id（* = golden 块）:")
for cid in ids:
    print("     %s %s" % (cid, "<-- golden" if cid in GOLD_BLOCKS else ""))
missing = GOLD_BLOCKS - set(ids)
print()
print("  **golden 块缺失: %s**" % (sorted(missing) if missing else "无（2/2 都在）"))

print()
print("=" * 100)
print("生成结果")
print("=" * 100)
print("  拒答: %s" % ("**是**" if is_refusal(gen) else "否"))
print("  内容: %s" % gen[:500].replace("\n", " "))
print()
print(
    "  命中 golden（61.35 + TurboPuffer）: %s"
    % ("**是**" if ("61.35" in gen and "turbopuffer" in gen.lower()) else "否")
)
