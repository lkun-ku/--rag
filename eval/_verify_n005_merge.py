"""端到端验证：接入 `merge_table_slices` 后，n005 是否从「3/3 拒答」变答对。

对照组来自 `eval/_exp_n005_table.py`（接入前）：

    A 生产上下文（9 块 / 16,182 字符）-> **3/3 稳定拒答**
    B 完整表   （1 块 /  8,614 字符）-> 2/3 答对

本脚本跑**真实链路**（`run_with_trace`），跑 3 次，记录：
块数变化（合并是否生效）、是否拒答、是否命中 61.35%。
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

from utils.citation import is_refusal  # noqa: E402
from graph2.generate_node2 import format_docs, merge_table_slices  # noqa: E402
from graph2.graph_2 import run_with_trace  # noqa: E402

QID = "n005"
RUNS = 3
cases = {}
for line in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        cases[d.get("id")] = d
C = cases[QID]
Q, GOLD = C["q"], C.get("answer")

print("=" * 100)
print("n005 接入「同表切片合并」后的端到端验证（%d 次）" % RUNS)
print("=" * 100)
print("  问题  : %s" % Q)
print("  golden: %s" % GOLD)
print()
print("  接入前对照: 9 块 / 16,182 字符 -> **3/3 拒答**")
print()

results = []
for i in range(RUNS):
    out = run_with_trace(Q, save=False)
    docs = out.get("documents") or []
    if not isinstance(docs, list):
        docs = [docs]
    gen = (out.get("generation") or "").strip()
    refused = is_refusal(gen)
    hit = ("61.35" in gen) and ("turbopuffer" in gen.lower())
    ctx = format_docs(docs)
    results.append((refused, hit))
    print("  第%d次: 块数=%d 字符=%d 拒答=%-3s 命中=%-3s"
          % (i + 1, len(docs), len(ctx), "是" if refused else "否",
             "是" if hit else "否"))
    print("         %s" % gen[:150].replace("\n", " "))

print()
print("=" * 100)
print("判读")
print("=" * 100)
n_ref = sum(1 for r, _ in results if r)
n_hit = sum(1 for _, h in results if h)
print("  拒答 %d/%d · 命中 %d/%d" % (n_ref, RUNS, n_hit, RUNS))
if n_ref == 0 and n_hit == RUNS:
    print("  **完全修复**：从 3/3 拒答 -> 3/3 稳定答对 ✅")
elif n_hit >= 2:
    print("  **显著改善**：3/3 拒答 -> %d/3 答对（仍有抖动，与完整表对照组一致）" % n_hit)
elif n_hit >= 1:
    print("  **部分改善**：但仍有较多拒答，需进一步排查")
else:
    print("  **未改善**：合并未生效或另有原因")
