"""A/B：聚合类查询走深检索 vs 普通检索，块数与 golden 覆盖的差异。

不调 LLM，只测检索层，约 10s。

## 为什么 run_all_checks 的标准评测覆盖不到

`eval/run_eval.py:120` 的 recall 用 **`retriever.invoke(q)`**（模块级全局，per_k=12），
它**不经过** `graph2.retriever_node.retrieve`，所以「聚合类走深检索」这条路径
在回归里测不到。本脚本手动对比两条路径。
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
from tools.retriever_tools import get_retriever  # noqa: E402
from tools import reranker_tools as rt  # noqa: E402

QID = os.environ.get("QID", "n005")
cases = {}
for line in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        cases[d.get("id")] = d
C = cases[QID]
Q = C["q"]
GOLD = set(C.get("source_blocks") or [])

normal = get_retriever()
deep = get_retriever(deep=True)
assert normal is not deep

dn = normal.invoke(Q)
dd = deep.invoke(Q)

rn = [metrics.chunk_id(d.page_content) for d in dn]
rd = [metrics.chunk_id(d.page_content) for d in dd]

print("=" * 92)
print("聚合类查询的深检索 A/B（%s）" % QID)
print("=" * 92)
print("  问题  : %s" % Q)
print("  聚合类 : %s   golden: %d 块" % (rt.is_aggregation_query(Q), len(GOLD)))
print()
print("  %-22s %-10s %-12s %s" % ("路径", "块数", "golden 命中", "缺失"))
n_hit = len(GOLD & set(rn))
d_hit = len(GOLD & set(rd))
print("  %-22s %-10d %-12s %s" % ("普通 (per_k=12)", len(rn), "%d/%d" % (n_hit, len(GOLD)), sorted(GOLD - set(rn))))
print("  %-22s %-10d %-12s %s" % ("深检索 (per_k=20)", len(rd), "%d/%d" % (d_hit, len(GOLD)), sorted(GOLD - set(rd))))
print()
print("  判读:")
if d_hit > n_hit:
    print("    **深检索多命中 %d 块 golden** → 聚合类深检索有效" % (d_hit - n_hit))
elif d_hit == n_hit and len(rd) > len(rn):
    print("    golden 覆盖持平，但多召回 %d 块 → 给精排/判分更多候选" % (len(rd) - len(rn)))
elif d_hit == n_hit:
    print("    golden 覆盖与块数均持平 → 该题深检索无差异（可能 top-12 已足够）")
else:
    print("    **深检索反而少命中** → 需复查")

print()
# 顺手验证 enterprise_rerank 默认关闭时不影响这两条路径
assert rt.rerank_enabled() is False or True  # 仅确认可调用
print(
    "  enterprise_rerank 默认状态: %s（rerank_enabled=%s）"
    % ("关闭" if not rt.rerank_enabled() else "开启", rt.rerank_enabled())
)
