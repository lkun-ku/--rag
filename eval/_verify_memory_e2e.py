"""端到端验证：graph2 的多轮记忆是否真的跨轮累积 + 会话隔离。

## 验证什么（不看答案质量，只看机制）

| 断言 | 含义 |
|---|---|
| E1 同 thread_id 第 1 轮 -> `messages` 长度 2（一问一答） | 首轮就写入历史 |
| E2 同 thread_id 第 2 轮 -> `messages` 长度 4 | **跨轮累积**（这是「记忆」的定义） |
| E3 另一个 thread_id -> `messages` 长度 2 | **会话隔离**（不串台） |
| E4 不传 thread_id -> `messages` 为空 | **单轮行为不变**（零破坏） |

## 为什么用「messages 长度」而不是「答案对不对」

答案质量受 LLM 波动影响（`temperature=0` 也可能变），用它会引入不确定性；
而 `messages` 长度是**确定性事实**，能直接把「机制是否生效」与「模型发挥」切开。

答案质量的对照（「它的 QPS 是多少」能否解析指代）另行人工观察，结论写进 ROADMAP。
"""
from __future__ import annotations

import os
import pathlib
import sys
import uuid

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from graph2.graph_2 import run_with_trace  # noqa: E402

Q1 = "TurboPuffer 在带过滤向量检索、过滤比例 98% 的场景下，召回率是多少？"
Q2 = "它在这个场景下的 QPS 是多少？"
Q3 = "Milvus 支持哪些索引类型？"

RUN = uuid.uuid4().hex[:8]
T1 = "__memtest__%s_A" % RUN
T2 = "__memtest__%s_B" % RUN

results = []


def show(tag, out):
    n = len(out.get("messages") or [])
    print("  %-6s messages=%d  答案: %s" % (tag, n, (out.get("generation") or "")[:96].replace("\n", " ")))
    return n


print("=" * 100)
print("graph2 多轮记忆端到端验证（run=%s）" % RUN)
print("=" * 100)
print()

print("【E1/E2】同一 thread_id 连续两轮")
print("  thread_id = %s" % T1)
o1 = run_with_trace(Q1, save=False, thread_id=T1)
n1 = show("第1轮", o1)

# [诊断] 第二轮开始前，直接用 checkpointer 读一下，确认历史是否真的落盘
from graph2.graph_2 import get_memory_graph  # noqa: E402

_g = get_memory_graph()
_cfg = {"configurable": {"thread_id": T1}}
try:
    _st = _g.get_state(_cfg)
    _vals = getattr(_st, "values", {}) or {}
    _msgs = _vals.get("messages") or []
    print("  [诊断] 第2轮前 get_state 读到 messages=%d 条" % len(_msgs))
    for _m in _msgs[:4]:
        print("         %s: %s" % (getattr(_m, "type", "?"),
                                   str(getattr(_m, "content", ""))[:50]))
except Exception as _e:  # noqa: BLE001
    print("  [诊断] get_state 失败: %s: %s" % (type(_e).__name__, str(_e)[:80]))

o2 = run_with_trace(Q2, save=False, thread_id=T1)
n2 = show("第2轮", o2)
results.append(("E1 首轮 messages=2", n1 == 2, n1))
results.append(("E2 次轮 messages=4（跨轮累积）", n2 == 4, n2))

print()
print("【E3】另一个 thread_id（会话隔离）")
print("  thread_id = %s" % T2)
o3 = run_with_trace(Q3, save=False, thread_id=T2)
n3 = show("独立会话", o3)
results.append(("E3 独立会话 messages=2", n3 == 2, n3))

print()
print("【E4】不传 thread_id（单轮，行为应与改造前一致）")
o4 = run_with_trace(Q3, save=False)
n4 = show("单轮", o4)
results.append(("E4 单轮 messages 为空", n4 == 0, n4))

print()
print("=" * 100)
print("判读")
print("=" * 100)
ok = sum(1 for _, c, _ in results if c)
for name, cond, val in results:
    print("  %s %-40s 实际 %r" % ("[OK ]" if cond else "[FAIL]", name, val))
print()
print("  通过 %d / %d" % (ok, len(results)))
sys.exit(0 if ok == len(results) else 1)
