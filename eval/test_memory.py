"""graph2 多轮记忆 —— 契约测试（ROADMAP 4.41）。

## 锁定的五条契约

| # | 契约 | 为什么重要 |
|---|---|---|
| C1 | `_format_history` 只取**最近 N 轮**、只认 human/ai、空历史返回空串 | 历史无限增长会挤爆上下文；空串是「是否走历史模板」的判据 |
| C2 | `create_graph()` **默认无 checkpointer** | 现有 22 个提交的调用方（评测/回归/`run_with_trace` 默认路径）都不传 `thread_id`，挂了就会报错 |
| C3 | `create_graph(checkpointer=...)` 能挂上 | 记忆模式的入口 |
| C4 | `get_memory_graph()` 是**惰性单例** | 重复构造会叠加 SQLite 连接；多轮必须共享同一个 saver |
| C5 | `run_with_trace` 接受 `thread_id`，默认 None | 向后兼容的签名 |

**真实两轮记忆**（会调 LLM，约 30s）不在本文件里跑 —— 见
`eval/_verify_memory_e2e.py`，由人工触发并记录结论。
"""
from __future__ import annotations

import inspect
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from graph2.generate_node2 import MAX_HISTORY_TURNS, _format_history  # noqa: E402
from graph2.graph_2 import create_graph, get_memory_graph, run_with_trace  # noqa: E402

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


print("=" * 100)
print("C1 _format_history 纯函数")
print("=" * 100)

check("空历史 -> 空串", _format_history([]) == "")
check("None -> 空串", _format_history(None) == "")

msgs = [HumanMessage(content="第一个问题"), AIMessage(content="第一个回答")]
h = _format_history(msgs)
check("1 轮 -> 含双方", "用户：第一个问题" in h and "助手：第一个回答" in h, repr(h[:60]))

# 只保留最近 MAX_HISTORY_TURNS 轮
many = []
for i in range(MAX_HISTORY_TURNS + 3):
    many.append(HumanMessage(content="Q%d" % i))
    many.append(AIMessage(content="A%d" % i))
h = _format_history(many)
check("超限只取最近 %d 轮" % MAX_HISTORY_TURNS,
      "Q0" not in h and ("Q%d" % (len(many) // 2 - 1)) in h, repr(h[:80]))

# system / 其它角色被忽略
h = _format_history([SystemMessage(content="系统提示"), HumanMessage(content="问"), AIMessage(content="答")])
check("忽略 system 角色", "系统提示" not in h and "问" in h, repr(h[:60]))

# 单条超长被截断
long_msg = [HumanMessage(content="长" * 1000), AIMessage(content="答")]
h = _format_history(long_msg)
check("单条截断到 300 字", len(h) < 400, "实际 %d" % len(h))

print()
print("=" * 100)
print("C2/C3 create_graph 的 checkpointer")
print("=" * 100)
g_plain = create_graph()
check("默认无 checkpointer（零破坏）", g_plain.checkpointer is None,
      "实际 %r" % type(g_plain.checkpointer).__name__)

saver = MemorySaver()
g_mem = create_graph(checkpointer=saver)
check("传 checkpointer 能挂上", g_mem.checkpointer is saver)

print()
print("=" * 100)
print("C4 get_memory_graph 惰性单例")
print("=" * 100)
m1 = get_memory_graph()
m2 = get_memory_graph()
check("两次调用返回同一对象", m1 is m2)
check("挂着 checkpointer", m1.checkpointer is not None)

print()
print("=" * 100)
print("C5 run_with_trace 签名")
print("=" * 100)
params = inspect.signature(run_with_trace).parameters
check("有 thread_id 且默认 None",
      "thread_id" in params and params["thread_id"].default is None,
      str(list(params)))

print()
print("=" * 100)
# [注意] 必须输出 `X/Y 通过` 格式 —— `run_all_checks.py` 靠正则解析通过率。
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
