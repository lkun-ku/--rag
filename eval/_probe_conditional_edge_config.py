"""探针：LangGraph 的**条件边函数**能否拿到 config？

## 为什么补这个探针

前两个探针只验证了「**节点**」（`add_node`）能拿到 config —— 但条件边是
**另一套调度路径**（`add_conditional_edges`）。而 `grade_generation` 恰恰是条件边：

    workflow.add_conditional_edges("generate",
        _traced_router("grade_generation", grade_generation_v_documents_and_question), {...})

它的内部有两次 LLM 判定。如果条件边拿不到 config，那两次调用的 token
**依然不可见** —— 实测（4.25 改完之后）total token 与改前**完全相同（31,465）**，
强烈提示「config 没传进去」。

这是同一个盲区的**第三次**（前两次：`tracer.traced` 包装器、节点签名），
教训是：**每加一类新的被包装对象，都要单独验证它的 config 通路。**

## 探针内容

1. 两参数的**条件边**函数 -> 能否拿到 config？
2. 三参数（state, config, runtime）？—— 排除「LangGraph 版本用了别的签名」
"""
from __future__ import annotations

import pathlib
import sys
from typing import TypedDict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402


class _CB(BaseCallbackHandler):
    pass


class S(TypedDict, total=False):
    x: int
    seen: bool


SEEN = {}


def node_start(state):
    return {"x": 1}


def cond_two_args(state, config=None):
    """两参数条件边函数。"""
    cbs = (config or {}).get("callbacks") if isinstance(config, dict) else None
    SEEN["cond_two_args"] = ("有 config" if config else "无 config",
                             "有 callbacks" if cbs else "无 callbacks")
    return "next_node"


def next_node(state):
    return {"x": state.get("x", 0) + 1}


print("=" * 96)
print("探针：条件边函数能否拿到 config")
print("=" * 96)

g = StateGraph(S)
g.add_node("start", node_start)
g.add_node("next_node", next_node)
g.set_entry_point("start")
g.add_conditional_edges("start", cond_two_args, {"next_node": "next_node"})
g.add_edge("next_node", END)
app = g.compile()

CB = [_CB()]
app.invoke({"x": 0}, config={"callbacks": CB})

print()
print("  [两参数条件边] %r" % (SEEN.get("cond_two_args"),))
res = SEEN.get("cond_two_args")
if res and res[1] == "有 callbacks":
    print("  -> **条件边能拿到 callbacks**（config 通路是通的）")
else:
    print("  -> **条件边拿不到 config**（或拿不到 callbacks）")
    print("     => 需要在 `_traced_router` 里换一条路径，或借助 LangGraph 的 runtime 参数")
