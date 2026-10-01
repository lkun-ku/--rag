"""可行性探针：LangGraph 的节点函数能否拿到 `graph.stream(config=...)` 传下来的 config？

## 为什么先探这个

4.20.7（本次提交）定位到：`grade_documents_node._grade_one` 里手动
`retrieval_grader_chain.invoke(...)` **没有传 config**，导致判分的 token
不被 `run_with_trace` 的 callback 统计。

修法取决于「节点函数能不能拿到上层 config」——
* 若能（节点签名 `fn(state, config)`）-> 直接把 callbacks 透传给 `.invoke()`
* 若不能 -> 得换别的路子（如 `get_config()` 上下文变量）

**这条决定整个修法，必须先探明，不能猜。**
"""

from __future__ import annotations

from typing import TypedDict



print("=" * 96)
print("探针：LangGraph 节点能否收到 config")
print("=" * 96)

try:
    import langgraph

    print("  langgraph 版本: %s" % getattr(langgraph, "__version__", "未知"))
except Exception as e:  # noqa: BLE001
    print("  [!] import langgraph 失败: %s" % e)

from langgraph.graph import StateGraph, END  # noqa: E402


class S(TypedDict, total=False):
    x: int
    seen: str


SEEN = {}


def node_two_args(state, config=None):
    """签名 (state, config) —— LangGraph 是否会把 config 传进来？"""
    SEEN["two_args"] = config
    return {"x": state.get("x", 0) + 1}


def node_one_arg(state):
    """签名 (state) —— 对照组。"""
    return {"x": state.get("x", 0) + 1}


g = StateGraph(S)
g.add_node("two", node_two_args)
g.add_node("one", node_one_arg)
g.set_entry_point("two")
g.add_edge("two", "one")
g.add_edge("one", END)
app = g.compile()

# 注意：callbacks 必须是**真正的 BaseCallbackHandler 实例**，不能塞字符串
# （首版探针传了 `["__FAKE_CALLBACK__"]`，LangChain 在 `on_chain_start` 里
#  访问 `handler.ignore_chain` 直接抛 AttributeError —— 这是探针的错，不是被测代码的）
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402


class _ProbeCB(BaseCallbackHandler):
    """最小回调：只用来给「config 是否透传」做标记。"""

    def __init__(self, name: str = "probe"):
        self.name = name


FAKE = [_ProbeCB()]
app.invoke({"x": 1}, config={"callbacks": FAKE})

cfg = SEEN.get("two_args")
print()
print("  [两参数节点] 收到的 config: %r" % (cfg,))
if cfg is None:
    print("  -> **没收到**（LangGraph 不传 config 给两参数节点）")
else:
    cbs = cfg.get("callbacks") if isinstance(cfg, dict) else getattr(cfg, "get", None)
    got = cfg.get("callbacks") if isinstance(cfg, dict) else None
    print("  -> 收到，callbacks=%r" % (got,))
    print("  -> **可取到 callbacks**: %s" % (got == FAKE))

# 再探一条备选路：Runnable 内部用 get_config()
print()
print("  --- 备选路：Runnable 内部的 get_config() ---")
try:
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.config import get_config

    def _inner(payload):
        c = get_config()
        cm = c.get("callbacks")
        return {"seen": "yes" if cm else "no", "n": len(cm) if cm else 0}

    r = RunnableLambda(_inner)
    out = r.invoke({"a": 1}, config={"callbacks": FAKE})
    print("  Runnable 内 get_config() -> %r" % (out,))
    if out.get("seen") == "yes":
        print("  -> **这条可行**：在 Runnable 内部可用 get_config() 取到 callbacks")
    else:
        print("  -> 这条取不到 callbacks")
except Exception as e:  # noqa: BLE001
    print("  [!] 探针失败: %s: %s" % (type(e).__name__, e))

# 再探第三条：节点内直接调 chain.invoke 时，get_config() 是否还有效
print()
print("  --- 第三条：在**节点函数内**（非 Runnable）调 get_config() ---")
SEEN2 = {}


def node_get_config(state, config=None):
    try:
        from langchain_core.runnables.config import get_config

        c = get_config()
        SEEN2["callbacks"] = c.get("callbacks")
    except Exception as e:  # noqa: BLE001
        SEEN2["error"] = "%s: %s" % (type(e).__name__, e)
    return {"x": state.get("x", 0)}


g2 = StateGraph(S)
g2.add_node("n", node_get_config)
g2.set_entry_point("n")
g2.add_edge("n", END)
g2.compile().invoke({"x": 1}, config={"callbacks": FAKE})
print("  节点内 get_config() -> callbacks=%r  error=%r" % (SEEN2.get("callbacks"), SEEN2.get("error")))
