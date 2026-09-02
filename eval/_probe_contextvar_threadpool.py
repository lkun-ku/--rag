"""探针：LangChain 的 config 靠 **contextvar** 传递，而**线程池不继承 contextvar**。

## 要验证的假设（它解释了一连串现象）

| 调用点 | 是否显式传 config | 是否用线程池 | token 是否统计到 |
|---|---|---|---|
| `route_question` | ❌ 没传 | ❌ 无线程池 | ✅ **统计到了** |
| `generate` | ❌ 没传 | ❌ 无线程池 | ✅ **统计到了** |
| `grade_documents`（首版） | ❌ 没传 | ✅ **有线程池** | ❌ **统计不到** |
| `grade_generation`（4.24 前） | ❌ 没传 | ❌ 串行 | ✅ 统计到了（所以 4.25 的 token 没变**不是矛盾**） |

**假设**：LangChain 的 `Runnable.invoke(..., config=None)` 会从 **contextvar**
取「当前 config」；而**线程池里的工作线程没有那个 contextvar**（Python 的
`contextvars` 不会自动跨线程传播），所以线程池内的调用拿不到 callbacks。

**若成立**，则：
1. 4.25.4 那个「token 该 +897 却没变」的疑问**有答案了** —— 4.24 时它们**本来就被统计**
   （串行、无线程池），并非漏统计；
2. `generate_node2.py` / `graph_2.py` 里那两处「没传 config 也能工作」是**隐式依赖** ——
   一旦将来改成并发就会**静默失效**，应当改为**显式传递**。

## 探针设计（三条对照，缺一不可）

    A. 无线程池 + 不传 config  -> 期望 **能**统计（contextvar 继承）
    B. 线程池   + 不传 config  -> 期望 **不能**统计（contextvar 不跨线程）
    C. 线程池   + 显式传 config -> 期望 **能**统计（这就是我们的修法）
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from llm_models.all_llm import llm  # noqa: E402

PROMPT = "只回答一个词：好"


def tokens_of(u) -> int:
    m = dict(getattr(u, "usage_metadata", {}) or {})
    return sum(int(v.get("total_tokens", 0) or 0) for v in m.values())


chain = RunnableLambda(lambda _x: llm.invoke(PROMPT))

print("=" * 100)
print("探针：contextvar 在线程池内是否失效")
print("=" * 100)

# A. 无线程池，不传 config
uA = UsageMetadataCallbackHandler()
try:
    chain.invoke({"dummy": 1}, config={"callbacks": [uA]})
except Exception as e:  # noqa: BLE001
    print("  [A] 调用失败: %s: %s" % (type(e).__name__, e))
print("  [A] 无线程池 + 不传 config -> token=%d" % tokens_of(uA))

# B. 线程池，不传 config（**关键对照**）
uB = UsageMetadataCallbackHandler()


def _work_no_config(_i):
    # 注意：这里**不传** config，依赖 contextvar 继承
    chain.invoke({"dummy": 1})
    return 1


with cf.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(_work_no_config, range(2)))
print("  [B] 线程池   + 不传 config -> token=%d" % tokens_of(uB))

# C. 线程池，显式传 config（我们的修法）
uC = UsageMetadataCallbackHandler()
CFG = {"callbacks": [uC]}


def _work_with_config(_i):
    chain.invoke({"dummy": 1}, config=CFG)
    return 1


with cf.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(_work_with_config, range(2)))
print("  [C] 线程池   + 显式传 config -> token=%d" % tokens_of(uC))

print()
print("=" * 100)
print("判读")
print("=" * 100)
a, b, c = tokens_of(uA), tokens_of(uB), tokens_of(uC)
print("  A=%d  B=%d  C=%d" % (a, b, c))
print()
print("  预期: A>0（contextvar 继承）、B==0（线程池丢失）、C>0（显式传递有效）")
if a > 0 and b == 0 and c > 0:
    print("  -> **假设成立**：contextvar 在线程池内失效，必须显式传 config。")
elif a > 0 and b > 0:
    print("  -> B 也有 token：说明该版本 LangChain 会把 contextvar 复制到线程 ——")
    print("     那么判分首版统计不到的原因**另有其他**，需重新定位。")
else:
    print("  -> 结果与预期不符，需要进一步排查（不要据此下结论）。")
