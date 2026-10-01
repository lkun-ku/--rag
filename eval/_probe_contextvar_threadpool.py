"""探针：LangChain 的 config 是否靠 **contextvar** 传递，而**线程池不继承它**？

## ⚠️ 这是**第二版** —— 首版的 B 组对照不成立（二次筛查发现）

首版的 B 组是这样写的：

    uB = UsageMetadataCallbackHandler()
    def _work_no_config(_i):
        chain.invoke({"dummy": 1})          # 想测「线程池内不传 config」
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(_work_no_config, range(2)))
    print(tokens_of(uB))                    # -> 0

**问题**：`uB` **从来没有被注册到任何调用**（没有一处 `config={"callbacks": [uB]}`），
所以 `token=0` 是**必然结果**，与「contextvar 是否跨线程」**毫无关系**。
**-> 首版结论（「contextvar 在线程池内失效」）没有证据。**

## 本版的三条对照（这次真的成立）

关键在于：**必须先让「当前上下文」里带上 callbacks**，再观察池内调用能否继承它。
用一层**外层 Runnable**（带 config）包住线程池，就能构造出这个状态：

    A. 无线程池 + 内层不传 config  -> 期望 **> 0**（contextvar 继承生效）
    B. 线程池   + 内层不传 config  -> 期望 **0**（若假设成立：池内拿不到外层 contextvar）
    C. 线程池   + 内层显式传 config -> 期望 **> 0**（显式传递有效 —— 我们的修法）

**A 与 B 的唯一差别是「在不在线程池里」**，C 用来证明「显式传能救回来」。

## 如果 B 也 > 0 会怎样

那说明「contextvar 会随线程复制」（某些 LangChain/anyio 版本会主动复制上下文），
**4.29 的「线程池内失效」就是错的** —— 那么当初判分首版统计不到 token 的原因
**另有其他**，需要重新定位（不能沿用旧结论）。
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


# 内层链：**自己不带 config** —— 它能否统计到，完全取决于「当前上下文里有没有 callbacks」
inner_chain = RunnableLambda(lambda _x: llm.invoke(PROMPT))

print("=" * 100)
print("探针（第二版）：contextvar 在线程池内是否失效")
print("=" * 100)
print()

# ---- A：无线程池，内层不传 config（外层 Runnable 带 config）------------------
uA = UsageMetadataCallbackHandler()
outer_A = RunnableLambda(lambda _x: inner_chain.invoke({"d": 1}))
outer_A.invoke({"d": 1}, config={"callbacks": [uA]})
a = tokens_of(uA)
print(
    "  [A] 无线程池 + 内层不传 config -> token=%d  %s"
    % (a, "✅（继承生效）" if a > 0 else "❌（连主线程都继承不了，说明另有问题）")
)

# ---- B：线程池内，内层不传 config（**同一个外层上下文**）---------------------
uB = UsageMetadataCallbackHandler()


def _work_no_config(_i):
    inner_chain.invoke({"d": 1})  # 故意不传 config
    return 1


def _run_pool(_x):
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(_work_no_config, range(2)))
    return 1


outer_B = RunnableLambda(_run_pool)
outer_B.invoke({"d": 1}, config={"callbacks": [uB]})
b = tokens_of(uB)
print(
    "  [B] 线程池   + 内层不传 config -> token=%d  %s"
    % (b, "✅（池内拿不到，假设成立）" if b == 0 else "⚠️（池内也能拿到 —— 假设不成立）")
)

# ---- C：线程池内，显式传 config ---------------------------------------------
uC = UsageMetadataCallbackHandler()
CFG = {"callbacks": [uC]}


def _work_with_config(_i):
    inner_chain.invoke({"d": 1}, config=CFG)  # 显式传
    return 1


def _run_pool_c(_x):
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(_work_with_config, range(2)))
    return 1


RunnableLambda(_run_pool_c).invoke({"d": 1})
c = tokens_of(uC)
print("  [C] 线程池   + 显式传 config -> token=%d  %s" % (c, "✅（显式传递有效）" if c > 0 else "❌（显式传也不行？）"))

print()
print("=" * 100)
print("判读")
print("=" * 100)
print("  A=%d  B=%d  C=%d" % (a, b, c))
print()
if a > 0 and b == 0 and c > 0:
    print("  -> **假设成立**：contextvar 在线程池内失效，必须显式传 config。")
    print("     4.29 的结论**有了证据**（首版没有，本版补上）。")
elif a > 0 and b > 0:
    print("  -> **假设不成立**：B 也拿到了 token，说明该版本会把上下文带进线程。")
    print("     那么 4.29 的结论（以及它指导的改动）需要**重新论证** ——")
    print("     判分首版统计不到 token 的原因另有其他（例如当时压根没透传 config）。")
else:
    print("  -> 结果不符合任一预期（A 都没拿到），**先别下结论**，需查环境。")
