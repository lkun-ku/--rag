"""探针：区分「连不上」与「响应慢于 20s」——用户观察到端点侧**有 token 消耗**。

## 为什么值得再测

`eval/check_llm.py` 的结果是 `APITimeoutError`（耗时 41.4s ≈ 2 × 20s 超时）。
但用户观察到 **DeepSeek 侧有实际消耗** —— 那说明**请求已经送达并被执行了**，
只是**响应慢于客户端 20s 超时**。

**两者修法完全不同**：
  * 「连不上」-> 网络/代理问题
  * 「响应慢」-> **把客户端超时放宽**即可（服务端其实在正常工作）

## 本探针做三件事

1. **裸 SDK 调用**（绕开 LangChain 封装），超时给足 **120s**，看单次真实耗时；
2. 若成功，打印**响应内容与耗时**；
3. 顺带测**项目 `llm` 对象**（LangChain `ChatOpenAI`）在不同 timeout 下的行为，
   为「要不要给 `ChatOpenAI` 显式设 timeout」提供依据
   （代码审查提到过：`all_llm.py` 的 `ChatOpenAI` 未设 timeout）。
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from utils.env_utils import (
    DEEPSEEK_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,  # noqa: E402
    OPENAI_API_KEY,
)

print("=" * 100)
print("探针：区分「连不上」与「响应慢」")
print("=" * 100)
print("  base_url : %s" % LLM_BASE_URL)
print("  model    : %s" % LLM_MODEL)
print("  api_key  : %s" % ("已配置" if (DEEPSEEK_API_KEY or OPENAI_API_KEY or OPENAI_API_KEY) else "**缺失**"))
print()

# ---- 1. 裸 SDK，超时 120s ---------------------------------------------------
print("[1] 裸 openai SDK，timeout=120s")
try:
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY or OPENAI_API_KEY or OPENAI_API_KEY, base_url=LLM_BASE_URL, timeout=120.0)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "只回答两个字：正常"}],
        max_tokens=16,
    )
    el = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    print("    成功 ✅ 耗时 %.1fs | 回复=%r | usage=%s" % (el, text[:40], usage))
    print("    -> **端点可用**，此前 20s 超时过短（真实耗时 %.1fs）" % el)
except Exception as e:  # noqa: BLE001
    el = time.time() - t0 if 't0' in dir() else -1
    print("    失败 ❌ %s: %s（耗时 %.1fs）" % (type(e).__name__, str(e)[:120], el))

# ---- 2. 项目 llm 对象（LangChain），不设 timeout（默认）---------------------
print()
print("[2] 项目 llm 对象（ChatOpenAI，未显式设 timeout）")
try:
    from llm_models.all_llm import llm

    t0 = time.time()
    r = llm.invoke("只回答两个字：正常")
    el = time.time() - t0
    print("    成功 ✅ 耗时 %.1fs | 回复=%r" % (el, str(getattr(r, "content", r))[:40]))
except Exception as e:  # noqa: BLE001
    print("    失败 ❌ %s: %s" % (type(e).__name__, str(e)[:120]))

print()
print("=" * 100)
print("判读")
print("=" * 100)
print("  * 若 [1] 成功而 [2] 失败 -> 问题在 **LangChain/客户端超时**，不是网络；")
print("  * 若两者都成功 -> 端点已恢复，可以继续后续验证；")
print("  * 若两者都失败且**依然耗时很久** -> 才是真的不可达。")
