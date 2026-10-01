"""探针：Tavily key 填入后，web_search 能否真正联网？

## 为什么单独测

4.27 的全链路审计里，**库外 10 题全部走 `web_search` 但最终仍拒答** ——
原因是当时 `TAVILY_API_KEY` 是 placeholder，搜索调不通、无内容可答。

所以那条「库外拒答率 100%」**不代表生产行为**。现在 key 是有效的，
可以测到**之前测不到的那条分支**：联网成功 -> 有内容 -> 作答。

本探针只答一个问题：**Tavily 通不通**。通了再去跑全链路审计。
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

key = os.environ.get("TAVILY_API_KEY", "")
print("TAVILY_API_KEY = %s...%s（长度 %d）" % (key[:12], key[-6:], len(key)))
print()

# 直接测 Tavily（不经 LangChain 封装，先确认服务本身可用）
print("[1] 裸 Tavily API")
try:
    from tavily import TavilyClient

    t0 = time.time()
    c = TavilyClient(api_key=key)
    r = c.search("Milvus 是什么", max_results=2)
    el = time.time() - t0
    res = r.get("results", [])
    print("    成功 ✅ 耗时 %.1fs，结果 %d 条" % (el, len(res)))
    for x in res[:2]:
        print("      - %s" % str(x.get("title"))[:70])
        print("        %s" % str(x.get("content"))[:100].replace("\n", " "))
except Exception as e:  # noqa: BLE001
    print("    失败 ❌ %s: %s" % (type(e).__name__, str(e)[:180]))

# 项目实际用的封装
print()
print("[2] 项目封装（langchain_tavily / 或项目里的 web_search 工具）")
try:
    from tools.retriever_tools import web_search_tool

    t0 = time.time()
    r = web_search_tool.invoke("Milvus 是什么")
    el = time.time() - t0
    txt = r if isinstance(r, str) else str(r)
    print("    成功 ✅ 耗时 %.1fs，返回 %d 字符" % (el, len(txt)))
    print("    预览: %s" % txt[:220].replace("\n", " "))
except Exception as e:  # noqa: BLE001
    print("    失败 ❌ %s: %s" % (type(e).__name__, str(e)[:180]))

print()
print("=" * 100)
print("若两条都成功 -> 可以重跑 `eval/fullchain_audit.py`，那时才能看到")
print("「库外题联网成功后会不会作答」这条**之前测不到的分支**。")
