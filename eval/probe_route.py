"""探查问题路由的分布（ROADMAP 3.1 的第一个实际收益）。

起因：跑 tracer 时发现「Milvus 的混合检索是怎么实现的？」被路由到 **web_search**，
而它是明确的库内问题。评测层用 `generate` 直调（绕过路由），因此从未覆盖这一点。

本脚本对若干「明知属于知识库」的问题检查路由结果，判断是否存在系统性误判。
"""

import os


os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.graph2.query_route_chain import question_router_chain  # noqa: E402

# 明确属于知识库的问题（应 -> vectorstore）
IN_KB = [
    "Milvus 支持哪些索引类型？",
    "Milvus 的混合检索是怎么实现的？",
    "Milvus 如何做数据分片？",
    "什么是向量召回率？",
    "Milvus 的部署模式有哪些？",
    "LayoutParser 是什么？",
    "表格切片为什么要保留表头？",
]
# 明确不属于知识库（应 -> web_search 或应当拒答）
OUT_KB = [
    "今天北京的天气怎么样？",
    "2026 年世界杯在哪个国家举办？",
]

print("=" * 96)
print("问题路由检查")
print("=" * 96)
print("  %-6s %-16s %s" % ("期望", "实际", "问题"))
print("  " + "-" * 88)

bad = []
for q in IN_KB:
    try:
        src = question_router_chain.invoke({"question": q})
        got = src.datasource
    except Exception as e:  # noqa: BLE001
        got = "ERROR:%s" % type(e).__name__
    ok = got == "vectorstore"
    if not ok:
        bad.append(q)
    print("  %-6s %-16s %s" % ("vectorstore", got, q[:52]))

print()
for q in OUT_KB:
    try:
        src = question_router_chain.invoke({"question": q})
        got = src.datasource
    except Exception as e:  # noqa: BLE001
        got = "ERROR:%s" % type(e).__name__
    print("  %-6s %-16s %s" % ("web_search", got, q[:52]))

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("  库内问题误路由到 web_search: %d / %d" % (len(bad), len(IN_KB)))
for q in bad:
    print("    - %s" % q)
if bad:
    print()
    print("  影响：这些问题的答案明明在知识库里，却因走 web_search 而拿不到，")
    print("        最终表现为「拒答」或「答非所问」。**评测层完全没覆盖这条路径**。")
