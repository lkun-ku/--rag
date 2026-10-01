"""TDD：`web_search` 节点返回的 `documents` 必须是 **list**（与其它节点一致）。

## 问题（4.27.4 记录，本次修复）

`graph2/web_search_node.py` 返回**单个 `Document`**：

    web_results = Document(page_content=..., metadata={...})
    return {"documents": web_results, ...}          # 单个，不是列表

而**其它所有节点都返回 `List[Document]`**。于是任何假设「documents 是 list」
的代码，在**走了 web_search 的路径上都会崩**：

    TypeError: object of type 'Document' has no len()

**已确认的受害点**（代码审查扫出来的）：
`eval/fullchain_audit.py`（首版就这么崩的）、`eval/verify_e2e.py`、
`eval/_probe_real_tokens.py`、`eval/diag_grade_retention.py`、`eval/verify_final_k_ab.py`。

**为什么现在敢改**：唯一的下游是 `generate`，而它**本来就防御了两种情况**
（`documents if isinstance(documents, list) else [documents]`）——
所以改成 list 对它是**零影响**，而所有「假设 list」的代码都能因此工作。

## 契约

    C1 `web_search_node` 返回的 `documents` **是 list**（类型一致）
    C2 列表长度 1，且**语义不变**：内容与 metadata 与改造前一致
       （`source=web` / `filename` / `title` 三个标记必须还在 —— P0-9 的溯源依赖它们）

**注意**：search 本身会调 Tavily（本环境 key 是 placeholder、必然失败），
所以测试**不依赖真实的搜索结果**，只验证**返回值的形状**。
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

failures = []
passed = 0


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%s  实际=%s" % (name, want, got))


print("=" * 96)
print("web_search_node 的 documents 类型")
print("=" * 96)
print()

try:
    from graph2.web_search_node import web_search
except ImportError as e:  # noqa: BLE001
    print("  [**FAIL**] import 失败: %s" % e)
    sys.exit(1)

out = web_search({"question": "今天北京的天气怎么样？"})
docs = out.get("documents")

# --- C1 类型 -----------------------------------------------------------------
print("[C1] 返回的 documents 必须是 list")
check("isinstance(documents, list)", isinstance(docs, list), True)
if isinstance(docs, list):
    check("长度 == 1（单条网络结果）", len(docs), 1)
    # 「假设 list」的代码应当能正常工作 —— 这是本次修复的核心目标
    try:
        check("len()/迭代不再抛异常", len(list(docs)), 1)
    except TypeError as e:
        check("len()/迭代不再抛异常", "TypeError: %s" % e, True)

# --- C2 语义不变（P0-9 的溯源标记）-------------------------------------------
print()
print("[C2] 语义不变：metadata 的三个标记必须保留（P0-9 依赖它们）")
if isinstance(docs, list) and docs:
    d = docs[0]
    md = getattr(d, "metadata", {}) or {}
    check("metadata['source'] == 'web'", md.get("source"), "web")
    check("metadata 有 filename", bool(md.get("filename")), True)
    check("metadata 有 title", bool(md.get("title")), True)
    print("     filename=%r  title=%r" % (md.get("filename"), md.get("title")))
else:
    check("能取到第一个文档", False, True)

# --- 顺带：generate 对两种形状都要能工作（防回归）-----------------------------
print()
print("[回归保护] generate 对「list」与「单个 Document」都应能工作")
gen_src = (ROOT / "RAG_PROJECT" / "graph2" / "generate_node2.py").read_text(encoding="utf-8")
check("generate 仍保留 isinstance 兼容判断", "isinstance(documents, list) else [documents]" in gen_src, True)

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
