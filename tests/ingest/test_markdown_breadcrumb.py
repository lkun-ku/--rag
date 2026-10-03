"""Markdown 分块的标题面包屑 / 归属契约测试（ROADMAP 4.x）。

## 为什么有这组测试

`merge_title_content` 依赖 `unstructured` 的 `metadata.parent_id` 还原章节层级。
实测本项目 6 份语料中该字段**大量缺失**（不是悬空，是根本没有）：

    milvus_cli_reference.md   90 个标题里 59 个无父，36 个代码块全部无父
    milvus_troubleshooting.md 30 块里有 15 块 title 退化成文件名
    milvus_faq.md             41 个标题里 23 个无父

后果不是「少合并几次」，而是**块内容错乱**：子块开头粘着父块全文、表块全都叫
`Options`、代码块丢失命令上下文。本文件把正确行为固化成契约。

## 契约（观测点在 `MarkdownParser.parse_markdown_to_documents`）

| # | 契约 |
|---|---|
| P1 | 面包屑只含**标题链**，绝不把父级正文复制进子块 |
| P2 | `parent_id` 缺失时，按标题深度（`category_depth`）推断归属；代码块挂在所属标题下 |
| P3 | 表格块的标题必须带命令名（`命令 -> Options`），不能只有 `Options` |
| P4 | 只有标题、没有正文的章节不得污染任何 content 块 |

## 判据来源

P1/P3/P4 的期望值来自语料本身的结构（`## connect` 下先 `### Syntax` 再 `### Options`，
表是 connect 的参数表），不是照抄实现。P2 的判据是「代码块属于它前面那个标题」。
"""

from __future__ import annotations

import os
import sys
import tempfile

from langchain_core.documents import Document  # noqa: F401

from rag_kb.documents.markdown_parser import MarkdownParser
from rag_kb.utils.log_utils import log

log.remove()

FAILED = []
TOTAL = 0

# 正文都写到 50 字符以上：低于 MIN_CHUNK_CHARS 会触发「超短块合并」，
# 那会把断言变成在测合并策略而不是面包屑本身。
BODY_A = "本指南介绍 Milvus 的安装步骤，这一段是父级正文，长度足够长以避免被超短块合并逻辑处理掉。"
BODY_B = "环境要求正文：需要 Python 3.10 以及至少 8GB 内存，这一段同样是足够长的正文内容，用于验证子块是否被污染。"
BODY_C = "依赖组件正文：Milvus 依赖 etcd 与 minio 两个外部组件，这一段同样足够长，用于验证子块内容。"
BODY_CONNECT = "连接 Milvus 服务，使用 uri 参数指定地址，这一段正文足够长以避免被超短块合并逻辑处理掉。"
BODY_DISCONNECT = "断开与 Milvus 的连接，这一段正文足够长以避免被超短块合并逻辑处理掉，用于验证空章节。"


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if cond:
        print("  [OK ] %s" % name)
    else:
        print("  [FAIL] %s  %s" % (name, detail))
        FAILED.append(name)


def parse(md_text, name="fixture.md"):
    """写临时 md 并走真实解析链路。"""
    tmp = tempfile.mkdtemp(prefix="kb_test_breadcrumb_")
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_text)
    try:
        return MarkdownParser().parse_markdown_to_documents(path)
    finally:
        for p in os.listdir(tmp):
            os.remove(os.path.join(tmp, p))
        os.rmdir(tmp)


def find(docs, needle):
    return [d for d in docs if needle in d.page_content]


def show(docs):
    for d in docs:
        print(f"      cat={str(d.metadata.get('category')):8s} title={str(d.metadata.get('title'))[:40]!r}")
        print(f"        {d.page_content[:120]!r}")


print("=" * 100)
print("P1 面包屑只含标题链，不复制父级正文")
print("=" * 100)
docs = parse(
    "\n".join([
        "# 安装指南",
        "",
        BODY_A,
        "",
        "## 环境要求",
        "",
        BODY_B,
        "",
        "### 依赖组件",
        "",
        BODY_C,
        "",
    ])
)
leaf = find(docs, "依赖组件正文")
check("P1 依赖组件正文有对应块", len(leaf) == 1, len(leaf))
if leaf:
    text = leaf[0].page_content
    check("P1 不含父级正文（安装指南）", "父级正文" not in text, text[:80])
    check("P1 不含父级正文（环境要求）", "环境要求正文" not in text, text[:80])
    check("P1 含完整标题链", "安装指南" in text and "环境要求" in text, text[:80])
else:
    show(docs)

print()
print("=" * 100)
print("P2 parent_id 缺失时按标题深度推断归属（代码块挂在所属标题下）")
print("=" * 100)
docs = parse(
    "\n".join([
        "# 命令参考",
        "",
        "## connect",
        "",
        BODY_CONNECT,
        "",
        '<h3 id="connect">Syntax</h3>',
        "",
        "```shell",
        "connect [-uri http://127.0.0.1:19530]",
        "```",
        "",
    ])
)
code = find(docs, "connect [-uri http://127.0.0.1:19530]")
check("P2 代码块有对应块", len(code) == 1, len(code))
if code:
    title = str(code[0].metadata.get("title"))
    check("P2 代码块 title 含命令名 connect", "connect" in title, title)
    check("P2 代码块 title 含小节名 Syntax", "Syntax" in title, title)
    check("P2 代码块 title 不是文件名", title != "fixture.md", title)
else:
    show(docs)

print()
print("=" * 100)
print("P3 表格块标题带命令名（不能只有 Options）")
print("=" * 100)
docs = parse(
    "\n".join([
        "# 命令参考",
        "",
        "## connect",
        "",
        BODY_CONNECT,
        "",
        '<h3 id="connect">Options</h3>',
        "",
        "| Option | Full name | Description |",
        "| --- | --- | --- |",
        "| -uri | --uri | 连接的地址，默认 http://127.0.0.1:19530 |",
        "| -t | --token | 认证用的 token |",
        "",
    ])
)
tables = [d for d in docs if d.page_content.startswith("表格：")]
check("P3 表块存在", len(tables) == 1, len(tables))
if tables:
    title = str(tables[0].metadata.get("title"))
    check("P3 表块 title 含命令名 connect", "connect" in title, title)
    check("P3 表块正文含命令名 connect", "connect" in tables[0].page_content, tables[0].page_content[:60])
else:
    show(docs)

print()
print("=" * 100)
print("P4 只有标题没有正文的章节，不得把标题文字拼进 content 块")
print("=" * 100)
docs = parse(
    "\n".join([
        "# 命令参考",
        "",
        "## connect",
        "",
        BODY_CONNECT,
        "",
        "### Options",
        "",
        "## disconnect",
        "",
        BODY_DISCONNECT,
        "",
    ])
)
leaked = [d for d in docs if d.metadata.get("category") == "content" and "Options" in d.page_content]
check("P4 空章节标题未污染 content 块", not leaked, [d.page_content[:80] for d in leaked])
check("P4 正文块仍在", bool(find(docs, "连接 Milvus 服务")), len(docs))

print()
print("=" * 100)
print("P5 同级小节（同一父标题下的第二个 h3）也要拿到父命令名")
print("=" * 100)
# 真实语料的形状：`## connect` 下先 `### Syntax`（带代码块）再 `### Options`（带表格），
# 后一个小节的 parent_id 在 unstructured 里同样缺失（实测 CLI 文档 24 张表全部如此）。
docs = parse(
    "\n".join([
        "# 命令参考",
        "",
        "## connect",
        "",
        BODY_CONNECT,
        "",
        '<h3 id="connect">Syntax</h3>',
        "",
        "```shell",
        "connect [-uri http://127.0.0.1:19530]",
        "```",
        "",
        '<h3 id="connect">Options</h3>',
        "",
        "| Option | Full name | Description |",
        "| --- | --- | --- |",
        "| -uri | --uri | 连接的地址，默认 http://127.0.0.1:19530 |",
        "",
    ])
)
tables = [d for d in docs if d.page_content.startswith("表格：")]
check("P5 表块存在", len(tables) == 1, len(tables))
if tables:
    title = str(tables[0].metadata.get("title"))
    check("P5 同级小节的表块 title 含命令名 connect", "connect" in title, title)

print()
print("=" * 100)
print("P6 只有面包屑、没有正文的标题块不应产出块")
print("=" * 100)
docs = parse(
    "\n".join([
        "# 命令参考",
        "",
        "## connect",
        "",
        BODY_CONNECT,
        "",
        "### Options",
        "",
    ])
)
shells = [d for d in docs if d.page_content.strip() == str(d.metadata.get("title")).strip()]
check("P6 无空壳块（page_content == title）", not shells, [d.page_content[:40] for d in shells])
check("P6 正文块仍在", bool(find(docs, "连接 Milvus 服务")), len(docs))

print()
print("=" * 100)
print("P7 同一张表只入库一次（表名前缀不得影响孪生判定）")
print("=" * 100)
# 真实语料的形状：H1 很长、表挂在 h3 下，表名 = 完整面包屑（约 63 字符）。
# 这段前缀是我们自己加的，不能算进「结构化块长度」，否则长度比被压低、
# 孪生判定翻成 False，同一张表就以「扁平文本 + 结构化表格」两份入库。
docs = parse(
    "\n".join([
        "# Milvus_CLI Command Reference",
        "",
        "## create database",
        "",
        BODY_CONNECT,
        "",
        '<h3 id="create database">Options</h3>',
        "",
        "| Option | Full name | Description |",
        "| :--- | :--- | :--- |",
        "| -db | --db_name | [Required] The database name in milvus. |",
        "| --help |  | Displays help for using the command. |",
        "",
    ])
)
hits = [d for d in docs if "--db_name" in d.page_content]
check("P7 该表只有一块（不重复入库）", len(hits) == 1, [d.page_content[:70] for d in hits])
check("P7 保留的是结构化表格", bool(hits) and hits[0].page_content.startswith("表格："),
      hits[0].page_content[:40] if hits else "")

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
