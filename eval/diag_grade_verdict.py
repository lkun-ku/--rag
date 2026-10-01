"""两件事一起做：

  1. 验证 P0-9：`build_citation_meta` 能否把「网络搜索」来源与知识库块区分开
  2. 诊断 P0-9 问题一：为什么「LayoutParser 是什么？」的 18 块被**连续三轮全判不相关**

诊断的关键是分清楚两种可能：
    (a) 语料里**确实没有**答案（那 web_search 兜底是合理的，属语料问题）
    (b) 语料里有答案但**判分链把它判掉了**（属判分问题，可修）
所以逐块打印：判分结果 + 文件名 + 内容摘要。
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.documents import Document  # noqa: E402

from utils.citation import build_citation_meta  # noqa: E402

# ---------------------------------------------------------------------------
# 1. P0-9 验证：来源区分
# ---------------------------------------------------------------------------
print("=" * 104)
print("1. P0-9 验证：build_citation_meta 是否区分「网络搜索」与「知识库」")
print("=" * 104)
docs = [
    Document(page_content="知识库内容", metadata={"filename": "overview.md", "title": "概述"}),
    Document(
        page_content="网页内容",
        metadata={"source": "web", "filename": "（网络搜索，非知识库）", "title": "（网络搜索，非知识库）"},
    ),
]
meta = build_citation_meta(docs, [1, 2])
for m in meta:
    print(
        "  [%s] n=%s from_web=%-6s filename=%s"
        % ("OK  " if m["from_web"] == (m["n"] == 2) else "FAIL", m["n"], m["from_web"], m["filename"])
    )
print("  => %s" % ("通过" if meta[0]["from_web"] is False and meta[1]["from_web"] is True else "失败"))

# ---------------------------------------------------------------------------
# 2. 判分误杀诊断
# ---------------------------------------------------------------------------
Q = "LayoutParser 是什么？它提供了哪些能力？"

print()
print("=" * 104)
print("2. 判分诊断：%s" % Q)
print("=" * 104)

from graph2.grader_chain import retrieval_grader_chain  # noqa: E402
from tools.retriever_tools import get_retriever  # noqa: E402

retr = get_retriever()
docs = retr.invoke(Q)
print("检索到 %d 块" % len(docs))
print()

n_yes = 0
for i, d in enumerate(docs, 1):
    txt = d.page_content or ""
    try:
        score = retrieval_grader_chain.invoke({"question": Q, "document": txt})
        verdict = str(getattr(score, "binary_score", "?")).strip().lower()
    except Exception as e:  # noqa: BLE001
        verdict = "ERR:%s" % type(e).__name__
    n_yes += verdict == "yes"
    fn = str(d.metadata.get("filename", ""))[:30]
    cat = str(d.metadata.get("category", ""))
    print(
        "  %2d. [%-4s] %-30s cat=%-8s %3d字 | %s" % (i, verdict.upper(), fn, cat, len(txt), txt[:64].replace("\n", " "))
    )

print()
print(
    "  保留 %d / %d 块（%s）"
    % (n_yes, len(docs), "**全部被丢弃——判分链对本题系统性判 no**" if n_yes == 0 else "有块被保留")
)

# 判分链的 prompt（确认判据是否过严）
print()
print("=" * 104)
print("3. 判分链的 prompt（判断判据是否过严）")
print("=" * 104)
from graph2.grader_chain import retrieval_grader_chain as rc  # noqa: E402

for attr in ("first", "steps", "last"):
    node = getattr(rc, attr, None)
    if node is None:
        continue
    prompt = getattr(node, "prompt", None) or getattr(getattr(node, "first", None), "prompt", None)
    if prompt is not None:
        try:
            tpl = prompt.template if hasattr(prompt, "template") else str(prompt)
            print(tpl[:1400])
        except Exception as e:  # noqa: BLE001
            print("  (取 prompt 失败: %s)" % type(e).__name__)
        break
