"""验证 P0-3：区分「幻觉引用」与「复述被检索块原文自带的编号」。

背景（ROADMAP P0-3）：
    修复前 `validate_citations` 把**任意超出 1..n_docs 的方括号数字**都判为幻觉引用。
    而 `papers` 集合的块正文本身就含参考文献编号（`[28]`、`[7]`）与年份（`[2024]`），
    LLM 复述原文时会带上它们 -> 被判越界 -> graph_2 走 `not supported`
    -> **触发重生成（最多 2 次）**，白烧 LLM 调用且可能把正确答案改坏。
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

from utils.citation import _numbers_in_docs, validate_citations  # noqa: E402


def D(t: str) -> Document:
    return Document(page_content=t, metadata={"filename": "x.md"})


docs = [
    D("LayoutParser 提供了统一的版面分析接口 [28]。"),
    D("该工作由 Smith 等人完成 [2024]，后续被多次引用。"),
    D("更多细节见 [7, 9]。"),
]

print("=" * 100)
print("_numbers_in_docs 提取结果（应含 28 / 2024 / 7 / 9）")
print("=" * 100)
nums = sorted(_numbers_in_docs(docs))
print("  %s" % nums)
assert nums == [7, 9, 28, 2024], nums

CASES = [
    ("正常引用（范围内）", "[3]", 5, docs, [], []),
    ("**原文编号 [28]**（修复前被判幻觉）", "[28]", 5, docs, [], [28]),
    ("**原文年份 [2024]**", "[2024]", 5, docs, [], [2024]),
    ("原文 [7, 9] 里的 9（逗号形式）", "[9]", 5, docs, [], [9]),
    ("真幻觉 [99]（原文里没有）", "[99]", 5, docs, [99], []),
    ("混合：合法 [2] + 原文 [28] + 幻觉 [99]", "答案 [2][28][99]", 5, docs, [99], [28]),
    ("不传 docs（向后兼容：越界一律 invalid）", "[28]", 5, None, [28], []),
    ("多原文编号", "[28][2024][7]", 5, docs, [2024, 28, 7] if False else [], [28, 2024, 7]),
]

print()
print("=" * 100)
print("validate_citations 验证")
print("=" * 100)
n_ok = 0
for name, answer, n_docs, ds, want_invalid, want_quoted in CASES:
    r = validate_citations(answer, n_docs, docs=ds)
    ok = (
        sorted(r["invalid"]) == sorted(want_invalid)
        and sorted(r["quoted"]) == sorted(want_quoted)
        and r["valid"] == (not want_invalid)
    )
    n_ok += ok
    print()
    print("  [%s] %s" % ("OK  " if ok else "FAIL", name))
    print("        cites=%s invalid=%s quoted=%s valid=%s" % (r["cites"], r["invalid"], r["quoted"], r["valid"]))
    if not ok:
        print("        期望 invalid=%s quoted=%s" % (want_invalid, want_quoted))

print()
print("  => %d/%d 通过" % (n_ok, len(CASES)))

# is_refusal 不应受影响
print()
print("=" * 100)
print("拒答判定未受影响")
print("=" * 100)
r = validate_citations("知识库中未找到依据。", 5, docs=docs)
print("  is_refusal=%s valid=%s invalid=%s" % (r["is_refusal"], r["valid"], r["invalid"]))
print("  => %s" % ("OK" if r["is_refusal"] and r["valid"] else "FAIL"))
