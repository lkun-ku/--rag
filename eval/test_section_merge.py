"""按章节聚合（检索后父块）—— 契约测试（ROADMAP 4.49）。

## 契约

| # | 契约 |
|---|---|
| P1 | **默认关闭** -> 原样返回（对既有指标零影响）|
| P2 | 同 `(filename, title)` 的 >=2 块合并成 1 块 |
| P3 | 不同章节（title 不同）**不合并** |
| P4 | 单块不合并（没有合并意义）|
| P5 | 超过 `max_chars` -> **拆成多个父块**，而不是截断（不丢内容）|
| P6 | 合并块带 `merged_from` / `merged_chars` 元数据（便于排查）|
| P7 | 合并后的内容**包含全部原块文本**（不丢）|
| P8 | 输出保持原有相对顺序 |
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.documents import Document  # noqa: E402

from tools import section_merge as sm  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if cond:
        print("  [OK ] %s" % name)
    else:
        print("  [FAIL] %s  %s" % (name, detail))
        FAILED.append(name)


def mk(text, title="检索性能", fn="bench.md"):
    return Document(page_content=text, metadata={"filename": fn, "title": title})


print("=" * 100)
print("P1 默认关闭 -> 原样返回")
print("=" * 100)
sm.SECTION_MERGE_ENABLED = False
docs = [mk("段落一"), mk("段落二"), mk("段落三")]
out = sm.merge_section_blocks(docs)
check("块数不变", len(out) == 3, len(out))
check("内容不变", [d.page_content for d in out] == ["段落一", "段落二", "段落三"])

print()
print("=" * 100)
print("P2/P6/P7 开启后：同章节合并 + 元数据 + 内容不丢")
print("=" * 100)
sm.SECTION_MERGE_ENABLED = True
docs = [mk("段落一"), mk("段落二"), mk("段落三")]
out = sm.merge_section_blocks(docs, max_chars=10000)
check("P2 3 块 -> 1 块", len(out) == 1, len(out))
merged_text = out[0].page_content if out else ""
check("P7 含全部原文", all(x in merged_text for x in ("段落一", "段落二", "段落三")), merged_text[:60])
if out:
    md = out[0].metadata
    check("P6 merged_from=3", md.get("merged_from") == 3, md.get("merged_from"))
    check("P6 merged_chars > 0", (md.get("merged_chars") or 0) > 0, md.get("merged_chars"))

print()
print("=" * 100)
print("P3/P4 不同章节不合并、单块不合并")
print("=" * 100)
docs = [
    mk("A1", title="章节A"),
    mk("A2", title="章节A"),
    mk("B1", title="章节B"),
    mk("B2", title="章节B"),
    mk("C1", title="章节C"),
]
out = sm.merge_section_blocks(docs, max_chars=10000)
check("3 个章节 -> 3 块", len(out) == 3, len(out))
check("章节A 合并了", any("A1" in d.page_content and "A2" in d.page_content for d in out))
check("单块的章节C 保留", any(d.page_content.strip() == "C1" for d in out))

# 不同 filename 同 title 也不合并
docs = [mk("X1", title="同一标题", fn="a.md"), mk("X2", title="同一标题", fn="b.md")]
out = sm.merge_section_blocks(docs, max_chars=10000)
check("P3 不同 filename 不合并", len(out) == 2, len(out))

print()
print("=" * 100)
print("P5 超 max_chars -> 拆成多个父块（不截断、不丢内容）")
print("=" * 100)
sm.SECTION_MERGE_ENABLED = True
big = [mk("甲" * 60), mk("乙" * 60), mk("丙" * 60), mk("丁" * 60)]
out = sm.merge_section_blocks(big, max_chars=130)  # 每块 ~60 -> 最多 2 块合并
check("拆成 >=2 个父块", len(out) >= 2, len(out))
all_text = "".join(d.page_content for d in out)
check("P5 不丢内容（4 个标记都在）", all(x in all_text for x in ("甲", "乙", "丙", "丁")), len(all_text))
check(
    "P5 每块不超 max_chars 太多", all(len(d.page_content) <= 130 + 20 for d in out), [len(d.page_content) for d in out]
)

print()
print("=" * 100)
print("P8 顺序保持")
print("=" * 100)
docs = [mk("第一", title="A"), mk("第二", title="A"), mk("第三", title="B")]
out = sm.merge_section_blocks(docs, max_chars=10000)
texts = [d.page_content for d in out]
check("合并块在第一个出现位置", len(texts) == 2 and "第一" in texts[0] and "第三" in texts[1], texts)

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
