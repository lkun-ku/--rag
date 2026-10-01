"""结构感知父块 / 子块切分 —— 契约测试（ROADMAP 4.51，TDD 先行）。

## 设计前提：**与语料无关**

切分规则只依赖「Markdown 结构」与「字符数上限」，**不依赖任何真实文档**——
所以用构造的 Markdown 就能把契约锁死。等真实数据到了，只需调
`parent_max_chars` / `child_max_chars` 两个参数，不改代码。

## 契约

| # | 契约 |
|---|---|
| P1 | 按 `#` 一级标题切**父块**（2 个 `#` -> 2 个父块）|
| P2 | 父块内按「段落 / 表格」切**子块** |
| P3 | `parent_id` **稳定可复现**（同输入两次结果一致）|
| P4 | 子块的 `parent_id` == 其父块的 `parent_id` |
| P5 | `chunk_index` 在同一父块内从 0 递增 |
| P6 | 章节超 `parent_max_chars` 时**在子块边界处切开**，不切断段落 |
| P7 | 表格块保留 Markdown 表头 |
| P8 | 空文档 -> `([], [])`，不抛 |
| P9 | 无标题文档 -> **1 个兜底父块** |
| P10 | `heading_path` 记录层级（`["第一章", "1.1 小节"]`）|
| P11 | 父块 `is_parent=True`、子块 `is_parent=False` |
| P12 | 子块 `page_content` 是其父块内容的一部分（不丢字）|
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from rag_kb.documents.parent_child_splitter import split_parent_child  # noqa: E402

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


SAMPLE = """# 第一章

正文 A。这里是第一章的开头段落。

## 1.1 小节

正文 B。这是小节里的内容。

| 产品 | 召回率 |
|---|---|
| TurboPuffer | 61.35% |

# 第二章

正文 C。这是第二章。
"""

print("=" * 100)
print("P1/P2 父块数、子块数")
print("=" * 100)
parents, children = split_parent_child(SAMPLE, "sample.md",
                                       parent_max_chars=4000, child_max_chars=800)
check("P1 2 个父块（按 # 切）", len(parents) == 2, len(parents))
check("P2 子块数 > 父块数", len(children) > len(parents),
      "父 %d / 子 %d" % (len(parents), len(children)))

print()
print("=" * 100)
print("P3/P4/P5 parent_id 与 chunk_index")
print("=" * 100)
p2, c2 = split_parent_child(SAMPLE, "sample.md",
                            parent_max_chars=4000, child_max_chars=800)
check("P3 parent_id 可复现",
      [d.metadata["parent_id"] for d in parents] ==
      [d.metadata["parent_id"] for d in p2])
check("P3 两个父块 id 不同",
      parents[0].metadata["parent_id"] != parents[1].metadata["parent_id"])
pid0 = parents[0].metadata["parent_id"]
kids0 = [d for d in children if d.metadata["parent_id"] == pid0]
check("P4 子块 parent_id 指向父块", len(kids0) >= 2, len(kids0))
check("P5 chunk_index 从 0 连续",
      [d.metadata["chunk_index"] for d in kids0] == list(range(len(kids0))),
      [d.metadata["chunk_index"] for d in kids0])

print()
print("=" * 100)
print("P7/P10/P11/P12 元数据与内容")
print("=" * 100)
all_child_text = "\n".join(d.page_content for d in children)
check("P7 表格保留表头", "| 产品 | 召回率 |" in all_child_text,
      all_child_text[:80].replace("\n", " "))
check("P7 表格保留数值", "61.35%" in all_child_text)
check("P10 heading_path 记录层级",
      any("1.1 小节" in str(d.metadata.get("heading_path", "")) for d in parents + children),
      [d.metadata.get("heading_path") for d in parents])
check("P11 父块 is_parent=True", all(d.metadata.get("is_parent") is True for d in parents))
check("P11 子块 is_parent=False", all(d.metadata.get("is_parent") is False for d in children))
check("P12 子块文本都属于父块文本",
      all(any(d.page_content.strip() in p.page_content for p in parents)
          for d in children if d.page_content.strip()))

print()
print("=" * 100)
print("P6 超长章节在子块边界切开")
print("=" * 100)
long_md = "# 长章\n\n" + "\n\n".join("第%d段：%s" % (i, "内容" * 60) for i in range(1, 8))
pl, cl = split_parent_child(long_md, "long.md", parent_max_chars=600, child_max_chars=200)
check("P6 超长章节被切成多个父块", len(pl) > 1, len(pl))
check("P6 切分不切断段落（每段仍完整出现在某个父块里）",
      all(("第%d段" % i) in "".join(p.page_content for p in pl) for i in range(1, 8)))
check("P6 父块未超限太多",
      all(len(p.page_content) <= 600 + 260 for p in pl),
      [len(p.page_content) for p in pl])

print()
print("=" * 100)
print("P8/P9 边界")
print("=" * 100)
check("P8 空文档 -> ([], [])", split_parent_child("", "e.md") == ([], []))
check("P8 纯空白 -> ([], [])", split_parent_child("   \n\n  ", "e.md") == ([], []))
pn, cn = split_parent_child("没有标题的一段正文。\n\n第二段。", "n.md")
check("P9 无标题 -> 1 个兜底父块", len(pn) == 1, len(pn))
check("P9 兜底父块含全部内容",
      "没有标题的一段正文" in pn[0].page_content and "第二段" in pn[0].page_content)

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
