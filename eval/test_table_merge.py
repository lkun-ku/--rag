"""同表切片合并 —— 契约测试（TDD，先于实现）。

## 为什么做这个功能

`eval/_exp_n005_table.py` 实测（每组 3 次）：

| 组 | 上下文 | 结果 |
|---|---|---|
| A | 生产：9 块 / 16,182 字符（含 2 个表格碎片） | **3/3 稳定拒答** |
| B | 完整表：1 块 / 8,614 字符 / 91 行 | **2/3 答对** |

**同一张表被切成 4 片，模型看不到完整表 -> 无法确认「哪个最低」-> 保守拒答。**
而 `generate_node2.py` 的注释早就点名了 n005 是「跨行统计类提问」的真实案例。

## 契约（本测试锁定的行为）

1. **同表多切片 -> 合并成 1 块**：`filename` + `title` 相同、且都含 markdown 表格行；
2. **按行号排序**：切片文本形如 `表格：…（第 31-60 行）`，按起始行号升序拼接；
3. **表头只保留一次**：分隔行（`|---|`）与表头行去重；
4. **不同表不合并**：`filename` 或 `title` 不同 -> 各自保留；
5. **超过上限放弃合并**：合并后字符数 > `max_chars` -> 返回原样
   （避免把 20 个切片拼成超长上下文，反而拖慢/超窗）；
6. **非表格切片不参与合并**：纯文本块即使 filename 相同也不合并。

## 为什么在 `generate` 里做而不是检索层

`run_eval.py:120` 的 recall 用 `retriever.invoke(q)` 的**原始块**计算，
与图内最终 `state["documents"]` 无关 —— 所以在生成层合并**不影响检索层指标**；
而 `grade_generation` 的引用校验用的是 `state["documents"]`，
`generate` 返回合并后的 documents 后仍然自洽。
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.documents import Document  # noqa: E402
from graph2.generate_node2 import merge_table_slices  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [OK ] %s" % name)
    else:
        print("  [FAIL] %s  %s" % (name, detail))
        FAILED.append(name)


def tbl(rows, start, end, fname="bench.xlsx.md", title="检索性能实测"):
    head = ("表格：%s（第 %d-%d 行）\n" % (title, start, end)
            + "| 场景 | 产品 | 召回率 |\n|---|---|---|\n")
    body = "\n".join("| %s | P%d | %.2f%% |" % (r[0], r[1], r[2]) for r in rows)
    return Document(page_content=head + body, metadata={"filename": fname, "title": title})


print("=" * 100)
print("C1 同表多切片 -> 合并为 1 块")
print("=" * 100)
d1 = tbl([("静态", 1, 83.52), ("静态", 2, 89.93)], 1, 30)
d2 = tbl([("带过滤", 3, 61.35), ("带过滤", 4, 96.81)], 31, 60)
out = merge_table_slices([d1, d2])
check("块数 2 -> 1", len(out) == 1, "实际 %d" % len(out))
if out:
    txt = out[0].page_content
    check("含 61.35", "61.35" in txt)
    check("含 83.52", "83.52" in txt)
    check("表头只出现一次", txt.count("| 场景 | 产品 | 召回率 |") == 1,
          "出现 %d 次" % txt.count("| 场景 | 产品 | 召回率 |"))
    check("分隔行只出现一次", txt.count("|---|---|---|") == 1,
          "出现 %d 次" % txt.count("|---|---|---|"))

print()
print("=" * 100)
print("C2 顺序：按「第 N-M 行」的起始行号排序（乱序输入也要正确）")
print("=" * 100)
out = merge_table_slices([d2, d1])      # 故意倒序
if out:
    txt = out[0].page_content
    i_static = txt.find("83.52")
    i_filter = txt.find("61.35")
    check("83.52 在 61.35 之前（行号小在前）", 0 <= i_static < i_filter,
          "索引 %d vs %d" % (i_static, i_filter))

print()
print("=" * 100)
print("C3 不同表不合并")
print("=" * 100)
other = tbl([("x", 9, 1.0)], 1, 30, fname="other.md", title="别的表")
out = merge_table_slices([d1, other])
check("不同 filename -> 2 块", len(out) == 2, "实际 %d" % len(out))

same_file_other_title = tbl([("y", 8, 2.0)], 1, 30, title="另一个标题")
out = merge_table_slices([d1, same_file_other_title])
check("同 filename 不同 title -> 2 块", len(out) == 2, "实际 %d" % len(out))

print()
print("=" * 100)
print("C4 非表格切片不参与合并")
print("=" * 100)
plain1 = Document(page_content="这是一段普通说明文字。", metadata={"filename": "a.md", "title": "t"})
plain2 = Document(page_content="另一段说明文字。", metadata={"filename": "a.md", "title": "t"})
out = merge_table_slices([plain1, plain2])
check("两个纯文本块 -> 不合并（2 块）", len(out) == 2, "实际 %d" % len(out))

print()
print("=" * 100)
print("C5 超过上限 -> 放弃合并（避免拼出超长上下文）")
print("=" * 100)
big1 = tbl([("a", 1, 1.0)] * 300, 1, 300)
big2 = tbl([("b", 2, 2.0)] * 300, 301, 600)
n_chars = len(big1.page_content) + len(big2.page_content)
out = merge_table_slices([big1, big2], max_chars=1000)
check("超限 -> 返回原样（2 块）", len(out) == 2,
      "原长 %d 字符，实际块数 %d" % (n_chars, len(out)))

print()
print("=" * 100)
print("C6 单块输入 -> 原样返回（幂等）")
print("=" * 100)
out = merge_table_slices([d1])
check("单块 -> 1 块且内容不变", len(out) == 1 and out[0].page_content == d1.page_content)

print()
print("=" * 100)
print("C7 混合场景：表格合并 + 非表格保留（块数减少但内容不丢）")
print("=" * 100)
mixed = [plain1, d1, d2, other]
out = merge_table_slices(mixed)
check("4 块 -> 3 块（d1+d2 合并，其余保留）", len(out) == 3, "实际 %d" % len(out))
check("普通文本仍在", any("普通说明文字" in d.page_content for d in out))
check("其它表仍在", any("别的表" in d.page_content for d in out))

print()
print("=" * 100)
if FAILED:
    print("失败 %d 项：%s" % (len(FAILED), FAILED))
    sys.exit(1)
print("全部通过")
