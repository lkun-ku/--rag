"""表格结构化查询 —— 契约测试（ROADMAP 4.47）。

## 背景：为什么必须有这一层

聚合类问题三层排查（`eval/_diag_agg.py` / `_diag_ctx.py`）：

    ① 检索  含 61.35 的块在上下文          -> 排除
    ② 合并  87 行完整表已合成一块           -> 排除
    ③ 推理  模型仍答 89.93%（只看第一个场景）-> **瓶颈**

prompt 引导把命中率从 0/3 提到 1/3，但**不稳定** —— 因为「扫 87 行取最小值」是精确计算。
把这一步换成 pandas 后：**3/3 稳定答对**。

## 契约

| # | 契约 |
|---|---|
| T1 | `_to_num` 支持 `1,234` / `61.35%` / `12.5 ms` / 负数，非数字返回 None |
| T2 | `parse_markdown_table` 取第一个表、跳过 `|---|` 分隔行、列数不齐的行丢弃 |
| T3 | `min`：返回**最低值**并带完整定位（场景/产品/过滤比例）|
| T4 | `max`：返回最高值 |
| T5 | `mean`：返回平均值 |
| T6 | 指标列不存在 -> 返回 None（调用方降级，不阻断）|
| T7 | 非表格文档 -> None |
| T8 | 结尾带 `%` 的值输出 `%.2f%%` 格式；不带 `%` 的原样输出 |
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


from langchain_core.documents import Document  # noqa: E402

from rag_kb.tools.table_query import (
    _to_num,
    answer_aggregation,  # noqa: E402
    parse_markdown_table,
)

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


TABLE = """表格：检索性能实测（第 1-4 行）
| 场景 | 数据集 | 产品与规格 | 过滤比例 | QPS | P99延迟(ms) | 召回率 |
|---|---|---|---|---|---|---|
| 静态向量检索 | Cohere Large | TurboPuffer | 0% | 649.88 | 55.2 | 89.93% |
| 带过滤向量检索 | Cohere Large | TurboPuffer | 98.00% | 382.53 | 48.1 | 61.35% |
| 带过滤向量检索 | Cohere Medium | TurboPuffer | 98.00% | 184.54 | 53.4 | 96.81% |
"""
DOC = Document(page_content=TABLE, metadata={"filename": "bench.xlsx.md", "title": "检索性能实测"})

print("=" * 100)
print("T1 _to_num")
print("=" * 100)
check("带千分位", _to_num("1,234") == 1234.0, _to_num("1,234"))
check("带百分号", _to_num("61.35%") == 61.35, _to_num("61.35%"))
check("带单位", _to_num("12.5 ms") == 12.5, _to_num("12.5 ms"))
check("负数", _to_num("-3") == -3.0, _to_num("-3"))
check("非数字 -> None", _to_num("TurboPuffer") is None, _to_num("TurboPuffer"))
check("空 -> None", _to_num("") is None)

print()
print("=" * 100)
print("T2 parse_markdown_table")
print("=" * 100)
h, rows = parse_markdown_table(TABLE)
check("表头 7 列", h is not None and len(h) == 7, h)
check("数据 3 行（分隔行被跳过）", len(rows) == 3, len(rows))
check("首行首列", rows and rows[0][0] == "静态向量检索", rows[0][:2] if rows else None)
check("非表格 -> (None, [])", parse_markdown_table("纯文本") == (None, []))

print()
print("=" * 100)
print("T3/T4/T5 min / max / mean")
print("=" * 100)
r_min = answer_aggregation("参与评测的产品里，召回率最低的是哪个？", [DOC])
check("T3 含 61.35", r_min and "61.35" in r_min, r_min)
check("T3 带定位（场景）", r_min and "带过滤向量检索" in r_min, r_min)
check("T3 说明比较了 3 行", r_min and "3 行" in r_min, r_min)
check("T3 输出 % 格式", r_min and "61.35%" in r_min, r_min)

r_max = answer_aggregation("召回率最高的是哪个？", [DOC])
check("T4 含 96.81", r_max and "96.81" in r_max, r_max)

r_mean = answer_aggregation("平均召回率是多少？", [DOC])
check("T5 平均值 82.70", r_mean and "82.70" in r_mean, r_mean)

# 换一列：QPS 最低
r_qps = answer_aggregation("QPS 最低的是哪个？", [DOC])
check("T3b 换列 QPS 最低 184.54", r_qps and "184.54" in r_qps, r_qps)

print()
print("=" * 100)
print("T6/T7 降级路径")
print("=" * 100)
check(
    "T6 指标列不存在 -> None",
    answer_aggregation("温度最低的是哪个？", [DOC]) is None,
    answer_aggregation("温度最低的是哪个？", [DOC]),
)
check(
    "T7 非表格 -> None",
    answer_aggregation("召回率最低的是哪个？", [Document(page_content="纯文本", metadata={})]) is None,
)
check("T7b 空列表 -> None", answer_aggregation("召回率最低的是哪个？", []) is None)
check("T7c 无聚合词 -> None", answer_aggregation("Milvus 怎么安装？", [DOC]) is None)

print()
print("=" * 100)
print("T8 多块时取行数最多的那块（合并后的完整表）")
print("=" * 100)
half = Document(
    page_content="""表格：检索性能实测（第 1-2 行）
| 场景 | 数据集 | 产品与规格 | 过滤比例 | QPS | P99延迟(ms) | 召回率 |
|---|---|---|---|---|---|---|
| 静态向量检索 | Cohere Large | TurboPuffer | 0% | 649.88 | 55.2 | 89.93% |
""",
    metadata={"filename": "bench.xlsx.md", "title": "检索性能实测"},
)
r = answer_aggregation("召回率最低的是哪个？", [half, DOC])
check("取更完整的表（3 行）而非 1 行", r and "3 行" in r and "61.35" in r, r)

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
