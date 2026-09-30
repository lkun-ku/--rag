"""表格结构化查询：把「跨行比较」交给代码算，不交给 LLM 猜（ROADMAP 4.47）。

## 为什么做这个

聚合类问题（「召回率最低的是哪个」）的三层排查结果：

    ① 检索   -> 含 61.35 的块在上下文里            ✅ 排除
    ② 合并   -> 87 行完整表已合成一块（含 61.35/89.93）✅ 排除
    ③ 推理   -> 模型仍答 89.93%，只比较了第一个场景   ❌ 这才是瓶颈

加 prompt 引导后从 0/3 提升到 **1/3**，改善但不稳定 —— 因为「扫描 87 行 × 8 列 ≈ 700 个单元格
取最小值」本质上是**精确计算**，是 LLM 的固有弱项。

**本模块把这步换成 pandas 计算**：算出确定结果，作为「事实」注入 prompt，
模型只需陈述，不需要自己扫表。

## 边界（不夸大）

    能处理：单块 markdown 表格内的**单列**聚合（min/max/mean/sum/count）
    不能处理：跨多表联合、多列条件筛选、时间序列趋势、图表内数值
    失败即返回 None（调用方降级回原路径，不阻断）
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

# 聚合操作识别（顺序有意义：先判极端值，再判均值/合计/计数）
_OP_PATTERNS = [
    ("min", r"最低|最小|最少|最差|最小的是"),
    ("max", r"最高|最大|最多|最好|最高的是"),
    ("mean", r"平均|均值"),
    ("sum", r"总和|合计|总计"),
    ("count", r"多少个|几条|几行|多少条|数量"),
]

# 指标提示词 -> 用于在表头里定位目标列
_METRIC_PATTERNS = [
    r"召回率", r"准确率", r"精度", r"QPS", r"P99", r"P95", r"延迟", r"耗时",
    r"吞吐", r"占比", r"百分比", r"分数", r"得分", r"过滤比例",
]

_OP_LABEL = {"min": "最低", "max": "最高", "mean": "平均", "sum": "合计", "count": "数量"}


def _to_num(s: str) -> Optional[float]:
    """把单元格转成数字：支持 `1,234` / `61.35%` / `12.5 ms` / `-3`。"""
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("，", "")
    t = re.sub(r"[%％]\s*$", "", t)
    t = re.sub(r"\s*(ms|s|MB|GB|个|次)\s*$", "", t, flags=re.I)
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t)
    return None


def parse_markdown_table(text: str):
    """从一块文本里解析出**第一个** markdown 表格。

    Returns:
        (headers, rows) —— rows 是 list[list[str]]；无表格返回 (None, []).
    """
    if not text:
        return None, []
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return None, []

    def cells(line: str):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = cells(lines[0])
    body = []
    for l in lines[1:]:
        # 跳过分隔行 |---|---|
        if set(l.replace("|", "").replace(" ", "")) <= set("-:"):
            continue
        r = cells(l)
        if len(r) == len(headers):
            body.append(r)
    return (headers if body else None), body


def _pick_column(question: str, headers: Sequence[str]) -> Optional[int]:
    """在表头里找问题所指的指标列。优先命中问题里出现的指标词。"""
    for pat in _METRIC_PATTERNS:
        if re.search(pat, question, re.I):
            for i, h in enumerate(headers):
                if re.search(pat, str(h), re.I):
                    return i
    return None


def _pick_op(question: str) -> Optional[str]:
    for op, pat in _OP_PATTERNS:
        if re.search(pat, question):
            return op
    return None


def answer_aggregation(question: str, docs) -> Optional[str]:
    """对聚合类问题做**确定性计算**，返回可直接陈述的事实文本。

    Args:
        question: 用户问题。
        docs: `list[Document]`（已过同表合并）。

    Returns:
        形如「按『召回率』列比较 85 行：最低 61.35%（场景=…, 产品=…）」；
        **无法处理时返回 None**（调用方降级，不阻断主流程）。
    """
    op = _pick_op(question)
    if not op:
        return None

    # 找到含目标列的第一张表（多块时取行数最多的那块，通常是合并后的完整表）
    best = None
    best_rows = 0
    for d in docs or []:
        text = getattr(d, "page_content", "") or ""
        headers, rows = parse_markdown_table(text)
        if not headers or not rows:
            continue
        col = _pick_column(question, headers)
        if col is None:
            continue
        if len(rows) > best_rows:
            best, best_rows = (headers, rows, col, d), len(rows)

    if not best:
        return None

    headers, rows, col, doc = best
    pairs = []
    for r in rows:
        v = _to_num(r[col])
        if v is not None:
            pairs.append((v, r))
    if not pairs:
        return None

    if op == "count":
        return "按『%s』列统计：共 %d 行有数值。" % (headers[col], len(pairs))

    if op == "min":
        target = min(pairs, key=lambda x: x[0])
    elif op == "max":
        target = max(pairs, key=lambda x: x[0])
    elif op == "mean":
        avg = sum(v for v, _ in pairs) / len(pairs)
        return ("按『%s』列比较 %d 行：平均值为 %.2f（原始值见表格）。"
                % (headers[col], len(pairs), avg))
    else:  # sum
        return ("按『%s』列统计 %d 行：合计 %.2f。"
                % (headers[col], len(pairs), sum(v for v, _ in pairs)))

    val, row = target
    # 拼「定位信息」：取除目标列外的其它列（最多 4 个），便于回答里交代条件
    ctx = []
    for i, h in enumerate(headers):
        if i == col:
            continue
        cell = str(row[i]).strip()
        if cell and cell != "-":
            ctx.append("%s=%s" % (h, cell))
        if len(ctx) >= 4:
            break

    return ("按『%s』列比较**全表 %d 行**：%s值为 %.2f%%（位于：%s）。"
            % (headers[col], len(pairs), _OP_LABEL[op], val, ", ".join(ctx))
            if "%" in str(row[col]) else
            "按『%s』列比较**全表 %d 行**：%s值为 %s（位于：%s）。"
            % (headers[col], len(pairs), _OP_LABEL[op], row[col], ", ".join(ctx)))
