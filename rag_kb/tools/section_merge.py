"""按章节聚合：**检索后**把同章节的子块拼成父块（ROADMAP 4.49）。

## 为什么用「检索后聚合」而不是「建父块索引」

Web 调研（2026，Atlan / Future AGI）的结论：

    Parent-Child 是生产默认（ARAGOG 2024 benchmark 排名第一），
    但**建两层索引**的代价是：存储翻倍 + 源文档更新要刷两层 +
    现有 523 块全量重灌 + golden 32 题的 source_blocks 全部失效要重标。

Atlan 同时指出一个更轻的变体：

    "The **sentence-window variant requires no hierarchical index — simpler to implement**"

**本模块走这条路**：索引仍按子块存（零改动），只在**生成前**把「同一章节的子块」
聚合成一个父块。效果等价于 small-to-big 的**生成侧收益**，代价几乎为零。

## 与 `merge_table_slices` 的分工（互补，不重复）

    merge_table_slices   : 只合并**表格**块（同一张表被切碎时）
    merge_section_blocks : 合并**整个章节**（段落 + 表格都参与）-> 父块

调用顺序：先 `merge_table_slices`（表格拼回整表），再 `merge_section_blocks`
（整表 + 同章节段落一起成父块）。这样父块里既不会出现「半张表」，
也不会丢掉章节内的上下文。

## 默认关闭

`SECTION_MERGE_ENABLED=0`（默认）-> 直接返回原列表，**对既有指标零影响**。
开启会让上下文显著变长（父块可达数千字），**必须先 A/B 验证 token 与质量**，
不能默认开。这是本项目一贯做法（reranker 也默认关）。
"""

from __future__ import annotations

import os

from rag_kb.utils.log_utils import log

# [ROADMAP 4.53] 新名 `KB_SECTION_MERGE` 优先，旧名 `SECTION_MERGE_ENABLED` 兜底 —— 两种 .env 写法都可用。
SECTION_MERGE_ENABLED = (os.getenv("KB_SECTION_MERGE") or os.getenv("SECTION_MERGE_ENABLED", "0")) not in (
    "",
    "0",
    "false",
    "False",
)
# 截断会丢内容，而一个章节拆成两段仍比散落的子块更有上下文。
SECTION_MERGE_MAX_CHARS = int(os.getenv("SECTION_MERGE_MAX_CHARS", "8000"))
# 至少要几块才值得合并（1 块没有合并意义）
_MERGE_MIN_BLOCKS = 2


def merge_section_blocks(docs, max_chars: int | None = None):
    """把同 `(filename, title)` 的块按顺序拼成父块。

    Args:
        docs: `list[Document]`（建议是 `merge_table_slices` 之后的）。
        max_chars: 单父块字符上限；超出则切成多个父块。None 用配置默认。

    Returns:
        list[Document]：合并后的列表；未启用或无可合并时**原样返回**。

    合并规则：
      * 分组键 = `(filename, title)` —— 与 `merge_table_slices` 一致；
      * 组内保持**原有顺序**（检索顺序反映相关性排序，不重排）；
      * 累计超过 `max_chars` 就另起一个父块（**不截断**，避免丢内容）；
      * 合并块的 metadata 取组内第一块 + `merged_from` 标记（便于排查）。
    """
    if not SECTION_MERGE_ENABLED:
        return docs
    if not isinstance(docs, list) or len(docs) < _MERGE_MIN_BLOCKS:
        return docs

    limit = SECTION_MERGE_MAX_CHARS if max_chars is None else max_chars

    groups: dict[tuple, list[tuple[int, object]]] = {}
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", None) or {}
        key = (md.get("filename"), md.get("title"))
        groups.setdefault(key, []).append((i, d))

    out: list[tuple[int, object]] = []
    merged_groups = 0
    for _key, items in groups.items():
        if len(items) < _MERGE_MIN_BLOCKS:
            out.extend(items)
            continue

        buf_lines: list[str] = []
        buf_size = 0
        buf_items: list[tuple[int, object]] = []

        def flush():
            nonlocal buf_lines, buf_size, buf_items
            if not buf_items:
                return
            if len(buf_items) == 1:
                out.append(buf_items[0])
            else:
                head = buf_items[0][1]
                md = dict(getattr(head, "metadata", None) or {})
                md["merged_from"] = len(buf_items)
                md["merged_chars"] = buf_size
                out.append((buf_items[0][0], type(head)(page_content="\n\n".join(buf_lines), metadata=md)))
            buf_lines, buf_size, buf_items = [], 0, []

        for idx, d in items:
            txt = getattr(d, "page_content", "") or ""
            if buf_items and buf_size + len(txt) > limit:
                flush()
            buf_lines.append(txt)
            buf_size += len(txt) + 2  # +2 是 "\n\n" 分隔
            buf_items.append((idx, d))
        flush()
        merged_groups += 1

    out.sort(key=lambda x: x[0])
    result = [d for _, d in out]
    if merged_groups:
        log.info("[section_merge] %d 个章节被聚合：%d 块 -> %d 块" % (merged_groups, len(docs), len(result)))
    return result
