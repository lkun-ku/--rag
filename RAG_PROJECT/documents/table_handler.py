"""表格结构保护与表格感知切分（ROADMAP 2.4）。

## 背景（均为本项目实测结论，非通用假设）

1. `UnstructuredMarkdownLoader` 会把 Markdown 表格解析成 `category == 'Table'` 的元素，
   并**吞掉 `|` 列分隔符**，`page_content` 退化成扁平文本：

       | 并发数 | P50(ms) | P99(ms) |  ...  | 1 | 12 | 30 |
       -> '并发数 P50(ms) P99(ms) 1 12 30'

   列与值的从属关系被压平。

2. 但**结构没有丢** —— 完整保存在 `metadata['text_as_html']` 里：

       '<table><thead><tr><th>并发数</th><th>P50(ms)</th>...</thead>
        <tbody><tr><td>1</td><td>12</td><td>30</td></tr>...'

3. 表格数据语义高度相似（`10 12 30 20 24 60`），`SemanticChunker` 找不到语义断点，
   实测 200 行 × 6 列（5528 字符）**仍被当成一个块** —— 不会被切碎，
   但会形成超大块，稀释向量语义。

## 本模块做三件事

- **L1 结构恢复**：`text_as_html` -> Markdown 表格，并加「表格：{表名}」+「列名：...」前缀，
  让每个块**脱离上下文也能被读懂**。
- **L2 表格感知切分**：≤ `max_rows` 行整块保留；超出则按行切分 + **每块重复完整表头 + 标注行范围**
  （即「表头注入」）。行切天然对齐记录边界，不会让一条记录跨块。
- **L3 噪点清理**：`NaN` -> 空；整数被写成 `1200.0` -> `1200`。

## 为什么这些处理放在「合并标题」之前

`MarkdownParser.merge_title_content` 会把带 `parent_id` 的元素合并进父标题块。
若在合并之后处理，表格已与正文混成一段文本，**边界信息丢失、无法按行切**。
因此本模块在合并阶段接管 Table 元素，产出独立的内容块（表名从父标题取），
使表格拥有自己的块粒度与检索入口。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from utils.log_utils import log

# 默认行数阈值：超过则触发行切 + 表头注入。
# 取 30 与业界经验一致；实测 60 行表切成 2 块（667 / 715 字符），远小于语义切割阈值 5000。
DEFAULT_MAX_ROWS = 30

# 单块**字节数**上限。Milvus schema 的 text 字段是 VARCHAR(6000)（按**字节**计），
# 超长会导致整批写入失败（实测报 `MilvusException code=1100`）。
#
# [FIX / P0-5] 原先这里是 `DEFAULT_MAX_CHARS = 4000`（**字符**），
# 与 Milvus 的字节约束不一致 —— 中文 3 字节/字，4000 字符 ≈ 12000 字节，
# 是上限的 2 倍。换真实语料后立刻暴露：
#     milvus_capability_matrix.xlsx 的一个表块做完行切后仍有 **20184 字节**（3.4 倍上限）
#     -> papers 源 175 块只写入 88、manuals 源 97 块只写入 63（合计 121 块被拒）
# 现在统一改为按字节判定，并留 200 字节余量（表名/列名前缀、换行符等）。
DEFAULT_MAX_BYTES = 5800

# 表名长度上限（字符）。表名会写入 Milvus 的 title 字段（VARCHAR 1000 字节）。
# 实测教训：表名取自「父标题」，而 merge_title_content 会让父标题不断吸收子元素，
# 取到时可能已累积成 1496 字符的长文本，直接写库会抛
# MilvusException code=1100 (length of varchar field title exceeds max length)。
MAX_TABLE_NAME_CHARS = 200

# 空值与数值噪点。
# 说明：只清理「独立的」nan/None 记号，不碰 SKU 编号等正常文本。
_NAN_TOKENS = {'nan', 'none', 'null', 'n/a', 'na'}
# 整数值被写成浮点（1200.0 -> 1200）；带小数的（0.05）保持不变。
_INT_FLOAT_RE = re.compile(r'^(\d+)\.0+$')


def clean_cell(value: Optional[str]) -> str:
    """清理单元格文本：空值 -> 空串；整数值去多余的 .0。

    实测 markitdown 会把空单元格写成 `NaN`（该记号会进入块内容并被用户看到），
    并把整数写成 `1200.0`。
    """
    if value is None:
        return ''
    s = str(value).strip()
    if not s:
        return ''
    if s.lower() in _NAN_TOKENS:
        return ''
    m = _INT_FLOAT_RE.match(s)
    if m:
        return m.group(1)
    return s


class _TableHTMLParser(HTMLParser):
    """把 HTML 表格解析成二维网格，并展开 rowspan / colspan。

    为什么要展开：合并单元格会让后续行列错位（同一行取值偏移），
    展开后每行的长度一致，行切与表头注入才有意义。
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell_parts: Optional[List[str]] = None
        # 记录被 rowspan 占用的位置：{列下标: [剩余行数, 值]}
        self._pending: Dict[int, List] = {}
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == 'tr':
            self._row = []
            self._flush_pending()
        elif tag in ('td', 'th'):
            self._cell_parts = []
            self._in_cell = True
            self._cur_span = 1
            self._cur_rowspan = 1
            for k, v in attrs:
                if k == 'colspan':
                    try:
                        self._cur_span = max(1, int(v))
                    except (TypeError, ValueError):
                        pass
                elif k == 'rowspan':
                    try:
                        self._cur_rowspan = max(1, int(v))
                    except (TypeError, ValueError):
                        pass

    def _flush_pending(self) -> None:
        """把上一行遗留下来、需要向下延续的单元格补进当前行。"""
        if self._row is None:
            return
        for col in sorted(self._pending):
            info = self._pending[col]
            while len(self._row) <= col:
                self._row.append('')
            self._row[col] = info[1]

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ('td', 'th') and self._row is not None and self._cell_parts is not None:
            value = clean_cell(''.join(self._cell_parts))
            self._in_cell = False
            self._cell_parts = None
            span = getattr(self, '_cur_span', 1)
            rowspan = getattr(self, '_cur_rowspan', 1)

            # 避开被上方 rowspan 占用的列
            col = 0
            while col < len(self._row) or col in self._pending:
                if col < len(self._row) or col in self._pending:
                    col += 1
                else:
                    break
            while len(self._row) < col:
                self._row.append('')
            for _ in range(span):
                self._row.append(value)
            if rowspan > 1:
                for offset in range(span):
                    # [FIX / P0-2] 这里必须存 **rowspan**，不是 rowspan-1。
                    #
                    # 计数在 `</tr>` 里递减（每过一行 -1），所以「还能延续几行」必须在
                    # **本行结束时**仍 > 0：
                    #     rowspan=2 --存 2--> 本行 `</tr>` 减到 1（保留）-> 下一行 flush 补位
                    #                -> 再下一行 `</tr>` 减到 0 并清理     ✓
                    # 原先存 rowspan-1：
                    #     rowspan=2 --存 1--> **本行** `</tr>` 就减到 0 被 pop
                    #                -> 下一行 `_flush_pending()` 拿不到延续值 -> 后续行整体左移一列  ✗
                    # 后果是**静默数据错位**（LLM 读到错误的行列对应关系），
                    # 正是 2.4 表格结构恢复要解决的问题本身。
                    self._pending[col + offset] = [rowspan, value]
        elif tag == 'tr' and self._row is not None:
            if any(c for c in self._row):
                while len(self.rows) and len(self.rows[-1]) < len(self._row):
                    self.rows[-1].append('')
                self.rows.append(self._row)
            # 递减 rowspan 计数
            for col in list(self._pending):
                info = self._pending[col]
                info[0] -= 1
                if info[0] <= 0:
                    self._pending.pop(col, None)
            self._row = None


def parse_table_html(html: str) -> List[List[str]]:
    """HTML 表格 -> 二维网格（第一行为表头）。解析失败返回空列表。"""
    if not html or '<t' not in html.lower():
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - HTML 畸形时降级为「不恢复」，不阻断解析
        return []
    rows = [[c for c in row] for row in parser.rows if any(c for c in row)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r + [''] * (width - len(r)) for r in rows]


def grid_to_markdown(
    rows: Sequence[Sequence[str]], table_name: str = '', row_range: Optional[Tuple[int, int]] = None
) -> str:
    """二维网格 -> 自解释的 Markdown 文本块。

    前缀三个要素（表名 / 行范围 / 列名）是刻意的：
      - 表名与行范围便于引用与去重（对应 ROADMAP 2.4 的引用粒度决策）
      - 「列名：」行让列名独立成串，对关键词检索（BM25）与 LLM 阅读都更友好
    """
    if not rows:
        return ''
    header = list(rows[0])
    body = [list(r) for r in rows[1:]]

    lines: List[str] = []
    if table_name:
        rng = f'（第 {row_range[0]}-{row_range[1]} 行）' if row_range else ''
        lines.append(f'表格：{table_name}{rng}')
    lines.append('列名：' + ' | '.join(header))
    lines.append('')
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header)) + '|')
    for r in body:
        lines.append('| ' + ' | '.join(r[: len(header)]) + ' |')
    return '\n'.join(lines).strip()


def _nbytes(s: str) -> int:
    """按 UTF-8 字节数计长（Milvus 的 VARCHAR 约束是字节，不是字符）。"""
    return len(s.encode('utf-8'))


def build_table_chunks(
    html: str, table_name: str = '', max_rows: int = DEFAULT_MAX_ROWS, max_bytes: int = DEFAULT_MAX_BYTES
) -> List[str]:
    """HTML 表格 -> 1~N 个自解释文本块（L1 + L2 + L3 的完整实现）。

    决策流程（ROADMAP 2.4.2）：
        行数 <= max_rows 且**字节数** <= max_bytes -> 整表一块（小表不切）
        任一超限                                  -> 按行分组，每块重复完整表头并标注行范围

    为何需要 max_bytes 这道闸（而不是只看行数）：
        Milvus 的 text 是 VARCHAR(6000) 且**按字节**计。行数不多但单元格很长
        （如整段说明文字）的表仍会超限，并导致**整批写入失败**。

    [FIX] 防御 `max_rows <= 0` 导致的死循环（P0-6）：
        原实现里 `end = min(start + max_rows, len(body))`，当 max_rows<=0 时
        `end <= start`，随后 `start = end` 不前进 -> 无限循环。
    """
    max_rows = max(1, int(max_rows or 1))
    rows = parse_table_html(html)
    if not rows:
        return []
    header, body = rows[0], rows[1:]

    whole = grid_to_markdown(rows, table_name)
    if len(body) <= max_rows and _nbytes(whole) <= max_bytes:
        return [whole]

    chunks: List[str] = []
    start = 0
    while start < len(body):
        end = min(start + max_rows, len(body))
        # 在行数上限内按**字节**上限收缩（至少保留 1 行，避免死循环）
        while end > start + 1:
            cand = grid_to_markdown([header] + body[start:end], table_name, (start + 1, end))
            if _nbytes(cand) <= max_bytes:
                break
            end -= 1
        chunks.append(grid_to_markdown([header] + body[start:end], table_name, (start + 1, end)))
        start = end
    return chunks


# ---------------------------------------------------------------------------
# 扁平孪生判定（ROADMAP 4.17）
#
# ## 要解决的问题
#
# 一个 unstructured `Table` 元素会被**两条路径**处理：
#   1. 其 `page_content` 是**扁平文本**（表格被压成一串词），走原有合并逻辑；
#   2. 其 `text_as_html` 被 `restore_table_documents()` 重建为**结构化 Markdown 块**。
#
# 当前实现**两份都入库**（`markdown_parser` 的注释写「少量重复，可接受」）。
# 实测**「少量」不成立**：库里 528 块中有 **236 块（44.7%）**
# 落在「同 filename+title」组里，且**直接损害指标** —— n005 的 4 个 golden 中
# 有 2 个是扁平版，`recall` 因此从 1.00 掉到 0.50（详见 ROADMAP 4.15 / 4.16）。
#
# ## 判据：三维同时满足（**严格向：宁可漏判，不误删**）
#
#   1. **正向覆盖** —— 结构化块里的长值**全部**出现在扁平文本里；
#   2. **反向覆盖** —— 扁平文本里的长词**几乎全部**能在结构化块里找到；
#   3. **长度比** —— 扁平 / 结构化总长在合理区间。
#
# ## 阈值全部由实测标定（`eval/measure_flat_twin.py`，库里 15 组）
#
#     正向覆盖率  min 0.98  中位 1.00  max 1.00
#     反向覆盖率  min 0.41  中位 1.00  max 1.00
#     长度比      min 0.74  中位 0.84  max 1.47
#
# **反向阈值取 0.95**，落在实测的空隙里：纯孪生全是 1.00，含额外正文的最高 0.69。
#
# **为什么必须有反向覆盖**（本次最关键的一课）：
# 首版只看正向覆盖，而两个真实样本的**正向覆盖率都是 1.00**，性质却完全不同 ——
#   * `milvus_capability_matrix / 来源说明`：扁平 = 结构化去掉 `|`/`---`（纯孪生，可删）；
#   * `milvus_index_spec / 6.2 依赖组件版本`：扁平里多了
#     「6 部署环境要求 Before installing a Milvus Standalone instance, check…」
#     「出处：milvus.io v2.6.x — Requirements…」
#     —— 该扁平块**已被父标题合并逻辑并入了周围正文**，直接删会**丢内容**。
#
# **不能**用「扁平 / 结构化最长块」当判据：该比值实测跨 0.74~7.13
# （一个表格元素会被切成多个行段，分母随切段数变化）。
# ---------------------------------------------------------------------------
_FLAT_TWIN_MIN_FORWARD = 0.98
_FLAT_TWIN_MIN_REVERSE = 0.95
_FLAT_TWIN_LEN_RATIO: Tuple[float, float] = (0.5, 2.0)
# 取 >= 4 字符、并排除纯数字：短值（`0%`、`40`）在扁平文本里几乎必然出现，
# 没有判别力；且格式差异（`61.35%` vs `61.35 %`）会造成误判。
_FLAT_TWIN_MIN_VALUE_CHARS = 4
_FLAT_TWIN_MIN_WORD_CHARS = 6

_NUMERICISH_RE = re.compile(r'^[\d.,%\s/\-]+$')
_FLAT_TWIN_WORD_RE = re.compile(r'[A-Za-z\u4e00-\u9fff0-9./_-]{%d,}' % _FLAT_TWIN_MIN_WORD_CHARS)
_HAS_LETTER_RE = re.compile(r'[A-Za-z\u4e00-\u9fff]')


def _norm_ws(s: str) -> str:
    """把连续空白压成单空格 —— 扁平文本与 Markdown 表格的空白布局必然不同。"""
    return re.sub(r'\s+', ' ', s)


def _flat_twin_values(chunks: Sequence[str]) -> set:
    """结构化块里「有判别力」的长值集合（Markdown 表格行中 `|` 分隔的长单元格）。"""
    vals = set()
    for c in chunks or []:
        for line in (c or '').splitlines():
            line = line.strip()
            # 只取表格数据行；`|---|---|` 这类分隔行要排除
            if not line.startswith('|') or set(line) <= set('|- '):
                continue
            for v in line.strip('|').split('|'):
                v = v.strip()
                if len(v) >= _FLAT_TWIN_MIN_VALUE_CHARS and not _NUMERICISH_RE.match(v):
                    vals.add(_norm_ws(v))
    return vals


def is_flat_twin(flat_text: str, table_chunks: Sequence[str]) -> bool:
    """该扁平文本是否为「已产出结构化块的表格」的孪生（是则可安全丢弃）。

    Args:
        flat_text: 表格元素的 `page_content`（扁平文本）。
        table_chunks: 由 `restore_table_documents()` 从**同一元素**产出的结构化块。

    Returns:
        True 表示扁平文本的内容已被结构化块完整覆盖，丢弃它不会丢内容。
        **宁可靠 False**：任一条判据不满足（或无法判定）都返回 False。

    判据与阈值来源见本节顶部注释（三维：正向覆盖 / 反向覆盖 / 长度比）。
    实测样本见 `eval/test_flat_twin.py`（正例与负例均取自库中真实数据）。
    """
    if not flat_text or not table_chunks:
        return False

    # 判据 1：正向覆盖 —— 结构化块的长值是否都在扁平文本里
    vals = _flat_twin_values(table_chunks)
    if not vals:
        # 提取不到有判别力的值（如只有表头）-> 无法判定 -> 不删（严格向）
        return False
    flat_norm = _norm_ws(flat_text)
    if sum(1 for v in vals if v in flat_norm) / len(vals) < _FLAT_TWIN_MIN_FORWARD:
        return False

    # 判据 2：反向覆盖 —— 扁平文本里有没有结构化块「完全没有」的内容
    # （若没有这一维，「已被父标题合并进周围正文」的扁平块会被误删）
    st_all = '\n'.join(table_chunks)
    words = {w for w in _FLAT_TWIN_WORD_RE.findall(flat_text) if _HAS_LETTER_RE.search(w)}
    if words and sum(1 for w in words if w in st_all) / len(words) < _FLAT_TWIN_MIN_REVERSE:
        return False

    # 判据 3：长度比 —— 防止「同内容被重复多遍」之类的畸形输入
    total = sum(len(c) for c in table_chunks)
    if total <= 0:
        return False
    lo, hi = _FLAT_TWIN_LEN_RATIO
    return lo <= len(flat_text) / total <= hi


def has_restorable_table(doc: Document) -> bool:
    """该元素是否为「可恢复结构的表格」。

    两个条件缺一不可：category 为 Table，且 metadata 里有可解析的 text_as_html。
    只满足前者时（如扫描件 OCR 出的裸文本表）走原有合并逻辑，不做特殊处理。
    """
    if doc.metadata.get('category') != 'Table':
        return False
    return bool(doc.metadata.get('text_as_html'))


def restore_table_documents(doc: Document, table_name: str = '', max_rows: int = DEFAULT_MAX_ROWS) -> List[Document]:
    """把一个 Table 元素转换成 1~N 个「结构已恢复」的内容块。

    Args:
        doc: 原始 Table 元素（含 text_as_html）。
        table_name: 表名，通常取父标题（面包屑的最后一段）。
        max_rows: 行切阈值。

    Returns:
        可直接进入切片的 Document 列表（category='content'）。
        若 HTML 无法解析，返回空列表 —— 调用方应回落到原有合并逻辑，避免丢内容。
    """
    # 表名会被写入 Milvus 的 title 字段（VARCHAR 1000 字节）。调用方传入的是「父标题」，
    # 而父标题在 merge 阶段会不断吸收子元素，实测曾累积到 1496 字符 ->
    # 直接写库抛 MilvusException code=1100。这里先截断到安全长度。
    if table_name:
        table_name = table_name.strip()[:MAX_TABLE_NAME_CHARS]

    chunks = build_table_chunks(doc.metadata.get('text_as_html', ''), table_name=table_name, max_rows=max_rows)
    if not chunks:
        return []

    out: List[Document] = []
    # [FIX / P1-8] 原先用 `enumerate(chunks, 1)`，但那个序号 `i` **从未被使用**。
    for text in chunks:
        meta = dict(doc.metadata)
        meta.pop('text_as_html', None)  # schema 无此字段；且已还原进正文，无需再存
        meta['category'] = 'content'
        # 保留原有 title（若有），否则用表名兜底 —— Milvus schema 要求非空。
        if table_name and not meta.get('title'):
            meta['title'] = table_name
        out.append(Document(page_content=text, metadata=meta))
    # [FIX / P1-8] 原先这里是 `total = len(chunks)` + `if total > 1: pass` ——
    # 一个**空分支**（注释写「分片信息只用于日志」却没有任何日志语句，`total` 也只被它用到）。
    # 改为真的记一条：分片信息对排查「跨行统计类问题为何不稳定」有用
    # （见 4.8.6 的高频抖动题 与 P0-1 的 n005）。
    if len(chunks) > 1:
        log.info(f"表格「{table_name or '未命名'}」切分为 {len(chunks)} 块（含表头注入与行范围标注）")
    return out
