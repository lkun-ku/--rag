"""PPTX 专用解析器（ROADMAP 2.6）：逐页结构化提取 -> 一页一块。

## 为什么需要它

`markitdown` 对 PPTX 的产出是「同一层级的 `#` 标题 + 正文」，经现有 Markdown 管道后
会被合并 —— 实测 3 页幻灯片变成 1 个块。而 **PPT 的天然语义单元是「一页」**：
一页通常只讲一个观点，把多页平均成一个向量后，与任何具体问题都不够近。

实测对照（3 页样本）：

| 路径 | 结果 |
|---|---|
| markitdown（原路径） | **1 块 / 92 字符** —— 三页内容混在一起 |
| 本解析器 | **3 块 / 32+42+34 字符** —— 一页一块，各带页码前缀 |

同时 `markitdown` **不解析** PPTX 的这些部分（存在则必丢）：

- **演讲备注**（`notesSlides/`）—— 指南指出「备注往往才是真正的叙事内容」
- **SmartArt**（`diagrams/data*.xml`）—— `python-pptx` 也不支持，需自己解析
- **图表数据**（`charts/`、`embeddings/*.xlsx`）—— 只剩图表标题
- **组合形状内部文字**（`grpSp` 需递归展开）
- **表格结构**（会退化成乱序文字）

## 与现有管道的关系

产出仍是 Markdown，因此照常走 `MarkdownParser.parse_markdown_to_documents_from_text`
—— 检索层与工作流层**零改动**。

页码前缀 `## 幻灯片 N：标题` 有两个作用：
1. 提供页码语义（满足 ROADMAP 决策 5 的引用粒度要求）
2. 让 `_merge_short_chunks` 识别为结构块而**豁免合并**（见 `STRUCT_PREFIXES`）
   —— 否则每页仅 30 余字符会被合并回一个块，方案就白做了
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from utils.log_utils import log

# SmartArt 的 diagram data 部件的命名空间
_NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'dgm': 'http://schemas.openxmlformats.org/drawingml/2006/diagram',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
_DGM_URI = 'http://schemas.openxmlformats.org/drawingml/2006/diagram'

# 单页字符数低于该值且该页含图片/SmartArt 时，说明内容主要在「图里」，
# 属于标记该页为「提取不充分」的阈值（暂只记日志，不触发额外处理）
SPARSE_PAGE_CHARS = 80


class PPTParser:
    """把 PPTX 解析成与 Markdown 路径完全一致的 Document 块（一页一块）。"""

    def __init__(self, keep_temp_md: bool = False, ocr_images: bool = False,
                 ocr_dpi: int = 200):
        """
        Args:
            keep_temp_md: 保留中间 Markdown 供人工核对（默认 False）。
            ocr_images: 是否对页面内的图片做 OCR。
                **默认 False**：复用 2.3 已装的 RapidOCR，但默认关闭以免拖慢解析；
                当前语料无图片型 PPT。
            ocr_dpi: 图片 OCR 的渲染分辨率。
        """
        self.keep_temp_md = keep_temp_md
        self.ocr_images = ocr_images
        self.ocr_dpi = ocr_dpi

    # ------------------------------------------------------------------ 形状处理
    @staticmethod
    def _iter_shapes(shapes):
        """递归展开组合形状（`grpSp` 可嵌套）。

        为什么必须递归：python-pptx 的 `shapes` 迭代**不会**展开组合内部，
        组内文字会整块丢失。
        """
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        for shp in shapes:
            if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from PPTParser._iter_shapes(shp.shapes)
            else:
                yield shp

    @staticmethod
    def _sort_by_position(shapes):
        """按视觉顺序（先上后下、先左后右）排序。

        为什么需要：**阅读顺序 != XML 顺序**，多栏页会串读。
        用 1e5 EMU（≈0.1 英寸）做行容差，避免轻微错位导致跨行误排。
        """
        def key(s):
            top = s.top if s.top is not None else 10 ** 12
            left = s.left if s.left is not None else 0
            return (round(top / 100000), left)

        return sorted(shapes, key=key)

    @staticmethod
    def _shape_role(shp) -> str:
        """用占位符类型判断角色（TITLE / BODY / TEXT）。"""
        try:
            pf = shp.placeholder_format
            if pf is None or pf.type is None:
                return 'TEXT'
            return str(pf.type)
        except (AttributeError, ValueError, KeyError):
            return 'TEXT'

    # ------------------------------------------------------------------ 元素转换
    @staticmethod
    def _table_to_markdown(shape) -> str:
        """PPT 原生表格 -> Markdown（逐格读，避免合并单元格错位）。"""
        try:
            tbl = shape.table
        except Exception:  # noqa: BLE001
            return ''
        rows = []
        for r in tbl.rows:
            rows.append([c.text.strip().replace('\n', ' ') for c in r.cells])
        if not rows:
            return ''
        width = max(len(r) for r in rows)
        rows = [r + [''] * (width - len(r)) for r in rows]
        head = '| ' + ' | '.join(rows[0]) + ' |'
        sep = '|' + '|'.join(['---'] * width) + '|'
        body = ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
        return '\n'.join([head, sep] + body)

    @staticmethod
    def _chart_to_markdown(shape) -> str:
        """图表 -> Markdown 表格（取类别与各系列数值）。"""
        try:
            chart = shape.chart
            plot = chart.plots[0]
            cats = [str(c) for c in plot.categories]
            series = list(chart.series)
        except Exception:  # noqa: BLE001
            return ''
        if not cats and not series:
            return ''
        header = ' | '.join(['类别'] + [(s.name or ('系列%d' % (i + 1))) for i, s in enumerate(series)])
        lines = [header, ' | '.join(['---'] * (len(series) + 1))]
        for i, cat in enumerate(cats):
            vals = []
            for s in series:
                try:
                    v = s.values[i]
                except Exception:  # noqa: BLE001
                    v = ''
                vals.append('' if v is None else str(v))
            lines.append(' | '.join([cat] + vals))
        return '\n'.join(lines)

    @staticmethod
    def _smartart_texts(slide) -> List[str]:
        """抽取本页 SmartArt 的文字。

        python-pptx **完全不支持** SmartArt：文字存在 `ppt/diagrams/data*.xml`，
        需经形状的 `graphicData` -> `dgm:relIds` -> 关系表定位到部件。
        """
        from lxml import etree

        out: List[str] = []
        try:
            shapes = list(slide.shapes)
        except Exception:  # noqa: BLE001
            return out
        for shp in shapes:
            el = getattr(shp, '_element', None)
            if el is None:
                continue
            gd = el.find('.//a:graphicData', _NS)
            if gd is None or _DGM_URI not in (gd.get('uri') or ''):
                continue
            rel_ids = gd.find('dgm:relIds', _NS)
            if rel_ids is None:
                continue
            rid = rel_ids.get('{%s}dm' % _NS['r'])       # r:dm 指向 diagram data 部件
            if not rid:
                continue
            try:
                part = slide.part.rels[rid].target_part
                root = etree.fromstring(part.blob)
            except Exception:  # noqa: BLE001
                continue
            out.extend(t.text for t in root.findall('.//a:t', _NS) if t.text)
        return out

    # ------------------------------------------------------------------ 单页提取
    def _extract_slide(self, slide, index: int) -> dict:
        """提取单页：标题 / 正文 / 表格 / 图表 / SmartArt / 备注。"""
        title = ''
        blocks: List[str] = []
        tables: List[str] = []
        charts: List[str] = []
        has_image = False

        try:
            shapes = self._sort_by_position(list(self._iter_shapes(slide.shapes)))
        except Exception:  # noqa: BLE001
            shapes = []

        for shp in shapes:
            role = self._shape_role(shp)

            if getattr(shp, 'has_table', False):
                md = self._table_to_markdown(shp)
                if md:
                    tables.append(md)
                continue

            if getattr(shp, 'has_chart', False):
                md = self._chart_to_markdown(shp)
                if md:
                    charts.append(md)
                continue

            try:
                from pptx.enum.shapes import MSO_SHAPE_TYPE
                if shp.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    has_image = True
                    if self.ocr_images:
                        t = self._ocr_picture(shp)
                        if t:
                            blocks.append('图片文字：' + t)
                    continue
            except Exception:  # noqa: BLE001
                pass

            if not getattr(shp, 'has_text_frame', False):
                continue
            # 按段落取，而不是按 run 拼 —— 否则自动换行会把一句话拆成多行
            text = '\n'.join(p.text for p in shp.text_frame.paragraphs).strip()
            if not text:
                continue

            if 'TITLE' in role and not title:
                title = text.split('\n')[0]
            else:
                blocks.append(text)

        smartart = self._smartart_texts(slide)

        notes = ''
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:  # noqa: BLE001
            notes = ''

        return {
            'index': index,
            'title': title,
            'blocks': blocks,
            'tables': tables,
            'charts': charts,
            'smartart': smartart,
            'notes': notes,
            'has_image': has_image,
        }

    def _ocr_picture(self, shape) -> str:
        """对页面内的图片做 OCR（复用 2.3 的 RapidOCR）。"""
        try:
            import io

            import numpy as np
            from PIL import Image
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return ''
        try:
            img = Image.open(io.BytesIO(shape.image.blob)).convert('RGB')
            arr = np.asarray(img)
            raw = RapidOCR()(arr)
            result = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
            texts = []
            for item in (result or []):
                t = str(item[1]).strip() if len(item) > 1 else ''
                score = float(item[2]) if len(item) > 2 else 1.0
                if t and score >= 0.5:
                    texts.append(t)
            return ' '.join(texts)
        except Exception:  # noqa: BLE001
            return ''

    # ------------------------------------------------------------------ Markdown 组装
    def to_markdown(self, file_path: str) -> str:
        """逐页提取并组装成 Markdown。每页一个 `## 幻灯片 N：标题`。"""
        from pptx import Presentation

        prs = Presentation(file_path)
        deck = os.path.basename(file_path)

        pages = []
        for i, slide in enumerate(prs.slides, start=1):
            pages.append(self._extract_slide(slide, i))

        # 母版污染过滤：跨页高频出现的短文本（页脚 / 页码 / logo 文字）
        noise = self._detect_master_noise(pages)

        parts: List[str] = []
        for p in pages:
            heading = p['title'] or '（无标题）'
            body: List[str] = []
            for b in p['blocks']:
                if b.strip() in noise:
                    continue
                body.append(b)
            for t in p['tables']:
                body.append('表格：\n' + t)
            for c in p['charts']:
                body.append('图表数据：\n' + c)
            if p['smartart']:
                body.append('图示文字：' + ' / '.join(p['smartart']))
            if p['notes']:
                body.append('演讲备注：' + p['notes'])

            text = '\n'.join(['## 幻灯片 %d：%s' % (p['index'], heading)] + body)
            # 纯装饰页 / 空白页：标题与正文都没有实质内容则跳过
            if len(text) < 30 and not p['title']:
                continue
            if len(text) < SPARSE_PAGE_CHARS and (p['has_image'] or p['smartart']):
                log.warning(
                    f'[PPTX] 第 {p["index"]} 页提取内容偏少（{len(text)} 字符）'
                    f'且含图片/SmartArt，可能有内容未被提取: {deck}'
                )
            parts.append(text)

        return '\n'.join(parts) + '\n'

    @staticmethod
    def _detect_master_noise(pages: List[dict]) -> set:
        """识别跨页重复出现的短文本（母版/页脚污染）。

        规则：出现在 **超过半数页** 且长度 < 30 的文本，视为版式噪声。
        这是「不依赖母版解析」的实用近似 —— 页脚/页码/公司名都有这个特征。
        """
        if len(pages) < 2:
            return set()
        from collections import Counter

        cnt = Counter()
        for p in pages:
            for b in set(p['blocks']):
                if len(b) < 30:
                    cnt[b] += 1
        threshold = max(2, len(pages) // 2 + 1)
        return {t for t, c in cnt.items() if c >= threshold}

    # ------------------------------------------------------------------ 主入口
    def parse_pptx_to_documents(self, file_path: str) -> List[Document]:
        """解析 PPTX 为 Document 块（一页一块）。"""
        from documents.markdown_parser import MarkdownParser

        basename = os.path.basename(file_path)
        log.info(f'PPTX 解析: {basename}')

        try:
            md_text = self.to_markdown(file_path)
        except ImportError as e:
            # 显式回落，不静默降级（见 ROADMAP 1.3.6）
            raise ValueError(
                f'PPTX 结构化解析需要 python-pptx：{e}。'
                f'若确实无法安装，请在 OfficeParser 中改走 markitdown 路径。'
            ) from e

        stripped = md_text.strip()
        if not stripped:
            raise ValueError(
                f'PPTX 解析后内容为空（可能全部是图片或装饰页）: {basename}。'
                f'请检查文件是否有效，或改用 ocr_images=True 尝试 OCR 图片文字。'
            )

        helper = MarkdownParser()
        docs = helper.parse_markdown_to_documents_from_text(
            md_text, source_name=basename + '.md', keep_temp=self.keep_temp_md
        )
        content = [d for d in docs if d.metadata.get('category') == 'content']
        log.info(f'PPTX 解析完成: {basename} -> {len(content)} 个内容块（一页一块）')
        return docs
