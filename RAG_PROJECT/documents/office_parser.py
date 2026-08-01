"""Office 文档解析器：docx / xlsx / pptx -> Markdown -> 复用现有 MarkdownParser 管道。

设计（依据见 ROADMAP 1.3）：

    ① markitdown.convert()        转 Markdown（保留标题层级与表格）
    ② MarkdownParser              合并 / 切片 / 字段兜底（与原生 .md 完全相同）
    ③ Document 块                  可直接写入 Milvus，检索层零改动

各格式的实测表现（同一套测试文件）：

    | 格式   | 转出的 Markdown 结构                     | content 占比 |
    |--------|------------------------------------------|--------------|
    | .docx  | 标题层级（h1+h2）+ 表格                  | 100%         |
    | .xlsx  | **每个 sheet 一个 `##` 标题** + Markdown 表格 | 100%      |
    | .pptx  | **每页 slide 一个 `#` 标题** + 正文      | 100%         |

依赖注意（实测踩过的坑）：
  - **docx 需要 `mammoth`**，不是 `python-docx`。
    缺它时 markitdown 抛 `DocxConverter threw MissingDependencyException`。
  - `markitdown -> magika -> onnxruntime` 与 `unstructured-inference` 的
    `onnxruntime>=1.25.0` 要求冲突，需把 onnxruntime 升到 1.25+（实测 1.30.0 可用）。
"""

from __future__ import annotations

import os
import re
from typing import List

from langchain_core.documents import Document

# [FIX 2026-10-11 / P1-5] `MIN_MARKDOWN_CHARS` 已提升为共享常量（值不变），
# 与 pdf_parser 统一 —— 详见 documents/markdown_parser.py 的定义处注释。
from documents.markdown_parser import MIN_MARKDOWN_CHARS, MarkdownParser
from utils.log_utils import log

# markitdown 转 pptx 时会插入 `<!-- Slide number: N -->` 这类 HTML 注释。
# 它们是结构标记而非文档内容，直接入库会污染块的正文，故默认清理。
_HTML_COMMENT_RE = re.compile(r'^\s*<!--.*?-->\s*$', re.MULTILINE)


class OfficeParser:
    """把 docx / xlsx / pptx 解析成与 Markdown 路径完全一致的 Document 块。

    与 MarkdownParser 输出格式一致，因此可写入同一 Milvus 集合、
    检索层的 `category == "content"` 过滤照常生效、标题面包屑照常生成。
    """

    SUPPORTED_EXTENSIONS = ('.docx', '.xlsx', '.pptx')

    # markitdown 实例持有文件类型检测模型，构造有开销，故进程内复用。
    _md_converter = None

    def __init__(self, keep_temp_md: bool = False, strip_html_comments: bool = True,
                 use_pptx_structurer: bool = True, ocr_pptx_images: bool = False):
        """
        Args:
            keep_temp_md: 是否保留中间产出的临时 Markdown（默认 False）。
            strip_html_comments: 是否清理 markitdown 插入的 HTML 注释
                （如 pptx 的 `<!-- Slide number: 1 -->`）。默认清理。
            use_pptx_structurer: [ROADMAP 2.6] pptx 是否走专用结构化解析器
                （逐页提取 + 一页一块，见 documents/pptx_parser.py）。
                **默认 True**：实测 markitdown 路径会把 3 页合并成 1 块，
                且不解析演讲备注 / SmartArt / 图表数据。
            ocr_pptx_images: pptx 中图片是否做 OCR（默认 False，复用 2.3 的 RapidOCR）。
        """
        self.keep_temp_md = keep_temp_md
        self.strip_html_comments = strip_html_comments
        self.use_pptx_structurer = use_pptx_structurer
        self.ocr_pptx_images = ocr_pptx_images
        self._helper = MarkdownParser()

    @classmethod
    def _get_converter(cls):
        if cls._md_converter is None:
            from markitdown import MarkItDown  # 局部导入：未装时不影响其他格式

            cls._md_converter = MarkItDown()
        return cls._md_converter

    @classmethod
    def to_markdown(cls, file_path: str) -> str:
        """把 Office 文档转成 Markdown 文本。"""
        result = cls._get_converter().convert(file_path)
        return result.text_content or ''

    def parse_office_to_documents(self, file_path: str) -> List[Document]:
        """解析单个 Office 文档，返回与 Markdown 路径同构的 Document 块。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 扩展名不支持，或转换结果为空/过少。
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Office 文档不存在: {file_path}')

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'不支持的 Office 格式: {ext}（支持 {self.SUPPORTED_EXTENSIONS}）'
            )

        basename = os.path.basename(file_path)

        # [ROADMAP 2.6] pptx 优先走专用结构化解析器：逐页提取 + 一页一块。
        # 实测对比（3 页样本）：markitdown 路径 -> 1 块；专用解析器 -> 3 块（各带页码）。
        # 且专用路径能提取演讲备注 / SmartArt / 图表数据 / 组合形状内文字，这些是
        # markitdown 完全不解析的部分。
        if ext == '.pptx' and self.use_pptx_structurer:
            try:
                from documents.pptx_parser import PPTParser

                return PPTParser(
                    keep_temp_md=self.keep_temp_md,
                    ocr_images=self.ocr_pptx_images,
                ).parse_pptx_to_documents(file_path)
            except Exception as e:  # noqa: BLE001
                # 显式回落并记 warning，不静默降级（见 ROADMAP 1.3.6）
                log.warning(
                    f'PPTX 结构化解析失败，回落到 markitdown 路径: '
                    f'{type(e).__name__}: {str(e)[:160]}'
                )

        md_text = self.to_markdown(file_path)

        if self.strip_html_comments:
            md_text = _HTML_COMMENT_RE.sub('', md_text)

        stripped = md_text.strip()
        if len(stripped) < MIN_MARKDOWN_CHARS:
            # 显式失败，不静默返回空列表（见 ROADMAP 1.3.6）
            raise ValueError(
                f'Office 文档转 Markdown 后内容过少（{len(stripped)} 字符 < {MIN_MARKDOWN_CHARS}）: {basename}。'
                f'请确认文件非空、且对应格式的转换依赖已安装（docx 需 mammoth）。'
            )

        log.info(f'Office 转 Markdown: {basename} -> {len(md_text)} 字符')

        docs = self._helper.parse_markdown_to_documents_from_text(
            md_text,
            source_name=f'{basename}.md',
            keep_temp=self.keep_temp_md,
        )
        log.info(f'Office 解析完成: {basename} -> {len(docs)} 个块')
        return docs


if __name__ == '__main__':
    from utils.env_utils import DATAS_DIR

    # 演示：解析 datas/office 下的所有文档，保留中间 md 便于核对
    office_dir = os.path.join(DATAS_DIR, 'office')
    if not os.path.isdir(office_dir):
        print(f'目录不存在：{office_dir}（可把 docx/xlsx/pptx 放入后重试）')
    else:
        files = [f for f in sorted(os.listdir(office_dir))
                 if f.lower().endswith(OfficeParser.SUPPORTED_EXTENSIONS)]
        if not files:
            print(f'未找到 Office 文档：{office_dir}')
        parser = OfficeParser(keep_temp_md=True)
        for f in files:
            try:
                docs = parser.parse_office_to_documents(os.path.join(office_dir, f))
                print(f'{f} -> {len(docs)} 个块')
                for d in docs[:2]:
                    print(f"    title={d.metadata.get('title')!r} category={d.metadata.get('category')!r}")
            except Exception as e:  # noqa: BLE001
                print(f'{f} 解析失败: {type(e).__name__}: {e}')
