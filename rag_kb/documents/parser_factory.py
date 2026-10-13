"""解析器分派：parser 类型 -> (实例, 单文件解析函数)。

从 `write_milvus.py` 拆出。`parser='auto'` 时按扩展名逐文件分派，
解析器实例按类型缓存，避免逐文件重复构造。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from rag_kb.documents.markdown_parser import MarkdownParser
from rag_kb.documents.office_parser import OfficeParser
from rag_kb.documents.pdf_parser import PDFParser

if TYPE_CHECKING:  # 仅类型标注
    from langchain_core.documents import Document

ParseFn = Callable[[str], List['Document']]


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔环境变量；只认 1/true/yes/on（`off`、`no` 不会被误判为真）。"""
    raw = (os.getenv(name) or '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


# 扩展名 -> parser 类型
_EXT_TO_PARSER = {
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.pdf': 'pdf',
    '.docx': 'office',
    '.xlsx': 'office',
    '.pptx': 'office',
    '.html': 'html',
    '.htm': 'html',
    '.txt': 'text',
}

# parser 实例缓存：auto 分派会逐文件查表，实例构造留在缓存里避免反复创建
_PARSER_CACHE: Dict[str, Tuple[Any, ParseFn]] = {}


class _AutoDispatcher:
    """auto 模式的占位 parser：只用于让 `parser in known` 类检查通过。"""

    def __repr__(self) -> str:  # pragma: no cover
        return 'AutoParser(按扩展名分派)'


def _auto_parse(path: str) -> List[Any]:
    """按扩展名选择解析器并解析单个文件。"""
    ext = os.path.splitext(path)[1].lower()
    parser_type = _EXT_TO_PARSER.get(ext)
    if parser_type is None:
        raise ValueError(f'不支持的文件类型: {ext}（{os.path.basename(path)}）')
    if parser_type not in _PARSER_CACHE:
        _PARSER_CACHE[parser_type] = build_parser(parser_type)
    _, fn = _PARSER_CACHE[parser_type]
    return fn(path)


def build_parser(parser_type: str) -> Tuple[Any, ParseFn]:
    """返回 (parser 实例, 单文件解析函数)；解析函数统一产出同构的 Document 块。"""
    if parser_type == 'auto':
        # 自动源允许混放 md / pdf / office / html / txt，按扩展名逐文件分派。
        return _AutoDispatcher(), _auto_parse
    # 每个分支用独立变量名：不同解析器是无关类型，复用同一个变量会让类型
    # 被首个分支固定住（mypy 曾据此报「MarkdownParser 没有 parse_pdf_to_documents」）。
    if parser_type == 'markdown':
        md_parser = MarkdownParser()
        return md_parser, md_parser.parse_markdown_to_documents
    if parser_type == 'pdf':
        # OCR 开关走环境变量（子进程时代无法传参；默认值与历史一致）
        pdf_parser = PDFParser(ocr_fallback=_env_flag('PDF_OCR_FALLBACK'), use_ocr=_env_flag('PDF_USE_OCR', True))
        return pdf_parser, pdf_parser.parse_pdf_to_documents
    if parser_type == 'office':
        office_parser = OfficeParser()
        return office_parser, office_parser.parse_office_to_documents
    if parser_type == 'text':
        # TXT 先补文件名作一级标题，再复用 Markdown 的合并/切片
        from rag_kb.documents.plain_converters import TextDocParser

        text_parser = TextDocParser()
        return text_parser, text_parser.parse
    if parser_type == 'html':
        # HTML -> Markdown -> MarkdownParser（与 Office 同链路）
        from rag_kb.documents.plain_converters import HTMLDocParser

        html_parser = HTMLDocParser()
        return html_parser, html_parser.parse
    raise ValueError(f'未知的 parser 类型: {parser_type!r}（可选: markdown / pdf / office / text / html）')
