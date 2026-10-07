import os
import re
import shutil
import tempfile
from typing import List
# [4.48] 走 make_splitter 封装：避免实验包 sunset 后 import 直接崩溃。

# [UPGRADE ] 改用本地 bge_embedding：分块语义须与入库向量同源（路线图坑①）
from rag_kb.llm_models.embeddings_model import get_embedding
from rag_kb.utils.env_utils import MD_DIR
from rag_kb.tools.semantic_splitter import make_splitter
from rag_kb.utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# [UPGRADE  / ROADMAP 2.4] 表格结构保护与表格感知切分
from rag_kb.documents.table_handler import (
    DEFAULT_MAX_ROWS,       # 表格行数阈值（默认 30）
    has_restorable_table,   # 能否从 text_as_html 恢复结构
    is_flat_twin,           # 扁平文本是否已被结构化块完整覆盖
    restore_table_documents,
)

# 转换后最少字符数（低于视为失败）；取 50：误拒小文件的代价更高。
MIN_MARKDOWN_CHARS = 50


class UTF8MarkdownLoader(UnstructuredMarkdownLoader):
    """强制 UTF-8 读取的 Markdown 加载器，绕开 unstructured 的编码猜测。

    partition_md 未指定编码时用 charset_normalizer 猜，曾把英文论文误判成 hp_roman8
    并解出乱码；而它调 read_txt_file 时没有透传 encoding，只能自己读文本再传 text=。
    """

    def _get_elements(self):
        from unstructured.partition.md import partition_md

        # errors="strict"：读不出来就报错，不静默替换字符。
        with open(self.file_path, encoding='utf-8') as f:
            text = f.read()

        # partition_md 不接受 encoding 参数（它只透传给 partition_html），需剔除
        kwargs = {k: v for k, v in self.unstructured_kwargs.items() if k != 'encoding'}
        return partition_md(
            text=text,
            metadata_filename=os.path.basename(str(self.file_path)),
            **kwargs,
        )


class MarkdownParser:
    """
    专门负责markdown文件的解析和切片
    """

    # schema「非空且无默认值」字段；unstructured 未必产出，缺则报 DataNotMatchException。
    REQUIRED_FIELD_DEFAULTS = {
        'category': '',
        'source': '',
        'filename': '',
        'filetype': '',
        'title': '',
        'category_depth': 0,
    }

    # schema VARCHAR 上限（按字节），超长会让整批写入失败（code=1100）。
    FIELD_MAX_BYTES = {
        'category': 1000,
        'source': 1000,
        'filename': 1000,
        'filetype': 1000,
        'title': 1000,
    }

    # 单块正文字节上限 5800（schema text 为 6000），切片拦不住时兜底。
    MAX_CONTENT_BYTES = 5800  # 留 200 字节余量给 metadata 拼接与换行

    # 版式噪声（每页重复或纯装饰），不作为独立内容块入库。
    NOISE_CATEGORIES = {'Header', 'Footer', 'PageNumber'}

    # frontmatter 按内容识别并排除（与正文冗余，会挤占检索候选）。
    _FRONTMATTER_RE = re.compile(r'^\s*id\s*:\s*\S+\.(?:md|mdx)\b', re.IGNORECASE)

    # [2.6] 结构块（"表格：" / "幻灯片 "）是独立语义单元，不参与超短块合并。
    STRUCT_PREFIXES = ('\u8868\u683c\uff1a', '\u5e7b\u706f\u7247 ')  # "表格：" / "幻灯片 "

    # 超短块合并阈值 50：碎片块向量更集中、易挤掉正常候选（实测 28/36 字符的块被挤出 top-6）。
    MIN_CHUNK_CHARS = 50

    # 超过此字符数才走语义切块（SemanticChunker）。提为类常量便于调整与测试。
    MAX_SEMANTIC_CHUNK_CHARS = 5000

    def __init__(self, table_max_rows: int = DEFAULT_MAX_ROWS):
        """
        Args:
            table_max_rows: [ROADMAP 2.4] 表格行数阈值。超过则按行切分，
                且每块重复完整表头（表头注入）。默认 30 行。
        """
        # [4.48] 可用时仍是 SemanticChunker(percentile)，不可用时降级 + WARNING。
        # 惰性：真正构造解析器时才加载嵌入模型（import 期不加载）
        self.text_splitter = make_splitter(get_embedding())
        self.table_max_rows = table_max_rows

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        """主入口：文件路径 -> Document 块列表（解析/合并/切块/短块/字节/字段 六步）。"""
        documents = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)

        log.info(f'文件合并后的长度: {len(merged_documents)}')

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度: {len(chunk_documents)}')

        # [2.5] 合并超短块放在切片之后、字段兜底之前。
        chunk_documents = self._merge_short_chunks(chunk_documents)
        log.info(f'合并超短块后的长度: {len(chunk_documents)}')

        # [FIX] 字节上限兜底（见 MAX_CONTENT_BYTES 的说明）。
        n_before = len(chunk_documents)
        chunk_documents = self._enforce_byte_limit(chunk_documents)
        if len(chunk_documents) != n_before:
            log.warning(f'超长块切分: {n_before} -> {len(chunk_documents)} （按 {self.MAX_CONTENT_BYTES} 字节上限）')

        # 补齐 Milvus schema 必需但 unstructured 未必产出的字段（见 _ensure_required_fields）
        chunk_documents = self._ensure_required_fields(chunk_documents)
        return chunk_documents

    def parse_markdown(self, md_file: str) -> List[Document]:
        """① 解析：用 UTF8MarkdownLoader 读成元素列表（尚未切块）。"""
        # 用 UTF8MarkdownLoader：原 loader 会猜编码（曾把英文论文误判为 hp_roman8）。
        loader = UTF8MarkdownLoader(file_path=md_file, mode='elements', strategy='fast')
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        """把「标题 + 其下元素」合并成带面包屑的内容块。

        两条硬约束（契约见 tests/test_markdown_breadcrumb.py）：
        1. 面包屑只由**标题链**构成，绝不用 page_content（否则父级正文会被复制进每个子块）；
        2. 归属不假设 parent_id 一定存在 —— 缺失时按 category_depth 回退到最近的上层标题，
           否则代码块会退化成 title=文件名的孤立块。
        """
        merged_data = []
        extra_table_docs: List[Document] = []  # 额外产出的表格块（最后追加，不打乱既有顺序）
        parent_dict = {}  # 是一个字典，保存所有的父document， key为当前父document的ID
        # 标题链与正文分开存：拼在一起会让 title 无限增长且语义错乱。
        crumb: dict = {}
        # [(category_depth, element_id)]：parent_id 缺失时用来判定「当前所属章节」
        stack: List[tuple] = []
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            # [2.4] 表格产出独立结构化块；扁平文本已被覆盖时才跳过父块累积。
            if has_restorable_table(document):
                # 表名用完整面包屑（只取最后一段会丢掉章节信息）。
                table_anchor = parent_id if parent_id in crumb else (stack[-1][1] if stack else None)
                table_name = crumb.get(table_anchor, '')
                table_chunks = restore_table_documents(document, table_name=table_name, max_rows=self.table_max_rows)
                if table_chunks:
                    extra_table_docs.extend(table_chunks)
                    # restore_table_documents 返回 List[Document]，is_flat_twin 要文本列表。
                    if is_flat_twin(document.page_content, [d.page_content for d in table_chunks]):
                        continue

            # 版式噪声：挂在标题下的并入父块，否则丢弃。
            if category in self.NOISE_CATEGORIES and not parent_id:
                continue

            if category == 'Title':
                heading = document.page_content.strip()
                depth = metadata.get('category_depth')
                depth = depth if isinstance(depth, int) else 0
                # 归属优先用 parent_id；本语料常缺失，需回退到标题栈。
                anchor = parent_id if parent_id in crumb else None
                if anchor is None:
                    # 在标题栈里找「最近的上层标题」，不能只看栈顶。
                    anchor = next((eid for d, eid in reversed(stack) if d < depth), None)
                parent_crumb = crumb.get(anchor, '')
                # 面包屑只拼接**标题**，绝不拼接父块的正文（见 crumb 的说明）
                crumb[element_id] = f'{parent_crumb} -> {heading}' if parent_crumb else heading
                # title 存面包屑：表格块要复用最后一段当表名，检索侧也靠它区分同名小节
                document.metadata['title'] = crumb[element_id]
                document.page_content = crumb[element_id]
                parent_dict[element_id] = document
                # 维护「当前标题栈」：同级或更浅的旧标题出栈
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, element_id))
                continue

            # 正文归属：显式 parent_id -> 当前标题。
            anchor = parent_id if parent_id in parent_dict else (stack[-1][1] if stack else None)
            if anchor is not None:
                parent = parent_dict[anchor]
                parent.page_content = parent.page_content + ' ' + document.page_content
                parent.metadata['category'] = 'content'
                continue

            # 无标题可归属时（文档开头/整篇无标题）独立成块。
            if self._FRONTMATTER_RE.match(document.page_content):
                continue
            document.metadata.setdefault('title', document.metadata.get('filename', ''))
            # 必须直接赋值：category 键已存在，setdefault 不生效（否则检索层过滤时漏块）。
            document.metadata['category'] = 'content'
            merged_data.append(document)

        # 处理字典
        # 只输出「吸收了正文」的标题块：page_content 仍等于面包屑，说明该章节下面只有子标题、
        # 自己没有任何正文。这种块既过不了检索层的 category=='content' 过滤，也没有信息量，
        # 纯占位（实测 CLI 文档 13/114 块是这种空壳，形如 "... -> show replicas -> Options"）。
        for doc in parent_dict.values():
            if doc.page_content.strip() and doc.page_content.strip() != str(doc.metadata.get('title', '')).strip():
                merged_data.append(doc)

        # [2.4] 表格块最后追加，不打乱既有顺序。
        merged_data.extend(extra_table_docs)

        return merged_data

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        """③ 语义切块：仅对超过 5000 字符的块调 split_documents，其余原样通过。"""
        new_docs = []
        for d in datas:
            if len(d.page_content) > self.MAX_SEMANTIC_CHUNK_CHARS:  # 超过阈值才语义切
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs

    @staticmethod
    def _merge_short_chunks(datas: List[Document], min_chars: int = MIN_CHUNK_CHARS) -> List[Document]:
        """把过短的块并入同一来源文件的前一块（只往前找，找不到则原样保留）。

        结构块（表格/幻灯片）既不参与合并、也不作为目标；只改 page_content。
        """
        if not datas:
            return datas

        prefixes = MarkdownParser.STRUCT_PREFIXES

        # 按来源文件分组（保持原顺序），未标注 filename 的块单独成组不参与合并
        out: List[Document] = []
        i = 0
        total = len(datas)
        while i < total:
            doc = datas[i]
            fname = doc.metadata.get("filename", "")
            text = doc.page_content
            # [2.6] 结构块（表格/幻灯片）是独立语义单元，不参与短块合并。
            is_structural = text.startswith(prefixes)

            # 源块与目标块都必须是 content，否则会并入 Title、内容被过滤。
            if is_structural or not fname or len(text) >= min_chars or doc.metadata.get('category') != 'content':
                out.append(doc)
                i += 1
                continue

            # 超短块并入前一个同文件的 content 块（只往前找，找不到则保留）。
            def _joinable(d):
                return (
                    d is not None
                    and d.metadata.get("filename") == fname
                    and d.metadata.get("category") == "content"
                    # 结构块也不作为合并目标（否则表名/页码会混入无关文本）。
                    and not d.page_content.startswith(prefixes)
                )

            prev_doc = out[-1] if out and _joinable(out[-1]) else None
            if prev_doc is None:
                # 没有可安全并入的对象（如该文件的第一个块）—— 原样保留，绝不丢内容
                out.append(doc)
                i += 1
                continue

            prev_doc.page_content = (prev_doc.page_content + " " + text).strip()
            log.info(f'[合并超短块] {fname}: {len(text)} 字符的块并入前一块 (合并后 {len(prev_doc.page_content)} 字符)')
            i += 1
        return out

    @staticmethod
    def _enforce_byte_limit(datas: List[Document]) -> List[Document]:
        """把超字节上限的块切开（切出的片段继承原 metadata）。"""
        out: List[Document] = []
        for d in datas:
            if len(d.page_content.encode('utf-8')) <= MarkdownParser.MAX_CONTENT_BYTES:
                out.append(d)
                continue
            for piece in MarkdownParser._split_by_bytes(d.page_content, MarkdownParser.MAX_CONTENT_BYTES):
                out.append(Document(page_content=piece, metadata=dict(d.metadata)))
        return out

    @staticmethod
    def _ensure_required_fields(datas: List[Document]) -> List[Document]:
        """为缺少 schema 必需字段的文档补默认值，并截断超出 Milvus 上限的字段。"""
        for d in datas:
            for key, default in MarkdownParser.REQUIRED_FIELD_DEFAULTS.items():
                d.metadata.setdefault(key, default)
            for key, limit in MarkdownParser.FIELD_MAX_BYTES.items():
                v = d.metadata.get(key)
                if isinstance(v, str) and len(v.encode('utf-8')) > limit:
                    d.metadata[key] = MarkdownParser._truncate_utf8(v, limit)
        return datas

    def parse_markdown_to_documents_from_text(self, md_text: str, source_name: str = 'converted.md', keep_temp: bool = False) -> List[Document]:
        """从 Markdown 文本解析成 Document 块（供 PDF/Office/HTML/TXT 复用）。

        转换产物走与原生 .md 相同的合并、切片、字段兜底流程，检索层零改动。
        落临时文件而非 BytesIO：loader 从文件名推导 filename / source（schema 必需）。

        Args:
            md_text: Markdown 全文。
            source_name: 临时文件名，同时决定入库 filename。
            keep_temp: True 时保留临时文件供核对（默认删除）。
        """
        # 空内容显式失败，否则「解析失败」会伪装成「文档为空」。
        if not md_text or not md_text.strip():
            raise ValueError(f'Markdown 内容为空，转换环节可能失败: {source_name}')

        tmp_dir = tempfile.mkdtemp(prefix='kb_convert_')
        tmp_path = os.path.join(tmp_dir, source_name)
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            return self.parse_markdown_to_documents(tmp_path)
        finally:
            if keep_temp:
                log.info(f'临时 Markdown 已保留（keep_temp=True）: {tmp_path}')
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _split_by_bytes(text: str, limit: int) -> List[str]:
        """按 UTF-8 字节上限切分，且**不切坏多字节字符**。

        为什么不在换行处优雅切分：进到这里的是"切片器都切不动"的异常块
        （语义高度相似的长表格 / 长数字列表），此时保证**字节合规**优先于**可读性**。
        """
        raw = text.encode('utf-8')
        if len(raw) <= limit:
            return [text]
        out, start = [], 0
        while start < len(raw):
            end = min(start + limit, len(raw))
            # 回退到字符边界（UTF-8 续字节的高两位是 10）
            while end < len(raw) and (raw[end] & 0xC0) == 0x80:
                end -= 1
            if end <= start:  # 单个字符就超限的极端情况
                end = min(start + limit, len(raw))
            out.append(raw[start:end].decode('utf-8', errors='ignore'))
            start = end
        return out

    @staticmethod
    def _truncate_utf8(value, max_bytes: int) -> str:
        """按 UTF-8 字节数截断，保证不切坏多字节字符。"""
        s = str(value)
        raw = s.encode('utf-8')
        if len(raw) <= max_bytes:
            return s
        return raw[:max_bytes].decode('utf-8', errors='ignore')


if __name__ == '__main__':
    # 用法：python markdown_parser.py [路径]；不传则取 MD_DIR 下第一个 .md。
    import glob
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob(os.path.join(MD_DIR, '*.md')))
        if not candidates:
            raise SystemExit('MD_DIR 下没有 .md 文件，请显式传路径：python markdown_parser.py <file.md>')
        file_path = candidates[0]
        print(f'未指定路径，使用: {file_path}')

    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    print(f'共 {len(docs)} 块\n')
    for item in docs:
        print(f"元数据: {item.metadata}")
        print(f"标题: {item.metadata.get('title')}")
        print(f"内容: {item.page_content[:300]}\n")
        print("------" * 10)
