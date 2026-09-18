import os
import re
import shutil
import tempfile
from typing import List
from langchain_experimental.text_splitter import SemanticChunker

# [UPGRADE ] 改用本地 bge_embedding：分块语义须与入库向量同源（路线图坑①）
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MD_DIR
from utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# [UPGRADE  / ROADMAP 2.4] 表格结构保护与表格感知切分
from documents.table_handler import (
    DEFAULT_MAX_ROWS,
    has_restorable_table,
    is_flat_twin,
    restore_table_documents,
)

# 转换后最少应有多少字符。低于此值几乎可以断定「转换失败」或「文件为空」，
# 此时**显式报错**（不静默返回空内容 —— 那会伪装成「文档本来就没内容」，见 ROADMAP 1.3.6）。
#
# [FIX / P1-5] 原先 `pdf_parser` 与 `office_parser` **各定义了一份**
# 同名同义的常量，但取值不同（100 vs 50）—— 于是**同类文件会得出不同结论**：
# 一份转换后只剩 60 字符的文档，走 PDF 路径会报错、走 Office 路径却放行。
#
# 统一取 **50**（较小的那个）的理由：
#   * 报错的代价是「入库中断 + 需人工处理」，**误报（正常小文件被拒）代价更高**；
#     而放行的代价只是「内容偏少」，还有后续体检与告警可兜底。
#   * PDF 的「扫描件」判定有**独立机制**（`MIN_TEXT_LAYER_CHARS` + `is_scanned`），
#     不需要靠抬高这个阈值来兜底。
MIN_MARKDOWN_CHARS = 50


class UTF8MarkdownLoader(UnstructuredMarkdownLoader):
    """强制以 UTF-8 读取的 Markdown 加载器，修复 unstructured 的编码误判。

    ## 问题（本项目实测）

    `unstructured` 的 `partition_md` 在未显式指定编码时，会调用
    `charset_normalizer.detect()` **自动猜测**文件编码。实测它把 `pymupdf4llm`
    产出的英文论文 Markdown（46 KB）误判成 **`hp_roman8`**（置信度 0.908，
    高于其内部阈值 0.8），于是整份文档按错误编码解码：

        en dash U+2013 (UTF-8: E2 80 93)  ->  U+00E3 U+0080 U+0093

    实测数据：正确的文档含 25 个 en dash；走原路径后 en dash 归零，
    整篇出现 36 个 `U+00E3`。原生 `.md`（datas/md）因内容较简单会被正确
    识别为 utf-8/ascii，只有 PDF / Office 转换产物会中招。

    ## 为什么不能简单传参修复

    `UnstructuredMarkdownLoader` 只暴露 `file_path / mode / unstructured_kwargs`；
    而 `partition_md` 内部调用 `read_txt_file(filename=...)` 时**根本没有把
    encoding 透传下去**，`**kwargs` 是流向 `partition_html` 的。因此无法通过参数指定。

    ## 修复方式

    自己以 UTF-8 读出文本，改走 `partition_md(text=...)` 分支 ——
    **该分支不解码字节，天然绕开编码检测**。同时传 `metadata_filename`，
    保证 `filename` / `source` 等 schema 必需字段与原来一致（已验证）。
    """

    def _get_elements(self):
        from unstructured.partition.md import partition_md

        # errors="strict"：管道里所有 Markdown 都是我们自己写出的 UTF-8
        # （原生 .md / pymupdf4llm / markitdown 产物），读不出来就是真有问题，
        # 应当显式报错而不是静默替换字符（见 ROADMAP 1.3.6）。
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

    # Milvus schema（见 documents/milvus_db.py）中声明为「非空且无默认值」的字段。
    # unstructured 并不会对所有元素都产出这些键（例如顶层正文块没有 category_depth），
    # 缺字段会在 add_documents 时报 DataNotMatchException。这里统一兜底，
    # 只补 metadata，不改动任何正文内容。
    REQUIRED_FIELD_DEFAULTS = {
        'category': '',
        'source': '',
        'filename': '',
        'filetype': '',
        'title': '',
        'category_depth': 0,
    }

    # Milvus schema 的 VARCHAR 上限（见 documents/milvus_db.py），**按字节**计
    # （中文 3 字节/字）。超长会让整批写入失败：实测表格块的 title 取自卫标题，
    # 而父标题在合并阶段会不断吸收子元素，曾累积到 1496 字符 ->
    # MilvusException code=1100 length of varchar field title exceeds max length。
    # 这里统一兜底：正常长度的字段值不变，只在超限时截断。
    FIELD_MAX_BYTES = {
        'category': 1000,
        'source': 1000,
        'filename': 1000,
        'filetype': 1000,
        'title': 1000,
    }

    # 单个块的 **page_content 字节上限**（Milvus 的 text 是 VARCHAR(6000)，按字节计）。
    #
    # [FIX] 此前**没有任何针对正文的字节约束**，换真实语料后立刻暴露：
    #     papers  源 175 块只写入  88（87 块被拒）
    #     manuals 源  97 块只写入  63（34 块被拒）
    #     报错全部是 MilvusException code=1100 length of varchar field text exceeds max length
    #     最长的一块达 20184 字节（上限的 3.4 倍）
    #
    # 为什么切片阶段没能拦住：
    #   ① `text_chunker` 的阈值 5000 是**字符**（中文 5000 字 ≈ 15000 字节），
    #      而且 `SemanticChunker` 在语义高度相似时**找不到断点**（表格/数字密集的块尤甚），
    #      于是整块原样保留；
    #   ② `table_handler` 的闸原先也是**字符**（P0-5，已改为字节）。
    # 所以这里补一道**与切分器无关的最终兜底**：无论前面怎么切，出管道前一定 ≤ 上限。
    MAX_CONTENT_BYTES = 5800        # 留 200 字节余量给 metadata 拼接与换行

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
            if end <= start:                    # 单个字符就超限的极端情况
                end = min(start + limit, len(raw))
            out.append(raw[start:end].decode('utf-8', errors='ignore'))
            start = end
        return out

    @staticmethod
    def _enforce_byte_limit(datas: List[Document]) -> List[Document]:
        """把超字节上限的块切开（切出的片段继承原 metadata）。"""
        out: List[Document] = []
        for d in datas:
            if len(d.page_content.encode('utf-8')) <= MarkdownParser.MAX_CONTENT_BYTES:
                out.append(d)
                continue
            for piece in MarkdownParser._split_by_bytes(
                    d.page_content, MarkdownParser.MAX_CONTENT_BYTES):
                out.append(Document(page_content=piece, metadata=dict(d.metadata)))
        return out

    @staticmethod
    def _truncate_utf8(value, max_bytes: int) -> str:
        """按 UTF-8 字节数截断，保证不切坏多字节字符。"""
        s = str(value)
        raw = s.encode('utf-8')
        if len(raw) <= max_bytes:
            return s
        return raw[:max_bytes].decode('utf-8', errors='ignore')

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

    def __init__(self, table_max_rows: int = DEFAULT_MAX_ROWS):
        """
        Args:
            table_max_rows: [ROADMAP 2.4] 表格行数阈值。超过则按行切分，
                且每块重复完整表头（表头注入）。默认 30 行。
        """
        self.text_splitter = SemanticChunker(
            bge_embedding, breakpoint_threshold_type="percentile"
        )
        self.table_max_rows = table_max_rows

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            if len(d.page_content) > 5000:  # 内容超出了阈值，则按照语义再切割
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs
        
    def parse_markdown(self, md_file: str) -> List[Document]:
        # [UPGRADE ] 由 UnstructuredMarkdownLoader 换成 UTF8MarkdownLoader：
        # 前者会让 unstructured 自动猜测文件编码，实测把 PDF 转出的 Markdown 误判成
        # hp_roman8，把 U+2013 等非 ASCII 字符解成乱码（详见 UTF8MarkdownLoader 说明）。
        loader = UTF8MarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    # [ROADMAP 2.7] 版式噪声类别：这些元素是每页重复出现的内容（或纯装饰），
    # 不应作为独立内容块入库。实测 187 个 hi_res JSON 里 Header 39 个、PageNumber 15 个，
    # 直接入库会污染召回（见 2.3 的页眉页脚结论）。
    NOISE_CATEGORIES = {'Header', 'Footer', 'PageNumber'}

    # [ROADMAP 2.7] frontmatter（YAML 元数据头）识别。
    #
    # 为什么必须按内容识别、而不能靠 category：实测 unstructured 把 frontmatter
    # 识别为 **NarrativeText** 且无父标题 —— 与正文段落的 category **完全相同**，
    # 无法从类别上区分。
    #
    # 为什么必须排除：它描述的是「文档自身」（id / summary / related_key），
    # 而不是文档内容，且与正文高度冗余（summary 就是正文的浓缩）。
    # 实测放进来会让 milvus_docs 从 87 块涨到 93 块、context_recall 从 0.8856
    # 掉到 0.8464 —— 冗余块挤占了检索候选窗口。
    # 项目早先的设计也明确不让它入库（见 eval/export_chunks.py 的注释）。
    _FRONTMATTER_RE = re.compile(r'^\s*id\s*:\s*\S+\.(?:md|mdx)\b', re.IGNORECASE)

    # [ROADMAP 2.6] 带「结构前缀」的块：它们是明确的独立语义单元
    # （表格按行切分的分片、PPTX 的每页块），**不参与**超短块合并。
    STRUCT_PREFIXES = ('\u8868\u683c\uff1a', '\u5e7b\u706f\u7247 ')   # "表格：" / "幻灯片 "

    # [ROADMAP 2.5] 超短块合并阈值（字符）。
    #
    # 为什么要做这件事：PPTX / docx 的解析会产生大量「一个要点一个元素」的碎片块。
    # 实测 manuals 集合里有 5 个 26~43 字符的块，内容有效但**过短** ——
    # 短文本的向量更「集中」，检索时反而容易排到前面，把正常的块挤出候选窗口。
    # 实测证据：q042 / q043 的目标块正是 28 / 36 字符，在集合块数从 8 增到 16 后
    # 被挤出 top-6（调大 PER_COLLECTION_K 也无效，因为瓶颈是排名而非 K）。
    #
    # 阈值取 50：扫描显示 <30 / <40 / <50 / <60 分别会合并 3 / 5 / 6 / 7 个块，
    # 50 是「覆盖 PDF 里 `1, pp.`（6 字符）这类残留噪声」与「不过度合并」的平衡点。
    MIN_CHUNK_CHARS = 50

    @staticmethod
    def _merge_short_chunks(datas: List[Document],
                            min_chars: int = MIN_CHUNK_CHARS) -> List[Document]:
        """把过短的块并入**同一来源文件**的相邻块。

        设计约束：
          * **不跨文档合并** —— 不同文件的块拼接会产生无意义的混合语义。
          * **结构块不参与合并，也不作为合并目标** —— 表格 / 幻灯片块带固定的结构与行范围
            语义（「表格：X（第 N-M 行）」「幻灯片 N：…」），被拼接后会失效。
          * **只并入前一块，不向后找**；找不到可并入的对象时**原样保留**。
            [修正  / P1-7] docstring 原先写的是「若它是该文件首块，则并入后一块」——
            与实现**相反**，而且那正是早期版本踩过的坑：后一块此时**还没被 append 进 out**，
            跳过它（`i += 2`）会让它**永远不出现在结果里**（实测把 pptx 的 3 个块吞成 1 个）。
            只往前合并则天然无此风险 —— 被合并者必然已安全落在 `out` 中。
            实现处的注释见下方（`_joinable` 附近）。
          * 只改 `page_content`，元数据沿用被并入的那一块，避免引入空 title 等问题。
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
            # [ROADMAP 2.6] 带结构前缀的块（"表格：" / "幻灯片 "）是独立的语义单元，
            # 短也不该被合并 —— 否则 PPTX 的每页块会被吞成一个（实测 3 页 -> 1 块）。
            is_structural = text.startswith(prefixes)

            if is_structural or not fname or len(text) >= min_chars:
                out.append(doc)
                i += 1
                continue

            # 这是一个超短块：并入**前一个**同文件、且 category 为 content 的块。
            #
            # 两点设计约束（都来自实测踩坑）：
            #  1) 必须限定 category == 'content' —— 只有 content 块会入库与被检索。
            #     若并入 Title 块，内容会随 Title 一起被过滤掉，属于静默丢失。
            #  2) **只并入前一块，不向后找** —— 早期版本曾「前块不可用就并入后一块」，
            #     但后一块这时还没被 append 进 out，跳过它（i += 2）会让它
            #     **永远不出现在结果里**。实测把 pptx 的 3 个块吞成 1 个。
            #     只往前合并则天然无此风险：被合并者必然已安全落在 out 中。
            def _joinable(d):
                return (d is not None
                        and d.metadata.get("filename") == fname
                        and d.metadata.get("category") == "content"
                        # 结构块既不被合并，也**不作为合并目标**（否则页码/表名会被
                        # 混入一段无关文本，破坏其结构语义）
                        and not d.page_content.startswith(prefixes))

            prev_doc = out[-1] if out and _joinable(out[-1]) else None
            if prev_doc is None:
                # 没有可安全并入的对象（如该文件的第一个块）—— 原样保留，绝不丢内容
                out.append(doc)
                i += 1
                continue

            prev_doc.page_content = (prev_doc.page_content + " " + text).strip()
            log.info(
                f'[合并超短块] {fname}: {len(text)} 字符的块并入前一块 '
                f'(合并后 {len(prev_doc.page_content)} 字符)'
            )
            i += 1
        return out

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        """把「标题 + 其下元素」合并成带面包屑的内容块。

        [ROADMAP 2.4] 表格元素额外产出一份**结构化独立块**（表格感知切分），
        但**不从原有合并流程中移除**。

        为什么是「追加」而不是「抽离」—— 这是一处实测踩坑后的修正：
            最初实现让可恢复结构的表格（带 `text_as_html`）产出独立块后
            直接 `continue`（不再并入父标题）。结果父标题累积的内容少了表格部分，
            而含表格的文档（PDF/Office）里有大量 **4000+ 字符的超长块**，
            它们会走 `SemanticChunker` 语义切割 —— 内容一变，**语义断点整体位移**，
            导致周围非表格块的边界连锁重排。实测证据：
                papers  旧 24 块 - 完全相同 16 - 改变 8（都是 4000+ 的超长块）
                milvus_docs（无表格、块小不触发切割）则**完全未变**
            现在改为「表格照旧并入父标题（原块边界 100% 不变）+ 额外追加独立表格块」，
            代价是表格文本会同时存在于原位置与独立块中（少量重复），
            换来的是**对既有语料零扰动**，且表格获得了可精确检索的结构化表示。
        """
        merged_data = []
        extra_table_docs: List[Document] = []  # 额外产出的表格块（最后追加，不打乱既有顺序）
        parent_dict = {}  # 是一个字典，保存所有的父document， key为当前父document的ID
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            # [ROADMAP 2.4] 表格：恢复结构 + 表格感知切分，产出一份独立的结构化块。
            #
            # [FIX / ROADMAP 4.17] 原先这里**不** continue（注释写
            # 「以保证父标题的累积内容与改造前完全一致，块边界不变」），于是表格的
            # **扁平文本**（`page_content`）还会走下面的合并逻辑、被拼进**父标题块**：
            #
            #     parent.page_content = parent.page_content + ' ' + document.page_content
            #
            # 结果同一张表**以两种形态同时入库**（结构化块 + 父块里的扁平文本）。
            # 2.4 当时判断这是「少量重复，可接受」，但实测「少量」不成立：
            #   * 库里 528 块中 **236 块（44.7%）** 落在「同 filename+title」组里；
            #   * 且**直接损害指标** —— n005 的 4 个 golden 里有 2 个是扁平版，
            #     `recall` 因此从 1.00 掉到 0.50（见 4.15 / 4.16）。
            #
            # 现在的做法：**只在能证明「扁平文本已被结构化块完整覆盖」时**才跳过累积
            # （`is_flat_twin`，三维判据 + 实测阈值）。这样：
            #   * 纯孪生（如 `capability_matrix / 来源说明`）-> 父块不再被塞入表格文本；
            #   * 含额外正文的（如 `index_spec / 6.2 依赖组件版本`，父块里还有
            #     「6 部署环境要求 Before installing…」）-> **照旧累积**，绝不丢内容；
            #   * 判定不出（HTML 解析失败 / 提取不到可判别值）-> 照旧累积（严格向）。
            #
            # 为何用 continue 是安全的：`category == 'Table'` 不可能是 `Title`，
            # 因此跳过下面的 Title 分支不影响面包屑的构建。
            if has_restorable_table(document):
                table_name = ''
                if parent_id and parent_id in parent_dict:
                    # 父标题可能已是面包屑（"A -> B"），表名取最后一段
                    table_name = parent_dict[parent_id].page_content.split(' -> ')[-1]
                table_chunks = restore_table_documents(
                    document, table_name=table_name, max_rows=self.table_max_rows
                )
                if table_chunks:
                    extra_table_docs.extend(table_chunks)
                    # 注意 `restore_table_documents()` 返回的是 **List[Document]**，
                    # 而 `is_flat_twin()` 要的是**文本列表**。首版直接传 Documents，
                    # 在真实 md 文件上抛 `AttributeError: 'Document' object has no
                    # attribute 'splitlines'`（由 B 段管道验证抓到 —— 见
                    # eval/verify_flat_twin_pipeline.py；单元测试测不到，因为
                    # 那里传的是裸字符串）。
                    if is_flat_twin(document.page_content,
                                    [d.page_content for d in table_chunks]):
                        continue

            # [ROADMAP 2.7] 顶层正文块（无父标题）独立成块。
            #
            # 原实现的条件是 `category == 'NarrativeText'`，但 unstructured 对**纯段落**
            # 常识别为 `UncategorizedText` 而非 NarrativeText —— 这类元素如果同时没有
            # `parent_id`，下面三条分支**全都不命中**，于是被**静默丢弃**。
            # 实测：输入两段无标题的正文，整个解析返回 0 个块。
            #
            # 现放宽为「任何非 Title、非版式噪声的元素，只要没有父标题就独立成块」。
            # 必须排除 Header/Footer/PageNumber：它们是每页重复的版式内容，
            # 放进来会污染检索（见 2.3 关于页眉页脚的实测结论）。
            #
            # 另：Milvus schema 中 title 是非空 VARCHAR，不补齐会在 add_documents 时报
            # DataNotMatchException: Insert missed an field `title`。
            # 这里只补 metadata，不改动任何正文内容；取值用文件名，缺失时退化为空串。
            if (category != 'Title' and parent_id is None
                    and category not in self.NOISE_CATEGORIES
                    # frontmatter 不是正文内容，且与正文冗余（见 _FRONTMATTER_RE 的说明）
                    and not self._FRONTMATTER_RE.match(document.page_content)):
                document.metadata.setdefault('title', document.metadata.get('filename', ''))
                # 必须**直接赋值**而非 setdefault：`category` 键本来就存在
                # （值是 unstructured 的元素类型，如 UncategorizedText / NarrativeText），
                # setdefault 对已存在的键不生效，会导致「块产出了但 category 不是 content」
                # -> 检索层按 `category == 'content'` 过滤，等于白产出（实测踩过这个坑）。
                #
                # 注意这与 Title 分支的处理不同：Title 一律保持 'Title' 而不入库
                # （标题只作为面包屑来源），非 Title 的正文块才改写为 'content'。
                document.metadata['category'] = 'content'
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent = parent_dict.get(parent_id)
                if parent is not None:
                    parent.page_content = parent.page_content + ' ' + document.page_content
                    parent.metadata['category'] = 'content'
                else:
                    # 悬空父：该元素的 parent_id 指向的元素不在本批中（实测 PDF hi_res 会把
                    # 页眉识别成 `Header`，而它并未进入 parent_dict）。原实现在此处直接
                    # parent_dict[parent_id] 取键，会抛 KeyError 中断**整个文件**的解析
                    # （datas/output 那 187 个 JSON 正是因此从未被消费）。
                    # 这里降级为独立内容块，保住内容且不中断解析。
                    document.metadata.setdefault('title', document.metadata.get('filename', ''))
                    document.metadata['category'] = 'content'
                    merged_data.append(document)

        # 处理字典
        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        # [ROADMAP 2.4] 追加表格块：放在最后，保证既有块的内容与顺序都不受影响
        # （表格是「额外增加」的可检索表示，而非对既有块的替换）。
        merged_data.extend(extra_table_docs)

        return merged_data

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)

        log.info(f'文件合并后的长度: {len(merged_documents)}')

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度: {len(chunk_documents)}')

        # [ROADMAP 2.5] 合并过短块。放在切片之后：切片本身不会再产出超短块，
        # 但合并要在字段兜底之前，保证并入后的块也走一遍 schema 字段校验。
        chunk_documents = self._merge_short_chunks(chunk_documents)
        log.info(f'合并超短块后的长度: {len(chunk_documents)}')

        # [FIX] 字节上限兜底（见 MAX_CONTENT_BYTES 的说明）。
        n_before = len(chunk_documents)
        chunk_documents = self._enforce_byte_limit(chunk_documents)
        if len(chunk_documents) != n_before:
            log.warning(f'超长块切分: {n_before} -> {len(chunk_documents)} '
                        f'（按 {self.MAX_CONTENT_BYTES} 字节上限）')

        # 补齐 Milvus schema 必需但 unstructured 未必产出的字段（见 _ensure_required_fields）
        chunk_documents = self._ensure_required_fields(chunk_documents)
        return chunk_documents

    def parse_markdown_to_documents_from_text(
        self, md_text: str, source_name: str = 'converted.md', keep_temp: bool = False
    ) -> List[Document]:
        """从 Markdown **文本内容**解析成 Document 块。

        供 PDF / Office 转换结果复用本管道（见 documents/pdf_parser.py、
        documents/office_parser.py）——它们先转成 Markdown，再走与原生 .md
        **完全相同**的合并、切片与字段兜底流程，因此检索层与工作流层零改动。

        为什么落地临时文件而不直接用 file=BytesIO：
            UnstructuredMarkdownLoader 会从**文件名**推导 filename / source 等
            metadata，而这些是 Milvus schema 的必需字段。走 BytesIO 会让它们缺失。
            临时文件写在系统临时目录（不污染 datas/md/），解析完成后即删除。

        Args:
            md_text: 已转换好的 Markdown 全文。
            source_name: 临时文件名，同时决定入库时 metadata 里的 filename。
                建议传「原文件名 + .md」（如 report.pdf -> report.pdf.md），
                便于追溯来源。
            keep_temp: True 时保留临时文件供人工核对（默认 False，解析后删除）。
        """
        # 空内容必须显式失败：静默返回空列表会让「解析失败」伪装成「文档为空」，
        # 是最难排查的故障形态（见 ROADMAP 1.3.6）。
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


if __name__ == '__main__':
    # 路径改为配置化：默认取包内 datas/md，可用 DATAS_DIR 覆盖
    file_path = os.path.join(MD_DIR, 'tech_report_0tfhhamx.md')
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"元数据: {item.metadata}")
        print(f"标题: {item.metadata.get('title', None)}")
        print(f"doc的内容: {item.page_content}\n")
        print("------" * 10)
