"""PDF 解析器：PDF -> Markdown -> 复用现有 MarkdownParser 管道。

设计（依据见 ROADMAP 1.3）：

    ① pymupdf4llm.to_markdown()   结构化转 Markdown（保留标题层级与表格）
    ② MarkdownParser              合并 / 切片 / 字段兜底（与原生 .md 完全相同）
    ③ Document 块                  可直接写入 Milvus，检索层零改动

为什么不用 UnstructuredPDFLoader：
    实测 `strategy="fast"` 对同一份 PDF **返回 0 个元素且不报错**（静默失败）；
    `strategy="auto"` 会走 OCR 并因缺 `unstructured_pytesseract` 直接抛错；
    `hi_res` 需要 Tesseract 系统二进制。
    而 `pymupdf4llm` 零系统依赖，实测产出 44,017 字符 / 15 个标题 / 18 行表格，
    走完管道 25 块、其中 24 块 `category=='content'`（能过检索层过滤）。

一个必须澄清的区分：
    那篇文章批评的是「**无结构地抽纯文本**」（丢失标题与表格）；
    本模块是「**结构化转 Markdown**」（保留标题与表格）。
    本项目管道是结构驱动的（Title -> 面包屑 -> category='content'），
    所以扁平文本必然失败（过不了 `expr` 过滤），带结构 Markdown 才能无缝接入。
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

# [FIX / P1-5] `MIN_MARKDOWN_CHARS` 已提升为共享常量（原为本地 100），
# 与 office_parser 统一 —— 详见 rag_kb/documents/markdown_parser.py 的定义处注释。
from rag_kb.documents.markdown_parser import MIN_MARKDOWN_CHARS, MarkdownParser
from rag_kb.utils.log_utils import log

# PDF 自带文本层少于该字符数即判为扫描件。实测样本（把文字渲染成图片再嵌回 PDF）
# 的文本层为 0 字符，而正常电子版 PDF 为数千至上万字符，阈值取 50 有充足余量。
MIN_TEXT_LAYER_CHARS = 50


def _pixmap_to_ndarray(pix):
    """把 PyMuPDF 的 Pixmap 转成 RapidOCR 可接受的 ndarray（H×W×3, uint8）。

    RapidOCR 接受图片路径或 numpy 数组；这里用内存数组可以避免为每页
    落一个 MB 级临时 PNG（扫描件渲染后体积很大）。
    """
    import numpy as np

    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA -> RGB
        arr = arr[:, :, :3]
    elif pix.n == 1:  # 灰度 -> RGB
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


class PDFParser:
    """把 PDF 解析成与 Markdown 路径完全一致的 Document 块。

    与 MarkdownParser 输出格式一致，因此：
      - 可写入同一 Milvus 集合（schema 完全相同）
      - 检索层的 `expr: category == "content"` 过滤照常生效
      - 标题面包屑（"A -> B -> 正文"）照常生成
    """

    def __init__(
        self,
        keep_temp_md: bool = False,
        ocr_fallback: bool = False,
        ocr_dpi: int = 200,
        ocr_min_score: float = 0.5,
        use_ocr: bool = True,
    ):
        """
        Args:
            keep_temp_md: 是否保留中间产出的临时 Markdown（默认 False，解析后删除）。
                排查切片问题时设为 True 可拿到转换结果人工核对。
            ocr_fallback: [ROADMAP 2.3] 文本层为空/过少时是否回落到 OCR。
                **默认 False**：OCR 慢（实测中文 2.5s/页、英文 8.5s/页），
                且当前语料并无扫描件；自动触发会让「解析 1 个 PDF」突然慢数倍。
            ocr_dpi: OCR 前把页面渲染成图片的分辨率。200 是实测的质量/耗时平衡点。
            ocr_min_score: OCR 行置信度阈值，低于该值的行会被丢弃（滤掉明显噪声）。
        """
        self.keep_temp_md = keep_temp_md
        self.ocr_fallback = ocr_fallback
        self.ocr_dpi = ocr_dpi
        self.ocr_min_score = ocr_min_score
        # [P1-4] pymupdf4llm 是否对页面图片区域做 OCR。默认 True（与修复前一致）；
        # 需要与阶段一基线严格对齐时传 False（`write_milvus` 通过 PDF_USE_OCR 暴露）。
        self.use_ocr = use_ocr
        # 复用 MarkdownParser 的合并 + 切片 + 字段兜底能力
        self._helper = MarkdownParser()

    @staticmethod
    def to_markdown(file_path: str, use_ocr: bool = True) -> str:
        """把 PDF 转成 Markdown 文本（结构化，保留标题与表格）。

        Args:
            use_ocr: 是否让 pymupdf4llm 对页面中的图片区域做 OCR。

        [UPGRADE ] 为什么必须**显式**指定 use_ocr：
            实测发现 pymupdf4llm 会根据「环境里是否装了 OCR 后端」自行决定行为，
            装上 `rapidocr_onnxruntime` 之后，同一份 layout-parser-paper.pdf 的结果
            从 44,017 字符变成 45,673 字符（+3.8%），耗时 6.9s -> 19.4s。
            这种「随环境漂移」的行为会让解析结果与评测基线无法对齐，
            因此把它固定为显式参数。

            选 True 的理由：OCR 能提取图片内的文字（图注、图内标签等），
            实测同一份 PDF 多出 1,656 字符可检索内容，与 ROADMAP 2.3
            「图片内容理解」的方向一致。若需与阶段一基线严格对齐，传 False。
        """
        import pymupdf4llm  # 局部导入：未装时不影响其他格式的解析

        return pymupdf4llm.to_markdown(file_path, use_ocr=use_ocr)

    @staticmethod
    def text_layer_chars(file_path: str) -> int:
        """统计 PDF **自带文本层**的字符数。0（或极少）表示是扫描件。

        为什么用它判定而不是看 `pymupdf4llm` 的转换结果：
            实测 `pymupdf4llm(use_ocr=True)` **自己也能 OCR 扫描件**（能读出 188 字符），
            因此「转换结果太少」这个条件永远不会成立、OCR 分支形同虚设。
            但它的产出**不带页码**，而扫描件恰恰最需要页码（引用粒度要求）。
            直接查文本层可以准确区分「电子版」与「扫描件」，从而把扫描件交给
            带页码标题的整页 OCR 路径。
        """
        import pymupdf

        doc = pymupdf.open(file_path)
        try:
            return len(''.join(p.get_text() for p in doc).strip())
        finally:
            doc.close()

    @staticmethod
    def ocr_to_markdown(file_path: str, dpi: int = 200, min_score: float = 0.5) -> str:
        """[ROADMAP 2.3] 用 RapidOCR 逐页识别扫描件，拼成 Markdown。

        为什么选 RapidOCR 而不是 `unstructured` hi_res：
            hi_res 需要 Tesseract 系统二进制 **+ 语言包**（实测本机 tessdata 为空）
            + pytesseract + unstructured_pytesseract + YOLO 版面模型（HF 下载），
            共 4 项待补依赖；而 RapidOCR 只要 3 个纯 pip 包，无系统依赖，
            实测中文行级命中 6/6、平均置信度 0.980、2.5s/页。

        输出结构（决定切片质量）：
            `# {文件名}`  +  每页一个 `## 第 N 页`  +  该页各行文本
            页标题让每个块天然带页码语义，也保证能通过 `category == 'content'` 过滤。

        Raises:
            ValueError: RapidOCR 不可用，或所有页都未识别出有效文本。
        """
        import pymupdf  # PyMuPDF

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:  # 显式报错，不静默降级
            raise ValueError(f'OCR 回退需要 rapidocr-onnxruntime（以及 onnxruntime / pyclipper / shapely）：{e}') from e

        engine = RapidOCR()
        basename = os.path.basename(file_path)
        parts: List[str] = [f'# {basename}', '']
        total_lines = 0

        doc = pymupdf.open(file_path)
        # 先取出页数：finally 里会 close，之后再访问 doc.page_count 会抛
        # ValueError: document closed（实测踩过）。
        page_count = doc.page_count
        try:
            for page_no in range(page_count):
                page = doc[page_no]
                # 渲染为图片：RapidOCR 接受 ndarray，这里用 pixmap 的原始像素构造
                pix = page.get_pixmap(dpi=dpi)
                img = _pixmap_to_ndarray(pix)

                raw = engine(img)
                # rapidocr 不同版本返回 (result, elapse) 或直接返回 result，两种都兼容
                result = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw

                lines: List[str] = []
                for item in result or []:
                    # item 形如 [box, text, score]
                    text = str(item[1]).strip() if len(item) > 1 else ''
                    score = float(item[2]) if len(item) > 2 else 1.0
                    if not text or score < min_score:
                        continue  # 低置信度行直接丢弃（OCR 会产出少量噪声行）
                    lines.append(text)

                if lines:
                    parts.append(f'## 第 {page_no + 1} 页')
                    parts.append('')
                    parts.extend(lines)
                    parts.append('')
                    total_lines += len(lines)
        finally:
            doc.close()

        if total_lines == 0:
            raise ValueError(f'OCR 未从任何页面识别出文本（置信度阈值 {min_score}）: {basename}')
        log.info(f'OCR 完成: {basename} -> {total_lines} 行 / {page_count} 页')
        return '\n'.join(parts)

    def parse_pdf_to_documents(self, file_path: str) -> List[Document]:
        """解析单个 PDF，返回与 Markdown 路径同构的 Document 块。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 转换结果为空或过少（多为扫描件，需 OCR/hi_res）。
        """
        if not os.path.isfile(file_path):
            # 不静默跳过：调用方应知道文件缺失
            raise FileNotFoundError(f'PDF 不存在: {file_path}')

        basename = os.path.basename(file_path)

        # [ROADMAP 2.3] 先判定是否为扫描件。
        # 为什么不看「pymupdf4llm 的转换结果是否太少」：实测它自带 OCR，对扫描件
        # 也能读出内容，该条件永不成立。但它的产出**不带页码**，而扫描件最需要页码
        # （引用粒度要求），因此按「PDF 自带文本层」判定，并让扫描件走整页 OCR 路径。
        text_layer = self.text_layer_chars(file_path)
        is_scanned = text_layer < MIN_TEXT_LAYER_CHARS

        if is_scanned and self.ocr_fallback:
            log.warning(f'PDF 文本层仅 {text_layer} 字符，判定为扫描件，走整页 OCR: {basename}')
            md_text = self.ocr_to_markdown(file_path, self.ocr_dpi, self.ocr_min_score)
        else:
            md_text = self.to_markdown(file_path, use_ocr=self.use_ocr)

        stripped = md_text.strip()

        if len(stripped) < MIN_MARKDOWN_CHARS:
            # 结果仍过少：要么是扫描件但未开 OCR，要么转换本身失败。
            # 两种都**显式报错**，不静默返回空内容（见 ROADMAP 1.3.6）。
            if is_scanned and not self.ocr_fallback:
                raise ValueError(
                    f'PDF 文本层仅 {text_layer} 字符，判定为扫描件（无文本层）: {basename}。'
                    f'可传 ocr_fallback=True 启用整页 OCR（见 ROADMAP 2.3）。'
                )
            raise ValueError(
                f'PDF 转 Markdown 后内容过少（{len(stripped)} 字符 < {MIN_MARKDOWN_CHARS}）: {basename}。'
                f'自带文本层 {text_layer} 字符。请确认文件有效、未加密且非纯图片。'
            )

        log.info(f'PDF 转 Markdown: {basename} -> {len(md_text)} 字符')

        # 走与 Markdown 完全一致的合并 -> 切片 -> 字段兜底流程。
        # source_name 用「原名 + .md」，让入库后的 metadata.filename 可追溯来源。
        docs = self._helper.parse_markdown_to_documents_from_text(
            md_text,
            source_name=f'{basename}.md',
            keep_temp=self.keep_temp_md,
        )
        log.info(f'PDF 解析完成: {basename} -> {len(docs)} 个块')
        return docs


if __name__ == '__main__':
    from rag_kb.utils.env_utils import DATAS_DIR

    # 演示：解析 datas/pdf 下的第一个 PDF，保留中间 md 便于核对
    pdf_dir = os.path.join(DATAS_DIR, 'pdf')
    pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')] if os.path.isdir(pdf_dir) else []
    if not pdfs:
        print(f'未找到 PDF：{pdf_dir}（可把 PDF 放入该目录后重试）')
    else:
        target = os.path.join(pdf_dir, pdfs[0])
        parser = PDFParser(keep_temp_md=True)
        docs = parser.parse_pdf_to_documents(target)
        print(f'解析 {pdfs[0]} -> {len(docs)} 个块')
        for d in docs[:3]:
            print(f"  title={d.metadata.get('title')!r} category={d.metadata.get('category')!r}")
            print(f"  {d.page_content[:100]}...")
