"""`.html` / `.txt` 的解析器：**统一转 Markdown 后复用同一条管道**（ROADMAP 4.50）。

## 为什么这两个能低成本支持（纠正之前的说法）

之前我在文档里写「HTML 不支持」，**这个说法不准确** —— 准确说法是「**没实现 parser**」，
而不是「做不到」。依据：

* `table_handler.parse_table_html` 用 **Python 标准库 `html.parser`** 就能解析表格，
  说明 HTML 解析在本项目**已有基础**；
* `markitdown==0.1.8` **已装且实测可用**：`<h1>T</h1><p>hello</p>` -> `# T\\n\\nhello`；
* 项目**本来就用 markitdown 处理 Office**（`requirements.txt` 注释：
  "Office（docx / xlsx / pptx）：统一转 Markdown 后复用同一条管道"）。

**所以只要把 html/txt 也转成 Markdown，就能复用 `MarkdownParser` 的全部能力**
（标题层级、表格处理、语义分块、元数据），**零新增依赖**。

## 两者的差别

**HTML**：有显式结构（`<h1>`~`<h6>` / `<table>` / `<ul>`），markitdown 转出来天然带
Markdown 标题与表格 -> 切片质量与 `.md` 文件一致。

**TXT**：**没有任何结构**。所以必须人工补一个「有意义的标题」，
否则切片质量差（这正是 `DOC_SOURCES` 里 `notes` 源当初 `enabled=False` 的原因）。
本实现用 **文件名（去扩展名）当一级标题** —— 文件名通常就描述了内容主题，
比「自动补的弱标题」可靠。若文件名是 `新建文本文档` 这类无意义名字，
切片质量仍会差，**这是 `.txt` 的固有限制，不是实现问题**。

## 与 `MarkdownParser` 的关系

两个 parser 都是**薄适配器**：转成 Markdown 文本 -> 写临时 `.md` -> 交给 `MarkdownParser`。
这样标题合并、表格恢复、语义分块、元数据注入**全部复用**，不重复实现。
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import List

from langchain_core.documents import Document
from rag_kb.utils.log_utils import log


def _to_markdown_via_tempfile(md_text: str, stem: str) -> List[Document]:
    """把 Markdown 文本写成临时 `.md` 再交给 `MarkdownParser` 解析。

    为什么要落盘：`MarkdownParser` 的接口是「按文件路径解析」（它内部用
    `UnstructuredMarkdownLoader` + 文件级元数据）。走临时文件能**零改动复用**
    它的全部逻辑；代价是一次磁盘 IO（毫秒级），比复制一套切片逻辑更安全。
    """
    from rag_kb.documents.markdown_parser import MarkdownParser

    fd, tmp = tempfile.mkstemp(suffix=".md", prefix="conv_", text=False)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(md_text)
        docs = MarkdownParser().parse_markdown_to_documents(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    # 把临时文件名换成真实文件名 —— 元数据与引用溯源都依赖它
    for d in docs:
        d.metadata["filename"] = "%s.md" % stem
        d.metadata.setdefault("filetype", "md")
    return docs


class HTMLDocParser:
    """`.html` / `.htm` -> Markdown（markitdown）-> `MarkdownParser`。"""

    def __init__(self):
        from markitdown import MarkItDown

        self._md = MarkItDown()

    def parse(self, file_path: str) -> List[Document]:
        path = pathlib.Path(file_path)
        result = self._md.convert(str(path))
        md_text = getattr(result, "text_content", "") or ""
        if not md_text.strip():
            log.warning("[html] %s 转换后为空，跳过" % path.name)
            return []
        log.info("[html] %s -> Markdown %d 字符" % (path.name, len(md_text)))
        return _to_markdown_via_tempfile(md_text, path.stem)


class TextDocParser:
    """`.txt` -> 补一级标题（文件名）-> Markdown -> `MarkdownParser`。

    为什么用文件名当标题：`.txt` 没有结构，而 `MarkdownParser` 的「标题合并」
    依赖标题。文件名通常描述内容主题，比自动生成的弱标题可靠。
    若文件名为 `新建文本文档` 这类无意义名，切片质量仍会差 —— **这是 `.txt` 的固有限制**。
    """

    # 纯文本里「像标题」的行：短、无句末标点、可能是中文数字分节
    _HEADING_HINT = (
        "一、",
        "二、",
        "三、",
        "四、",
        "五、",
        "六、",
        "七、",
        "八、",
        "九、",
        "十、",
        "第",
        "Chapter",
        "CHAPTER",
    )

    def parse(self, file_path: str) -> List[Document]:
        path = pathlib.Path(file_path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            log.warning("[txt] %s 为空，跳过" % path.name)
            return []

        lines = raw.splitlines()
        out: list[str] = ["# %s" % path.stem, ""]
        promoted = 0
        for ln in lines:
            s = ln.strip()
            # 把「像标题」的短行提升为二级标题，给切片器一个结构信号
            if (
                s
                and len(s) <= 40
                and not s.endswith(("。", ".", "；", ";", "！", "!"))
                and any(s.startswith(h) for h in self._HEADING_HINT)
                and "|" not in s
            ):
                out.append("## %s" % s)
                out.append("")
                promoted += 1
            else:
                out.append(ln)
        md_text = "\n".join(out)
        log.info("[txt] %s -> Markdown %d 字符（提升 %d 个小标题）" % (path.name, len(md_text), promoted))
        return _to_markdown_via_tempfile(md_text, path.stem)
