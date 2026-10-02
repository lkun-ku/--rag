"""结构感知的父块 / 子块切分（ROADMAP 4.51）。

## 为什么用「结构」而不是「语义相似度」

本项目的语料以 **Markdown / 转成 Markdown 的 PDF 与 Office** 为主，**结构信号天然显式存在**：

    # / ## / ### 标题层级  ·  表格有行边界  ·  段落有空行

而 `SemanticChunker` 靠「相邻句 embedding 相似度」猜断点，在**表格/数字密集**处
语义高度相似 -> **找不到断点**（`markdown_parser.py:135` 已记录该缺陷）。

**所以父块按标题层级切、子块按段落/表格切 —— 用强信号，不用弱信号。**

## 与 `merge_section_blocks`（4.49）的关系

    4.49 是「检索后临时聚合」——零索引成本，但父块无法单独检索、无法按命中次数重排
    本模块是「入库时就分好父子」——需要建两层数据，但支持真正的 small-to-big 检索

两者**可以并存**：默认走 4.49（零成本），需要更强效果时切到真父子索引。

## 与数据无关 / 有关

    **无关**（本模块已固定）：按标题切父块、按段落/表格切子块、parent_id 生成规则
    **有关**（用参数暴露）：`parent_max_chars` / `child_max_chars`

等真实语料到位后，只需调这两个参数，**不需要改代码**。
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from langchain_core.documents import Document

# 默认上限：**通用占位值**，等真实语料的章节/段落长度分布到位后按 P90 调整
DEFAULT_PARENT_MAX_CHARS = 4000
DEFAULT_CHILD_MAX_CHARS = 800

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|")


def make_parent_id(filename: str, heading_path: List[str], seq: int) -> str:
    """父块 ID：**只依赖「文件名 + 标题路径 + 序号」，与内容无关**。

    这样同一个文件重新切分时 ID 稳定，便于「增量更新」时判断哪些父块变了。
    取 sha256 前 16 hex（64 bit）—— 在万级块规模下碰撞概率可忽略。
    """
    raw = "%s|%s|%d" % (filename, ">".join(heading_path), seq)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _iter_blocks(md_text: str) -> List[dict]:
    """把 Markdown 拆成「结构块」序列。

    每个块：`{"kind": "heading"|"text"|"table", "text": str, "level": int, "title": str}`

    只做最必要的结构识别（标题 / 表格 / 段落），**不引入额外依赖** ——
    因为目标是把「已有的结构信号」用起来，而不是再解析一遍 Markdown。
    """
    blocks: List[dict] = []
    buf: List[str] = []
    in_table = False

    def flush_text():
        nonlocal buf
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append({"kind": "text", "text": body})
            buf = []

    for line in md_text.splitlines():
        m = _HEADING_RE.match(line)
        is_table_line = bool(_TABLE_LINE_RE.match(line))

        if m:
            flush_text()
            in_table = False
            blocks.append({
                "kind": "heading",
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
                "text": line.strip(),
            })
            continue

        # 表格：连续的 `|` 行算一块（含表头与分隔行）
        if is_table_line:
            if not in_table:
                flush_text()
                in_table = True
            buf.append(line)
            continue

        if in_table:
            body = "\n".join(buf).strip()
            if body:
                blocks.append({"kind": "table", "text": body})
            buf = []
            in_table = False

        if line.strip():
            buf.append(line)
        else:
            flush_text()

    if in_table:
        body = "\n".join(buf).strip()
        if body:
            blocks.append({"kind": "table", "text": body})
        buf = []
    flush_text()
    return blocks


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """单块超过 `max_chars` 时按段落边界再切；极端情况下按字符硬切（保底不超限）。"""
    if len(text) <= max_chars:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(p) > max_chars:                    # 单段就超限 -> 硬切
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(p), max_chars):
                out.append(p[i:i + max_chars])
            continue
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out or [text]


def split_parent_child(
    md_text: str,
    filename: str,
    parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS,
    child_max_chars: int = DEFAULT_CHILD_MAX_CHARS,
) -> Tuple[List[Document], List[Document]]:
    """把 Markdown 切成 (父块列表, 子块列表)。

    规则：
      * **父块**：按 `#`~`######` 标题层级聚合；一个标题及其下属内容构成一个父块；
        超过 `parent_max_chars` 时**在子块边界处**切开（不切断段落/表格）。
      * **子块**：父块内按「段落 / 表格」切；单块超 `child_max_chars` 时按段落再切。
      * **无标题文档**：整篇作为 **1 个兜底父块**（否则无法建立父子关系）。

    Returns:
        `(parents, children)`；空输入返回 `([], [])`。
    """
    if not md_text or not md_text.strip():
        return [], []

    blocks = _iter_blocks(md_text)
    if not blocks:
        return [], []

    # ---- 1. 按一级标题（#）分段：每段构成「一个父块候选」 ----
    sections: List[dict] = []
    cur = {"heading_path": [], "blocks": []}
    for b in blocks:
        if b["kind"] == "heading" and b["level"] == 1:
            if cur["blocks"]:
                sections.append(cur)
            cur = {"heading_path": [b["title"]], "blocks": [b]}
        else:
            if b["kind"] == "heading":
                # 二级及以下：并入当前父块，同时更新 heading_path（取最近一层）
                path = cur["heading_path"][:1] + [b["title"]]
                cur["heading_path_current"] = path  # 供子块记录
            cur["blocks"].append(b)

    # 无一级标题 -> 整篇一个兜底父块
    if not sections and cur["blocks"]:
        cur["heading_path"] = ["(未命名章节)"]
        sections.append(cur)
    elif cur["blocks"]:
        sections.append(cur)

    # ---- 2. 生成父块与子块 ----
    parents: List[Document] = []
    children: List[Document] = []

    for seq, sec in enumerate(sections):
        hpath = sec["heading_path"] or ["(未命名章节)"]
        pid = make_parent_id(filename, hpath, seq)

        # 2a. 收集该父块内的「子块文本」（段落 / 表格，已按 child_max_chars 切）
        child_texts: List[str] = []
        sub_path: List[str] = list(hpath)
        for b in sec["blocks"]:
            if b["kind"] == "heading":
                if b["level"] >= 2:
                    sub_path = hpath[:1] + [b["title"]]
                continue
            for piece in _split_long_text(b["text"], child_max_chars):
                child_texts.append((piece, list(sub_path)))

        # 2b. 按 parent_max_chars 把子块**成组**分配（超限时切开父块，不切断子块）
        groups: List[List[Tuple[str, List[str]]]] = []
        cur_group: List[Tuple[str, List[str]]] = []
        cur_len = 0
        for item in child_texts:
            ln = len(item[0])
            if cur_group and cur_len + ln + 2 > parent_max_chars:
                groups.append(cur_group)
                cur_group, cur_len = [], 0
            cur_group.append(item)
            cur_len += ln + 2
        if cur_group:
            groups.append(cur_group)

        # 2c. 落库：每个 group 一个父块（首组沿用 pid，后续组派生子 ID）
        for gi, group in enumerate(groups):
            gpid = pid if gi == 0 else make_parent_id(filename, hpath, seq * 100 + gi)
            gtext = "\n\n".join(t for t, _ in group)
            parents.append(Document(
                page_content=gtext,
                metadata={
                    "filename": filename,
                    "parent_id": gpid,
                    "is_parent": True,
                    "chunk_index": 0,
                    "heading_path": hpath,
                    "part": gi,
                },
            ))
            for ci, (t, sp) in enumerate(group):
                children.append(Document(
                    page_content=t,
                    metadata={
                        "filename": filename,
                        "parent_id": gpid,
                        "is_parent": False,
                        "chunk_index": ci,
                        "heading_path": sp,
                    },
                ))

    return parents, children
