"""引用溯源与拒答支撑（ROADMAP 2.1）。

## 要解决的问题

RAG 的答案如果没有可核验的来源标注，用户无法判断它是「检索到的」还是「编造的」。
本项目实测：给模型加上强约束 prompt 后，**引用标注率 100%**（两次独立运行编号一致），
拒答也正确（问「如何用 Rust 写贪吃蛇」时输出「知识库中未找到依据。」）。

## 本模块提供四件事

1. `format_context`   —— 上下文**编号化**（`[1] ... [2] ...`），编号与 documents 下标严格对应
2. `parse_citations`  —— 从答案中提取引用编号
3. `validate_citations` —— 校验编号是否越界（**模型可能引用不存在的编号**，
   这种「幻觉引用」比「没有引用」更危险：它让用户误以为答案有依据）
4. `is_refusal` / `build_citation_meta` —— 区分设计内拒答、构造溯源映射

## 编号必须在「工具层」生成的原因

graph_react 的上下文由 `retriever_tool` 输出，到 `generate` 节点时已经是**拼接好的字符串**，
块边界已丢失，无法再编号。因此编号在工具层完成（见 `tools/retriever_tools.py`）。
实测 `create_retriever_tool` 的 `document_prompt` 变量只能取自 document.metadata，
**不支持序号**，故该工具改为自定义实现。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

# 引用编号形如 [1]、[12]；允许多个连续标注（[1][3]）
CITE_RE = re.compile(r"\[(\d+)\]")

# 设计内的拒答文案。校验层据此区分「拒答」与「幻觉」：
# 拒答是允许的（不触发熔断重试），而编号越界必须重试。
REFUSAL_TEXT = "知识库中未找到依据。"


def format_context(docs: Sequence[Document], start: int = 1) -> str:
    """把 Document 列表编号拼成上下文。

    编号与 `docs` 的下标严格对应（`start` 默认为 1），下游校验层据此反查原文块。

    Args:
        docs: 检索到的文档块。
        start: 起始编号（默认 1，即 [1] 对应 docs[0]）。
    """
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start))


def parse_citations(answer: str) -> List[int]:
    """提取答案中的引用编号（**保序去重**，保留首次出现的顺序）。"""
    seen, out = set(), []
    for m in CITE_RE.findall(answer or ""):
        n = int(m)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# 被检索块**原文自带**的方括号编号，形如 `[28]` / `[28, 29]` / `[28,29]`。
# 用途见 validate_citations 的 `quoted`（P0-3）。
_QUOTED_NUM_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers_in_docs(docs: Sequence[Document]) -> set:
    """收集被检索块正文里出现过的方括号编号。

    只处理 `[N]` 与 `[N, M]` 两种最常见形态。区间写法（`[28-30]`）不展开 ——
    当前语料（`papers` 集合转出的 Markdown）实测都是前两种。
    """
    out = set()
    for d in docs or []:
        content = getattr(d, "page_content", "") or ""
        for m in _QUOTED_NUM_RE.finditer(content):
            for part in m.group(1).split(","):
                part = part.strip()
                if part.isdigit():
                    out.add(int(part))
    return out


def validate_citations(
    answer: str, n_docs: int, start: int = 1, docs: Optional[Sequence[Document]] = None
) -> Dict:
    """校验引用编号是否落在实际上下文范围内。

    为什么必须校验：模型可能引用不存在的编号（幻觉的一种形式）。
    这类引用比「没有引用」更危险 —— 它让用户以为答案有依据。

    Args:
        answer: 模型输出。
        n_docs: 本次提供的上下文块数。
        start: 编号起始值（与 format_context 保持一致）。
        docs: 本次提供的上下文块。**传入后可区分「幻觉引用」与「复述原文编号」**
            （见 `quoted`）；不传则行为与旧版一致（越界一律算 invalid）。

    Returns:
        {"cites": [...], "invalid": [...], "quoted": [...], "valid": bool,
         "is_refusal": bool}

    `quoted` 与 `invalid` 的区别（**[FIX / P0-3]**）：

        越界编号分两类，处理方式必须不同：
          * **quoted**：该编号**确实存在于被检索块的原文里** —— 例如 `papers`
            集合的块正文含参考文献编号 `[28]`，LLM 复述原文时把它带了出来。
            这**不是幻觉**（它没编造依据），不应触发熔断重试。
          * **invalid**：编号在原文里找不到 —— 这才是模型编造的引用。

        修复前的判据是「超出 1..n_docs 就算错」，于是 `[28]` 这类原文编号会被
        误判 → `crag_graph` 走 `not supported` → **触发重生成（最多 2 次）**：
        白烧 LLM 调用，且**可能把原本正确的答案改坏**。

        判据选「是否出现在原文」而不是「数值是否大于某个量级」的原因：
        `[28]`（真实参考文献）与 `[19]`（幻觉，n_docs=18 时）在数值上无法区分，
        而「原文里有没有」是确定性的。
    """
    cites = parse_citations(answer)
    upper = start + n_docs - 1
    out_of_range = [c for c in cites if c < start or c > upper]

    quoted: List[int] = []
    invalid: List[int] = []
    if out_of_range and docs:
        in_docs = _numbers_in_docs(docs)
        for c in out_of_range:
            (quoted if c in in_docs else invalid).append(c)
    else:
        invalid = out_of_range

    return {
        "cites": cites,
        "invalid": invalid,
        "quoted": quoted,
        "valid": not invalid,
        "is_refusal": is_refusal(answer),
    }


def is_refusal(answer: str) -> bool:
    """是否为设计内的拒答输出。

    注意：拒答**不应**触发熔断重试 —— 它是「资料不足」的正确响应，
    与「编造了不存在的引用」是两回事。
    """
    return REFUSAL_TEXT in (answer or "")


def build_citation_meta(docs: Sequence[Document], cites: Sequence[int], start: int = 1) -> List[Dict]:
    """构造「编号 -> 原文块信息」的映射，供前端/调用方渲染溯源。

    Returns:
        [{"n": 3, "filename": "产品手册.docx", "title": "索引类型", "chars": 220}, ...]
        只返回确实被引用、且编号有效的条目。
    """
    out: List[Dict] = []
    for n in cites:
        idx = n - start
        if 0 <= idx < len(docs):
            d = docs[idx]
            from_web = d.metadata.get("source") == "web"
            out.append(
                {
                    "n": n,
                    "filename": d.metadata.get("filename", ""),
                    # 表格块的行范围信息藏在正文前缀里（如「表格：X（第 31-60 行）」），
                    # 这里透传正文首行，便于前端展示「引用的是哪张表的哪一段」。
                    "title": d.metadata.get("title", ""),
                    "chars": len(d.page_content),
                    # [FIX / P0-9] 该来源是否来自**网络搜索**而非知识库。
                    # 不区分的话，网页内容会被当成知识库文档展示给用户。
                    "from_web": from_web,
                    # 联网来源的原网页地址：让用户能点开核对，而不是只看到一句占位符
                    "url": d.metadata.get("url", "") if from_web else "",
                }
            )
    return out
