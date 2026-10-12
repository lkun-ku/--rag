"""语义分块器封装：隔离 `langchain-experimental` 的 sunset 风险（ROADMAP 4.48）。

## 为什么要这层

`langchain-experimental` 官方已标记 **sunset、不再维护**（`requirements.txt` 钉在 0.4.2）。
它现在能跑，但**升级 LangChain 主版本时可能 import 失败** —— 一旦失败，
`documents/markdown_parser.py` 在 import 阶段就炸，**整个入库链路瘫痪**。

## 这层做什么（**只保命，不换策略**）

    优先：SemanticChunker(bge_embedding, breakpoint_threshold_type="percentile")
          -> **与改造前行为完全一致**，分块结果不变，**不需要重建索引**
    降级：RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
          -> 仅在 import 失败时启用，并打 WARNING（不许静默）

## 为什么不在这一层直接换「结构感知分块」

换了分块策略 -> 523 块全量重灌 -> **golden 集 32 题的 `source_blocks` 全部失效、
必须重新标注块 ID**。这是**人力成本**，不是机器成本。

所以换策略必须走 A/B 实验（同套 golden、同套门禁、对比 recall），
见 ROADMAP 4.48 的后续计划。**本模块只消除「未来 import 失败」这个尾部风险。**

## 本项目的语料特点（决定后续该换什么）

    Markdown 有 `# ## ###` 标题层级、表格有行边界 —— **结构信号本来就是显式的**
    而 `SemanticChunker` 靠「相邻句 embedding 相似度」猜断点，
    在**表格/数字密集**处语义高度相似 -> **找不到断点**（`markdown_parser.py:135` 已记录）
    所以后续候选是「结构感知分块」，而不是更复杂的语义方案。
"""

from __future__ import annotations

from typing import Any

from rag_kb.utils.log_utils import log

# 降级时的结构分块参数（与语义分块的平均块长量级相当）
FALLBACK_CHUNK_SIZE = 800
FALLBACK_CHUNK_OVERLAP = 100

# import 探测（模块加载时一次性）。
#
# 用 `Any` 显式声明而不是 `type: ignore`：CI 的静态检查环境不安装
# langchain_experimental，此时 `ignore_missing_imports` 不会产生错误，
# `type: ignore` 反而会因「未使用」被 warn_unused_ignores 判为失败。
SemanticChunker: Any = None
_SEMANTIC_IMPORT_ERROR: str | None = None
try:
    from langchain_experimental.text_splitter import SemanticChunker as _SemanticChunker

    SemanticChunker = _SemanticChunker
except Exception as e:  # noqa: BLE001
    _SEMANTIC_IMPORT_ERROR = "%s: %s" % (type(e).__name__, str(e)[:120])

SEMANTIC_AVAILABLE = SemanticChunker is not None


def make_splitter(embedding=None, *, force_fallback: bool = False):
    """构造分块器。**优先语义分块（行为不变）**，不可用时降级为结构分块。

    Args:
        embedding: 语义分块要用的 embedding 对象（本项目传 `bge_embedding`）。
        force_fallback: 仅供测试 —— 强制走降级路径，模拟「实验包不可用」。

    Returns:
        一个 `TextSplitter`（有 `split_documents` / `split_text`）。
    """
    if SEMANTIC_AVAILABLE and not force_fallback:
        return SemanticChunker(embedding, breakpoint_threshold_type="percentile")

    # ---- 降级路径 ----
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    reason = "force_fallback" if force_fallback else (_SEMANTIC_IMPORT_ERROR or "unknown")
    log.warning(
        "[splitter] langchain-experimental 不可用（%s），"
        "已降级为 RecursiveCharacterTextSplitter(size=%d, overlap=%d)。"
        "**分块结果会变化**，如需重建索引请先做 A/B 对比。" % (reason, FALLBACK_CHUNK_SIZE, FALLBACK_CHUNK_OVERLAP)
    )
    return RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )


def status() -> dict:
    """当前分块器状态（供 `/health` 或诊断脚本展示）。"""
    return {
        "semantic_available": SEMANTIC_AVAILABLE,
        "import_error": _SEMANTIC_IMPORT_ERROR,
        "fallback_chunk_size": FALLBACK_CHUNK_SIZE,
        "fallback_chunk_overlap": FALLBACK_CHUNK_OVERLAP,
    }
