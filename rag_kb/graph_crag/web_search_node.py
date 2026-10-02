from urllib.parse import urlsplit

from langchain_core.documents import Document

from rag_kb.llm_models.all_llm import web_search_tool
from rag_kb.utils.log_utils import log


def web_search(state):
    """
    基于优化后的问题进行网络搜索

    Args:
        state (dict): 当前图状态，包含优化后的问题

    Returns:
        state (dict): 更新后的状态，documents字段替换为网络搜索结果
    """
    log.info("---WEB SEARCH---")  # 阶段标识
    question = state["question"]  # 获取优化后的问题

    # 执行网络搜索
    raw = web_search_tool.invoke({"query": question})
    # 工具返回形态随版本/成败而变，逐一兼容（都是实测踩到的）：
    #   1) dict {'results': [...]} —— langchain_tavily 的 TavilySearch 默认返回原始响应
    #   2) dict {'error': ...}     —— 同上；失败时把异常包成 dict（handle_tool_error=True）
    #   3) list[dict]              —— 旧版 TavilySearchResults 的 content 形态
    #   4) (list, raw) 元组        —— response_format='content_and_artifact'
    #   5) str                     —— 更早版本出错时返回异常字符串（如 "HTTPError('401 ...')"）
    if isinstance(raw, tuple):
        results = raw[0]
    elif isinstance(raw, dict):
        if raw.get("error"):
            log.warning(f"web_search 失败：{raw['error']}")
            results = []
        else:
            results = raw.get("results") or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = raw

    if isinstance(results, str):
        # 老版本的错误字符串：不要把它当作「资料」喂给生成，直接判为无结果
        log.warning(f"web_search 返回异常文本：{results[:120]}")
        results = []
    web_results = "\n".join(str(d.get("content", "")) for d in results).strip()

    if not web_results:
        log.warning("web_search 没有可用结果（网络不可用或未命中），返回空 documents")
        return {"documents": [], "question": question}
    # [FIX / P0-9] 必须标记「来源是网络搜索」，不能只建成裸 Document。
    #
    # 原因：这条分支的结果会与知识库检索结果走**同一条**编号化路径
    # （`format_context` -> `[1][2]...`，见 generate_node），并被
    # `build_citation_meta` 渲染成溯源条目。若不标记，用户在溯源里看到的是一个
    # 「知识库文档」样式的条目（带 filename/title），**实际内容却来自网页** ——
    # 比"没有溯源"更容易误导。
    # [FIX / ROADMAP 4.32] **必须包成 list**，与其它所有节点保持一致。
    #
    # 原实现返回**单个 `Document`**，而 `retrieve` / `grade_documents` 等节点返回
    # `List[Document]`。这个类型不一致是**潜在的雷**：任何假设「documents 是 list」
    # 的代码，在**走了 web_search 的路径上都会崩**：
    #
    #     TypeError: object of type 'Document' has no len()
    #
    # （4.27 的全链路审计脚本首版就这么崩的；同批被扫出的还有
    #  `tests/verify_e2e.py`、`_probe_real_tokens.py`、`diag_grade_retention.py`、
    #  `verify_final_k_ab.py`。）
    #
    # **为什么现在才敢改**：唯一的下游是 `generate`，而它**本来就防御了两种情况**
    # （`documents if isinstance(documents, list) else [documents]`）——
    # 所以改成 list 对它是**零影响**，而所有「假设 list」的代码都能因此工作。
    # [2026-11] 改为「一条搜索结果 = 一个 Document」，不再把所有内容拼成一块。
    #
    # 原先全部拼接后，溯源只能列出一条 `（网络搜索，非知识库）·（网络搜索，非知识库）`——
    # 既看不出引自哪个网页、也无法跳转核对，而且 filename/title 填的是同一个占位符，
    # 渲染出来就是重复的两段文字。
    #
    # 拆成多条后：编号 [1][2] 与网页一一对应，metadata 带真实 title 与 url，
    # `_print_answer` / API 的 Capture 都能展示可点击来源（见 citation.build_citation_meta）。
    docs_out = []
    for item in results:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        url = str(item.get("url", "") or "")
        # 域名比完整 URL 更适合做 filename 列（展示宽度有限）
        host = urlsplit(url).netloc if url else ""
        docs_out.append(
            Document(
                page_content=content,
                metadata={
                    "source": "web",
                    "url": url,
                    "filename": host or "（联网结果）",
                    "title": str(item.get("title", "") or "").strip() or host or "（无标题）",
                },
            )
        )

    if not docs_out:
        log.warning("web_search 结果均无正文，返回空 documents")
        return {"documents": [], "question": question, "route": "web"}

    # [2026-11] 标记本轮走的是联网分支（见 crag_graph.grade_generation 的分支保护）
    return {"documents": docs_out, "question": question, "route": "web"}
