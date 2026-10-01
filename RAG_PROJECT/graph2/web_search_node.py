from langchain_core.documents import Document

from llm_models.all_llm import web_search_tool
from utils.log_utils import log


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
    docs = web_search_tool.invoke({"query": question})  # 调用网络搜索工具
    # 修复：TavilySearchResults 的 response_format 默认为 "content_and_artifact"，
    # invoke() 返回的是 (clean_results, raw_results) 元组，而下面按 list 迭代取
    # d["content"] 会对 list 取下标而抛 TypeError（0.3.21 与 0.4.2 行为一致，
    # 故这是项目既有缺陷）。这里只做正确的解包，取值逻辑保持不变。
    # 工具可能返回三种形态，必须逐一处理（后两种是实测踩到的）：
    #   1) (list, raw) 元组 —— response_format='content_and_artifacts' 的正常返回
    #   2) list         —— 直接就是结果列表
    #   3) str          —— 工具执行出错时，langchain 会把异常信息包成字符串返回
    #                      （例如 Tavily key 无效时会拿到 "HTTPError('401 ...')"）
    if isinstance(docs, tuple):
        results = docs[0]
    elif isinstance(docs, list):
        results = docs
    else:
        log.warning(f"web_search 未返回结果列表，收到 {type(docs).__name__}，按原文使用")
        results = docs

    if isinstance(results, str):
        web_results = results
    else:
        web_results = "\n".join([d["content"] for d in results])  # 合并搜索结果
    # [FIX / P0-9] 必须标记「来源是网络搜索」，不能只建成裸 Document。
    #
    # 原因：这条分支的结果会与知识库检索结果走**同一条**编号化路径
    # （`format_context` -> `[1][2]...`，见 generate_node2），并被
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
    #  `eval/verify_e2e.py`、`_probe_real_tokens.py`、`diag_grade_retention.py`、
    #  `verify_final_k_ab.py`。）
    #
    # **为什么现在才敢改**：唯一的下游是 `generate`，而它**本来就防御了两种情况**
    # （`documents if isinstance(documents, list) else [documents]`）——
    # 所以改成 list 对它是**零影响**，而所有「假设 list」的代码都能因此工作。
    web_results = [
        Document(
            page_content=web_results,
            metadata={
                "source": "web",
                "filename": "（网络搜索，非知识库）",
                "title": "（网络搜索，非知识库）",
            },
        )
    ]

    return {"documents": web_results, "question": question}  # 返回更新状态
