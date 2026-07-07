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
    web_results = Document(page_content=web_results)  # 转换为文档格式

    return {"documents": web_results, "question": question}  # 返回更新状态