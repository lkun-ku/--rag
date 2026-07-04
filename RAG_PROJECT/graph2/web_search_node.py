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
    results = docs[0] if isinstance(docs, tuple) else docs
    web_results = "\n".join([d["content"] for d in results])  # 合并搜索结果
    web_results = Document(page_content=web_results)  # 转换为文档格式

    return {"documents": web_results, "question": question}  # 返回更新状态