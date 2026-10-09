from rag_kb.tools.retriever_tools import get_retriever
from rag_kb.tools import reranker_tools as rt
from rag_kb.utils.log_utils import log


def retrieve(state, config=None):
    """
    检索相关文档
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        state (dict): 更新后的状态，新增包含检索结果的documents字段
    """
    log.info("---去知识库中检索文档---")  # 打印当前阶段标识
    question = state["question"]  # 从状态中获取用户问题

    # [ROADMAP 4.42] 聚合类查询（「哪个召回率最低」）走**深检索**：
    # 这类问题要跨行比较，必须看到完整表格，粗排每集合 12 块不够用。
    # 代价是粗排多取一批，但由下游 reranker 收敛（若未启用 reranker，
    # 则由既有的 LLM 判分继续过滤，最终仍只留 final_k 附近）。
    is_agg = rt.is_aggregation_query(question)
    if is_agg:
        log.info("---识别为聚合类查询，启用深检索---")

    documents = get_retriever(deep=is_agg).invoke(question)

    # [ROADMAP 4.42] reranker 精排 + 分数门槛。**默认关闭**：
    # RERANK_ENABLED=0 时原样返回，不影响任何既有指标；
    # 企业级部署设 RERANK_ENABLED=1 即可启用（此时它同时承担 conditions ① 和 ④）。
    documents = rt.enterprise_rerank(question, documents)

    # [2026-11] 标记本轮走的是知识库分支。`grade_generation` 据此判断
    # 「判分不通过时该不该转去改写/重查」—— 联网分支不能转到这里（见 crag_graph）。
    return {"documents": documents, "question": question, "route": "vectorstore"}
