from tools.retriever_tools import get_retriever
from tools import reranker_tools as rt
from utils.log_utils import log


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

    # [ROADMAP 3.3] get_retriever() 会读 ContextVar 里的租户上下文，
    # 有租户时返回只含授权集合的检索器；无租户时返回全局检索器，
    # **行为与改动前完全一致**（单租户模式下零差异）。
    documents = get_retriever(deep=is_agg).invoke(question)

    # [ROADMAP 4.42] reranker 精排 + 分数门槛。**默认关闭**：
    # RERANK_ENABLED=0 时原样返回，不影响任何既有指标；
    # 企业级部署设 RERANK_ENABLED=1 即可启用（此时它同时承担 conditions ① 和 ④）。
    documents = rt.enterprise_rerank(question, documents)

    return {"documents": documents, "question": question}  # 返回更新后的状态
