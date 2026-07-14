from tools.retriever_tools import get_retriever
from utils.log_utils import log


def retrieve(state):
    """
    检索相关文档
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        state (dict): 更新后的状态，新增包含检索结果的documents字段
    """
    log.info("---去知识库中检索文档---")  # 打印当前阶段标识
    question = state["question"]  # 从状态中获取用户问题
    # [ROADMAP 3.3] 改为 get_retriever()：它会读 ContextVar 里的租户上下文，
    # 有租户时返回只含授权集合的检索器；无租户时返回全局检索器，
    # **行为与改动前完全一致**（单租户模式下零差异）。
    documents = get_retriever().invoke(question)  # 调用检索器获取相关文档
    return {"documents": documents, "question": question}  # 返回更新后的状态
