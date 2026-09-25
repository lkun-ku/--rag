from typing import Annotated, Any, List, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    表示图处理流程的状态信息

    属性说明：
        question: 用户提出的问题文本
        generation: 语言模型生成的回答文本
        transform_count: 传换查询的次数
        generation_count: 生成回答的次数（用于熔断，避免 generate 自环无限循环）
        documents: 检索到的相关文档列表
        messages: 多轮对话历史（**仅在使用 checkpointer 时累积**）
    """

    question: str  # 存储当前处理的用户问题
    transform_count: int  # 传换查询的次数
    generation_count: int  # 生成回答的次数
    generation: str  # 存储LLM生成的回答内容
    documents: List[Document]  # 存储检索到的文档内容列表
    # [ROADMAP 2.1] 引用溯源：编号 -> 原文块的映射，供调用方渲染「答案依据来自哪里」。
    # 由 generate 节点产出；不参与流程控制（越界校验在判分函数里做）。
    citations: List[dict]
    # [ROADMAP 4.41] 多轮对话历史（阶段 4.41 新增）。
    #
    # `add_messages` 是 **reducer**：节点返回 `{"messages": [新消息]}` 时，
    # LangGraph 会**追加**到已有列表，而不是覆盖 —— 这是做多轮记忆的前提。
    #
    # 四个设计约束：
    #   1. `NotRequired` + 默认空：**单轮调用（不发 messages）时行为与改造前完全一致**；
    #   2. 只在 `generate` 里追加（一问一答两条），其它节点不碰它；
    #   3. **是否真的跨轮保留，取决于有没有挂 checkpointer** ——
    #      `create_graph()`（模块级 `graph`）不挂，所以单轮行为不变；
    #      `get_memory_graph()` 才挂 `SqliteSaver`，按 `thread_id` 持久化；
    #   4. 历史**只在 prompt 里注入最近若干轮**（见 `generate_node2._format_history`），
    #      不是把全部历史塞进去 —— 避免上下文无限增长。
    messages: NotRequired[Annotated[List[AnyMessage], add_messages]]
    # 测试/调试用：记录本次会话的标识（真正隔离由 checkpointer 的 thread_id 负责）
    session_id: NotRequired[str]
    # [ROADMAP 4.43] **更早内容的摘要**：当 messages 超过 26 轮时，
    # `generate` 会把「掉出最近 13 轮的部分」压成一段摘要存这里。
    # 只生成一次、之后复用（不每轮重算，见 `_summarize` 的成本取舍）。
    summary: NotRequired[str]
    # 便于类型标注（Any 仅用于兜底不可预期的附加键）
    _extra: NotRequired[Any]