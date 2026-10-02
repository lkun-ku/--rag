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

    question: str  # 存储当前处理的用户问题（可能已被 transform_query 改写）
    # [FIX 2026-11] **原始问题**：入口写入一次，之后只读不写。
    #
    # 为什么必须有它：`transform_query` 原本把改写结果写回 `question`，下一轮改写
    # 又拿它当输入 —— 每一轮都在上一次的改写结果上再改，几十轮后问题彻底失真。
    # 实测把「我上一个问题问的是是吗」连续改写 30+ 次，`question` 变成一段
    # 「请从对话历史中定位最近一条用户消息…」的元指令，最终被当作检索词原文
    # 送去联网搜索并复述给用户（提示词泄漏）。
    #
    # 现在：改写**永远基于 original_question**，`question` 只承载「本轮检索用的问法」。
    original_question: NotRequired[str]
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
    #   4. 历史**只在 prompt 里注入最近若干轮**（见 `generate_node._format_history`），
    #      不是把全部历史塞进去 —— 避免上下文无限增长。
    messages: NotRequired[Annotated[List[AnyMessage], add_messages]]
    # 测试/调试用：记录本次会话的标识（真正隔离由 checkpointer 的 thread_id 负责）
    session_id: NotRequired[str]
    # [FIX 2026-11] 本轮走的是哪条分支：`web` / `vectorstore` / `direct`。
    #
    # 用途：`grade_generation` 判「答案不合格」时，决定下一步往哪走。
    # 原实现不管哪条分支都返回 `not useful`，而它的边固定指向 `transform_query`
    # -> `retrieve` —— 于是**联网问题的失败会被转去查知识库**：
    # 实测问「今天星期几」时，联网没搜到日期，转去查库拿到 5 块 Milvus 运维文档，
    # 用它们生成了一个带 [1][5] 引用的答案，再判不合格、再转、再查……跑了 5 轮、
    # 28 秒，最后被 recursion_limit 强制中断。
    route: NotRequired[str]
    # [2026-11] 对外契约：本轮的答案**是否有知识库依据**。
    #
    # 语义从原来的二元（答 / 拒答）改为三态：
    #   grounded=True            有检索依据，答案带 [n] 引用
    #   grounded=False           降级作答（通用知识），答案带「非知识库资料」抬头
    #   来自联网                 见 citations 里的 from_web
    #
    # 有了它，调用方不必靠「答案里有没有『未找到依据』这句话」来判断是否可信。
    grounded: NotRequired[bool]
    # `knowledge_base` / `web` / `model_knowledge`（检索失败降级）/ `model_direct`（主动闲聊）
    source_type: NotRequired[str]
    # [ROADMAP 4.43] **更早内容的摘要**：当 messages 超过 26 轮时，
    # `generate` 会把「掉出最近 13 轮的部分」压成一段摘要存这里。
    # 只生成一次、之后复用（不每轮重算，见 `_summarize` 的成本取舍）。
    summary: NotRequired[str]
    # 便于类型标注（Any 仅用于兜底不可预期的附加键）
    _extra: NotRequired[Any]
