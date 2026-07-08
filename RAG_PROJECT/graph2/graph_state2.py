from typing import TypedDict, List

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    表示图处理流程的状态信息

    属性说明：
        question: 用户提出的问题文本
        generation: 语言模型生成的回答文本
        transform_count: 传换查询的次数
        generation_count: 生成回答的次数（用于熔断，避免 generate 自环无限循环）
        documents: 检索到的相关文档列表
    """

    question: str  # 存储当前处理的用户问题
    transform_count: int  # 传换查询的次数
    generation_count: int  # 生成回答的次数
    generation: str  # 存储LLM生成的回答内容
    documents: List[Document]  # 存储检索到的文档内容列表
    # [ROADMAP 2.1] 引用溯源：编号 -> 原文块的映射，供调用方渲染「答案依据来自哪里」。
    # 由 generate 节点产出；不参与流程控制（越界校验在判分函数里做）。
    citations: List[dict]