from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.env_utils import LLM_STRUCTURED_METHOD


# 数据模型 - 生成内容幻觉评分
class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""

    binary_score: str = Field(description="回答是否基于事实，取值为'yes'或'no'")


# 带函数调用的LLM初始化
structured_llm_grader = llm.with_structured_output(
    GradeHallucinations, method=LLM_STRUCTURED_METHOD
)  # 绑定结构化输出到评分模型

# 提示词模板
#
# [FIX 2026-11] 原 prompt 只有一句话：「回答是否基于给定事实」。实测判分**过严**：
# 一次完整链路里第二次生成的答案内容准确、带 [1][4] 引用，仍被判为「未基于参考文档」，
# 于是一次提问多跑一轮生成 + 一次改写（多花约 6 秒）。
#
# 原因：模型把「是否逐字复述资料」当成了判据，而正常作答必然包含
# 同义改写、多块归纳、结构化排版 —— 这些都不是幻觉。
#
# 现在把判据写清楚：只惩罚「资料里根本没有的新事实」，不惩罚改写与归纳。
system = """你是事实一致性评估器。判断【生成内容】中的**事实性陈述**能否由【资料】支持。

判 yes（满足任一即可）：
- 陈述能在资料中找到直接依据（允许同义改写、合并、换一种说法）；
- 陈述是对资料中多个片段的归纳或对比，且没有引入资料之外的新事实；
- 生成内容主动标注了「资料中未涉及 X」这类缺口说明。

判 no（满足任一即可）：
- 出现资料里完全没有的**具体**数字、名称、版本号、条目或结论；
- 把资料中的「部分/可能」表述成「全部/必然」；
- 引用编号指向的块与所述内容无关。

注意：语气词、连接词、标题层级、列表排版**不属于事实**，不影响判定。
不要因为「资料没这么写」就给 no —— 只要事实可由资料支持，就是 yes。

只输出 'yes' 或 'no'。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "资料: \n\n {documents} \n\n 生成内容: {generation}"),  # 用户输入模板
    ]
)

# 构建幻觉检测工作流
hallucination_grader_chain = (
    hallucination_prompt  # 使用幻觉检测提示模板
    | structured_llm_grader  # 调用结构化评分的LLM
)
