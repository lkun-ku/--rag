from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_models.all_llm import llm
from utils.citation import build_citation_meta, format_context, validate_citations

# [UPGRADE 2026-10-09 / ROADMAP 2.1] 强约束引用 prompt。
#
# 与原 prompt（「你是一个问答任务助手…如果不知道答案，请直接说明」）的区别：
#   1. 每个事实性句子**必须**标注来源编号 -> 答案可核验
#   2. 明确「禁止补充资料之外的知识」-> 抑制常识性编造
#   3. 拒答文案**固定**为「知识库中未找到依据。」-> 让下游能用字符串精确识别拒答，
#      从而把它与「幻觉」区分开（拒答不该触发熔断重试）
#
# 实测（ROADMAP 2.1.1）：引用标注率 100%（两种 prompt 各跑 2 次、编号一致），
# 拒答行为正确，且 LLM_TEMPERATURE=0 保证跨次稳定。
CITE_PROMPT = """你是一个严格基于给定资料作答的助手。规则：
1. 每个事实性句子后必须标注来源编号，格式为 [1]、[2]，多来源写 [1][3]。
2. 只允许使用给定资料，禁止补充资料之外的知识。
3. 若资料不足以回答，只回复：知识库中未找到依据。
4. 回答简洁，不要重复问题。

资料：
{context}

问题：{question}
回答："""


def format_docs(docs):
    """带编号拼接上下文。

    编号 [1]、[2] 必须与 `documents` 的下标**严格对应** —— 下游校验层
    （graph_2.grade_generation_v_documents_and_question）用它反查原文块，
    并判断模型有没有引用不存在的编号。
    """
    if isinstance(docs, list):
        return format_context(docs)
    return '[1] ' + docs.page_content


def generate(state):
    """
    生成回答
    Args:
        state (dict): 当前图状态，包含问题和检索结果
    Returns:
        state (dict): 更新后的状态，新增包含生成结果的generation字段
    """
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取检索到的文档

    prompt = PromptTemplate(
        template=CITE_PROMPT,
        input_variables=["question", "context"],
    )

    # 构建RAG处理链
    rag_chain = (
            prompt |  # 第一步：使用提示模板
            llm |  # 第二步：调用语言模型
            StrOutputParser()  # 第三步：解析模型输出为字符串
    )

    # RAG生成过程
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})  # 调用RAG链生成回答

    # [ROADMAP 2.1 ⑤] 构造「编号 -> 原文块」映射，供调用方渲染溯源信息。
    # 引用编号的**合法性校验**不在这里做 —— 它属于流程控制（越界要触发熔断重试），
    # 放在 graph_2.grade_generation_v_documents_and_question 里统一处理。
    doc_list = documents if isinstance(documents, list) else [documents]
    cites = validate_citations(generation, len(doc_list))["cites"]
    citations = build_citation_meta(doc_list, cites)

    # 累加生成次数：原先 graph_2.py 中 "not supported" -> generate 是无条件自环，
    # 幻觉检测反复失败会无限循环；该计数供判分函数做熔断判断。
    generation_count = state.get("generation_count", 0) + 1
    return {"documents": documents, "question": question, "generation": generation,
            "generation_count": generation_count, "citations": citations}
