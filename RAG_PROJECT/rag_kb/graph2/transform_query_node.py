from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph2.grade_documents_node import _sub_config
from llm_models.all_llm import llm
from utils.log_utils import log


def transform_query(state, config=None):
    """
    优化用户问题，生成更合适的查询语句

    Args:
        state (dict): 当前图状态，包含用户问题和检索结果

    Returns:
        state (dict): 更新后的状态，question字段替换为优化后的问题
    """
    log.info("---TRANSFORM QUERY---")  # 阶段标识
    question = state["question"]  # 获取原始问题
    documents = state["documents"]  # 获取当前文档
    transform_count = state.get("transform_count", 0)

    # 提示词模板 - 问题重写优化
    system = """作为问题重写器，您需要将输入问题转换为更适合向量数据库检索的优化版本。\n
         请分析输入问题并理解其背后的语义意图/真实含义。"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),  # 系统角色设定：问题优化器
            (
                "human",  # 用户输入模板
                "这是初始问题: \n\n {question} \n 请生成一个优化后的问题。",
            ),
        ]
    )

    # 构建问题重写处理链
    question_rewriter = (
        re_write_prompt  # 使用优化提示模板
        | llm  # 调用语言模型
        | StrOutputParser()  # 将输出解析为字符串
    )

    # 问题重写
    #
    # [FIX / 代码审查] **显式传 config** —— 这是同类坑的**第三处**。
    # 4.29.3 我当时写「全项目未显式传 config 的 LLM 调用只剩两处」，
    # 漏了这个文件（它不在我当时 grep 的 `chain.invoke(` 命中里，
    # 因为变量名叫 `question_rewriter`）。教训升级：
    # **grep 要按「调用形状」而非具体变量名**（`\.invoke\(` + 人工过一遍）。
    better_question = question_rewriter.invoke({"question": question}, config=_sub_config(config))  # 调用问题重写器
    return {"documents": documents, "question": better_question, "transform_count": transform_count + 1}  # 返回更新状态
