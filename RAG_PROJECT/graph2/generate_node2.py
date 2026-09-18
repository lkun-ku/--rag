from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from graph2.grade_documents_node import _sub_config
from llm_models.all_llm import llm
from utils.citation import build_citation_meta, format_context, validate_citations

# [UPGRADE  / ROADMAP 2.1] 强约束引用 prompt。
#
# 与原 prompt（「你是一个问答任务助手…如果不知道答案，请直接说明」）的区别：
#   1. 每个事实性句子**必须**标注来源编号 -> 答案可核验
#   2. 明确「禁止补充资料之外的知识」-> 抑制常识性编造
#   3. 拒答文案**固定**为「知识库中未找到依据。」-> 让下游能用字符串精确识别拒答，
#      从而把它与「幻觉」区分开（拒答不该触发熔断重试）
#
# 实测（ROADMAP 2.1.1）：引用标注率 100%（两种 prompt 各跑 2 次、编号一致），
# 拒答行为正确，且 LLM_TEMPERATURE=0 保证跨次稳定。
# [TRY & REVERT  / P0-1] 曾加过两条约束（「标注前回看该编号对应的块」+
# 「同一资料被切成多块时要选真正含数据的那一块」），试图修 P0-1 的「引用指错块」。
# **实测后回退，原因是「无法证明有效」**：
#
#   运行            库内拒答率       含引用   引用支持率        指错数
#   ------------------------------------------------------------------------
#   旧 prompt r1    9.4%  (3/32)    29      96.55% (28/29)   1
#   新 prompt r1    12.5% (4/32)    -       96.43% (27/28)   1
#   新 prompt r2    12.5% (4/32)    28      100.00% (28/28)  0
#   旧 prompt r2    12.5% (4/32)    28      100.00% (28/28)  0    <- 旧 prompt 也是 12.5%
#
# **关键：旧 prompt 第二轮同样是 12.5%** —— 差异不来自 prompt，只来自**唯一的抖动
# 样本 n005**（「召回率最低的产品」，4 轮里作答 1 次、拒答 3 次；**检索块数恒为 18**
# —— 抖动发生在生成层，不是检索层）。其余 31 题的拒答判定四轮完全一致。
#
# 由于样本仅 29 题、指错仅 1 个，而这一个抖动就能翻转全部指标，**本套评测无法判断
# 该 prompt 改动的效果**。既无法证明有效、又增加 prompt 复杂度 => 回退。
#
# ## 两条更深的方法论教训
#
# 1. **先测「基线自身的抖动」，再谈 A/B 差异。** 我第一轮看到 9.4% -> 12.5% 就判定
#    「拒答率确定上升」，补跑第二轮才发现**基线与实验组同值**。
#    在小样本 + 存在不稳定样本时，单轮对比是不可靠的。
# 2. **n005 是「跨行统计类提问」的真实案例**（问哪一行的值最小）。表格被切成多块后，
#    模型有时能聚合出答案、有时判「资料不足」拒答。这正是 ROADMAP 3.5.1「表格父子块」
#    当初标注的立项条件 ——「**需先出现跨行统计类提问的真实案例**」。**这个案例出现了。**
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


def generate(state, config=None):
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
    #
    # [FIX / ROADMAP 4.29] **显式传 config**，不要依赖 contextvar 隐式继承。
    #
    # 实测（`eval/_probe_contextvar_threadpool.py`，三条对照）：
    #     A. 无线程池 + 不传 config -> token=11   （contextvar 继承，能统计）
    #     B. 线程池   + 不传 config -> token=0    （**线程池内失效**）
    #     C. 线程池   + 显式传 config -> token=22 （显式传递有效）
    #
    # 本节点当前是**串行**执行的，所以靠 A 那条路径侥幸能统计到；
    # 但一旦有人为了提速把它改成并发（线程池），token 会**静默归零** ——
    # 正是 `grade_documents` 首版踩过的坑（判分用了线程池 -> 统计不到 -> 4.22 才发现）。
    # 显式传递把这个隐式依赖消掉。
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question},
                                  config=_sub_config(config))  # 调用RAG链生成回答

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
