from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag_kb.graph.get_human_message import get_last_human_message  # pyright: ignore[reportImplicitRelativeImport]
from rag_kb.graph2.generate_node2 import CITE_PROMPT  # pyright: ignore[reportImplicitRelativeImport]
from rag_kb.llm_models.all_llm import llm  # pyright: ignore[reportImplicitRelativeImport]
from rag_kb.utils.citation import parse_citations, validate_citations  # pyright: ignore[reportImplicitRelativeImport]
from rag_kb.utils.log_utils import log  # pyright: ignore[reportImplicitRelativeImport]


def generate(state):
    """
    生成答案

    参数:
        state (messages): 当前状态

    返回:
        dict: 包含重述问题的更新后状态

    [UPGRADE  / ROADMAP 2.1] 与 graph2 共用同一套引用规范：

    - **prompt 复用** `graph2.generate_node2.CITE_PROMPT`。两个工作流若各写一份，
      日后极易漂移成不同行为；共用一份定义可杜绝这种不一致。
    - **上下文已编号化**：`tools.retriever_tools.retriever_tool` 在工具层就产出
      `[1] ... [2] ...`，因此这里原样传入即可（此节点拿到的已是拼接字符串，
      块边界已丢失，**无法**在此处再编号）。
    - **校验范围说明**：graph1 的拓扑是 `generate -> END`，**没有熔断重试环路**。
      因此这里只解析「引用了哪些编号」并记录，**不做越界重试** ——
      重试逻辑属于 graph2（它有 `not supported -> generate` 回边）。
      graph1 若也要越界拦截，需要新增回边，属于拓扑变更，不在本次范围。
    """
    log.info("---生成最终的答案---")
    messages = state["messages"]
    question = get_last_human_message(messages).content
    last_message = messages[-1]

    docs = last_message.content  # 已编号化的上下文（来自检索工具）

    # 提示模板（与 graph2 共用，保证两个工作流行为一致）
    prompt = PromptTemplate(
        template=CITE_PROMPT,
        input_variables=["question", "context"],
    )
    # 处理链
    rag_chain = prompt | llm | StrOutputParser()
    # 执行
    response = rag_chain.invoke({"context": docs, "question": question})

    cites = parse_citations(response)
    is_refusal = validate_citations(response, n_docs=0)["is_refusal"]
    log.info(f"---答案引用的来源编号: {cites}---")
    if is_refusal:
        log.info("---模型按规则拒答（资料未找到依据）---")

    ai_message = AIMessage(
        content=response,
        additional_kwargs={"citations": [{"n": n} for n in cites]},
    )
    return {"messages": [ai_message]}
