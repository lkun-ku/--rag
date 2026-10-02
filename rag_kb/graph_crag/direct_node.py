"""直接回答节点（不检索、不判分）。

## 为什么需要它

原工作流是**二选一**：`vectorstore`（查库）或 `web_search`（联网）。任何问题都必须
先检索、再逐块判分（36 块 / 批 8 = 5 次 LLM 调用）、再生成、再做幻觉与匹配度判定 ——
合计约 9 次 LLM 调用。

对「你好」「我刚才问了什么」「帮我写个函数名」这类问题，这套链路：
* **必然检索不到**（知识库里没有对话上下文），于是 0/36 通过 -> 改写 -> 再检索 ->
  仍为 0 -> 转联网 -> 生成 -> 判定不过 -> 改写 …… 直到熔断；
* **答得生硬**：生成节点被要求「只能使用给定资料」，没有资料时只能拒答或硬凑；
* **很慢**：实测一次追问耗时 14 秒（熔断修复前是 213 秒）。

本节点提供第三条路径：**不检索，直接答**。耗时约 1 次 LLM 调用（2~4 秒）。

## 与 RAG 分支的边界

* 知识库主题内的具体问题 -> 仍然走 vectorstore，保留引用溯源；
* 实时信息 -> 仍然走 web_search；
* 只有「不需要外部资料」的请求才走这里（见 `query_route_chain` 的判定规则）。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_kb.graph_crag.grade_documents_node import _sub_config
from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.log_utils import log

_SYSTEM = """你是一个自然、专业的中文助手，直接回答用户的问题。

规则：
1. 用你自己的通用知识作答，语气自然、口语化，像人在对话；不要用「根据资料」
   「资料显示」这类开头 —— 本轮**没有检索任何资料**。
2. **不要输出 [1][2] 这样的引用编号**：没有资料可引，编造编号就是幻觉。
3. 如果问题在追问本次对话的上文（「我刚才问了什么」「上面那条再展开」），
   直接依据下方「对话历史」回答。
4. 涉及你不确定的具体事实（时间、数字、版本、人名），明确说「这点我不确定」，
   不要编造。
5. 需要展开就展开，不要为了显得简短而漏掉关键信息。"""

_HUMAN_WITH_HISTORY = """对话历史：
{history}

用户问题：{question}"""

_HUMAN_PLAIN = """用户问题：{question}"""

# 降级作答的固定抬头：必须让用户一眼看出「这不是知识库里的内容」。
FALLBACK_NOTICE = "【以下内容来自模型通用知识，非知识库资料】"

_SYSTEM_FALLBACK = """用户的问题在内部知识库中没有找到依据。请用你自己的通用知识给出一个**有帮助的回答**。

规则：
1. **必须在回答开头原样输出这一行**：`【以下内容来自模型通用知识，非知识库资料】`
   然后再另起一段回答，不要把它混进句子里。
2. 不要编造「资料显示」「知识库提到」这类来源 —— 本轮没有任何检索结果。
3. 不要输出 [1][2] 这样的引用编号。
4. 如果你确实不知道，就直接说不知道，不要硬凑。
5. 其余部分保持自然、口语化，像人在解释问题。"""


def generate_direct(state, config=None, *, mode: str = "direct"):
    """不检索，直接生成回答。

    Args:
        state: 图状态。`session_id` 存在时注入对话历史并回写 `messages`。
        config: LangGraph 运行配置，**必须透传** —— 否则本节点的 token 统计归零
            （与 `generate` / `grade_documents` 是同一个坑）。
        mode: `direct` = 用户主动闲聊/追问（不加标记）；
              `fallback` = **检索失败后的降级作答**（强制加「非知识库」抬头，
              并置 `grounded=False`）。

    Returns:
        dict: `generation` / `documents`（空）/ `citations`（空）/
              `grounded` / `source_type` 等增量。
    """
    log.info("---DIRECT ANSWER（不检索，mode=%s）---" % mode)
    # 用原始问题：本分支不经过 transform_query
    question = state.get("original_question") or state["question"]

    history_text = ""
    if state.get("session_id"):
        # 延迟 import，避免与 generate_node 形成模块级互相依赖
        from rag_kb.graph_crag.generate_node import _format_history

        history_text = _format_history(state.get("messages") or [])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_FALLBACK if mode == "fallback" else _SYSTEM),
            ("human", _HUMAN_WITH_HISTORY if history_text else _HUMAN_PLAIN),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    payload = {"question": question}
    if history_text:
        payload["history"] = history_text

    try:
        answer = (chain.invoke(payload, config=_sub_config(config)) or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("直接回答失败：%s: %s" % (type(e).__name__, str(e)[:100]))
        answer = "抱歉，这次没能生成回答，请重试或换个问法。"

    if not answer:
        answer = "抱歉，这次没能生成回答，请重试或换个问法。"

    # 降级模式：**代码强制加抬头**，不依赖模型是否听话。
    # 提示词里要求了，但模型偶尔会漏 —— 这是「是否误导用户」的合规问题，
    # 不能指望概率性服从。
    if mode == "fallback" and FALLBACK_NOTICE not in answer:
        answer = FALLBACK_NOTICE + "\n\n" + answer

    updates = {
        "generation": answer,
        # 本分支没有检索：显式清空，避免上游残留的 documents 被下游误当作依据
        "documents": [],
        "citations": [],
        # [2026-11] 对外契约：是否有知识库依据、来源类型。
        # 前端/调用方据此决定是否展示「非知识库」标记。
        "grounded": False,
        "source_type": "model_knowledge" if mode == "fallback" else "model_direct",
        "generation_count": state.get("generation_count", 0) + 1,
    }
    # 与 generate 一致：记忆模式下把「问 + 答」成对追加，供下一轮解析指代
    if state.get("session_id"):
        updates["messages"] = [HumanMessage(content=question), AIMessage(content=answer)]
    return updates


def generate_fallback(state, config=None):
    """检索失败后的**降级作答**（图节点入口）。

    与 `generate_direct` 同一实现，区别只在 `mode="fallback"`：
    回答开头强制加「非知识库资料」抬头，并置 `grounded=False`。
    """
    return generate_direct(state, config, mode="fallback")
