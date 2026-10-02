from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_kb.graph_crag.grade_documents_node import _sub_config
from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.log_utils import log

# 改写结果的合理长度上限。
# 正常改写是一句话（本项目中位 ~30 字）；超过这个长度基本是把 prompt 条款
# 或解释文字复述回来了 —— 直接判为改写失败、退回原问题。
_MAX_REWRITE_CHARS = 160

# 模型偶尔会给改写结果加前缀，这里统一剥掉（否则会当作检索词的一部分）。
_PREFIXES = (
    "优化后的问题：",
    "优化后的问题:",
    "改写后的问题：",
    "改写后的问题:",
    "独立问句：",
    "独立问句:",
    "问题：",
    "问题:",
)

_SYSTEM = """你是一个问题重写器。把用户的提问改写成**一句可直接用于向量检索的独立问句**。

规则：
1. 若问题里含有指代（「它」「这个」「那第二点」「上一个问题」等），结合对话历史还原出具体所指；
2. 只输出**一行改写后的问句**，不要解释、不要加「优化后的问题：」这类前缀；
3. 不要回答问题，也不要引入对话历史里没有的信息；
4. 若无法确定指代，就原样返回用户问题。"""

_HUMAN_WITH_HISTORY = """对话历史（仅用于理解指代）：
{history}

用户问题：{question}

改写后的独立问句："""

_HUMAN_PLAIN = """用户问题：{question}

改写后的独立问句："""


def _clean(text: str) -> str:
    """取首行、剥前缀、去引号。"""
    s = (text or "").strip()
    if not s:
        return ""
    # 有的模型会输出「问句\n\n解释」，只取第一段非空行
    s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    s = s.strip('"').strip("'").strip("「").strip("」")
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p) :].strip()
            break
    return s


def transform_query(state, config=None):
    """把用户问题改写成更适合向量检索的自包含问句。

    Args:
        state (dict): 当前图状态。

    Returns:
        dict: `question` 换成改写结果（仅本轮检索用），`transform_count` 累加。

    [FIX 2026-11] 两处根本性修改：

    1. **永远基于 `original_question` 改写**，不拿上一次的改写结果当输入。
       原实现每轮覆盖 `question`，下一轮又在它之上改写 —— 连续改写会把问题
       逐轮"漂移"成一段元指令，实测 30+ 次后检索词变成
       「请从对话历史中定位最近一条用户消息…」，并被泄漏进最终答案。

    2. **注入对话历史**。像「我上一个问题问的是是吗」这类追问，本身不含检索
       所需的实体，不结合历史改写的话必然检索不到任何东西（实测 0/36）。
    """
    log.info("---TRANSFORM QUERY---")
    original = state.get("original_question") or state["question"]
    documents = state["documents"]
    transform_count = state.get("transform_count", 0)

    history_text = ""
    if state.get("session_id"):
        # 延迟 import：generate_node 体量较大，且两者不应在模块级互相依赖
        from rag_kb.graph_crag.generate_node import _format_history

        history_text = _format_history(state.get("messages") or [])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", _HUMAN_WITH_HISTORY if history_text else _HUMAN_PLAIN),
        ]
    )
    question_rewriter = prompt | llm | StrOutputParser()

    payload = {"question": original}
    if history_text:
        payload["history"] = history_text

    try:
        raw = question_rewriter.invoke(payload, config=_sub_config(config))
        better = _clean(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("问题改写失败，沿用原问题：%s: %s" % (type(e).__name__, str(e)[:80]))
        better = ""

    # 改写失败/空/过长/等于原句 -> 退回原问题，绝不让脏结果进入检索
    if not better or len(better) > _MAX_REWRITE_CHARS or better == original:
        better = original
    else:
        log.info("问题改写：%s -> %s" % (original[:40], better[:80]))

    return {"documents": documents, "question": better, "transform_count": transform_count + 1}
