"""文档相关性判分节点（CRAG 的 filtering 环节）。

[UPGRADE 2026-10-10 / ROADMAP 3.1.3] 由**串行**改为**并发**判分。

## 为什么要改

3.1 的 tracer 实测：该节点占单次问答总耗时的 **68%**（10.83s / 15.86s）。
原因在实现方式 —— 原代码对 documents **串行遍历**，每块一次 LLM 调用：

    for d in documents:                       # 18 块
        score = retrieval_grader_chain.invoke(...)   # 18 次串行网络往返

18 次串行 LLM 调用累计约 10.8s，而这一步本身是**相互独立**的
（每块只看「自己 + 同一个问题」，块之间无依赖）—— 天然适合并发。

## 为什么用线程池而不是 asyncio

LangGraph 节点签名是同步的 `fn(state)`，改成 async 需要连带调整图的调用方式。
`retrieval_grader_chain.invoke` 是**网络 IO 密集**（LLM API），
线程池足以让等待时间重叠；实现改动小、风险低。

## 行为保持不变的保证

- **判分逻辑与阈值不变**：仍是 `retrieval_grader_chain` 的 yes/no 二元判定
- **顺序不变**：`executor.map` 保证结果顺序与输入一致
- **失败即视为不相关**：单块判分异常时按原语义丢弃该块（不中断整体）
- **空输入短路**：无文档时不做任何调用，与原来一致
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph2.grader_chain import retrieval_grader_chain
from llm_models.all_llm import llm
from utils.env_utils import LLM_STRUCTURED_METHOD
from utils.log_utils import log

# 并发度上限。判分是网络 IO 密集，适度并发即可显著缩短总耗时；
# 取 8 是因为 18 块在 8 并发下约 3 轮完成，再高收益递减且更容易触发 API 限流。
MAX_GRADE_WORKERS = 8


def _sub_config(config):
    """从 LangGraph 传来的 config 里取出**判分链需要的那部分**。

    [FIX 2026-10-16 / ROADMAP 4.20.7] 为什么需要这个：

    判分是**最大**的 LLM 调用方（`FINAL_K=36` 时 36 次/题，占调用次数约 90%），
    但它的 token **不被 `run_with_trace` 的 tracer 统计** —— 因为节点内部手动
    `retrieval_grader_chain.invoke(...)` 时没传 config，而上层的 callbacks 是经
    `graph.stream(..., config={"callbacks": ...})` 下传的，**只覆盖 LangGraph
    直接调度的 Runnable**，节点函数内手动的 `.invoke()` 拿不到。

    后果：**生产成本可见性缺失** —— 最大的成本项在监控里不可见，
    导致任何「降成本」的改动都无法验证效果。

    为什么只取 `callbacks`（而不是整个 config 透传）：
    LangGraph 的 config 里还有 `__pregel_send`、`__pregel_scratchpad` 等
    内部对象（deque / 函数 / 弱引用），原样传进 LangChain 的 `.invoke()`
    会带来无谓的复制与潜在序列化问题。判分链只需要 callbacks。
    """
    if not config:
        return None
    cbs = config.get("callbacks") if isinstance(config, dict) else None
    return {"callbacks": cbs} if cbs else None


def _grade_one(question: str, document, config=None):
    """对单个文档判分。

    Args:
        question: 用户问题。
        document: 待判分的块。
        config: LangGraph 传来的运行配置（可选）。**只用于透传 callbacks**，
            让上层 tracer / 监控能统计到判分的 token 消耗（见 `_sub_config`）。
            传 None 时行为与改造前**完全一致**（不触发任何回调）。

    Returns:
        (document, is_relevant)。

    **异常时返回 (document, True)（保留）** —— 这一点与原实现不同，是有意为之：
        * 判 `no` 是 LLM **明确说**「不相关」-> 按 CRAG 语义丢弃
        * 抛异常是**无法判定**（API 抖动、超时、限流），此时保留更保守 ——
          宁可多留一块让下游的 generate 看到，也不要因为一次网络抖动
          把正确块静默丢掉（那会直接表现为「知识库中未找到依据」的误拒）。
        * 原实现没有异常处理，一旦某块判分抛错会**中断整个节点**，
          比「保留」更糟。

    为什么这里必须考虑抖动：实测同一批 18 块、同一问题，
    串行与并发两次判分结果完全一致（0 异常），但**不同时刻**的判定会变化
    （同一问题曾得到「保留 3 块」与「保留 0 块」两种结果）——
    说明判分本身存在服务端不确定性，异常兜底因此是必要的。
    """
    try:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": document.page_content},
            config=_sub_config(config),
        )
        return document, str(getattr(score, "binary_score", "")).strip().lower() == "yes"
    except Exception as e:  # noqa: BLE001
        log.warning(f"---GRADE: 判分失败，保守保留该块: {type(e).__name__}: {str(e)[:80]}---")
        return document, True


def grade_documents(state, config=None):
    """评估检索到的文档与问题的相关性。

    Args:
        state (dict): 当前图状态，包含问题与检索结果。
        config: LangGraph 注入的运行配置（含 callbacks / metadata）。
            **[FIX 2026-10-16] 必须接收并透传** —— 判分是最大的 LLM 调用方，
            不透传则它的 token 消耗在生产监控里不可见（见 `_grade_one`）。

    Returns:
        dict: `documents` 仅保留被判为相关的文档（顺序与输入一致）。
    """
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取待评估文档

    if not documents:
        return {"documents": [], "question": question}

    # [ROADMAP 4.23] 批量判分（默认开启）—— 实测省 40% token、86% 调用次数，
    # 且**不损伤质量**（6 题 / 215 块实测：16 个判定分歧**全部落在非 golden 块**上，
    # 其中唯一的 golden 分歧还是「批量把逐块误杀的答案块救回来」）。
    # 见 `eval/measure_grade_batch_saving.py`；开关与理由见 `USE_BATCH_GRADE`。
    if USE_BATCH_GRADE:
        verdicts = grade_batch(question, documents, config)
        filtered_docs = [d for d, ok in zip(documents, verdicts) if ok]
        kept, total = len(filtered_docs), len(documents)
        log.info(f"---GRADE(batch): 判分完成 {kept}/{total} 块保留"
                 f"（批大小 {BATCH_SIZE}，共 {-(-total // BATCH_SIZE)} 次调用）---")
        return {"documents": filtered_docs, "question": question}

    # 文档评分与过滤（并发；executor.map 保序）
    workers = max(1, min(MAX_GRADE_WORKERS, len(documents)))
    if workers == 1:
        graded = [_grade_one(question, d, config) for d in documents]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            graded = list(ex.map(lambda d: _grade_one(question, d, config), documents))

    filtered_docs = []
    for doc, ok in graded:
        if ok:
            log.info("---GRADE: 打印相关标识---")
            filtered_docs.append(doc)
        else:
            log.info("---GRADE: 打印不相关标识,并丢掉doc---")

    log.info(f"---GRADE: 判分完成 {len(filtered_docs)}/{len(documents)} 块保留"
             f"（并发度 {workers}）---")
    return {"documents": filtered_docs, "question": question}  # 返回仅含相关文档的状态


# ===========================================================================
# 批量判分（ROADMAP 4.23）—— 一次 LLM 调用判多块，压掉判分的成本
# ===========================================================================
#
# ## 为什么需要它
#
# 4.22 实测：判分占单题真实成本的 **约 76%**（41,868 token 里约 31K），
# 根因是**逐块调用** —— `FINAL_K=36` 时每块一次 LLM 请求，
# **system prompt 与问题被重复发送 36 遍**。
#
# ## 与逐块版的关系
#
# 两者**语义必须一致**（同一批块给出同样的 yes/no）。
# 是否切换由 `USE_BATCH_GRADE` 控制 —— **先实测一致率与节省，达标才默认打开**。
#
# ## 设计要点
#
# * **编号提交**：块按 `【块 N】` 编号，模型按 index 回填，避免顺序错位；
# * **保守默认 True**：判定缺失或整批异常时**保留**该块 —— 与 `_grade_one`
#   的异常兜底语义一致（宁可多留，不要因一次抖动把正确块静默丢掉）；
# * **分批**：`BATCH_SIZE=8` —— 太小省得少，太大则单次请求的 token 与
#   「块间干扰」风险都上升；8 与 `MAX_GRADE_WORKERS` 一致，便于对照；
# * **只取 callbacks** 传下去（与 `_sub_config` 同理，LangGraph 的 config
#   里含 `__pregel_send` 等内部对象）。


class _BlockVerdict(BaseModel):
    """单个块的判定。"""

    index: int = Field(description="块的编号，从 1 开始，与资料块的编号一致")
    relevant: bool = Field(description="该块是否与用户问题相关")


class _BatchVerdicts(BaseModel):
    """整批判定。"""

    verdicts: List[_BlockVerdict] = Field(
        description="每个块的判定，**必须覆盖所有块且每个块只出现一次**")


_BATCH_SYSTEM = """你是一个评估检索文档与用户问题相关性的评分器。
下面会给出若干个**已编号**的资料块，请**逐块**判断它与用户问题是否相关。

判断标准：只要该块包含与用户问题相关的关键词或语义含义，即评为相关。
不需要非常严格的测试，目的是过滤掉错误的检索结果。

要求：
1. **必须对每一个块都给出判定**，不能遗漏、不能合并、不能新增；
2. 每个块的判定相互独立，不要因为别的块判了「相关」就跟着判「相关」；
3. 输出里 index 必须与资料块的编号一一对应。"""

_BATCH_HUMAN = """用户问题：
{question}

待判定的资料块（共 {n} 个）：

{blocks}"""

_batch_chain = None


def _get_batch_chain():
    """惰性构造批量判分链（模块级缓存，避免每次重复构造）。"""
    global _batch_chain
    if _batch_chain is None:
        structured = llm.with_structured_output(_BatchVerdicts,
                                                method=LLM_STRUCTURED_METHOD)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _BATCH_SYSTEM),
            ("human", _BATCH_HUMAN),
        ])
        _batch_chain = prompt | structured
    return _batch_chain


# 每批的块数。8 与 `MAX_GRADE_WORKERS` 一致，便于与逐块版对照。
BATCH_SIZE = 8

# 是否在生产链路里启用批量判分。
#
# [启用 2026-10-16 / ROADMAP 4.23] 由 `eval/measure_grade_batch_saving.py` 实测后开启：
#
#   规模: 6 题 / 215 块（对比同一批块的「逐块判分」与「批量判分」）
#
#   成本（实测，非推算）:
#     total token  173,265 -> 104,378   **省 40%**
#     调用次数        215 -> 30        **省 86%**
#     单题         28,878 -> 17,396 token/题（省 11,481）
#
#   质量:
#     一致率 199/215 = 92.6%
#     **16 个分歧块全部落在「非 golden 块」之外**（逐块核对过）——
#     唯一一个 golden 分歧是 n003 块12，方向为「逐块 False -> 批量 True」，
#     即**批量把逐块误杀的答案块救回来了**（正向）。
#     判分后保留块数 9.7 -> 10.7 块/题（更松 -> 不会漏答案块，代价是生成层略增）。
#
# 为什么敢开：**省的是「重复发送的 system prompt + 问题」**（36 次 -> 5 次），
# 块内容一个字没改 —— 所以它的质量风险本质上只有「块间干扰」，
# 而实测显示干扰只影响噪声块的取舍。
USE_BATCH_GRADE = True


def _format_blocks(docs) -> str:
    """把块按 `【块 N】` 编号后拼成一段（编号是回填 index 的唯一依据）。"""
    return "\n\n".join(
        "【块 %d】\n%s" % (i, (d.page_content or "")) for i, d in enumerate(docs, 1)
    )


def grade_batch(question: str, documents, config=None,
                batch_size: int = BATCH_SIZE,
                max_workers: int = MAX_GRADE_WORKERS) -> List[bool]:
    """批量判分：一次 LLM 调用判 `batch_size` 个块，返回每块是否相关。

    Args:
        question: 用户问题。
        documents: 待判分的块（任意数量，内部会分批）。
        config: LangGraph 传来的运行配置（只取 callbacks 透传，见 `_sub_config`）。
        batch_size: 每批块数。
        max_workers: **批间并发度**。

    Returns:
        `List[bool]`，长度 == `len(documents)`，**顺序与输入一致**。

    契约（由 `eval/test_grade_batch.py` 覆盖）：
        * 长度恒等于输入块数（结构，确定性）；
        * 明显相关的块 True、明显无关的 False（语义，用极端样本验证）；
        * 单块判定缺失/整批异常时**保守保留（True）**，不中断；
        * **多批时顺序与结果不变**（并发的正确性，C4 保护性测试）。

    ## 批间并发（ROADMAP 4.24）

    首版是**串行分批**（`for start in range(...)`），实测端到端耗时因此从 7.2s
    升到 8.1s —— 5 批依次发，而逐块版是 8 并发。

    改为并发后有**两个收益**：

    1. **耗时**：判分阶段 ≈ 单批时间（~0.9s）而非「批数 × 单批时间」；
    2. **可扩展性**（更重要）：耗时从「**随块数线性增长**」变为
       「`ceil(批数 / 并发度)` × 单批时间」—— 即**块数再涨也只多几轮，
       不会线性劣化**。这对「语料/配额继续放大」的场景是必需的：
       例如 `final_k` 到 72 时，串行要 9 批（~8s），并发 8 只要 2 轮（~1.8s）。

    `executor.map` 保证结果顺序与输入一致（与逐块版同一套保证）。
    """
    docs = list(documents or [])
    if not docs:
        return []

    # 默认 True（保守保留）—— 与 _grade_one 的异常兜底语义一致
    out: List[bool] = [True] * len(docs)
    chain = _get_batch_chain()
    size = max(1, int(batch_size or 1))
    batches = [(s, docs[s:s + size]) for s in range(0, len(docs), size)]

    def _run_batch(item):
        """跑一批。返回 (start, chunk, result_or_None, error_or_None)。"""
        start, chunk = item
        try:
            res = chain.invoke(
                {"question": question, "n": len(chunk), "blocks": _format_blocks(chunk)},
                config=_sub_config(config),
            )
            return start, chunk, res, None
        except Exception as e:  # noqa: BLE001
            return start, chunk, None, e

    workers = max(1, min(int(max_workers or 1), len(batches)))
    if workers == 1:
        results = [_run_batch(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_batch, batches))

    for start, chunk, res, err in results:
        no = start // size + 1
        if err is not None:
            log.warning(f"---GRADE(batch): 第 {no} 批判分失败，"
                        f"保守保留该批 {len(chunk)} 块: {type(err).__name__}: {str(err)[:80]}---")
            continue

        verdicts = getattr(res, "verdicts", None) or []
        if not verdicts:
            log.warning(f"---GRADE(batch): 第 {no} 批未返回任何判定，"
                        f"保守保留该批 {len(chunk)} 块---")
            continue

        got = 0
        for v in verdicts:
            try:
                idx = int(getattr(v, "index", 0))
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(chunk):
                out[start + idx - 1] = bool(getattr(v, "relevant", True))
                got += 1
        if got < len(chunk):
            log.warning(f"---GRADE(batch): 第 {no} 批只返回 {got}/"
                        f"{len(chunk)} 个判定，缺失的块保守保留---")

    return out
