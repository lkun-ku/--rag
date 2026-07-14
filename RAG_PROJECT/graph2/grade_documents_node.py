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

from graph2.grader_chain import retrieval_grader_chain
from utils.log_utils import log

# 并发度上限。判分是网络 IO 密集，适度并发即可显著缩短总耗时；
# 取 8 是因为 18 块在 8 并发下约 3 轮完成，再高收益递减且更容易触发 API 限流。
MAX_GRADE_WORKERS = 8


def _grade_one(question: str, document):
    """对单个文档判分。

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
            {"question": question, "document": document.page_content}
        )
        return document, str(getattr(score, "binary_score", "")).strip().lower() == "yes"
    except Exception as e:  # noqa: BLE001
        log.warning(f"---GRADE: 判分失败，保守保留该块: {type(e).__name__}: {str(e)[:80]}---")
        return document, True


def grade_documents(state):
    """评估检索到的文档与问题的相关性。

    Args:
        state (dict): 当前图状态，包含问题与检索结果。

    Returns:
        dict: `documents` 仅保留被判为相关的文档（顺序与输入一致）。
    """
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取待评估文档

    if not documents:
        return {"documents": [], "question": question}

    # 文档评分与过滤（并发；executor.map 保序）
    workers = max(1, min(MAX_GRADE_WORKERS, len(documents)))
    if workers == 1:
        graded = [_grade_one(question, d) for d in documents]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            graded = list(ex.map(lambda d: _grade_one(question, d), documents))

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
