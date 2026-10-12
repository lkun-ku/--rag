import os

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from rag_kb.utils.log_utils import log
from rag_kb.tools import history_store
from rag_kb.tools.section_merge import merge_section_blocks

from rag_kb.graph_crag.grade_documents_node import _sub_config
from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.citation import build_citation_meta, format_context, is_refusal, validate_citations

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
2. 只能使用给定资料，禁止补充资料之外的知识（包括你自己的先验知识）。
3. 资料与问题相关时，**能答多少就答多少**：按资料实际覆盖的范围组织回答，
   不要因为资料不够全面、不够「详解」就整体拒答。
4. 若资料只覆盖问题的一部分，先回答已覆盖的部分，再用一句话说明「资料中未涉及 X」，
   不要编造未覆盖的内容。
5. 只有当资料与问题**完全无关**（问的是 A，资料讲的却是 B）时，才只回复：
   知识库中未找到依据。
6. 回答自然、有信息量，符合正常问答的语感；不要机械复述问题，也不要用
   「根据资料」这类套话开头。

资料：
{context}

问题：{question}
回答："""


def _is_table_text(text: str) -> bool:
    """是否包含 markdown 表格（至少 2 行以 `|` 开头）。"""
    return sum(1 for line in text.splitlines() if line.strip().startswith("|")) >= 2


def _start_line(text: str) -> int:
    """切片形如 `表格：…（第 31-60 行）`，取起始行号用于排序。"""
    import re

    m = re.search(r"第\s*(\d+)\s*-\s*(\d+)\s*行", text)
    return int(m.group(1)) if m else 10**9


def merge_table_slices(docs, max_chars: int = 20000):
    """把**同一张表**的多个切片拼回完整表（ROADMAP 4.40）。

    为什么需要：`eval/_exp_n005_table.py` 实测（各 3 次）——

        A 生产上下文（9 块 / 16,182 字符，含 2 个表格碎片）: **3/3 稳定拒答**
        B 完整表   （1 块 /  8,614 字符 / 91 行）          : **2/3 答对**

    同一张表被切成 4 片后，模型看不到完整表，**无法确认「哪个最低」**，
    于是按 prompt 规则 3 保守拒答。而 `generate_node.py` 的注释早就点名
    n005 是「跨行统计类提问」的真实案例（ROADMAP 3.5.1 的立项条件）。

    合并规则（严格，宁可不合并也不合并错）：

    1. 分组键 = `(filename, title)`；
    2. **组内每一块都必须是表格**才合并（有一块是纯文本就整组不动）；
    3. 按切片自带的「第 N-M 行」**起始行号升序**拼接；
    4. **表头与分隔行只保留一次**；`表格：…` / `列名：…` 元信息行丢弃；
    5. 合并后**超过 `max_chars` 就放弃**（返回原样）—— 避免拼出超长上下文；
    6. 合并块的 metadata 取组内第一块的。

    Args:
        docs: `list[Document]`（或单个 Document —— 原样返回）。
        max_chars: 合并结果字符上限。

    Returns:
        list[Document]：合并后的列表；不满足条件时原样返回。
    """
    if not isinstance(docs, list) or len(docs) < 2:
        return docs

    # 按首次出现顺序分组
    groups: dict[tuple, list[tuple[int, Document]]] = {}
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", None) or {}
        key = (md.get("filename"), md.get("title"))
        groups.setdefault(key, []).append((i, d))

    out: list[tuple[int, Document]] = []
    for _key, items in groups.items():
        # [FIX / 演示暴露] **容错**：只要求组内**至少 2 个表格块**，不再要求「每一块都是表格」。
        #
        # 原判据 `not all(_is_table_text(...))` 过于严格：同一 title 下往往还有一个
        # **纯文本标题块**（内容就是表名、1 行），它会让**整组放弃合并**。
        # 实测（vector_db_benchmark.xlsx.md，6 块）：
        #     title="检索性能实测" 组 = 1 个标题块 + 3 个表格块（第 1-30 / 31-60 / 61-85 行）
        #     -> 因标题块非表格 -> 整组不合并 -> 仍是 3 张「各自带表头」的碎片
        #     -> 模型当成三张独立的表分别看，只报告第一块里的最低值（89.93%），
        #        而全局最低 61.35% 在第二块里。
        # 改为：非表格块**原样保留**，只把表格块合并。
        table_items = [(i, d) for i, d in items if _is_table_text(d.page_content)]
        if len(table_items) < 2:
            out.extend(items)
            continue
        for i, d in items:
            if not _is_table_text(d.page_content):
                out.append((i, d))  # 非表格块原样保留
        items = table_items

        ordered = sorted(items, key=lambda x: _start_line(x[1].page_content))
        merged_lines: list[str] = []
        header: str | None = None
        kept_header = False  # **第一块要保留表头**，后续块才去重
        for _idx, d in ordered:
            lines = [line for line in d.page_content.splitlines() if line.strip()]
            tbl_pos = [j for j, line in enumerate(lines) if line.strip().startswith("|")]
            if not tbl_pos:
                continue
            local_head = lines[tbl_pos[0]].strip()
            skip: set[int] = set()
            if header is None:
                header = local_head
            if local_head == header and kept_header:
                skip.add(tbl_pos[0])  # 重复表头
                nxt = tbl_pos[0] + 1
                if nxt < len(lines) and set(lines[nxt].strip()) <= set("|-: "):
                    skip.add(nxt)  # 重复分隔行
            elif local_head == header:
                kept_header = True
            for j, line in enumerate(lines):
                s = line.strip()
                if s.startswith("表格：") or s.startswith("列名："):
                    continue
                if j in skip:
                    continue
                merged_lines.append(line)

        merged = "\n".join(merged_lines)
        if not merged or len(merged) > max_chars:
            out.extend(items)
            continue
        out.append((items[0][0], type(ordered[0][1])(page_content=merged, metadata=dict(ordered[0][1].metadata))))

    out.sort(key=lambda x: x[0])
    return [d for _, d in out]


def format_docs(docs):
    """带编号拼接上下文。

    编号 [1]、[2] 必须与 `documents` 的下标**严格对应** —— 下游校验层
    （crag_graph.grade_generation_v_documents_and_question）用它反查原文块，
    并判断模型有没有引用不存在的编号。
    """
    if isinstance(docs, list):
        return format_context(docs)
    return '[1] ' + docs.page_content


# [ROADMAP 4.41] **带对话历史**的 prompt。
#
# 为什么不直接改 `CITE_PROMPT`：
#   `graph/generate_node.py:6` 直接 `from graph_crag.generate_node import CITE_PROMPT`，
#   给原模板加一个 `{history}` 变量会让 graph_react 在格式化时**因缺参数报错**
#   （`PromptTemplate` 会校验 `input_variables`）。
#   所以新增一个模板，**只在有历史时才用**，原路径完全不变。
CITE_PROMPT_MEMORY = """你是一个严格基于给定资料作答的助手。规则：
1. 每个事实性句子后必须标注来源编号，格式为 [1]、[2]，多来源写 [1][3]。
2. 只能使用给定资料，禁止补充资料之外的知识（包括你自己的先验知识）。
3. 资料与问题相关时，**能答多少就答多少**：按资料实际覆盖的范围组织回答，
   不要因为资料不够全面、不够「详解」就整体拒答。
4. 若资料只覆盖问题的一部分，先回答已覆盖的部分，再用一句话说明「资料中未涉及 X」，
   不要编造未覆盖的内容。
5. 只有当资料与问题**完全无关**（问的是 A，资料讲的却是 B）时，才只回复：
   知识库中未找到依据。
6. 回答自然、有信息量，符合正常问答的语感；不要机械复述问题，也不要用
   「根据资料」这类套话开头。
7. 下方「对话历史」只用来理解**指代与上下文**（如「它」「这个产品」「那第二点呢」），
   **事实依据必须来自「资料」**；历史里出现但资料中没有的内容不得作为答案。

对话历史：
{history}

资料：
{context}

问题：{question}
回答："""

# 注入 prompt 的历史轮数上限（一问一答算一轮）。
#
# [ROADMAP 4.43] 由 3 提到 **13** —— 依据 `eval/_budget_history.py` 的真实数据：
#      实测 max_turns=13 时历史文本 4,672 字符（≈2,300 token），占整个 prompt 约 22%，
#      远低于「挤占检索上下文」的警戒线（本项目检索上下文约 16,182 字符）。
#      且 question 中位 27 字、golden answer 中位 12 字，本项目真实答案普遍短。
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "13"))

# 单条消息的字符上限。**它不再是主约束** —— 主约束是下面的总预算。
# 提到 500 是防御性的：万一未来有长答案（表格罗列、多条目汇总），
# 3 轮时代的 `[:300]` 会把每条砍残，13 轮时会把 13 条全砍残。
MAX_PER_MSG_CHARS = int(os.getenv("MAX_PER_MSG_CHARS", "500"))

# **历史总预算**（字符）。超出时**从最老的一端开始丢，保住最近**。
#
# 为什么用总预算而不是单条截断：
#   单条截断在「答案都短」时形同虚设（本项目正好如此，实测几乎不触发），
#   在「答案都长」时会把每一轮都削掉一截，得到 13 段残缺记忆；
#   总预算则相反 —— 短答案时能容纳满 13 轮，长答案时优雅退化成「只记最近几轮」。
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "6000"))

# 生成阶段的**上下文预算**（字符）。
#
# [FIX 2026-10-28] 实测（同一问题「milvus有哪些内容」，同一批检索结果）：
#     6 块 / 4,993 字符  -> 正常回答
#     12 块 / 12,089 字符 -> 正常回答
#     20 块 / 17,385 字符 -> 正常回答
#     36 块 / 29,063 字符 -> **拒答**（"知识库中未找到依据"）
# 上下文过长时模型会按 prompt 规则 3 保守拒答，表现为「明明有资料却说没有依据」。
# 这里按**检索顺序**（RRF 已排序）截断；批量问题（36 块全部判分通过）正是最容易被它打中的场景。
MAX_GENERATE_CONTEXT_CHARS = int(os.getenv("GENERATE_MAX_CONTEXT_CHARS", "24000"))


def _truncate_context(docs, budget: int = MAX_GENERATE_CONTEXT_CHARS):
    """按检索顺序截断上下文到字符预算（至少保留 1 块），返回新列表。

    截断后必须**把新列表返回给 state**：引用编号 [1]..[n] 是按列表下标生成的，
    若只截断 prompt 而 state 保留全量，下游的引用校验会错位。
    """
    if not isinstance(docs, list):
        return docs
    out: list[Document] = []
    used = 0
    for d in docs:
        n = len(getattr(d, "page_content", "") or "")
        if out and used + n > budget:
            log.info("[generate] 上下文超预算，截断 %d -> %d 块（%d/%d 字符）" % (len(docs), len(out), used, budget))
            return out
        out.append(d)
        used += n
    return docs  # 未超预算：原样返回，不做多余拷贝


# 触发摘要压缩的轮数阈值：超过它才把「比最近 N 轮更早」的内容压成摘要。
# 默认 2 × MAX_HISTORY_TURNS = 26 轮。
SUMMARY_TRIGGER_TURNS = 2 * MAX_HISTORY_TURNS


def _looks_like_refusal(text: str) -> bool:
    """文本是否是一条「资料不足」的拒答（容忍前缀/包裹）。

    为什么不用纯 is_refusal()：历史向量库里存的是「标题 + 正文」拼接的块，
    精确相等会漏掉带前缀的拒答，导致它仍然被注入 prompt 继续锚定模型。
    """
    t = (text or '').strip()
    return is_refusal(t) or ('知识库中未找到依据' in t)


def _pairs_of(messages):
    """只保留 human / ai 两类消息，按出现顺序。"""
    if not messages:
        return []
    return [m for m in messages if getattr(m, "type", "") in ("human", "ai")]


def _drop_refusal_turns(messages):
    """把「助手这一轮是拒答」的问答轮**整对**丢掉。

    [FIX 2026-10-28] 为什么必须丢：拒答文案里没有任何信息，但会**锚定后续回答**。
    实测同一会话：第一轮拒答后，第二轮即使 35/36 块判分通过、上下文完整，
    模型仍继续回「知识库中未找到依据。」；把拒答轮从历史里去掉后恢复正常。
    整对丢弃（连用户那一问一起）是为了避免出现「有问无答」的悬空历史。
    """
    if not messages:
        return []
    out = []
    i = 0
    while i < len(messages):
        cur = messages[i]
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if getattr(cur, "type", "") == "human" and getattr(nxt, "type", "") == "ai":
            if _looks_like_refusal(str(getattr(nxt, "content", "") or "")):
                i += 2  # 整对跳过
                continue
            out.extend([cur, nxt])
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _format_history(messages, max_turns: int = MAX_HISTORY_TURNS, budget: int | None = None) -> str:
    """把 `messages` 渲染成「用户: … / 助手: …」的纯文本，供 prompt 使用。

    设计要点：
    * 只取**最近** `max_turns` 轮（一轮 = 一条 human + 一条 ai）；
    * `budget` 给定时，**从最新往最旧累加**，超预算即停 ——
      即「先保最近，再保更早」，与直觉一致（绝对不会为了留旧内容而丢掉最新一轮）；
    * 忽略 `system` / `tool` 等其它角色（本流程只产生 human/ai）；
    * 空历史返回空串 —— 调用方据此决定**是否走带历史的模板**；
    * **纯函数、无 LLM 调用**，便于单测（`tests/test_history_budget.py`）。
    """
    keep = _pairs_of(_drop_refusal_turns(messages))[-max_turns * 2 :] if messages else []
    if not keep:
        return ""

    limit = MAX_HISTORY_CHARS if budget is None else budget
    lines: list[str] = []
    total = 0
    for m in reversed(keep):  # 从最新往最旧
        role = "用户" if m.type == "human" else "助手"
        text = str(getattr(m, "content", "")).strip().replace("\n", " ")
        if not text:
            continue
        line = "%s：%s" % (role, text[:MAX_PER_MSG_CHARS])
        # 已经攒下内容、再加就超预算 -> 停止（更老的丢掉）
        if limit and limit > 0 and lines and total + len(line) > limit:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()  # 恢复时间正序
    return "\n".join(lines)


def _should_summarize(messages):
    """判断是否需要摘要压缩，并返回「待摘要的旧内容」。

    Returns:
        (need: bool, old_text: str)
        need=True 时 old_text 是「比最近 MAX_HISTORY_TURNS 轮更早」的对话原文，
        调用方（generate）应把它交给 LLM 压成摘要后再注入 prompt；
        need=False 时 old_text 为 ""（单轮 / 轮数未超阈值）。

    **纯函数**：只做阈值判断与文本切片，不调 LLM。
    """
    pairs = _pairs_of(messages)
    if len(pairs) <= SUMMARY_TRIGGER_TURNS * 2:
        return False, ""
    old = pairs[: -(MAX_HISTORY_TURNS * 2)]
    lines = []
    for m in old:
        role = "用户" if m.type == "human" else "助手"
        text = str(getattr(m, "content", "")).strip().replace("\n", " ")
        if text:
            lines.append("%s：%s" % (role, text[:MAX_PER_MSG_CHARS]))
    return True, "\n".join(lines)


def build_history_prompt_part(messages, budget: int | None = None, summary: str = "") -> dict:
    """把历史整理成「摘要 + 最近若干轮」两部分，供 prompt 使用。

    **纯函数**：摘要文本由调用方生成后传入（它需要一次 LLM 调用，
    不应藏在这个函数里——否则单测也要走 LLM）。

    Returns:
        {"summary": str, "history": str}
        summary 为空串表示没有更早内容需要摘要。
    """
    return {"summary": (summary or "").strip(), "history": _format_history(messages, budget=budget)}


# 摘要压缩的 prompt。[ROADMAP 4.43]
_SUMMARY_PROMPT = """把下面的多轮对话压缩成不超过 300 字的摘要。

必须保留：① 讨论过的主题；② 关键实体（产品名 / 指标名 / 具体数值）；
③ 用户表达过的偏好或约束。
不要编造，只总结给定内容；不要评价。

对话：
{history}

摘要："""


def _summarize(old_text: str, config=None) -> str:
    """调一次 LLM 把旧对话压成摘要。**失败返回空串，不阻断主流程**。

    成本取舍（ROADMAP 4.43）：摘要只在**首次超过阈值**时生成一次，
    之后复用 `state["summary"]`，不每轮重算 ——
    否则 40 轮会话要额外 14 次 LLM 调用，收益却递减（最近 13 轮原文已覆盖）。
    """
    if not old_text or not old_text.strip():
        return ""
    try:
        from langchain_core.output_parsers import StrOutputParser

        chain = PromptTemplate(template=_SUMMARY_PROMPT, input_variables=["history"]) | llm | StrOutputParser()
        text = chain.invoke({"history": old_text}, config=_sub_config(config))
        return (text or "").strip()[:500]  # 兜底截断，防模型不听话
    except Exception as e:  # noqa: BLE001
        log.warning("[history] 摘要生成失败，降级为无摘要: %s: %s" % (type(e).__name__, str(e)[:120]))
        return ""


# [ROADMAP 4.46] **聚合类问题的额外引导**。
#
# 实测背景（`eval/_diag_ctx.py`）：修好「标题块阻断合并」后，87 行的完整表
# **确实进入了上下文**（同时含 61.35 与 89.93），但模型仍答 89.93% ——
# 它只比较了第一个场景（静态检索），没扫完整个表。
#
# 所以问题不在检索、也不在合并，而在**「怎么让模型读完这张表」**。
# 这段引导只对 `is_aggregation_query()` 命中的问题追加，不影响其它问题；
# 且**只拼进 payload 的 question，不写入 messages**，避免污染对话历史。
AGG_HINT = (
    "\n\n（本题需要**跨行比较**：请逐行扫描表格中的**所有场景 / 所有行**，"
    "先列出每个候选值，再给出最小（或最大）者与它的完整条件。"
    "不要只看第一行、第一个场景或第一个分组。）"
)


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

    # [ROADMAP 4.40] **同表切片合并** —— 把同一张表的多块拼回完整表再生成。
    #
    # 依据：`eval/_exp_n005_table.py` 实测（各 3 次）
    #     A 生产上下文（9 块 / 16,182 字符，含 2 个表格碎片）-> **3/3 稳定拒答**
    #     B 完整表   （1 块 /  8,614 字符 / 91 行）           -> **2/3 答对**
    # n005（「召回率最低的产品」）是「跨行统计类」提问，需要看完整表才能确认最小值；
    # 切成 4 片后模型只能保守拒答（这符合 prompt 规则 3，是**正确但无益**的行为）。
    #
    # 为什么放在这里而不是检索层：`run_eval.py:120` 的 recall 用
    # `retriever.invoke(q)` 的**原始块**算，与最终 state 无关 —— 所以此处合并不影响
    # 检索层指标；而 `grade_generation` 的引用校验用 `state["documents"]`，
    # 本函数返回合并后的 documents，编号仍然自洽。
    documents = merge_table_slices(documents)

    # [ROADMAP 4.49] **按章节聚合**（Parent-Child 的「检索后」轻量版）。
    # 顺序很重要：先把表格拼回整表（merge_table_slices），再把整表 + 同章节段落
    # 一起聚成父块（merge_section_blocks）—— 否则父块里会出现「半张表」。
    # **默认关闭**（SECTION_MERGE_ENABLED=0）：开启会让上下文显著变长，
    # 必须先 A/B 验证 token 与质量，不能默认开（与 reranker 同一原则）。
    documents = merge_section_blocks(documents)

    # [FIX 2026-10-28] 上下文预算：过长会让模型保守拒答（见 MAX_GENERATE_CONTEXT_CHARS 的实测）
    documents = _truncate_context(documents)

    # [ROADMAP 4.41] **有历史才用带历史的模板** —— 单轮调用走原路径，行为完全不变。
    #
    # ⚠️ 判据用 `session_id` 而**不是** `messages is not None`：
    #    实测发现 LangGraph 会把 `TypedDict` 里声明的 `messages` **初始化为 `[]`**
    #    （不是缺失、也不是 None），所以 `state.get("messages") is not None` **恒为真** ——
    #    单轮调用也会被误判成「记忆模式」并往里写历史（`_verify_memory_e2e.py` 的 E4 抓到）。
    #
    # `session_id` 由 `run_with_trace` 在**记忆模式下**注入，单轮时不传 -> None ->
    # 判据为假，行为与改造前完全一致。
    has_memory = bool(state.get("session_id"))
    # [ROADMAP 4.43] 摘要压缩：只在「超阈值且尚未生成过摘要」时调一次 LLM，
    # 之后每轮复用 state["summary"]（checkpointer 会把它带回来）。
    _msgs = state.get("messages") or []
    _need_sum, _old_text = _should_summarize(_msgs)
    _summary = state.get("summary") or ""
    if has_memory and _need_sum and not _summary:
        _summary = _summarize(_old_text, config)

    history_text = _format_history(_msgs)
    if _summary:
        # 摘要放在**前面**、最近原文放在**后面** —— 越近的内容离问题越近，权重越高
        history_text = "【更早内容的摘要】\n%s\n\n【最近的对话】\n%s" % (_summary, history_text)

    # [ROADMAP 4.44] **历史向量检索**：找回「含具体数值的早期轮次」。
    #
    # 三层记忆各有不可替代的分工：
    #     最近 13 轮原文 -> 近期指代（「它」）
    #     摘要          -> 主题脉络（做过什么）
    #     ★本节         -> 早期**具体事实**（「第 3 轮那个 184.54」）
    # 只有这一层能在长会话里找回已掉出窗口的具体数值。
    if has_memory:
        _hits = history_store.lookup_history(state.get("session_id"), question)
        if _hits:
            _lines = []
            for i, d in enumerate(_hits[: history_store.HISTORY_TOP_K], 1):
                _md = getattr(d, "metadata", None) or {}
                _body = (d.page_content or "").strip()
                # [FIX 2026-10-28] 拒答轮同样要从这里剔除：它是「另一个通道」——
                # messages 过滤只管最近 N 轮，而本层从历史向量库直接捞旧轮次，
                # 会把历史上的「知识库中未找到依据」重新注入 prompt，继续锚定模型。
                if _looks_like_refusal(_body):
                    continue
                _lines.append("【%d】（%s）%s" % (i, _md.get("title", "更早轮次"), _body[:600]))
            if _lines:
                history_text += "\n\n【更早但相关的轮次（语义检索）】\n" + "\n".join(_lines)

    if history_text:
        prompt = PromptTemplate(
            template=CITE_PROMPT_MEMORY,
            input_variables=["question", "context", "history"],
        )
    else:
        prompt = PromptTemplate(
            template=CITE_PROMPT,
            input_variables=["question", "context"],
        )

    # 构建RAG处理链
    rag_chain = (
        prompt  # 第一步：使用提示模板
        | llm  # 第二步：调用语言模型
        | StrOutputParser()  # 第三步：解析模型输出为字符串
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
    # [ROADMAP 4.46/4.47] 聚合类问题：**结构化计算 + 引导**。
    #
    # 三层排查证明瓶颈在「模型扫不完 87 行表格」（见 rag_kb/tools/table_query.py 的文档）。
    # 所以这里先把跨行比较**交给 pandas 算准**，把结果作为「事实」注入 context，
    # 模型只需陈述；算不出来（返回 None）时才退回「让模型自己扫 + 引导」。
    #
    # 注意：`updates["messages"]` 用的仍是**原始 `question`** —— 引导语与事实
    # 都不进历史，否则下一轮会把它们当成用户说过的话。
    from rag_kb.tools.reranker_tools import is_aggregation_query
    from rag_kb.tools.table_query import answer_aggregation

    _is_agg = is_aggregation_query(question)
    _q = question + AGG_HINT if _is_agg else question
    _ctx = format_docs(documents)
    if _is_agg:
        _fact = answer_aggregation(question, documents)
        if _fact:
            log.info("[generate] 聚合类问题已用代码精确计算：%s" % _fact[:80])
            _ctx = ("【已由代码精确计算的事实 —— 请直接采用该结论，不要自行重新比较】\n%s\n\n" % _fact) + _ctx

    payload = {"context": _ctx, "question": _q}
    if history_text:
        payload["history"] = history_text
    generation = rag_chain.invoke(payload, config=_sub_config(config))  # 调用RAG链生成回答

    # [ROADMAP 2.1 ⑤] 构造「编号 -> 原文块」映射，供调用方渲染溯源信息。
    # 引用编号的**合法性校验**不在这里做 —— 它属于流程控制（越界要触发熔断重试），
    # 放在 crag_graph.grade_generation_v_documents_and_question 里统一处理。
    doc_list = documents if isinstance(documents, list) else [documents]
    cites = validate_citations(generation, len(doc_list))["cites"]
    citations = build_citation_meta(doc_list, cites)

    # 累加生成次数：原先 crag_graph.py 中 "not supported" -> generate 是无条件自环，
    # 幻觉检测反复失败会无限循环；该计数供判分函数做熔断判断。
    generation_count = state.get("generation_count", 0) + 1
    # [2026-11] 对外契约：本答案**有检索依据**。
    # 联网结果的 metadata.source == "web"，据此区分 knowledge_base / web。
    _has_web = any((getattr(d, "metadata", None) or {}).get("source") == "web" for d in doc_list)
    updates = {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_count": generation_count,
        "citations": citations,
        "grounded": True,
        "source_type": "web" if (_has_web and not is_refusal(generation)) else "knowledge_base",
    }

    # [ROADMAP 4.41] **只在记忆模式下追加历史**。
    #
    # `add_messages` 是 reducer，返回 `{"messages": [...]}` 会**追加**到已有列表。
    # 判据同样是「state 里有没有 messages 键」—— 单轮调用不写，state 结构不变。
    #
    # 为什么把「问 + 答」成对追加：下一轮的 prompt 需要看到「上一轮问了什么、
    # 答了什么」才能解析「它 / 那个 / 第二点」这类指代。
    if has_memory:
        updates["messages"] = [HumanMessage(content=question), AIMessage(content=generation)]
        if _summary:
            updates["summary"] = _summary
    return updates
