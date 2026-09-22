from langchain_core.messages import AIMessage, HumanMessage
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


def _is_table_text(text: str) -> bool:
    """是否包含 markdown 表格（至少 2 行以 `|` 开头）。"""
    return sum(1 for l in text.splitlines() if l.strip().startswith("|")) >= 2


def _start_line(text: str) -> int:
    """切片形如 `表格：…（第 31-60 行）`，取起始行号用于排序。"""
    import re
    m = re.search(r"第\s*(\d+)\s*-\s*(\d+)\s*行", text)
    return int(m.group(1)) if m else 10 ** 9


def merge_table_slices(docs, max_chars: int = 20000):
    """把**同一张表**的多个切片拼回完整表（ROADMAP 4.40）。

    为什么需要：`eval/_exp_n005_table.py` 实测（各 3 次）——

        A 生产上下文（9 块 / 16,182 字符，含 2 个表格碎片）: **3/3 稳定拒答**
        B 完整表   （1 块 /  8,614 字符 / 91 行）          : **2/3 答对**

    同一张表被切成 4 片后，模型看不到完整表，**无法确认「哪个最低」**，
    于是按 prompt 规则 3 保守拒答。而 `generate_node2.py` 的注释早就点名
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
    groups: dict[tuple, list[tuple[int, object]]] = {}
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", None) or {}
        key = (md.get("filename"), md.get("title"))
        groups.setdefault(key, []).append((i, d))

    out: list[tuple[int, object]] = []
    for _key, items in groups.items():
        if len(items) < 2 or not all(_is_table_text(d.page_content) for _, d in items):
            out.extend(items)
            continue

        ordered = sorted(items, key=lambda x: _start_line(x[1].page_content))
        merged_lines: list[str] = []
        header: str | None = None
        kept_header = False        # **第一块要保留表头**，后续块才去重
        for _idx, d in ordered:
            lines = [l for l in d.page_content.splitlines() if l.strip()]
            tbl_pos = [j for j, l in enumerate(lines) if l.strip().startswith("|")]
            if not tbl_pos:
                continue
            local_head = lines[tbl_pos[0]].strip()
            skip: set[int] = set()
            if header is None:
                header = local_head
            if local_head == header and kept_header:
                skip.add(tbl_pos[0])                      # 重复表头
                nxt = tbl_pos[0] + 1
                if nxt < len(lines) and set(lines[nxt].strip()) <= set("|-: "):
                    skip.add(nxt)                         # 重复分隔行
            elif local_head == header:
                kept_header = True
            for j, l in enumerate(lines):
                s = l.strip()
                if s.startswith("表格：") or s.startswith("列名："):
                    continue
                if j in skip:
                    continue
                merged_lines.append(l)

        merged = "\n".join(merged_lines)
        if not merged or len(merged) > max_chars:
            out.extend(items)
            continue
        out.append((items[0][0],
                    type(ordered[0][1])(page_content=merged,
                                        metadata=dict(ordered[0][1].metadata))))

    out.sort(key=lambda x: x[0])
    return [d for _, d in out]


def format_docs(docs):
    """带编号拼接上下文。

    编号 [1]、[2] 必须与 `documents` 的下标**严格对应** —— 下游校验层
    （graph_2.grade_generation_v_documents_and_question）用它反查原文块，
    并判断模型有没有引用不存在的编号。
    """
    if isinstance(docs, list):
        return format_context(docs)
    return '[1] ' + docs.page_content


# [ROADMAP 4.41] **带对话历史**的 prompt。
#
# 为什么不直接改 `CITE_PROMPT`：
#   `graph/generate_node.py:6` 直接 `from graph2.generate_node2 import CITE_PROMPT`，
#   给原模板加一个 `{history}` 变量会让 graph1 在格式化时**因缺参数报错**
#   （`PromptTemplate` 会校验 `input_variables`）。
#   所以新增一个模板，**只在有历史时才用**，原路径完全不变。
CITE_PROMPT_MEMORY = """你是一个严格基于给定资料作答的助手。规则：
1. 每个事实性句子后必须标注来源编号，格式为 [1]、[2]，多来源写 [1][3]。
2. 只允许使用给定资料，禁止补充资料之外的知识。
3. 若资料不足以回答，只回复：知识库中未找到依据。
4. 回答简洁，不要重复问题。
5. 下方「对话历史」只用来理解**指代与上下文**（如「它」「这个产品」「那第二点呢」），
   **事实依据必须来自「资料」**；历史里出现但资料中没有的内容不得作为答案。

对话历史：
{history}

资料：
{context}

问题：{question}
回答："""

# 注入 prompt 的历史轮数上限（一问一答算一轮）。
# 为什么限轮数：`messages` 会随轮次无限增长，全塞进去会挤爆上下文窗口，
# 也会让早期无关对话干扰当前判断。3 轮足以覆盖「它/那个/第二点」这类指代。
MAX_HISTORY_TURNS = 3


def _format_history(messages, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """把 `messages` 渲染成「用户: … / 助手: …」的纯文本，供 prompt 使用。

    设计要点：
    * 只取**最近** `max_turns` 轮（一轮 = 一条 human + 一条 ai）；
    * 忽略 `system` / `tool` 等其它角色（本流程只产生 human/ai）；
    * 空历史返回空串 —— 调用方据此决定**是否走带历史的模板**；
    * **纯函数、无 LLM 调用**，便于单测（`eval/test_memory.py` C4）。
    """
    if not messages:
        return ""
    # 只保留 human / ai 两类，按出现顺序
    pairs = [m for m in messages
             if getattr(m, "type", "") in ("human", "ai")]
    if not pairs:
        return ""
    keep = pairs[-max_turns * 2:]          # 最近 max_turns 轮
    lines = []
    for m in keep:
        role = "用户" if m.type == "human" else "助手"
        text = str(getattr(m, "content", "")).strip().replace("\n", " ")
        if text:
            lines.append("%s：%s" % (role, text[:300]))   # 单条截断，防单轮超长
    return "\n".join(lines)


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
    history_text = _format_history(state.get("messages") or [])

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
    payload = {"context": format_docs(documents), "question": question}
    if history_text:
        payload["history"] = history_text
    generation = rag_chain.invoke(payload, config=_sub_config(config))  # 调用RAG链生成回答

    # [ROADMAP 2.1 ⑤] 构造「编号 -> 原文块」映射，供调用方渲染溯源信息。
    # 引用编号的**合法性校验**不在这里做 —— 它属于流程控制（越界要触发熔断重试），
    # 放在 graph_2.grade_generation_v_documents_and_question 里统一处理。
    doc_list = documents if isinstance(documents, list) else [documents]
    cites = validate_citations(generation, len(doc_list))["cites"]
    citations = build_citation_meta(doc_list, cites)

    # 累加生成次数：原先 graph_2.py 中 "not supported" -> generate 是无条件自环，
    # 幻觉检测反复失败会无限循环；该计数供判分函数做熔断判断。
    generation_count = state.get("generation_count", 0) + 1
    updates = {"documents": documents, "question": question, "generation": generation,
               "generation_count": generation_count, "citations": citations}

    # [ROADMAP 4.41] **只在记忆模式下追加历史**。
    #
    # `add_messages` 是 reducer，返回 `{"messages": [...]}` 会**追加**到已有列表。
    # 判据同样是「state 里有没有 messages 键」—— 单轮调用不写，state 结构不变。
    #
    # 为什么把「问 + 答」成对追加：下一轮的 prompt 需要看到「上一轮问了什么、
    # 答了什么」才能解析「它 / 那个 / 第二点」这类指代。
    if has_memory:
        updates["messages"] = [HumanMessage(content=question),
                               AIMessage(content=generation)]
    return updates
