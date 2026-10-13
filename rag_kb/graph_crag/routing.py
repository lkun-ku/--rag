"""CRAG 条件边：生成质量判定、检索分支判定、问题路由。

从 `crag_graph.py` 拆出：原文件同时承担**节点定义、条件边、图装配**三件事，
600+ 行不便定位。本模块只放条件边函数与其常量，图装配仍在 `crag_graph.py`。
"""
import inspect
import time
from concurrent.futures import ThreadPoolExecutor

from rag_kb.graph_crag.grade_answer_chain import answer_grader_chain
from rag_kb.graph_crag.grade_documents_node import _sub_config
from rag_kb.graph_crag.grade_hallucinations_chain import hallucination_grader_chain
from rag_kb.graph_crag.query_route_chain import question_router_chain
from rag_kb.utils import tracer
from rag_kb.utils.log_utils import log

# 生成答案的最大重试次数：达到上限后改走 "not useful"（受 transform_count 保护）。
MAX_GENERATION_RETRY = 2
# 问题改写次数上限（与 decide_to_generate 里的阈值保持一致）
MAX_TRANSFORM_RETRY = 2


def grade_generation_v_documents_and_question(state, config=None):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
        config: LangGraph 注入的运行配置。**必须接收并透传** —— 本函数内部有
            两次 LLM 判定（幻觉 + 回答质量），不透传则它们的 token 不可见
            （这与 `tracer.traced` / `grade_documents` 是同一个坑，见 ROADMAP 4.25）。
    Returns:
        str: 下一节点的名称（useful/not useful/not supported）
    """
    log.info("---检查生成内容是否存在幻觉---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取参考文档
    generation = state["generation"]  # 获取生成结果

    # ---- [ROADMAP 2.1 ④] 先做确定性的引用校验 ----
    # 为什么放在 LLM 判定之前：这是**确定性**检查（字符串 + 数值比较），
    # 零成本、零抖动，且「引用了不存在的编号」是最危险的幻觉形态 ——
    # 它让用户以为答案有依据，比「没有引用」更糟。
    from rag_kb.utils.citation import validate_citations  # 局部导入：避免与判分链的循环依赖

    doc_list = documents if isinstance(documents, list) else [documents]
    # 传入 docs 后才能区分「幻觉引用」与「复述被检索块原文自带的编号」（P0-3）
    check = validate_citations(generation, len(doc_list) or 1, docs=doc_list)

    if check["is_refusal"]:
        # 设计内拒答：资料不足时的**正确**响应，不应触发熔断重试
        # （若在这里重试，只会反复得到同样的拒答，白白消耗 LLM 调用）。
        log.info("---判定：模型按规则拒答（资料不足），结束流程---")
        return "useful"

    if check["quoted"]:
        # 复述了被检索块原文里的参考文献编号（如 papers 块里的 [28]）。
        # 这不是幻觉 —— 编号确实来自资料，只是不属于本次的 1..n 编号体系。
        # 不触发重试，但记一条日志便于排查（P0-3）。
        log.info(f"---注：答案复述了原文编号 {check['quoted']}（非幻觉，不触发重试）---")

    if check["invalid"]:
        generation_count = state.get("generation_count", 0)  # 熔断
        log.warning(f"---判定：引用编号越界 {check['invalid']}（幻觉引用），已生成 {generation_count} 次---")
        if generation_count >= MAX_GENERATION_RETRY:
            # 熔断耗尽：改走受 transform_count 保护的分支，避免无限循环
            return "not useful"
        return "not supported"

    # ---- 两次判定：**并发**执行（ROADMAP 4.25 / 4.31）----
    #
    # 原实现是**串行 + 短路**：
    #     1) 先判「是否基于文档」；2) **只在第 1 次判 yes 时**才判「是否回答了问题」。
    # 实测该环节 1.42s，是链路里仅次于判分的第二大头。
    #
    # **为什么可以并发**：两次调用的输入都能从 state 直接取到，第 2 次**不依赖第 1 次的
    # 输出**、只依赖它的**结论** —— 所以可以提前把它发出去。
    #
    # **短路语义严格保持不变**：第 2 次调用总是发出，但其结果**只在第 1 次判 yes 时使用**
    # （第 1 次判 no 时结果直接丢弃，走原 else 分支）。
    # 代价：第 1 次判 no 时那次调用是白跑的 —— 用少量浪费换延迟，**不改变判定结果**。
    #
    # `.result()` 的取用时机也是正确性的一部分：
    #   * 第 1 次在 `with` 内取（它总是需要）；
    #   * 第 2 次**出了 `with` 块再按需取**，并单独兜异常。若在 `with` 内无条件取，
    #     第 1 次判 no 时（原实现根本不会调用 answer_grader）它一旦抛错，
    #     异常会冒出来**炸掉整个节点** —— 那就改变了判定结果。
    #
    # > **[二次筛查纠正 ] 这里必须保留 `_cfg = _sub_config(config)`。**
    # > 上一轮改这段注释/取值时机时**把它删掉了**，却留下两处 `config=_cfg`，
    # > 于是**任何走到这里的正常请求都会抛 `NameError`**（P0）。
    # > 它**没被回归抓到**，因为回归里没有「完整工作流」这一层覆盖（见 4.33）。
    _cfg = _sub_config(config)
    with ThreadPoolExecutor(max_workers=2) as _ex:
        _f_hall = _ex.submit(
            hallucination_grader_chain.invoke, {"documents": documents, "generation": generation}, config=_cfg
        )
        _f_ans = _ex.submit(answer_grader_chain.invoke, {"question": question, "generation": generation}, config=_cfg)
        hall_score = _f_hall.result()

    grade = getattr(hall_score, "binary_score", "")

    # [FIX 2026-11] 联网分支的失败**不能**落在 `not useful` 上。
    #
    # `not useful` 的边固定指向 `transform_query`，而它的下一步是 `retrieve`（查知识库）。
    # 实测问「今天星期几」：联网没搜到日期 -> 判不合格 -> 转去查库 -> 拿到 5 块
    # Milvus 运维文档 -> 用它们生成带 [1][5] 引用的答案 -> 再判不合格 -> 再转……
    # 跑了 5 轮、28 秒，最后被 recursion_limit 掐断。
    #
    # 现在：联网分支答不上来就直接结束（保留「没有实时信息」这个诚实的答案），
    # 只有知识库分支才值得换问法重试。
    _is_web = state.get("route") == "web"

    def _retry_or_end(reason: str) -> str:
        if _is_web:
            log.info(f"---决策：{reason}；本轮走联网分支，不再转查知识库，结束---")
            return "give_up"
        return "not useful"

    if grade == "yes":  # 如果生成基于文档
        log.info("---判定：生成内容基于参考文档---")
        # 检查是否准确回答问题（结果在上面已并发算出，此处直接取用）
        log.info("---评估：生成回答与问题的匹配度---")
        try:
            ans_score = _f_ans.result()
            grade = getattr(ans_score, "binary_score", "")
        except Exception as e:  # noqa: BLE001
            # 第 2 次判定失败 -> **按 not useful 处理**（保守：无法确认回答了问题，
            # 让上游走 transform_query / 收尾，而不是抛异常中断整次问答）。
            log.warning(f"---判定：回答质量判定失败，按 not useful 处理: {type(e).__name__}: {str(e)[:80]}---")
            return _retry_or_end("回答质量判定失败")
        if grade == "yes":  # 如果正确回答问题
            log.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果
        else:  # 如果没有回答问题
            log.info("---判定：生成内容未能准确回答问题---")
            return _retry_or_end("生成内容未能准确回答问题")  # 返回无用结果
    else:  # 如果生成不基于文档
        generation_count = state.get("generation_count", 0)  # 熔断
        transform_count = state.get("transform_count", 0)
        # [FIX 2026-11] 两侧都到上限就**终止**，不再回 transform_query。
        # 原实现这里返回 "not useful"，而它的边指向 transform_query ——
        # 于是「生成不合格 -> 改写 -> 检索 -> 仍不合格 -> 再生成」形成**无终止边**的环。
        if generation_count >= MAX_GENERATION_RETRY and transform_count >= MAX_TRANSFORM_RETRY:
            log.info(
                f"---决策：已生成 {generation_count} 次、改写 {transform_count} 次仍未得到可靠答案，终止---"
            )
            return "give_up"
        if generation_count >= MAX_GENERATION_RETRY:
            log.info(f"---决策：已生成 {generation_count} 次仍未基于参考文档，换个问法再试---")
            return _retry_or_end("已生成 %d 次仍未基于参考文档" % generation_count)  # 还有改写额度，换问法重试
        log.info("---判定：生成内容未基于参考文档，将重新尝试---")
        return "not supported"  # 返回不支持结果


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和过滤后的文档

    Returns:
        str: 下一节点的名称（transform_query或generate）
    """
    log.info("---ASSESS GRADED DOCUMENTS---")  # 阶段标识

    filtered_documents = state["documents"]  # 获取已过滤文档
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:  # 如果没有相关文档
        if transform_count >= MAX_TRANSFORM_RETRY:
            # [2026-11] 改写两次仍无依据 -> **降级作答**，不再反复转圈。
            #
            # 原实现在这里转 `web_search`（联网兜底），但路由已经判断过
            # 「这是知识库问题」才会走到 retrieve；此时再联网，多半也是白跑一趟，
            # 而且会让链路变成 retrieve->transform->web->生成->判分->再 transform 的环
            # （实测「今天星期几」跑了 5 轮、28 秒）。
            log.info("---决策：改写 %d 次仍无依据，降级为通用知识作答（会标注非知识库）---" % transform_count)
            return "fallback"
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"  # 返回问题优化节点
    else:  # 如果有相关文档
        log.info("---决策：生成最终回答---")
        return "generate"  # 返回回答生成节点


def route_question(state, config=None):
    """
    路由问题到网络搜索或RAG流程
    Args:
        state (dict): 当前图状态，包含用户问题
        config: LangGraph 注入的运行配置。**[FIX / ROADMAP 4.29] 显式接收并
            透传** —— 本函数有一次 LLM 调用（路由判定），不要依赖 contextvar 隐式继承：
            实测（`eval/_probe_contextvar_threadpool.py`）显示 contextvar 在**线程池内失效**
            （无线程池时 token=11，线程池内不传 config 时 token=0）。
            本函数当前串行执行、侥幸能统计到，但一旦并发化就会**静默归零**。

    Returns:
        str: 下一节点的名称（web_search或vectorstore）
    """
    log.info("---ROUTE QUESTION---")  # 阶段标识
    question = state["question"]  # 获取用户问题

    # [2026-11] 把最近几轮对话一并交给路由器：
    # 「我刚才问了什么」「上面那条再展开」这类问题，**不看历史无法正确分流** ——
    # 它们本身不含任何检索线索，只看当前句子必然被判成「查库」或「联网」。
    route_input = question
    if state.get("session_id"):
        from rag_kb.graph_crag.generate_node import _format_history

        _hist = _format_history(state.get("messages") or [], max_turns=3)
        if _hist:
            route_input = "【最近的对话】\n%s\n\n【本轮问题】%s" % (_hist, question)

    source = question_router_chain.invoke({"question": route_input}, config=_sub_config(config))  # 调用问题路由器

    # 根据路由结果决定下一个节点。结构化输出可能是 pydantic 模型，也可能是
    # dict（取决于模型是否走 function_calling 回退），这里统一取字段。
    datasource = source.get("datasource") if isinstance(source, dict) else source.datasource
    if datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"
    if datasource == "direct":
        log.info("---路由到直接回答（不检索）---")
        return "direct"
    # 兜底走知识库：宁可多查一次，也不要漏掉库里的依据
    log.info("---路由到RAG系统---")
    return "vectorstore"


def _traced_router(name: str, fn):
    """包装条件边函数，记录**决策结果**（ROADMAP 3.1）。

    为什么单独包条件边：节点包装器只能看到 state 变化，看不出「为什么走这条分支」。
    而路由决策恰恰是排查时最想知道的信息（例如「为什么走了 web_search 而不是 RAG」）。

    ## [FIX / ROADMAP 4.25] 同样要透传 config

    这与 `utils.tracer.traced` 是**同一个坑的第二处实例**：条件边函数内部也可能有
    LLM 调用（`grade_generation` 的两次判定），`wrapper(state)` 只接 state
    就会把 LangGraph 传来的 config 丢掉，那些调用的 token 同样**不可见**。

    上一轮只修了节点包装器，这里补上 —— 教训是「同一个模式在两处出现，
    修的时候要一起找」（`grep` 一下同类包装器）。
    """

    # [FIX / 代码审查] 与 `utils.tracer.traced` 同步：**也要识别仅关键字
    # config**（`def fn(state, *, config=None)`），否则那种签名会被静默丢 config。
    try:
        _params = list(inspect.signature(fn).parameters.values())
        _n_pos = len([p for p in _params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        _kwonly_cfg = any(p.name == "config" and p.kind == p.KEYWORD_ONLY for p in _params)
    except (TypeError, ValueError):
        _n_pos, _kwonly_cfg = 1, False

    def wrapper(state, config=None):
        t0 = time.time()
        if _n_pos >= 2:
            decision = fn(state, config)
        elif _kwonly_cfg:
            decision = fn(state, config=config)
        else:
            decision = fn(state)
        t = tracer.current()
        if t is not None:
            info = {"decision": decision}
            docs = state.get("documents")
            if isinstance(docs, list):
                info["documents_in"] = len(docs)
            if name == "grade_generation":
                # 引用校验结果对排查最有用：越界/拒答/正常
                from rag_kb.utils.citation import validate_citations

                doc_list = docs if isinstance(docs, list) else [docs]
                chk = validate_citations(state.get("generation", ""), len(doc_list) or 1, docs=doc_list)
                info.update(
                    {
                        "cites": chk["cites"],
                        "invalid_cites": chk["invalid"],
                        "quoted_cites": chk["quoted"],
                        "is_refusal": chk["is_refusal"],
                    }
                )
            t.step(name, elapsed=time.time() - t0, **info)
        return decision

    wrapper.__name__ = name
    return wrapper
