import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph2.generate_node2 import generate
from graph2.grade_answer_chain import answer_grader_chain
from graph2.grade_documents_node import _sub_config, grade_documents
from graph2.grade_hallucinations_chain import hallucination_grader_chain
from graph2.graph_state2 import GraphState
from graph2.query_route_chain import question_router_chain
from graph2.retriever_node import retrieve
from graph2.transform_query_node import transform_query
from graph2.web_search_node import web_search
from utils import tracer
from utils.log_utils import log


# 生成答案的最大重试次数。
# 原实现中 "not supported" -> generate 是无条件自环，一旦幻觉检测持续失败就会无限循环
# （transform_count 只保护了 transform_query 分支）。达到上限后改走 "not useful"，
# 该分支受 transform_count 保护，整体循环次数因此有界。
MAX_GENERATION_RETRY = 2


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
    from utils.citation import validate_citations  # 局部导入：避免与判分链的循环依赖

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
        _f_hall = _ex.submit(hallucination_grader_chain.invoke,
                             {"documents": documents, "generation": generation},
                             config=_cfg)
        _f_ans = _ex.submit(answer_grader_chain.invoke,
                            {"question": question, "generation": generation},
                            config=_cfg)
        hall_score = _f_hall.result()

    grade = getattr(hall_score, "binary_score", "")

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
            log.warning(f"---判定：回答质量判定失败，按 not useful 处理: "
                        f"{type(e).__name__}: {str(e)[:80]}---")
            return "not useful"
        if grade == "yes":  # 如果正确回答问题
            log.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果
        else:  # 如果没有回答问题
            log.info("---判定：生成内容未能准确回答问题---")
            return "not useful"  # 返回无用结果
    else:  # 如果生成不基于文档
        generation_count = state.get("generation_count", 0)  # 熔断
        if generation_count >= MAX_GENERATION_RETRY:
            log.info(f"---决策：已生成 {generation_count} 次仍未基于参考文档，停止重试，转为变换问题---")
            return "not useful"  # 改走受 transform_count 保护的分支，避免无限循环
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
        if transform_count >= 2:
            log.info("---决策：所有文档都与问题无关,并且已经循环了2次，转为web查询问题---")
            return "web_search"  # 返回问题优化节点
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
    source = question_router_chain.invoke({"question": question},
                                          config=_sub_config(config))  # 调用问题路由器

    # 根据路由结果决定下一个节点
    if source.datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"
    elif source.datasource == "vectorstore":
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
        _n_pos = len([p for p in _params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        _kwonly_cfg = any(
            p.name == "config" and p.kind == p.KEYWORD_ONLY for p in _params
        )
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
                from utils.citation import validate_citations
                doc_list = docs if isinstance(docs, list) else [docs]
                chk = validate_citations(state.get("generation", ""), len(doc_list) or 1,
                                         docs=doc_list)
                info.update({"cites": chk["cites"], "invalid_cites": chk["invalid"],
                             "quoted_cites": chk["quoted"], "is_refusal": chk["is_refusal"]})
            t.step(name, elapsed=time.time() - t0, **info)
        return decision

    wrapper.__name__ = name
    return wrapper


def create_graph(checkpointer=None):
    """构建并编译工作流图，返回已编译的图对象。

    Args:
        checkpointer: 可选的检查点存储。**默认 None（不挂记忆）** ——
            模块级 `graph` 就是这样编译的，与改造前完全一致；
            需要多轮记忆时传 `SqliteSaver`（见 `get_memory_graph()`）。

            [ROADMAP 4.41] 为什么做成**可选参数**而不是直接挂上：
            挂了 checkpointer 后 LangGraph 要求每次调用必须带
            `config["configurable"]["thread_id"]`，否则报错；
            而现有调用方（`eval/*.py`、`test_pipeline_smoke`、
            甚至 `run_with_trace` 的默认路径）都不传 thread_id。
            做成可选参数 -> **零破坏**：默认行为不变，要用记忆才显式开启。

    依据 LangGraph 官方 Application structure 文档，langgraph.json 的 graphs 入口
    可以是「已编译的图对象」，也可以是「一个用于创建图的函数」。本函数即后者，
    使构图逻辑可被独立调用与测试。

    节点、条件边与熔断阈值均与迁移前一致。

    [ROADMAP 3.1] 所有节点与条件边都经过 `tracer` 包装 —— **只加观测，不改行为**：
        * 节点用 `tracer.traced()` 包：记录耗时 + state 摘要（documents 数量等）
        * 条件边用 `_traced_router()` 包：记录**路由决策结果**
    这样无需改动任何节点实现文件（`generate_node2.py` / `retriever_node.py` …）。
    """
    # 初始化工作流图
    workflow = StateGraph(GraphState)

    # 定义各状态节点（经 tracer 包装，记录耗时与状态摘要）
    workflow.add_node("web_search", tracer.traced("web_search")(web_search))
    workflow.add_node("retrieve", tracer.traced("retrieve")(retrieve))
    workflow.add_node("grade_documents", tracer.traced("grade_documents")(grade_documents))
    workflow.add_node("generate", tracer.traced("generate")(generate))
    workflow.add_node("transform_query", tracer.traced("transform_query")(transform_query))

    # 起始路由判断
    workflow.add_conditional_edges(
        START,
        _traced_router("route_question", route_question),
        {
            "web_search": "web_search",
            "vectorstore": "retrieve"
        }
    )

    # 添加固定边
    workflow.add_edge("web_search", "generate")  # 网络搜索后直接生成回答
    workflow.add_edge("retrieve", "grade_documents")  # 检索后评估文档相关性

    # 文档评估后的条件分支
    workflow.add_conditional_edges(
        'grade_documents',
        _traced_router("decide_to_generate", decide_to_generate)
    )

    # 生成结果评估后的条件分支
    workflow.add_conditional_edges(
        "generate",  # 生成节点
        _traced_router("grade_generation", grade_generation_v_documents_and_question),
        {
            "not supported": "generate",  # 生成不符合要求时重试
            "useful": END,  # 生成符合要求时结束
            "not useful": "transform_query",  # 生成无用结果时优化查询
        },
    )

    workflow.add_edge("transform_query", "retrieve")  # 查询优化后重新检索

    # 编译工作流
    # [ROADMAP 4.41] 传 checkpointer 时才启用多轮记忆（默认 None = 无记忆）
    return workflow.compile(checkpointer=checkpointer)


# 模块级导出已编译的图，保持原有 `from graph2.graph_2 import graph` 的用法不变
# **注意：这是「无记忆」的图** —— 单轮调用、评测、回归都走它。
graph = create_graph()


# [ROADMAP 4.41] 带记忆的图：**惰性单例**。
#
# 为什么惰性：`create_checkpointer()` 会建立 SQLite 连接并可能写表，
# 在 import 阶段执行会让「只想跑一次检索评测」的脚本也去开数据库。
#
# 为什么单例：`SqliteSaver` 持有连接，重复构造会叠加连接数；
# 而多轮记忆需要**同一个 saver 实例**才能看到彼此写入的检查点。
_memory_graph = None


def get_memory_graph():
    """返回挂了 `SqliteSaver` 的图（多轮记忆用）。

    与模块级 `graph` 的区别只有 checkpointer：
    同一 `thread_id` 的多次调用会**先读回上次的 state**（含 `messages`），
    因此 `generate` 能把历史注入 prompt。

    真正上生产应换 `PostgresSaver`（SQLite 单文件并发写是瓶颈）。
    """
    global _memory_graph
    if _memory_graph is None:
        from utils.checkpointer import create_checkpointer
        _memory_graph = create_graph(checkpointer=create_checkpointer())
    return _memory_graph


def run_with_trace(question: str, enabled: bool = True, save: bool = True,
                   thread_id: str | None = None) -> dict:
    """跑一次完整工作流，返回最终状态 + 执行轨迹（ROADMAP 3.1）。

    与直接 `graph.stream()` 的区别只在**观测**：
        * 通过 `config={"callbacks": ...}` 把 token 统计回调下传到所有 LLM 调用
        * 包装过的节点/条件边会把耗时与决策写入本次 trace

    Args:
        question: 用户问题。
        enabled: 是否记录（False 时等价于不观测，用于对比性能开销）。
        save: 是否把 trace 追加写入 `logs/trace.jsonl`。
        thread_id: **多轮会话标识**（ROADMAP 4.41）。
            * `None`（默认）-> 走**无记忆**的模块级 `graph`，行为与改造前**完全一致**；
            * 传值 -> 走 `get_memory_graph()`，同一 thread_id 的多次调用共享历史。

    Returns:
        {"generation": 最终答案, "trace": 轨迹 dict, "documents": ..., "messages": ...}
    """
    t = tracer.start(question, enabled=enabled)
    base_config = {"callbacks": t.callbacks} if t.callbacks else {}
    final_state: dict = {}

    if thread_id:
        g = get_memory_graph()
        # 注入 `session_id` 作为「记忆模式」的**显式标记**（`generate` 用它判断
        # 是否追加历史）。不能用 `messages is not None` —— LangGraph 会把它初始化为 []。
        #
        # `messages=[]` 只是首轮的初值；`add_messages` reducer 会把它与
        # checkpoint 读回的历史合并（实测：传 [] 不会覆盖已有历史）。
        inputs = {"question": question, "messages": [],
                  "session_id": str(thread_id)}
        config = {**base_config, "configurable": {"thread_id": str(thread_id)}}
    else:
        g = graph
        inputs = {"question": question}
        config = base_config or None

    for output in g.stream(inputs, config=config):
        for _node, value in output.items():
            if isinstance(value, dict):
                final_state.update(value)

    # [ROADMAP 4.41] **记忆模式下改用 checkpointer 的完整 state**。
    #
    # 为什么：`g.stream()` 产出的是**每个节点的增量**。对 `messages` 这类带
    # reducer（`add_messages`）的字段，图内 state 会累积成 4 条，但 `generate`
    # 节点的**增量**只有本轮新增的 2 条 —— 上面的 `final_state.update()` 会把
    # 累积结果覆盖成 2 条。实测（`_verify_memory_e2e.py` E2）：
    #     第2轮前 get_state 读到 2 条（历史在），第2轮返回却只有 2 条（被增量覆盖）。
    # 其它字段（question/generation/documents）都是覆盖语义，所以不受影响。
    if thread_id:
        try:
            _vals = getattr(g.get_state(config), "values", None) or {}
            if _vals:
                final_state = dict(_vals)
        except Exception:  # noqa: BLE001
            pass          # 读不到就退回增量拼装的结果，不阻断主流程

    gen = final_state.get("generation", "")
    t.result = gen
    if save:
        t.save()
    return {"generation": gen, "trace": t.to_dict(),
            "documents": final_state.get("documents", []),
            # [ROADMAP 4.41] 供调用方检查多轮历史是否在累积（单轮模式下恒为空）
            "messages": final_state.get("messages", []),
            "session_id": thread_id}

# draw_graph(graph, 'graph_rag2.png')

# 执行工作流

def main():
    """交互式运行入口。

    这段循环原先位于模块顶层，import 即执行、无法复用；现移入 main() 并由
    __main__ 守卫。交互逻辑与退出条件均未改动。
    """
    _printed = set()  # set集合，避免重复打印

    # 执行工作流
    while True:
        question = input('用户：')
        if question.lower() in ['q', 'exit', 'quit']:
            print('对话结束，拜拜！')
            break
        else:
            inputs = {
                "question": question
            }
            # 流式执行工作流
            for output in graph.stream(inputs):
                for key, value in output.items():
                    # 打印当前节点名称
                    pprint(f"Node '{key}':")  # 显示当前执行的节点名称
                    # 可选：打印每个节点的完整状态信息
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")  # 节点分隔线

            # 打印最终生成结果
            pprint(value["generation"])  # 输出最终生成的回答内容


if __name__ == '__main__':
    main()
