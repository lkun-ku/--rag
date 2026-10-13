import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

from langgraph.constants import START, END
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph

from rag_kb.graph_crag.direct_node import generate_direct, generate_fallback
from rag_kb.graph_crag.generate_node import generate
from rag_kb.graph_crag.crag_state import GraphState
from rag_kb.graph_crag.grade_documents_node import grade_documents
from rag_kb.graph_crag.retriever_node import retrieve
from rag_kb.graph_crag.transform_query_node import transform_query
from rag_kb.graph_crag.web_search_node import web_search
from rag_kb.utils import tracer
from rag_kb.utils import session_store
from rag_kb.utils.log_utils import log
from rag_kb.tools import history_store


# 单次调用的**总步数上限**（LangGraph recursion_limit）。
#
# [FIX 2026-11] 这是最后一道兜底。此前环 `generate ->(not useful)-> transform_query
# -> retrieve -> grade_documents ->(空)-> web_search -> generate` **没有任何终止边**：
# `grade_generation` 的熔断只是把 `not supported` 换成 `not useful`，而 `not useful`
# 同样指回 `transform_query`，于是环继续转。实测一次提问跑了 **36 轮**、213 秒，
# 期间问题被改写几十遍直至失真。
#
# 正常链路 5~6 步；两次生成 + 两次改写 + 一次联网约 13 步，20 足够。
MAX_RECURSION = int(os.getenv("GRAPH_RECURSION_LIMIT", "20"))

# 历史向量库的**后台写入队列**（单线程串行，见 run_with_trace 末尾的说明）。
# 放在模块级：整个进程共用一个 writer，避免每轮都新建线程。
_HISTORY_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="history-writer")

from rag_kb.graph_crag.routing import (
    _traced_router,
    decide_to_generate,
    grade_generation_v_documents_and_question,
    route_question,
)


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
    这样无需改动任何节点实现文件（`generate_node.py` / `retriever_node.py` …）。
    """
    # 初始化工作流图
    workflow = StateGraph(GraphState)

    # 定义各状态节点（经 tracer 包装，记录耗时与状态摘要）
    workflow.add_node("web_search", tracer.traced("web_search")(web_search))
    workflow.add_node("retrieve", tracer.traced("retrieve")(retrieve))
    workflow.add_node("grade_documents", tracer.traced("grade_documents")(grade_documents))
    workflow.add_node("generate", tracer.traced("generate")(generate))
    workflow.add_node("transform_query", tracer.traced("transform_query")(transform_query))
    # [2026-11] 第三条路径：不检索、不判分，直接回答（闲聊 / 追问上文 / 写作等）
    workflow.add_node("generate_direct", tracer.traced("generate_direct")(generate_direct))
    # [2026-11] 第四条路径：**检索失败后的降级作答**。
    #
    # 原行为是「检索不到 -> 反复改写 -> 转联网 -> 再判 -> 最后吐一句『知识库中未找到依据』」。
    # 用户体验上这是一次失败；而它对很多问题是多余的 —— 用户问「什么是递归」，
    # 知识库里没有，模型其实能直接答，只是需要**明确标注这不是内部资料**。
    workflow.add_node("generate_fallback", tracer.traced("generate_fallback")(generate_fallback))

    # 起始路由判断
    workflow.add_conditional_edges(
        START,
        _traced_router("route_question", route_question),
        {"web_search": "web_search", "vectorstore": "retrieve", "direct": "generate_direct"},
    )

    # 添加固定边
    workflow.add_edge("web_search", "generate")  # 网络搜索后直接生成回答
    workflow.add_edge("generate_direct", END)  # 直接回答：无需检索与判分
    workflow.add_edge("retrieve", "grade_documents")  # 检索后评估文档相关性

    # 文档评估后的条件分支
    workflow.add_conditional_edges(
        "grade_documents",
        _traced_router("decide_to_generate", decide_to_generate),
        # 显式给出映射：`fallback` 是新增的降级分支，不写死会出现隐式边
        {"generate": "generate", "transform_query": "transform_query", "fallback": "generate_fallback"},
    )
    workflow.add_edge("generate_fallback", END)  # 降级作答不做幻觉判分（没有资料可对照）

    # 生成结果评估后的条件分支
    workflow.add_conditional_edges(
        "generate",  # 生成节点
        _traced_router("grade_generation", grade_generation_v_documents_and_question),
        {
            "not supported": "generate",  # 生成不符合要求时重试
            "useful": END,  # 生成符合要求时结束
            "not useful": "transform_query",  # 生成无用结果时优化查询
            # [FIX 2026-11] 熔断终止边：生成与改写两侧都用尽后直接结束，
            # 否则图里存在一个**没有出口**的环（详见 MAX_RECURSION 的说明）。
            "give_up": END,
        },
    )

    workflow.add_edge("transform_query", "retrieve")  # 查询优化后重新检索

    # 编译工作流
    # [ROADMAP 4.41] 传 checkpointer 时才启用多轮记忆（默认 None = 无记忆）
    return workflow.compile(checkpointer=checkpointer)


# [2026-11] 模块级编译改为**惰性单例**。
#
# 原先 `graph = create_graph()` 在 import 阶段执行，而 create_graph 会 import 全部
# 节点模块（其中 history_store → milvus_db → embeddings_model 会拉起 1.3GB 模型），
# 实测 `import rag_kb.graph_crag.crag_graph` 18.48s。
#
# 现在 import 只定义函数；首次 `graph` 访问或 `get_graph()` 调用时才编译。
# **注意：这是「无记忆」的图** —— 单轮调用、评测、回归都走它。
_graph = None
_graph_lock = threading.Lock()


def get_graph():
    """无记忆图（惰性单例）。"""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = create_graph()
    return _graph


def __getattr__(name):
    """PEP 562：兼容 `from rag_kb.graph_crag.crag_graph import graph`。"""
    if name == "graph":
        return get_graph()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        from rag_kb.utils.checkpointer import create_checkpointer

        _memory_graph = create_graph(checkpointer=create_checkpointer())
    return _memory_graph


def run_with_trace(question: str, enabled: bool = True, save: bool = True, thread_id: str | None = None) -> dict:
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
        # [ROADMAP 4.45] 刷新会话活跃时间（**滑动 TTL** 的依据）。
        # 用 ensure_session 而非 touch_session：客户端可能自生成 UUID 而没走
        # `/sessions/new`，upsert 语义既宽容又不丢会话目录。
        session_store.ensure_session(thread_id, question[:200])
        # 注入 `session_id` 作为「记忆模式」的**显式标记**（`generate` 用它判断
        # 是否追加历史）。不能用 `messages is not None` —— LangGraph 会把它初始化为 []。
        #
        # `messages=[]` 只是首轮的初值；`add_messages` reducer 会把它与
        # checkpoint 读回的历史合并（实测：传 [] 不会覆盖已有历史）。
        inputs = {
            "question": question,
            # [FIX 2026-11] 改写节点永远读它，避免「在上一轮改写结果上再改写」
            "original_question": question,
            "messages": [],
            "session_id": str(thread_id),
        }
        config = {**base_config, "configurable": {"thread_id": str(thread_id)}, "recursion_limit": MAX_RECURSION}
    else:
        g = get_graph()
        inputs = {"question": question, "original_question": question}
        config = {**base_config, "recursion_limit": MAX_RECURSION}

    try:
        for output in g.stream(inputs, config=config):
            for _node, value in output.items():
                if isinstance(value, dict):
                    final_state.update(value)
    except GraphRecursionError:
        # 兜底：正常情况下 `give_up` 边会先结束流程，走到这里说明环的上限设置偏小
        # 或新增了未考虑的回边。**不抛给调用方**，用已有增量结果返回。
        log.warning("---达到 recursion_limit=%d，强制结束本次调用（可能存在未预期的环）---" % MAX_RECURSION)

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
            pass  # 读不到就退回增量拼装的结果，不阻断主流程

    gen = final_state.get("generation", "")
    t.result = gen

    # [ROADMAP 4.44] 记忆模式：把本轮问答写入**历史向量集合**。
    #
    # 为什么放在这里而不是 `generate` 节点内：`generate` 可能因幻觉判定被
    # 重试（`not supported -> generate`），写在节点里会导致**重复入库**；
    # 而在图结束后写，拿到的 `gen` 就是最终答案，**一轮只写一次**。
    #
    # turn 号由 messages 条数推：messages 是「一问一答」成对的，
    # 写完本轮后条数 = 2 × 轮次，所以 `// 2` 即当前轮次。
    #
    # [2026-11] **改为后台串行写入**：`record_turn` 内部要走一遍 embedding 编码
    # （~0.5s）再写 Milvus，实测单次 **2.39s**。原先同步执行，这 2.4 秒直接加在
    # 用户等待上 —— 问一句「你好」也要多等这么久。
    #
    # 为什么用 max_workers=1 而不是并发：`MilvusVectorSave` 是模块级单例，
    # 多线程同时写同一集合没有验证过；单线程队列既保证顺序，又完全不阻塞主流程。
    # 线程池是非 daemon 的，解释器退出时 CPython 会 join，不会丢最后一轮。
    if thread_id and gen:
        _turn = len(final_state.get("messages") or []) // 2
        _HISTORY_WRITER.submit(history_store.record_turn, thread_id, _turn, question, gen)

    if save:
        t.save()
    return {
        "generation": gen,
        "trace": t.to_dict(),
        "documents": final_state.get("documents", []),
        # [FIX / 演示暴露] **必须返回 citations** —— `generate` 产出的
        # 「编号 -> 原文块」映射（ROADMAP 2.1 引用溯源）此前没被带出来，
        # 导致调用方拿到的引用列表**恒为空**，答案里的 `[1]` 无从校验来源。
        "citations": final_state.get("citations", []),
        # [2026-11] 答案是否有知识库/联网依据；见 crag_state.grounded 的三态说明
        "grounded": final_state.get("grounded", False),
        "source_type": final_state.get("source_type", "unknown"),
        # [ROADMAP 4.41] 供调用方检查多轮历史是否在累积（单轮模式下恒为空）
        "messages": final_state.get("messages", []),
        "session_id": thread_id,
    }


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
            inputs = {"question": question}
            # 流式执行工作流
            for output in get_graph().stream(inputs):
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
