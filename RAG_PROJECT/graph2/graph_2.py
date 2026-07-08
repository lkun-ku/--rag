from pprint import pprint

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from draw_png import draw_graph
from graph2.generate_node2 import generate
from graph2.grade_answer_chain import answer_grader_chain
from graph2.grade_documents_node import grade_documents
from graph2.grade_hallucinations_chain import hallucination_grader_chain
from graph2.graph_state2 import GraphState
from graph2.query_route_chain import question_router_chain
from graph2.retriever_node import retrieve
from graph2.transform_query_node import transform_query
from graph2.web_search_node import web_search
from utils.log_utils import log


# 生成答案的最大重试次数。
# 原实现中 "not supported" -> generate 是无条件自环，一旦幻觉检测持续失败就会无限循环
# （transform_count 只保护了 transform_query 分支）。达到上限后改走 "not useful"，
# 该分支受 transform_count 保护，整体循环次数因此有界。
MAX_GENERATION_RETRY = 2


def grade_generation_v_documents_and_question(state):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
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
    check = validate_citations(generation, len(doc_list) or 1)

    if check["is_refusal"]:
        # 设计内拒答：资料不足时的**正确**响应，不应触发熔断重试
        # （若在这里重试，只会反复得到同样的拒答，白白消耗 LLM 调用）。
        log.info("---判定：模型按规则拒答（资料不足），结束流程---")
        return "useful"

    if check["invalid"]:
        generation_count = state.get("generation_count", 0)  # 熔断
        log.warning(f"---判定：引用编号越界 {check['invalid']}（幻觉引用），已生成 {generation_count} 次---")
        if generation_count >= MAX_GENERATION_RETRY:
            # 熔断耗尽：改走受 transform_count 保护的分支，避免无限循环
            return "not useful"
        return "not supported"

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":  # 如果生成基于文档
        log.info("---判定：生成内容基于参考文档---")
        # 检查是否准确回答问题
        log.info("---评估：生成回答与问题的匹配度---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
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


def route_question(state):
    """
    路由问题到网络搜索或RAG流程
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        str: 下一节点的名称（web_search或vectorstore）
    """
    log.info("---ROUTE QUESTION---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    source = question_router_chain.invoke({"question": question})  # 调用问题路由器

    # 根据路由结果决定下一个节点
    if source.datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"
    elif source.datasource == "vectorstore":
        log.info("---路由到RAG系统---")
        return "vectorstore"


def create_graph():
    """构建并编译工作流图，返回已编译的图对象。

    依据 LangGraph 官方 Application structure 文档，langgraph.json 的 graphs 入口
    可以是「已编译的图对象」，也可以是「一个用于创建图的函数」。本函数即后者，
    使构图逻辑可被独立调用与测试。

    节点、条件边与熔断阈值均与迁移前一致。
    """
    # 初始化工作流图
    workflow = StateGraph(GraphState)

    # 定义各状态节点
    workflow.add_node("web_search", web_search)  # 网络搜索节点
    workflow.add_node("retrieve", retrieve)  # 文档检索节点
    workflow.add_node("grade_documents", grade_documents)  # 文档相关性评分节点
    workflow.add_node("generate", generate)  # 回答生成节点
    workflow.add_node("transform_query", transform_query)  # 查询优化节点

    # 起始路由判断
    workflow.add_conditional_edges(
        START,
        route_question,
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
        decide_to_generate
    )

    # 生成结果评估后的条件分支
    workflow.add_conditional_edges(
        "generate",  # 生成节点
        grade_generation_v_documents_and_question,  # 生成质量评估函数
        {
            "not supported": "generate",  # 生成不符合要求时重试
            "useful": END,  # 生成符合要求时结束
            "not useful": "transform_query",  # 生成无用结果时优化查询
        },
    )

    workflow.add_edge("transform_query", "retrieve")  # 查询优化后重新检索

    # 编译工作流
    return workflow.compile()


# 模块级导出已编译的图，保持原有 `from graph2.graph_2 import graph` 的用法不变
graph = create_graph()

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
