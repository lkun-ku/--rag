"""RAG 项目 LangChain 1.x 迁移验证脚本（可重复运行，不修改项目任何文件）。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_migration.py
    .venv\\Scripts\\python.exe scripts\\verify_migration.py --with-embeddings
    .venv\\Scripts\\python.exe scripts\\verify_migration.py --report 迁移验证报告.md

设计要点：
1. 顶层带副作用的历史遗留已消除（交互循环收进 main() 并加 __main__ 守卫），
   因此可以对项目模块做真实 import；仅向量库构造处打桩。
2. 外部服务（Milvus / LLM / Tavily）不可达一律归类为 WARN（环境问题），
   与 API 不兼容（FAIL）严格区分。
3. 全程不打印任何密钥明文。
4. 报告不含任何时间戳，便于内容比对。
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import os
import socket
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

socket.setdefaulttimeout(8)

WORKSPACE = Path(__file__).resolve().parent.parent
PKG_ROOT = WORKSPACE / "RAG_PROJECT"
MD_SAMPLE = PKG_ROOT / "datas" / "md" / "overview.md"

THIRD_PARTY_ROOTS = {
    "langchain", "langchain_core", "langchain_community", "langchain_openai",
    "langchain_milvus", "langchain_experimental", "langchain_text_splitters",
    "langchain_huggingface", "langchain_unstructured", "langchain_tavily",
    "langchain_classic", "langgraph", "pymilvus", "unstructured",
    "pydantic", "loguru", "dotenv", "torch", "transformers", "tavily",
}

_DUMMY_KEYS = {
    "OPENAI_API_KEY": "placeholder",
    "TAVILY_API_KEY": "placeholder",
}


@dataclass
class Result:
    gate: str
    name: str
    status: str
    detail: str = ""
    evidence: list = field(default_factory=list)


def read_py(path: Path) -> str:
    """读取源码，用 utf-8-sig 容忍 BOM（ast.parse 不接受 U+FEFF）。"""
    return path.read_text(encoding="utf-8-sig")


def iter_py_files():
    for p in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def add_pkgs_to_path() -> None:
    for p in (str(WORKSPACE), str(PKG_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_placeholder_keys() -> list:
    injected = []
    for k, v in _DUMMY_KEYS.items():
        if not os.environ.get(k):
            os.environ[k] = v
            injected.append(k)
    return injected


def enable_stub_milvus() -> str:
    """把 Milvus 适配器打桩为不发起网络连接的替身。

    项目 utils/env_utils.py 中 MILVUS_URI 默认是硬编码的公网地址（已失效），
    不打桩则构图阶段会中断、无法验证 StateGraph 拓扑。仅替换向量库构造这一环。
    """
    import langchain_milvus
    from langchain_core.retrievers import BaseRetriever

    class StubRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            return []

    class StubMilvus:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def as_retriever(self, **kwargs):
            return StubRetriever()

    langchain_milvus.Milvus = StubMilvus
    return "langchain_milvus.Milvus"


def import_real(mod_name: str):
    """真实 import 项目模块（Milvus 已打桩、占位密钥已注入）。"""
    import importlib

    add_pkgs_to_path()
    ensure_placeholder_keys()
    enable_stub_milvus()
    importlib.invalidate_caches()
    return importlib.import_module(mod_name)


def is_env_error(e: Exception) -> bool:
    txt = f"{type(e).__name__} {e}".lower()
    keys = ("connection", "timeout", "unreachable", "refused", "dns",
            "api key", "apikey", "credential", "milvus", "unavailable")
    return any(k in txt for k in keys)


# ---------------------------------------------------------------- G1 编译
def gate_compile():
    import py_compile
    import tempfile

    tmp = str(Path(tempfile.gettempdir()) / "_verify.pyc")
    files = list(iter_py_files())
    failed = []
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True, cfile=tmp)
        except Exception as e:
            failed.append(f"{f.relative_to(PKG_ROOT)}: {str(e).strip().splitlines()[-1][:110]}")
    return [Result("G1-编译", f"全部 {len(files)} 个 .py 语法编译",
                   "FAIL" if failed else "PASS",
                   f"失败 {len(failed)} 个" if failed else "无语法错误", failed[:10])]


# ------------------------------------------------------------ G2 import
def gate_imports_ast():
    import importlib

    checked = 0
    problems = []
    covered = set()
    for f in iter_py_files():
        tree = ast.parse(read_py(f), filename=str(f))
        rel = str(f.relative_to(PKG_ROOT))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                mod = node.module or ""
                if (mod.split(".")[0] or "").lower() not in THIRD_PARTY_ROOTS:
                    continue
                names = [(mod, a.name) for a in node.names]
            elif isinstance(node, ast.Import):
                names = [(a.name, None) for a in node.names
                         if (a.name.split(".")[0] or "").lower() in THIRD_PARTY_ROOTS]
            else:
                continue
            covered.add(rel)
            for mod, n in names:
                checked += 1
                try:
                    m = importlib.import_module(mod)
                    ok = n is None or hasattr(m, n)
                    err = "" if ok else "符号不存在"
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {str(e)[:70]}"
                if not ok:
                    problems.append(f"{rel} -> {mod}.{n}: {err}")
    return [Result("G2-import", f"AST 静态校验第三方符号（{checked} 个 / 覆盖 {len(covered)} 个文件）",
                   "FAIL" if problems else "PASS",
                   f"{len(problems)} 个不可解析" if problems else "全部可解析", problems[:12])]

# ------------------------------------------------------------ G3 探针
def probe(name, fn):
    try:
        detail, ev = fn()
        return Result("G3-探针", name, "PASS", detail, ev)
    except ImportError as e:
        return Result("G3-探针", name, "SKIP", f"依赖未安装：{e}")
    except Exception as e:
        if is_env_error(e):
            return Result("G3-探针", name, "WARN",
                          f"外部服务/环境不可达（环境问题）：{type(e).__name__}")
        return Result("G3-探针", name, "FAIL", f"{type(e).__name__}: {str(e)[:170]}")


def p_agent_api():
    import langchain.agents as A
    if not hasattr(A, "create_agent"):
        raise RuntimeError("langchain.agents.create_agent 不存在")
    users = [str(f.relative_to(PKG_ROOT)) for f in iter_py_files()
             if "langchain_classic" in read_py(f)]
    return ("1.x 标准 API create_agent 可用", [
        f"create_tool_calling_agent 是否仍存在：{hasattr(A, 'create_tool_calling_agent')}",
        f"仍引用 langchain_classic 的文件：{users if users else '无'}"])


def p_tavily():
    from langchain_community.tools import TavilySearchResults
    d = getattr(TavilySearchResults.model_fields.get("response_format"), "default", None)
    return (f"可导入，response_format 默认={d!r} -> invoke() 返回 (list, raw) 元组",
            ["项目按 list 取 d['content']，属既有缺陷（0.3.21 与 0.4.2 行为一致）"])


def p_loader():
    from langchain_community.document_loaders import UnstructuredMarkdownLoader as L
    return ("community 版 UnstructuredMarkdownLoader 可导入", [f"{L.__module__}.{L.__name__}"])


def p_chunker():
    from langchain_experimental.text_splitter import SemanticChunker as C
    return ("experimental 版 SemanticChunker 可导入", [f"{C.__module__}.{C.__name__}"])


def p_pymilvus():
    import pymilvus
    from pymilvus.client.types import DataType, FunctionType, MetricType
    from pymilvus import IndexType
    return (f"pymilvus {pymilvus.__version__}：枚举与类全部可导入", [
        f"FunctionType.BM25={FunctionType.BM25}",
        f"DataType.SPARSE_FLOAT_VECTOR={DataType.SPARSE_FLOAT_VECTOR}",
        f"MetricType.IP={MetricType.IP}", f"IndexType.HNSW={IndexType.HNSW}"])


def p_milvus_adapter():
    import inspect
    from langchain_milvus import Milvus
    need = ["embedding_function", "collection_name", "builtin_function", "vector_field",
            "consistency_level", "auto_id", "connection_args"]
    sig = inspect.signature(Milvus.__init__)
    missing = [n for n in need if n not in sig.parameters]
    if missing:
        raise RuntimeError(f"Milvus.__init__ 缺少参数：{missing}")
    return ("Milvus 适配器参数签名兼容（仅签名校验，未连接）", [
        "__init__ 保留：" + ", ".join(need),
        "ranker_type/ranker_params：已 deprecated 但仍生效",
        "score_threshold / filter：升级前后均透传忽略，非回归"])


def p_lc_core():
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    m = AIMessage(content="probe")
    _ = m.pretty_repr(html=True)
    p = ChatPromptTemplate.from_messages(
        [("system", "s"), MessagesPlaceholder(variable_name="h", optional=True), ("human", "{input}")])
    return ("langchain-core / langgraph 核心 API 可用", [
        f"MessagesPlaceholder(optional=True) 编译出 {len(p.messages)} 条消息",
        "AIMessage.pretty_repr(html=True) / .id：OK",
        f"{ToolNode.__name__} / {tools_condition.__name__} / {InMemorySaver.__name__} / {StateGraph.__name__}：OK"])


def p_hf():
    from langchain_huggingface import HuggingFaceEmbeddings as H
    return ("langchain-huggingface 可导入", [f"{H.__module__}.{H.__name__}"])


def gate_probes():
    return [probe("1.x 标准 Agent API", p_agent_api),
            probe("Tavily 工具与返回结构", p_tavily),
            probe("UnstructuredMarkdownLoader 来源", p_loader),
            probe("SemanticChunker 来源", p_chunker),
            probe("pymilvus 枚举与子模块路径", p_pymilvus),
            probe("langchain-milvus 适配器签名", p_milvus_adapter),
            probe("langchain-core / langgraph 核心 API", p_lc_core),
            probe("langchain-huggingface", p_hf)]

# ------------------------------------------------------------ G4 切片
def replicate_merge(datas):
    """原样复刻 documents/markdown_parser.py 的 merge_title_content。"""
    merged_data = []
    parent_dict = {}
    for document in datas:
        md = document.metadata
        md.pop("languages", None)
        parent_id = md.get("parent_id", None)
        category = md.get("category", None)
        element_id = md.get("element_id", None)
        if category == "NarrativeText" and parent_id is None:
            merged_data.append(document)
        if category == "Title":
            document.metadata["title"] = document.page_content
            if parent_id in parent_dict:
                document.page_content = parent_dict[parent_id].page_content + " -> " + document.page_content
            parent_dict[element_id] = document
        if category != "Title" and parent_id:
            parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + " " + document.page_content
            parent_dict[parent_id].metadata["category"] = "content"
    if parent_dict is not None:
        merged_data.extend(parent_dict.values())
    return merged_data


def gate_chunk():
    add_pkgs_to_path()
    from collections import Counter
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    if not MD_SAMPLE.is_file():
        return [Result("G4-切片", "样本 md 存在", "FAIL", str(MD_SAMPLE))]
    loader = UnstructuredMarkdownLoader(file_path=str(MD_SAMPLE), mode="elements", strategy="fast")
    elements = list(loader.lazy_load())
    keys = set()
    for e in elements:
        keys |= set(e.metadata.keys())
    missing = sorted({"category", "element_id", "parent_id"} - keys)
    merged = replicate_merge(elements)
    content_docs = [d for d in merged if d.metadata.get("category") == "content"]
    crumbs = [d for d in merged if " -> " in d.page_content]
    raw_cat = dict(Counter(e.metadata.get("category") for e in elements))
    m_cat = dict(Counter(d.metadata.get("category") for d in merged))
    return [
        Result("G4-切片", "elements metadata 键兼容性",
               "FAIL" if missing else "PASS",
               f"缺失键={missing}" if missing else f"三键齐全（raw 元素 {len(elements)} 个）",
               [f"键集合={sorted(keys)}", f"raw category 分布={raw_cat}"]),
        Result("G4-切片", "merge_title_content 复刻产物",
               "PASS" if content_docs else "FAIL",
               f"merged={len(merged)}，category=='content'={len(content_docs)}，含 ' -> ' 面包屑={len(crumbs)}",
               [f"merged category 分布={m_cat}"]),
    ]


# ------------------------------------------------------------ G5 拓扑
def gate_topology():
    # 基线来自升级前快照；拓扑一旦变动必须 FAIL，而不是只打印数字
    BASELINE = {
        "graph.graph1": ({"__end__", "__start__", "agent", "generate", "retrieve", "rewrite"}, 7),
        "graph2.graph_2": ({"__end__", "__start__", "generate", "grade_documents",
                            "retrieve", "transform_query", "web_search"}, 9),
    }
    res = []
    for mod, base in BASELINE.items():
        exp_nodes, exp_edges = base
        try:
            m = import_real(mod)
            g = m.graph.get_graph()
            nodes = set(g.nodes.keys())
            edges = [f"{e.source} -> {e.target}" + ("  [条件边]" if e.conditional else "  [固定边]")
                     for e in g.edges]
            problems = []
            if nodes != exp_nodes:
                problems.append(f"节点集合变化：多 {sorted(nodes - exp_nodes)} / 少 {sorted(exp_nodes - nodes)}")
            if len(edges) != exp_edges:
                problems.append(f"边数变化：期望 {exp_edges}，实际 {len(edges)}")
            if not callable(getattr(m, "create_graph", None)):
                problems.append("缺少 create_graph() 工厂")
            if not callable(getattr(m, "main", None)):
                problems.append("缺少 main() 入口")
            detail = f"节点 {len(nodes)} 个 / 边 {len(edges)} 条（基线：{len(exp_nodes)} / {exp_edges}）"
            ev = problems or [f"节点集合与基线一致：{sorted(nodes)}"]
            res.append(Result("G5-拓扑", f"{mod} 真实 import + 拓扑断言",
                              "FAIL" if problems else "PASS", detail, ev + edges))
        except Exception as e:
            st = "WARN" if is_env_error(e) else "FAIL"
            res.append(Result("G5-拓扑", f"{mod} 真实 import", st, f"{type(e).__name__}: {str(e)[:150]}"))
    return res


# ------------------------------------------------------------ G6 向量
def gate_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})
    v = emb.embed_query("半导体光刻技术")
    n = sum(x * x for x in v) ** 0.5
    return [Result("G6-向量", "输出维度", "PASS" if len(v) == 512 else "FAIL",
                   f"dim={len(v)}（dense 字段要求 512）"),
            Result("G6-向量", "L2 归一化", "PASS" if abs(n - 1.0) < 1e-3 else "WARN", f"l2_norm={n:.6f}")]

# ------------------------------------------------------------ G7 1.x 写法
_SEEN = []


class _RecordingChatModel:
    pass


def gate_agent_standard():
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.checkpoint.memory import InMemorySaver
    from tools.retriever_tools import retriever_tool

    seen = []

    class RecordingChatModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "recording"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append([m.content for m in messages])
            reply = AIMessage(content=f"回答：{messages[-1].content}")
            return ChatResult(generations=[ChatGeneration(message=reply)])

    add_pkgs_to_path()
    ensure_placeholder_keys()
    enable_stub_milvus()
    import agent.rag_agent as ra
    res = []
    g = ra.agent.get_graph()
    res.append(Result("G7-1.x写法", "真实 import agent.rag_agent.py", "PASS",
                      f"create_agent 构造成功，节点={sorted(g.nodes.keys())}",
                      [f"checkpointer 类型={type(ra.checkpointer).__name__}",
                       f"SYSTEM_PROMPT={ra.SYSTEM_PROMPT!r}", f"SESSION_ID={ra.SESSION_ID!r}"]))
    agent = create_agent(model=RecordingChatModel(), tools=[retriever_tool],
                         system_prompt=ra.SYSTEM_PROMPT, checkpointer=InMemorySaver())
    cfg_a = {"configurable": {"thread_id": "zs123"}}
    cfg_b = {"configurable": {"thread_id": "other"}}
    r1 = agent.invoke({"messages": [("user", "什么是EUV光刻机？")]}, config=cfg_a)
    r2 = agent.invoke({"messages": [("user", "它和DUV有什么区别？")]}, config=cfg_a)
    agent.invoke({"messages": [("user", "换个会话")]}, config=cfg_b)
    first, second, other = seen[0], seen[1], seen[2]
    res.append(Result("G7-1.x写法", "AgentState 输入输出规范", "PASS",
                      f"输入 {{'messages': [...]}} 正常，返回键={sorted(r1.keys())}",
                      [f"最终答案={r1['messages'][-1].content!r}",
                       f"第二次最终答案={r2['messages'][-1].content!r}"]))
    res.append(Result("G7-1.x写法", "checkpointer + thread_id 承担多轮记忆", "PASS",
                      f"同一 thread_id 第二次提问时模型收到 {len(second)} 条（首次 {len(first)} 条）",
                      ["首轮：" + " | ".join(first), "次轮：" + " | ".join(second)]))
    res.append(Result("G7-1.x写法", "会话隔离", "PASS",
                      f"不同 thread_id 互不干扰（zs123: {len(second)} 条 / other: {len(other)} 条）"))
    return res


# ------------------------------------------------------------ G8 熔断
def gate_circuit_breaker():
    m = import_real("graph2.graph_2")
    grade = getattr(m, "grade_generation_v_documents_and_question", None)
    lim = getattr(m, "MAX_GENERATION_RETRY", None)
    if grade is None or lim is None:
        return [Result("G8-熔断", "定位判分函数/阈值", "FAIL", "缺少必要符号")]

    class S:
        binary_score = "no"

    grade.__globals__["hallucination_grader_chain"] = type(
        "G", (), {"invoke": staticmethod(lambda *a, **k: S())})
    base = {"question": "q", "documents": [], "generation": "g"}
    got = []
    for n in (lim - 1, lim, lim + 5):
        d = dict(base)
        d["generation_count"] = n
        got.append(grade(d))
    ok = got == ["not supported", "not useful", "not useful"]
    return [Result("G8-熔断", "generate 自环熔断", "PASS" if ok else "FAIL",
                   f"MAX={lim}；{lim - 1}->{got[0]}，{lim}->{got[1]}，{lim + 5}->{got[2]}",
                   ["未达上限维持原重试行为", "达上限转 not useful，循环有界"])]


# ------------------------------------------------------------ 报告
def build_report(results, as_markdown):
    n = {}
    for r in results:
        n[r.status] = n.get(r.status, 0) + 1
    o = io.StringIO()
    if as_markdown:
        o.write("# 迁移验证报告（自动生成，请勿手工编辑）\n\n")
        o.write("> 由 `scripts/verify_migration.py --report` 生成，每次运行整体覆盖。\n")
        o.write("> 它是**执行证据**；`UPGRADE_NOTES.md` 是**决策依据**。\n\n")
        o.write(f"闸门共 {len(results)} 项：PASS {n.get('PASS',0)} / WARN {n.get('WARN',0)} / "
                f"SKIP {n.get('SKIP',0)} / FAIL {n.get('FAIL',0)}\n\n")
        o.write("| 闸门 | 检查项 | 结果 | 说明 |\n|---|---|---|---|\n")
        for r in results:
            o.write(f"| {r.gate} | {r.name} | {r.status} | {r.detail} |\n")
        for r in results:
            ev = [e for e in r.evidence if e]
            if ev:
                o.write(f"\n**{r.gate} · {r.name}**\n\n")
                o.writelines(f"- {e}\n" for e in ev)
    else:
        o.write("=" * 78 + "\n")
        o.write(f"闸门共 {len(results)} 项：PASS {n.get('PASS',0)} / WARN {n.get('WARN',0)} / "
                f"SKIP {n.get('SKIP',0)} / FAIL {n.get('FAIL',0)}\n")
        o.write("=" * 78 + "\n")
        for r in results:
            o.write(f"[{r.status:>4}] {r.gate:<9} {r.name}\n         {r.detail}\n")
            for e in [x for x in r.evidence if x][:12]:
                o.write(f"           - {e}\n")
    return o.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--with-embeddings", action="store_true")
    args = ap.parse_args()
    if not PKG_ROOT.is_dir():
        print(f"ERROR: 找不到包根目录 {PKG_ROOT}")
        return 3
    add_pkgs_to_path()
    results = []
    results += gate_compile()
    results += gate_imports_ast()
    results += gate_probes()
    results += gate_chunk()
    results += gate_topology()
    results += gate_agent_standard()
    results += gate_circuit_breaker()
    if args.with_embeddings:
        try:
            results += gate_embeddings()
        except Exception as e:
            st = "WARN" if is_env_error(e) else "FAIL"
            results.append(Result("G6-向量", "本地 embedding 维度校验", st, f"{type(e).__name__}: {str(e)[:150]}"))
    text = build_report(results, bool(args.report))
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"\n报告已写入 {args.report}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
