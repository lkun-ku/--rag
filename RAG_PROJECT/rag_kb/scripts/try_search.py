"""试检索：交互式看新语料的检索与问答效果。

三种用法：

    # 1. 交互模式（推荐）—— 输入问题回车，看检索到的块；q 退出
    .venv\\Scripts\\python.exe RAG_PROJECT\\scripts\\try_search.py

    # 2. 单次检索
    .venv\\Scripts\\python.exe RAG_PROJECT\\scripts\\try_search.py -q "Milvus 支持哪些索引类型？"

    # 3. 完整问答（走 graph2 工作流：路由 -> 检索 -> 判分 -> 生成 -> 引用校验）
    .venv\\Scripts\\python.exe RAG_PROJECT\\scripts\\try_search.py --ask -q "Milvus 支持哪些索引类型？"

交互模式下可用开关（输入 `:help` 查看）：
    :ask      切换「只检索 / 完整问答」
    :k N      调整每集合召回条数（默认 6）
    :trace    切换是否打印执行轨迹
    :q        退出
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"


def print_retrieval(docs, show_full: bool = False):
    """按集合/文件分组打印检索结果。"""
    groups: dict = {}
    for i, d in enumerate(docs, 1):
        fn = d.metadata.get("filename", "?")
        groups.setdefault(fn, []).append((i, d))
    print("\n  检索到 %d 块，来自 %d 个文件：" % (len(docs), len(groups)))
    for fn, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("\n  ┌─ %s（%d 块）" % (fn, len(items)))
        for i, d in items:
            txt = d.page_content.replace("\n", " ")
            limit = 300 if show_full else 96
            print("  │ [%2d] %s" % (i, txt[:limit]))


def print_answer(res: dict):
    """打印完整问答结果（含引用映射）。"""
    print("\n  ── 答案 " + "─" * 70)
    for line in str(res.get("answer", "")).splitlines():
        print("  %s" % line)
    cites = res.get("citations") or []
    if cites:
        print("\n  ── 引用来源 " + "─" * 66)
        for c in cites:
            print("  [%s] %-40s | %s" % (c.get("n"), str(c.get("filename", ""))[:40], str(c.get("title", ""))[:34]))
    print("\n  用时 %.1fs | 拒答: %s" % (res.get("elapsed_sec", 0), res.get("refused")))


def do_retrieval(q: str, show_full: bool):
    from rag_kb.tools.retriever_tools import retriever

    t0 = time.time()
    docs = retriever.invoke(q)
    print("\n  [检索] %.1fs" % (time.time() - t0))
    print_retrieval(docs, show_full)


def do_ask(q: str, trace: bool):
    from rag_kb.graph2.graph_2 import run_with_trace

    t0 = time.time()
    r = run_with_trace(q, save=False)
    t = r.get("trace") or {}
    docs = r.get("documents") or []
    print("\n  [完整链路] %.1fs | 上下文 %d 块" % (time.time() - t0, len(docs)))
    if trace:
        print("\n  ── 执行轨迹 " + "─" * 66)
        for s in t.get("steps", []):
            extra = {k: v for k, v in s.items() if k not in ("node", "t", "elapsed_sec")}
            print(
                "  %6.2fs  %-18s %6.2fs  %s"
                % (s.get("t", 0), s.get("node", ""), s.get("elapsed_sec", 0), str(extra)[:70])
            )
    gen = {
        "answer": r.get("generation", ""),
        "citations": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "refused": None,
    }
    from rag_kb.utils.citation import build_citation_meta, is_refusal, parse_citations

    answer = r.get("generation", "") or ""
    cites = parse_citations(answer)
    if cites and docs:
        meta = build_citation_meta(docs, cites)
        gen["citations"] = [
            {"n": m.get("n"), "filename": m.get("filename", ""), "title": m.get("title", "")} for m in meta
        ]
        print("\n  ── 上下文里的块（编号即引用编号）" + "─" * 50)
        for i, d in enumerate(docs, 1):
            mark = "⭐" if i in cites else "  "
            print(
                "  %s[%2d] %-32s | %s"
                % (mark, i, d.metadata.get("filename", "?")[:32], d.page_content.replace("\n", " ")[:70])
            )
    gen["refused"] = is_refusal(answer)
    print_answer(gen)


def main() -> int:
    ap = argparse.ArgumentParser(description="试检索 / 试问答")
    ap.add_argument("-q", "--question", default="", help="直接给一个问题（不给则进交互模式）")
    ap.add_argument("--ask", action="store_true", help="走完整问答链路（含生成与引用）")
    ap.add_argument("--trace", action="store_true", help="打印执行轨迹（需 --ask）")
    ap.add_argument("--full", action="store_true", help="块内容显示更多（300 字）")
    args = ap.parse_args()

    from rag_kb.tools.retriever_tools import retriever

    print("=" * 86)
    print("RAG 试检索")
    print("=" * 86)
    colls = list(getattr(retriever, "retrievers", {}).keys())
    print("  覆盖集合: %s" % (colls or "（空库）"))
    print("  模式: %s" % ("完整问答" if args.ask else "仅检索"))
    print()

    if args.question:
        if args.ask:
            do_ask(args.question, args.trace)
        else:
            do_retrieval(args.question, args.full)
        return 0

    mode_ask, mode_trace, mode_full = args.ask, args.trace, args.full
    while True:
        try:
            q = input("问题（:help 看命令，:q 退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            break
        if q == ":help":
            print(
                "  :ask 切换模式（当前 %s） | :trace 切换轨迹（当前 %s） | "
                ":full 切换全文（当前 %s） | :q 退出" % ("问答" if mode_ask else "检索", mode_trace, mode_full)
            )
            continue
        if q == ":ask":
            mode_ask = not mode_ask
            print("  模式 -> %s" % ("完整问答" if mode_ask else "仅检索"))
            continue
        if q == ":trace":
            mode_trace = not mode_trace
            print("  轨迹 -> %s" % mode_trace)
            continue
        if q == ":full":
            mode_full = not mode_full
            print("  全文显示 -> %s" % mode_full)
            continue
        try:
            if mode_ask:
                do_ask(q, mode_trace)
            else:
                do_retrieval(q, mode_full)
        except Exception as e:  # noqa: BLE001
            import traceback

            print("  !! %s: %s" % (type(e).__name__, str(e)[:200]))
            traceback.print_exc()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
