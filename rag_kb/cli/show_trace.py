"""查看问答执行轨迹（ROADMAP 3.1）。

trace 由 `graph_crag.crag_graph.run_with_trace()` 产出，追加写入 `logs/trace.jsonl`。

用法：
    # 最近 1 条（默认）
    .venv/Scripts/python.exe rag_kb/cli/show_trace.py

    # 最近 5 条
    .venv/Scripts/python.exe rag_kb/cli/show_trace.py -n 5

    # 按总耗时倒序看最慢的 10 条（定位性能问题）
    .venv/Scripts/python.exe rag_kb/cli/show_trace.py --slow -n 10

    # 只看某类事件（如所有走了 web_search 的）
    .venv/Scripts/python.exe rag_kb/cli/show_trace.py --grep web_search

    # 汇总统计（各节点平均耗时、token 总量）
    .venv/Scripts/python.exe rag_kb/cli/show_trace.py --stats
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

from rag_kb.utils.tracer import TRACE_FILE


def load(path: pathlib.Path):
    if not path.exists():
        print("没有找到 trace 文件：%s" % path)
        print("先跑一次 run_with_trace()，例如：")
        print(
            "  python -c \"from rag_kb.graph_crag.crag_graph import run_with_trace;"
            " print(run_with_trace('Milvus 支持哪些索引？')['generation'])\""
        )
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def show_one(r: dict, idx: int):
    print("=" * 96)
    print("[%d] %s" % (idx, r.get("question", "")[:88]))
    tk = r.get("tokens", {})
    tok = (
        "in %d / out %d" % (tk.get("input_tokens", 0), tk.get("output_tokens", 0)) if tk.get("available") else "不可用"
    )
    print("    总耗时 %.2fs | 步骤 %d | tokens %s" % (r.get("total_sec", 0), r.get("n_steps", 0), tok))
    print("-" * 96)
    print("    %-6s %-16s %-9s %s" % ("t(s)", "节点/边", "耗时", "信息"))
    for s in r.get("steps", []):
        info = {k: v for k, v in s.items() if k not in ("node", "t", "elapsed_sec")}
        bits = []
        for k, v in info.items():
            if isinstance(v, list):
                v = "[" + ",".join(map(str, v[:6])) + ("…" if len(v) > 6 else "") + "]"
            bits.append("%s=%s" % (k, v))
        print(
            "    %-6s %-16s %-9s %s"
            % (s.get("t"), s.get("node"), "%.2fs" % s.get("elapsed_sec", 0), " ".join(bits)[:60])
        )
    prev = r.get("result_preview", "")
    if prev:
        print("-" * 96)
        print("    答案: %s" % prev[:150].replace("\n", " "))


def show_stats(rows):
    print("=" * 96)
    print("汇总统计（共 %d 条 trace）" % len(rows))
    print("=" * 96)
    by_node = defaultdict(list)
    for r in rows:
        for s in r.get("steps", []):
            by_node[s["node"]].append(s.get("elapsed_sec", 0))
    print("  %-18s %-8s %-12s %-12s %s" % ("节点/边", "出现次数", "平均耗时", "最大耗时", "占比"))
    total_all = sum(sum(v) for v in by_node.values()) or 1
    for node, vals in sorted(by_node.items(), key=lambda kv: -sum(kv[1])):
        print(
            "  %-18s %-8d %-12s %-12s %.1f%%"
            % (node, len(vals), "%.2fs" % (sum(vals) / len(vals)), "%.2fs" % max(vals), sum(vals) / total_all * 100)
        )
    print()
    tk_in = sum(r.get("tokens", {}).get("input_tokens", 0) for r in rows)
    tk_out = sum(r.get("tokens", {}).get("output_tokens", 0) for r in rows)
    print("  token 合计: 输入 %d / 输出 %d / 总计 %d" % (tk_in, tk_out, tk_in + tk_out))
    tot = [r.get("total_sec", 0) for r in rows]
    if tot:
        print("  单次耗时: 平均 %.2fs | 最快 %.2fs | 最慢 %.2fs" % (sum(tot) / len(tot), min(tot), max(tot)))
    # 路由分布
    routes = defaultdict(int)
    for r in rows:
        d = next((s.get("decision") for s in r.get("steps", []) if s["node"] == "route_question"), None)
        if d:
            routes[d] += 1
    if routes:
        print("  路由分布: %s" % dict(routes))


def main() -> int:
    ap = argparse.ArgumentParser(description="查看问答执行轨迹")
    ap.add_argument("-n", type=int, default=1, help="显示最近 N 条（默认 1）")
    ap.add_argument("--slow", action="store_true", help="按总耗时倒序（看最慢的）")
    ap.add_argument("--grep", default="", help="只显示包含该关键词的 trace")
    ap.add_argument("--stats", action="store_true", help="输出汇总统计")
    ap.add_argument("--file", default=str(TRACE_FILE), help="trace 文件路径")
    args = ap.parse_args()

    rows = load(pathlib.Path(args.file))
    if not rows:
        return 2

    if args.grep:
        rows = [r for r in rows if args.grep in json.dumps(r, ensure_ascii=False)]
        if not rows:
            print("没有匹配 %r 的 trace" % args.grep)
            return 2

    if args.stats:
        show_stats(rows)
        return 0

    if args.slow:
        rows = sorted(rows, key=lambda r: -r.get("total_sec", 0))

    for i, r in enumerate(rows[-args.n :] if not args.slow else rows[: args.n], 1):
        show_one(r, i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
