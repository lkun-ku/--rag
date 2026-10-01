"""拒答率审计（ROADMAP 3.0.1）—— 补齐 2.1.4 唯一未跑的验收项。

## 为什么需要它

引用系统（2.1）上线后，「该拒答时是否真拒答」「是否过度拒答」**完全没有数据**：
  - 过度拒答：用户问得到答案，却被告知「资料不足」—— 直接伤害可用性
  - 拒答失效：知识库外的问题被强行编造 —— 这正是 2.1 要防的核心风险

## 验收标准（2.1.4）

  - 知识库**外**问题拒答率 >= 95%
  - 知识库**内**问题拒答率 <= 20%

## 为什么走 generate 而不是跑完整工作流

与 `eval/metrics.generate_answer` 保持一致：评测要**控制变量**，上下文就是本次检索
到的那批块。跑完整工作流会引入路由决策与 web_search 降级，让「检索」与「生成」
纠缠在一起，无法归因。

**关键**：`generate_answer` 复用的是 `graph2/generate_node2.generate`，即**真实
生成链**（含 2.1 的 CITE_PROMPT 与引用校验），因此这里测到的拒答行为就是线上行为。

用法：
    $env:PYTHONPATH="RAG_PROJECT"
    .venv\\Scripts\\python.exe eval\\refusal_audit.py
    .venv\\Scripts\\python.exe eval\\refusal_audit.py --verbose --show-answers
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402

GOLDEN = HERE / "golden_dataset.jsonl"
REPORT = HERE / "refusal_audit.json"

# ---- 库外问题：明确不在知识库覆盖范围内 ----
# 覆盖多个领域，避免「某类问题恰好撞到检索噪声」被误判为拒答失效。
OUT_OF_SCOPE = [
    "今天北京的天气怎么样？",
    "红烧肉的家常做法是什么？",
    "2026 年世界杯在哪个国家举办？",
    "帮我写一首关于春天的短诗。",
    "Python 的 GIL 是什么，为什么会限制多线程性能？",
    "特斯拉最新一个交易日的收盘价是多少？",
    "推荐几本适合入门的小说。",
    "如何在线预约驾驶证考试？",
    "量子计算的 Shor 算法原理是什么？",
    "可口可乐的完整配方是什么？",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="拒答率审计（库外 + 库内）")
    ap.add_argument("--verbose", action="store_true", help="逐条打印")
    ap.add_argument("--show-answers", action="store_true", help="打印答案前 90 字")
    ap.add_argument("--no-cache", action="store_true", help="忽略判定/生成缓存")
    ap.add_argument("--limit-in", type=int, default=0, help="库内只跑前 N 条（调试）")
    args = ap.parse_args()

    print("=" * 90)
    print("拒答率审计（ROADMAP 3.0.1）")
    print("=" * 90)

    print("[健康检查] LLM 探活中...")
    llm_ok, llm_msg = metrics.llm_health_check()
    print("  结果: %s —— %s" % ("可用" if llm_ok else "不可用", llm_msg))
    if not llm_ok:
        print("\n  [!] LLM 不可用，无法判定拒答行为。这不是质量回退。")
        return 3
    print()

    cache = metrics.JudgeCache(enabled=not args.no_cache)

    from rag_kb.tools.retriever_tools import retriever  # noqa: E402
    from rag_kb.utils.citation import is_refusal  # noqa: E402

    cases = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit_in:
        cases = cases[: args.limit_in]

    t0 = time.time()
    results = {"in_scope": [], "out_of_scope": []}

    def run_one(q: str, label: str) -> dict:
        docs = retriever.invoke(q)
        gen, gen_cached = metrics.generate_answer(q, docs, cache)
        refused = is_refusal(gen)
        return {
            "label": label,
            "q": q,
            "refused": refused,
            "n_docs": len(docs),
            "answer": gen,
            "generated_cached": gen_cached,
        }

    print("=" * 90)
    print("A. 库外问题（期望：拒答率 >= 95%%，共 %d 题）" % len(OUT_OF_SCOPE))
    print("=" * 90)
    for i, q in enumerate(OUT_OF_SCOPE, 1):
        try:
            row = run_one(q, "out")
        except Exception as e:  # noqa: BLE001
            print("  [%2d] %-34s 失败: %s" % (i, q[:32], type(e).__name__))
            continue
        results["out_of_scope"].append(row)
        flag = "拒答" if row["refused"] else "**未拒答**"
        print(
            "  [%2d] %-34s %-12s (检索到 %d 块)%s"
            % (i, q[:32], flag, row["n_docs"], " [缓存]" if row["generated_cached"] else "")
        )
        if args.show_answers or (not row["refused"] and args.verbose):
            print("        答案: %s" % row["answer"].replace("\n", " ")[:90])
    print()

    print("=" * 90)
    print("B. 库内问题（期望：拒答率 <= 20%%，共 %d 题）" % len(cases))
    print("=" * 90)
    for i, c in enumerate(cases, 1):
        q = c["q"]
        try:
            row = run_one(q, c.get("source", "?"))
        except Exception as e:  # noqa: BLE001
            print("  [%2d] %s 失败: %s" % (i, c.get("id"), type(e).__name__))
            continue
        row["id"] = c.get("id")
        results["in_scope"].append(row)
        if row["refused"] or args.verbose:
            flag = "**拒答**" if row["refused"] else "正常"
            print("  [%2d] %-6s %-40s %s" % (i, c.get("id"), q[:38], flag))
            if row["refused"] and args.show_answers:
                print("        答案: %s" % row["answer"].replace("\n", " ")[:90])
    print()

    # ---- 汇总 ----
    out = results["out_of_scope"]
    inn = results["in_scope"]
    out_rate = (sum(1 for r in out if r["refused"]) / len(out)) if out else 0.0
    in_rate = (sum(1 for r in inn if r["refused"]) / len(inn)) if inn else 0.0

    out_pass = out_rate >= 0.95
    in_pass = in_rate <= 0.20

    print("=" * 90)
    print("汇总")
    print("=" * 90)
    print("  %-16s %-10s %-10s %-10s %s" % ("范围", "题数", "拒答数", "拒答率", "判定"))
    print("  " + "-" * 76)
    print(
        "  %-16s %-10d %-10d %-10.1f%% %s"
        % (
            "库外（期望>=95%）",
            len(out),
            sum(1 for r in out if r["refused"]),
            out_rate * 100,
            "PASS" if out_pass else "**FAIL**",
        )
    )
    print(
        "  %-16s %-10d %-10d %-10.1f%% %s"
        % (
            "库内（期望<=20%）",
            len(inn),
            sum(1 for r in inn if r["refused"]),
            in_rate * 100,
            "PASS" if in_pass else "**FAIL**",
        )
    )
    print()
    print("  耗时: %.1fs | 生成缓存命中 %d" % (time.time() - t0, cache.hits))

    if not out_pass:
        print()
        print("  [!] 库外拒答率不达标 —— 以下问题被强行回答（可能编造）：")
        for r in out:
            if not r["refused"]:
                print("      - %s" % r["q"])
                print("        -> %s" % r["answer"].replace("\n", " ")[:110])
    if not in_pass:
        print()
        print("  [!] 库内拒答率过高 —— 以下问题被误拒（可用性受损）：")
        for r in inn:
            if r["refused"]:
                print("      - [%s] %s" % (r.get("id"), r["q"]))
                print("        -> %s" % r["answer"].replace("\n", " ")[:110])

    REPORT.write_text(
        json.dumps(
            {
                "out_of_scope": {
                    "n": len(out),
                    "refused": sum(1 for r in out if r["refused"]),
                    "rate": round(out_rate, 4),
                    "pass": out_pass,
                    "detail": [
                        {k: v for k, v in r.items() if k != "answer"} | {"answer": r["answer"][:300]} for r in out
                    ],
                },
                "in_scope": {
                    "n": len(inn),
                    "refused": sum(1 for r in inn if r["refused"]),
                    "rate": round(in_rate, 4),
                    "pass": in_pass,
                    "detail": [
                        {k: v for k, v in r.items() if k != "answer"} | {"answer": r["answer"][:300]} for r in inn
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print("  报告已写入 %s" % REPORT.name)
    print("=" * 90)

    return 0 if (out_pass and in_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
