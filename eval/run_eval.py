"""评测主流程：跑 Golden Dataset，输出四项指标与门禁判定。

用法：
    $env:PYTHONPATH="RAG_PROJECT"

    # 只跑检索层（默认）—— 0 LLM 调用、0 抖动、秒级
    .venv\\Scripts\\python.exe eval\\run_eval.py

    # 额外跑生成层（需要可用的 LLM，会调用 LLM 生成 + 判定）
    .venv\\Scripts\\python.exe eval\\run_eval.py --with-generation

    # 调试：只跑前 5 条 / 显示每题详情 / 忽略缓存
    .venv\\Scripts\\python.exe eval\\run_eval.py --limit 5 --verbose --no-cache

产物：
    eval/baseline.json   本次结果（会被覆盖）
    eval/history.jsonl   历史记录（追加，用于区分「抖动」与「持续回退」）
    eval/cache.jsonl     LLM 判定缓存（消除重复判定的抖动与成本）

退出码：
    0 = PASS / WARN
    1 = FAIL（确定性指标未达标）
    3 = INCONCLUSIVE（LLM 不可用，生成层缺失 —— **不是**质量回退）
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# 把 eval/ 目录本身放进 sys.path，用 `import metrics` 而不是 `from eval import metrics` ——
# `eval` 与 Python 内置函数同名，作为包名容易引起困扰（也可能与第三方包冲突）。
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402

GOLDEN = HERE / "golden_dataset.jsonl"
BASELINE = HERE / "baseline.json"
HISTORY = HERE / "history.jsonl"


def load_golden(limit: int = 0):
    cases = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 项目评测（Golden Dataset）")
    ap.add_argument("--with-generation", action="store_true",
                    help="额外评测生成层（Faithfulness / Answer Relevancy），需要 LLM")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--verbose", action="store_true", help="打印每题详情")
    ap.add_argument("--no-cache", action="store_true", help="忽略判定缓存")
    ap.add_argument("--no-save", action="store_true",
                    help="不写 baseline/history（用于验证性测试，避免污染正式基线）")
    args = ap.parse_args()

    if not GOLDEN.exists():
        print(f"缺少 {GOLDEN}，请先运行 eval/build_dataset.py")
        return 2

    cases = load_golden(args.limit)
    print("=" * 84)
    print("RAG 评测（Golden Dataset）")
    print("=" * 84)
    print(f"用例数     : {len(cases)}")
    print(f"Milvus     : {MILVUS_URI}")
    print(f"生成层评测 : {'开启' if args.with_generation else '关闭（仅检索层）'}")
    print()

    # ---- LLM 探活（仅在需要生成层时）----
    llm_ok, llm_msg = True, "未使用"
    if args.with_generation:
        print("[健康检查] LLM 探活中...")
        llm_ok, llm_msg = metrics.llm_health_check()
        print(f"  结果: {'可用' if llm_ok else '不可用'} —— {llm_msg}")
        if not llm_ok:
            print()
            print("  [!] LLM 不可用 -> 生成层指标将标记为 SKIPPED，")
            print("      门禁判定为 INCONCLUSIVE（退出码 3），**不计为质量回退**。")
        print()

    cache = metrics.JudgeCache(enabled=not args.no_cache) if args.with_generation else None

    # ---- 检索器 ----
    from tools.retriever_tools import retriever  # noqa: E402

    covered = list(getattr(retriever, "retrievers", {}).keys()) or ["(默认集合)"]
    print(f"[检索器] {type(retriever).__name__}  覆盖集合: {covered}")
    print()

    # ---- 逐条评测 ----
    t0 = time.time()
    per_case = []
    l1 = {"faithfulness": [], "answer_relevancy": []}
    l1_errors = []

    for i, case in enumerate(cases, 1):
        q = case["q"]
        golden_ids = case["source_blocks"]
        row = {"id": case.get("id", f"#{i}"), "source": case.get("source", "?"),
               "q": q, "golden_blocks": golden_ids}

        # Layer 0
        try:
            docs = retriever.invoke(q)
            rids = [metrics.chunk_id(d.page_content) for d in docs]
            s = metrics.score_retrieval(golden_ids, rids)
        except Exception as e:  # noqa: BLE001
            print(f"  [{row['id']}] 检索失败: {type(e).__name__}: {str(e)[:120]}")
            row.update({"recall": 0.0, "precision": 0.0, "error": str(e)[:120]})
            per_case.append(row)
            continue

        row.update({"recall": s["recall"], "precision": s["precision"],
                    "hit": s["hit"], "retrieved_n": len(rids),
                    "missed": s["missed_ids"]})
        per_case.append(row)

        if args.verbose:
            flag = "OK " if s["recall"] == 1.0 else ("部分" if s["recall"] > 0 else "未命中")
            print(f"  [{row['id']}] R={s['recall']:.2f} P={s['precision']:.2f} "
                  f"({flag}) | {q[:44]}")
            if s["missed_ids"]:
                print(f"        漏召回: {s['missed_ids']}")

        # Layer 1（可选）
        if args.with_generation and llm_ok:
            try:
                gen, gen_cached = metrics.generate_answer(q, docs, cache)
                ctx = "\n\n".join(d.page_content for d in docs)
                f = metrics.judge_faithfulness(q, gen, ctx, cache)
                r = metrics.judge_answer_relevancy(q, gen, cache)
                l1["faithfulness"].append(f["value"])
                l1["answer_relevancy"].append(r["value"])
                if args.verbose:
                    print(f"        faith={f['raw']}({f['value']:.0f}) "
                          f"relev={r['raw']}({r['value']:.0f})"
                          f"{' [生成缓存]' if gen_cached else ''}"
                          f"{' [判定缓存]' if f['cached'] else ''}")
            except metrics.LLMUnavailable as e:
                l1_errors.append(str(e)[:140])
            except Exception as e:  # noqa: BLE001
                l1_errors.append(f"{type(e).__name__}: {str(e)[:140]}")

    elapsed = time.time() - t0

    # ---- 汇总 ----
    l0 = metrics.aggregate_layer0(per_case)
    scores = {
        "context_recall": l0["context_recall"],
        "hit_rate": l0["hit_rate"],
        "context_precision": l0["context_precision"],
        "faithfulness": (sum(l1["faithfulness"]) / len(l1["faithfulness"])
                         if l1["faithfulness"] else None),
        "answer_relevancy": (sum(l1["answer_relevancy"]) / len(l1["answer_relevancy"])
                             if l1["answer_relevancy"] else None),
    }

    print()
    print("=" * 84)
    print("指标汇总")
    print("=" * 84)
    th = metrics.THRESHOLDS
    print("%-20s %10s %10s %10s   %s" % ("指标", "分数", "硬门禁", "告警线", "说明"))
    print("-" * 84)
    labels = {
        "context_recall": "检索层（确定性）",
        "hit_rate": "检索层（确定性）",
        "context_precision": "检索层（仅监控）",
        "faithfulness": "生成层（LLM）",
        "answer_relevancy": "生成层（LLM）",
    }
    for k, v in scores.items():
        thk = th[k]
        vs = "SKIPPED" if v is None else "%.4f" % v
        print("%-20s %10s %10.2f %10.2f   %s" % (k, vs, thk["fail"], thk["warn"], labels[k]))

    verdict, reasons = metrics.gate(scores)

    print()
    print(f"耗时: {elapsed:.1f}s")
    if cache:
        print(f"判定缓存: 命中 {cache.hits} / 新增 {cache.misses}")
    if l1_errors:
        print(f"生成层错误 {len(l1_errors)} 条，示例: {l1_errors[0]}")
    print()
    print(f"门禁判定: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 84)

    # ---- 分源统计（验证集合隔离）----
    by_src = {}
    for row in per_case:
        by_src.setdefault(row["source"], []).append(row)
    if len(by_src) > 1:
        print()
        print("分源召回率（验证集合隔离与跨集合检索）")
        print("-" * 84)
        for s, rows in sorted(by_src.items()):
            avg = sum(r["recall"] for r in rows) / len(rows)
            print("  %-14s %3d 条  平均 Recall=%.4f" % (s, len(rows), avg))

    # ---- 落盘 ----
    result = {
        "verdict": verdict,
        "reasons": reasons,
        "scores": scores,
        "cases": len(cases),
        "elapsed_sec": round(elapsed, 1),
        "with_generation": bool(args.with_generation),
        "llm_available": llm_ok,
        "collections": covered,
        "per_source": {s: {"n": len(rows),
                           "recall": sum(r["recall"] for r in rows) / len(rows),
                           "precision": sum(r["precision"] for r in rows) / len(rows)}
                       for s, rows in by_src.items()},
        "per_case": per_case,
    }
    if args.no_save:
        print()
        print("[--no-save] 未写入 baseline/history（验证性运行）")
    else:
        BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in result.items() if k != "per_case"},
                               ensure_ascii=False) + "\n")
        print()
        print(f"已写入 {BASELINE.name} 与追加 {HISTORY.name}")

    if verdict == "FAIL":
        return 1
    if verdict == "INCONCLUSIVE":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
