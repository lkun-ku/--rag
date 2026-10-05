"""评测主流程：跑 Golden Dataset，输出四项指标与门禁判定。

用法：
    $env:PYTHONPATH="."

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

import pathlib as _pl
import sys as _sy

# 解释器护栏：裸 python（Anaconda）跑本脚本会 ModuleNotFoundError，自动切到项目 .venv
_HERE = _pl.Path(__file__).resolve().parent
if str(_HERE) not in _sy.path:
    _sy.path.insert(0, str(_HERE))
import venv_guard  # noqa: E402

venv_guard.ensure_venv()

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Sequence

HERE = pathlib.Path(__file__).resolve().parent
# 用 `import metrics` 而不是 `from eval import metrics`：`python eval/run_eval.py` 启动时
# 脚本目录本身就在 sys.path[0]，无需手工插入。`eval` 作包名还会与内置函数同名。


os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from rag_kb.utils.env_utils import MILVUS_URI  # noqa: E402

GOLDEN_DATASET = HERE / "golden_dataset.jsonl"   # 手工题集（绑定旧语料，语料变了会失效）
GOLDEN_AUTO = HERE / "golden_auto.jsonl"         # 自动题集（gen_testset 生成，与当前向量库对齐）


def _pick_golden(explicit: str, candidates: Sequence[pathlib.Path]) -> pathlib.Path:
    """评测集选择：显式 --golden > 自动题（与当前向量库对齐）> 手工题集 > 第一个存在的。"""
    if explicit:
        p = pathlib.Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"指定的评测集不存在: {p}")
        return p
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("没有找到任何评测集（候选: %s）" % ", ".join(str(p) for p in candidates))


def _output_paths(gold: pathlib.Path) -> tuple:
    """baseline/history 跟随评测集派生：不同题集的基线**互不覆盖**。"""
    if gold.name == "golden_dataset.jsonl":
        return HERE / "baseline.json", HERE / "history.jsonl"
    tag = gold.stem.replace("golden_", "") or "auto"
    return HERE / f"baseline_{tag}.json", HERE / f"history_{tag}.jsonl"


def goldens_of(case: dict) -> list:
    """题目的「应召回目标」：要点优先，其次块 ID。两者都空 = 拒答题。"""
    return list(case.get("key_points") or []) + list(case.get("source_blocks") or [])


def load_golden(limit: int = 0, path=None):
    src = pathlib.Path(path) if path else GOLDEN_DATASET
    cases = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 项目评测（Golden Dataset）")
    ap.add_argument(
        "--with-generation", action="store_true", help="额外评测生成层（Faithfulness / Answer Relevancy），需要 LLM"
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--verbose", action="store_true", help="打印每题详情")
    ap.add_argument("--no-cache", action="store_true", help="忽略判定缓存")
    ap.add_argument("--no-save", action="store_true", help="不写 baseline/history（用于验证性测试，避免污染正式基线）")
    ap.add_argument(
        "--golden", default="", help="评测集路径（默认 golden_dataset.jsonl；自动生成题为 golden_auto.jsonl）"
    )
    args = ap.parse_args()

    candidates = [pathlib.Path(args.golden)] if args.golden else [GOLDEN_AUTO, GOLDEN_DATASET]
    try:
        gold_path = _pick_golden(args.golden, candidates)
    except FileNotFoundError as e:
        print(str(e))
        if not args.golden:
            print("生成方式（换语料后）：")
            print("    python eval/export_chunks.py                    # 导出块清单")
            print("    python eval/gen_testset.py --per-collection 3   # 自动出题 -> eval/golden_auto.jsonl")
        return 2

    cases = load_golden(args.limit, gold_path)
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
    from rag_kb.tools.retriever_tools import retriever  # noqa: E402

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
        golden_ids = goldens_of(case)  # 要点 + 块 ID 都记录，诊断信息更完整
        row = {"id": case.get("id", f"#{i}"), "source": case.get("source", "?"), "q": q, "golden_blocks": golden_ids}

        # Layer 0
        # [2026-11] 改用 `score_case`：判据优先用 `key_points`（与语料内容解耦），
        # 没有时才退回 `source_blocks`；拒答题返回 recall=None（不参与聚合）。
        try:
            docs = retriever.invoke(q)
            recalled = [
                {
                    "id": metrics.chunk_id(d.page_content),
                    "text": d.page_content,
                    "filename": str((getattr(d, "metadata", None) or {}).get("filename") or ""),
                }
                for d in docs
            ]
            s = metrics.score_case(case, recalled)
        except Exception as e:  # noqa: BLE001
            print(f"  [{row['id']}] 检索失败: {type(e).__name__}: {str(e)[:120]}")
            is_refusal = not goldens_of(case)
            row.update(
                {
                    "recall": None if is_refusal else 0.0,
                    "precision": 0.0,
                    "hit": 0.0,
                    "basis": "refusal" if is_refusal else "error",
                    "error": str(e)[:120],
                }
            )
            per_case.append(row)
            continue

        row.update(
            {
                "recall": s["recall"],
                "precision": s["precision"],
                "hit": s["hit"],
                "basis": s["basis"],
                "doc_hit": s["doc_hit"],
                "retrieved_n": len(recalled),
                "missed": s["missed_points"],
            }
        )
        per_case.append(row)

        if args.verbose:
            if s["recall"] is None:
                print(f"  [{row['id']}] 拒答题（不参与召回统计）| {q[:40]}")
            else:
                flag = "OK " if s["recall"] == 1.0 else ("部分" if s["recall"] > 0 else "未命中")
                extra = " doc=%s" % ("hit" if s["doc_hit"] else "miss") if s["doc_hit"] is not None else ""
                print(
                    f"  [{row['id']}] R={s['recall']:.2f} ({flag}) [{s['basis']}]{extra} | {q[:40]}"
                )
                if s["missed_points"]:
                    print(f"        漏召回: {s['missed_points']}")

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
                    # 题目与回答一一对应展示，肉眼即可核对「答得对不对」
                    _gen = (gen or "").replace("\n", " ")
                    _gold = (case.get("answer") or "").replace("\n", " ")
                    print("        " + "─" * 60)
                    print(f"        题: {q[:70]}")
                    print(
                        f"        答({len(gen)}字): {_gen[:220]}{'…' if len(_gen) > 220 else ''}"
                    )
                    if _gold:
                        print(f"        参考: {_gold[:70]}")
                    print(
                        f"        判分: faith={f['raw']}({f['value']:.0f}) "
                        f"relev={r['raw']}({r['value']:.0f})"
                        f"{' [生成缓存]' if gen_cached else ''}"
                        f"{' [判定缓存]' if f['cached'] else ''}"
                    )
            except metrics.LLMUnavailable as e:
                l1_errors.append(str(e)[:140])
            except Exception as e:  # noqa: BLE001
                l1_errors.append(f"{type(e).__name__}: {str(e)[:140]}")

    elapsed = time.time() - t0

    # ---- 汇总 ----
    l0 = metrics.aggregate_layer0(per_case)
    n_refusal = l0.get("refusal_cases", 0)
    n_scored = len(per_case) - n_refusal
    scores = {
        "context_recall": l0["context_recall"],
        "hit_rate": l0["hit_rate"],
        "context_precision": l0.get("context_precision"),
        "faithfulness": (sum(l1["faithfulness"]) / len(l1["faithfulness"]) if l1["faithfulness"] else None),
        "answer_relevancy": (
            sum(l1["answer_relevancy"]) / len(l1["answer_relevancy"]) if l1["answer_relevancy"] else None
        ),
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
        if v is None:
            vs = "N/A"
            # None 有两种语义，必须区分说明（否则用户会以为检索层也出问题了）：
            #   生成层指标 -> 本次未启用 / LLM 不可用
            #   context_precision（要点口径）-> 该口径下指标无定义
            if k in ("faithfulness", "answer_relevancy"):
                why = "LLM 不可用" if args.with_generation else "未启用（加 --with-generation）"
                label = "%s（%s）" % (labels[k], why)
            else:
                label = labels[k] + "（要点口径下无定义）"
        else:
            vs = "%.4f" % v
            label = labels[k]
        print("%-20s %10s %10.2f %10.2f   %s" % (k, vs, thk["fail"], thk["warn"], label))

    verdict, reasons = metrics.gate(scores, gen_requested=args.with_generation)

    print()
    print("分母说明: 可评测 %d 道；拒答题 %d 道（不参与召回统计）" % (n_scored, n_refusal))
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
            scored = [r for r in rows if r["recall"] is not None]
            if not scored:
                print("  %-14s %3d 条  拒答题，不参与召回统计" % (s, len(rows)))
                continue
            avg = sum(r["recall"] for r in scored) / len(scored)
            print("  %-14s %3d 条  平均 Recall=%.4f" % (s, len(scored), avg))

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
        "per_source": {
            s: {
                "n": len([r for r in rows if r.get("recall") is not None]),
                "recall": (
                    sum(r["recall"] for r in rows if r.get("recall") is not None)
                    / max(1, len([r for r in rows if r.get("recall") is not None]))
                ),
                # 要点口径的题 precision 为 None，不参与平均
                "precision": (
                    sum(r["precision"] for r in rows if r.get("precision") is not None)
                    / max(1, len([r for r in rows if r.get("precision") is not None]))
                    if any(r.get("precision") is not None for r in rows)
                    else None
                ),
                "refusal_n": len([r for r in rows if r.get("recall") is None]),
            }
            for s, rows in by_src.items()
        },
        "per_case": per_case,
    }
    # [FIX] `--limit N` 是「只跑前 N 条」的调试模式：它的指标只代表子集，
    # 若照常写入 baseline 就会把正式基线覆盖成片面的数字（实测踩过：
    # `--limit 5` 把 context_recall 覆盖成 1.0、verdict 覆盖成 INCONCLUSIVE）。
    # 这里强制当作验证性运行，避免污染基线。
    #
    # [FIX 2026-11 同类第二处] `verdict == INCONCLUSIVE`（生成层未评测）的**不完整结果
    # 也不该落盘** —— 实测：不带 --with-generation 跑一次，baseline.json 被覆盖成
    # 「faithfulness/answer_relevancy 缺失」的残缺版，history.jsonl 追加了一条
    # INCONCLUSIVE 记录。基线/历史是用来对比「有没有变差」的，残缺记录混进去后
    # 「抖动 vs 持续回退」的判断就失真了。与项目原则一致：**未完成 ≠ 不合格，但未完成也不能当基线**。
    save_blocked = args.no_save or bool(args.limit) or verdict == "INCONCLUSIVE"
    if save_blocked:
        print()
        if args.no_save:
            print("[--no-save] 未写入 baseline/history（验证性运行）")
        elif args.limit:
            print(f"[--limit {args.limit}] 只跑了子集，已自动跳过 baseline/history 写入（避免污染正式基线）")
        else:
            print(
                "[未落盘] 门禁为 INCONCLUSIVE（生成层缺失），不完整结果不写入 baseline/history ——"
                "\n         补 `--with-generation` 得到完整结论后再落盘"
            )
    else:
        baseline_path, history_path = _output_paths(gold_path)
        baseline_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in result.items() if k != "per_case"}, ensure_ascii=False) + "\n")
        print()
        print(f"已写入 {baseline_path.name} 与追加 {history_path.name}")

    if verdict == "FAIL":
        return 1
    if verdict == "INCONCLUSIVE":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
