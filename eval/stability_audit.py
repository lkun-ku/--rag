"""评测稳定性审计 —— 量化「多轮抖动」，产出可用于 A/B 的稳定子集。

## 为什么需要它（来自 P0-1 的教训）

P0-1 复核时踩到：**同一 prompt、同一问题（n005），4 轮里作答 1 次、拒答 3 次**，
而**检索块数恒为 18** —— 抖动在生成层。后果是**单轮 A/B 对比不可靠**：
一个抖动样本就能翻转全部指标，使「prompt 改动有效/无效」无法判断。

## 做什么

1. **检索只跑一次、多轮复用** —— 这样能把「生成抖动」从「检索抖动」里分离出来。
   若多轮检索块数不同，说明检索本身也不稳（那是另一个问题，需单独查）。
2. **生成跑 N 轮**（无缓存，真实调用）。
3. 逐题比较三轮的：**拒答状态 / 引用编号 / 答案文本**。
4. 产出：
   - **critical 不稳定题**（拒答或引用编号跨轮不一致）—— 会污染指标，必须识别
   - **answer_text 差异**（仅措辞不同）—— 单独统计，**不算 critical**
     （LLM 措辞变化是正常的，只要意思/引用/拒答一致就不影响指标结论）
   - **稳定子集**（无 critical 抖动）的指标 —— 后续 A/B 应以它为准

## 用法

    .venv\\Scripts\\python.exe eval\\stability_audit.py --rounds 3
    .venv\\Scripts\\python.exe eval\\stability_audit.py --rounds 3 --limit 8   # 快速试点
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from tools.retriever_tools import get_retriever  # noqa: E402
from utils.citation import is_refusal, parse_citations  # noqa: E402

GOLDEN = HERE / "golden_dataset.jsonl"
OUT = HERE / "stability_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="评测稳定性审计")
    ap.add_argument("--rounds", type=int, default=3, help="生成轮数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（快速试点）")
    args = ap.parse_args()

    cases = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[:args.limit]
    print("=" * 96)
    print("评测稳定性审计：题数 %d | 轮数 %d" % (len(cases), args.rounds))
    print("=" * 96)

    # ---- 阶段一：检索（只做一次）----
    retr = get_retriever()
    t0 = time.time()
    print()
    print("阶段一：检索（只做一次，多轮复用 —— 以分离「检索抖动」与「生成抖动」）")
    docs_by_id, n_docs, retr_err = {}, {}, {}
    for c in cases:
        try:
            docs = retr.invoke(c["q"])
        except Exception as e:  # noqa: BLE001
            docs, retr_err[c["id"]] = [], "%s: %s" % (type(e).__name__, str(e)[:80])
        docs_by_id[c["id"]] = docs
        n_docs[c["id"]] = len(docs)
    dist = {}
    for v in n_docs.values():
        dist[v] = dist.get(v, 0) + 1
    print("  完成，耗时 %.1fs | 检索块数分布 %s" % (time.time() - t0, dict(sorted(dist.items()))))
    if retr_err:
        print("  检索失败 %d 题: %s" % (len(retr_err), list(retr_err)[:5]))

    # ---- 阶段二：多轮生成 ----
    print()
    print("阶段二：生成 %d 轮（无缓存，真实调用）" % args.rounds)
    rounds = []
    for r in range(1, args.rounds + 1):
        tr = time.time()
        rec = {}
        for c in cases:
            cid, q = c["id"], c["q"]
            docs = docs_by_id.get(cid) or []
            try:
                gen, _ = metrics.generate_answer(q, docs)
            except Exception as e:  # noqa: BLE001
                gen = ""
                print("    [%s] 生成失败: %s" % (cid, type(e).__name__))
            rec[cid] = {"refused": is_refusal(gen), "cites": parse_citations(gen), "answer": gen}
        rounds.append(rec)
        n_ref = sum(1 for v in rec.values() if v["refused"])
        print("  第 %d 轮: 拒答 %d/%d | 耗时 %.1fs" % (r, n_ref, len(rec), time.time() - tr))

    # ---- 分析 ----
    print()
    print("=" * 96)
    print("逐题跨轮一致性")
    print("=" * 96)
    detail, critical_ids, text_only_ids, stable_ids = {}, [], [], []
    refusal_unstable, cites_unstable = [], []
    for c in cases:
        cid = c["id"]
        states = [rounds[r][cid] for r in range(args.rounds)]
        refusals = [s["refused"] for s in states]
        # [修正 ] 用 **集合** 而非 tuple 比较引用编号。
        # `parse_citations` 是「保序去重」的，而顺序在不同轮之间会变
        # （实测 n025 `[15,12]` vs `[12,15]`、n026 `[3,12,6,15]` vs `[3,12,15,6]`）——
        # 顺序不影响引用是否正确，用 tuple 比会把这类**假抖动**算成 critical。
        cite_sets = [frozenset(s["cites"]) for s in states]
        cites = [tuple(s["cites"]) for s in states]
        answers = [s["answer"] for s in states]
        dims = []
        if len(set(refusals)) > 1:
            dims.append("refusal")
        if len(set(cite_sets)) > 1:
            dims.append("cites")
        if len(set(answers)) > 1:
            dims.append("answer_text")
        critical = [d for d in dims if d != "answer_text"]
        detail[cid] = {
            "q": c["q"][:110], "n_docs": n_docs[cid], "dims": dims,
            "refusals": refusals, "cites": [list(x) for x in cites],
            "answers": [a[:150] for a in answers],
        }
        if critical:
            critical_ids.append(cid)
        elif dims:
            text_only_ids.append(cid)
        else:
            stable_ids.append(cid)
        if "refusal" in dims:
            refusal_unstable.append(cid)
        if "cites" in dims:
            cites_unstable.append(cid)

    print("  稳定（三轮全一致）      : %d 题" % len(stable_ids))
    print("  仅措辞不同（answer_text）: %d 题  <- 不影响指标，单独统计" % len(text_only_ids))
    print("  **critical 抖动**        : %d 题  <- 拒答或引用编号跨轮不一致，会污染指标"
          % len(critical_ids))
    print()
    print("  按维度拆分（两者可能重叠）：")
    print("      拒答状态抖动: %d 题%s" % (
        len(refusal_unstable),
        "  <- 直接污染「拒答率」，最需警惕" if refusal_unstable else "  （稳定）"))
    print("      引用编号抖动: %d 题  （形态通常是「多标/少标一个来源」，"
          "会污染「引用支持度」）" % len(cites_unstable))

    if critical_ids:
        print()
        print("  critical 抖动明细：")
        for cid in critical_ids:
            d = detail[cid]
            print("    [%s] %s" % (cid, d["q"][:72]))
            print("         抖动维度: %s | 检索块数 %d" % (d["dims"], d["n_docs"]))
            print("         拒答: %s" % d["refusals"])
            print("         引用: %s" % d["cites"])
            for i, a in enumerate(d["answers"], 1):
                print("         第%d轮答案: %s" % (i, a.replace("\n", " ")[:96]))

    # ---- 指标：全体 vs 稳定子集 ----
    print()
    print("=" * 96)
    print("指标对比：全体 vs 稳定子集")
    print("=" * 96)
    print("  %-10s %-16s %-16s %s" % ("轮次", "全体拒答率", "稳定子集拒答率", "稳定子集题数"))
    print("  " + "-" * 78)
    per_round = []
    for r in range(args.rounds):
        all_ref = sum(1 for cid in [c["id"] for c in cases] if rounds[r][cid]["refused"])
        st_ref = sum(1 for cid in stable_ids if rounds[r][cid]["refused"])
        per_round.append({"all": all_ref, "stable": st_ref})
        print("  %-10d %-16s %-16s %s" % (
            r + 1,
            "%d/%d" % (all_ref, len(cases)),
            "%d/%d" % (st_ref, len(stable_ids)),
            len(stable_ids)))

    print()
    print("  **全体拒答率的跨轮波动: %d ~ %d 题（相差 %d）**" % (
        min(p["all"] for p in per_round), max(p["all"] for p in per_round),
        max(p["all"] for p in per_round) - min(p["all"] for p in per_round)))
    print("  稳定子集拒答率的跨轮波动: %d ~ %d 题" % (
        min(p["stable"] for p in per_round), max(p["stable"] for p in per_round)))
    print()
    print("  => 后续 A/B 应看**稳定子集**的指标；critical 抖动题须单独跟踪。")

    (OUT).write_text(json.dumps({
        "n_cases": len(cases), "rounds": args.rounds,
        "stable_ids": stable_ids, "answer_text_only_ids": text_only_ids,
        "critical_ids": critical_ids,
        "refusal_unstable_ids": refusal_unstable, "cites_unstable_ids": cites_unstable,
        "per_round_refusals": per_round,
        "retrieval": {"n_docs": n_docs, "errors": retr_err},
        "detail": detail,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("  结果 -> eval/stability_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
