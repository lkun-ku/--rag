"""32 题标准评测：确认 `per_k=8 / final_k=24` 在**全集**上也成立（ROADMAP 4.13.3 第 1 步）。

## 为什么必须再跑一次全集

A/B（`verify_final_k_ab.py`）只用了 **6 道稳定题**，而且是「有 golden」的题 ——
样本小、且不代表全集。这里用 `run_eval.py` 的**同一套口径**（`metrics.score_retrieval`
+ `metrics.aggregate_layer0`）跑全部 32 题，保证与基线 `0.7492` 可比。

## 为什么不用 `scan_per_collection_k.py`

那个脚本自带 `evaluate()`，**口径与 `run_eval.py` 不同**（它只统计有 golden 的题）。
要跟基线对得上，就必须用 `metrics` 里那套 —— 这也是「**测量口径必须一致**」的又一次应用。

## 检索层是确定性的

不调 LLM，所以**不存在服务漂移**（4.12.5 那个坑）—— 顺序跑两配置即可，无需交替。
"""

from __future__ import annotations

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
import tools.retriever_tools as rt  # noqa: E402

CONFIGS = [(6, 18), (8, 24)]  # (per_collection_k, final_k)

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print("=" * 100)
print("32 题标准评测对比（口径与 run_eval.py 一致）")
print("=" * 100)
print("  题数: %d（其中含 golden 的 %d 道）" % (len(cases), sum(1 for c in cases if c.get("source_blocks"))))

results = {}
for pk, fk in CONFIGS:
    retr = rt._build_multi_retriever(per_k=pk, final_k=fk)
    t0 = time.time()
    per_case = []
    errs = []
    for c in cases:
        try:
            docs = retr.invoke(c["q"])
            rids = [metrics.chunk_id(d.page_content) for d in docs]
            s = metrics.score_retrieval(c.get("source_blocks") or [], rids)
        except Exception as e:  # noqa: BLE001
            errs.append("%s: %s" % (c.get("id"), type(e).__name__))
            s = {"recall": 0.0, "precision": 0.0, "hit": False, "missed_ids": []}
            rids = []
        per_case.append(
            {
                "id": c.get("id"),
                "source": c.get("source", "?"),
                "recall": s["recall"],
                "precision": s["precision"],
                "hit": s["hit"],
                "retrieved_n": len(rids),
            }
        )
    l0 = metrics.aggregate_layer0(per_case)
    results[(pk, fk)] = {"l0": l0, "per_case": per_case, "errs": errs}
    print()
    print(
        "  per_k=%-3d final_k=%-3d  recall=%.4f  hit_rate=%.4f  precision=%.4f  平均块数=%.1f"
        % (
            pk,
            fk,
            l0["context_recall"],
            l0["hit_rate"],
            l0["context_precision"],
            sum(r["retrieved_n"] for r in per_case) / len(per_case),
        )
    )
    print("                      耗时 %.1fs | 检索失败 %d 题" % (time.time() - t0, len(errs)))
    if errs:
        print("                      %s" % errs[:3])

# ---- 对比 ----
a, b = results[CONFIGS[0]], results[CONFIGS[1]]
print()
print("=" * 100)
print("对比")
print("=" * 100)
print("  %-14s %-12s %-12s %-12s %s" % ("指标", "6/18（当前）", "8/24", "Δ", "判定"))
print("  " + "-" * 92)
for key, label, better in (
    ("context_recall", "context_recall", "up"),
    ("hit_rate", "hit_rate", "up"),
    ("context_precision", "context_precision", "up"),
):
    va, vb = a["l0"][key], b["l0"][key]
    d = vb - va
    ok = "✅ 提升" if d > 0.001 else ("≈ 持平" if abs(d) <= 0.001 else "❌ 下降")
    print("  %-14s %-12.4f %-12.4f %+-12.4f %s" % (label, va, vb, d, ok))

na = sum(r["retrieved_n"] for r in a["per_case"]) / len(cases)
nb = sum(r["retrieved_n"] for r in b["per_case"]) / len(cases)
print("  %-14s %-12.1f %-12.1f %+-12.1f %s" % ("平均块数", na, nb, nb - na, ""))

# 逐题变化（只看有 golden 的题）
print()
print("=" * 100)
print("逐题变化（有 golden 的题；只列 recall 有变的）")
print("=" * 100)
ma = {r["id"]: r for r in a["per_case"]}
mb = {r["id"]: r for r in b["per_case"]}
up = down = 0
for cid in sorted(ma):
    if cid not in mb:
        continue
    ra, rb = ma[cid]["recall"], mb[cid]["recall"]
    if abs(ra - rb) < 0.001:
        continue
    flag = "↑" if rb > ra else "↓"
    if rb > ra:
        up += 1
    else:
        down += 1
    print(
        "  %s %-6s recall %.3f -> %.3f   块数 %d -> %d   [%s]"
        % (
            flag,
            cid,
            ra,
            rb,
            ma[cid]["retrieved_n"],
            mb[cid]["retrieved_n"],
            (next((c["q"] for c in cases if c.get("id") == cid), ""))[:44],
        )
    )
print()
print("  提升 %d 题 / 下降 %d 题" % (up, down))

(HERE / "final_k_full.json").write_text(
    json.dumps(
        {
            "configs": CONFIGS,
            "n_cases": len(cases),
            "results": {
                "%d/%d" % k: {"l0": v["l0"], "per_case": v["per_case"], "errs": v["errs"]} for k, v in results.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/final_k_full.json")
