"""per_k 配额的全集 recall 对比：8/24(现行) vs 12/36 vs 16/48。

## 为什么要跑这个（而不是直接改）

`diag_n026_perk.py` 证明配额截断真实存在（n026 的 golden 排第 9、被 per_k=8 截掉），
但**四道题都显示收益递减**，且代价明确：上下文块数从 24 涨到 46~63，
判分的 LLM 调用量随之翻倍。所以要用**全 32 题**的口径拍板。

## 口径

**与 `run_eval` 一致**（`metrics.score_retrieval` + `aggregate_layer0`），
这样才能与基线 0.8501 直接比 —— 沿用 4.14.7 的教训：**口径不一致的两个数字不能比**。

检索层是确定性的（不调 LLM），所以顺序跑三配置即可，无需交替。
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
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
import tools.retriever_tools as rt  # noqa: E402

GRID = [(6, 18), (8, 24), (12, 36), (16, 48)]
cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

print("=" * 104)
print("per_k 配额的全集 recall 对比（口径与 run_eval 一致）")
print("=" * 104)
print("  题数 %d（含 golden %d）" % (len(cases), sum(1 for c in cases if c.get("source_blocks"))))
print()

out = {}
for pk, fk in GRID:
    r = rt._build_multi_retriever(per_k=pk, final_k=fk)
    t0 = time.time()
    per_case, errs = [], []
    for c in cases:
        try:
            docs = r.invoke(c["q"])
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
    avg_n = sum(x["retrieved_n"] for x in per_case) / len(per_case)
    out[(pk, fk)] = {"l0": l0, "per_case": per_case, "avg_n": avg_n, "errs": errs}
    print(
        "  per_k=%-3d final_k=%-3d  recall=%.4f  hit=%.4f  precision=%.4f  "
        "平均块数=%.1f  (%.1fs)"
        % (pk, fk, l0["context_recall"], l0["hit_rate"], l0["context_precision"], avg_n, time.time() - t0)
    )

print()
print("=" * 104)
print("对比（以现行 8/24 为基准）")
print("=" * 104)
print("  %-12s %-10s %-10s %-12s %s" % ("配置", "recall", "Δ", "平均块数", "判读"))
print("  " + "-" * 96)
base = out[(8, 24)]
for pk, fk in GRID:
    v = out[(pk, fk)]
    d = v["l0"]["context_recall"] - base["l0"]["context_recall"]
    tag = (
        ""
        if (pk, fk) == (8, 24)
        else ("基准" if abs(d) < 1e-9 else "有效" if d > 0.01 else "无明显收益" if d > -0.005 else "**下降**")
    )
    print(
        "  %-12s %-10.4f %+-10.4f %-12.1f %s"
        % (
            "%d/%d" % (pk, fk),
            v["l0"]["context_recall"],
            d,
            v["avg_n"],
            "**基准（当前）**" if (pk, fk) == (8, 24) else tag,
        )
    )

print()
print("  判读要点：放大配额只在「Δrecall 明显 > 0」时值得，因为块数增加的代价是")
print("  **判分的 LLM 调用量同比例上升**（每块一次判分）。")

(HERE / "compare_perk_recall.json").write_text(
    json.dumps(
        {
            "%d/%d" % k: {"l0": v["l0"], "avg_n": v["avg_n"], "errs": v["errs"], "per_case": v["per_case"]}
            for k, v in out.items()
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/compare_perk_recall.json")
