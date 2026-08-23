"""端到端对比：三个 `per_k` 配置下，**判分后保留的 golden 块数**。

## 为什么用这个指标，而不是 recall

`compare_perk_recall.py` 显示放大配额能提升 recall（0.8501 -> 0.8953），
**但四档的 `hit_rate` 完全相同（0.9375）** —— 没有命中新题，只是覆盖更多 golden 块。

而真正传导到答案的是**判分之后**上下文里还剩什么（4.19 实测判分只保留 20.2%）。
所以决策指标必须是「**判分后保留的 golden 块数**」：多召回的那些块，
**有多少能过判分**。

## 三个配置

    (8, 24)  现行
    (12, 36)
    (16, 48)

## 两个实现要点

1. **复用生产判分函数** `grade_documents_node._grade_one` —— 保证与线上**同一套
   召回/提示词/异常兜底**，避免「测出来的判定逻辑和线上不一样」。
2. **并发**（线程池，度数与生产一致 `MAX_GRADE_WORKERS=8`）——
   三配置合计约 3300 次判分，串行要一个多小时。

## 记录字段

    每题：检索块数 / 判分后块数 / golden 命中数（判分前）/ **golden 保留数（判分后）**
    汇总：保留率、golden 保留率、**被完全筛掉 golden 的题数**（最坏情形）
"""
from __future__ import annotations

import concurrent.futures as cf
import collections
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
from graph2.grade_documents_node import _grade_one, MAX_GRADE_WORKERS  # noqa: E402

GRID = [(8, 24), (12, 36), (16, 48)]
WORKERS = max(1, MAX_GRADE_WORKERS)

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")]

print("=" * 106)
print("端到端对比：判分后保留的 golden 块数（三个 per_k 配置）")
print("=" * 106)
print("  题数 %d（仅含 golden 的题）| 判分并发度 %d | 判分函数复用生产的 _grade_one"
      % (len(cases), WORKERS))
print()

out = {}
for pk, fk in GRID:
    r = rt._build_multi_retriever(per_k=pk, final_k=fk)
    t0 = time.time()
    rows = []
    n_calls = 0
    for ci, c in enumerate(cases, 1):
        gids = set(c["source_blocks"])
        docs = r.invoke(c["q"])
        with cf.ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(docs)))) as ex:
            graded = list(ex.map(lambda d: _grade_one(c["q"], d), docs))
        n_calls += len(docs)
        kept = [d for d, ok in graded if ok]
        ids_ret = {metrics.chunk_id(d.page_content) for d in docs}
        ids_kept = {metrics.chunk_id(d.page_content) for d in kept}
        row = {
            "id": c.get("id"), "n_golden": len(gids),
            "n_ret": len(docs), "n_kept": len(kept),
            "g_ret": len(gids & ids_ret),
            "g_kept": len(gids & ids_kept),
        }
        rows.append(row)
        print("  [%d/%d] %-6s golden=%2d  检索 %2d -> 判分后 %2d | "
              "golden 命中 %d -> **保留 %d**" % (
                  ci, len(cases), row["id"], row["n_golden"], row["n_ret"],
                  row["n_kept"], row["g_ret"], row["g_kept"]))
    el = time.time() - t0
    out[(pk, fk)] = {"rows": rows, "sec": round(el, 1), "calls": n_calls}
    print("  -> %d/%d 完成，%d 次判分，耗时 %.0fs" % (pk, fk, n_calls, el))
    print()

print("=" * 106)
print("汇总")
print("=" * 106)
print("  %-10s %-12s %-14s %-14s %-12s %s" % (
    "配置", "检索块数", "判分后块数", "golden 命中", "golden 保留", "完全筛掉 golden 的题"))
print("  " + "-" * 100)
summary = {}
for pk, fk in GRID:
    v = out[(pk, fk)]["rows"]
    n = len(v)
    avg_ret = sum(x["n_ret"] for x in v) / n
    avg_kept = sum(x["n_kept"] for x in v) / n
    g_ret = sum(x["g_ret"] for x in v)
    g_kept = sum(x["g_kept"] for x in v)
    zero = [x["id"] for x in v if x["g_ret"] > 0 and x["g_kept"] == 0]
    summary["%d/%d" % (pk, fk)] = {
        "avg_ret": round(avg_ret, 1), "avg_kept": round(avg_kept, 1),
        "g_ret": g_ret, "g_kept": g_kept,
        "g_keep_rate": round(g_kept / max(g_ret, 1), 4),
        "zero_golden_cases": zero, "calls": out[(pk, fk)]["calls"],
        "sec": out[(pk, fk)]["sec"],
    }
    print("  %-10s %-12.1f %-14.1f %-14d %-12s %s" % (
        "%d/%d" % (pk, fk), avg_ret, avg_kept, g_ret,
        "%d (%.0f%%)" % (g_kept, 100 * g_kept / max(g_ret, 1)),
        (", ".join(zero) if zero else "无")))

print()
print("=" * 106)
print("判读")
print("=" * 106)
b = summary["8/24"]
for k, v in summary.items():
    if k == "8/24":
        continue
    d_g = v["g_kept"] - b["g_kept"]
    d_c = v["calls"] - b["calls"]
    print("  %-8s vs 8/24:  golden 保留 %+d 块（%d -> %d）| 判分调用 %+d 次（%.2fx）"
          % (k, d_g, b["g_kept"], v["g_kept"], d_c, v["calls"] / max(b["calls"], 1)))
print()
print("  决策依据：**每多保留 1 个 golden 块，要付多少次额外判分调用**。")

(HERE / "compare_perk_after_grade.json").write_text(json.dumps(
    {"grid": GRID, "workers": WORKERS,
     "detail": {"%d/%d" % k: v["rows"] for k, v in out.items()},
     "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/compare_perk_after_grade.json")
