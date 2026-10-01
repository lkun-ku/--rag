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

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")]

print("=" * 106)
print("端到端对比：判分后保留的 golden 块数（三个 per_k 配置）")
print("=" * 106)
print("  题数 %d（仅含 golden 的题）| 判分并发度 %d | 判分函数复用生产的 _grade_one" % (len(cases), WORKERS))
print()

# [FIX / 二次筛查] **不再复用 `_grade_one`，改为显式调链 + 暴露异常**。
#
# 原因：`_grade_one` 在异常时返回 `(document, **True**)`（保守保留）——
# 那是**生产上正确的行为**，但在**测量**里会把「判分失败」混进「判分通过」，
# 于是「放大配额后多保留了 N 个 golden」**无法区分**是：
#     (a) 判分真的放宽了（配额有效），还是
#     (b) 判分调用失败被保守保留（配额无效，只是碰巧留下了）。
# 本脚本的结论曾被用来采纳 `PER_COLLECTION_K=12 / FINAL_K=36`，所以必须能分清。
#
# 现在显式返回三元组 `(doc, verdict, error)`，并在汇总里单列：
#     * `err`        —— 该配置下判分异常的**调用次数**
#     * `g_kept_err` —— 其中属于 golden 的块数（**这才是「假保留」的规模**）
#     * `g_kept_sure`—— 排除异常后的 golden 保留数（**结论应以它为准**）
from graph2.grader_chain import retrieval_grader_chain  # noqa: E402


def _grade_explicit(q: str, d):
    """显式判分：返回 (document, verdict, error)。异常时 verdict=True（与生产一致），
    但**把异常记下来** —— 这样「保留」与「因异常而保留」可以分开统计。"""
    try:
        sc = retrieval_grader_chain.invoke({"question": q, "document": d.page_content})
        v = str(getattr(sc, "binary_score", "")).strip().lower() == "yes"
        return d, v, None
    except Exception as e:  # noqa: BLE001
        return d, True, "%s: %s" % (type(e).__name__, str(e)[:60])


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
            graded = list(ex.map(lambda d: _grade_explicit(c["q"], d), docs))
        n_calls += len(docs)
        ids_ret = {metrics.chunk_id(d.page_content) for d in docs}
        kept = [d for d, ok, _e in graded if ok]
        # 排除「因异常而保留」的那部分 —— 它才是干净的口径
        kept_sure = [d for d, ok, e in graded if ok and e is None]
        err_docs = [d for d, _ok, e in graded if e is not None]
        ids_kept = {metrics.chunk_id(d.page_content) for d in kept}
        ids_kept_sure = {metrics.chunk_id(d.page_content) for d in kept_sure}
        ids_err = {metrics.chunk_id(d.page_content) for d in err_docs}
        row = {
            "id": c.get("id"),
            "n_golden": len(gids),
            "n_ret": len(docs),
            "n_kept": len(kept),
            "n_kept_sure": len(kept_sure),
            "n_err": len(err_docs),
            "g_ret": len(gids & ids_ret),
            "g_kept": len(gids & ids_kept),
            "g_kept_sure": len(gids & ids_kept_sure),
            "g_kept_err": len(gids & ids_err),  # golden 里「因异常而保留」的
        }
        rows.append(row)
        flag = "  ⚠️判分异常 %d 次" % row["n_err"] if row["n_err"] else ""
        print(
            "  [%d/%d] %-6s golden=%2d  检索 %2d -> 判分后 %2d（其中干净 %2d）| "
            "golden 命中 %d -> **保留 %d（干净 %d）**%s"
            % (
                ci,
                len(cases),
                row["id"],
                row["n_golden"],
                row["n_ret"],
                row["n_kept"],
                row["n_kept_sure"],
                row["g_ret"],
                row["g_kept"],
                row["g_kept_sure"],
                flag,
            )
        )
    el = time.time() - t0
    out[(pk, fk)] = {"rows": rows, "sec": round(el, 1), "calls": n_calls}
    _err = sum(x["n_err"] for x in rows)
    _gerr = sum(x["g_kept_err"] for x in rows)
    print(
        "  -> %d/%d 完成，%d 次判分，耗时 %.0fs；**判分异常 %d 次（其中 golden %d 块）**"
        % (pk, fk, n_calls, el, _err, _gerr)
    )
    print()

print("=" * 106)
print("汇总")
print("=" * 106)
print(
    "  %-10s %-10s %-10s %-12s %-14s %-12s %s"
    % ("配置", "检索块数", "判分后", "golden 命中", "golden 保留", "判分异常", "完全筛掉 golden 的题")
)
print("  " + "-" * 100)
summary = {}
for pk, fk in GRID:
    v = out[(pk, fk)]["rows"]
    n = len(v)
    avg_ret = sum(x["n_ret"] for x in v) / n
    avg_kept = sum(x["n_kept"] for x in v) / n
    g_ret = sum(x["g_ret"] for x in v)
    g_kept = sum(x["g_kept"] for x in v)
    g_sure = sum(x["g_kept_sure"] for x in v)
    n_err = sum(x["n_err"] for x in v)
    g_err = sum(x["g_kept_err"] for x in v)
    # 「完全筛掉」按**干净口径**判定 —— 保守保留不算「真的留下了」
    zero = [x["id"] for x in v if x["g_ret"] > 0 and x["g_kept_sure"] == 0]
    summary["%d/%d" % (pk, fk)] = {
        "avg_ret": round(avg_ret, 1),
        "avg_kept": round(avg_kept, 1),
        "g_ret": g_ret,
        "g_kept": g_kept,
        "g_kept_sure": g_sure,
        "n_err": n_err,
        "g_kept_err": g_err,
        "g_keep_rate": round(g_kept / max(g_ret, 1), 4),
        "g_keep_rate_sure": round(g_sure / max(g_ret, 1), 4),
        "zero_golden_cases": zero,
        "calls": out[(pk, fk)]["calls"],
        "sec": out[(pk, fk)]["sec"],
    }
    print(
        "  %-10s %-10.1f %-10.1f %-12d %-14s %-12s %s"
        % (
            "%d/%d" % (pk, fk),
            avg_ret,
            avg_kept,
            g_ret,
            "%d / %d" % (g_kept, g_sure),
            ("%d 次（golden %d）" % (n_err, g_err)) if n_err else "无",
            (", ".join(zero) if zero else "无"),
        )
    )

print()
print("  **口径说明**：`golden 保留` 列格式为 `含保守保留 / **干净口径**` ——")
print("    * **含保守保留** = 与生产一致（判分异常时留块，`_grade_one` 的行为）")
print("    * **干净口径**   = **排除判分异常**后的保留数，**结论应以它为准**")
print("    * 两值接近 -> 放大配额的收益是真实的；")
print("      干净口径明显更低 -> 「多保留的 golden」有相当部分是**判分失败被保守保留**，需谨慎。")

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
    print(
        "  %-8s vs 8/24:  golden 保留 %+d 块（%d -> %d）| 判分调用 %+d 次（%.2fx）"
        % (k, d_g, b["g_kept"], v["g_kept"], d_c, v["calls"] / max(b["calls"], 1))
    )
print()
print("  决策依据：**每多保留 1 个 golden 块，要付多少次额外判分调用**。")

(HERE / "compare_perk_after_grade.json").write_text(
    json.dumps(
        {
            "grid": GRID,
            "workers": WORKERS,
            "detail": {"%d/%d" % k: v["rows"] for k, v in out.items()},
            "summary": summary,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/compare_perk_after_grade.json")
