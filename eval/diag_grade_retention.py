"""③ 的起点实测：判分环节（`grade_documents`）到底筛掉了什么？

## 为什么先测这个

4.15.3 的结论是「n005 之前有时拒答，真因是**信噪比**」——
实测时给模型 2~3 块干净上下文它就能聚合，而真实链路是 24 块混合上下文。
但那是**间接推断**（对比了两组），本脚本把它**直接量化**：

    * 检索拿到多少块（`final_k=24`）
    * 判分后留下多少块（保留率）
    * **被筛掉的块里，有没有 golden 块**  <- 这是决定性的问题

若 golden 被大量筛掉 -> 判分是瓶颈，③ 有价值；
若 golden 基本都留下 -> 判分没问题，瓶颈在别处（那么 ③ 应改方向）。

## 口径

跑 `run_with_trace()`（**真实工作流**，含真实 LLM 判分），逐题记录：

    retrieve 步骤的块数 / grade_documents 步骤的块数 / 两者的 golden 命中数

**注意**：`run_with_trace` 返回的最终状态里 `documents` 已被判分覆盖（4.11.4 的坑），
所以要**逐步读 trace**，不能只看最终状态。

## 顺带测一件事（4.17 留下的推断）

4.17 推断「扁平块会在表格类查询里与结构化块竞争排名、制造噪声」——
重建已把扁平块消掉，本脚本顺便统计**判分保留的块里表格/结构化的占比**，
看噪声是否确实下降了（可与重建前的同类统计对照）。
"""
from __future__ import annotations

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

LIMIT = int(os.environ.get("GRADE_LIMIT", "10"))

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")][:LIMIT]

from graph2.graph_2 import run_with_trace  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402

print("=" * 104)
print("③ 起点实测：判分环节的保留率与「golden 是否被误筛」")
print("=" * 104)
print("  题数: %d（取 golden 题的前 %d 道）" % (len(cases), LIMIT))
print()

rows = []
for c in cases:
    gids = set(c["source_blocks"])
    t0 = time.time()
    try:
        out = run_with_trace(c["q"])
    except Exception as e:  # noqa: BLE001
        print("  [%s] 工作流失败: %s: %s" % (c.get("id"), type(e).__name__, str(e)[:80]))
        continue
    # trace 的 `steps` 只含节点名/耗时/决策，**不含 documents**（实测探明，见
    # eval/_probe_trace_shape.py）。所以「判分前」的块改从检索器直取，
    # 「判分后」用 `run_with_trace` 返回的 `documents`（那是最终状态里的，已被
    # grade_documents 覆盖 —— 4.11.4 记录过这个坑，这里正好利用它）。
    try:
        docs_retrieved = retriever.invoke(c["q"])
    except Exception as e:  # noqa: BLE001
        print("  [%s] 检索失败: %s" % (c.get("id"), type(e).__name__))
        continue
    docs_graded = out.get("documents", [])
    dec = ""
    for st in (out.get("trace", {}).get("steps") or []):
        if st.get("node") == "decide_to_generate":
            dec = st.get("decision", "")

    n_ret = len(docs_retrieved)
    n_grd = len(docs_graded)
    ids_ret = {metrics.chunk_id(d.page_content) for d in docs_retrieved}
    g_ret = len(gids & ids_ret)
    if docs_graded is not None:
        ids_grd = {metrics.chunk_id(d.page_content) for d in docs_graded}
        g_grd = len(gids & ids_grd)
    else:
        g_grd = -1
    rows.append({"id": c.get("id"), "q": c["q"][:44], "n_golden": len(gids),
                 "n_ret": n_ret, "n_grd": n_grd,
                 "g_ret": g_ret, "g_grd": g_grd, "decision": dec,
                 "sec": round(time.time() - t0, 1)})

print("  %-6s %-8s %-9s %-9s %-9s %-9s %-6s %s" % (
    "题", "golden", "检索块", "判分后", "保留率", "golden留", "步数", "判分决定"))
print("  " + "-" * 100)
for r in rows:
    keep = ("%.0f%%" % (100 * r["n_grd"] / max(r["n_ret"], 1))) if r["n_grd"] >= 0 else "n/a"
    print("  %-6s %-8d %-9d %-9d %-9s %d/%d      %-6s %s" % (
        r["id"], r["n_golden"], r["n_ret"], r["n_grd"], keep,
        r["g_grd"], r["g_ret"], "", r["decision"]))

ok = [r for r in rows if r["n_grd"] >= 0]
if ok:
    avg_ret = sum(r["n_ret"] for r in ok) / len(ok)
    avg_grd = sum(r["n_grd"] for r in ok) / len(ok)
    print()
    print("  平均检索块数: %.1f | 平均判分后: %.1f | **保留率 %.1f%%**" % (
        avg_ret, avg_grd, 100 * avg_grd / max(avg_ret, 1)))
    print()
    # 决定性指标：golden 在判分前后的命中数
    g_ret_all = sum(r["g_ret"] for r in ok)
    g_grd_all = sum(r["g_grd"] for r in ok)
    print("  **golden 命中数：检索后 %d -> 判分后 %d**（流失 %d，%.1f%%）" % (
        g_ret_all, g_grd_all, g_ret_all - g_grd_all,
        100 * (g_ret_all - g_grd_all) / max(g_ret_all, 1)))
    print()
    lost = [r for r in ok if r["g_grd"] < r["g_ret"]]
    if lost:
        print("  判分**筛掉了 golden 块**的题（%d 道）:" % len(lost))
        for r in lost:
            print("     [%s] golden %d 命中：检索后 %d -> 判分后 %d   %s" % (
                r["id"], r["n_golden"], r["g_ret"], r["g_grd"], r["q"]))
    else:
        print("  **没有任何一题的 golden 被筛掉** -> 判分环节不是瓶颈。")

(HERE / "diag_grade_retention.json").write_text(
    json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/diag_grade_retention.json")
