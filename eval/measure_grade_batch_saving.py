"""实测批量判分：**一致率**（质量）与 **token 节省**（成本）。

## 为什么必须实测节省，不能推算

批量判分减少的是**调用次数**（36 -> 5），但**块内容一个字没变** ——
输入 token 主要由「块长 × 块数」决定，所以**未必成比例下降**。
真正省下的只有「被重复发送的 system prompt + 问题」，以及 API 调用开销。

> 「调用少了 7 倍」≠「token 少了 7 倍」。**必须用数字说话** ——
> 本会话已多次因「想当然的比例」吃过亏（4.12 的耗时误判、4.19 的判分误杀）。

## 口径

对同一批（问题, 块）分别跑：

    A. **逐块**：`_grade_one` × N（现状，带 callbacks 计 token）
    B. **批量**：`grade_batch`（`BATCH_SIZE=8`）

记录三件事：

    1. **一致率** = 两种方式判定相同的块数 / 总块数      <- 质量指标
    2. **token**（input / output / total），用 `UsageMetadataCallbackHandler` 实测
    3. **调用次数**（逐块 = N；批量 = ceil(N/8)）

一致率**不作 pass/fail**：差异需人工判断是否可接受（LLM 的合理分歧 vs 真退化）。
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402

import metrics  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402
from graph2.grade_documents_node import (  # noqa: E402
    _grade_one, grade_batch, BATCH_SIZE,
)

QIDS = os.environ.get("SAVING_QIDS", "n001,n005,n026").split(",")

cases = {json.loads(l).get("id"): json.loads(l)
         for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
         if l.strip()}

print("=" * 106)
print("批量判分实测：一致率 + token 节省（BATCH_SIZE=%d）" % BATCH_SIZE)
print("=" * 106)
print()

rows = []
tot = {"a_in": 0, "a_out": 0, "b_in": 0, "b_out": 0, "a_calls": 0, "b_calls": 0,
       "same": 0, "all": 0}
for qid in QIDS:
    c = cases.get(qid)
    if not c:
        print("  [!] %s 不在 golden 里" % qid)
        continue
    q = c["q"]
    docs = retriever.invoke(q)

    # ---------- A：逐块 ----------
    ua = UsageMetadataCallbackHandler()
    ca = {"callbacks": [ua]}
    a_verdicts = []
    for d in docs:
        _, ok = _grade_one(q, d, ca)
        a_verdicts.append(ok)
    a_calls = len(docs)

    # ---------- B：批量 ----------
    ub = UsageMetadataCallbackHandler()
    b_verdicts = grade_batch(q, docs, config={"callbacks": [ub]})
    b_calls = math.ceil(len(docs) / max(BATCH_SIZE, 1))

    def _tk(u):
        m = dict(getattr(u, "usage_metadata", {}) or {})
        ti = sum(int(v.get("input_tokens", 0) or 0) for v in m.values())
        to = sum(int(v.get("output_tokens", 0) or 0) for v in m.values())
        return ti, to

    a_in, a_out = _tk(ua)
    b_in, b_out = _tk(ub)
    same = sum(1 for x, y in zip(a_verdicts, b_verdicts) if x == y)
    n = len(docs)

    rows.append({"id": qid, "n": n, "a_verdicts": a_verdicts, "b_verdicts": b_verdicts,
                 "same": same, "a_in": a_in, "a_out": a_out, "b_in": b_in, "b_out": b_out,
                 "a_calls": a_calls, "b_calls": b_calls})

    print("  [%s] %d 块" % (qid, n))
    print("     逐块: 调用 %2d 次  in=%6d out=%5d  total=%6d" % (
        a_calls, a_in, a_out, a_in + a_out))
    print("     批量: 调用 %2d 次  in=%6d out=%5d  total=%6d" % (
        b_calls, b_in, b_out, b_in + b_out))
    print("     一致率: %d/%d = %.1f%%" % (same, n, 100 * same / max(n, 1)))
    diff = [i for i, (x, y) in enumerate(zip(a_verdicts, b_verdicts), 1) if x != y]
    if diff:
        print("     分歧块: %s" % diff)
        for i in diff[:3]:
            print("        块%d: 逐块=%s 批量=%s | %s" % (
                i, a_verdicts[i - 1], b_verdicts[i - 1],
                docs[i - 1].page_content.replace("\n", " ")[:60]))
    print()

    tot["a_in"] += a_in; tot["a_out"] += a_out
    tot["b_in"] += b_in; tot["b_out"] += b_out
    tot["a_calls"] += a_calls; tot["b_calls"] += b_calls
    tot["same"] += same; tot["all"] += n

print("=" * 106)
print("汇总（%d 道题 / %d 块）" % (len(rows), tot["all"]))
print("=" * 106)
print("  %-14s %-12s %-12s %s" % ("口径", "调用次数", "input tok", "output tok / total"))
print("  " + "-" * 98)
print("  %-14s %-12d %-12d %d / %d" % (
    "逐块（现状）", tot["a_calls"], tot["a_in"], tot["a_out"], tot["a_in"] + tot["a_out"]))
print("  %-14s %-12d %-12d %d / %d" % (
    "批量", tot["b_calls"], tot["b_in"], tot["b_out"], tot["b_in"] + tot["b_out"]))
a_tot = tot["a_in"] + tot["a_out"]
b_tot = tot["b_in"] + tot["b_out"]
print()
print("  **一致率: %d/%d = %.1f%%**" % (
    tot["same"], tot["all"], 100 * tot["same"] / max(tot["all"], 1)))
print("  **调用次数: %d -> %d（%.2fx）**" % (
    tot["a_calls"], tot["b_calls"], tot["b_calls"] / max(tot["a_calls"], 1)))
print("  **input token: %d -> %d（%.2fx，省 %.0f%%）**" % (
    tot["a_in"], tot["b_in"], tot["b_in"] / max(tot["a_in"], 1),
    100 * (1 - tot["b_in"] / max(tot["a_in"], 1))))
print("  **total token: %d -> %d（%.2fx，省 %.0f%%）**" % (
    a_tot, b_tot, b_tot / max(a_tot, 1), 100 * (1 - b_tot / max(a_tot, 1))))
print()
print("  换算到单题（本批 %d 道）：逐块 %.0f token/题 -> 批量 %.0f token/题，"
      "**每题省 %.0f token**" % (
          len(rows), a_tot / max(len(rows), 1), b_tot / max(len(rows), 1),
          (a_tot - b_tot) / max(len(rows), 1)))

(HERE / "measure_grade_batch_saving.json").write_text(json.dumps(
    {"batch_size": BATCH_SIZE,
     # [FIX 2026-10-16 / 二次筛查] **必须记录本次实际跑了哪些题**。
     #
     # 原因：`SAVING_QIDS` 的默认值是 **3 题**，而落盘的证据（4.23 的
     # 「省 40% / 一致率 92.6%」）来自**6 题 215 块**。若不记 QIDS，
     # 任何人重跑一次（用默认值）都会**静默用 3 题的结果覆盖 6 题的证据** ——
     # 而 ROADMAP 里引用的数字会与 JSON 对不上，且无从察觉。
     # 记下 QIDS 后，「这条结论基于几道题」变成可核查的事实。
     "qids": QIDS,
     "n_cases": len(rows),
     "n_blocks": tot["all"],
     "rows": rows, "totals": tot,
     "consistent_rate": round(tot["same"] / max(tot["all"], 1), 4),
     "token_ratio": round(b_tot / max(a_tot, 1), 4),
     "call_ratio": round(tot["b_calls"] / max(tot["a_calls"], 1), 4)},
    ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  **本次证据覆盖**：%d 题（%s）/ %d 块" % (len(rows), ",".join(QIDS), tot["all"]))
print("  （ROADMAP 4.23 的「省 40%% / 一致率 92.6%%」来自 **6 题 / 215 块** ——")
print("   若本次题数不同，JSON 已记录 QIDS，可用于判断能否直接对比。）")
print()
print("  证据 -> eval/measure_grade_batch_saving.json")
