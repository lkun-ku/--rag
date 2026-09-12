"""并发压测：验证今晚新增的两处线程池在**多请求**下的行为。

## 要回答的问题

今晚加了两处线程池（`grade_batch` 批间 8 并发、`grade_generation` 2 并发），
但**只在单请求路径上验证过**。这个系统的形态是**服务**（`api_server` + SSE），
所以必须回答：

    1. 会不会触发 LLM 端点**限流 / 超时**？（C 请求 × 8 判分线程 = 8C 并发调用）
    2. **线程总数**有没有失控？（每请求 8+ 个线程 × N 请求 × FastAPI worker）
    3. **耗时**随并发如何变化？（是线性、还是劣化？）
    4. **结果正确性**是否受影响？（并发下答案是否仍然正确）

## 方法

用 `ThreadPoolExecutor` 同时发起 C 个 `run_with_trace(q)`，逐级加压
（1 -> 2 -> 4，**不一次拉满** —— 先看 2 的行为再决定要不要上 4）。

**只测链路层**（直调 `run_with_trace`），不起 FastAPI —— 因为线程池在链路层，
而 HTTP 层只多一次序列化开销。服务层的验证需要 SSE 客户端，留作后续。

## 成本

C 个请求 × 约 5s。总题数 = 1+2+4 = 7 次问答，**约 1 分钟**。

## 判读标准

* **无异常且答案正确** -> 该并发度安全；
* **出现 429/超时** -> 端点限流，说明并发度偏高，需要加信号量；
* **耗时非线性放大** -> 有资源争抢（Milvus 连接池 / GIL / 端点排队）。
"""
from __future__ import annotations

import concurrent.futures as cf
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

LEVELS = [int(x) for x in os.environ.get("LOAD_LEVELS", "1,2,4").split(",")]

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")]
# 取 4 道互不相同的题（并发时用不同问题，更接近真实）
QUESTIONS = [c["q"] for c in cases[:max(LEVELS)]]
GOLDS = [c for c in cases[:max(LEVELS)]]
print("题库（前 %d 道）:" % len(QUESTIONS))
for c in GOLDS:
    print("  %-6s %s" % (c.get("id"), c["q"][:58]))
print()

from graph2.graph_2 import run_with_trace  # noqa: E402


def one(idx: int) -> dict:
    q = QUESTIONS[idx % len(QUESTIONS)]
    t0 = time.time()
    try:
        out = run_with_trace(q)
        gen = (out.get("generation") or "").strip()
        return {"idx": idx, "sec": round(time.time() - t0, 1), "ok": True,
                "n_gen": len(gen), "head": gen[:60].replace("\n", " "),
                "err": None}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "sec": round(time.time() - t0, 1), "ok": False,
                "n_gen": 0, "head": "", "err": "%s: %s" % (type(e).__name__,
                                                           str(e)[:110])}


print("=" * 104)
print("并发压测（链路层直调 run_with_trace）")
print("=" * 104)
all_rows = []
for C in LEVELS:
    print()
    print("--- 并发度 C=%d（该级会同时发起 %d 个请求，每个内部还有 8 个判分线程）---"
          % (C, C))
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=C) as ex:
        rows = list(ex.map(one, range(C)))
    wall = time.time() - t0

    for r in rows:
        tag = "OK " if r["ok"] else "**FAIL**"
        print("  [%s] #%d  %5.1fs  generation %4d 字符 | %s" % (
            tag, r["idx"], r["sec"], r["n_gen"], r["head"] or r["err"] or ""))
    secs = [r["sec"] for r in rows]
    n_ok = sum(1 for r in rows if r["ok"])
    print("  -> 墙钟 %.1fs | 单请求 %.1f~%.1fs | 成功 %d/%d" % (
        wall, min(secs), max(secs), n_ok, C))
    all_rows.append({"C": C, "wall": round(wall, 1),
                     "min": min(secs), "max": max(secs),
                     "n_ok": n_ok, "n": C, "rows": rows})

print()
print("=" * 104)
print("汇总")
print("=" * 104)
print("  %-6s %-10s %-12s %-12s %-10s %s" % (
    "并发", "墙钟(s)", "单请求最快", "单请求最慢", "成功", "判定"))
print("  " + "-" * 96)
base = None
for r in all_rows:
    if base is None:
        base = r["wall"]
    flag = "✅ 正常" if r["n_ok"] == r["n"] else "❌ 有失败"
    if r["n_ok"] == r["n"] and r["max"] > 3 * (r["min"] or 1):
        flag = "⚠️ 耗时离散大（可能有排队）"
    print("  %-6d %-10.1f %-12.1f %-12.1f %-10s %s" % (
        r["C"], r["wall"], r["min"], r["max"], "%d/%d" % (r["n_ok"], r["n"]), flag))

print()
print("  理想情况：墙钟随时间接近「单请求耗时」（并发真的并行了）")
print("  若墙钟 ≈ C × 单请求耗时 -> 实际是串行（有全局锁或端点排队）")
print()
errs = [x for r in all_rows for x in r["rows"] if not x["ok"]]
if errs:
    print("  **失败详情**：")
    for x in errs[:5]:
        print("    #%d: %s" % (x["idx"], x["err"]))
else:
    print("  **无失败** —— 该并发范围内端点未限流、链路未出错。")

(HERE / "load_test_concurrency.json").write_text(json.dumps(
    {"levels": LEVELS, "results": all_rows}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print()
print("  证据 -> eval/load_test_concurrency.json")
