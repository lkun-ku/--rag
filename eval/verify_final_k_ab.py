"""final_k 端到端 A/B（ROADMAP 4.10.4）—— 验证 4.10 发现的杠杆是否真的可用。

## TDD 说明（本脚本的写法）

按 TDD 的「**先写判读标准，再跑数据**」：标准写死在 `ACCEPT` 里，
**在看到数据之前就定好**，避免事后找理由。若数据不满足 `ACCEPT`，
结论就是「不采纳」—— 不问「能不能网开一面」。

## 接缝（seam）

* 检索层：`MultiCollectionRetriever(final_k=...)` -> `invoke()`
* 端到端：`graph2.run_with_trace(question)` -> `{generation, trace, documents}`
  （trace 里有 `grade_documents` 的保留块数、token 消耗、各步耗时）

**不测**：RRF 融合的内部实现、`grade_documents` 的判分逻辑。

## 为什么必须端到端（而不是只看检索层 recall）

4.10 的检索层扫描显示 `final_k` 每 +6 块 recall 稳定 +0.05 —— 但那是**检索层**指标。
真正决定答案质量的是「`grade_documents` 过滤后喂给 `generate` 的块数」。
本脚本测的就是这一层：**多出来的块有没有通过判分、有没有进到 prompt 里**。

## 多轮对抗抖动（4.8 的方法论）

只取 `stability_audit.json` 里的**稳定题**，且每个配置跑 `--rounds` 轮取均值。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# 判读标准（**跑数据之前就写死**）
# ---------------------------------------------------------------------------
ACCEPT = {
    # 检索层：final_k=24 的 recall 需比 18 高出这个幅度（大于抖动才有意义）
    "recall_delta_min": 0.02,
    # 成本闸门：token 与端到端耗时不能暴涨
    "token_ratio_max": 1.5,
    "elapsed_ratio_max": 1.3,
    # 下游：判分保留的块数增加（说明多出来的块确实进了上下文）
    "kept_blocks_must_increase": True,
}

# (per_collection_k, final_k) 组合
#
# **[修正 2026-10-15] 首版是无效实验**：它只改了 FINAL_K 而没改 PER_COLLECTION_K ——
# 每集合仍只取 6 块，总候选只有 18 块，于是 `final_k=24` 是「**截断上限 > 输入**」，
# **两个配置完全等价**，A/B 必然测出「毫无差异」（实测连平均块数都相同：17.7 vs 17.7）。
# 这属于 TDD 里的 **tautological 反模式** —— 断言因构造而必然成立，永远无法与代码不同。
#
# 正确做法：`per_k` 与 `final_k` **同步调整**（4.10 的 B 段就是这么扫的），
# 并加**有效性自检**（两组候选规模必须不同，否则拒绝出结论）。
CONFIGS = [(6, 18), (8, 24)]


def main() -> int:
    ap = argparse.ArgumentParser(description="final_k 端到端 A/B")
    ap.add_argument("--cases", type=int, default=6, help="取几道稳定题")
    ap.add_argument("--rounds", type=int, default=2, help="每配置跑几轮")
    args = ap.parse_args()

    import tools.retriever_tools as rt
    from graph2.graph_2 import run_with_trace
    from utils.env_utils import DOC_SOURCES  # noqa: F401  (确保配置已加载)

    # 取稳定题（4.8 的产物）
    stab = HERE / "stability_audit.json"
    picked = []
    if stab.exists():
        d = json.loads(stab.read_text(encoding="utf-8"))
        stable_ids = set(d.get("stable_ids") or [])
        cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        picked = [c for c in cases if c["id"] in stable_ids and c.get("source_blocks")]
    if not picked:
        print("** 没有可用的稳定题（stability_audit.json 缺失或为空）—— 先跑 4.8 的审计")
        return 2
    picked = picked[:args.cases]
    print("参与 A/B 的稳定题: %s" % [c["id"] for c in picked])

    def fp(t):
        import hashlib
        return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]

    # ---------------------------------------------------------------------
    # [修正 2026-10-15 · 第四个实验设计缺陷] **交替执行**，不能「跑完 A 再跑 B」。
    #
    # 实测教训：顺序执行时 8/24 组在第 5 个样本后耗时整体从 6s 跳到 26s，而**同一题
    # 同一输入**（n001 两轮，tokens 完全一致 14104）耗时 6.7s vs 25.3s —— 那是
    # **LLM 服务的时间漂移**，却被算成了「配置导致耗时增加 2.88x」。
    # 交替执行让两组**共享同一时间窗**，把这类污染平摊掉。
    #
    # 注：`_build_multi_retriever(per_k=..., final_k=...)` 必须**显式传参** ——
    # 那两个值是 pydantic 字段，默认值在**类定义时**固定，改模块常量无效（见 4.11.8）。
    # ---------------------------------------------------------------------
    built = {(pk, fk): rt._build_multi_retriever(per_k=pk, final_k=fk)
             for (pk, fk) in CONFIGS}           # 预先构造；切换时只换引用，不重建

    results = {(pk, fk): [] for (pk, fk) in CONFIGS}
    block_order = []                            # **配置块**的执行顺序（供「交替自检」用）
    #
    # 注意：自检必须看「配置块」的顺序，不能看「样本」的顺序 ——
    # 首版用 `order[:2]`（前两个**样本**）判断，而那两个样本天然都在第一个配置块内，
    # 于是**交替明明生效了却被判成「未交替」**。（又一次「测量口径」的错。）

    for rnd in range(1, args.rounds + 1):
        for (pk, fk) in CONFIGS:                # <- 轮内交替
            rt.retriever = built[(pk, fk)]
            block_order.append((pk, fk))
            print()
            print("--- 第 %d 轮 / per_k = %d / final_k = %d ---" % (rnd, pk, fk))
            rows = results[(pk, fk)]
            for c in picked:
                t0 = time.time()
                try:
                    r = run_with_trace(c["q"], save=False)
                except Exception as e:  # noqa: BLE001
                    print("  [%s] 运行失败: %s" % (c["id"], type(e).__name__))
                    continue
                tr = r["trace"]
                docs = r.get("documents") or []
                gold = set(c["source_blocks"])
                # 注意：这里算的是**判分后**的召回（docs 已被 grade_documents 覆盖），
                # 不是检索层召回 —— 见下方打印处的说明。
                found = len(gold & {fp(d.page_content) for d in docs})
                # 检索到的块数与判分保留的块数**都要从 trace 的对应步骤取**。
                # [修正 2026-10-15] 首版把 `len(r["documents"])` 当成「检索到几块」，
                # 但那是**工作流最终状态**（已被判分覆盖），于是有效性自检拿
                # 「判分后块数」去比较两组配置，误判成「配置没生效」。
                # **同一个测量口径的坑，这是第二次踩** —— 教训：workflow 的 final state
                # 不保留中间态，任何「中间量」都必须从 trace 的步骤里取。
                n_retrieved = None
                kept = None
                for s in tr["steps"]:
                    if s.get("node") == "retrieve" and "documents" in s:
                        n_retrieved = s["documents"]
                    if s.get("node") == "grade_documents" and "documents" in s:
                        kept = s["documents"]
                rows.append({
                    "id": c["id"], "round": rnd,
                    "recall": round(found / max(len(gold), 1), 3),
                    "n_docs": n_retrieved,      # 检索到的块数（从 trace 的 retrieve 步骤取）
                    "kept": kept,
                    "tokens": (tr["tokens"].get("total_tokens") or 0),
                    "elapsed": round(time.time() - t0, 1),
                    "refused": "知识库中未找到依据" in (r["generation"] or ""),
                })
            if rows:
                last = rows[-1]
                print("  第 %d 轮末题: recall=%.3f kept=%s tokens=%s elapsed=%.1fs" % (
                    rnd, last["recall"], last["kept"], last["tokens"], last["elapsed"]))
        # （不再需要 `results[(pk, fk)] = rows` —— rows 就是该配置列表的引用）

    # ---- 交替自检（对应 TDD 的「验证必须能失败」）----
    # 若脚本被改回「顺序执行」，前两个**配置块**会是同一个配置 —— 这里必须能发现并拒绝。
    head = set(block_order[:len(CONFIGS)])
    if len(head) < len(CONFIGS):
        print()
        print("  **[实验无效]** 执行顺序未交替（前 %d 个配置块同属 %s）——"
              " 无法排除服务时间漂移的污染，结论不可用。"
              % (len(CONFIGS), sorted(head)))
        return 4
    print()
    print("  交替自检通过：前 %d 个配置块覆盖 %d 个配置 ✅（块序列: %s）"
          % (len(CONFIGS), len(head),
             ["%d/%d" % b for b in block_order[:6]]))

    # ---- 对比 ----
    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else float("nan")

    print()
    print("=" * 100)
    print("A/B 对比（%d 题 × %d 轮）" % (len(picked), args.rounds))
    print("=" * 100)
    # [修正 2026-10-15] 这一列**不是**「检索层召回」——`run_with_trace()` 返回的
    # `documents` 是**工作流最终状态**里的，已被 `grade_documents` 覆盖（只留相关的）。
    # 首版把它标成 `retrieval_recall`，极易被误读成「检索层 recall 只有 0.58」
    # 而去错误地优化检索。**测量口径必须标清**。
    print("  %-13s %-11s %-11s %-11s %-9s %-9s %s" % (
        "per_k/final_k", "检索块数", "判分保留", "判分后召回", "tokens", "耗时", "拒答"))
    stat = {}
    for (pk, fk) in CONFIGS:
        rows = results[(pk, fk)]
        st = {k: agg(rows, k) for k in ("recall", "kept", "tokens", "elapsed", "n_docs")}
        st["refused"] = sum(1 for r in rows if r["refused"])
        stat[(pk, fk)] = st
        print("  %-13s %-11.1f %-11.2f %-11.4f %-9.0f %-9.1f %d" % (
            "%d/%d" % (pk, fk), st["n_docs"], st["kept"], st["recall"],
            st["tokens"], st["elapsed"], st["refused"]))

    a, b = stat[CONFIGS[0]], stat[CONFIGS[1]]

    # ---- 有效性自检（**关键**：防止再次出现「两组等价」的无效实验）----
    if abs(a["n_docs"] - b["n_docs"]) < 0.5:
        print()
        print("  **[实验无效]** 两组的候选块数几乎相同（%.1f vs %.1f）—— "
              "配置没有真正生效，结论不可用。" % (a["n_docs"], b["n_docs"]))
        print("  提示：改 `final_k` 时**必须同时**把 `per_k` 调大，否则候选池填不满 final_k。")
        return 3
    print()
    print("=" * 100)
    print("按**事先写死的** ACCEPT 判读")
    print("=" * 100)
    checks = []
    d_rec = b["recall"] - a["recall"]
    checks.append(("recall 提升 > %.2f" % ACCEPT["recall_delta_min"],
                   d_rec > ACCEPT["recall_delta_min"], "Δ=%.4f" % d_rec))
    tr_ratio = (b["tokens"] / a["tokens"]) if a["tokens"] else float("inf")
    checks.append(("token 增幅 <= %sx" % ACCEPT["token_ratio_max"],
                   tr_ratio <= ACCEPT["token_ratio_max"], "%.2fx" % tr_ratio))
    el_ratio = (b["elapsed"] / a["elapsed"]) if a["elapsed"] else float("inf")
    checks.append(("耗时增幅 <= %sx" % ACCEPT["elapsed_ratio_max"],
                   el_ratio <= ACCEPT["elapsed_ratio_max"], "%.2fx" % el_ratio))
    checks.append(("判分保留块数增加",
                   b["kept"] > a["kept"], "%.2f -> %.2f" % (a["kept"], b["kept"])))

    for name, ok, detail in checks:
        print("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", name, detail))
    all_ok = all(ok for _n, ok, _d in checks)
    print()
    print("  => **%s**" % ("采纳 final_k=%d" % FINAL_KS[1] if all_ok
                          else "不采纳（未满足事先定义的标准）"))

    (HERE / "final_k_ab.json").write_text(json.dumps(
        {"accept": ACCEPT, "configs": CONFIGS,
         # tuple key 不能直接进 JSON，转成 "per_k/final_k" 形式
         "results": {"%d/%d" % k: v for k, v in results.items()},
         "stats": {"%d/%d" % k: v for k, v in stat.items()},
         "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
         "verdict": "adopt" if all_ok else "reject"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("  结果 -> eval/final_k_ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
