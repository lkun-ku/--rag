"""PER_COLLECTION_K 扫描 —— 验证 P1-15「集合内部的同质块互相挤占」是否成立。

## 背景（P1-15）

4.7.4 的观察：问「召回率受什么影响」时，`performance_faq.md` 被**同一集合内**的
`operational_faq.md` / `product_faq.md` 挤出 top-6。即 `PER_COLLECTION_K=6` 对
「**文档多且主题相近**」的集合不够用 —— **集合内部也会发生配额竞争**。

修法方向是「调大每集合配额」。本脚本负责**先验证它值不值得做**。

## 实验设计（关键是只动一个变量）

`MultiCollectionRetriever` 是 pydantic 模型，`per_collection_k` 与 `final_k` 是
**独立字段**，因此可以只调「每集合取几块」而**固定最终交给下游的 18 块** ——
这样跨集合 RRF 融合会**真正参与取舍**（而不是把每集合配额全取走）。

**固定 `fetch_k = 12`**（3.0.3 已扫描出它最优）：`fetch_k` 是「每路（dense/sparse）
预取数」，必须 ≥ `k`，否则候选池不足、`k` 形同虚设。
所以扫描点限制在 **per_k ≤ 12**：3 / 4 / 6 / 8 / 10 / 12。

两段对照：

    A. per_k 变化，final_k 固定 18    -> 检验 P1-15（融合参与取舍）
    B. final_k = 3 × per_k（同步放大） -> 对照：连 final_k 一起放大有何不同

**判读标准**（事先写清，避免事后找理由）：

* 若 A 段在某个 per_k 上 **recall 提升且 precision 不降** -> P1-15 成立，值得改；
* 若 A 段 **recall 基本不动** -> 说明「6 块已够」，集合内挤占不是当前瓶颈 -> P1-15 降级为已知边界；
* 若 A 段 **precision 大幅下滑** -> 与 P0-8 的结论一致（给得多不等于给得准），维持现状。

**注意样本与抖动**：当前评测集 32 题，且 4.8 已量化「4 题存在 critical 抖动」。
因此**小于抖动幅度的差异不足为据** —— 本脚本只做「方向性判断」，
不追求小数点后的精确比较。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent


os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.documents.milvus_db import MilvusVectorSave  # noqa: E402
from rag_kb.tools.retriever_tools import (  # noqa: E402
    FETCH_K_PER_ROUTE,
    MultiCollectionRetriever,
    _build_search_kwargs,
    _discover_collections,
)

OUT = HERE / "scan_per_collection_k.json"
cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")]
print("参与统计的题目: %d 道（已排除库外题）" % len(cases))


def fp(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


_STORES = {}


def build(per_k: int, final_k: int, fetch_k: int = FETCH_K_PER_ROUTE) -> MultiCollectionRetriever:
    if not _STORES:
        for coll in _discover_collections():
            mv = MilvusVectorSave(collection_name=coll)
            mv.create_connection()
            _STORES[coll] = mv.vector_store_saved
    kwargs = dict(_build_search_kwargs())
    kwargs["k"] = per_k
    kwargs["fetch_k"] = fetch_k
    built = {
        coll: store.as_retriever(search_type="similarity", search_kwargs=kwargs) for coll, store in _STORES.items()
    }
    return MultiCollectionRetriever(retrievers=built, per_collection_k=per_k, final_k=final_k)


def evaluate(retr: MultiCollectionRetriever) -> dict:
    rec = prec = 0.0
    hit = 0
    n_blocks = 0
    per_src = {}
    for c in cases:
        gold = set(c["source_blocks"])
        try:
            docs = retr.invoke(c["q"])
        except Exception as e:  # noqa: BLE001
            print("    检索失败 [%s]: %s" % (c["id"], type(e).__name__))
            docs = []
        rids = [fp(d.page_content) for d in docs]
        found = gold & set(rids)
        r = len(found) / max(len(gold), 1)
        rec += r
        prec += len(found) / max(len(rids), 1)
        hit += 1 if r > 0 else 0
        n_blocks += len(rids)
        per_src.setdefault(c.get("source", "?"), []).append(r)
    n = len(cases)
    return {
        "recall": round(rec / n, 4),
        "hit_rate": round(hit / n, 4),
        "precision": round(prec / n, 4),
        "avg_blocks": round(n_blocks / n, 1),
        "by_src": {s: round(sum(v) / len(v), 4) for s, v in sorted(per_src.items())},
    }


def run(label: str, points, final_of) -> list:
    print()
    print("=" * 104)
    print("%s" % label)
    print("=" * 104)
    print(
        "  %-8s %-9s %-9s %-9s %-11s %-9s %s"
        % ("per_k", "final_k", "recall", "hit_rate", "precision", "平均块数", "分源 recall")
    )
    print("  " + "-" * 96)
    rows = []
    for pk in points:
        fk = final_of(pk)
        t0 = time.time()
        m = evaluate(build(pk, fk))
        m.update({"per_k": pk, "final_k": fk, "sec": round(time.time() - t0, 1)})
        rows.append(m)
        src = " ".join("%s=%.3f" % (s[:8], v) for s, v in m["by_src"].items())
        print(
            "  %-8d %-9d %-9.4f %-9.4f %-11.4f %-9.1f %s"
            % (pk, fk, m["recall"], m["hit_rate"], m["precision"], m["avg_blocks"], src)
        )
    return rows


# ---- A 段：只动 per_k，final_k 固定 18（检验 P1-15）----
FIXED_FINAL = 18
rows_a = run(
    "A. per_k 变化 / final_k 固定 %d（跨集合融合真正参与取舍）" % FIXED_FINAL,
    [3, 4, 6, 8, 10, 12],
    lambda pk: FIXED_FINAL,
)

base_a = next(r for r in rows_a if r["per_k"] == 6)  # 当前生产值
print()
print("  相对当前值（per_k=6）的变化：")
print("  %-8s %-12s %-12s %-12s %s" % ("per_k", "Δrecall", "Δprecision", "Δ平均块数", "判读"))
for r in rows_a:
    d_rec = r["recall"] - base_a["recall"]
    d_pre = r["precision"] - base_a["precision"]
    d_blk = r["avg_blocks"] - base_a["avg_blocks"]
    if r["per_k"] == 6:
        note = "（当前生产值）"
    elif d_rec >= 0.02 and d_pre >= -0.005:
        note = "**recall 提升且 precision 不降 -> P1-15 成立**"
    elif d_rec <= -0.02:
        note = "recall 下降"
    else:
        note = "变化未超抖动幅度，不足为据"
    print("  %-8d %+-12.4f %+-12.4f %+-12.1f %s" % (r["per_k"], d_rec, d_pre, d_blk, note))

# ---- B 段：per_k 与 final_k 同步放大（对照）----
rows_b = run(
    "B. per_k 与 final_k 同步放大（final_k = 3 × per_k，对照）",
    [6, 8, 10, 12],
    lambda pk: pk * len(_discover_collections()),
)
print()
print("  与 A 段同 per_k 对比：")
print("  %-8s %-24s %-24s %s" % ("per_k", "A: final_k=18", "B: final_k=3×per_k", "说明"))
for rb in rows_b:
    ra = next((r for r in rows_a if r["per_k"] == rb["per_k"]), None)
    if ra is None:
        continue
    print(
        "  %-8d recall %.4f/prec %.4f   recall %.4f/prec %.4f   %s"
        % (
            rb["per_k"],
            ra["recall"],
            ra["precision"],
            rb["recall"],
            rb["precision"],
            "块更多 -> " + ("recall 升" if rb["recall"] > ra["recall"] else "recall 未升"),
        )
    )

OUT.write_text(
    json.dumps({"fixed_final": FIXED_FINAL, "A": rows_a, "B": rows_b}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print()
print("  结果 -> eval/scan_per_collection_k.json")
