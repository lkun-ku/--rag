"""fetch_k 参数扫描：寻找「召回与排序」的平衡点，判断是否能不依赖 reranker。

## 最终取值：`fetch_k = 12`（恰为 2 × PER_COLLECTION_K）

结论已写入 `tools/retriever_tools.FETCH_K_PER_ROUTE`。当时的扫描结果：

    fetch_k     recall      未满分题数
    ------------------------------------
    默认(4)     0.8801      8
    8           0.9181      6
    12          0.9327      5   <- 采用
    16          0.9197      8
    32          0.8947      9
    50          0.8626      9
    120         0.8012      13

**太小**（候选池不足 → q043 那种「正确块进不了融合」）与**太大**
（RRF 的 `1/(k+rank)` 让「两路都中等」的块压过「单路极强」的块）都不好，
12 是平衡点。

> [说明 2026-10-11 / P2-2] 本 docstring 原先写「fetch_k=50（修复值）」——
> 那是**中间版本**的值（最终扫描定为 12）；题数也仍写 57（旧语料，现为 32）。
> 两处均已更正。**实际生效值以 `tools/retriever_tools.py` 为准**，脚本里不再写死。

方法：直接构造不同 fetch_k 的跨集合检索器，跑**当前评测集**的检索层指标
（不调 LLM，秒级）。
"""
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from documents.milvus_db import MilvusVectorSave  # noqa: E402
from tools.retriever_tools import (  # noqa: E402
    FETCH_K_PER_ROUTE, FINAL_K, PER_COLLECTION_K, RRF_K, MultiCollectionRetriever,
    _build_search_kwargs, _discover_collections,
)

E = ROOT / "eval"
cases = [json.loads(l) for l in (E / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def fp(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


def build(fetch_k):
    kwargs = dict(_build_search_kwargs())
    kwargs["k"] = PER_COLLECTION_K
    if fetch_k is None:
        kwargs.pop("fetch_k", None)
    else:
        kwargs["fetch_k"] = fetch_k
    built = {}
    for coll in _discover_collections():
        try:
            mv = MilvusVectorSave(collection_name=coll)
            mv.create_connection()
            built[coll] = mv.vector_store_saved.as_retriever(
                search_type="similarity", search_kwargs=kwargs)
        except Exception as e:  # noqa: BLE001
            print("  集合 %s 跳过: %s" % (coll, type(e).__name__))
    return MultiCollectionRetriever(retrievers=built)


def evaluate(retr):
    per_src, hits, total, n_ret = {}, 0, 0, 0
    miss = []
    for c in cases:
        gid = set(c["source_blocks"])
        try:
            docs = retr.invoke(c["q"])
        except Exception as e:  # noqa: BLE001
            print("    [%s] 检索失败 %s" % (c["id"], type(e).__name__))
            continue
        rids = [fp(d.page_content) for d in docs]
        found = gid & set(rids)
        rec = len(found) / len(gid) if gid else 0.0
        total += rec
        n_ret += len(rids)
        hits += 1 if rec > 0 else 0
        per_src.setdefault(c.get("source", "?"), []).append(rec)
        if rec < 1.0:
            miss.append(c["id"])
    return {
        "recall": total / len(cases),
        "hit_rate": hits / len(cases),
        "avg_blocks": n_ret / len(cases),
        "by_src": {s: sum(v) / len(v) for s, v in per_src.items()},
        "miss": miss,
    }


print("=" * 104)
print("fetch_k 扫描（%d 题检索层，PER_COLLECTION_K=%d / FINAL_K=%d）"
      % (len(cases), PER_COLLECTION_K, FINAL_K))
print("=" * 104)
print("  %-9s %-9s %-9s %-9s %s" % ("fetch_k", "recall", "hit_rate", "平均块数", "分源 recall"))
print("  " + "-" * 100)

rows = []
for fk in (None, 4, 8, 12, 16, 24, 32, 50, 80, 120):
    t0 = time.time()
    r = evaluate(build(fk))
    r["fetch_k"] = fk
    r["sec"] = time.time() - t0
    rows.append(r)
    label = "默认(4)" if fk is None else str(fk)
    src = " ".join("%s=%.3f" % (k[:9], v) for k, v in sorted(r["by_src"].items()))
    print("  %-9s %-9.4f %-9.4f %-9.2f %s" % (
        label, r["recall"], r["hit_rate"], r["avg_blocks"], src))

best = max(rows, key=lambda r: (r["recall"], r["hit_rate"]))
print()
print("=" * 104)
print("结论")
print("=" * 104)
print("  最佳 recall : fetch_k=%s -> %.4f（hit_rate %.4f，平均 %.1f 块，耗时 %.0fs）" % (
    best["fetch_k"] if best["fetch_k"] else "默认",
    best["recall"], best["hit_rate"], best["avg_blocks"], best["sec"]))
# [FIX 2026-10-11 / P2-2] 不再写死「当前值 50」—— 直接从生产配置读，
# 避免下次改了 FETCH_K_PER_ROUTE 又忘了改这里（这正是原先脱节的成因）。
_cur = next((r["recall"] for r in rows if r["fetch_k"] == FETCH_K_PER_ROUTE), None)
print("  当前值 %-4s: recall %s" % (
    FETCH_K_PER_ROUTE, ("%.4f" % _cur) if _cur is not None else "（不在扫描点内）"))
print("  旧默认 4   : recall %.4f" % next(r["recall"] for r in rows if r["fetch_k"] is None))
print()
print("  各配置未满分题:")
for r in rows:
    print("    fetch_k=%-8s 未满分 %d 题: %s" % (
        r["fetch_k"] if r["fetch_k"] else "默认(4)", len(r["miss"]), r["miss"][:12]))
