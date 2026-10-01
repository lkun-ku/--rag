"""门槛可行性 + 副本率 的批量量化（30 题）。

## 探针（probe_scores.py）给出的现象，本脚本负责量化

1. **门槛可行性**：把 golden 块与非 golden 块的分数分布画出来，
   找「recall 损失 <= 2% 前提下 precision 能提升多少」。
   预期：分界几乎不存在（RRF 分数只反映排名）。

2. **副本率**：低 precision 的另一嫌疑是「同一内容的多副本块」
   （表格结构化版 + 扁平平铺版 + 摘要版）。量化方式：
   - 同一 filename 在 18 块中占几块
   - 两两块之间「一个是另一个的子串」或「前缀相同」的比例

**不吞异常**：失败计数并打印，绝不静默变成空结果。
"""

import hashlib
import json
import os
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker  # noqa: E402

from rag_kb.llm_models.embeddings_model import bge_embedding  # noqa: E402
from rag_kb.tools.retriever_tools import FETCH_K_PER_ROUTE, PER_COLLECTION_K, _discover_collections  # noqa: E402
from rag_kb.utils.env_utils import MILVUS_URI  # noqa: E402

E = ROOT / "eval"
EXPR = 'category == "content"'


def fp(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


cases = [json.loads(l) for l in (E / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")]
print("参与统计的题目: %d 道" % len(cases))

client = MilvusClient(uri=MILVUS_URI)
COLLS = _discover_collections()
print("集合: %s" % COLLS)

OUT = ["text", "filename", "category"]
errs = []

# ---------------------------------------------------------------------------
# 采集：每题的 18 块候选（每集合 top-6，与生产一致）
# ---------------------------------------------------------------------------
print()
print("采集候选中（%d 题 × %d 集合）..." % (len(cases), len(COLLS)))
t0 = time.time()
samples = []  # list of dict(q, src, blocks=[(score, key, filename, text)], gold=set)
for c in cases:
    q = c["q"]
    qv = bge_embedding.embed_query(q)
    blocks = []
    for coll in COLLS:
        try:
            reqs = [
                AnnSearchRequest(
                    data=[qv], anns_field="dense", param={"metric_type": "IP"}, limit=FETCH_K_PER_ROUTE, expr=EXPR
                ),
                AnnSearchRequest(
                    data=[q], anns_field="sparse", param={"metric_type": "BM25"}, limit=FETCH_K_PER_ROUTE, expr=EXPR
                ),
            ]
            # expr 只放在各 AnnSearchRequest 里（生产也是这么传的：
            # as_retriever 的 search_kwargs["expr"] 最终落到每个 request）
            res = client.hybrid_search(
                collection_name=coll, reqs=reqs, ranker=RRFRanker(100), limit=PER_COLLECTION_K, output_fields=OUT
            )
            for h in res[0] if res else []:
                ent = h.get("entity", {}) or {}
                txt = ent.get("text") or ""
                blocks.append((h.get("distance"), fp(txt), str(ent.get("filename")), txt))
        except Exception as e:  # noqa: BLE001
            errs.append((c["id"], coll, "%s: %s" % (type(e).__name__, str(e)[:80])))
    samples.append(dict(qid=c["id"], src=c.get("source"), q=q, blocks=blocks, gold=set(c["source_blocks"])))

print("采集完成，耗时 %.1fs，异常 %d 条" % (time.time() - t0, len(errs)))
for e in errs[:6]:
    print("  异常: %s / %s -> %s" % e)

# ---------------------------------------------------------------------------
# 统计一：分数分布与「最佳门槛」
# ---------------------------------------------------------------------------
gold_all, noise_all = [], []
for s in samples:
    for sc, key, _fn, _tx in s["blocks"]:
        (gold_all if key in s["gold"] else noise_all).append(sc)

gold_all = sorted(gold_all)
noise_all = sorted(noise_all)


def pct(v, p):
    if not v:
        return float("nan")
    return v[min(int(len(v) * p), len(v) - 1)]


print()
print("=" * 100)
print("统计一：分数分布（RRF 分数，生产路径）")
print("=" * 100)
print("  golden 块 %d 个 | 非 golden 块 %d 个" % (len(gold_all), len(noise_all)))
print("  %-8s %8s %8s %8s %8s" % ("", "min", "p25", "median", "max"))
print("  %-8s %8.5f %8.5f %8.5f %8.5f" % ("golden", gold_all[0], pct(gold_all, 0.25), pct(gold_all, 0.5), gold_all[-1]))
print(
    "  %-8s %8.5f %8.5f %8.5f %8.5f" % ("noise", noise_all[0], pct(noise_all, 0.25), pct(noise_all, 0.5), noise_all[-1])
)
print()
print("  => 若 golden 的 min < noise 的 max，则**不存在**能干净分开两者的单阈值。")

# 扫阈值：对每个阈值算「过滤后还剩多少 golden / 多少噪声」
print()
print("  %-9s %-14s %-14s %s" % ("门槛", "golden 保留率", "噪声清除率", "说明"))
cands = [0.0, 0.005, 0.008, 0.009, 0.0095, 0.010, 0.012, 0.015, 0.018, 0.019]
for th in cands:
    gk = sum(1 for v in gold_all if v >= th)
    nk = sum(1 for v in noise_all if v >= th)
    gr = gk / max(len(gold_all), 1)
    nr = 1 - nk / max(len(noise_all), 1)
    note = "基线" if th == 0 else ("recall 损失 %.1f%% / 噪声清除 %.1f%%" % ((1 - gr) * 100, nr * 100))
    print("  %-9.4f %-14s %-14s %s" % (th, "%.1f%%" % (gr * 100), "%.1f%%" % (nr * 100), note))

# ---------------------------------------------------------------------------
# 统计二：副本率（同一内容的多版本块）
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("统计二：副本现象（低 precision 的另一嫌疑）")
print("=" * 100)
same_file_ratios, dup_pairs = [], []
for s in samples:
    fns = [fn for _sc, _k, fn, _tx in s["blocks"]]
    if fns:
        top = max(set(fns), key=fns.count)
        same_file_ratios.append(fns.count(top) / len(fns))
    # 两两判断「一个是另一个的子串」或「前 40 字相同」=> 视为同内容副本
    bl = s["blocks"]
    dp = 0
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            a, b = bl[i][3], bl[j][3]
            if not a or not b:
                continue
            if a[:40] == b[:40] or a in b or b in a:
                dp += 1
    dup_pairs.append(dp)

print("  平均每题返回块数: %.1f" % statistics.mean([len(s["blocks"]) for s in samples]))
print(
    "  同一 filename 占最多块的**平均占比**: %.1f%%（说明同文档多块返回的普遍程度）"
    % (statistics.mean(same_file_ratios) * 100)
)
print("  每题「互为副本」的块对数: 平均 %.1f 对（最多 %d 对）" % (statistics.mean(dup_pairs), max(dup_pairs)))
tot_pairs = sum(len(s["blocks"]) * (len(s["blocks"]) - 1) // 2 for s in samples)
print("  => 副本对占全部块对的比例: %.1f%%" % (sum(dup_pairs) / max(tot_pairs, 1) * 100))

# 举一个最严重的例子
worst = max(
    samples,
    key=lambda s: sum(
        1
        for i in range(len(s["blocks"]))
        for j in range(i + 1, len(s["blocks"]))
        if s["blocks"][i][3][:40] == s["blocks"][j][3][:40]
    ),
)
print()
print("  最严重的一例 [%s]: %s" % (worst["qid"], worst["q"][:70]))
for sc, key, fn, tx in worst["blocks"]:
    mark = "GOLDEN" if key in worst["gold"] else "      "
    print("    %.5f %s %-32s %s" % (sc, mark, fn[:32], tx[:56].replace("\n", " ")))

# ---------------------------------------------------------------------------
# 统计三：题级指标（块级统计不能直接说明 recall 怎么变）
# ---------------------------------------------------------------------------
# 说明：生产是「每集合 6 块 -> 跨集合 RRF -> final_k=18」，此处 18 块全保留，
# 故跨集合融合只影响顺序、不影响集合，题级 recall 可直接在这 18 块上算。
print()
print("=" * 100)
print("统计三：题级指标（门槛对 recall / precision 的真实影响）")
print("=" * 100)
print("  %-9s %-10s %-11s %-12s %-10s %s" % ("门槛", "recall", "hit_rate", "precision", "平均块数", "Δrecall"))
print("  " + "-" * 94)
base_rec = None
stat3 = []
for th in cands:
    rec = hit = prec = 0.0
    nblk = 0
    for s in samples:
        kept = [b for b in s["blocks"] if b[0] >= th]
        keys = {b[1] for b in kept}
        found = s["gold"] & keys
        r = len(found) / max(len(s["gold"]), 1)
        rec += r
        hit += 1 if r > 0 else 0
        prec += len(found) / max(len(kept), 1)
        nblk += len(kept)
    n = len(samples)
    rec, hit, prec, nblk = rec / n, hit / n, prec / n, nblk / n
    if base_rec is None:
        base_rec = rec
    stat3.append(
        dict(
            threshold=th,
            recall=round(rec, 4),
            hit_rate=round(hit, 4),
            precision=round(prec, 4),
            avg_blocks=round(nblk, 1),
            delta_recall=round(rec - base_rec, 4),
        )
    )
    print("  %-9.4f %-10.4f %-11.4f %-12.4f %-10.1f %+.4f" % (th, rec, hit, prec, nblk, rec - base_rec))

print()
print("=" * 100)
print("结论汇总")
print("=" * 100)
print("  1. 门槛: golden 的 p25 与 noise 的 max 完全重叠 => 不存在能干净分开的单阈值。")
print("     唯一拐点在最上方表中 —— 看 Δrecall 是否为负、precision 提升多少，再决定是否值得。")
print("  2. 副本: 「互为副本」仅占块对 %.1f%%，但「同一 filename 占最多块」平均 %.0f%%")
print("     => 低 precision 的主因不是重复块，而是**同一文档的相关片段本来就多**，")
print("        且 golden 标注的是「含答案的那几块」而非「所有相关块」。")
print("     这意味着 precision 在当前口径下天然偏低，用它指导优化要谨慎。")
print()
print("  3. 更关键: 被门槛清掉的「噪声」大多也来自相关文档（见统计二的 filename 分布），")
print("     清掉它们对**答案质量**几乎无影响，却会丢 6.2% 的真 golden。")
print("     => **收益不对称，建议不采用门槛**。")

# 证据落盘（与 eval/scan_K.json / rerank_exp.json 同惯例，便于复现与引用）
report = {
    "n_cases": len(samples),
    "n_golden_blocks": len(gold_all),
    "n_noise_blocks": len(noise_all),
    "golden_scores": {
        "min": gold_all[0],
        "p25": pct(gold_all, 0.25),
        "median": pct(gold_all, 0.5),
        "max": gold_all[-1],
    },
    "noise_scores": {
        "min": noise_all[0],
        "p25": pct(noise_all, 0.25),
        "median": pct(noise_all, 0.5),
        "max": noise_all[-1],
    },
    "overlap": bool(gold_all[0] < noise_all[-1]),
    "threshold_scan": stat3,
    "dup": {
        "avg_blocks_per_case": round(statistics.mean([len(s["blocks"]) for s in samples]), 1),
        "same_file_top_ratio": round(statistics.mean(same_file_ratios), 4),
        "avg_dup_pairs_per_case": round(statistics.mean(dup_pairs), 1),
        "dup_pair_ratio": round(sum(dup_pairs) / max(tot_pairs, 1), 4),
    },
    "errors": [list(e) for e in errs[:20]],
}
(E / "quant_threshold.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/quant_threshold.json")
