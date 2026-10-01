"""reranker 离线效果实验（ROADMAP 3.4.2 的评估）。

## 要回答的问题

reranker **不改变检索返回的集合**，只改变**顺序** —— 因此对 recall（是否召回）
本身没有影响。它真正的作用是：**在截断时保留更相关的块**。

所以实验设计为：检索拿 18 块（FINAL_K）→ rerank → 取 **top-N** → 测指标。

  * N=18（不截断）：recall 不变，precision 不变（只是顺序变了）
  * N=6 ：若 recall 保持高位而 precision 大幅提升 -> reranker 值得接入

## 为什么关心

线上把 18 块全喂给 LLM 会稀释语义、干扰判分（grade_documents_node 还要过滤）。
若能只给最相关的 6 块且不漏目标，就是纯收益。

## 耗时预期

CPU 推理约 0.5~1s/块，57 题 × 18 块 ≈ 15~20 分钟。
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

from sentence_transformers import CrossEncoder  # noqa: E402

from tools.retriever_tools import retriever  # noqa: E402

E = ROOT / "eval"
cases = [json.loads(l) for l in (E / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
MODEL = ROOT / "RAG_PROJECT" / "models" / "bge-reranker-v2-m3"
OUT = E / "rerank_exp.json"


def fp(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


print("加载 reranker...", flush=True)
t0 = time.time()
model = CrossEncoder(str(MODEL), max_length=512)
print("  完成 %.1fs" % (time.time() - t0), flush=True)

CUTOFFS = (3, 6, 9, 12, 18)
agg = {n: {"rec": 0.0, "prec": 0.0, "hit": 0} for n in CUTOFFS}
agg["base"] = {"rec": 0.0, "prec": 0.0, "hit": 0}  # 不 rerank 的基线
per_case = []

t0 = time.time()
for i, c in enumerate(cases, 1):
    q, gid = c["q"], set(c["source_blocks"])
    try:
        docs = retriever.invoke(q)
    except Exception as e:  # noqa: BLE001
        print("  [%s] 检索失败 %s" % (c["id"], type(e).__name__), flush=True)
        continue

    rids = [fp(d.page_content) for d in docs]

    # 基线（原始 RRF 顺序）
    found = len(gid & set(rids))
    agg["base"]["rec"] += found / len(gid)
    agg["base"]["prec"] += found / max(len(rids), 1)
    agg["base"]["hit"] += 1 if found else 0

    # rerank
    pairs = [(q, d.page_content) for d in docs]
    scores = model.predict(pairs, batch_size=8)
    order = sorted(range(len(docs)), key=lambda k: -float(scores[k]))
    ranked = [rids[k] for k in order]

    row = {"id": c["id"], "source": c.get("source"), "n_docs": len(docs), "base_recall": found / len(gid), "ranks": {}}
    for n in CUTOFFS:
        top = ranked[:n]
        f = len(gid & set(top))
        agg[n]["rec"] += f / len(gid)
        agg[n]["prec"] += f / max(len(top), 1)
        agg[n]["hit"] += 1 if f else 0
        row["ranks"][n] = f / len(gid)
    per_case.append(row)

    if i % 5 == 0 or i == len(cases):
        el = time.time() - t0
        print("  %2d/%d  已用 %.0fs  预计剩余 %.0fs" % (i, len(cases), el, el / i * (len(cases) - i)), flush=True)

N = len(per_case)
print()
print("=" * 96)
print("结果（%d 题，reranker=bge-reranker-v2-m3，检索 fetch_k=12）" % N)
print("=" * 96)
print("  %-14s %-12s %-12s %-10s %s" % ("配置", "context_recall", "precision", "hit_rate", "说明"))
print("  " + "-" * 88)


def show(label, d, note=""):
    print("  %-14s %-12.4f %-12.4f %-10.4f %s" % (label, d["rec"] / N, d["prec"] / N, d["hit"] / N, note))


show("基线(不重排)", agg["base"], "18 块全给下游")
for n in CUTOFFS:
    note = "截断到 top-%d" % n
    if n == 18:
        note = "全量但重排（对照）"
    show("rerank top-%d" % n, agg[n], note)

print()
print("  结论判据：若 top-6 的 recall 接近基线且 precision 更高 -> 值得接入")
res = {
    "baseline": {k: v / N for k, v in agg["base"].items()},
    "rerank": {str(n): {k: v / N for k, v in agg[n].items()} for n in CUTOFFS},
    "per_case": per_case,
    "n_cases": N,
}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("  结果已写入 %s" % OUT.name)
