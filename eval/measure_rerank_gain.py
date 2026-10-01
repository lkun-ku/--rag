"""P3-4 前置实测：**reranker 到底能带来多少增益**（值不值得上 GPU）。

## 为什么先测这个

ROADMAP 3.4.2 只证明了「**单题**里 `bge-reranker-v2-m3` 能把第 30 名提到第 1 名」——
但**从未量化它对当前指标的影响**。而上 GPU 的成本不小（装 CUDA torch / 租 AutoDL /
**改检索器** —— reranker 需要「候选 + 每块分数」，而 `as_retriever` 把它们封装死了）。

**先花几分钟量增益，再决定要不要动那一串。**

## 对照设计（关键：**固定预算下比精度**）

reranker 只改**顺序**，不增删块。所以它唯一的价值是：
**在「只取前 K 块」的固定预算下，让更相关的块进前 K**。

    【基线】   检索 36 块 -> 取前 K -> 判分 -> 看 golden 保留数
    【reranker】检索 36 块 -> rerank -> 取前 K -> 判分 -> 看 golden 保留数

K 取 **12**（≈ 判分实际保留数 7 的 1.7 倍，给判分留余量）。

**为什么要过判分而不是只比「前 K 里有多少 golden」**：因为**判分是下游的真实闸门**，
最终进上下文的是「判分后保留的块」——直接比「前 K 的 golden 数」会高估 reranker 的作用
（判分本来就会把不相关的筛掉）。

## 另一个关键前提（会在结论里核对）

③ 实测：`12/36` 配置下 **golden 保留 56/57 = 98%** —— 判分**几乎不筛掉 golden**。
若这个数字成立，reranker 的**天花板就是 2%**（它再怎么排也超不过「全部保留」）。
本脚本会把这个天花板一并算出来，避免「提升 2% 却以为是大改善」。

## 成本

CPU 上 reranker 约 1s/块 → 36 块 ≈ 36s/题。取 LIMIT 题（默认 8）≈ 5 分钟。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
# [FIX] **本脚本不能设 HF_HUB_OFFLINE=1**（其它 eval 脚本为了不联网都设了，
# 但 reranker 首次运行需要下载 ~2.3GB 权重）。这里允许联网，并默认走**国内镜像**
# （huggingface.co 在本网络不可达，hf-mirror.com 实测 2.3s 可连）。
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import metrics  # noqa: E402
from rag_kb.tools.retriever_tools import retriever  # noqa: E402
from rag_kb.graph2.grader_chain import retrieval_grader_chain  # noqa: E402

LIMIT = int(os.environ.get("RERANK_LIMIT", "8"))
TOPK = int(os.environ.get("RERANK_TOPK", "12"))
MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
cases = [c for c in cases if c.get("source_blocks")][:LIMIT]

print("=" * 104)
print("P3-4 前置实测：reranker 的增益（模型 %s）" % MODEL)
print("=" * 104)
print("  题数 %d | 预算 K=%d 块 | **两档都要过判分**（判分是下游真实闸门）" % (len(cases), TOPK))
print()

# ---- 加载 reranker ---------------------------------------------------------
t0 = time.time()
try:
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(MODEL, max_length=512, device="cpu")
    print("  reranker 加载完成（%.1fs，CPU）" % (time.time() - t0))
except Exception as e:  # noqa: BLE001
    print("  [**无法加载 reranker**] %s: %s" % (type(e).__name__, str(e)[:160]))
    print("  （若是缺模型文件，需要联网下载 ~2.3GB；本机的 HF_HUB_OFFLINE=1 会阻止下载）")
    sys.exit(2)


def grade_keep(q, docs):
    """对这些块跑判分，返回 (保留数, 保留的 golden 数, golden 总数)。"""
    keep_ids, g_ids = set(), set()
    for d in docs:
        cid = metrics.chunk_id(d.page_content)
        try:
            sc = retrieval_grader_chain.invoke({"question": q, "document": d.page_content})
            ok = str(getattr(sc, "binary_score", "")).strip().lower() == "yes"
        except Exception:  # noqa: BLE001
            ok = True  # 与生产一致：异常保守保留
        if ok:
            keep_ids.add(cid)
    return keep_ids


rows = []
t_all = time.time()
for i, c in enumerate(cases, 1):
    q = c["q"]
    gids = set(c["source_blocks"])
    docs = retriever.invoke(q)
    ids = [metrics.chunk_id(d.page_content) for d in docs]
    g_in_ret = len(gids & set(ids))

    # 检索原始顺序的前 K
    base_docs = docs[:TOPK]

    # reranker 重排（对**全部**候选打分，再取前 K —— 这是 reranker 的真实用法）
    t0 = time.time()
    pairs = [(q, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    rank_docs = [d for _s, d in sorted(zip(scores, docs), key=lambda x: -x[0])][:TOPK]
    rerank_sec = time.time() - t0

    keep_base = grade_keep(q, base_docs)
    keep_rr = grade_keep(q, rank_docs)
    gb = len(gids & keep_base)
    gr = len(gids & keep_rr)

    rows.append(
        {
            "id": c.get("id"),
            "n_golden": len(gids),
            "g_in_ret": g_in_ret,
            "n_ret": len(docs),
            "base_keep": len(keep_base),
            "rr_keep": len(keep_rr),
            "g_base": gb,
            "g_rr": gr,
            "rerank_sec": round(rerank_sec, 1),
        }
    )
    print(
        "  [%d/%d] %-6s golden=%2d（检索命中 %d）| 基线前%d: 保留%2d/golden %d"
        "  ->  rerank前%d: 保留%2d/golden **%d**  (rerank %.0fs)"
        % (
            i,
            len(cases),
            c.get("id"),
            len(gids),
            g_in_ret,
            TOPK,
            len(keep_base),
            gb,
            TOPK,
            len(keep_rr),
            gr,
            rerank_sec,
        )
    )

print()
print("  rerank 总耗时 %.0fs（平均 %.1fs/题）" % (time.time() - t_all, (time.time() - t_all) / max(len(cases), 1)))

# ---- 汇总 ------------------------------------------------------------------
n = len(rows)
g_base = sum(r["g_base"] for r in rows)
g_rr = sum(r["g_rr"] for r in rows)
g_ret = sum(r["g_in_ret"] for r in rows)
g_all = sum(r["n_golden"] for r in rows)
better = sum(1 for r in rows if r["g_rr"] > r["g_base"])
worse = sum(1 for r in rows if r["g_rr"] < r["g_base"])

print()
print("=" * 104)
print("汇总（%d 题）" % n)
print("=" * 104)
print("  golden 总数            : %d（其中被检索命中 %d = %.0f%%）" % (g_all, g_ret, 100 * g_ret / max(g_all, 1)))
print("  **基线**前%d 且判分后保留 : **%d**" % (TOPK, g_base))
print("  **rerank**后前%d 且判分保留: **%d**" % (TOPK, g_rr))
print(
    "  -> 增益 **%+d 块**（%.1f%%）| 提升 %d 题 / 下降 %d 题"
    % (g_rr - g_base, 100 * (g_rr - g_base) / max(g_base, 1), better, worse)
)
print()
print("  **天花板**：检索命中的 golden 共 %d 块 —— 这是 reranker 无论如何排都超不过的上限。" % g_ret)
print(
    "             当前基线已拿到 %d（%.0f%%），**剩余空间只有 %d 块（%.1f%%）**。"
    % (g_base, 100 * g_base / max(g_ret, 1), g_ret - g_base, 100 * (g_ret - g_base) / max(g_ret, 1))
)
print()
print("  判读：")
if g_rr - g_base <= max(1, int(0.03 * max(g_base, 1))):
    print("    **增益在 3%% 以内** -> reranker 的收益不足以抵消成本（GPU/改检索器），")
    print("    建议 P3-4 关闭（或降级为「等有 GPU 时再评估」）。")
else:
    print("    **增益显著** -> 值得推进：装 CUDA 版 torch（本机 4060 可用），")
    print("    并按 4.17 的设想改造 MultiCollectionRetriever 的内部实现。")

(HERE / "measure_rerank_gain.json").write_text(
    json.dumps(
        {
            "model": MODEL,
            "topk": TOPK,
            "n_cases": n,
            "rows": rows,
            "g_all": g_all,
            "g_in_ret": g_ret,
            "g_base": g_base,
            "g_rr": g_rr,
            "gain": g_rr - g_base,
            "ceiling_left": g_ret - g_base,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/measure_rerank_gain.json")
