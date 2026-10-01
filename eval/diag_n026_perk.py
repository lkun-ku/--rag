"""n026 的 per_k 对比实测：7 个 golden 块「在库里却没召回」，是不是 per_k=8 截断造成的？

## 假设（待验证，不采信）

4.14 把 `PER_COLLECTION_K` 定在 8（每集合取 8 块）、`FINAL_K=24`。
n026 的 8 个 golden 分布在 4 个文件、**至少跨 manuals 与 papers 两个集合**。
若某个集合里与问题相关的块多于 8 个，前 8 名之外的就永远进不了上下文。

## 做法

对**同一问题**，用不同的 `per_k` 各跑一次检索，记录：
    * 返回总块数
    * **golden 命中数**（关键指标）
    * 每个 golden 块在各配置下的**排名**（若被召回）

`per_k` 取 8 / 16 / 32 三档，`final_k` 同步放大（保持生产里"每集合配额 × 集合数"的关系）。

**这能直接回答**：召回不到是「配额截断」还是「相似度不够」（后者放大配额也没用）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
import tools.retriever_tools as rt  # noqa: E402

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

TARGETS = ["n026", "n027", "n022", "n025"]
GRID = [(8, 24), (16, 48), (32, 96)]

print("=" * 104)
print("per_k 对比实测：配额放大能否召回更多 golden？")
print("=" * 104)
print("  %-6s %-8s %-8s %-10s %-9s %s" % ("题", "golden", "per_k", "返回块数", "命中", "说明"))
print("  " + "-" * 100)

report = {}
for cid in TARGETS:
    c = next((x for x in cases if x.get("id") == cid), None)
    if not c or not c.get("source_blocks"):
        continue
    gids = set(c["source_blocks"])
    report[cid] = {"q": c["q"], "n_golden": len(gids), "runs": []}
    for pk, fk in GRID:
        try:
            r = rt._build_multi_retriever(per_k=pk, final_k=fk)
            docs = r.invoke(c["q"])
        except Exception as e:  # noqa: BLE001
            print("  %-6s %-8d %-8d **失败** %s" % (cid, len(gids), pk, type(e).__name__))
            continue
        ids = [metrics.chunk_id(d.page_content) for d in docs]
        hit = gids & set(ids)
        ranks = {g: (ids.index(g) + 1) for g in gids if g in ids}
        report[cid]["runs"].append({"per_k": pk, "final_k": fk, "n": len(ids), "hit": len(hit), "ranks": ranks})
        print(
            "  %-6s %-8d %-8d %-10d %d/%-7d %s"
            % (cid, len(gids), pk, len(ids), len(hit), len(gids), "命中排名 %s" % sorted(ranks.values()))
        )

print()
print("=" * 104)
print("判读")
print("=" * 104)
for cid, v in report.items():
    runs = v["runs"]
    if len(runs) < 2:
        continue
    a, b = runs[0], runs[-1]
    if b["hit"] > a["hit"]:
        print(
            "  %-6s 命中 %d -> %d（per_k %d -> %d）=> **配额截断**是主因，放大配额有效"
            % (cid, a["hit"], b["hit"], a["per_k"], b["per_k"])
        )
    elif b["hit"] == a["hit"]:
        print(
            "  %-6s 命中 %d -> %d（per_k %d -> %d）=> 放大配额**无改善**，"
            "瓶颈在相似度/语料本身" % (cid, a["hit"], b["hit"], a["per_k"], b["per_k"])
        )

(HERE / "diag_n026_perk.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/diag_n026_perk.json")
