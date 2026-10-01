"""3.5.1 前置实测：**用块 ID 精确对照** —— 跨行统计题的失败在检索层还是生成层？

## 为什么必须先测这个（而不是直接上父子块）

3.5.1「表格父子块」的立项依据是「一张表被切成多块后，模型**有时能聚合出答案、
有时判『资料不足』拒答**」。这句话**没有区分两种可能**，而两者修法完全不同：

    A. **检索层**：同一张表只有部分块被召回 -> 模型确实看不到全表
                  -> 「命中子块就带上父表其余行段」有用；
    B. **生成层**：同一张表的多块**都已经在上下文里**，模型仍聚合不好
                  -> 父子块帮助有限，瓶颈在 prompt / 判分 / 模型能力。

`ROADMAP 4.14` 已把 `final_k` 提到 24（上下文 17.9 -> 23.9 块），**A 的可能性下降** ——
所以这条实测更必要：若结论是 B，3.5.1 的优先级应下调，**不该去动 schema、重建集合**。

## v1 的两个坑（已修，记下来）

1. 查库用了 `content` 字段 -> `MilvusException code=1100 field content not exist`
   —— **真实字段名是 `text`**（`describe_collection` 可查）。不该凭印象写字段名。
2. 识别表格块只看正文前缀 `表格：` -> 会漏掉「**同一张表的另一种形态**」。
   实测 n007 的 golden 里就有这样一个块：正文以 `检索性能实测 场景 数据集 …` 开头，
   **没有 `表格：` 前缀**，却是 golden。

## v2 口径：不猜形态，直接用**块 ID** 对照

扫全库算 `metrics.chunk_id`（与 golden 同一套指纹），建立
`块 ID -> (集合, filename, 正文)` 映射；再逐题打印
「每个 golden 块长什么样、是否在召回列表里、排第几」。**这样形态无关、不会漏判。**
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024", "kb_manuals_bge_large_1024"]

# ---------------------------------------------------------------------------
# 第一步：全库建索引（块 ID -> 正文），字段名用实测确认的 `text`
# ---------------------------------------------------------------------------
client = MilvusClient(uri=MILVUS_URI)
by_id: dict = {}
n_rows = 0
for coll in COLLECTIONS:
    rows = client.query(collection_name=coll, filter="", output_fields=["filename", "text", "title"], limit=4000)
    n_rows += len(rows)
    for r in rows:
        txt = r.get("text") or ""
        by_id[metrics.chunk_id(txt)] = {
            "coll": coll,
            "filename": r.get("filename", "?"),
            "title": r.get("title", ""),
            "text": txt,
        }

print("=" * 100)
print("第一步：全库索引（用块 ID 对照，不靠正文前缀猜形态）")
print("=" * 100)
print("  块总数 %d（去重后 %d）" % (n_rows, len(by_id)))


# 「表格」块的两种形态（用于分类展示，判定本身用 ID）
def shape(txt: str) -> str:
    head = txt[:80].replace("\n", " ")
    if "表格：" in head:
        return "表格块(带前缀)"
    if "列名：" in txt[:400]:
        return "表格块(列名式)"
    return "非表格"


# ---------------------------------------------------------------------------
# 第二步：逐题对照 —— 每个 golden 块在召回列表里吗？排第几？
# ---------------------------------------------------------------------------
cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print()
print("=" * 100)
print("第二步：32 题逐题对照（golden 块 vs 召回列表）")
print("=" * 100)

from tools.retriever_tools import retriever  # noqa: E402

report = []
for c in cases:
    gids = c.get("source_blocks") or []
    if not gids:
        continue
    try:
        docs = retriever.invoke(c["q"])
    except Exception as e:  # noqa: BLE001
        print("  [%s] 检索失败: %s: %s" % (c.get("id"), type(e).__name__, e))
        continue
    rids = [metrics.chunk_id(d.page_content) for d in docs]
    s = metrics.score_retrieval(gids, rids)
    pos = {rid: i for i, rid in enumerate(rids, 1)}

    # 这一题的「表格相关性」：召回列表里含几个表格块
    shapes = collections.Counter(shape(d.page_content) for d in docs)
    n_tbl = sum(v for k, v in shapes.items() if k.startswith("表格"))

    entry = {"id": c.get("id"), "q": c["q"], "recall": s["recall"], "n_docs": len(docs), "n_table": n_tbl, "golden": []}
    for g in gids:
        info = by_id.get(g)
        if info is None:
            entry["golden"].append({"id": g, "found": "不在库里"})
            continue
        entry["golden"].append(
            {
                "id": g,
                "found": "召回第 %d 位" % pos[g] if g in pos else "**未召回**",
                "shape": shape(info["text"]),
                "filename": info["filename"],
                "head": info["text"][:70].replace("\n", " "),
            }
        )
    report.append(entry)

# 只打印「召回不全」的题（recall < 1），这才是要诊断的
bad = [e for e in report if e["recall"] < 1.0]
print("  有 golden 的题 %d 道，其中 recall < 1 的 %d 道（下面逐块展开）" % (len(report), len(bad)))
for e in bad:
    print()
    print("  [%s] recall=%.3f  块数=%d  其中表格块=%d" % (e["id"], e["recall"], e["n_docs"], e["n_table"]))
    print("       Q: %s" % e["q"][:86])
    for g in e["golden"]:
        if "found" not in g:
            continue
        print(
            "       golden %s  %-12s %-24s | %s"
            % (g["id"], g.get("found", "?"), g.get("shape", "?")[:24], g.get("head", "")[:56])
        )

# ---------------------------------------------------------------------------
# 第三步：核心汇总 —— 未召回的 golden 是什么形态？
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("第三步：核心汇总 —— 未召回的 golden 块都是什么形态？")
print("=" * 100)
missed_shape = collections.Counter()
missed_rows = []
for e in report:
    for g in e["golden"]:
        if g.get("found") == "**未召回**":
            missed_shape[g["shape"]] += 1
            missed_rows.append((e["id"], g))
print("  未召回 golden 共 %d 块，按形态:" % len(missed_rows))
for k, v in missed_shape.most_common():
    print("     %-22s %d" % (k, v))
print()
for cid, g in missed_rows[:20]:
    print("     [%s] %-22s %-26s | %s" % (cid, g["shape"][:22], g["filename"][:26], g["head"][:52]))

print()
print("  判读:")
print("    若未召回的 golden 多为「表格块」 -> 问题在**检索层**，3.5.1 有直接价值；")
print("    若 golden 基本都召回了、只是模型没聚合 -> 问题在**生成层**，3.5.1 价值有限。")

(HERE / "diag_table_parent_child.json").write_text(
    json.dumps(
        {"n_blocks": len(by_id), "report": report, "missed": [{"id": c, **g} for c, g in missed_rows]},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/diag_table_parent_child.json")
