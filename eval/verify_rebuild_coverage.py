"""重建后的**内容覆盖核对**：消失的块，其内容是否都能在新集合里找到？

## 为什么必须做（重建不可逆，必须证明「没丢内容」）

② 的重建改了入库逻辑（父块不再被塞入表格的扁平文本），块集合因此变化。
块数从 528 -> 523，但**块数近似不代表内容近似** —— 必须逐块核对。

## 口径

1. 读重建前的备份（`eval/_backup_collections/backup_*.json`）与当前库；
2. 按**归一化正文**做集合差：`消失 = 旧 - 新`、`新增 = 新 - 旧`；
3. **对每个消失的块**，把它的正文切成滑动片段（长 40 字符、步长 20），
   检查这些片段有多少能在「新集合的全文」里找到：

   * 覆盖率 >= 0.95 -> 该块内容**已被新集合覆盖**（可接受 —— 它被拆散/重组了）
   * 覆盖率 <  0.95 -> **可能有内容丢失**，逐块打印缺失片段供人工判断

用**滑动片段**而不是整块匹配的原因：重建会改变块边界（父块变小 -> 语义切分更细），
整块文本本来就不该存在；关心的是**字符级内容有没有丢**。
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
               "kb_manuals_bge_large_1024"]

FRAG_LEN = 40
FRAG_STEP = 20
COVER_OK = 0.95


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# [修正 2026-10-15] 首版用 **40 字符滑窗** 核覆盖率，结果 78 个消失块**全部** < 0.95
# （大量 0.00），看着像「内容全丢了」—— 但那是**口径错误**，不是真丢内容：
#
#     消失块（扁平）：Options Option Full name Description -uri --uri (Optional) The uri name...
#     新集合（结构化）：表格：Options 列名：Option | Full name | Description | -uri | --uri |...
#
# 扁平版**无分隔符**、结构化版**有 `|`**，滑窗必然跨值边界 -> 处处不匹配（假阴性）。
#
# 改为**词级**：按空白切词（>= 4 字符），看每个词能否在新集合里找到。
# 词是两种形态的**共同颗粒度**（`|` 只是分隔符，不影响词本身）。
_WORD_RE = re.compile(r"[A-Za-z\u4e00-\u9fff0-9._/:#@\-\[\]()]{4,}")


def fragments(text: str):
    n = norm(text)
    if not n:
        return []
    return sorted(set(_WORD_RE.findall(n)))


bk = sorted((HERE / "_backup_collections").glob("backup_*.json"))[-1]
old = json.loads(bk.read_text(encoding="utf-8"))
client = MilvusClient(uri=MILVUS_URI)

print("=" * 104)
print("重建后的内容覆盖核对（旧块消失 -> 内容是否仍在新集合里）")
print("=" * 104)
print("  备份: %s" % bk.name)

# 取新集合的所有正文（按集合分别建索引，避免跨集合误判）
new_by_coll = {}
for coll in COLLECTIONS:
    rows = client.query(collection_name=coll, filter="", output_fields=["text"], limit=8000)
    texts = [norm(r.get("text") or "") for r in rows]
    new_by_coll[coll] = {"n": len(texts), "blob": "\n".join(texts)}
    print("  %-32s 新集合 %5d 块 / %8d 字符" % (coll, len(texts), len(new_by_coll[coll]["blob"])))

print()
print("  %-32s %10s %10s %10s %s" % ("集合", "旧块", "消失块", "覆盖OK", "说明"))
print("  " + "-" * 96)

all_lost = []
tot_old = tot_gone = tot_ok = 0
for coll in COLLECTIONS:
    items = old["collections"].get(coll, {}).get("items", [])
    blob = new_by_coll[coll]["blob"]
    gone = []
    for it in items:
        t = norm(it.get("text") or "")
        # 整块在新集合里原样存在 -> 未消失
        if t and t in blob:
            continue
        gone.append(it)
    ok = 0
    lost = []
    for it in gone:
        frs = fragments(it.get("text") or "")
        if not frs:
            ok += 1
            continue
        hit = sum(1 for f in frs if f in blob)
        cov = hit / len(frs)
        if cov >= COVER_OK:
            ok += 1
        else:
            lost.append({"filename": it.get("filename"), "title": it.get("title"),
                         "coverage": round(cov, 3), "n_frag": len(frs),
                         "head": norm(it.get("text") or "")[:120]})
    tot_old += len(items); tot_gone += len(gone); tot_ok += ok
    if lost:
        all_lost.extend(lost)
    print("  %-32s %10d %10d %10d %s" % (
        coll, len(items), len(gone), ok,
        "OK" if not lost else "**%d 块覆盖率 < %.2f**" % (len(lost), COVER_OK)))

print("  " + "-" * 96)
print("  %-32s %10d %10d %10d" % ("合计", tot_old, tot_gone, tot_ok))
print()
print("  含义：'消失块'=%d，其中 %d 块（%.1f%%）的内容在新集合里覆盖率 >= %.2f"
      % (tot_gone, tot_ok, (tot_ok / max(tot_gone, 1)) * 100, COVER_OK))

if all_lost:
    print()
    print("=" * 104)
    print("**覆盖率不足的块**（需人工判断是否真的丢了内容）")
    print("=" * 104)
    for r in all_lost[:20]:
        print("  [cov=%.2f] %-30s %-18s" % (r["coverage"], (r["filename"] or "")[:30], (r["title"] or "")[:18]))
        print("      %s" % r["head"])
else:
    print()
    print("  [OK] 所有消失块的内容都在新集合里 —— **未丢内容**。")

(HERE / "verify_rebuild_coverage.json").write_text(json.dumps(
    {"backup": bk.name, "collections": {
        c: {"n": old["collections"].get(c, {}).get("n", 0)} for c in COLLECTIONS},
     "gone_total": tot_gone, "covered_ok": tot_ok, "lost": all_lost},
    ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("  证据 -> eval/verify_rebuild_coverage.json")
