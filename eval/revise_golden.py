"""修订 `golden_dataset.jsonl`：把重建后**已消失的扁平孪生块**从 `source_blocks` 里摘掉。

## 依据（全部来自实测）

重建后核对（`eval/verify_rebuild_coverage.py` 与失效形态核对）得到三条硬事实：

1. **36 个 golden 块失效，形态统计为 `{扁平: 33, 扁平(PS版): 3}`** ——
   **零个结构化块失效**。也就是说，消失的正是本次要去掉的冗余，
   golden 里对它们的标注**本来就该撤掉**（要求召回一个已不存在的冗余块没有意义）。
2. 消失块的内容**没有被丢**（78 个消失块中 75 个词级覆盖 >= 0.95；
   另 3 个是扫描件 OCR 重跑导致的逐字差异，与本次改动无关）。
3. 所以修订规则很简单：**摘掉失效 ID，其余不动**。

## 唯一需要补块的情形

若某题摘掉后 `source_blocks` **为空**（如 n030 只有 1 个 golden 且是扁平块），
则用**内容匹配**从新库里补一个进来：取该题「失效旧块」的正文，
在新库里找**词重叠率最高**的块（同一文件优先）。**匹配不到就不补**，
并把它列进报告供人工处理 —— 宁可少一道题，不硬塞一个错块。

## 用法

    python eval/revise_golden.py              # dry-run：只输出预览，不改文件
    python eval/revise_golden.py --apply      # 真正写入（会先备份原文件）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
import metrics  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
               "kb_manuals_bge_large_1024"]
_WORD_RE = re.compile(r"[A-Za-z\u4e00-\u9fff0-9._/:#@\-]{4,}")

ap = argparse.ArgumentParser(description="按重建后的块集合修订 golden 的 source_blocks")
ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
args = ap.parse_args()

# ---- 新库索引 ----
client = MilvusClient(uri=MILVUS_URI)
new_blocks = []          # [(chunk_id, filename, text)]
new_ids = set()
for coll in COLLECTIONS:
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "text"], limit=8000):
        t = r.get("text") or ""
        cid = metrics.chunk_id(t)
        new_ids.add(cid)
        new_blocks.append((cid, r.get("filename", "?"), t))

# ---- 旧库（备份）索引：失效块的正文与来源 ----
bk = sorted((HERE / "_backup_collections").glob("backup_*.json"))[-1]
old = json.loads(bk.read_text(encoding="utf-8"))
old_by_id = {}
for _coll, v in old["collections"].items():
    for it in v.get("items", []):
        old_by_id[it["chunk_id"]] = it


def words(t: str):
    return set(_WORD_RE.findall(t or ""))


def best_match(old_text: str, old_fn: str):
    """在**同一文件**的新块里找词重叠率最高者（同文件优先；找不到再看全库）。"""
    ow = words(old_text)
    if not ow:
        return None, 0.0
    def scan(pool):
        best, best_r = None, 0.0
        for cid, fn, t in pool:
            nw = words(t)
            if not nw:
                continue
            r = len(ow & nw) / len(ow)
            if r > best_r:
                best, best_r = (cid, fn, t), r
        return best, best_r
    same = [b for b in new_blocks if b[1] == old_fn]
    best, r = scan(same) if same else (None, 0.0)
    if best is None or r < 0.5:
        b2, r2 = scan(new_blocks)
        if r2 > r:
            best, r = b2, r2
    return best, r


golden_path = HERE / "golden_dataset.jsonl"
cases = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines()
         if l.strip()]

# 人工指定的补块（自动匹配失效、但内容确实还在的情形）。
#
# n030「扫描版性能 FAQ 里提到了设置 nlist / nprobe 的问题吗？」：
# 旧 golden 是扫描件第 1 页（`31109bf8138c`，2144 字符）。该文件走 **OCR**，
# 重跑一次 OCR 的逐字结果就会不同（`Settingnlist…` -> `Setingnlist …`），
# 于是自动匹配的最佳重叠只有 **0.23**（阈值 0.5）。
# 但内容确实还在 —— 重建后按 FAQ 条目切成了 6 块。人工核对后钉住
# `28321778ec2b`（title 就是「Howtosetnlist andnprobeforIVFindexes?」，744 字符，
# 正文含 nlist/nprobe 的完整回答）。
MANUAL_FIX = {"n030": "28321778ec2b"}

print("=" * 104)
print("golden 修订预览（%s）" % ("**APPLY**" if args.apply else "dry-run，不改文件"))
print("=" * 104)
print("  规则：摘掉已消失的块 ID；若摘成空则按内容匹配补一个（同文件优先，重叠 >= 0.5）")
print()

n_changed = n_removed = n_refilled = n_emptied = 0
refills = []
for cc in cases:
    gids = list(cc.get("source_blocks") or [])
    if not gids:
        continue
    gone = [g for g in gids if g not in new_ids]
    if not gone:
        continue
    kept = [g for g in gids if g in new_ids]
    note = ""
    if not kept and cc.get("id") in MANUAL_FIX:
        kept = [MANUAL_FIX[cc["id"]]]
        n_refilled += 1
        note = "人工指定 %s（OCR 重跑导致自动匹配失效，已人工核对）" % MANUAL_FIX[cc["id"]]
        refills.append((cc.get("id"), MANUAL_FIX[cc["id"]], 1.0))
    elif not kept:
        # 需要补块：用第一个失效旧块的正文去匹配
        it = old_by_id.get(gone[0])
        if it:
            m, r = best_match(it.get("text") or "", it.get("filename") or "")
            if m and r >= 0.5:
                kept = [m[0]]
                n_refilled += 1
                note = "补块 %s（重叠 %.2f，同源于 %s）" % (m[0], r, (it.get("filename") or "")[:26])
                refills.append((cc.get("id"), m[0], round(r, 3)))
            else:
                n_emptied += 1
                note = "**补块失败（最佳重叠 %.2f）-> 该题将无 golden**" % r
        else:
            n_emptied += 1
            note = "**失效块不在备份里 -> 无法补块**"
    cc["source_blocks"] = kept
    n_changed += 1
    n_removed += len(gone)
    print("  [%s] golden %d -> %d  摘掉 %d %s" % (
        cc.get("id"), len(gids), len(kept), len(gone), note))

print()
print("  汇总：修订 %d 题 / 摘掉 %d 个失效 ID / 补块 %d 题 / 补块失败 %d 题"
      % (n_changed, n_removed, n_refilled, n_emptied))
if refills:
    print()
    print("  补块明细:")
    for cid, nid, r in refills:
        print("     [%s] + %s（重叠 %.2f）" % (cid, nid, r))

if args.apply:
    shutil.copy2(golden_path, golden_path.with_suffix(".jsonl.bak"))
    with golden_path.open("w", encoding="utf-8") as f:
        for cc in cases:
            f.write(json.dumps(cc, ensure_ascii=False) + "\n")
    print()
    print("  [APPLY] 已写入 %s（原文件备份为 golden_dataset.jsonl.bak）" % golden_path.name)
else:
    print()
    print("  这是 **dry-run**：文件未改动。确认无误后加 --apply 执行。")
