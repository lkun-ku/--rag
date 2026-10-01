"""重建前的备份：导出三个集合的全部块（块 ID / 文本 / metadata）为 JSON。

## 为什么需要（重建不可逆）

② 的重建走 `write_milvus.ingest_source(recreate=True)` —— **删除并重建集合**，
不可逆。备份保证两件事可核对：

  1. **内容覆盖**：重建后消失的块，其内容能否在新集合里找到（这是 4.17 的核心主张）；
  2. **golden 修订**：`golden_dataset.jsonl` 的 `source_blocks` 是**内容指纹**，
     可以对新旧两套块集合分别求值 —— 哪些 golden 会失效、失效后应换成哪个新块，
     都能从这份备份推出来。

## 输出

    eval/_backup_collections/backup_<UTC 时间戳>.json

结构与 `MilvusClient.query` 的字段一致（filename / title / text / category / source），
外加每块的 `chunk_id`（= `metrics.chunk_id(text)`，与 golden 同一套指纹）。
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
import metrics  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024", "kb_manuals_bge_large_1024"]

out_dir = HERE / "_backup_collections"
out_dir.mkdir(exist_ok=True)

client = MilvusClient(uri=MILVUS_URI)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
payload = {"stamp": stamp, "milvus_uri": MILVUS_URI, "collections": {}}

print("=" * 96)
print("备份三个集合（重建前）")
print("=" * 96)
total = 0
for coll in COLLECTIONS:
    try:
        rows = client.query(
            collection_name=coll,
            filter="",
            output_fields=["filename", "title", "text", "category", "source"],
            limit=8000,
        )
    except Exception as e:  # noqa: BLE001
        print("  %-30s 查询失败: %s: %s" % (coll, type(e).__name__, e))
        payload["collections"][coll] = {"error": "%s: %s" % (type(e).__name__, e)}
        continue
    items = []
    for r in rows:
        t = r.get("text") or ""
        items.append(
            {
                "chunk_id": metrics.chunk_id(t),
                "filename": r.get("filename", ""),
                "title": r.get("title", ""),
                "category": r.get("category", ""),
                "source": r.get("source", ""),
                "text": t,
            }
        )
    payload["collections"][coll] = {"n": len(items), "items": items}
    total += len(items)
    print("  %-30s %5d 块" % (coll, len(items)))

payload["total"] = total
path = out_dir / ("backup_%s.json" % stamp)
path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
size_mb = path.stat().st_size / 1024 / 1024
print()
print("  总计 %d 块 -> %s（%.1f MB）" % (total, path.relative_to(ROOT), size_mb))

# 同时备份 golden（重建后要修订它，先留一份原始件）
import shutil  # noqa: E402

g_src = HERE / "golden_dataset.jsonl"
g_dst = out_dir / ("golden_%s.jsonl" % stamp)
if g_src.exists():
    shutil.copy2(g_src, g_dst)
    n = sum(1 for l in g_src.read_text(encoding="utf-8").splitlines() if l.strip())
    print("  golden %d 题 -> %s" % (n, g_dst.relative_to(ROOT)))

print()
print("  提示：这是**核对用**备份（可据此重建等价块集合），不是回滚快照。")
