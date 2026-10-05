"""导出块清单（`chunk_index.json`）—— 换语料后重造评测的第一步。

## 为什么需要它

`golden_dataset.jsonl` 的标注指向「具体的块」。块的 ID 是**内容指纹**
（`md5(text)[:12]`），所以：

    换一批文档 / 重新切片 / 改一个标点  ->  ID 变了  ->  旧标注意义消失

要重造标注，先得知道「现在库里有哪些块」。这个脚本就干这件事。

## 用法

    # 导出全部集合（默认）
    .venv\\Scripts\\python.exe eval\\export_chunks.py

    # 只看某个集合 / 抽样
    .venv\\Scripts\\python.exe eval\\export_chunks.py --collection kb_frontend_bge_large_1024 --limit 50

产物：`eval/chunk_index.json`
    [{"id","text","filename","title","source","collection"}, ...]

`id` 的算法与 `metrics.chunk_id` **必须完全一致**，否则标注对不上。
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sy

# 解释器护栏：裸 python（Anaconda）跑本脚本会 ModuleNotFoundError，自动切到项目 .venv
_HERE = _pl.Path(__file__).resolve().parent
if str(_HERE) not in _sy.path:
    _sy.path.insert(0, str(_HERE))
import venv_guard  # noqa: E402

venv_guard.ensure_venv()

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "chunk_index.json"

sys.path.insert(0, str(HERE))
from rag_kb.utils.env_utils import DOC_SOURCES, MILVUS_URI  # noqa: E402


def chunk_id(text: str) -> str:
    """与 `metrics.chunk_id` 保持字节级一致。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _target_collections(explicit: str = "") -> list:
    if explicit:
        return [explicit]
    seen, out = set(), []
    for s in DOC_SOURCES:
        if not s.get("enabled", True):
            continue
        c = s.get("collection")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def export(collection: str, limit: int, client) -> list:
    """分页拉取一个集合的块，返回统一结构的字典列表。"""
    rows, offset, page = [], 0, 1000
    while True:
        batch = client.query(
            collection_name=collection,
            filter="id >= 0",
            output_fields=["text", "filename", "title", "source"],
            limit=page,
            offset=offset,
        )
        if not batch:
            break
        for r in batch:
            text = r.get("text") or ""
            rows.append(
                {
                    "id": chunk_id(text),
                    "text": text,
                    "filename": r.get("filename") or "",
                    "title": r.get("title") or "",
                    "source": r.get("source") or "",
                    "collection": collection,
                }
            )
        offset += len(batch)
        if len(batch) < page or (limit and len(rows) >= limit):
            break
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 Milvus 块清单，供评测标注使用")
    ap.add_argument("--collection", default="", help="只导出指定集合（默认全部 enabled 源）")
    ap.add_argument("--limit", type=int, default=0, help="每个集合最多导出多少块（0 = 全部）")
    ap.add_argument("--out", default=str(OUT), help="输出路径")
    args = ap.parse_args()

    from pymilvus import MilvusClient

    targets = _target_collections(args.collection)
    print("=" * 84)
    print("导出块清单")
    print("=" * 84)
    print("Milvus :", MILVUS_URI)
    print("集合   :", targets)
    print()

    client = MilvusClient(uri=MILVUS_URI)
    all_rows = []
    for c in targets:
        try:
            rows = export(c, args.limit, client)
        except Exception as e:  # noqa: BLE001
            print("  [跳过] %-40s %s: %s" % (c, type(e).__name__, str(e)[:80]))
            continue
        print("  %-40s %5d 块" % (c, len(rows)))
        all_rows.extend(rows)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print("总计 %d 块 -> %s" % (len(all_rows), out))
    print("下一步：python eval/gen_testset.py --per-collection 3   （自动出题，生成 golden_auto.jsonl）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
