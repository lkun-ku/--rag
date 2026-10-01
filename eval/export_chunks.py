"""从 Milvus 导出块清单，作为 Golden Dataset 的 `source_blocks` 依据。

产物：`eval/chunk_index.json`

```json
{
  "source": "milvus",
  "server_version": "3.0.1",
  "total_chunks": 135,
  "collections": {
    "kb_milvus_bge_large_1024": {
      "count": 102,
      "chunks": [
        {"chunk_id": "a1b2c3d4e5f6", "filename": "overview.md",
         "title": "...", "chars": 547, "preview": "..."}
      ]
    }
  }
}
```

chunk_id 用 `md5(page_content)[:12]` —— 内容指纹，与 ROADMAP 1.5.1 的设计一致。
这样 Golden Dataset 里标注的 `source_blocks` 是**内容指纹**而非行 ID，
即使重建集合（换维度/换模型）也能对应上（只要块内容不变）。

用法：
    $env:PYTHONPATH="RAG_PROJECT"
    .venv\\Scripts\\python.exe eval\\export_chunks.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")

from pymilvus import MilvusClient  # noqa: E402

from utils.env_utils import DOC_SOURCES, MILVUS_URI  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "chunk_index.json"
# Milvus query 单次上限（本项目块数远小于此值，留足余量）
QUERY_LIMIT = 16384


def chunk_id(text: str) -> str:
    """内容指纹：与检索无关，只保证「同样内容 -> 同样 ID」。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    print("=" * 78)
    print("导出块清单（Golden Dataset 的 source_blocks 依据）")
    print("=" * 78)
    print(f"MILVUS_URI = {MILVUS_URI}")

    client = MilvusClient(uri=MILVUS_URI)
    try:
        version = client.get_server_version()
    except Exception:  # noqa: BLE001
        version = "unknown"
    print(f"服务端版本 = {version}")

    existing = set(client.list_collections())
    index = {
        "source": "milvus",
        "server_version": str(version),
        "total_chunks": 0,
        "collections": {},
    }

    for s in DOC_SOURCES:
        if not s.get("enabled", True):
            continue
        coll = s["collection"]
        if coll not in existing:
            print(f"  [跳过] 集合不存在: {coll}")
            continue

        # 只导出**可被检索到**的块（category == 'content'），
        # 因为 Golden Dataset 标注的是「检索应该命中什么」，
        # 而非 content 的块（frontmatter / 纯标题）本来就不会被召回。
        rows = client.query(
            collection_name=coll,
            filter="category == 'content'",
            output_fields=["text", "filename", "title", "category"],
            limit=QUERY_LIMIT,
        )

        chunks = []
        for r in rows:
            text = r.get("text") or ""
            chunks.append(
                {
                    "chunk_id": chunk_id(text),
                    "filename": r.get("filename") or "",
                    "title": (r.get("title") or "")[:80],
                    "chars": len(text),
                    "preview": text[:160].replace("\n", " "),
                    # 完整文本仅用于本地核对，体积较大但便于人工标注
                    "text": text,
                }
            )
        chunks.sort(key=lambda c: (c["filename"], c["chunk_id"]))

        index["collections"][coll] = {
            "name": s["name"],
            "parser": s["parser"],
            "count": len(chunks),
            "chunks": chunks,
        }
        index["total_chunks"] += len(chunks)
        print(f"  {coll:<34} {len(chunks):>4} 块  (源 {s['name']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"总计 {index['total_chunks']} 个可检索块")
    print(f"已写入: {OUT}")

    # 重复指纹检查：同内容不同来源会让标注产生歧义
    seen = {}
    dups = []
    for coll, info in index["collections"].items():
        for c in info["chunks"]:
            if c["chunk_id"] in seen:
                dups.append((c["chunk_id"], seen[c["chunk_id"]], coll))
            else:
                seen[c["chunk_id"]] = coll
    if dups:
        print()
        print(f"[!] 发现 {len(dups)} 个跨集合重复的内容指纹（标注时需注意区分来源）:")
        for cid, c1, c2 in dups[:5]:
            print(f"    {cid}  同时出现在 {c1} 与 {c2}")
    else:
        print("无跨集合重复指纹（每个块内容唯一，标注无歧义）")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
