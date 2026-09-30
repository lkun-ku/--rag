"""诊断：同一份 xlsx 的切片，title 是否相同？61.35 与 89.93 是否在**同一张表**里？

如果它们在不同 title 的块里 -> `merge_table_slices` 按 (filename, title) 分组不会合并它们
-> 模型只看到 89.93 那张表就以为「这是全部」。
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402

TARGET = "vector_db_benchmark.xlsx.md"
cli = MilvusClient(uri=MILVUS_URI)

rows = []
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
             "kb_manuals_bge_large_1024"):
    try:
        rs = cli.query(coll, filter='filename == "%s"' % TARGET,
                       output_fields=["filename", "title", "text"], limit=200)
    except Exception:  # noqa: BLE001
        continue
    rows.extend(rs)

print("=" * 100)
print("`%s` 的切片分布（共 %d 块）" % (TARGET, len(rows)))
print("=" * 100)
print("%-3s %-22s %6s %7s %8s %8s  %s"
      % ("#", "title", "行数", "字符", "含61.35", "含89.93", "正文首行"))
print("-" * 100)
for i, r in enumerate(rows, 1):
    t = r.get("text") or ""
    ls = t.splitlines()
    head = next((l for l in ls if l.strip()), "")
    m = re.search(r"第\s*(\d+)\s*-\s*(\d+)\s*行", head)
    rng = m.group(0) if m else ""
    print("%-3d %-22s %6d %7d %8s %8s  %s"
          % (i, str(r.get("title"))[:22], len(ls), len(t),
             "是" if "61.35" in t else "-",
             "是" if "89.93" in t else "-",
             (rng + " " + head[:40])[:50]))

print()
titles = {}
for r in rows:
    titles.setdefault(str(r.get("title")), []).append(r)
print("按 title 分组：")
for t, items in titles.items():
    has6135 = sum(1 for x in items if "61.35" in (x.get("text") or ""))
    print("  %-24s %d 块 | 含 61.35 的块 %d 个" % (t[:24], len(items), has6135))

print()
print("=" * 100)
print("判读")
print("=" * 100)
if len(titles) > 1:
    print("  **同一文件有 %d 个不同的 title** -> `merge_table_slices` 按 (filename,title) 分组")
    print("  不会跨 title 合并 -> 含 61.35 的表与含 89.93 的表被当成两张互不相关的表。")
else:
    print("  所有切片同 title -> 理论上应合并；若仍未合并，检查是否被 max_chars=20000 拒绝。")
