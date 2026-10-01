"""统计：本项目能处理哪些数据类型、边界在哪。**全部来自代码事实**。"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

print("=" * 100)
print("1. DOC_SOURCES 声明的格式（唯一权威配置）")
print("=" * 100)
from utils.env_utils import DOC_SOURCES  # noqa: E402

print("  %-12s %-10s %-34s %-9s %s"
      % ("源名", "格式", "集合", "parser", "启用"))
for s in DOC_SOURCES:
    print("  %-12s %-10s %-34s %-9s %s"
          % (s["name"], ",".join(s["formats"]), s["collection"],
             s["parser"], "是" if s.get("enabled", True) else "**否**"))

print()
print("=" * 100)
print("2. write_milvus 的 parser 分派（决定「能不能入库」）")
print("=" * 100)
ls = (ROOT / "RAG_PROJECT/documents/write_milvus.py").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(ls, 1):
    st = l.strip()
    if "parser_type ==" in st or "NotImplementedError" in st or "raise ValueError" in st:
        print("  %4d| %s" % (i, st[:94]))

print()
print("=" * 100)
print("3. 各 parser 支持的扩展名（从源码里找）")
print("=" * 100)
for f in ("markdown_parser", "pdf_parser", "office_parser", "pptx_parser"):
    p = ROOT / ("RAG_PROJECT/documents/%s.py" % f)
    if not p.exists():
        print("  [缺失] %s" % f)
        continue
    t = p.read_text(encoding="utf-8")
    exts = sorted(set(re.findall(r"['\"]\.(md|pdf|docx|xlsx|pptx|doc|xls|txt|csv|html?)['\"]", t)))
    cls = re.findall(r"^class (\w+)", t, re.M)
    print("  %-18s 类=%-28s 扩展名=%s" % (f, ",".join(cls) or "-", exts or "-"))

print()
print("=" * 100)
print("4. 硬限制（写进源码的约束）")
print("=" * 100)
mv = (ROOT / "RAG_PROJECT/documents/milvus_db.py").read_text(encoding="utf-8")
for i, l in enumerate(mv.splitlines(), 1):
    if "max_length" in l or "VARCHAR" in l:
        print("  milvus_db.py:%-4d %s" % (i, l.strip()[:88]))
mp = (ROOT / "RAG_PROJECT/documents/markdown_parser.py").read_text(encoding="utf-8")
for i, l in enumerate(mp.splitlines(), 1):
    if "5000" in l or "table_max_rows" in l and "=" in l:
        print("  markdown_parser.py:%-4d %s" % (i, l.strip()[:88]))
pp = (ROOT / "RAG_PROJECT/documents/pdf_parser.py").read_text(encoding="utf-8")
for i, l in enumerate(pp.splitlines(), 1):
    if "ocr" in l.lower() and ("use_ocr" in l or "ocr_fallback" in l):
        print("  pdf_parser.py:%-4d %s" % (i, l.strip()[:88]))

print()
print("=" * 100)
print("5. 实际语料（当前库里的真实分布）")
print("=" * 100)
from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402

cli = MilvusClient(uri=MILVUS_URI)
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
             "kb_manuals_bge_large_1024"):
    try:
        rows = cli.query(coll, filter="", output_fields=["filename"], limit=8000)
    except Exception:  # noqa: BLE001
        continue
    from collections import Counter
    c = Counter()
    for r in rows:
        fn = str(r.get("filename") or "")
        c[pathlib.Path(fn).suffix.lower() or "(无后缀)"] += 1
    print("  %-32s %4d 块  %s" % (coll, len(rows), dict(c)))
