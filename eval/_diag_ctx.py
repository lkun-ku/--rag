"""诊断：聚合类问题实际进入生成的 8 块，分别来自哪个文件、什么 title、含什么关键值。"""

from __future__ import annotations

import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.graph2.graph_2 import run_with_trace  # noqa: E402

Q = "参与评测的产品里，召回率最低的是哪个？召回率是多少？"
out = run_with_trace(Q, save=False)
docs = out.get("documents") or []

print("=" * 108)
print("聚合类问题：进入生成的 %d 块" % len(docs))
print("=" * 108)
print("%-3s %-34s %-16s %5s %7s %7s %s" % ("#", "filename", "title", "行数", "含61.35", "含89.93", "正文前 46 字"))
print("-" * 108)
for i, d in enumerate(docs, 1):
    md = getattr(d, "metadata", None) or {}
    t = d.page_content or ""
    body = " ".join(t.split())[:46]
    print(
        "%-3d %-34s %-16s %5d %7s %7s %s"
        % (
            i,
            str(md.get("filename"))[:34],
            str(md.get("title"))[:16],
            len(t.splitlines()),
            "是" if "61.35" in t else "-",
            "是" if "89.93" in t else "-",
            body,
        )
    )

print()
has = [i for i, d in enumerate(docs, 1) if "61.35" in (d.page_content or "")]
no = [i for i, d in enumerate(docs, 1) if "89.93" in (d.page_content or "")]
print("  含 61.35 的块: %s" % (has or "无"))
print("  含 89.93 的块: %s" % (no or "无"))
print()
print("  答案: %s" % (out.get("generation") or "")[:120])
