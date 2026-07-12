"""引用准确性抽样核对（ROADMAP 3.0.2）。

## 为什么需要它

`faithfulness = 0.9020` 只回答「答案是否忠于检索内容」，**不回答**：
  答案里标注的 `[N]` 是否**真的指向支持该结论的那个块**。

即：可能存在「引用格式正确、编号合法（没过 validate_citations 的越界检查），
但指错了块」的隐性错误 —— 这类错误让用户按图索骥却找不到依据。

## 怎么判定

对每题输出：
  1. 问题
  2. 答案（含 [N] 标注）
  3. **每个被引用块的原文摘要**（按编号对应到实际 Document）

然后人工核对「块内容是否支持答案里引用它的那句话」。
这属于**人工判据**，脚本只负责把证据并排呈现在一起（避免人工在多个文件间来回跳）。

## 数据来源

直接复用 `eval/refusal_audit.json` 里已生成的答案 —— 那份报告是走
`graph2/generate_node2.generate`（真实生成链）产出的，**不需要重新调 LLM**。

用法：
    .venv\\Scripts\\python.exe eval\\citation_audit.py                 # 全部 57 题
    .venv\\Scripts\\python.exe eval\\citation_audit.py --sample 20     # 均匀抽样 20 题
    .venv\\Scripts\\python.exe eval\\citation_audit.py --only-with-cites
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

REPORT = HERE / "refusal_audit.json"
OUT = HERE / "citation_audit.txt"
CITE_RE = re.compile(r"\[(\d+)\]")


def main() -> int:
    ap = argparse.ArgumentParser(description="引用准确性抽样核对")
    ap.add_argument("--sample", type=int, default=0, help="均匀抽样 N 题（0=全部）")
    ap.add_argument("--only-with-cites", action="store_true", help="只输出含引用的题")
    ap.add_argument("--block-chars", type=int, default=190, help="每个引用块展示的字符数")
    args = ap.parse_args()

    if not REPORT.exists():
        print("缺少 %s —— 请先运行 eval/refusal_audit.py" % REPORT.name)
        return 2

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = report["in_scope"]["detail"]

    from tools.retriever_tools import retriever  # noqa: E402

    if args.sample and args.sample < len(rows):
        step = len(rows) / args.sample
        rows = [rows[int(i * step)] for i in range(args.sample)]

    lines = []
    lines.append("=" * 100)
    lines.append("引用准确性核对（ROADMAP 3.0.2）")
    lines.append("=" * 100)
    lines.append("  判定方法：核对「答案里引用 [N] 的那句话」是否被「第 N 个块的原文」支持")
    lines.append("  数据来源：eval/refusal_audit.json（真实生成链产出，未重新调用 LLM）")
    lines.append("")

    stats = {"total": 0, "refused": 0, "with_cite": 0, "no_cite": 0, "invalid": 0}
    detail = []

    for row in rows:
        q = row["q"]
        ans = row.get("answer", "")
        stats["total"] += 1
        if row.get("refused"):
            stats["refused"] += 1
            if args.only_with_cites:
                continue

        cites = sorted({int(x) for x in CITE_RE.findall(ans)})
        if not cites:
            stats["no_cite"] += 1
            if not args.only_with_cites and row.get("refused"):
                continue
        else:
            stats["with_cite"] += 1

        # 重新检索拿到「编号 -> 块」的对应（numbering 由 format_context 生成，从 1 开始）
        try:
            docs = retriever.invoke(q)
        except Exception as e:  # noqa: BLE001
            lines.append("  [%s] 检索失败: %s" % (row.get("id"), type(e).__name__))
            continue

        lines.append("-" * 100)
        lines.append("[%s] %s" % (row.get("id"), q))
        lines.append("")
        lines.append("  答案: %s" % ans.replace("\n", " ").strip()[:400])
        lines.append("")

        n = len(docs)
        entry = {"id": row.get("id"), "q": q, "cites": cites, "n_docs": n,
                 "answer": ans[:400], "blocks": []}
        for c in cites:
            if c < 1 or c > n:
                stats["invalid"] += 1
                lines.append("    [%d] -> **越界（上下文只有 %d 块）**" % (c, n))
                entry["blocks"].append({"n": c, "error": "out_of_range"})
                continue
            d = docs[c - 1]
            meta = d.metadata
            body = d.page_content.replace("\n", " ").strip()
            lines.append("    [%d] %s | %s" % (
                c, meta.get("filename", "?"), str(meta.get("title", ""))[:34]))
            lines.append("        %s" % body[:args.block_chars])
            entry["blocks"].append({
                "n": c, "filename": meta.get("filename", ""),
                "title": str(meta.get("title", ""))[:60], "text": body[:600],
            })
        lines.append("")
        detail.append(entry)

    lines.append("=" * 100)
    lines.append("统计")
    lines.append("=" * 100)
    lines.append("  题目数        : %d" % stats["total"])
    lines.append("  其中拒答      : %d" % stats["refused"])
    lines.append("  含引用的答案  : %d" % stats["with_cite"])
    lines.append("  不含引用的答案: %d" % stats["no_cite"])
    lines.append("  **编号越界次数**: %d  （应为 0 —— generate 节点已做校验）" % stats["invalid"])
    lines.append("")

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    # 同时写一份结构化数据，便于后续比对
    (HERE / "citation_audit.json").write_text(
        json.dumps({"stats": stats, "detail": detail}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(text[:4000])
    print("...")
    print("完整结果已写入 %s（%d 行）与 %s" % (OUT.name, len(lines), "citation_audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
