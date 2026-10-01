"""端到端验证：确认改动没有破坏完整工作流。

关注点：
  1. graph_2 的 grade_generation 步骤里出现新增的 `quoted_cites` 字段
  2. 链路能正常走完（不因 citation.py 的返回结构变化而 KeyError）
  3. 问题**取自当前语料的评测集**，不用硬编码

> **[教训 ] 为什么问题必须随语料走**
>
> 本脚本最初硬编码了「LayoutParser 是什么？」（沿用上一轮的验证问题）。
> 但语料已更换 —— 新的 `papers` 集合是 Milvus / HNSW / RAG 论文，
> **LayoutParser 的文档已不在语料里**。于是：
>
>     检索 18 块（全是 Milvus 系文档）-> 判分 0/18 保留 -> 走 transform_query 两轮
>     -> 最后回退 web_search 答对
>
> 我当时把这个**正常的兜底行为**误判成「判分链系统性误杀」，还写进了 ROADMAP。
> 追查后才确认判分完全正确 —— 库里确实没有这个答案。
>
> 所以现在改为从 `eval/golden_dataset.jsonl` 读题：**评测集是随语料重建的**，
> 用它的题做端到端测试，才能保证「问的问题在当前语料里真的有答案」。
"""

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.graph2.graph_2 import run_with_trace  # noqa: E402

# 取评测集里**有 golden 块**的前两题（库外题不算端到端问答的典型路径）
_cases = [
    json.loads(l)
    for l in (ROOT / "eval" / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
_cases = [c for c in _cases if c.get("source_blocks")]
QUESTIONS = [c["q"] for c in _cases[:2]]
print("取自 eval/golden_dataset.jsonl 的题目: %s" % [c["id"] for c in _cases[:2]])

for q in QUESTIONS:
    print()
    print("#" * 100)
    print("问题: %s" % q)
    print("#" * 100)
    r = run_with_trace(q, save=False)
    t = r["trace"]
    print(
        "  总耗时 %.2fs | 步骤 %d | tokens in=%s out=%s"
        % (t["total_sec"], t["n_steps"], t["tokens"].get("input_tokens"), t["tokens"].get("output_tokens"))
    )
    for s in t["steps"]:
        extra = {k: v for k, v in s.items() if k not in ("node", "t", "elapsed_sec")}
        print("    %6.2fs  %-18s %6.2fs  %s" % (s["t"], s["node"], s["elapsed_sec"], str(extra)[:88]))
    print()
    print("  答案: %s" % r["generation"][:220].replace(chr(10), " "))
    print()
    print("  documents: %d 块" % len(r.get("documents") or []))
