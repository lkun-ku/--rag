"""量化「引用支持度」—— 回答 ROADMAP 4.5 的前置问题 2。

## 为什么需要

`citation_audit.py` 只统计「编号越界」（=0）。但复核发现 q006 的答案标了 [4]，
而 [4] 是「表面能改性」（讲等离子体），与「配置文件为何不生效」毫无关系 ——
**编号合法但指错了块**。这类错误靠字符串规则查不出来，必须做语义判断。

## 判据（LLM-as-judge）

对每题：把「答案」与「被引用的块原文」一起给 LLM，问它
「被引用的块是否支持答案中的陈述」。返回 yes/no。

    support     引用确实支持答案（正确溯源）
    not_support 引用不支持答案（指错块 —— P0-1 的目标）

数据源：eval/citation_audit.json（已含每题答案与被引块文本，无需重跑检索与生成）。
结果写 eval/citation_support.json。
"""

import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag_kb.llm_models.all_llm import llm  # noqa: E402

E = ROOT / "eval"
audit = json.loads((E / "citation_audit.json").read_text(encoding="utf-8"))
detail = audit["detail"]

PROMPT = """你是一个严格的引用核查员。

给定「答案」与「被引用的资料块」，判断：**资料块是否真的支持答案中的陈述**？

判定标准：
- 若资料块包含答案所需的关键信息（数据、结论、定义、步骤等），判 yes。
- 若资料块讲的是完全不同的主题，或虽有相同关键词但无法支撑该结论，判 no。
- 只判「支持性」，不要去评价答案本身是否正确。
- 资料块可能很长（表格/文档整段），**请在其全部内容中仔细查找**支持答案的关键信息
  （数字、结论、定义、步骤），只要找到即判 yes。不要因为关键信息不在开头就判 no。
- 只要**任意一个**被引块支持答案，就判 yes。

资料块：
{blocks}

答案：
{answer}

只回复一个词：yes 或 no。"""

results = []
t0 = time.time()
n_yes = n_no = 0

print("=" * 100)
print("引用支持度审计（%d 题，每题 1 次 LLM 判断）" % len(detail))
print("=" * 100)

for i, it in enumerate(detail, 1):
    qid = it["id"]
    answer = it.get("answer", "")
    blocks = it.get("blocks") or []
    if not blocks:
        continue
    # [修正 ] 原先截到 1200 字符，导致长块里的关键信息（如表格第 31-50 行块）
    # 根本没进判定上下文，LLM 只能判「不支持」-> 大量误报（实测 q033/q042/q046 等
    # 引用明显正确却被判指错）。这里放宽到 6000 字符。
    ctx = "\n\n".join(
        "[%s] %s | %s\n%s"
        % (b.get("n"), b.get("filename", ""), str(b.get("title", ""))[:60], str(b.get("text", ""))[:6000])
        for b in blocks
    )
    try:
        resp = llm.invoke(PROMPT.format(blocks=ctx, answer=answer[:1500]))
        raw = str(getattr(resp, "content", resp)).strip().lower()
        verdict = "yes" if raw.startswith("yes") else ("no" if raw.startswith("no") else "?")
    except Exception as e:  # noqa: BLE001
        verdict = "ERR:%s" % type(e).__name__

    ok = verdict == "yes"
    n_yes += 1 if ok else 0
    n_no += 0 if ok else 1
    results.append(
        {
            "id": qid,
            "q": it.get("q"),
            "cites": it.get("cites"),
            "verdict": verdict,
            "answer": answer[:200],
            "cited_files": [b.get("filename") for b in blocks],
        }
    )
    flag = "OK " if ok else ("ERR" if verdict.startswith("ERR") else "**指错**")
    print("  %2d/%d [%s] %-6s cites=%-10s %s" % (i, len(detail), flag, qid, it.get("cites"), str(it.get("q"))[:44]))
    if not ok and not verdict.startswith("ERR"):
        print("        答案: %s" % answer[:110].replace("\n", " "))
        for b in blocks:
            print(
                "        引到: [%s] %s | %s"
                % (b.get("n"), b.get("filename"), str(b.get("text", ""))[:80].replace("\n", " "))
            )

print()
print("=" * 100)
print("汇总")
print("=" * 100)
tot = len(results)
print("  题数            : %d" % tot)
print("  引用支持答案    : %d" % n_yes)
print("  **引用指错块**  : %d  <- P0-1 的规模" % n_no)
if tot:
    print("  支持率          : %.1f%%" % (n_yes / tot * 100))
print("  耗时            : %.1fs" % (time.time() - t0))

(E / "citation_support.json").write_text(
    json.dumps(
        {
            "n": tot,
            "support": n_yes,
            "not_support": n_no,
            "support_rate": round(n_yes / tot, 4) if tot else None,
            "detail": results,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print("  结果 -> eval/citation_support.json")
