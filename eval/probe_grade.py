"""诊断 grade_documents 的过滤行为（3.1 的第二个实际收益）。

起因：跑 run_with_trace 时发现「什么是归一化？」检索到 18 块，
但 grade_documents 只留下 **1** 块，且耗时 10.83s（占单次总耗时 68%）。

要区分两种可能：
  A. 判分过严  —— 块其实相关却被判 no（prompt/LLM 行为问题）
  B. 检索确实差 —— 18 块里真的只有 1 块相关（precision 问题，reranker 能救）

判据：逐块打印判分结果 + 块内容 + 是否为 golden 目标块。
若目标块被判 no -> A；若目标块本就不在 18 块里 -> B。

同时统计「过滤后剩几块」以评估「过滤过严导致拒答」的风险。
"""

import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from graph2.grader_chain import retrieval_grader_chain  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402

E = ROOT / "eval"
cases = {
    c["id"]: c
    for c in (json.loads(l) for l in (E / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
}


def fp(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


# 选 3 题：两道曾被拒答的 + 一道正常题
TARGETS = ["q017", "q042", "q001"]

print("=" * 100)
print("grade_documents 逐块判分诊断")
print("=" * 100)

for qid in TARGETS:
    c = cases.get(qid)
    if not c:
        continue
    q = c["q"]
    gid = set(c["source_blocks"])
    docs = retriever.invoke(q)
    print()
    print("=" * 100)
    print("[%s] %s" % (qid, q))
    print("  目标块: %s | 检索到 %d 块" % (sorted(gid), len(docs)))
    print("-" * 100)
    kept, dropped, target_dropped = 0, 0, False
    for i, d in enumerate(docs, 1):
        cid = fp(d.page_content)
        is_target = cid in gid
        try:
            sc = retrieval_grader_chain.invoke({"question": q, "document": d.page_content})
            grade = str(getattr(sc, "binary_score", "")).strip().lower()
        except Exception as e:  # noqa: BLE001
            grade = "ERR:%s" % type(e).__name__
        if grade == "yes":
            kept += 1
        else:
            dropped += 1
            if is_target:
                target_dropped = True
        print(
            "  %2d. [%-3s] %-10s len=%-4d %s"
            % (i, grade, "<== 目标块" if is_target else "", len(d.page_content), d.page_content[:60].replace("\n", " "))
        )
    print("-" * 100)
    print(
        "  保留 %d / 丢弃 %d  |  目标块被判 no: %s" % (kept, dropped, "**是（判分过严）**" if target_dropped else "否")
    )
