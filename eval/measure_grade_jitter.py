"""B：**判分抖动实测** —— 同一「问题 + 块」组合重复判分，量出判定翻转率。

## 为什么要测（来自代码里已记录的已知问题）

`graph2/grade_documents_node.py` 的 docstring 写着：

    实测同一批 18 块、同一问题，串行与并发两次判分结果完全一致（0 异常），
    但**不同时刻**的判定会变化（同一问题曾得到「保留 3 块」与「保留 0 块」两种结果）
    —— 说明判分本身存在服务端不确定性

**这条从未被量化**。它与 4.19 观察到的「保留率 20.2%」叠加后风险很大：
如果单块判定的翻转率有 10%，那 24 块里就会有若干块在各次运行间反复进出上下文 ——
表现为「同一问题、同样的检索结果，这次答得出、下次答不出」。

## 口径（不猜，直接重复测）

对选定的题，取它的**检索结果**（`final_k=24` 的一部分），
把每个「问题 + 块」组合**重复判分 ROUNDS 次**，然后统计：

    * **翻转块**：ROUNDS 次里既出现过 yes 又出现过 no 的块；
    * **翻转率** = 翻转块数 / 总块数；
    * 逐块打印 yes/no 序列，便于看是「偶发翻转」还是「持续摇摆」。

同时记录**每一步的原始返回**（避免把「结构化输出解析失败」误当成「判定为 no」——
那会让抖动被高估）。

## 样本选择

取 4 道有 golden 的题（覆盖不同来源），每题取**前 8 块**（保持总调用量可控：
4 题 × 8 块 × 5 轮 = 160 次判分）。**同时标注哪些是 golden 块** ——
若翻转集中在 golden 块上，后果更严重（正确块时有时无）。
"""

from __future__ import annotations

import json
import os
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from rag_kb.tools.retriever_tools import retriever  # noqa: E402
from rag_kb.graph2.grader_chain import retrieval_grader_chain  # noqa: E402
from rag_kb.utils.env_utils import LLM_TEMPERATURE, LLM_MODEL  # noqa: E402

ROUNDS = int(os.environ.get("JITTER_ROUNDS", "5"))
PER_Q = int(os.environ.get("JITTER_PER_Q", "8"))
QIDS = os.environ.get("JITTER_QIDS", "n001,n005,n008,n022").split(",")

cases = {
    json.loads(l).get("id"): json.loads(l)
    for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}

print("=" * 104)
print("B：判分抖动实测 —— 同一「问题 + 块」重复判分 %d 轮" % ROUNDS)
print("=" * 104)
print("  LLM: %s  temperature=%s" % (LLM_MODEL, LLM_TEMPERATURE))
print(
    "  样本: %d 道题 × 每题前 %d 块 × %d 轮 = %d 次判分"
    % (len(QIDS), PER_Q, ROUNDS, len(QIDS) * PER_Q * ROUNDS if False else 0)
)
print()

# 先取好样本（检索一次，避免重复检索引入差异）
samples = []  # (qid, q, idx, chunk_id, text)
for qid in QIDS:
    c = cases.get(qid)
    if not c:
        print("  [!] %s 不在 golden 里" % qid)
        continue
    docs = retriever.invoke(c["q"])
    for i, d in enumerate(docs[:PER_Q], 1):
        samples.append(
            (qid, c["q"], i, metrics.chunk_id(d.page_content), d.page_content, set(c.get("source_blocks") or []))
        )

print("  实际样本数: %d" % len(samples))
print()

results = []  # {…, seq: ['yes','no',…], raw: [...]}
t0 = time.time()
for si, (qid, q, idx, cid, text, gold) in enumerate(samples, 1):
    seq, raws = [], []
    for r in range(ROUNDS):
        try:
            sc = retrieval_grader_chain.invoke({"question": q, "document": text})
            v = str(getattr(sc, "binary_score", "")).strip().lower()
            raw = getattr(sc, "binary_score", None)
        except Exception as e:  # noqa: BLE001
            v = "error"
            raw = "%s: %s" % (type(e).__name__, str(e)[:60])
        seq.append(v)
        raws.append(raw)
    results.append(
        {
            "qid": qid,
            "q": q,
            "idx": idx,
            "chunk_id": cid,
            "is_golden": cid in gold,
            "seq": seq,
            "raw": raws,
            "head": text.replace("\n", " ")[:56],
        }
    )
    print(
        "  [%2d/%d] %s #%-2d %s  序列=%s"
        % (si, len(samples), qid, idx, "<GOLD>" if cid in gold else "      ", ",".join(seq))
    )

print()
print("  判分耗时 %.1fs" % (time.time() - t0))

# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
print()
print("=" * 104)
print("统计")
print("=" * 104)


def summarize(rows, label):
    n = len(rows)
    if not n:
        return
    flip = [r for r in rows if len({x for x in r["seq"] if x in ("yes", "no")}) > 1]
    errs = [r for r in rows if "error" in r["seq"]]
    allno = [r for r in rows if all(x == "no" for x in r["seq"])]
    allyes = [r for r in rows if all(x == "yes" for x in r["seq"])]
    print(
        "  %-14s 块数 %3d | **翻转 %2d（%.0f%%）** | 恒 no %2d | 恒 yes %2d | 含异常 %d"
        % (label, n, len(flip), 100 * len(flip) / n, len(allno), len(allyes), len(errs))
    )
    return flip


allflip = summarize(results, "全部")
g = [r for r in results if r["is_golden"]]
ng = [r for r in results if not r["is_golden"]]
gflip = summarize(g, "其中 golden 块")
summarize(ng, "非 golden 块")

print()
if allflip:
    print("  **翻转块的明细**（这类块在各次运行间反复进出上下文）:")
    print("  %-6s %-4s %-8s %-8s %s" % ("题", "序", "golden", "序列", "块首行"))
    print("  " + "-" * 96)
    for r in allflip:
        print(
            "  %-6s #%-3d %-8s %-8s %s"
            % (r["qid"], r["idx"], "GOLD" if r["is_golden"] else "", ",".join(r["seq"]), r["head"][:44])
        )
    print()
    print("  原始返回值（确认不是「解析失败被当成 no」）:")
    for r in allflip[:6]:
        print("     [%s #%d] raw=%s" % (r["qid"], r["idx"], r["raw"]))
else:
    print("  **没有翻转块** —— 本批样本上判分是稳定的。")

(HERE / "grade_jitter.json").write_text(
    json.dumps(
        {"rounds": ROUNDS, "llm": LLM_MODEL, "temperature": LLM_TEMPERATURE, "results": results},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/grade_jitter.json")
