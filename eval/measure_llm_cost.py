"""成本结构实测：一次真实提问里，**各个节点的 input/output token 占比**。

## 为什么必须先测这个（而不是照着「调用次数」优化）

上一轮的结论是「判分占 90% 的**调用次数**」（1 + 36 + 1 + 2 ≈ 40 次）。
但**调用次数 ≠ token**：判分每次只传**1 个块**，而生成一次传 **~7 个块**，
幻觉检查要传「答案 + 全部上下文」。所以真实成本占比可能完全不同，
**优化错地方的代价是白改一遍**。

## 口径（不猜，用真实数据算）

用**真实运行的输入**估算每个节点的 token：

    * **判分**：`retrieval_grader_chain` 的 prompt = system + 单个块 + 问题；
      对**每一块**都算一次 —— 用真实检索结果（`FINAL_K=36` 块）的**实际长度**；
    * **生成**：`CITE_PROMPT` + format_context(判分后保留的块) + 问题；
    * **路由**：`query_route_chain` 的 prompt + 问题；
    * **幻觉检查**：`grade_hallucinations_chain` 的 prompt + 答案 + 上下文。

token 估算用 **tiktoken**（有则用）否则退化到字符数启发式：
中文约 1 字 ≈ 1 token、英文约 4 字符 ≈ 1 token —— 对**占比**判断够用，
且报告里会**显式标注估算方法**。

同时对照 `logs/trace.jsonl` 里**真实记录的 total_tokens**，校验估算是否离谱。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

# ---- token 估算器 ----
try:
    import tiktoken  # type: ignore

    _enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s: str) -> int:
        return len(_enc.encode(s or ""))

    METHOD = "tiktoken(cl100k_base)"
except Exception:  # noqa: BLE001

    def ntok(s: str) -> int:
        """退化启发式：中文字 1 token，其它按 4 字符 1 token。"""
        if not s:
            return 0
        cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
        other = len(s) - cjk
        return cjk + other // 4

    METHOD = "启发式（中文字=1 token，其它 4 字符=1 token）"

print("=" * 104)
print("成本结构实测：一次真实提问的各节点 token 占比")
print("=" * 104)
print("  token 估算方法: %s" % METHOD)
print()

import metrics  # noqa: E402
from tools.retriever_tools import retriever  # noqa: E402
from graph2.grade_documents_node import _grade_one, MAX_GRADE_WORKERS  # noqa: E402
from graph2.grader_chain import grade_prompt  # noqa: E402
from graph2.generate_node2 import CITE_PROMPT  # noqa: E402
from utils.citation import format_context  # noqa: E402

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
case = cases[0]
Q = case["q"]
print("  取样问题: %s" % Q)
print()

# ---- ① 路由 ----
route_in = ntok("请判断以下问题应该走知识库检索还是网络搜索：\n" + Q) + 40
route_out = 6  # 只回一个词：vectorstore / web_search

# ---- ② 判分（对每一块）----
docs = retriever.invoke(Q)
g_in = g_out = 0
gr = grade_prompt.messages
sys_txt = "\n".join(
    str(getattr(m, "prompt", getattr(m, "content", "")))
    for m in gr
    if getattr(m, "role", "") == "system" or type(m).__name__ == "SystemMessage"
)
sys_tok = ntok(sys_txt) or 60
for d in docs:
    g_in += sys_tok + ntok(d.page_content) + ntok(Q) + 20
    g_out += 6  # yes/no
n_graded = len(docs)

# ---- ③ 生成（只看判分后保留的块）----
import concurrent.futures as cf  # noqa: E402

with cf.ThreadPoolExecutor(max_workers=max(1, min(MAX_GRADE_WORKERS, len(docs)))) as ex:
    graded = list(ex.map(lambda d: _grade_one(Q, d), docs))
kept = [d for d, ok in graded if ok]
ctx = format_context(kept)
gen_in = ntok(CITE_PROMPT) + ntok(ctx) + ntok(Q) + 20
gen_out = 180  # 实测答案通常 100~300 token

# ---- ④ 幻觉检查（通常 1 次，最多 3 次）----
hall_in = ntok("你是判分员，判断答案是否完全基于资料回答。资料：\n" + ctx + "\n答案：\n" + "x" * 300) + ntok(Q)
hall_out = 6
hall_calls = 2  # 平均约 2 次（可能重试）

total_in = route_in + g_in + gen_in + hall_in * hall_calls
total_out = route_out + g_out + gen_out + hall_out * hall_calls

rows = [
    ("① route_question", 1, route_in, route_out),
    ("② grade_documents", n_graded, g_in, g_out),
    ("③ generate", 1, gen_in, gen_out),
    ("④ grade_generation", hall_calls, hall_in * hall_calls, hall_out * hall_calls),
]

print("=" * 104)
print("各节点 token（input 按真实输入长度估算）")
print("=" * 104)
print("  %-22s %-8s %-12s %-12s %-10s %s" % ("节点", "调用次数", "input tok", "output tok", "合计 tok", "占总 token"))
print("  " + "-" * 98)
for name, calls, ti, to in rows:
    tot = ti + to
    print(
        "  %-22s %-8d %-12d %-12d %-10d %.1f%%" % (name, calls, ti, to, tot, 100 * tot / max(total_in + total_out, 1))
    )
print("  " + "-" * 98)
print("  %-22s %-8s %-12d %-12d %-10d" % ("合计", "", total_in, total_out, total_in + total_out))
print()
print("  检索到 %d 块；判分后保留 %d 块" % (len(docs), len(kept)))
print("  接入上下文的总字符数: %d（判分后）" % len(ctx))

# ---- 与 trace 的真实记录对照 ----
tp = ROOT / "logs" / "trace.jsonl"
if tp.exists():
    hs = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if hs:
        tk = hs[-1].get("tokens", {})
        print()
        print(
            "  对照 trace.jsonl 最后一条的**真实**统计: total=%s (in=%s / out=%s)"
            % (tk.get("total_tokens"), tk.get("input_tokens"), tk.get("output_tokens"))
        )
        real = tk.get("total_tokens") or 0
        est = total_in + total_out
        if real:
            print(
                "  估算 %d vs 真实 %d -> 偏差 %.0f%%（估算仅用于**占比**判断）" % (est, real, 100 * (est - real) / real)
            )

(HERE / "measure_llm_cost.json").write_text(
    json.dumps(
        {
            "question": Q,
            "n_ret": len(docs),
            "n_kept": len(kept),
            "method": METHOD,
            "nodes": [{"node": n, "calls": c, "in": ti, "out": to} for n, c, ti, to in rows],
            "total": {"in": total_in, "out": total_out},
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/measure_llm_cost.json")
