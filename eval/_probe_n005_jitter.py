"""定位 n005 的抖动来源：**判分层** 还是 **生成层**？

## 已知事实

n005 问：「参与评测的产品里，召回率最低的是哪个？召回率是多少？」

* **4.20.3**：同一问题 4 轮里**作答 1 次、拒答 3 次**
* **4.27 / 4.34.2**：全链路审计与 8 并发压测里**又拒答**（8 并发时它 7.2s 返回，
  是 8 个请求里最快的 —— 说明**早早走了拒答分支**）
* **判分后保留 9 块** —— **块是充足的**，所以不是「判分把上下文清空」

## 两种可能（决策树）

    A. **判分抖**：某些轮次判分把**含答案的块**筛掉了
       -> 上下文里根本没有答案 -> 只能拒答
       -> 修法：收紧判分 / 提高 per_k / 判分加白名单

    B. **生成抖**：含答案的块**在上下文里**，但模型这一轮仍说「未找到依据」
       -> 修法：看 prompt 与块的组织方式（例如块太多、答案淹没）

**本探针逐轮记录这两件事，用数据判定是哪一种。**

## 判读标准

* 「golden 不在保留块里」占多数 -> 判分问题（A）
* 「golden 在但拒答」占多数    -> 生成问题（B）
* 两者都出现                  -> 双重抖动（要分别处理）
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.citation import is_refusal  # noqa: E402

ROUNDS = int(os.environ.get("N005_ROUNDS", "10"))
QID = os.environ.get("N005_QID", "n005")

cases = {
    json.loads(l).get("id"): json.loads(l)
    for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}
C = cases[QID]
Q = C["q"]
GOLD = set(C.get("source_blocks") or [])

print("=" * 104)
print("n005 抖动定位（%d 轮）" % ROUNDS)
print("=" * 104)
print("  问题 : %s" % Q)
print("  golden: %d 块" % len(GOLD))
print()

from graph2.graph_2 import run_with_trace  # noqa: E402

rows = []
for i in range(1, ROUNDS + 1):
    try:
        out = run_with_trace(Q)
    except Exception as e:  # noqa: BLE001
        print("  [%2d] **异常** %s: %s" % (i, type(e).__name__, str(e)[:70]))
        rows.append({"i": i, "err": "%s: %s" % (type(e).__name__, str(e)[:70])})
        continue

    docs = out.get("documents") or []
    if not isinstance(docs, list):
        docs = [docs]
    ids = {metrics.chunk_id(d.page_content) for d in docs}
    g_kept = len(GOLD & ids)  # **关键**：含答案的块有没有进上下文
    gen = (out.get("generation") or "").strip()
    refused = is_refusal(gen)
    r = {"i": i, "n_ctx": len(docs), "g_kept": g_kept, "refused": refused, "gen": gen[:150].replace("\n", " ")}
    rows.append(r)
    print(
        "  [%2d] 上下文 %2d 块 | **golden 保留 %d/%d** | %-8s | %s"
        % (i, len(docs), g_kept, len(GOLD), "**拒答**" if refused else "作答", gen[:78])
    )

ok = [r for r in rows if "err" not in r]
n_ref = sum(1 for r in ok if r["refused"])
n_ref_no_gold = sum(1 for r in ok if r["refused"] and r["g_kept"] == 0)
n_ref_has_gold = sum(1 for r in ok if r["refused"] and r["g_kept"] > 0)
n_ans = sum(1 for r in ok if not r["refused"])

print()
print("=" * 104)
print("判读（%d 轮有效）" % len(ok))
print("=" * 104)
print("  作答 %d 轮 | 拒答 %d 轮（%.0f%%）" % (n_ans, n_ref, 100 * n_ref / max(len(ok), 1)))
print()
print("  拒答中：")
print("    · **含答案的块不在上下文里**（g_kept=0）: %d 轮  -> 指向 **(A) 判分抖**" % n_ref_no_gold)
print("    · **含答案的块在上下文里**（g_kept>0）: %d 轮  -> 指向 **(B) 生成抖**" % n_ref_has_gold)
print()
if n_ref == 0:
    print("  本次 %d 轮**全部作答** —— 未复现拒答。" % len(ok))
    print("  说明该题的不稳定可能是**偶发**（或与当时的服务端状态相关），")
    print("  而非稳定的逻辑缺陷。建议：加大轮数（N005_ROUNDS=30）再观察。")
elif n_ref_no_gold > n_ref_has_gold:
    print("  -> **主要是判分抖**：含答案的块被筛掉了。")
    print("     修法方向：提高 per_k / 放宽判分，或对含答案特征（表格行）加保护。")
    print("     **但注意**：判分总体 golden 保留率是 98%（4.20.5），若这里频繁丢，")
    print("     说明**这道题的块在判分器眼里特别「不像答案」**（如纯数字表格行）。")
elif n_ref_has_gold > n_ref_no_gold:
    print("  -> **主要是生成抖**：答案在上下文里，但模型仍说未找到。")
    print("     修法方向：看 prompt（是否块太多导致淹没）、看块顺序（答案块排第几）、")
    print("     看该题的答案形态（是否是「最低/最差」这类**需要跨块比较**的问题）。")
else:
    print("  -> **两种都出现**，需要分别处理（判分层 + 生成层各看一次）。")

# 作答与拒答时的上下文规模对比（容易看出的差异）
if ok:
    a = [r["n_ctx"] for r in ok if not r["refused"]]
    b = [r["n_ctx"] for r in ok if r["refused"]]
    ga = [r["g_kept"] for r in ok if not r["refused"]]
    gb = [r["g_kept"] for r in ok if r["refused"]]
    print()
    print("  上下文规模：作答轮 %s 块 | 拒答轮 %s 块" % (a or "-", b or "-"))
    print("  golden 保留：作答轮 %s | 拒答轮 %s" % (ga or "-", gb or "-"))

(HERE / "probe_n005_jitter.json").write_text(
    json.dumps(
        {
            "qid": QID,
            "question": Q,
            "rounds": ROUNDS,
            "rows": rows,
            "n_ans": n_ans,
            "n_ref": n_ref,
            "ref_no_gold": n_ref_no_gold,
            "ref_has_gold": n_ref_has_gold,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print()
print("  证据 -> eval/probe_n005_jitter.json")
