"""全链路审计：**用户实际看到的是什么**（对比 `refusal_audit.py` 的简化口径）。

## 与 `refusal_audit.py` 的口径差异（这是本脚本存在的理由）

`refusal_audit.py` 为**控制变量**刻意跳过了判分 —— 它把检索到的**全部块**
直接交给 `generate`，测的是「生成层的拒答行为」：

    docs = retriever.invoke(q)                 # 36 块全给
    gen  = metrics.generate_answer(q, docs)    # 测生成层

**真实链路不是这样**，中间还有两级：

    检索 36 块 -> 判分（保留 ~10 块）-> decide_to_generate
                                            |-- 有块 -> generate
                                            `-- 空 -> transform_query ×2 -> web_search

所以**库外题的真实行为可能根本不是「拒答」** —— 而是
「放弃知识库、上网搜索并回答」（P0-9 的「LayoutParser」就是这条路径）。
用户看到的是**网络答案**，不是「知识库中未找到依据」。

**两者都对，但回答的是不同问题**：

| 审计 | 回答的问题 |
|---|---|
| `refusal_audit.py` | 生成层在「给了全量块」时会不会拒答？（控制变量） |
| **本脚本** | **用户真实遇到的是什么？**（端到端） |

## 记录什么

逐题跑 `run_with_trace`，记录：

    * **路由决策**（vectorstore / web_search）
    * 检索块数 -> **判分后块数**（中间变量）
    * **最终是否拒答**（is_refusal）
    * 是否走了 `web_search`、走了几轮（从前面的 trace 步骤推断）
    * 答案摘要（便于人工核对）

## 注意

这是**测量**脚本，没有代码契约要守，因此不适用 TDD
（TDD 用于「实现新逻辑」，此处只观测既有链路的真实行为）。
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

from rag_kb.utils.citation import is_refusal  # noqa: E402

LIMIT_IN = int(os.environ.get("FC_LIMIT_IN", "0"))  # 0 = 全部库内题
LIMIT_OUT = int(os.environ.get("FC_LIMIT_OUT", "0"))  # 0 = 全部库外题

# 与 refusal_audit.py 用同一批库外题（保证可比）
OUT_OF_SCOPE = [
    "今天北京的天气怎么样？",
    "红烧肉的家常做法是什么？",
    "2026 年世界杯在哪个国家举办？",
    "帮我写一首关于春天的短诗。",
    "Python 的 GIL 是什么，为什么会限制多线程性能？",
    "特斯拉最新一个交易日的收盘价是多少？",
    "推荐几本适合入门的小说。",
    "如何在线预约驾驶证考试？",
    "帮我订一张明天去上海的机票。",
    "介绍一下最近上映的电影。",
]

from rag_kb.graph_crag.crag_graph import run_with_trace  # noqa: E402

cases = [
    json.loads(line)
    for line in (HERE.parents[1] / "eval" / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if LIMIT_IN:
    cases = cases[:LIMIT_IN]
if LIMIT_OUT:
    OUT_OF_SCOPE = OUT_OF_SCOPE[:LIMIT_OUT]


def _as_list(docs):
    """把 `documents` 统一成 list。

    **[实测发现 ]** `web_search` 节点返回的是**单个 `Document`**
    （见 `graph_crag/web_search_node.py`：`web_results = Document(page_content=...)`
    然后 `return {"documents": web_results, ...}`），而**其它节点都返回列表**。
    于是任何假设「documents 是 list」的下游代码都会在**走了 web_search 的路径**上崩：

        TypeError: object of type 'Document' has no len()

    （本脚本首版就是这么崩的。）这个类型不一致**本身值得记录** —— 见 ROADMAP。
    """
    if docs is None:
        return []
    if isinstance(docs, list):
        return docs
    return [docs]


def probe(q: str) -> dict:
    t0 = time.time()
    try:
        out = run_with_trace(q)
    except Exception as e:  # noqa: BLE001
        return {"q": q, "error": "%s: %s" % (type(e).__name__, str(e)[:80])}
    steps = [s.get("node") for s in (out.get("trace", {}).get("steps") or [])]
    dec = ""
    for s in out.get("trace", {}).get("steps") or []:
        if s.get("node") == "route_question":
            dec = s.get("decision", "")
    gen = out.get("generation") or ""
    docs = _as_list(out.get("documents"))
    return {
        "q": q,
        "sec": round(time.time() - t0, 1),
        "route": dec,
        "n_kept": len(docs),
        "docs_is_list": isinstance(out.get("documents"), list),
        "n_web_search": steps.count("web_search"),
        "n_transform": steps.count("transform_query"),
        "n_retrieve": steps.count("retrieve"),
        "refused": is_refusal(gen),
        "answer": gen[:220].replace("\n", " "),
        "steps": " -> ".join(steps),
    }


print("=" * 108)
print("全链路审计：用户实际看到的是什么")
print("=" * 108)
print("  库内 %d 题（golden）| 库外 %d 题" % (len(cases), len(OUT_OF_SCOPE)))
print()

res = {"out": [], "in": []}

print("─" * 108)
print("A. 库外问题 —— 期望：**不是拒答**，而是「放弃知识库、上网搜索并回答」")
print("─" * 108)
for i, q in enumerate(OUT_OF_SCOPE, 1):
    r = probe(q)
    res["out"].append(r)
    if r.get("error"):
        print("  [%2d] %-30s **失败** %s" % (i, q[:30], r["error"]))
        continue
    tag = "拒答" if r["refused"] else ("走网络搜索" if r["n_web_search"] else "库内作答")
    print(
        "  [%2d] %-30s %-10s 路由=%-11s 判分后=%2d 检索%d次 网络搜索%d次 %.1fs"
        % (i, q[:30], tag, r["route"], r["n_kept"], r["n_retrieve"], r["n_web_search"], r["sec"])
    )

print()
print("─" * 108)
print("B. 库内问题 —— 期望：判分后仍有块、能作答、不拒答")
print("─" * 108)
for i, c in enumerate(cases, 1):
    r = probe(c["q"])
    r["id"] = c.get("id")
    res["in"].append(r)
    if r.get("error"):
        print("  [%2d] %-6s **失败** %s" % (i, c.get("id"), r["error"]))
        continue
    tag = "**拒答**" if r["refused"] else "作答"
    print(
        "  [%2d] %-6s %-10s 路由=%-11s 判分后=%2d %.1fs | %s"
        % (i, c.get("id"), tag, r["route"], r["n_kept"], r["sec"], c["q"][:34])
    )

# ---- 汇总 ----
print()
print("=" * 108)
print("汇总：全链路 vs refusal_audit（后者跳过判分、给全量块）")
print("=" * 108)

o = [r for r in res["out"] if not r.get("error")]
n = [r for r in res["in"] if not r.get("error")]
o_ref = sum(1 for r in o if r["refused"])
o_web = sum(1 for r in o if r["n_web_search"] > 0)
n_ref = sum(1 for r in n if r["refused"])
n_zero = sum(1 for r in n if r["n_kept"] == 0)

print()
print("  %-22s %-16s %-16s %s" % ("口径", "库外拒答率", "库外走网络搜索", "库内拒答率"))
print("  " + "-" * 96)

# [FIX / 二次筛查] **对照行不再硬编码**。
#
# 原先这里写死了 `"100% (10/10)"` / `"—（不经路由）"` / `"9.4% (3/32)"` ——
# 那是 4.27 那次运行的数字。硬编码的后果是：**对照会永远"看起来正常"**，
# 即使 `refusal_audit.py` 的结果早已变化（语料换了、判分改了、题换了）。
# 那正是「拿旧数字当基准」的经典陷阱（本项目在 4.13 已经吃过一次：
# 拿上一条 trace 当本次对照，得出 266% 的假偏差）。
#
# 现在：**优先从 `refusal_audit.json` 读实际值**；读不到就显式标注「未找到」，
# 而不是伪造一个看起来合理的数字。
_ref_path = HERE.parent / "refusal_audit.json"
_ref_in = _ref_out = None
if _ref_path.exists():
    try:
        _d = json.loads(_ref_path.read_text(encoding="utf-8"))
        _ref_out = _d.get("out_of_scope", {})
        _ref_in = _d.get("in_scope", {})
    except Exception:  # noqa: BLE001
        _ref_out = _ref_in = None

if _ref_out and _ref_in and _ref_out.get("n"):
    _ro_rate = "%.1f%% (%d/%d)" % (100 * _ref_out["rate"], _ref_out["refused"], _ref_out["n"])
    _ri_rate = "%.1f%% (%d/%d)" % (100 * _ref_in["rate"], _ref_in["refused"], _ref_in["n"])
    _note = (
        "（读自 refusal_audit.json，本次运行于 %s）" % _d.get("stamp", "未知时间")
        if isinstance(_d, dict) and _d.get("stamp")
        else "（读自 refusal_audit.json）"
    )
    print("  %-22s %-16s %-16s %s" % ("refusal_audit（跳过判分）", _ro_rate, "—（不经路由）", _ri_rate))
    print("  %s" % _note)
else:
    print(
        "  %-22s %-16s %-16s %s"
        % ("refusal_audit（跳过判分）", "**未找到 refusal_audit.json**", "—（不经路由）", "**无法对照**")
    )

print(
    "  %-22s %-16s %-16s %s"
    % (
        "**全链路（本脚本）**",
        "%.1f%% (%d/%d)" % (100 * o_ref / max(len(o), 1), o_ref, len(o)),
        "%.1f%% (%d/%d)" % (100 * o_web / max(len(o), 1), o_web, len(o)),
        "%.1f%% (%d/%d)" % (100 * n_ref / max(len(n), 1), n_ref, len(n)),
    )
)
print()
print(
    "  注：**两组口径不同**（本脚本走完整链路含判分/路由/联网；refusal_audit 跳过判分"
    "直接给全量块），所以差异本身不是\"对错\"，而是\"两种问法\"。"
)
print()
print("  库内题中「判分后 0 块」的: %d/%d（这些会走 transform_query / web_search）" % (n_zero, len(n)))
if n_ref:
    print()
    print("  库内被拒答的题（可用性受损）:")
    for r in n:
        if r["refused"]:
            print("     [%s] %s" % (r.get("id"), r["q"][:56]))
            print("        判分后 %d 块，路由 %s，步骤 %s" % (r["n_kept"], r["route"], r["steps"][:80]))

(HERE.parent / "fullchain_audit.json").write_text(
    json.dumps({"out_of_scope": res["out"], "in_scope": res["in"]}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print()
print("  证据 -> eval/fullchain_audit.json")
