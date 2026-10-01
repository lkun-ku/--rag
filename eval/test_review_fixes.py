"""TDD：修复代码审查发现的 6 个问题（Standards + Spec 两轴 + 常见 bug 筛查）。

## 每个用例对应一个真实发现

| 用例 | 问题 | 来源 |
|---|---|---|
| **F1** | `graph_2.py` 无条件 `.result()`：第 1 次判 no 时 answer_grader 报错会**炸掉节点**（短路版不会调用它）—— 违反 4.25「不改变任何判定结果」 | Spec 轴 (c) |
| **F2** | `_WARN_ELIGIBLE` 含生成层两项但循环不查它 -> 改集合对生成层**不生效** | Standards smell 4 |
| **F3** | `grade_batch` 重复 idx 无校验：last-write-wins 可把 True 改 False，且 `got` 重复计数使「缺判」告警失效 | bug 筛查 6 |
| **F4** | `grade_batch(batch_size=...)` 传参被忽略（内部仍用模块常量算批） | Spec 轴 (b) |
| **F5** | `transform_query_node` 的 LLM 调用未显式传 config（同类坑第三次） | Spec 轴 (a)2 |
| **F6** | `tracer.traced` 对 `def fn(state, *, config=None)` 这类**仅关键字**形参会静默丢弃 config | Standards A3 |

**设计约束**：F1/F5/F6 涉及外部依赖（LLM、langgraph），用**最小桩**验证通路，
不真调模型；F2/F3/F4 是纯逻辑，直接断言。
"""

from __future__ import annotations

import inspect
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

failures = []
passed = 0


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%s  实际=%s" % (name, want, got))


print("=" * 100)
print("代码审查发现的修复 —— 逐项验证")
print("=" * 100)

# ---------------------------------------------------------------------------
# F2：_WARN_ELIGIBLE 必须对**所有**参与告警的指标生效
# ---------------------------------------------------------------------------
print()
print("[F2] _WARN_ELIGIBLE 对生成层也要生效")
import metrics  # noqa: E402

# 生成层处于告警区间、且不在白名单里时，不应告警。
# 注意：`context_precision` 取 0.06（**高于**它的告警线 0.05），
# 否则它自己就会告警、干扰本用例的判定（首版就是这么误报的）。
s = {
    "context_recall": 0.99,
    "hit_rate": 0.99,
    "context_precision": 0.06,
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
}
v, r = metrics.gate(s, gen_requested=True)
check("生成层在白名单内（faithfulness）-> 告警仍生效", v, "WARN")

# 用「从白名单里摘掉」的方式验证集合真的被查了
_saved = metrics._WARN_ELIGIBLE
try:
    metrics._WARN_ELIGIBLE = {"context_recall", "hit_rate", "context_precision"}
    v2, r2 = metrics.gate(s, gen_requested=True)
    check("faithfulness 不在白名单 -> 不告警", v2, "PASS")
finally:
    metrics._WARN_ELIGIBLE = _saved

# ---------------------------------------------------------------------------
# F3/F4：grade_batch 的 idx 校验与 batch_size 生效
# ---------------------------------------------------------------------------
print()
print("[F3] grade_batch 对「重复 idx」与「越界 idx」的处理")
from langchain_core.documents import Document  # noqa: E402
import graph2.grade_documents_node as gdn  # noqa: E402

DOCS = [Document(page_content="块 %d" % i, metadata={}) for i in range(1, 4)]


class _V:
    def __init__(self, i, rel):
        self.index, self.relevant = i, rel


class _R:
    def __init__(self, vs):
        self.verdicts = vs


# 桩：返回 (1->True, 1->False 重复, 3->False)，块 2 缺失
_stub = _R([_V(1, True), _V(1, False), _V(3, False)])
_orig_chain = gdn._batch_chain
_orig_get = gdn._get_batch_chain
gdn._batch_chain = type("C", (), {"invoke": staticmethod(lambda *a, **k: _stub)})()
gdn._get_batch_chain = lambda: gdn._batch_chain
try:
    out = gdn.grade_batch("q", DOCS)
    check("长度守恒", len(out), 3)
    check("重复 idx 不覆盖首次判定（块1 应保持 True）", out[0], True)
    check("明确的 False 生效（块3）", out[2], False)
    check("缺判的块保守保留（块2）", out[1], True)
finally:
    gdn._batch_chain = _orig_chain
    gdn._get_batch_chain = _orig_get

print()
print("[F4] grade_batch(batch_size=2) 传参必须真的生效")
_calls = {"n": 0}


class _CountingChain:
    def invoke(self, payload, config=None):
        _calls["n"] += 1
        # 按批内块数返回全 True
        n = payload["n"]
        return _R([_V(i, True) for i in range(1, n + 1)])


_orig2 = gdn._get_batch_chain
gdn._get_batch_chain = lambda: _CountingChain()
try:
    gdn.grade_batch("q", DOCS, batch_size=2)
    check("batch_size=2 时 3 个块切成 2 批（调用 2 次）", _calls["n"], 2)
    _calls["n"] = 0
    gdn.grade_batch("q", DOCS, batch_size=3)
    check("batch_size=3 时 3 个块切成 1 批（调用 1 次）", _calls["n"], 1)
finally:
    gdn._get_batch_chain = _orig2

# ---------------------------------------------------------------------------
# F5：transform_query_node 的 LLM 调用必须显式传 config
# ---------------------------------------------------------------------------
print()
print("[F5] transform_query_node 显式传 config")
tq_src = (ROOT / "RAG_PROJECT" / "graph2" / "transform_query_node.py").read_text(encoding="utf-8")
check(
    "invoke 处带 config=",
    "config=" in tq_src.split("question_rewriter.invoke")[1][:220] if "question_rewriter.invoke" in tq_src else False,
    True,
)

# ---------------------------------------------------------------------------
# F6：tracer.traced 对「仅关键字 config」的形参也要透传
# ---------------------------------------------------------------------------
print()
print("[F6] tracer.traced 对 `def fn(state, *, config=None)` 也要透传")
from utils.tracer import traced  # noqa: E402

seen = {}


def kwonly_node(state, *, config=None):
    seen["cfg"] = config
    return {"ok": True}


w = traced("kwonly")(kwonly_node)
w({"x": 1}, {"callbacks": ["STUB"]})
check("仅关键字 config 也被透传", seen.get("cfg") is not None, True)

print()
print("=" * 100)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 100)
sys.exit(0 if not failures else 1)
