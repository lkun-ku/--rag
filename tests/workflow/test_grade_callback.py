"""TDD 测试：`grade_documents` 必须把上层 config（callbacks）透传给判分链。

## 要解决的实测问题（4.20.7）

判分是**最大的 LLM 调用方**（36 次/题，占调用次数 90%），但它的 token
**不被 `run_with_trace` 统计** —— 根因是节点内手动调用没传 config：

    score = retrieval_grader_chain.invoke(          # 没有 config
        {"question": question, "document": document.page_content}
    )

后果：**最大的成本项在生产监控里不可见**，导致任何降成本改动都无法验证。

## 接缝与契约（TDD：先定接缝，再写测试）

    接缝：`graph_crag.grade_documents_node.grade_documents(state, config=None)`
    契约：若传入 `config`（含 `callbacks`），则**判分链的每次 LLM 调用
          都会触发该 callback 的 `on_llm_end`** —— 即 token 可被统计。

## 期望值来源（避免同义反复）

不用「实现怎么算就怎么期望」，而是**观察一个真实回调的副作用**：

    * 3 个文档 + 传 config  ->  `on_llm_end` 触发 **3 次**（一块一次，与实现无关）
    * 3 个文档 + 不传 config ->  `on_llm_end` 触发 **0 次**（对照组，
      证明这个差异**来自 config 透传**，而不是别的）

对照组是必需的：否则「3 次」可能来自别处的回调，测不出真正原因。

判分用真实 LLM（只 3 块，成本可忽略）；`_grade_one` 在异常时会保守保留，
所以即便 API 抖动也不影响计数（调用仍发生）。

探针先行：`eval/_probe_langgraph_config.py` 已实测确认「LangGraph 会把 config
传给两参数节点」（`config['callbacks']` 是 `CallbackManager` 对象）。
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

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


class CountingCallback(BaseCallbackHandler):
    """只记数：LLM 调用结束了几次。"""

    def __init__(self):
        self.llm_ends = 0

    def on_llm_end(self, response, **kwargs):  # noqa: D401
        self.llm_ends += 1


Q = "Milvus-16c7c-sq4u-fp16-force_merge 的 QPS 是多少？"
DOCS = [
    Document(
        page_content="表格：检索性能实测（第 1-30 行）\n| 场景 | QPS |\n| 静态向量检索 | 9575.69 |",
        metadata={"filename": "t.md"},
    ),
    Document(page_content="Milvus 是一个向量数据库，支持混合检索与稀疏向量。", metadata={"filename": "t.md"}),
    Document(page_content="本文讨论 RAG 系统的七个常见失败点，与向量检索无关。", metadata={"filename": "t.md"}),
]

print("=" * 96)
print("grade_documents 的 config 透传测试")
print("=" * 96)
print()

try:
    from rag_kb.graph_crag.grade_documents_node import grade_documents
except ImportError as e:  # noqa: BLE001
    print("  [**FAIL**] 无法 import grade_documents: %s" % e)
    sys.exit(1)

import rag_kb.graph_crag.grade_documents_node as gdn  # noqa: E402

# --- 用例 1：传 config -> 判分链的每次 LLM 调用都应触发 callback -------------
#
# 4.23 启用批量判分后，调用次数随模式变化（3 块：逐块=3 次、批量=1 次），
# 所以这里**两种模式都要覆盖** —— 否则测试会与实现脱节（首版就是这么红的）。
print("[用例1a] 逐块模式（USE_BATCH_GRADE=False）-> 期望 on_llm_end 触发 3 次")
_orig_flag = gdn.USE_BATCH_GRADE
gdn.USE_BATCH_GRADE = False
cb1 = CountingCallback()
try:
    out1 = grade_documents({"question": Q, "documents": DOCS}, {"callbacks": [cb1]})
except TypeError as e:
    print("  [**FAIL**] 节点不接受第二个参数（config）: %s" % e)
    print("            -> 这正是当前缺陷：节点签名是 `fn(state)`，拿不到上层 config")
    out1 = None
if out1 is not None:
    check("逐块模式：3 块 -> 3 次 LLM 调用都被统计", cb1.llm_ends, 3)
    check("仍返回 filtered documents", isinstance(out1.get("documents"), list), True)

print()
print("[用例1b] 批量模式（USE_BATCH_GRADE=True）-> 期望 on_llm_end 触发 1 次（3 块 < BATCH_SIZE，合成 1 批）")
gdn.USE_BATCH_GRADE = True
cb1b = CountingCallback()
try:
    out1b = grade_documents({"question": Q, "documents": DOCS}, {"callbacks": [cb1b]})
except Exception as e:  # noqa: BLE001
    print("  [**FAIL**] 批量模式调用失败: %s: %s" % (type(e).__name__, e))
    out1b = None
if out1b is not None:
    check("批量模式：3 块 -> 1 次 LLM 调用（且 token 可统计）", cb1b.llm_ends, 1)
    check("批量模式：仍返回 filtered documents", isinstance(out1b.get("documents"), list), True)
gdn.USE_BATCH_GRADE = _orig_flag

# --- 用例 2（对照）：不传 config -> 不应有任何回调触发 ----------------------
print()
print("[用例2·对照] 不传 config -> 期望 on_llm_end 触发 0 次（证明差异来自 config）")
cb2 = CountingCallback()
try:
    out2 = grade_documents({"question": Q, "documents": DOCS})
except TypeError as e:
    print("  [**FAIL**] 不传 config 时调用失败（应仍然可用）: %s" % e)
    out2 = None
if out2 is not None:
    check("无 config 时不触发回调", cb2.llm_ends, 0)

# --- 用例 3（**真实链路**）：经 tracer.traced 包装后，config 仍须透传 ----------
#
# 这一条才是真正定位到根因的用例：`crag_graph.py` 注册的**不是**裸节点，而是
#   workflow.add_node("grade_documents", tracer.traced("grade_documents")(grade_documents))
# 首版 wrapper 的签名是 `wrapper(state)` —— 它**丢弃**了 LangGraph 传下来的 config，
# 于是 `grade_documents` 永远拿到 `config=None`，判分 token 依旧不被统计
# （实测：修复前后 total 只差 637，而不是 36 次判分应带来的 ~17K）。
print()
print("[用例3·真实链路] 经 tracer.traced 包装后 -> on_llm_end 须 > 0（用**当前默认配置**）")
from rag_kb.utils.tracer import traced  # noqa: E402

wrapped = traced("grade_documents")(grade_documents)
cb3 = CountingCallback()
try:
    out3 = wrapped({"question": Q, "documents": DOCS}, {"callbacks": [cb3]})
except TypeError as e:
    print("  [**FAIL**] 包装器不接受/不透传 config（**这就是根因**）: %s" % e)
    out3 = None
if out3 is not None:
    # 断言「> 0」而不是具体次数：次数随 USE_BATCH_GRADE 变化
    # （逐块=3、批量=1），而**这条测试要守的是「config 穿过包装器」**，
    # 不是「调用几次」。把它写死成某个数字会与实现脱节（首版就是这么红的）。
    check("包装后的节点 token 可被统计（on_llm_end > 0）", cb3.llm_ends > 0, True)
    print("     实际触发 %d 次（当前 USE_BATCH_GRADE=%s）" % (cb3.llm_ends, gdn.USE_BATCH_GRADE))

# --- 用例 4：包装器不能破坏「不接受 config 的节点」 ---------------------------
# （其它节点如 retrieve/generate 的签名仍是 `fn(state)`，包装器要能兼容）
print()
print("[用例4·兼容性] 包装器对只接受 state 的节点仍须正常工作")
calls = {"n": 0}


def plain_node(state):
    calls["n"] += 1
    return {"x": 1}


w4 = traced("plain")(plain_node)
try:
    w4({"x": 0}, {"callbacks": [CountingCallback()]})
    check("只接受 state 的节点未被破坏", calls["n"], 1)
except TypeError as e:
    print("  [**FAIL**] 包装器给不接受的节点传了 config: %s" % e)
    check("只接受 state 的节点未被破坏", calls["n"], 1)

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
