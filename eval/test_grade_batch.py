"""TDD：批量判分 `grade_batch()` —— 一次 LLM 调用判多块。

## 为什么做它

判分占真实成本的 **76%**（4.22 实测：单题 41,868 token 里判分约 31K），
原因是它**逐块调用**（`FINAL_K=36` 时 36 次/题）：

    graded = list(ex.map(lambda d: _grade_one(question, d, config), documents))

**每块都要重新发一遍 system prompt + 问题**，36 次就是 36 遍。

## 接缝与契约

    接缝：`graph2.grade_documents_node.grade_batch(question, documents, config=None)`
    输入：问题 + N 个 Document（N 可大于批大小，内部自会分批）
    输出：`List[bool]`，长度 == N，**顺序与输入一致**（每项 = 该块是否相关）

契约三条，**分开验证**（TDD：先定契约，再写测试）：

    C1 **结构**（确定性，硬断言）：返回长度恒等于输入块数、类型为 bool
       —— 与 LLM 的表现无关，必须 100% 成立。
    C2 **语义**（依赖 LLM，用**极端样本**）：明显相关的块 -> True；
       明显无关的块 -> False。用极端样本是为了让断言稳定，不赌模型细节。
    C3 **容错**：单块判定缺失/异常时，**不能中断整批**，且按保守语义
       保留（True）—— 与 `_grade_one` 的异常兜底一致。

## 一致率**只作报告，不作断言**

「与逐个判定的结果一致率」是**质量指标**，要用数据判断是否可接受，
不是 pass/fail —— 把它写成断言会导致测试因为 LLM 的合理分歧而假红。
本文件的测试只覆盖 C1/C2/C3；一致率与 token 节省由
`eval/measure_grade_batch_saving.py` 实测。
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

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


Q = "Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 是多少？"

# 极端样本：前两块**明显相关**（含问题里的型号 + QPS 数值），后两块**明显无关**
DOCS = [
    Document(page_content="表格：检索性能实测（第 1-30 行）\n"
                          "| 场景 | 产品与规格 | QPS |\n"
                          "| 静态向量检索 | Milvus-16c64g-sq4u-fp16-force_merge | 9575.69 |",
             metadata={"filename": "t.md"}),
    Document(page_content="Milvus-16c64g-sq4u-fp16-force_merge 在静态向量检索场景下"
                          "QPS 为 9575.69，P99 延迟 2.3ms。",
             metadata={"filename": "t.md"}),
    Document(page_content="春天是万物复苏的季节，诗人们常用柳树与桃花来描写它。",
             metadata={"filename": "t.md"}),
    Document(page_content="本公司的报销流程如下：先填写申请单，再交由部门主管审批。",
             metadata={"filename": "t.md"}),
]

print("=" * 96)
print("grade_batch() 测试")
print("=" * 96)
print()

try:
    from graph2.grade_documents_node import grade_batch
except ImportError as e:  # noqa: BLE001
    print("  [**FAIL**] 无法 import grade_batch（尚未实现）: %s" % e)
    print()
    print("Red 阶段符合预期：函数不存在。")
    sys.exit(1)

# --- C1 结构 --------------------------------------------------------------
print("[C1·结构] 返回长度 == 输入块数、类型为 bool")
try:
    out = grade_batch(Q, DOCS)
    check("返回长度", len(out), len(DOCS))
    check("元素类型都是 bool", all(isinstance(x, bool) for x in out), True)
except Exception as e:  # noqa: BLE001
    print("  [**FAIL**] 调用抛异常: %s: %s" % (type(e).__name__, e))
    out = None

# --- C2 语义（极端样本）---------------------------------------------------
print()
print("[C2·语义] 明显相关的前两块 -> True；明显无关的后两块 -> False")
if out is not None:
    check("块1（含型号+QPS 表格行）", out[0], True)
    check("块2（含型号+QPS 的正文）", out[1], True)
    check("块3（讲春天，明显无关）", out[2], False)
    check("块4（讲报销流程，明显无关）", out[3], False)
    print("     实际返回: %s" % out)

# --- C3 容错 ---------------------------------------------------------------
print()
print("[C3·容错] 空输入不能抛异常")
try:
    r = grade_batch(Q, [])
    check("空输入返回空列表", r, [])
except Exception as e:  # noqa: BLE001
    print("  [**FAIL**] 空输入抛异常: %s: %s" % (type(e).__name__, e))
    check("空输入返回空列表", "抛异常", [])

# --- C4 并发：多批（块数 > BATCH_SIZE）时结果与顺序不变 ----------------------
#
# 4.24 给批量判分加了「批内并发」。并发最容易引入的缺陷是**顺序错乱**
# （`executor.map` 保序，但要**测出来**），以及**批次间结果串位**。
# 这里用 12 块（> BATCH_SIZE=8，必然走多批）验证：
#   * 长度与类型（结构）；
#   * **前 4 块的判定与单批时一致**（顺序正确性的直接证据）。
print()
print("[C4·并发] 12 块（> BATCH_SIZE，必然多批）-> 长度/类型/顺序都不变")
MANY = DOCS + [
    Document(page_content="Milvus 的索引类型包括 FLAT、IVF_FLAT、HNSW 等。",
             metadata={"filename": "t.md"}),
    Document(page_content="本店营业时间为每天上午九点到晚上十点。",
             metadata={"filename": "t.md"}),
    Document(page_content="Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 是 9575.69。",
             metadata={"filename": "t.md"}),
    Document(page_content="今天天气晴朗，适合外出散步。", metadata={"filename": "t.md"}),
    Document(page_content="Python 的 GIL 限制了多线程的并行计算能力。",
             metadata={"filename": "t.md"}),
    Document(page_content="Milvus-16c64g-sq4u-fp16-force_merge 的 P99 延迟为 2.3ms。",
             metadata={"filename": "t.md"}),
    Document(page_content="火锅的底料通常由牛油、辣椒和花椒组成。",
             metadata={"filename": "t.md"}),
    Document(page_content="向量数据库的召回率与索引参数 nlist、nprobe 有关。",
             metadata={"filename": "t.md"}),
]
try:
    out_many = grade_batch(Q, MANY)
    check("多批：返回长度恒等于输入块数", len(out_many), len(MANY))
    check("多批：元素类型都是 bool", all(isinstance(x, bool) for x in out_many), True)
    if out is not None:
        check("多批：前 4 块判定与单批一致（顺序正确）", out_many[:4], out)
    print("     多批返回: %s" % out_many)
except Exception as e:  # noqa: BLE001
    print("  [**FAIL**] 多批调用抛异常: %s: %s" % (type(e).__name__, e))
    check("多批：返回长度恒等于输入块数", "抛异常", len(MANY))

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
