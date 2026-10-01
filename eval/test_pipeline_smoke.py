"""TDD：**完整工作流冒烟测试** —— 补上回归缺失的那一层。

## 为什么需要它（4.33 的教训）

二次筛查发现 `graph_2.py:101/104` 用了 `_cfg` 却**没有定义它**（P0）——
**任何走到该分支的正常请求都会抛 `NameError`**。

**而当时回归是 10/10 全绿**，因为覆盖是分层的、恰好漏了这一层：

    run_eval.py        只走**检索层**（检索器直调，不跑工作流）
    test_*.py          只测**单个函数**（grade_batch / gate / is_flat_twin …）
    run_all_checks     串联上述，但**没有任何一项端到端跑一次图**

于是「图能不能跑通」这件事**从来没被回归覆盖过** —— P0 就这么藏进来了。

## 契约

    C1 一道库内题跑完整工作流 -> **不抛异常**
    C2 返回的 `generation` **非空**（且不是空串/纯空白）
    C3 关键步骤都在（route_question / retrieve / grade_documents / generate）
    C4 **不走 web_search**（该题答案在库里，走网络说明链路有问题）

**为什么只跑 1 题**：冒烟测试的目标是「通路有没有断」，不是「质量好不好」——
质量由 `run_eval` 负责。1 题约 5s，可以放心进快速档回归。
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


import json  # noqa: E402

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
case = next(c for c in cases if c.get("source_blocks"))  # 取第一道有 golden 的题

print("=" * 96)
print("完整工作流冒烟：%s" % case.get("id"))
print("=" * 96)
print("  问题: %s" % case["q"])
print()

from graph2.graph_2 import run_with_trace  # noqa: E402

try:
    out = run_with_trace(case["q"])
    crashed = None
except Exception as e:  # noqa: BLE001
    out, crashed = None, "%s: %s" % (type(e).__name__, str(e)[:160])

# --- C1 不抛异常 ------------------------------------------------------------
print("[C1] 完整工作流不抛异常")
check("无异常", crashed, None)
if crashed:
    print("     **异常**: %s" % crashed)
    print("     -> 这正是本测试存在的意义：此前回归没有任何一项端到端跑图。")

if out is not None:
    steps = [s.get("node") for s in (out.get("trace", {}).get("steps") or [])]
    gen = out.get("generation") or ""

    # --- C2 generation 非空 ------------------------------------------------
    print()
    print("[C2] 返回的 generation 非空")
    check("generation 有内容", bool(gen.strip()), True)

    # --- C3 关键步骤齐全 ----------------------------------------------------
    print()
    print("[C3] 关键步骤齐全")
    for node in ("route_question", "retrieve", "grade_documents", "generate"):
        check("步骤 %s 出现" % node, node in steps, True)
    print("     实际步骤: %s" % " -> ".join(steps))

    # --- C4 不走 web_search（库内题）----------------------------------------
    print()
    print("[C4] 库内题不应回退到 web_search")
    check("未出现 web_search", "web_search" not in steps, True)

    print()
    print("  答案预览: %s" % gen[:110].replace("\n", " "))

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
