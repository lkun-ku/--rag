"""TDD：`gate()` 不应把 `context_precision` 当**告警**项（它只作监控）。

## 要修的「代码与注释不一致」

`eval/metrics.py` 的注释明确写着：

    context_precision  在「宽召回 + 严判分」架构下**天然很低**：为命中 1 个必需块
                       要返回约 15 个块，理论上限就只有 1/15 ≈ 0.067。
                       它低**不代表检索差**，因此**只作监控**…**不作为主要门禁**。

但 `gate()` 里它**是被判定的一项**：

    for name in ("context_recall", "hit_rate", "context_precision"):
        ...
        elif v < th["warn"]:
            warned = True

于是每次回归只要 precision 落进 `[0.03, 0.05)` 就报 WARN，
而当前实测 **0.0496**（差 0.0004 就触发）—— **成了纯噪音**。

**为什么 0.0496 是「必然」而不是「变差」**：注释里那个「理论上限 1/15 ≈ 0.067」
是 `final_k=18` 时代算的（15 ≈ 18 块里约 1 个必需块）。现在 `final_k=36`，
上限降到约 `1/36`，precision 因此**必然下降** —— 这是 4.14/4.21
放大配额的**数学后果**，与检索质量无关。

## 契约三条（分开验证）

    C1 precision 处于 `[fail, warn)` -> **不产生告警**（verdict 不被它影响）
    C2 recall    处于 `[fail, warn)` -> **仍然告警**（不能把整个 gate 关掉）
    C3 precision 低于 **fail**       -> **仍然 FAIL**
       （保留注释说的「发现检索严重发散」这条兜底）
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402

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


def verdict(scores, gen=True):
    """跑一次 gate，返回 (verdict, reasons)。gen=True 表示生成层已给分（避免 INCONCLUSIVE）。"""
    s = dict(scores)
    if gen:
        s.setdefault("faithfulness", 0.95)
        s.setdefault("answer_relevancy", 0.95)
    return metrics.gate(s, gen_requested=gen)


print("=" * 96)
print("gate() 对 context_precision 的处理")
print("=" * 96)
print("  阈值: %s" % metrics.THRESHOLDS["context_precision"])
print()

# --- C1：precision 在告警区间 -> 不应告警 -----------------------------------
print("[C1] precision=0.0496（告警区间内，当前实测值）-> 期望 PASS（不告警）")
v, r = verdict({"context_recall": 0.8664, "hit_rate": 0.9375, "context_precision": 0.0496})
check("verdict == PASS（precision 不参与告警）", v, "PASS")
print("     reasons=%s" % r)

print()
print("[C1b] precision=0.031（刚过硬门禁）-> 期望 PASS")
v, r = verdict({"context_recall": 0.8664, "hit_rate": 0.9375, "context_precision": 0.031})
check("verdict == PASS", v, "PASS")

# --- C2：recall 在告警区间 -> 仍须告警 --------------------------------------
print()
print("[C2] recall=0.75（告警区间）-> 期望 **WARN**（不能把 gate 一起关掉）")
v, r = verdict({"context_recall": 0.75, "hit_rate": 0.9375, "context_precision": 0.0496})
check("verdict == WARN", v, "WARN")
check("reasons 里提到 context_recall", any("context_recall" in x for x in r), True)

# --- C3：precision 低于硬门禁 -> 仍须 FAIL ----------------------------------
print()
print("[C3] precision=0.01（< 硬门禁 0.03）-> 期望 **FAIL**（保留「严重发散」兜底）")
v, r = verdict({"context_recall": 0.8664, "hit_rate": 0.9375, "context_precision": 0.01})
check("verdict == FAIL", v, "FAIL")
check("reasons 里提到 context_precision", any("context_precision" in x for x in r), True)

# --- C4：回归保护：recall 硬门禁失效仍须 FAIL ------------------------------
print()
print("[C4·保护] recall=0.5（< 硬门禁 0.60）-> 期望 FAIL")
v, r = verdict({"context_recall": 0.5, "hit_rate": 0.9375, "context_precision": 0.0496})
check("verdict == FAIL", v, "FAIL")

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
