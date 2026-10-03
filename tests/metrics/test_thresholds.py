"""评测阈值外置 —— 契约测试。

## 为什么需要它

原阈值写死在 `metrics.py` 里，是按「含 2 道必错题」的旧口径定的
（context_recall warn=0.80，而真实水平是 0.8584）。换语料后旧阈值毫无意义，
但改阈值又要动代码 —— 于是「换材料」变成了一次代码变更。

外置后的契约：

    C1 文件不存在  -> 用内置默认值（行为与改造前完全一致）
    C2 文件存在    -> 覆盖对应指标，未列出的保持默认（**增量合并**，不是整体替换）
    C3 只写一半    -> 只覆盖写了的那个键（如只写 fail，warn 保持默认）
    C4 非法 JSON   -> 回退默认值，**不抛异常**（评测工具不该因配置错误崩掉）
    C5 未知指标名  -> 忽略，不污染结果

## 运行

    python tests/metrics/test_thresholds.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))

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
        print("  [**FAIL**] %s  期望=%r  实际=%r" % (name, want, got))


print("=" * 96)
print("评测阈值外置")
print("=" * 96)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="thr_"))
DEFAULT_WARN = metrics.DEFAULT_THRESHOLDS["context_recall"]["warn"]
DEFAULT_FAIL = metrics.DEFAULT_THRESHOLDS["context_recall"]["fail"]

# ---------------------------------------------------------------------------
print()
print("[C1] 文件不存在 -> 默认值")
missing = tmp / "nope.json"
th = metrics.load_thresholds(missing)
check("context_recall.warn 等于默认", th["context_recall"]["warn"], DEFAULT_WARN)
check("返回的是副本（改它不影响默认）", th is not metrics.DEFAULT_THRESHOLDS, True)

# ---------------------------------------------------------------------------
print()
print("[C2] 文件存在 -> 增量覆盖")
f2 = tmp / "full.json"
f2.write_text(json.dumps({"context_recall": {"fail": 0.70, "warn": 0.84}}), encoding="utf-8")
th2 = metrics.load_thresholds(f2)
check("context_recall.warn 被覆盖为 0.84", th2["context_recall"]["warn"], 0.84)
check("context_recall.fail 被覆盖为 0.70", th2["context_recall"]["fail"], 0.70)
check(
    "未列出的指标保持默认",
    th2["hit_rate"]["warn"],
    metrics.DEFAULT_THRESHOLDS["hit_rate"]["warn"],
)

# ---------------------------------------------------------------------------
print()
print("[C3] 只写一半 -> 只覆盖写了的键")
f3 = tmp / "half.json"
f3.write_text(json.dumps({"hit_rate": {"warn": 0.90}}), encoding="utf-8")
th3 = metrics.load_thresholds(f3)
check("hit_rate.warn == 0.90", th3["hit_rate"]["warn"], 0.90)
check(
    "hit_rate.fail 保持默认",
    th3["hit_rate"]["fail"],
    metrics.DEFAULT_THRESHOLDS["hit_rate"]["fail"],
)

# ---------------------------------------------------------------------------
print()
print("[C4] 非法 JSON -> 回退默认，不抛异常")
f4 = tmp / "bad.json"
f4.write_text("{not json at all", encoding="utf-8")
try:
    th4 = metrics.load_thresholds(f4)
    check("回退到默认值", th4["context_recall"]["warn"], DEFAULT_WARN)
    check("没有抛异常", True, True)
except Exception as e:  # noqa: BLE001
    check("没有抛异常（实际抛了 %s）" % type(e).__name__, False, True)

# ---------------------------------------------------------------------------
print()
print("[C5] 未知指标名 -> 忽略")
f5 = tmp / "unknown.json"
f5.write_text(
    json.dumps({"context_recall": {"warn": 0.9}, "unheard_metric": {"fail": 0.1}}), encoding="utf-8"
)
th5 = metrics.load_thresholds(f5)
check("已知指标仍生效", th5["context_recall"]["warn"], 0.9)
check("未知指标没有被塞进来", "unheard_metric" in th5, False)

# ---------------------------------------------------------------------------
print()
print("[C6] 模块级 THRESHOLDS 仍可用（向后兼容）")
check("THRESHOLDS 是 dict", isinstance(metrics.THRESHOLDS, dict), True)
check("包含五个指标", sorted(metrics.THRESHOLDS.keys()), sorted(metrics.DEFAULT_THRESHOLDS.keys()))

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
