"""评测产物路径的通用化 —— 契约测试。

## 为什么需要它

换语料后，评测集由 `gen_testset.py` 重新生成（golden_auto.jsonl），但
`run_eval.py` 把**评测集文件名**和**产物文件名**（baseline/history）都写死了：

    评测集默认 golden_dataset.jsonl（换语料后已失效）
    baseline.json / history.jsonl 固定 —— 自动题与手工题的基线会**互相覆盖**

## 契约

    C1 默认评测集选择：golden_auto.jsonl 存在 -> 优先用它（与当前向量库对齐）
    C2 自动题不存在、旧手工题存在 -> 用旧手工题（向后兼容）
    C3 显式 --golden 永远优先
    C4 产物路径跟随评测集：
       golden_dataset.jsonl -> baseline.json / history.jsonl（旧名，向后兼容）
       golden_auto.jsonl    -> baseline_auto.jsonl / history_auto.jsonl
    C5 逐题诊断记录的是 goldens_of（要点 + 块），不再只有块 ID

## 运行

    python tests/eval/test_eval_paths.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402  (模块级只有 os.environ.setdefault，无副作用)

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
print("评测产物路径通用化")
print("=" * 96)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="gold_"))
auto = tmp / "golden_auto.jsonl"
manual = tmp / "golden_dataset.jsonl"
other = tmp / "golden_mydomain.jsonl"
auto.write_text("{}\n", encoding="utf-8")
manual.write_text("{}\n", encoding="utf-8")
other.write_text("{}\n", encoding="utf-8")

# ---------------------------------------------------------------------------
print()
print("[C1] 自动题存在 -> 默认用它")
check("pick(auto 优先)", run_eval._pick_golden("", [auto, manual]), auto)
check("pick(只有一个就用那个)", run_eval._pick_golden("", [manual]), manual)

# ---------------------------------------------------------------------------
print()
print("[C2] 显式 --golden 永远优先")
check("pick(显式 other)", run_eval._pick_golden(str(other), [auto, manual]), other)

# ---------------------------------------------------------------------------
print()
print("[C3] 什么都不存在 -> 明确报错路径")
try:
    run_eval._pick_golden("", [])
    check("应抛 FileNotFoundError", False, True)
except FileNotFoundError:
    check("应抛 FileNotFoundError", True, True)

# ---------------------------------------------------------------------------
print()
print("[C4] 产物路径跟随评测集（不互相覆盖）")
b1, h1 = run_eval._output_paths(manual)
check("手工题 -> baseline.json（旧名兼容）", b1.name, "baseline.json")
check("手工题 -> history.jsonl", h1.name, "history.jsonl")
b2, h2 = run_eval._output_paths(auto)
check("自动题 -> baseline_auto.json", b2.name, "baseline_auto.json")
check("自动题 -> history_auto.jsonl", h2.name, "history_auto.jsonl")
b3, h3 = run_eval._output_paths(other)
check("自定义域 -> baseline_mydomain.json", b3.name, "baseline_mydomain.json")
check("自定义域 -> history_mydomain.jsonl", h3.name, "history_mydomain.jsonl")

# ---------------------------------------------------------------------------
print()
print("[C5] 逐题诊断用 goldens_of（要点 + 块，不再只有块 ID）")
case = {
    "q": "QPS 是多少",
    "key_points": ["9575.69"],
    "source_blocks": ["af0fc19664f3"],
}
check("goldens_of 返回要点与块", run_eval.goldens_of(case), ["9575.69", "af0fc19664f3"])

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
