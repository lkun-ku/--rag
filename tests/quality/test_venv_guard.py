"""解释器护栏（eval/venv_guard）—— 契约测试。

## 为什么需要它

eval 工具依赖项目 .venv，但 Windows 裸 `python` 常指向 Anaconda ——
直接跑会 ModuleNotFoundError。护栏负责「不在 venv 就自动切换重跑」。

## 契约

    C1 在项目 venv 里运行时：不需要 reexec
    C2 环境变量 EVAL_NO_REEXEC=1 时：永不 reexec
    C3 .venv 不存在的环境：不 reexec（交给后续 import 自然报错）
    C4 venv_python() 能返回存在的解释器路径

## 运行

    python tests/quality/test_venv_guard.py
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import venv_guard as vg  # noqa: E402

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
print("eval 解释器护栏")
print("=" * 96)

# ---------------------------------------------------------------------------
print()
print("[C1] 测试本身运行在项目 venv 里 -> 不需要 reexec")
check("needs_reexec() == False", vg.needs_reexec(), False)

# ---------------------------------------------------------------------------
print()
print("[C2] EVAL_NO_REEXEC=1 -> 永不 reexec")
old = os.environ.get("EVAL_NO_REEXEC")
os.environ["EVAL_NO_REEXEC"] = "1"
try:
    check("EVAL_NO_REEXEC=1 时 needs_reexec() == False", vg.needs_reexec(), False)
finally:
    if old is None:
        os.environ.pop("EVAL_NO_REEXEC", None)
    else:
        os.environ["EVAL_NO_REEXEC"] = old

# ---------------------------------------------------------------------------
print()
print("[C3] 模拟「解释器不是 venv」-> 需要 reexec")
_real_exe = sys.executable
try:
    sys.executable = r"C:\Anaconda\python.exe"
    check("非 venv 解释器时 needs_reexec() == True", vg.needs_reexec(), True)
    os.environ["EVAL_NO_REEXEC"] = "1"
    check("（配合开关可关闭）", vg.needs_reexec(), False)
    os.environ.pop("EVAL_NO_REEXEC", None)
finally:
    sys.executable = _real_exe

# ---------------------------------------------------------------------------
print()
print("[C4] venv_python() 返回存在的解释器")
vp = vg.venv_python()
check("路径存在", vp is not None and pathlib.Path(vp).exists(), True)

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
