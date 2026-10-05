"""确保 eval 工具脚本运行在项目 `.venv` 里；否则用 venv 解释器重启自身。

## 为什么需要它

`eval/` 下的工具（export_chunks / gen_testset / run_eval）依赖项目 venv 里的
`rag_kb` / `langchain_core`。而 Windows 上裸 `python` 常指向 Anaconda/系统解释器 ——
直接跑会得到一串 ModuleNotFoundError（实测：用户按文档跑
`python eval/export_chunks.py` 时三个工具全部如此）。

`main.py` 有同样的自动切换（`run_with_project_venv`），但 eval 脚本是独立入口，
覆盖不到 —— 本模块把这套保护补齐到工具侧。

## 用法（脚本最顶部、任何第三方 import 之前）

    import pathlib as _pl
    import sys as _sy
    _HERE = _pl.Path(__file__).resolve().parent
    if str(_HERE) not in _sy.path:
        _sy.path.insert(0, str(_HERE))
    import venv_guard
    venv_guard.ensure_venv()

## 为什么用 subprocess 而不是 `os.execv`

Windows 上 execv 是「原地替换进程」：PowerShell 会认为命令已结束并打印提示符，
而替换后的进程还在往同一控制台写 —— 提示符与输出交错（main.py 实测踩过）。
子进程方式父进程会等待，输出顺序正常。

设置 `EVAL_NO_REEXEC=1` 可关闭（与 main.py 的 `KB_NO_REEXEC` 同语义）。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def venv_python() -> pathlib.Path | None:
    """项目 venv 解释器路径；.venv 不存在时返回 None。"""
    return VENV_PYTHON if VENV_PYTHON.exists() else None


def needs_reexec() -> bool:
    """当前解释器**不是**项目 venv（且 .venv 存在、未被禁用）时为 True。

    判不出来（路径解析失败）时保守返回 True —— 走一次子进程重跑，
    最坏情况是多一层进程，不会因为「判错为已在 venv」而 import 失败。
    """
    if os.environ.get("EVAL_NO_REEXEC") == "1":
        return False
    vp = venv_python()
    if vp is None:
        return False
    try:
        return pathlib.Path(sys.executable).resolve() != vp.resolve()
    except OSError:
        return True


def reexec() -> None:
    """用项目 venv 的解释器子进程重跑当前脚本，结束后以子进程退出码退出。"""
    script = pathlib.Path(sys.argv[0]).resolve()
    cmd = [str(VENV_PYTHON), str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


def ensure_venv() -> None:
    """入口：需要时执行 reexec（不返回）。"""
    if needs_reexec():
        reexec()
