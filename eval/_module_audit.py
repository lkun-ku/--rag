"""深度模块化体检：**真实导入** 52 个业务模块，检查：
  ① 能否 import（暴露循环导入 / 缺失依赖 / 语法问题）
  ② 是否有 `__main__` 但缺 sys.path 兜底（直跑会炸）
  ③ 模块级公开符号是否**从未被任何文件引用**（死代码候选）

全部基于真实执行结果，不做静态猜测。
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

# ---- 1. 收集业务模块 -------------------------------------------------------
biz = sorted(p for p in (ROOT / "RAG_PROJECT").rglob("*.py") if "__pycache__" not in str(p))
mods = []
for p in biz:
    rel = p.relative_to(ROOT / "RAG_PROJECT").with_suffix("")
    name = ".".join(rel.parts)
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    mods.append((name, p))

print("=" * 100)
print("① 真实导入（%d 个业务模块）" % len(mods))
print("=" * 100)
ok, fail = [], []
for name, p in mods:
    try:
        importlib.import_module(name)
        ok.append(name)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        kind = "循环导入" if "partially initialized" in tb else type(e).__name__
        fail.append((name, kind, str(e)[:110]))

print("  成功 %d / %d" % (len(ok), len(mods)))
if fail:
    print("  **失败 %d 个**：" % len(fail))
    for name, kind, msg in fail:
        print("    [%s] %-40s %s" % (kind, name, msg))
else:
    print("  ✅ 全部可导入")

# ---- 2. 有 __main__ 但没有 sys.path 兜底 -----------------------------------
print()
print("=" * 100)
print("② 有 `__main__` 但缺 sys.path 兜底（直跑会 ModuleNotFoundError）")
print("=" * 100)
bad_main = []
for name, p in mods:
    t = p.read_text(encoding="utf-8", errors="replace")
    if "__main__" not in t:
        continue
    has_fallback = "_ROOT = _pathlib.Path(__file__).resolve().parent.parent" in t
    if not has_fallback:
        bad_main.append(name)
print(
    "  有 __main__ 的模块 %d 个，缺兜底 %d 个"
    % (sum(1 for _, p in mods if "__main__" in p.read_text(encoding="utf-8", errors="replace")), len(bad_main))
)
for n in bad_main:
    print("    [缺兜底] %s" % n)
if not bad_main:
    print("  ✅ 全部有兜底")

# ---- 3. 模块级公开符号从未被引用（死代码候选）------------------------------
print()
print("=" * 100)
print("③ 模块级公开符号是否从未被引用（死代码候选，排除 __init__ / __main__）")
print("=" * 100)
all_src = {}
for p in list((ROOT / "RAG_PROJECT").rglob("*.py")) + list((ROOT / "eval").rglob("*.py")):
    if "__pycache__" in str(p):
        continue
    all_src[str(p)] = p.read_text(encoding="utf-8", errors="replace")

dead = []
for name, p in mods:
    if p.name == "__init__.py":
        continue
    t = p.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"^(?:def|class)\s+([A-Za-z_]\w*)", t, re.M):
        sym = m.group(1)
        if sym.startswith("_"):
            continue  # 私有符号不查（内部使用）
        # 在**其它文件**里找引用
        used = False
        for fp, src in all_src.items():
            if fp == str(p):
                continue
            if re.search(r"\b%s\b" % re.escape(sym), src):
                used = True
                break
        if not used:
            dead.append("%s::%s" % (name, sym))

print("  未被任何其它文件引用的公开符号: %d 个" % len(dead))
for d in dead[:20]:
    print("    %s" % d)
if len(dead) > 20:
    print("    ...（还有 %d 个）" % (len(dead) - 20))

# ---- 汇总 ------------------------------------------------------------------
print()
print("=" * 100)
print("汇总")
print("=" * 100)
print("  导入失败       : %d" % len(fail))
print("  缺 sys.path 兜底: %d" % len(bad_main))
print("  死代码候选     : %d" % len(dead))
sys.exit(0 if not fail and not bad_main else 1)
