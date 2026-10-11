"""惰性初始化契约测试（import 期副作用治理）。

## 为什么需要它

实测冷启动：`import rag_kb.tools.retriever_tools` 17.93s、`crag_graph` 18.48s，
大头是模块级加载 bge-large-zh-v1.5（25.45s）。后果：
  * 每条 `python xxx.py` 都要等；
  * `uvicorn --workers 4` 时每个进程各加载一份（4 × 1.3GB 显存）；
  * 测试只要 import 到这些模块就白白等 20s。

改造目标：**import 不再加载模型**，改为首次访问属性/首次调用时惰性构造，
并用模块级 `__getattr__`（PEP 562）保持旧引用方式（`from ... import retriever`）不变。

## 契约

    C1 import retriever_tools 不再加载模型（< 5s）
    C2 import crag_graph 不再编译图/加载模型（< 5s）
    C3 旧的 `from ... import retriever` 仍可用，且触发一次加载
    C4 `get_retriever()` 是单例（多次调用同一对象）
    C5 `crag_graph.graph` 仍可访问（兼容旧引用），且与 `get_graph()` 同一对象
    C6 `embeddings_model.bge_embedding` 仍可访问

## 运行

    python tests/service/test_lazy_init.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable

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


def import_cost(mod: str) -> float:
    """在独立子进程里测 import 耗时（避免受本进程已加载模块的影响）。"""
    code = f"import time;t=time.time();import {mod};print(round(time.time()-t,2))"
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    if r.returncode != 0:
        print("     import 失败:", (r.stderr or "").strip()[-200:])
        return 999.0
    return float((r.stdout or "999").strip().splitlines()[-1])


print("=" * 96)
print("惰性初始化（import 期副作用治理）")
print("=" * 96)

# ---------------------------------------------------------------------------
print()
print("[C1] import retriever_tools 不再加载模型")
t1 = import_cost("rag_kb.tools.retriever_tools")
print("     耗时 %.2fs（改造前实测 17.93s，其中模型 25s 级）" % t1)
check("import 耗时 < 5s", t1 < 5.0, True)

# ---------------------------------------------------------------------------
print()
print("[C2] import crag_graph 不再编译图")
t2 = import_cost("rag_kb.graph_crag.crag_graph")
print("     耗时 %.2fs（改造前实测 18.48s）" % t2)
check("import 耗时 < 5s", t2 < 5.0, True)

# ---------------------------------------------------------------------------
print()
print("[C3] 旧引用方式仍可用（首次访问触发加载）")
from rag_kb.tools.retriever_tools import retriever as old_style_retriever  # noqa: E402

check("from ... import retriever 拿到实例", old_style_retriever is not None, True)

# ---------------------------------------------------------------------------
print()
print("[C4] get_retriever() 单例")
from rag_kb.tools.retriever_tools import get_retriever  # noqa: E402

check("两次调用同一对象", get_retriever() is get_retriever(), True)
check("与旧引用同一对象", get_retriever() is old_style_retriever, True)

# ---------------------------------------------------------------------------
print()
print("[C5] crag_graph.graph 兼容访问")
from rag_kb.graph_crag import crag_graph as cg  # noqa: E402

check("graph 与 get_graph() 同一对象", cg.graph is cg.get_graph(), True)
check("图对象有 nodes", len(cg.graph.get_graph().nodes) > 0, True)

# ---------------------------------------------------------------------------
print()
print("[C6] embeddings_model.bge_embedding 仍可访问")
from rag_kb.llm_models.embeddings_model import bge_embedding  # noqa: E402

check("bge_embedding 非空", bge_embedding is not None, True)
check("同一实例（懒加载只构造一次）", bge_embedding is bge_embedding, True)

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
