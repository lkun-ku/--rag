"""可行性验证 0：流式首字延迟 + import 分阶段耗时（只读，不改项目代码）。

用法：python _probe_baseline.py
"""

import asyncio
import subprocess
import sys
import time


async def probe_stream():
    from rag_kb.llm_models.all_llm import llm

    prompt = "用大约 200 字介绍向量数据库的索引类型，要求分点说明。"

    # A. 非流式：总耗时
    t = time.time()
    resp = await llm.ainvoke(prompt)
    total = time.time() - t
    n_chars = len(getattr(resp, "content", "") or "")

    # B. 流式：首 token 与总耗时
    t = time.time()
    first = None
    chunks, chars = 0, 0
    async for chunk in llm.astream(prompt):
        if first is None:
            first = time.time() - t
        chunks += 1
        chars += len(getattr(chunk, "content", "") or "")
    total2 = time.time() - t

    print("[流式] 非流式总耗时 %.2fs（%d 字）" % (total, n_chars))
    print("[流式] 流式首 token %.2fs，共 %d 块 / %d 字，总耗时 %.2fs" % (first, chunks, chars, total2))
    print("[流式] 首字提前量 %.2fs（%.0f%%）" % (total - first, (total - first) / total * 100))


def probe_import():
    mods = [
        "rag_kb.utils.env_utils",
        "rag_kb.llm_models.all_llm",
        "rag_kb.llm_models.embeddings_model",
        "rag_kb.tools.retriever_tools",
        "rag_kb.graph_crag.crag_graph",
    ]
    print("\n[import] 分阶段冷启动耗时（每个模块独立进程）")
    for m in mods:
        code = (
            "import time,os;os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');"
            f"t=time.time();import {m};print(round(time.time()-t,2))"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
        val = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "ERR"
        print("  %-40s %s s" % (m, val))


if __name__ == "__main__":
    asyncio.run(probe_stream())
    probe_import()
