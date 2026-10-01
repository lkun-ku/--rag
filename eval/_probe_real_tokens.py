"""跑一次真实工作流，抓**可信的** token 统计（并核对我上一版估算的口径问题）。

上一版 `measure_llm_cost.py` 用「按真实输入长度估算」的方式算 token，
然后拿 `logs/trace.jsonl` 的**最后一条**去对照 —— 但那条是**之前某次运行**的记录，
口径不同，对照无效（偏差 266% 就是这么来的）。这里改成：**自己跑一次**，
再读自己刚写进去的那条。
"""

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

Q = "Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 是多少？"

from graph2.graph_2 import run_with_trace  # noqa: E402

tp = ROOT / "logs" / "trace.jsonl"
before = tp.stat().st_size if tp.exists() else 0

t0 = time.time()
out = run_with_trace(Q)
el = time.time() - t0

print("=" * 100)
print("真实工作流的 token 统计")
print("=" * 100)
print("  问题: %s" % Q)
print("  耗时: %.1fs" % el)
tk = out.get("trace", {}).get("tokens", {})
print("  total=%s  input=%s  output=%s" % (tk.get("total_tokens"), tk.get("input_tokens"), tk.get("output_tokens")))
bm = tk.get("by_model", {})
for m, v in bm.items():
    print(
        "  模型 %s: total=%s (in=%s / out=%s)"
        % (m, v.get("total_tokens"), v.get("input_tokens"), v.get("output_tokens"))
    )
print()
print("  步数 %d:" % len(out.get("trace", {}).get("steps", [])))
for st in out.get("trace", {}).get("steps", []):
    print("     %-22s %.2fs  %s" % (st.get("node"), st.get("elapsed_sec", 0), st.get("decision", "")))
print()
print("  答案: %s" % (out.get("generation") or "")[:100].replace("\n", " "))
print("  判分后上下文块数: %d" % len(out.get("documents", [])))

# 读本次新写入的那条，核对一致
if tp.exists() and tp.stat().st_size > before:
    new = [l for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
    last = json.loads(new[-1])
    print()
    print(
        "  trace.jsonl 新增条目: total=%s  问题匹配=%s"
        % (last.get("tokens", {}).get("total_tokens"), last.get("question", "")[:20] == Q[:20])
    )
else:
    print()
    print("  [!] trace.jsonl 未新增（可能 save=False 或被覆盖）")
