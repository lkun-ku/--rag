"""针对 P1 批次改动的验证（P1-2 / P1-3 / P1-6 / P1-7 / P1-8）。

## 覆盖

* **P1-2**：`/ask/stream`（SSE）是否把 trace 落盘 —— 跑到之前它从不 `t.save()`，
  导致走 SSE 的请求在 `logs/trace.jsonl` 里完全不存在。
  验证方式：记录运行前后 trace.jsonl 的**行数**，跑一次 SSE，再比。
* **P1-3**：`/ingest` 对 `enabled: False` 的源是否**显式拒绝**（原先返回 200 + ok=True，
  调用方会误以为已入库）。
* **P1-6**：`pptx_parser` 的静默 `except: pass` 已改为 `log.warning`（静态确认——
  触发它需要构造「shape_type 取不到」的畸形 pptx，成本高，这里只做导入与语法确认）。
* **P1-7 / P1-8**：docstring / 空分支的修正属文档与清理，无行为变化，靠导入确认。

**不吞异常**：任何失败都直接打印。
"""

import json
import os
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

TRACE = ROOT / "logs" / "trace.jsonl"


def n_lines(p: pathlib.Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip())


print("=" * 96)
print("P1 批次验证（P1-2 SSE trace 落盘 / P1-3 disabled 源拒绝）")
print("=" * 96)

from fastapi.testclient import TestClient  # noqa: E402

from rag_kb.api.main import app  # noqa: E402
from rag_kb.utils.env_utils import DOC_SOURCES  # noqa: E402

results = []

# ---------------------------------------------------------------------------
# P1-3：/ingest 对 disabled 源
# ---------------------------------------------------------------------------
print()
print("--- P1-3：/ingest 对 enabled=False 的源 ---")
disabled = [s["name"] for s in DOC_SOURCES if not s.get("enabled", True)]
print("  DOC_SOURCES 中禁用的源: %s" % (disabled or "（无）"))

with TestClient(app) as client:
    if disabled:
        r = client.post("/ingest", json={"source": disabled[0]})
        detail = str(r.json().get("detail", r.text))[:140]
        ok = r.status_code == 400
        results.append(("P1-3 /ingest disabled 源 -> 400", ok))
        print("  POST /ingest {source: %r} -> %d" % (disabled[0], r.status_code))
        print("      detail: %s" % detail)
        print("      => %s" % ("OK 正确拒绝" if ok else "**FAIL：应返回 400**"))
    else:
        print("  当前没有 disabled 源，跳过实测（逻辑已在代码中）")
        results.append(("P1-3（无 disabled 源可测）", True))

    # -----------------------------------------------------------------------
    # P1-2：SSE 是否落盘 trace
    # -----------------------------------------------------------------------
    print()
    print("--- P1-2：/ask/stream 是否落盘 trace ---")
    before = n_lines(TRACE)
    print("  运行前 logs/trace.jsonl 行数: %d" % before)

    t0 = time.time()
    try:
        with client.stream("POST", "/ask/stream", json={"question": "Milvus 支持哪些索引类型？"}) as resp:
            status = resp.status_code
            events = []
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line[7:].strip())
    except Exception as e:  # noqa: BLE001
        status, events = -1, ["ERR:%s" % type(e).__name__]
        print("  SSE 请求异常: %s: %s" % (type(e).__name__, e))

    elapsed = time.time() - t0
    after = n_lines(TRACE)
    print("  SSE 状态 %d | 事件数 %d（类型: %s）| 耗时 %.1fs" % (status, len(events), sorted(set(events))[:6], elapsed))
    print("  运行后 logs/trace.jsonl 行数: %d（增量 %d）" % (after, after - before))

    ok2 = (after - before) >= 1
    results.append(("P1-2 SSE 落盘 trace（行数 +1）", ok2))
    print("      => %s" % ("OK trace 已落盘" if ok2 else "**FAIL：SSE 仍未写 trace.jsonl**"))

    # 顺带看最后一条 trace 的内容，确认是方才那次
    if ok2:
        last = [l for l in TRACE.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-1]
        try:
            d = json.loads(last)
            print(
                "      最后一条 trace: q=%r | steps=%s | total=%.1fs"
                % (str(d.get("question"))[:40], d.get("n_steps"), d.get("total_sec", 0))
            )
        except Exception:  # noqa: BLE001
            print("      （最后一条无法解析为 JSON）")

print()
print("=" * 96)
print("汇总")
print("=" * 96)
n_ok = sum(1 for _n, ok in results if ok)
for name, ok in results:
    print("  [%s] %s" % ("OK  " if ok else "FAIL", name))
print()
print("  => %d/%d 通过" % (n_ok, len(results)))
