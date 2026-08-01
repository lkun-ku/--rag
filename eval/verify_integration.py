"""系统性深度测试 · 阶段 4：**系统联动测试**。

## 为什么要专门测「多租户 × SSE」

刚修掉的那个严重回归恰好出在这个组合上：

    with tenant_scope(tenant):        # ContextVar.reset(token)
        for ... in graph.stream(...):
            yield ...                 # 生成器跨上下文迭代 -> reset 抛 ValueError
                                      # -> 整个 SSE 端点形同废弃

而**单租户下的验证完全覆盖不到它**（`tenant_scope(None)` 同样会 set/reset，
也一样抛）。也就是说：`verify_p1_batch` 能测出「SSE 挂了」，但**测不出「为什么只在
租户模式下挂」**，更测不出「租户隔离是否真的生效」。

本脚本按「配置 × 入口」的**矩阵**跑，覆盖三条联动路径：

    场景 A  单租户     + 同步 /ask      （基线）
    场景 B  多租户启用 + 同步 /ask      （带 X-Tenant-Id；覆盖检索器层隔离）
    场景 C  多租户启用 + SSE  /ask/stream（**最关键** —— 回归就出在这里）
    场景 D  多租户启用 + 无租户头        （应拒绝：400）
    场景 E  多租户启用 + 未知租户        （应拒绝：403）

每个场景在**子进程**里跑（`env_utils` 在 import 期读环境变量，同进程改不了）。

用法：
    .venv\\Scripts\\python.exe eval\\verify_integration.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RAG = ROOT / "RAG_PROJECT"

# 用一个真实存在的源名（见 utils/env_utils.py 的 DOC_SOURCES）
TENANTS_JSON = '{"acme": ["milvus_docs"], "beta": ["manuals"]}'

# 子进程里跑的探针：跑指定场景并打印一行 JSON 结果
PROBE = r'''
import json, os, pathlib, sys
ROOT = pathlib.Path(r"{root}")
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi.testclient import TestClient
from api.main import app

SCEN = "{scen}"
Q = "Milvus 支持哪些索引类型？"
out = {{"scenario": SCEN, "ok": False, "detail": {{}}}}

with TestClient(app) as client:
    try:
        if SCEN == "A":
            r = client.post("/ask", json={{"question": Q}})
            b = r.json()
            out["detail"] = {{"http": r.status_code, "refused": b.get("refused"),
                             "n_cites": len(b.get("citations") or []),
                             "answer_len": len(b.get("answer") or "")}}
            out["ok"] = r.status_code == 200 and (b.get("answer") or "")
        elif SCEN == "B":
            r = client.post("/ask", json={{"question": Q}},
                            headers={{"X-Tenant-Id": "acme"}})
            b = r.json()
            files = sorted({{c.get("filename") for c in (b.get("citations") or [])}})
            out["detail"] = {{"http": r.status_code, "refused": b.get("refused"),
                             "tenant": b.get("tenant"), "cited_files": files[:4]}}
            out["ok"] = r.status_code == 200 and b.get("tenant") == "acme"
        elif SCEN == "C":
            events = []
            answer_payload = {{}}
            with client.stream("POST", "/ask/stream", json={{"question": Q}},
                               headers={{"X-Tenant-Id": "acme"}}) as resp:
                http = resp.status_code
                for raw in resp.iter_lines():
                    line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                    if line.startswith("event: "):
                        events.append(line[7:].strip())
                    elif line.startswith("data: ") and events and events[-1] == "answer":
                        answer_payload = json.loads(line[6:])
            out["detail"] = {{"http": http, "event_types": sorted(set(events)),
                             "has_answer": "answer" in events,
                             "has_error": "error" in events,
                             "tenant": answer_payload.get("tenant"),
                             "answer_len": len(answer_payload.get("answer") or "")}}
            out["ok"] = ("answer" in events) and ("error" not in events)
        elif SCEN == "D":
            r = client.post("/ask", json={{"question": Q}})
            out["detail"] = {{"http": r.status_code, "body": str(r.json())[:90]}}
            out["ok"] = r.status_code == 400
        elif SCEN == "E":
            r = client.post("/ask", json={{"question": Q}},
                            headers={{"X-Tenant-Id": "nobody"}})
            out["detail"] = {{"http": r.status_code, "body": str(r.json())[:90]}}
            out["ok"] = r.status_code == 403
    except Exception as e:
        out["detail"] = {{"exception": "%s: %s" % (type(e).__name__, str(e)[:160])}}

print("__RESULT__" + json.dumps(out, ensure_ascii=False))
'''


def run_scenario(scen: str, with_tenants: bool) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if with_tenants:
        env["KB_TENANTS"] = TENANTS_JSON
    else:
        env.pop("KB_TENANTS", None)
    code = PROBE.format(root=str(ROOT), scen=scen)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       cwd=str(ROOT))
    for line in (r.stdout or "").splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"scenario": scen, "ok": False,
            "detail": {"error": "no result", "stderr": (r.stderr or "")[-200:]}}


SCENARIOS = [
    ("A", False, "单租户 + 同步 /ask（基线）"),
    ("B", True, "多租户 + 同步 /ask（acme）"),
    ("C", True, "多租户 + SSE /ask/stream（acme）**关键：回归出在这里**"),
    ("D", True, "多租户 + 未带租户头（应 400）"),
    ("E", True, "多租户 + 未知租户（应 403）"),
]

print("=" * 100)
print("阶段 4 · 系统联动测试（多租户 × 入口 矩阵）")
print("=" * 100)
print("  KB_TENANTS = %s" % TENANTS_JSON)
print()

results = []
for scen, with_tenants, label in SCENARIOS:
    print("  [%s] 运行中 ..." % scen, flush=True)
    got = run_scenario(scen, with_tenants)
    results.append((scen, label, got))
    print("      -> %s" % ("OK" if got.get("ok") else "**FAIL**"))

print()
print("=" * 100)
print("结果明细")
print("=" * 100)
for scen, label, got in results:
    print()
    print("  [%s] %s" % ("OK  " if got.get("ok") else "FAIL", label))
    for k, v in (got.get("detail") or {}).items():
        print("        %-14s %s" % (k, str(v)[:110]))

n_ok = sum(1 for _s, _l, g in results if g.get("ok"))
print()
print("=" * 100)
print("合计 %d/%d 通过" % (n_ok, len(results)))
print("=" * 100)

(HERE / "integration_verify.json").write_text(json.dumps(
    [{"scenario": s, "label": l, "ok": g.get("ok"), "detail": g.get("detail")}
     for s, l, g in results], ensure_ascii=False, indent=2), encoding="utf-8")
print("  结果 -> eval/integration_verify.json")
