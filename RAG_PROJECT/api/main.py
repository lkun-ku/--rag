"""RAG 知识库 HTTP 服务（ROADMAP 3.2）。

## 端点

    GET  /health        健康检查（Milvus 连通性 + LLM 探活 + 集合清单）
    POST /ask           问答（同步返回答案 + 引用溯源）
    POST /ask/stream    问答（SSE 流式，逐节点推送进度）
    POST /ingest        按源目录入库（复用阶段一的 DOC_SOURCES 配置）

## 为什么用 FastAPI + 模块级单例

模型（bge-large 嵌入、CrossEncoder）与 Milvus 连接都是**重资源**，
若每次请求重建会有秒级开销。这里沿用项目既有的模块级单例风格
（`tools.retriever_tools.retriever`、`graph2.graph_2.graph` 都是 import 时构造），
配合 lifespan 预热，让**首个请求**也不必承担加载成本。

## 流式为什么用 SSE 而不是 WebSocket

`graph.stream()` 本身就是生成器、天然单向推送；SSE 在 FastAPI 里只需
`StreamingResponse` + `text/event-stream`，无需额外依赖与握手协议。
若将来需要双向（如多轮打断），再换 WebSocket。

启动：
    $env:PYTHONPATH="RAG_PROJECT"
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

    # 或用项目脚本
    .venv\\Scripts\\python.exe RAG_PROJECT/api/run.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

# 允许 `python RAG_PROJECT/api/main.py` 直接运行（正常应走 PYTHONPATH）
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from utils import tracer  # noqa: E402
from utils.citation import build_citation_meta, is_refusal, parse_citations  # noqa: E402
from utils.env_utils import DOC_SOURCES, MILVUS_URI, TENANTS  # noqa: E402
from utils.log_utils import log  # noqa: E402

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    with_trace: bool = Field(False, description="是否在响应中附带执行轨迹（调试用）")


class Citation(BaseModel):
    n: int = Field(..., description="答案中出现的引用编号")
    filename: str = ""
    title: str = ""
    chars: int = 0
    from_web: bool = Field(False, description="该来源是否来自网络搜索（而非知识库，见 P0-9）")


class AskResponse(BaseModel):
    answer: str
    refused: bool = Field(..., description="是否为设计内拒答（知识库无依据）")
    citations: List[Citation] = []
    elapsed_sec: float = 0.0
    tenant: Optional[str] = Field(None, description="本次请求生效的租户（单租户模式下为 null）")
    trace: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    source: str = Field(..., description="DOC_SOURCES 里的源名，如 milvus_docs / papers")
    reset: bool = Field(False, description="入库前是否清空该集合（危险，默认否）")


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG 企业知识库",
    version=APP_VERSION,
    description="多格式文档摄入 + 跨集合混合检索 + 引用溯源",
)


@app.on_event("startup")
def _warmup():
    """预热：import 时构造的重资源（嵌入模型 / Milvus 连接 / 工作流图）在此完成加载。

    `tools.retriever_tools` 与 `graph2.graph_2` 都在模块级构造单例，
    这里显式 import 一次即可触发 —— 目的是让**首个请求**不必等模型加载。
    """
    t0 = time.time()
    try:
        import tools.retriever_tools  # noqa: F401  触发检索器 + 嵌入模型加载
        import graph2.graph_2  # noqa: F401        触发工作流图编译
        log.info(f"[api] 预热完成，耗时 {time.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        # 预热失败不阻断启动：具体问题留给 /health 暴露，便于区分
        # 「服务起不来」与「依赖未就绪」两种情况。
        log.error(f"[api] 预热失败（服务仍启动，请查看 /health）: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查：分别探测各依赖，**不把失败糊成一个 false**。

    这样调用方能直接看出是哪一环断了（Milvus / LLM / 检索器 / 图）。
    """
    out: Dict[str, Any] = {"version": APP_VERSION, "ok": True, "checks": {}}

    # Milvus
    try:
        from pymilvus import MilvusClient

        cols = MilvusClient(uri=MILVUS_URI).list_collections()
        out["checks"]["milvus"] = {"ok": True, "uri": MILVUS_URI,
                                   "collections": sorted(cols)}
    except Exception as e:  # noqa: BLE001
        out["checks"]["milvus"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out["ok"] = False

    # 检索器（覆盖集合）
    try:
        from tools.retriever_tools import retriever

        covered = list(getattr(retriever, "retrievers", {}).keys())
        out["checks"]["retriever"] = {"ok": bool(covered), "collections": covered}
        if not covered:
            out["ok"] = False
    except Exception as e:  # noqa: BLE001
        out["checks"]["retriever"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out["ok"] = False

    # LLM（探活一次）
    try:
        from llm_models.all_llm import llm

        resp = llm.invoke("回复一个字：好")
        out["checks"]["llm"] = {"ok": True,
                                "reply": str(getattr(resp, "content", resp))[:20]}
    except Exception as e:  # noqa: BLE001
        # LLM 不可用时刻意保留 ok=False 但**不抛异常** —— 健康检查本身要能返回
        out["checks"]["llm"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out["ok"] = False

    # 多租户（只读配置，不触发检索器构造）
    out["checks"]["tenants"] = {
        "enabled": bool(TENANTS),
        "configured": {t: sorted(v or []) for t, v in TENANTS.items()} if TENANTS else {},
        "note": "未配置 KB_TENANTS -> 单租户模式（可访问全部集合）" if not TENANTS else "",
    }

    return out


# ---------------------------------------------------------------------------
# 多租户（ROADMAP 3.3）
# ---------------------------------------------------------------------------
def _resolve_tenant(x_tenant_id: Optional[str]) -> Optional[str]:
    """把 HTTP 头解析成租户 ID，并**在进入检索前**校验合法性。

    为什么在 API 层就校验而不是等检索时报错：
        越权应当快速失败（400/403），而不是让请求跑到检索器构造那里才抛异常。
        另外 `KB_TENANTS` 未配置时 header 一律忽略 —— 单租户模式不该被一个
        随手的 header 改变行为。
    """
    if not TENANTS:
        return None
    if not x_tenant_id:
        # 配置了多租户但请求未带租户头 -> 拒绝，而不是悄悄放行到全部集合
        raise HTTPException(status_code=400,
                            detail="服务已启用多租户，请在 X-Tenant-Id 头中指定租户")
    if x_tenant_id not in TENANTS:
        raise HTTPException(status_code=403,
                            detail="未知租户 %r" % x_tenant_id)
    return x_tenant_id


def _tenant_collections_of(tenant: Optional[str]):
    """给健康检查/调试用：列出某租户实际可访问的集合。"""
    if not tenant:
        return None
    try:
        from tools.retriever_tools import tenant_collections

        return sorted(tenant_collections(tenant))
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}


# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------
def _run_once(question: str, with_trace: bool) -> Dict[str, Any]:
    """跑一次完整工作流并组装响应（同步与流式共用）。"""
    from graph2.graph_2 import run_with_trace

    t0 = time.time()
    result = run_with_trace(question, save=False)
    docs = result.get("documents") or []
    answer = result.get("generation", "") or ""

    cites = parse_citations(answer)
    meta = build_citation_meta(docs, cites) if cites and docs else []

    payload: Dict[str, Any] = {
        "answer": answer,
        "refused": is_refusal(answer),
        "citations": [
            {"n": m.get("n"), "filename": m.get("filename", ""),
             "title": m.get("title", ""), "chars": m.get("chars", 0),
             "from_web": m.get("from_web", False)}
            for m in meta
        ],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if with_trace:
        payload["trace"] = result.get("trace")
    return payload


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest,
        x_tenant_id: Optional[str] = Header(None, description="租户 ID（多租户模式下必填）")
        ) -> Dict[str, Any]:
    """同步问答：返回答案 + 引用溯源（+ 可选轨迹）。

    多租户：请求头 `X-Tenant-Id` 指定租户，检索被限制在该租户授权的集合内。
    未配置 `KB_TENANTS` 时该头被忽略（单租户模式）。
    """
    from tools.retriever_tools import tenant_scope

    tenant = _resolve_tenant(x_tenant_id)
    try:
        with tenant_scope(tenant):
            out = _run_once(req.question, req.with_trace)
        out["tenant"] = tenant
        return out
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # 显式 500 而不是返回空答案 —— 见 ROADMAP 1.3.6「不得静默降级」
        log.error(f"[api] /ask 失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500,
                            detail=f"问答失败: {type(e).__name__}: {str(e)[:200]}") from e


@app.post("/ask/stream")
def ask_stream(req: AskRequest,
               x_tenant_id: Optional[str] = Header(None, description="租户 ID（多租户模式下必填）")):
    """SSE 流式问答：逐节点推送进度，最后推 event=answer。

    为什么值得流式：完整链路实测可达 42s（含 multi-round 的判分与改写），
    用户等待期间能看到「正在检索 / 正在判分 / 正在生成」比空等体验好得多。
    """
    from tools.retriever_tools import tenant_scope

    tenant = _resolve_tenant(x_tenant_id)

    def gen():
        from graph2.graph_2 import graph

        t = tracer.start(req.question, enabled=True)
        config = {"callbacks": t.callbacks} if t.callbacks else None
        final: Dict[str, Any] = {}
        try:
            # 租户上下文必须包住整个 stream —— retriever_node 在生成器内部执行，
            # 若只在外层设置、而生成器在别的上下文里迭代，会读不到租户。
            with tenant_scope(tenant):
                for output in graph.stream({"question": req.question}, config=config):
                    for node, value in output.items():
                        if isinstance(value, dict):
                            final.update(value)
                        yield "event: step\ndata: %s\n\n" % json.dumps(
                            {"node": node, "t": round(time.time() - t.t0, 2)},
                            ensure_ascii=False)

            answer = final.get("generation", "") or ""
            docs = final.get("documents") or []
            cites = parse_citations(answer)
            meta = build_citation_meta(docs, cites) if cites and docs else []
            yield "event: answer\ndata: %s\n\n" % json.dumps({
                "answer": answer,
                "refused": is_refusal(answer),
                "tenant": tenant,
                "citations": [{"n": m.get("n"), "filename": m.get("filename", ""),
                               "title": m.get("title", ""), "chars": m.get("chars", 0),
                               "from_web": m.get("from_web", False)}
                              for m in meta],
                "elapsed_sec": round(time.time() - t.t0, 2),
            }, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            yield "event: error\ndata: %s\n\n" % json.dumps(
                {"error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """按源入库。

    说明：入库是**长任务**（解析 + 向量化 + 写入，多格式语料可达分钟级），
    这里同步执行并返回统计 —— 适合本地/内网按需触发。
    若要支持并发长任务，应改为提交后台任务 + 轮询状态（未做，避免过早复杂化）。
    """
    key = req.source
    src = next((s for s in DOC_SOURCES if s.get("name") == key), None)
    if src is None:
        raise HTTPException(
            status_code=400,
            detail="未知源 %r；可用: %s" % (key, [s.get("name") for s in DOC_SOURCES]),
        )

    try:
        from documents.write_milvus import main as ingest_main

        # write_milvus.main(sources=<dict 列表>, batch_size, recreate)
        #
        # 注意 recreate 的语义：write_milvus 里 **默认 True**（全量重灌，会先删集合），
        # 而 HTTP 接口的 reset 默认 **False** —— 这是刻意的：入库是不可逆操作，
        # 默认必须是「增量追加」，清空集合要显式传 reset=true。
        stats = ingest_main(sources=[src], recreate=bool(req.reset))
        return {"ok": True, "source": key, "collection": src.get("collection"),
                "recreate": bool(req.reset), "stats": stats}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.error(f"[api] /ingest 失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500,
                            detail=f"入库失败: {type(e).__name__}: {str(e)[:200]}") from e


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "RAG 企业知识库", "version": APP_VERSION,
            "docs": "/docs", "endpoints": ["/health", "/ask", "/ask/stream", "/ingest"]}
