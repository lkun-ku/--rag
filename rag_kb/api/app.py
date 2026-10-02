"""RAG 知识库 HTTP 服务（ROADMAP 3.2）。

## 端点

    GET  /health        健康检查（Milvus 连通性 + LLM 探活 + 集合清单）
    POST /ask           问答（同步返回答案 + 引用溯源）
    POST /ask/stream    问答（SSE 流式，逐节点推送进度）
    POST /ingest        按源目录入库（复用阶段一的 DOC_SOURCES 配置）

## 为什么用 FastAPI + 模块级单例

模型（bge-large 嵌入、CrossEncoder）与 Milvus 连接都是**重资源**，
若每次请求重建会有秒级开销。这里沿用项目既有的模块级单例风格
（`tools.retriever_tools.retriever`、`graph_crag.crag_graph.graph` 都是 import 时构造），
配合 lifespan 预热，让**首个请求**也不必承担加载成本。

## 流式为什么用 SSE 而不是 WebSocket

`graph.stream()` 本身就是生成器、天然单向推送；SSE 在 FastAPI 里只需
`StreamingResponse` + `text/event-stream`，无需额外依赖与握手协议。
若将来需要双向（如多轮打断），再换 WebSocket。

启动（前置：`pip install -e .`，见 pyproject.toml 的说明）：
    .venv\\Scripts\\python.exe -m uvicorn rag_kb.api.app:app --host 127.0.0.1 --port 8000

    # 或用项目脚本
    .venv\\Scripts\\python.exe rag_kb/api/run.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_kb.utils import tracer
from rag_kb.utils.citation import build_citation_meta, is_refusal, parse_citations
from rag_kb.utils.env_utils import DOC_SOURCES, MILVUS_URI, TENANTS
from rag_kb.utils.log_utils import log

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    with_trace: bool = Field(False, description="是否在响应中附带执行轨迹（调试用）")
    # [ROADMAP 4.41] 多轮会话标识。**不传 = 单轮（行为与改造前完全一致）**；
    # 传同一个值 -> 该会话的问答历史会被持久化并在后续轮次注入 prompt
    # （例如第二轮问「它的 QPS 是多少」，模型能知道「它」指上一轮的产品）。
    session_id: Optional[str] = Field(None, max_length=128, description="多轮会话 ID；不传则按单轮处理（无记忆）")


class Citation(BaseModel):
    n: int = Field(..., description="答案中出现的引用编号")
    filename: str = ""
    title: str = ""
    chars: int = 0
    from_web: bool = Field(False, description="该来源是否来自网络搜索（而非知识库，见 P0-9）")
    url: str = Field("", description="联网来源的原网页地址；知识库来源为空")


class AskResponse(BaseModel):
    answer: str
    refused: bool = Field(..., description="是否为设计内拒答（知识库无依据）")
    grounded: bool = Field(
        False,
        description="答案是否有知识库/联网依据；False 表示**降级作答**（通用知识，已在 answer 抬头标注）",
    )
    source_type: str = Field(
        "unknown",
        description="knowledge_base=知识库 / web=联网 / model_knowledge=检索失败降级 / model_direct=主动闲聊",
    )
    citations: List[Citation] = []
    elapsed_sec: float = 0.0
    tenant: Optional[str] = Field(None, description="本次请求生效的租户（单租户模式下为 null）")
    trace: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    source: str = Field(..., description="DOC_SOURCES 里的源名，即 datas/ 下的目录名，如 前端复习笔记")
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

    `tools.retriever_tools` 与 `graph_crag.crag_graph` 都在模块级构造单例，
    这里显式 import 一次即可触发 —— 目的是让**首个请求**不必等模型加载。
    """
    t0 = time.time()
    try:
        import rag_kb.tools.retriever_tools  # noqa: F401  触发检索器 + 嵌入模型加载
        import rag_kb.graph_crag.crag_graph  # noqa: F401        触发工作流图编译

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
        out["checks"]["milvus"] = {"ok": True, "uri": MILVUS_URI, "collections": sorted(cols)}
    except Exception as e:  # noqa: BLE001
        out["checks"]["milvus"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out["ok"] = False

    # 嵌入设备（**必须暴露**：换 GPU 后最大的坑是静默回落 CPU，性能掉约 40x 却无信号）
    try:
        from rag_kb.llm_models.embeddings_model import EMBEDDING_INFO

        # pid 放进健康检查：多 worker 部署时，调用方可以直接确认「每个请求落到了哪个进程」
        out["checks"]["embedding"] = {"ok": True, "pid": os.getpid(), **EMBEDDING_INFO}

        if EMBEDDING_INFO.get("backend") == "http":
            # 远程后端：探活共享的 embed 服务（它才是真正持有模型/显存的进程）
            url = str(EMBEDDING_INFO.get("url") or "")
            try:
                from rag_kb.utils.http_local import make_client

                with make_client(url, timeout=2.0) as client:
                    resp = client.get(url + "/health")
                server = resp.json() if resp.status_code == 200 else {"ok": False, "status": resp.status_code}
                out["checks"]["embedding"]["server"] = server
                if not server.get("ok"):
                    out["ok"] = False
            except Exception as e:  # noqa: BLE001
                out["checks"]["embedding"]["server"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
                out["ok"] = False
        elif EMBEDDING_INFO.get("cuda_available") and EMBEDDING_INFO.get("device") != "cuda":
            # CUDA 明明可用却在跑 CPU，说明配置写成了 cpu（或显式要求了 cpu）——提示但不判失败
            out["checks"]["embedding"]["note"] = "CUDA 可用但当前使用 CPU（检查 EMBEDDING_DEVICE）"
    except Exception as e:  # noqa: BLE001
        import traceback

        out["checks"]["embedding"] = {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)[:160]}",
            "traceback": traceback.format_exc()[-500:],
        }
        out["ok"] = False

    # 检索器（覆盖集合）
    try:
        from rag_kb.tools.retriever_tools import retriever

        covered = list(getattr(retriever, "retrievers", {}).keys())
        out["checks"]["retriever"] = {"ok": bool(covered), "collections": covered}
        if not covered:
            out["ok"] = False
    except Exception as e:  # noqa: BLE001
        out["checks"]["retriever"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out["ok"] = False

    # LLM（探活一次）
    try:
        from rag_kb.llm_models.all_llm import llm

        resp = llm.invoke("回复一个字：好")
        out["checks"]["llm"] = {"ok": True, "reply": str(getattr(resp, "content", resp))[:20]}
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
        raise HTTPException(status_code=400, detail="服务已启用多租户，请在 X-Tenant-Id 头中指定租户")
    if x_tenant_id not in TENANTS:
        raise HTTPException(status_code=403, detail="未知租户 %r" % x_tenant_id)
    return x_tenant_id


def _tenant_collections_of(tenant: Optional[str]):
    """给健康检查/调试用：列出某租户实际可访问的集合。"""
    if not tenant:
        return None
    try:
        from rag_kb.tools.retriever_tools import tenant_collections

        return sorted(tenant_collections(tenant))
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}


# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 会话管理（ROADMAP 4.45）：新开对话 / 列表 / 删除
# ---------------------------------------------------------------------------
class NewSessionRequest(BaseModel):
    title: str = Field("", max_length=200, description="会话标题（空则用首条问题填充）")


@app.post("/sessions/new")
def sessions_new(req: NewSessionRequest) -> Dict[str, Any]:
    """**新开对话**：服务端生成 session_id 并建档。

    之后 `/ask` 带这个 `session_id` 即可获得多轮记忆；
    `/ask/stream` 同理。
    """
    from rag_kb.utils import session_store

    sid = session_store.create_session(req.title)
    return {"session_id": sid, "ttl_seconds": session_store.SESSION_TTL_SECONDS}


@app.get("/sessions")
def sessions_list() -> Dict[str, Any]:
    """会话列表（按最后活跃倒序）。"""
    from rag_kb.utils import session_store

    items = session_store.list_sessions()
    return {"count": len(items), "sessions": items}


@app.delete("/sessions/{session_id}")
def sessions_delete(session_id: str) -> Dict[str, Any]:
    """删除会话：历史向量块 + LangGraph 检查点 + 元数据，三处一起清。"""
    from rag_kb.utils import session_store

    ok = session_store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或删除失败")
    return {"deleted": session_id}


@app.on_event("startup")
def _start_session_janitor():
    """启动会话清理线程（ROADMAP 4.45）。

    [取舍] 用 `@app.on_event("startup")` 而非新的 `lifespan` API：
    后者要改 app 构造签名，影响面更大；`on_event` 虽在新版 FastAPI 标记 deprecated，
    但语义稳定、且本项目是**单进程**部署（uvicorn 单 worker），
    不存在「多 worker 各起一份」的额外问题。

    线程设为 daemon：服务退出时随主进程结束，不阻塞 shutdown。
    """
    import threading

    from rag_kb.utils import session_store

    t = threading.Thread(target=session_store.janitor_loop, name="session-janitor", daemon=True)
    t.start()
    log.info(
        "[api] 会话清理线程已启动（TTL=%ds，间隔=%ds）"
        % (session_store.SESSION_TTL_SECONDS, session_store.SESSION_JANITOR_INTERVAL)
    )


def _run_once(question: str, with_trace: bool, session_id: Optional[str] = None) -> Dict[str, Any]:
    """跑一次完整工作流并组装响应（同步接口）。

    [ROADMAP 4.41] `session_id` 透传给 `run_with_trace(thread_id=...)`：
    None -> 无记忆（原行为）；有值 -> 该会话历史持久化。
    """
    from rag_kb.graph_crag.crag_graph import run_with_trace

    t0 = time.time()
    result = run_with_trace(question, save=False, thread_id=session_id)
    docs = result.get("documents") or []
    answer = result.get("generation", "") or ""

    cites = parse_citations(answer)
    meta = build_citation_meta(docs, cites) if cites and docs else []

    payload: Dict[str, Any] = {
        "answer": answer,
        "refused": is_refusal(answer),
        # [2026-11] 直接透传 meta：原先手工投影成 5 个字段，把 `url` 丢了，
        # 导致联网来源在 API 里也没有可点开的地址（与 main.py 是同一个漏）。
        "citations": meta,
        # [2026-11] 三态来源：knowledge_base / web / model_knowledge / model_direct
        "grounded": result.get("grounded", False),
        "source_type": result.get("source_type", "unknown"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if with_trace:
        payload["trace"] = result.get("trace")
    return payload


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest, x_tenant_id: Optional[str] = Header(None, description="租户 ID（多租户模式下必填）")
) -> Dict[str, Any]:
    """同步问答：返回答案 + 引用溯源（+ 可选轨迹）。

    多租户：请求头 `X-Tenant-Id` 指定租户，检索被限制在该租户授权的集合内。
    未配置 `KB_TENANTS` 时该头被忽略（单租户模式）。
    """
    from rag_kb.tools.retriever_tools import tenant_scope

    tenant = _resolve_tenant(x_tenant_id)
    try:
        with tenant_scope(tenant):
            out = _run_once(req.question, req.with_trace, req.session_id)
        out["tenant"] = tenant
        return out
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # 显式 500 而不是返回空答案 —— 见 ROADMAP 1.3.6「不得静默降级」
        log.error(f"[api] /ask 失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {type(e).__name__}: {str(e)[:200]}") from e


@app.post("/ask/stream")
def ask_stream(req: AskRequest, x_tenant_id: Optional[str] = Header(None, description="租户 ID（多租户模式下必填）")):
    """SSE 流式问答：逐节点推送进度，最后推 event=answer。

    为什么值得流式：完整链路实测可达 42s（含 multi-round 的判分与改写），
    用户等待期间能看到「正在检索 / 正在判分 / 正在生成」比空等体验好得多。
    """
    # [FIX  · 严重] SSE 分支**不能用** `tenant_scope()`（见 gen() 内注释），
    # 改用 `set_tenant`。
    from rag_kb.tools.retriever_tools import set_tenant

    tenant = _resolve_tenant(x_tenant_id)

    def gen():
        from rag_kb.graph_crag.crag_graph import get_memory_graph, graph

        t = tracer.start(req.question, enabled=True)
        config: Optional[Dict[str, Any]] = {"callbacks": t.callbacks} if t.callbacks else {}
        final: Dict[str, Any] = {}

        # [ROADMAP 4.41] 流式同样支持多轮：传了 session_id 就走带记忆的图。
        # 与 `run_with_trace` 的逻辑保持一致（同一 thread_id 才共享历史）。
        if req.session_id:
            from rag_kb.utils import session_store

            g = get_memory_graph()
            # [FIX / 文档审计] 必须同时注入 `session_id` —— `generate` 用它判断
            # 「是否处于记忆模式」（`has_memory = bool(state.get("session_id"))`）。
            # 首版只传了 `messages`，导致流式路径下三层记忆**完全不生效**
            # （既不注入历史，也不写历史）。与 `/ask` 保持一致。
            session_store.ensure_session(req.session_id, req.question[:200])
            inputs: Dict[str, Any] = {"question": req.question, "messages": [], "session_id": req.session_id}
            config = {**(config or {}), "configurable": {"thread_id": req.session_id}}
        else:
            g = graph
            inputs = {"question": req.question}
            config = config or None

        # [FIX  · 严重回归] **不能用 `tenant_scope()` 的 token-reset**。
        #
        # 根因：SSE 的 `gen()` 是**生成器**，由 FastAPI 在**另一个上下文**
        # （AnyIO worker thread）里迭代；而 `ContextVar.reset(token)` 要求
        # token 与 reset 在**同一 Context** 中，否则抛
        #     ValueError: Token ... was created in a different Context
        # 实测后果：SSE 直接发 `event: error`，**连 answer 事件都拿不到**，SSE 形同废弃。
        #
        # 这个 bug 是 3.3 引入多租户时植入的（当时的 `verify_api.py` 是在 3.3 **之前**
        # 跑的，之后 SSE 再没被测过 —— 直到本次 P1-2 修复顺带暴露）。
        #
        # 改用「set + finally 归零」：只设值、不碰 token，因此与上下文无关。
        # `retriever_node` 每次迭代都在同一上下文读取该值，语义等价。
        set_tenant(tenant)
        try:
            for output in g.stream(inputs, config=config):
                for node, value in output.items():
                    if isinstance(value, dict):
                        final.update(value)
                    yield "event: step\ndata: %s\n\n" % json.dumps(
                        {"node": node, "t": round(time.time() - t.t0, 2)}, ensure_ascii=False
                    )

            answer = final.get("generation", "") or ""

            # [FIX / P1-2] **落盘 trace**。
            # 原先 SSE 分支起了 tracer 却从不 save —— 结果是**走 SSE 的请求在
            # logs/trace.jsonl 里完全不存在**。而 SSE 是前端的默认路径，
            # 恰恰是最需要可观测的那条：它耗时最长（实测可达 42s），
            # 也最容易暴露多轮循环问题（见 3.1.3）。
            # run_with_trace() 有这两行，这里当初漏了。
            t.result = answer
            t.save()

            docs = final.get("documents") or []
            cites = parse_citations(answer)
            meta = build_citation_meta(docs, cites) if cites and docs else []
            yield "event: answer\ndata: %s\n\n" % json.dumps(
                {
                    "answer": answer,
                    "refused": is_refusal(answer),
                    "tenant": tenant,
                    "citations": [
                        {
                            "n": m.get("n"),
                            "filename": m.get("filename", ""),
                            "title": m.get("title", ""),
                            "chars": m.get("chars", 0),
                            "from_web": m.get("from_web", False),
                        }
                        for m in meta
                    ],
                    "elapsed_sec": round(time.time() - t.t0, 2),
                },
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001
            yield "event: error\ndata: %s\n\n" % json.dumps(
                {"error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False
            )
        finally:
            # 归零租户。用 `set(None)` 而不是 token.reset()（跨上下文会抛，见上）。
            # 不归零的后果很严重：同一 worker 的**后续请求会继承上一个租户** ——
            # 那是越权风险（租户 A 的查询会被限制到租户 B 的集合上）。
            set_tenant(None)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """按源入库。

    说明：入库是**长任务**（解析 + 向量化 + 写入，多格式语料可达分钟级），
    这里同步执行并返回统计 —— 适合本地/内网按需触发。
    若要支持并发长任务，应改为提交后台任务 + 轮询状态（未做，避免过早复杂化）。

    响应语义（两个字段各管一件事，别混着看）：
      * ``ok``      —— 请求**是否被受理并执行完**（配置错误/未知源/禁用源会直接 400/500）；
      * ``partial`` —— 执行过程中**是否出现了失败或跳过**（errors / skipped / aborted）。
        `ok=true, partial=true` 表示「跑完了，但没全部入库」，调用方必须看 ``stats``。
        加这个字段而不是改 ``ok``，是为了不破坏既有调用方（实测「解析器配置错」
        和「目录为空」两种情况下 ok 都是 true，只有 stats 里能看出没入库）。
    """
    key = req.source
    src = next((s for s in DOC_SOURCES if s.get("name") == key), None)
    if src is None:
        raise HTTPException(
            status_code=400,
            detail="未知源 %r；可用: %s" % (key, [s.get("name") for s in DOC_SOURCES]),
        )
    if not src.get("enabled", True):
        # [FIX / P1-3] 原先对 `enabled: False` 的源**也返回 200 + ok=True**：
        # `write_milvus` 的入口内部会再过滤一次 `enabled`，于是 stats 是空列表，
        # 但 HTTP 仍报成功 —— 调用方会**误以为已经入库**（这类「假成功」最难排查）。
        # 这里显式拒绝并说明原因。
        raise HTTPException(
            status_code=400,
            detail="源 %r 在 DOC_SOURCES 中已禁用（enabled=False），不会入库；如需启用请改配置并重启服务" % key,
        )

    try:
        # 调 ingest_many（纯函数）而不是 main()：main 是 CLI 入口，内部会 print 汇总表，
        # 在服务进程里会把表格打到 stdout。响应结构不变（to_dict 与旧 stats dict 字段一致）。
        from rag_kb.documents.write_milvus import ingest_many

        # 注意 recreate 的语义：write_milvus 里 **默认 True**（全量重灌，会先删集合），
        # 而 HTTP 接口的 reset 默认 **False** —— 这是刻意的：入库是不可逆操作，
        # 默认必须是「增量追加」，清空集合要显式传 reset=true。
        stats = [s.to_dict() for s in ingest_many(sources=[src], recreate=bool(req.reset))]
        # 任何失败 / 跳过 / 熔断都算「部分成功」——这是原先只返回 ok=True 时丢失的信息
        partial = any(s.get("errors") or s.get("skipped") or s.get("aborted") for s in stats)
        return {
            "ok": True,
            "partial": partial,
            "source": key,
            "collection": src.get("collection"),
            "recreate": bool(req.reset),
            "stats": stats,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.error(f"[api] /ingest 失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败: {type(e).__name__}: {str(e)[:200]}") from e


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "RAG 企业知识库",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/ask", "/ask/stream", "/ingest"],
    }
