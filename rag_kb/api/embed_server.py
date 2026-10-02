"""独立的 embedding 服务：多 worker / 多服务共用**一份**模型。

启动：
    .venv\\Scripts\\python.exe -m rag_kb.api.embed_server            # 默认 127.0.0.1:8001
    .venv\\Scripts\\python.exe -m rag_kb.api.embed_server --port 8002

配套使用（主服务侧）：
    $env:EMBEDDING_BACKEND="http"; $env:EMBEDDING_URL="http://127.0.0.1:8001"
    .venv\\Scripts\\python.exe rag_kb/api/run.py --workers 4 --embed-url http://127.0.0.1:8001

为什么单独一个进程，而不是在主服务里加个 /embed 端点：
  * 主服务的每个 worker 都会 import 整个 app，模型会被加载 N 次；
  * 独立进程只有一份模型、一份显存，worker 数量可以自由扩；
  * 也让「扩 worker」与「换模型/换卡」解耦。

本模块启动时会**强制 local 后端**（否则 http 模式下会自己调自己，形成递归）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

# 必须在 import embeddings_model 之前设置：本进程必须使用本地模型
os.environ['EMBEDDING_BACKEND'] = 'local'

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from rag_kb.utils.log_utils import log  # noqa: E402

app = FastAPI(title='RAG embedding server', version='1.0')

_model = None


def _model_or_load():
    """惰性加载模型：`/health` 可以在不加载模型的情况下先回答（便于探活）。"""
    global _model
    if _model is None:
        from rag_kb.llm_models.embeddings_model import bge_embedding

        _model = bge_embedding
        log.info('embed server: 模型已加载')
    return _model


class DocumentsRequest(BaseModel):
    texts: List[str]


class QueryRequest(BaseModel):
    text: str


@app.get('/health')
def health() -> Dict[str, Any]:
    """健康检查：返回 pid（便于确认多 worker 是否真的共享了本服务）与设备信息。"""
    from rag_kb.llm_models.embeddings_model import EMBEDDING_INFO

    return {'ok': True, 'pid': os.getpid(), 'model_loaded': _model is not None, **EMBEDDING_INFO}


@app.post('/embed_documents')
def embed_documents(req: DocumentsRequest) -> Dict[str, Any]:
    if not req.texts:
        return {'vectors': []}
    try:
        return {'vectors': _model_or_load().embed_documents(req.texts)}
    except Exception as e:  # noqa: BLE001
        # 显式 500 + 原因：客户端会把状态码与消息一起抛出，便于定位
        raise HTTPException(status_code=500, detail=f'{type(e).__name__}: {str(e)[:300]}') from e


@app.post('/embed_query')
def embed_query(req: QueryRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail='text 不能为空')
    try:
        return {'vector': _model_or_load().embed_query(req.text)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'{type(e).__name__}: {str(e)[:300]}') from e


def serve(host: str = '127.0.0.1', port: int = 8001) -> None:
    """启动 embedding 服务（供 `main.py embed-server` 与本模块 CLI 共用）。"""
    import uvicorn

    log.info(f'embed server 启动: http://{host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level='info')


def main() -> None:
    ap = argparse.ArgumentParser(description='独立 embedding 服务（多 worker 部署时共用一份模型）')
    ap.add_argument('--host', default='127.0.0.1', help='监听地址（默认仅本机）')
    ap.add_argument('--port', type=int, default=8001)
    args = ap.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
