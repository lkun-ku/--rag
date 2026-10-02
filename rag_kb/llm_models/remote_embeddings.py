"""远程 embedding 客户端：多 worker / 多服务共用一个模型进程。

## 为什么需要

`bge-large-zh-v1.5` 权重 1.24GB。若每个 uvicorn worker 各自加载一份：
  * 4 worker = 4 份显存（RTX 4060 Laptop 只有 8GB，两份就吃掉一半）；
  * 每个 worker 启动都要 +2s（GPU）/+13s（CPU）；
  * 同一块 GPU 上多个进程抢算力，吞吐并不随 worker 数线性增长。

所以多 worker 部署时应把模型放进**一个独立进程**（`rag_kb.api.embed_server`），
其它进程用本模块通过 HTTP 调用。单 worker 部署仍可用本地模型（默认行为）。

## 契约

  * `embed_documents(texts)` -> 与输入**等长同序**的向量列表；内部按 batch_size 分批。
  * `embed_query(text)` -> 单个向量。
  * 任何失败（连不上 / 非 200）**一律抛 RuntimeError**，不让 httpx 的异常类型泄漏给调用方 ——
    调用方只需要处理一种异常，且消息里必须带可定位的信息（URL / 状态码）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx
from langchain_core.embeddings import Embeddings

from rag_kb.utils.log_utils import log

# 与 embed_server 的路径约定（改一处必须同步另一处）
PATH_DOCUMENTS = '/embed_documents'
PATH_QUERY = '/embed_query'


class RemoteEmbeddings(Embeddings):
    """把 embedding 计算转发给独立的 embed 服务（见 `rag_kb.api.embed_server`）。"""

    def __init__(
        self,
        base_url: str,
        batch_size: int = 16,
        timeout: float = 120.0,
        retries: int = 1,
    ) -> None:
        """
        Args:
            base_url: embed 服务地址，例如 ``http://127.0.0.1:8001``。
            batch_size: 每次 HTTP 请求携带多少条文本（服务端仍会自己再分批）。
            timeout: 单次请求超时（秒）。首次请求可能包含 CUDA 预热，别设太小。
            retries: 连接类失败的重试次数（状态码错误不重试 —— 那是服务端逻辑问题）。
        """
        self.base_url = base_url.rstrip('/')
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self.retries = max(0, int(retries))
        # 本机地址必须绕开系统代理，否则会被转发成 404（见 utils/http_local.py）
        from rag_kb.utils.http_local import make_client

        self._client = make_client(self.base_url, timeout=timeout)

    # ---- Embeddings 接口 ---------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            payload = self._post(PATH_DOCUMENTS, {'texts': chunk})
            got = payload.get('vectors') or []
            if len(got) != len(chunk):
                raise RuntimeError(
                    f'embed 服务返回条数不符：请求 {len(chunk)} 条，返回 {len(got)} 条（{self.base_url}）'
                )
            vectors.extend(got)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        payload = self._post(PATH_QUERY, {'text': text})
        vec = payload.get('vector')
        if not vec:
            raise RuntimeError(f'embed 服务未返回向量（{self.base_url}）')
        return vec

    # ---- 内部 --------------------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.post(url, json=payload)
            except httpx.HTTPError as e:  # 连接 / 超时 / 协议错误
                last_error = e
                if attempt < self.retries:
                    log.warning(f'embed 请求失败，重试 {attempt + 1}/{self.retries}：{type(e).__name__}: {e}')
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
            if resp.status_code != 200:
                # 4xx/5xx 不重试：重试不会改变服务端的逻辑错误，只会拖长故障时间
                raise RuntimeError(f'embed 服务返回 {resp.status_code}（{url}）：{resp.text[:200]}')
            return resp.json()
        raise RuntimeError(
            f'连不上 embed 服务 {url}（{type(last_error).__name__}: {last_error}）。'
            f'请确认 rag_kb.api.embed_server 已启动，或改用 EMBEDDING_BACKEND=local'
        )

    def close(self) -> None:
        self._client.close()


def build_remote_embeddings(base_url: str, batch_size: int = 16) -> RemoteEmbeddings:
    """按 `EMBEDDING_URL` 构造远程客户端（供 embeddings_model 的工厂调用）。"""
    return RemoteEmbeddings(base_url=base_url, batch_size=batch_size)
