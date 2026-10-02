"""本地 HTTP 调用：对 loopback 地址**绕过系统代理**。

为什么需要：httpx 默认 `trust_env=True`，在 Windows 上会从注册表读取系统代理。
本机实测代理指向 `127.0.0.1:26561`，于是 `POST http://127.0.0.1:<port>/ask`
被代理转发并返回 **404** —— 服务明明在跑，却表现为「路由不存在」。

规则：目标是 127.0.0.1 / localhost / ::1 时一律 `trust_env=False`；
真正的远程地址仍沿用系统代理（走公司网络代理时才有意义）。
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

LOOPBACK = {'127.0.0.1', 'localhost', '::1', '[::1]'}


def is_loopback(url: str) -> bool:
    """判断 URL 是否指向本机。"""
    return (urlparse(url).hostname or '').lower() in LOOPBACK


def host_of(url: str) -> str:
    """取 URL 的主机名（解析失败返回空串）。"""
    return (urlparse(url).hostname or '').lower()


def make_client(url: str, timeout: float = 60.0) -> httpx.Client:
    """按目标地址创建 httpx.Client：本机不走代理，远程走系统代理。"""
    return httpx.Client(timeout=timeout, trust_env=not is_loopback(url))


def bypass_proxy(*hosts: str) -> None:
    """把指定主机加入 NO_PROXY / no_proxy（幂等），让这些域名的请求走直连。

    背景（实测）：Windows 的系统代理（本机 127.0.0.1:26561）会对 HTTPS 做 TLS 中间人，
    而 Python 侧用的是 certifi 证书库、不认代理的根证书 —— 表现为
    ``SSLCertVerificationError: unable to get local issuer certificate``，
    LLM 调用和 Tavily 搜索都会失败；把这些域名从代理排除后立刻正常。
    如果你的环境**必须**走公司代理才能出网，设 ``KB_KEEP_PROXY=1`` 关闭该行为。
    """
    if os.getenv('KB_KEEP_PROXY') == '1':
        return
    wanted = [h for h in hosts if h]
    if not wanted:
        return
    for key in ('NO_PROXY', 'no_proxy'):
        current = [p for p in os.environ.get(key, '').split(',') if p]
        missing = [h for h in wanted if h not in current]
        if missing:
            os.environ[key] = ','.join(current + missing)
