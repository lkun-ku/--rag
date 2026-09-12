"""用 **ModelScope** 下载 `bge-reranker-v2-m3`。

## 为什么换 ModelScope

HF 官方站在本网络不可达；`hf-mirror.com` 虽可连，但实测 **20 分钟只下了 300MB**
（约 15KB/s，2.3GB 要 2 小时+）。ModelScope（阿里）在国内通常快一个数量级。

下载后把路径打印出来（供 `measure_rerank_gain.py` 通过环境变量 `RERANK_MODEL` 指向）。
"""
from __future__ import annotations

import pathlib

CACHE = pathlib.Path.home() / ".cache" / "modelscope"
print("cache_dir = %s" % CACHE)
print("开始下载 BAAI/bge-reranker-v2-m3 ...")

from modelscope import snapshot_download  # noqa: E402

p = snapshot_download("BAAI/bge-reranker-v2-m3", cache_dir=str(CACHE))
print("OK -> %s" % p)

d = pathlib.Path(p)
files = sorted(d.rglob("*"))
tot = sum(f.stat().st_size for f in files if f.is_file())
print("文件数 %d，合计 %.1f MB" % (len([f for f in files if f.is_file()]), tot / 1024**2))
print()
print("**把下面这个路径填给 RERANK_MODEL 即可**：")
print(p)
