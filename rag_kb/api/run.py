"""启动 RAG API 服务（ROADMAP 3.2）。

用法：
    .venv/Scripts/python.exe rag_kb/api/run.py                 # 默认 127.0.0.1:8000
    .venv/Scripts/python.exe rag_kb/api/run.py --host 0.0.0.0 --port 8080
    .venv/Scripts/python.exe rag_kb/api/run.py --reload        # 开发模式热重载

为什么单独写这个脚本而不是让用户记 uvicorn 命令：
    1. 把 `--reload` 与模型预热的冲突提示内置（每次重载都会重新加载 1.3GB
       嵌入模型），避免误用；
    2. 给 uvicorn 传 `app_dir`，让热重载只监听包根目录，而不是整个仓库。

前置：包需先 `pip install -e .` 安装（见 pyproject.toml 的说明），
否则 `api.app` 与 `utils.*` / `graph_crag.*` 都 import 不到。
"""

from __future__ import annotations

import argparse
import pathlib

# uvicorn 的 `app_dir`（热重载监听目录）指向项目根（含 rag_kb/ 的那一层）；
# 导入解析已由 editable 安装负责
ROOT = pathlib.Path(__file__).resolve().parents[2]


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    embed_url: str | None = None,
) -> None:
    """启动 API 服务（供 `main.py serve` 与 `run.py` 的 CLI 共用）。"""
    import os

    import uvicorn

    if embed_url:
        # 必须在 uvicorn 导入 app 之前设置：worker 子进程会继承这份环境变量
        os.environ["EMBEDDING_BACKEND"] = "http"
        os.environ["EMBEDDING_URL"] = embed_url
        print(f"[配置] embedding 后端 = http（{embed_url}），本进程不加载本地模型")
    elif workers > 1:
        print(
            "[警告] --workers > 1 且未指定 --embed-url：每个 worker 都会各自加载一份 "
            "bge-large（1.24GB 显存 + 约 2s 启动），并且在同一块 GPU 上互相抢算力。\n"
            "        多 worker 的正确做法：\n"
            "          1) 先起共享服务: python main.py embed-server --port 8001\n"
            "          2) 再起主服务  : python main.py serve --workers 4 --embed-url http://127.0.0.1:8001"
        )

    if workers > 1 and reload:
        raise SystemExit("--reload 与 --workers 不能同时使用（uvicorn 的限制）")

    if reload:
        print(
            "[提示] --reload 模式下每次改动都会重新加载嵌入模型（约 1.3GB），"
            "首次请求会明显变慢。生产/评测请去掉该参数。"
        )
    uvicorn.run(
        "rag_kb.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        app_dir=str(ROOT),
    )


def main():
    ap = argparse.ArgumentParser(description="启动 RAG API 服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载（会反复加载模型，慎用）")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker 进程数（默认 1）。>1 时**必须**配合 --embed-url，否则每个 worker 各加载一份 1.24GB 模型",
    )
    ap.add_argument(
        "--embed-url",
        default=None,
        help="独立 embedding 服务地址（如 http://127.0.0.1:8001）。给了它才会启用 EMBEDDING_BACKEND=http",
    )
    args = ap.parse_args()
    serve(host=args.host, port=args.port, workers=args.workers, reload=args.reload, embed_url=args.embed_url)


if __name__ == "__main__":
    main()
