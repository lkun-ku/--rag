"""启动 RAG API 服务（ROADMAP 3.2）。

用法：
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py                 # 默认 127.0.0.1:8000
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py --host 0.0.0.0 --port 8080
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py --reload        # 开发模式热重载

为什么单独写这个脚本而不是让用户记 uvicorn 命令：
    1. 把 `--reload` 与模型预热的冲突提示内置（每次重载都会重新加载 1.3GB
       嵌入模型），避免误用；
    2. 给 uvicorn 传 `app_dir`，让热重载只监听包根目录，而不是整个仓库。

前置：包需先 `pip install -e .` 安装（见 pyproject.toml 的说明），
否则 `api.main` 与 `utils.*` / `graph2.*` 都 import 不到。
"""

from __future__ import annotations

import argparse
import pathlib

# uvicorn 的 `app_dir`（热重载监听目录）指向 RAG_PROJECT/（含 rag_kb/ 的那一层）；
# 导入解析已由 editable 安装负责
ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description="启动 RAG API 服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载（会反复加载模型，慎用）")
    args = ap.parse_args()

    import uvicorn

    if args.reload:
        print(
            "[提示] --reload 模式下每次改动都会重新加载嵌入模型（约 1.3GB），"
            "首次请求会明显变慢。生产/评测请去掉该参数。"
        )
    uvicorn.run("rag_kb.api.main:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
