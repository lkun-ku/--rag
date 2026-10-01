"""启动 RAG API 服务（ROADMAP 3.2）。

用法：
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py                 # 默认 127.0.0.1:8000
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py --host 0.0.0.0 --port 8080
    .venv/Scripts/python.exe RAG_PROJECT/api/run.py --reload        # 开发模式热重载

为什么单独写这个脚本而不是让用户记 uvicorn 命令：
    1. 需要把它所在目录的**上一级**（RAG_PROJECT/）加入 sys.path，
       否则 `api.main` 与 `utils.*` / `graph2.*` 都 import 不到；
    2. `--reload` 与模型预热不兼容（每次重载都会重新加载 1.3GB 嵌入模型），
       这里显式提示，避免误用。
"""

from __future__ import annotations

# [FIX] 允许「直接 python xxx.py」运行：包内绝对导入需要把 RAG_PROJECT/ 加入 sys.path。
# 用 `python -m` 运行时该路径已在 path 中，此段自动跳过（幂等）。
import pathlib as _pathlib
import sys as _sys

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
