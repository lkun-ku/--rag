"""兼容入口：保留 `python main.py <子命令>` 的用法。

CLI 实现位于 `rag_kb/cli/main.py`，安装后也可直接用 console script：
    rag-kb <子命令>
"""
import sys

from rag_kb.cli.main import main, run_with_project_venv

if __name__ == '__main__':
    try:
        rc = run_with_project_venv(sys.argv[1:])
        raise SystemExit(main() if rc is None else rc)
    except KeyboardInterrupt:
        print('\n已中断（Ctrl+C）')
        raise SystemExit(130) from None
