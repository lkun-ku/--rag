r"""RAG 企业知识库 —— 统一入口（实现；根目录 main.py 为兼容 shim）。

不带子命令 = 打印本帮助（不加载模型）。
────────────────────────────────────────────────────────────────────────────
 ① 环境自检（换环境 / 出问题时先跑这个）
────────────────────────────────────────────────────────────────────────────
    python main.py doctor                     # Python / GPU / embedding / Milvus / 清单

────────────────────────────────────────────────────────────────────────────
 ② 日常问答
────────────────────────────────────────────────────────────────────────────
    python main.py chat                        # 终端连续问答（闲聊/追问/查库自动分流）
    python main.py ask "你的问题"               # 问一句，返回答案 + 引用
    python main.py serve                       # 启动 HTTP API（127.0.0.1:8000）
    python main.py serve --workers 4 --embed-url http://127.0.0.1:8001
    python main.py embed-server --port 8001    # 多 worker 共用的 embedding 服务

────────────────────────────────────────────────────────────────────────────
 ③ 换文档 / 加文档（三步，全程不改代码）
────────────────────────────────────────────────────────────────────────────
    # 第 1 步：把文档放进 datas/ 下的一个**主题目录**里。
    #   目录名随意（它就是源名），md/pdf/docx/xlsx/pptx/html/txt 可混放，支持子目录：
    #       datas/产品手册/
    #       datas/docs/
    python main.py ingest --list               # 确认识别到多少文件
    python main.py ingest                      # 入库（--no-recreate 为增量，只处理变化文件）
    python main.py ingest --sources docs  # 只处理指定源

    # 第 2 步：导出块清单（评测标注的依据）
    python eval\export_chunks.py

    # 第 3 步：自动出题（LLM 从语料生成问题 + 要点，写入 golden_auto.jsonl）
    python eval\gen_testset.py --per-collection 5

────────────────────────────────────────────────────────────────────────────
 ④ 质量评测
────────────────────────────────────────────────────────────────────────────
    python eval\run_eval.py --no-save          # 检索层（0 次 LLM，秒级；默认用自动题集）
    python eval\run_eval.py --with-generation --no-save   # 加生成层（回答 + 两道判分，需 LLM）
    python tests\run_all_checks.py             # 全量回归（23 项，含记忆/判分/冒烟）

    # 产物：baseline_auto.jsonl（本次分数）/ history_auto.jsonl（追加历史）
    # 阈值在 eval\thresholds.json —— 换语料后按新基线调整，不改代码

────────────────────────────────────────────────────────────────────────────
 ⑤ 说明
────────────────────────────────────────────────────────────────────────────
  * 裸 `python` 也能跑：检测到不是项目 venv 会自动切换（KB_NO_REEXEC=1 关闭）；
    eval 工具同理（EVAL_NO_REEXEC=1 关闭）。
  * `--no-save` / `--limit N` / 门禁 INCONCLUSIVE 时：结果**不落盘**，
    避免残缺数据覆盖正式基线。
  * 原有入口仍可用：`rag_kb/api/run.py`、`python -m rag_kb.doctor`、
    `python -m rag_kb.documents.write_milvus` —— 行为不变。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

# 项目根：editable 安装时 = 仓库根（rag_kb/cli/main.py -> parents[2]）；
# 非 editable 安装（site-packages）时不存在「项目根」，退回当前工作目录。
ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (ROOT / 'pyproject.toml').is_file():
    ROOT = pathlib.Path.cwd()

VENV_PYTHON = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

# 运行期真正需要的包（`version` 不需要）。用于在「用错解释器」时给出可执行的提示，
# 而不是甩一段 ModuleNotFoundError 的 traceback。
REQUIRED_DEPS = (
    'torch',
    'transformers',
    'sentence_transformers',
    'pymilvus',
    'langchain_huggingface',
    'fastapi',
    'uvicorn',
    'httpx',
)


def missing_deps(deps: tuple[str, ...] = REQUIRED_DEPS) -> list[str]:
    """当前解释器缺哪些关键依赖（纯函数，便于测试）。"""
    import importlib.util

    return [name for name in deps if importlib.util.find_spec(name) is None]


def run_with_project_venv(argv: list[str]) -> int | None:
    """若当前不是项目 venv 且有 .venv，则用 venv 的解释器**子进程**重跑本入口。

    为什么需要：Windows 上 `python` 常常指向 Anaconda/系统解释器，那里没有本项目依赖，
    直接跑会得到一串 ModuleNotFoundError（实测踩过）。设置 ``KB_NO_REEXEC=1`` 可关闭。

    为什么用 subprocess 而不是 `os.execv`：execv 是「原地替换进程」，在 Windows 上
    PowerShell 会先认为命令结束并打印提示符，而替换后的进程仍在往同一控制台写 ——
    结果就是**提示符与输出交错、命令行飘到奇怪的位置**（实测踩过）。
    改为子进程后，父进程会一直等到它结束，shell 的作业控制与输出顺序都正常。

    Returns:
        子进程退出码；无需切换时返回 None（调用方继续用当前进程执行）。
    """
    if os.environ.get('KB_NO_REEXEC') == '1' or not VENV_PYTHON.exists():
        return None
    try:
        same = pathlib.Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except OSError:
        same = False
    if same:
        return None
    print(f'[提示] 当前解释器不是项目 venv，改用：{VENV_PYTHON}')
    env = {**os.environ, 'KB_NO_REEXEC': '1'}  # 防止递归
    try:
        return subprocess.call(
            [str(VENV_PYTHON), '-m', 'rag_kb.cli.main', *argv], env=env, cwd=str(ROOT)
        )
    except KeyboardInterrupt:
        # Ctrl+C 会同时送达父子进程：子进程（uvicorn/chat）自己会优雅收尾，
        # 父进程不要再抛一段 traceback
        return 130


def _dep_error(missing: list[str]) -> str:
    venv_cmd = f'{VENV_PYTHON}'
    return (
        f'[错误] 当前 Python 解释器缺少依赖：{", ".join(missing)}\n'
        f'       解释器：{sys.executable}\n'
        f'       这通常是「没有用项目 venv 运行」导致的。两种解决方式：\n'
        f'         1) 用项目环境：{venv_cmd} main.py <子命令>\n'
        f'         2) 安装依赖  ：{venv_cmd} -m pip install -r requirements.txt\n'
        f'       （若确认要用当前解释器，设置环境变量 KB_NO_REEXEC=1 后重试）'
    )


# ---------------------------------------------------------------------------
# 子命令实现（重活都在函数内 import，保证 --help 与 doctor 不被模型加载拖慢）
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    from rag_kb.doctor import collect, format_report
    import json
    import os

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    report = collect(include_llm=args.with_llm)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0 if report['ok'] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from rag_kb.api.run import serve

    serve(host=args.host, port=args.port, workers=args.workers, reload=args.reload, embed_url=args.embed_url)
    return 0


def cmd_embed_server(args: argparse.Namespace) -> int:
    from rag_kb.api.embed_server import serve

    serve(host=args.host, port=args.port)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag_kb.utils.env_utils import DOC_SOURCES

    if args.list:
        # 只读操作走轻量模块：不 import write_milvus（否则会白加载 1.24GB 模型）
        from rag_kb.utils.file_scan import format_source_table

        print(format_source_table(DOC_SOURCES))
        return 0

    from rag_kb.documents import write_milvus as wm

    stats = wm.main(
        sources=wm._pick_sources(args.sources),
        batch_size=args.batch_size,
        recreate=not args.no_recreate,
    )
    # 有失败/熔断就返回非 0，便于 CI 或脚本判断（原 main() 只打印不返回状态）
    bad = sum(len(s.get('errors', [])) for s in stats) + sum(1 for s in stats if s.get('aborted'))
    return 1 if bad else 0


def _ask_once(question: str, url: str | None = None, session_id: str | None = None, trace: bool = False) -> dict:
    """问一次，返回 `{answer, refused, citations, elapsed_sec}`。

    ``url`` 为空时在**当前进程**里跑完整工作流（会加载模型）；给了 url 则复用已启动的服务
    （省掉 ~30s 冷启动，也不额外占一份显存）。
    """
    import time

    t0 = time.time()
    if url:
        from rag_kb.utils.http_local import make_client

        try:
            with make_client(url, timeout=600) as client:
                resp = client.post(
                    url.rstrip('/') + '/ask',
                    json={'question': question, 'session_id': session_id, 'with_trace': trace},
                )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f'连不上服务 {url}（{type(e).__name__}: {e}）') from e
        if resp.status_code != 200:
            raise RuntimeError(f'服务返回 {resp.status_code}: {resp.text[:200]}')
        return resp.json()

    from rag_kb.graph_crag.crag_graph import run_with_trace
    from rag_kb.utils.citation import build_citation_meta, is_refusal, parse_citations

    result = run_with_trace(question, save=False, thread_id=session_id)
    answer = result.get('generation') or ''
    docs = result.get('documents') or []
    cites = parse_citations(answer)
    meta = build_citation_meta(docs, cites) if cites and docs else []
    payload = {
        'answer': answer,
        'refused': is_refusal(answer),
        # 直接透传 build_citation_meta 的结果：原先只挑 n/filename/title 三个字段，
        # 把 from_web 与 url 丢掉了，导致联网引用在终端里显示成
        # 「（网络搜索，非知识库）·（网络搜索，非知识库）」且无法点开核对。
        'citations': meta,
        # [2026-11] 三态来源：knowledge_base / web / model_knowledge / model_direct
        'grounded': result.get('grounded', False),
        'source_type': result.get('source_type', 'unknown'),
        'elapsed_sec': round(time.time() - t0, 2),
    }
    if trace:
        payload['trace'] = result.get('trace')
    return payload


def _print_answer(payload: dict, prefix: str = '') -> None:
    """打印答案 + 引用 + 耗时（ask 与 chat 共用）。"""
    print(f'{prefix}{payload.get("answer") or "（空答案）"}')
    cites = payload.get('citations') or []
    if cites:
        print('引用：')
        for m in cites:
            if m.get('from_web'):
                # 联网来源：显示真实 URL，方便点开核对；没有 URL 时退回 filename
                loc = m.get('url') or m.get('filename') or '（联网结果）'
                title = (m.get('title') or '').strip()
                tail = f' · {title}' if title and title != loc else ''
                print(f"  [{m.get('n')}] {loc}{tail}（联网）")
            else:
                print(f"  [{m.get('n')}] {m.get('filename')} · {m.get('title')}")
    src = payload.get('source_type')
    tag = ''
    if src == 'model_knowledge':
        tag = ' | 降级作答（通用知识，非知识库）'
    elif src == 'model_direct':
        tag = ' | 直接回答（未检索）'
    elif src == 'web':
        tag = ' | 来源：联网'
    print(f"耗时 {payload.get('elapsed_sec')}s" + (' | 模型判定为拒答' if payload.get('refused') else '') + tag)


def cmd_ask(args: argparse.Namespace) -> int:
    """命令行问一句：答案 + 引用（无需先起服务）。"""
    import json

    try:
        payload = _ask_once(args.question, args.url, args.session_id, args.trace)
    except Exception as e:  # noqa: BLE001
        print(_service_hint(args.url, e) if args.url else f'[错误] {e}')
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_answer(payload)
    return 0


def _service_hint(url: str, err: object = '') -> str:
    """连不上服务时的可执行提示（--url 模式专用）。"""
    return (
        f'[错误] 连不上服务 {url}：{err}\n'
        f'       `--url` 模式要求服务**在另一个窗口常驻运行**：\n'
        f'         窗口 A: python main.py serve\n'
        f'         窗口 B: python main.py chat --url {url}\n'
        f'       或者去掉 --url，直接在本窗口跑（首次加载模型约 20-30s）：\n'
        f'         python main.py chat\n'
        f'       （`serve` 是常驻进程，Ctrl+C 会把它停掉）'
    )


def _probe_service(url: str) -> str | None:
    """探活服务；可达返回 None，否则返回错误描述。"""
    from rag_kb.utils.http_local import make_client

    try:
        with make_client(url, timeout=3) as client:
            client.get(url.rstrip('/') + '/health')
        # 只要端口有 HTTP 响应就算「服务在跑」（/health 可能不存在，或被网关代答）；
        # 真正的错误留给每次提问时的 /ask 响应暴露。
        return None
    except Exception as e:  # noqa: BLE001
        return f'{type(e).__name__}: {e}'


# 终端命令的典型开头：误粘到 chat 里会被当成"问题"送去联网搜索，白花一次调用
_SHELL_HEADS = {
    'pip', 'pip3', 'cd', 'ls', 'dir', 'git', 'conda', 'uvicorn',
    'pytest', 'docker', 'powershell', 'cmd', 'activate', 'deactivate',
}
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def _looks_like_shell(line: str) -> bool:
    """粗判输入是不是终端命令（宁可漏判，也不要误伤正常提问）。

    只认「一眼就是命令」的形态：
      python main.py xxx / python -m xxx / python xxx.py / pip install ...
      .venv\\Scripts\\... / & c:/... / cd datas ...

    像 "python 的 GIL 是什么"、"pip 和 conda 有什么区别" 这类带中文的提问
    **一律不算命令** —— 终端命令基本不含中文，靠这一条把误伤挡掉。
    """
    s = line.strip()
    low = s.lower()
    # 强特征：无论是否含中文都按命令处理（多为带中文路径的整条命令）
    if low.startswith(('.venv\\', '.venv/', '& c:/', '& c:\\', '& "c:', '$env:')):
        return True
    parts = low.split()
    if not parts:
        return False
    head = parts[0]
    if head in ('python', 'python3', 'py'):
        # 只有带上脚本名 / -m / -c 才当作命令
        return any(t in ('-m', '-c') or t.endswith('.py') for t in parts[1:])
    if head in _SHELL_HEADS:
        return not _CJK_RE.search(s)
    return False


def cmd_chat(args: argparse.Namespace) -> int:
    """交互式问答：启动后在终端里连续提问。"""
    import time

    if args.url:
        problem = _probe_service(args.url)
        if problem:
            # 启动即检查：否则用户每问一句都只看到一行「连不上」，不知道该怎么办
            print(_service_hint(args.url, problem))
            return 1

    session = args.session_id or f'cli-{int(time.time())}'
    where = f'复用服务 {args.url}' if args.url else '进程内'
    print(f'=== 交互问答（{where}）===')
    print('直接输入问题回车；命令：:help 帮助 / :new 新会话 / :q 退出')
    print(f'会话: {session}')
    if not args.url:
        # 预热放在提示之前：否则第一条问题会静默等 20-50s，像卡住了。
        #
        # [2026-11] 惰性化之后 import 不再加载模型，必须**显式构造**（与 API 的
        # `_warmup` 一致）；只 `from ... import graph` 会让模型加载推迟到首题，
        # 实测首题因此多付 52s。
        print('正在加载模型与检索器（首次约 20-50s，只加载一次）…', flush=True)
        try:
            from rag_kb.graph_crag.crag_graph import get_graph
            from rag_kb.llm_models.embeddings_model import get_embedding_info
            from rag_kb.tools.retriever_tools import get_global_retriever

            get_embedding_info()
            get_global_retriever()
            get_graph()
            print('就绪。')
        except Exception as e:  # noqa: BLE001
            # 预热失败不退出：让用户看到具体原因，首个问题会再试一次
            print(f'[警告] 预热未完成（{type(e).__name__}: {str(e)[:160]}）；首个问题会重新尝试。')
    while True:
        try:
            line = input('\n你> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见。')
            return 0
        if not line:
            continue
        cmd = line.lower()
        if cmd in (':q', ':quit', ':exit', 'quit', 'exit'):
            print('再见。')
            return 0
        if cmd in (':help', ':h', 'help'):
            print('  直接输入问题即可；:new 开一个新会话（清除上下文）；:q 退出')
            print('  提示：终端命令请 :q 退出后执行，这里只接收自然语言问题。')
            continue
        if cmd == ':new':
            session = f'cli-{int(time.time())}'
            print(f'  已开新会话: {session}')
            continue
        if _looks_like_shell(line):
            print('  这看起来是终端命令，不是提问。请先 :q 退出，再到 PowerShell 里执行。')
            continue
        try:
            payload = _ask_once(line, args.url, session, False)
        except Exception as e:  # noqa: BLE001
            if args.url:
                print(_service_hint(args.url, e))
            else:
                print(f'[错误] {e}')
            continue
        _print_answer(payload, prefix='\n助手> ')


def cmd_eval(args: argparse.Namespace) -> int:
    script = ROOT / 'eval' / 'run_eval.py'
    cmd = [sys.executable, str(script)]
    if args.with_generation:
        cmd.append('--with-generation')
    if args.limit is not None:
        cmd += ['--limit', str(args.limit)]
    if args.no_save:
        cmd.append('--no-save')
    if args.no_cache:
        cmd.append('--no-cache')
    if args.extra:
        cmd += args.extra
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_version(_: argparse.Namespace) -> int:
    """打印版本 / Python / 项目根。

    版本优先取**已安装包的元数据**（wheel 安装后仓库里没有 pyproject.toml，
    只剩 site-packages 的 dist-info），未安装时再解析源码里的 pyproject.toml。
    """
    version = 'unknown'
    try:
        from importlib.metadata import version as _dist_version

        version = _dist_version('rag-kb')
    except Exception:  # noqa: BLE001  未安装（直接跑源码）时走下面的文件解析
        import tomllib

        try:
            with open(ROOT / 'pyproject.toml', 'rb') as f:
                data = tomllib.load(f)
            version = data.get('project', {}).get('version') or data.get('tool', {}).get('poetry', {}).get(
                'version', 'unknown'
            )
        except Exception:  # noqa: BLE001
            version = 'unknown'
    print(f'rag-kb {version} | python {sys.version.split()[0]} | {ROOT}')
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EPILOG = r"""\
────────────────────────────────────────────────────────────────
 ① 环境自检（换环境 / 出问题时先跑这个）
    python main.py doctor

 ② 日常问答
    python main.py chat                        终端连续问答（闲聊/追问/查库自动分流）
    python main.py ask "你的问题"               问一句，返回答案 + 引用
    python main.py serve                       启动 HTTP API（127.0.0.1:8000）

 ③ 换文档 / 加文档（三步，全程不改代码）
    1) 把文档放进 datas/ 下的一个**主题目录**（md/pdf/docx/xlsx/pptx
       可混放，支持子目录），目录名即源名
    python main.py ingest --list               确认识别到的文件数
    python main.py ingest                      入库（--no-recreate 为增量）
    python eval\export_chunks.py               导出块清单
    python eval\gen_testset.py --per-collection 5   自动出题

 ④ 质量评测
    python eval\run_eval.py --no-save          检索层（0 次 LLM，秒级）
    python eval\run_eval.py --with-generation --no-save   加回答与判分（需 LLM）
    python tests\run_all_checks.py             全量回归（23 项）

────────────────────────────────────────────────────────────────
说明：
  * 裸 `python` 也能跑：自动切到项目 venv（KB_NO_REEXEC / EVAL_NO_REEXEC=1 关闭）
  * 阈值在 eval\thresholds.json，换语料后按新基线调整
  * --no-save / --limit N / 门禁 INCONCLUSIVE 时结果不落盘，避免污染基线
  * 上面每行可直接复制执行；[--opt] 只表示可选参数
"""


def _prog_name() -> str:
    """帮助里的命令名：安装后显示 `rag-kb`，源码运行时显示 `main.py`。"""
    name = pathlib.Path(sys.argv[0]).name if sys.argv else ''
    return name[:-4] if name.lower().endswith('.exe') else (name or 'rag-kb')


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=_prog_name(),
        description='RAG 企业知识库统一入口（服务 / 入库 / 评测 / 自检）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    sub = ap.add_subparsers(dest='command', metavar='{doctor,serve,embed-server,ask,chat,ingest,eval,version}')

    p = sub.add_parser('doctor', help='环境自检：Python / GPU / embedding / Milvus / 清单')
    p.add_argument('--json', action='store_true', help='输出 JSON（便于脚本消费）')
    p.add_argument('--with-llm', action='store_true', help='额外测一次 LLM（有费用）')
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser('serve', help='启动 API 服务')
    p.add_argument('--host', default='127.0.0.1', help='监听地址（默认仅本机）')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--workers', type=int, default=1, help='worker 数；>1 需配合 --embed-url')
    p.add_argument('--reload', action='store_true', help='开发热重载（会反复加载模型）')
    p.add_argument('--embed-url', default=None, help='共享 embedding 服务地址，如 http://127.0.0.1:8001')
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser('embed-server', help='启动独立 embedding 服务（多 worker 共用一份模型）')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    p.set_defaults(func=cmd_embed_server)

    p = sub.add_parser('ingest', help='入库（默认全量重建；--no-recreate 走增量）')
    p.add_argument('--sources', nargs='*', default=None, help='只处理指定源名（默认全部 enabled）')
    p.add_argument('--batch-size', type=int, default=20, help='每批写入块数（默认 20）')
    p.add_argument('--no-recreate', action='store_true', help='不重建集合，按文件指纹增量')
    p.add_argument('--list', action='store_true', help='只列出源与文件数，不入库')
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser('ask', help='命令行问一句（答案 + 引用，不需要先起服务）')
    p.add_argument('question', help='要问的问题，例如 "Milvus 支持哪些索引类型？"')
    p.add_argument('--json', action='store_true', help='输出 JSON（含 refused/citations/elapsed）')
    p.add_argument('--trace', action='store_true', help='附带执行轨迹（调试用）')
    p.add_argument('--session-id', default=None, help='多轮会话 ID；不传=单轮')
    p.add_argument('--url', default=None, help='走已启动的服务（如 http://127.0.0.1:8000），省掉冷启动')
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser('chat', help='交互式问答：启动后在终端里连续提问')
    p.add_argument('--url', default=None, help='复用已启动的服务（推荐：不起第二份模型）')
    p.add_argument('--session-id', default=None, help='会话 ID（默认自动生成，支持多轮追问）')
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser('eval', help='跑检索/生成评测（透传 eval/run_eval.py）')
    p.add_argument('--with-generation', action='store_true', help='加生成层指标（LLM 判分，有费用）')
    p.add_argument('--limit', type=int, default=None, help='只跑前 N 条（会跳过写基线）')
    p.add_argument('--no-save', action='store_true', help='不写 baseline/history')
    p.add_argument('--no-cache', action='store_true', help='忽略判定缓存')
    p.add_argument('extra', nargs='*', help='其余参数原样透传给 run_eval.py')
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser('version', help='查看版本')
    p.set_defaults(func=cmd_version)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        # 不带子命令 = 打印用法并正常退出（不加载模型、不报错）
        parser.print_help()
        print('\n提示：先跑 `python main.py doctor` 做环境自检。')
        return 0

    # 依赖预检：用错解释器时给可执行的提示（退出码 3），而不是 traceback
    if args.command != 'version':
        missing = missing_deps()
        if missing:
            print(_dep_error(missing))
            return 3

    try:
        return int(args.func(args) or 0)
    except ModuleNotFoundError as e:  # 兜底：间接依赖缺失
        print(_dep_error([e.name or 'unknown']))
        return 3


if __name__ == '__main__':
    # 只在「作为脚本运行」时自动切 venv；被 import 时不介入
    try:
        rc = run_with_project_venv(sys.argv[1:])
        raise SystemExit(main() if rc is None else rc)
    except KeyboardInterrupt:
        # 模型加载/生成中途 Ctrl+C：给一行提示，不甩 traceback
        print('\n已中断（Ctrl+C）')
        raise SystemExit(130) from None
