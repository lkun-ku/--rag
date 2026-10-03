"""统一入口 main.py 的契约测试。

C1 `--help` 能列出全部子命令，且不加载模型（快）
C2 未知子命令 -> 非 0（argparse 的 2）
C3 `doctor --json` -> 结构化输出，含 python/torch/embedding/milvus/manifest，
   且 embedding 真的算了一次（dim 与配置一致）
C4 `ingest --list` -> 打印源清单，退出码 0
C5 `version` -> 打印版本
C6 每个子命令的 `--help` 都能正常退出（参数定义没写坏）
"""

import json
import os
import pathlib
import time
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MAIN = str(ROOT / 'main.py')
PY = sys.executable

passed = 0
failures = []


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print('  [OK ] %s' % name)
    else:
        failures.append(name)
        print('  [**FAIL**] %s  期望=%r 实际=%r' % (name, want, got))


def run(*args, timeout=300):
    r = subprocess.run([PY, MAIN, *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def extract_json(text: str) -> dict:
    """日志可能混在输出里：取第一个「独占一行的 {」到最后一个「独占一行的 }」。"""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == '{')
    end = max(i for i, ln in enumerate(lines) if ln.strip() == '}')
    return json.loads('\n'.join(lines[start : end + 1]))


print('=' * 96)
print('main.py 统一入口')
print('=' * 96)

print('\n[C0] 不带子命令（应打印用法并正常退出，而不是 argparse 报错）')
rc, out = run()
check('退出码 0', rc, 0)
check('打印了 usage', 'usage: main.py' in out, True)
check('列出可用子命令', 'doctor' in out and 'serve' in out, True)
check('没有 error:', 'error:' in out, False)

print('\n[C1] --help')
rc, out = run('--help')
check('退出码 0', rc, 0)
for sub in ('doctor', 'serve', 'embed-server', 'ask', 'chat', 'ingest', 'eval', 'version'):
    check(f'列出 {sub}', sub in out, True)

print('\n[C2] 未知子命令')
rc, _ = run('不存在')
check('非 0', rc != 0, True)

print('\n[C3] doctor --json')
rc, out = run('doctor', '--json')
check('退出码 0（环境应自检通过）', rc, 0)
data = extract_json(out)
for key in ('python', 'torch', 'embedding', 'milvus', 'manifest'):
    check(f'包含 {key}', key in data['checks'], True)
check('embedding 可用', data['checks']['embedding'].get('ok'), True)
check('embedding 维度与配置一致', data['checks']['embedding']['dim'], data['checks']['embedding']['expected_dim'])
check('Milvus 可达', data['checks']['milvus'].get('ok'), True)

print('\n[C4] ingest --list')
t0 = time.time()
rc, out = run('ingest', '--list')
cost = time.time() - t0
check('退出码 0', rc, 0)
# 源名 = datas/ 下的目录名（自动源机制），不写死具体名字
check('打印了源清单（表头 + 文件数列）', ('源' in out) and ('文件数' in out), True)
# 只读命令不该加载 1.24GB 嵌入模型（否则每跑一次白等 ~5s + 占显存）
check('没有加载嵌入模型', 'Loading weights' in out or 'embedding 后端' in out, False)
check('足够快（<5s）', cost < 5, True)

print('\n[C5] version')
rc, out = run('version')
check('退出码 0', rc, 0)
check('打印 rag-kb', 'rag-kb' in out, True)

print('\n[C6] 各子命令 --help')
for sub in ('doctor', 'serve', 'embed-server', 'ask', 'chat', 'ingest', 'eval', 'version'):
    rc, _ = run(sub, '--help', timeout=120)
    check(f'{sub} --help', rc, 0)

print('\n[C12] chat --url 连不上时给出可执行提示')
r = subprocess.run(
    [PY, MAIN, 'chat', '--url', 'http://127.0.0.1:59999'],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding='utf-8',
    input='',
    timeout=120,
)
out = r.stdout + r.stderr
check('退出码 1', r.returncode, 1)
check('提示在另一个窗口起服务', '另一个窗口' in out, True)
check('提示可去掉 --url', '去掉 --url' in out, True)
check('没有 Traceback', 'Traceback' in out, False)

print('\n[C10] 本机 HTTP 必须绕过系统代理')
from rag_kb.utils.http_local import is_loopback, make_client  # noqa: E402

check('127.0.0.1 判定为本机', is_loopback('http://127.0.0.1:8000'), True)
check('localhost 判定为本机', is_loopback('http://localhost:8001'), True)
check('远程地址不算本机', is_loopback('https://api.deepseek.com'), False)
c_local = make_client('http://127.0.0.1:8000')
check('本机 client 关闭 trust_env', c_local.trust_env, False)
c_local.close()
c_remote = make_client('https://api.deepseek.com')
check('远程 client 保留 trust_env', c_remote.trust_env, True)
c_remote.close()

# 系统代理会做 TLS 中间人（certifi 不认它的根证书）→ 这些域名必须直连
from rag_kb.utils.http_local import bypass_proxy, host_of  # noqa: E402

check('host_of 取主机名', host_of('https://api.deepseek.com/v1'), 'api.deepseek.com')
_old = os.environ.get('NO_PROXY')
os.environ.pop('NO_PROXY', None)
bypass_proxy('api.deepseek.com', 'api.tavily.com')
check('NO_PROXY 已写入', {'api.deepseek.com', 'api.tavily.com'} <= set(os.environ.get('NO_PROXY', '').split(',')), True)
bypass_proxy('api.deepseek.com')  # 幂等
check('重复调用不重复追加', os.environ['NO_PROXY'].count('api.deepseek.com'), 1)
os.environ['KB_KEEP_PROXY'] = '1'
os.environ.pop('NO_PROXY', None)
bypass_proxy('api.deepseek.com')
check('KB_KEEP_PROXY=1 时不改环境', os.environ.get('NO_PROXY'), None)
os.environ.pop('KB_KEEP_PROXY', None)
if _old is not None:
    os.environ['NO_PROXY'] = _old

print('\n[C11] chat 交互问答（用 stub 服务，不依赖真实模型）')


def _start_stub_ask_server():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            import json as _json

            n = int(self.headers.get('Content-Length', 0))
            body = _json.loads(self.rfile.read(n) or b'{}')
            payload = {
                'answer': f"[stub] 回答：{body.get('question', '')}",
                'refused': False,
                'citations': [{'n': 1, 'filename': 'a.md', 'title': '标题'}],
                'elapsed_sec': 0.01,
            }
            raw = _json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


srv, stub_port = _start_stub_ask_server()
r = subprocess.run(
    [PY, MAIN, 'chat', '--url', f'http://127.0.0.1:{stub_port}'],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding='utf-8',
    input='Milvus 是什么\n:q\n',
    timeout=120,
)
out = r.stdout + r.stderr
srv.shutdown()
check('退出码 0', r.returncode, 0)
check('给了输入提示', '你>' in out, True)
check('输出了答案', '[stub] 回答：Milvus 是什么' in out, True)
check('输出了引用', '[1] a.md' in out, True)
check('退出时不留异常', 'Traceback' in out, False)

print('\n[C7] 依赖预检（纯函数）')
sys.path.insert(0, str(ROOT))
import main as cli  # noqa: E402

check('缺失项被识别', cli.missing_deps(('json', 'definitely_absent_xyz')), ['definitely_absent_xyz'])
check('齐全时返回空', cli.missing_deps(('json', 'pathlib')), [])

# 用「非 venv 的解释器」跑，验证不再甩 traceback（Windows 下 base_prefix 即 Anaconda）
base_py = pathlib.Path(sys.base_prefix) / 'python.exe'
if base_py.exists() and base_py.resolve() != pathlib.Path(sys.executable).resolve():
    print('\n[C8] 用错解释器 + 禁止自动切换 -> 友好报错')
    env = {**os.environ, 'KB_NO_REEXEC': '1'}
    r = subprocess.run(
        [str(base_py), MAIN, 'ingest', '--list'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env,
        timeout=300,
    )
    out = r.stdout + r.stderr
    check('退出码 3', r.returncode, 3)
    check('提示缺少依赖', '缺少依赖' in out, True)
    check('没有 Traceback', 'Traceback' in out, False)

    print('\n[C9] 用错解释器（默认）-> 自动切到项目 venv 并成功')
    r = subprocess.run(
        [str(base_py), MAIN, 'version'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        env={k: v for k, v in os.environ.items() if k != 'KB_NO_REEXEC'},
        timeout=300,
    )
    out = r.stdout + r.stderr
    check('退出码 0', r.returncode, 0)
    check('提示切换解释器', '不是项目 venv' in out, True)
    check('拿到版本信息', 'rag-kb' in out, True)

    # 用子进程（而非 execv）实现切换：子进程退出码必须原样回传，否则 CI 判断失真
    r = subprocess.run(
        [str(base_py), MAIN, '不存在的子命令'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        env={k: v for k, v in os.environ.items() if k != 'KB_NO_REEXEC'},
        timeout=300,
    )
    check('子进程退出码被透传（argparse 错误=2）', r.returncode, 2)
else:
    print('\n[C8/C9] 跳过：找不到非 venv 解释器')

print('\n' + '=' * 96)
print('结果：%d/%d 通过' % (passed, passed + len(failures)))
if failures:
    print('失败项：%s' % failures)
print('=' * 96)
sys.exit(0 if not failures else 1)
