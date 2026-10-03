"""TDD：远程 embedding 客户端（多 worker 场景的共用模型服务）。

## 为什么需要它

多 worker 部署时，如果每个 worker 各自加载 bge-large（1.24GB），
4 个 worker 就是 4 份显存 + 4 次 13s 启动，GPU 还会互相抢 —— 这正是 P3 要解决的。
做法是把模型放进一个独立进程（embed server），其它进程通过 HTTP 调用。

## 契约（客户端侧，用 stub 服务验证，不依赖真实模型）

    C1 `embed_documents` 返回与输入**等长的向量列表**，顺序与输入一致
    C2 超过 batch_size 时**自动分批**，且顺序仍与输入一致
    C3 `embed_query` 返回单个向量
    C4 服务端 5xx -> 抛 RuntimeError，消息里带状态码（不允许静默返回空向量）
    C5 连接失败 -> 抛 RuntimeError（而不是把异常泄漏成 httpx 类型，调用方只认一种）
"""

import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

passed = 0
failures = []


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%r 实际=%r" % (name, want, got))


class _Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):  # 静音
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.calls.append((self.path, body))
        if self.path == "/embed_documents":
            if body.get("texts") == ["boom"]:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"detail":"model crashed"}')
                return
            payload = {"vectors": [[0.5, 0.5, 0.5, 0.5] for _ in body["texts"]]}
        elif self.path == "/embed_query":
            payload = {"vector": [0.25, 0.25, 0.25, 0.25]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


srv = HTTPServer(("127.0.0.1", 0), _Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"

print("=" * 96)
print("远程 embedding 客户端（stub 服务 @ %s）" % base)
print("=" * 96)

from rag_kb.llm_models.remote_embeddings import RemoteEmbeddings  # noqa: E402

emb = RemoteEmbeddings(base_url=base, batch_size=2, timeout=5.0)

print("\n[C1/C2] 批量与分批")
docs = ["t%d" % i for i in range(5)]
vecs = emb.embed_documents(docs)
check("返回条数等于输入条数", len(vecs), 5)
check("每条维度一致", {len(v) for v in vecs}, {4})
check("按 batch_size=2 分成 3 次请求", len([c for c in _Handler.calls if c[0] == '/embed_documents']), 3)
check(
    "分批后顺序不变（请求里的文本顺序）",
    [t for c in _Handler.calls if c[0] == '/embed_documents' for t in c[1]['texts']],
    docs,
)

print("\n[C3] 单条查询")
one = emb.embed_query("hello")
check("返回单个向量", one, [0.25, 0.25, 0.25, 0.25])

print("\n[C4] 服务端 5xx")
try:
    emb.embed_documents(["boom"])
    check("抛出异常", False, True)
except RuntimeError as e:
    check("RuntimeError 且带状态码", "500" in str(e), True)
except Exception as e:  # noqa: BLE001
    check("异常类型应为 RuntimeError", type(e).__name__, "RuntimeError")

print("\n[C5] 连接失败")
dead = RemoteEmbeddings(base_url="http://127.0.0.1:9", timeout=1.0)
try:
    dead.embed_query("x")
    check("抛出异常", False, True)
except RuntimeError as e:
    check("RuntimeError（不泄漏 httpx 类型）", type(e).__name__, "RuntimeError")
    check("消息里说明连不上", "连不上" in str(e) or "失败" in str(e), True)

emb.close()
dead.close()
srv.shutdown()

print("\n" + "=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：%s" % failures)
print("=" * 96)
sys.exit(0 if not failures else 1)
