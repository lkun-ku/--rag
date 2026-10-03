"""后端选择（P3）：`EMBEDDING_BACKEND` 决定「本地加载模型」还是「走共享服务」。

## 为什么必须测

多 worker 的收益完全取决于一件事：**http 模式下进程绝对不能加载本地模型**。
一旦加载，4 个 worker 就是 4 份 1.24GB 显存 —— 比不做还糟（还会互相抢 GPU）。
这件事没法靠读代码确认（模块级构造容易漏），所以用子进程分别验证：

    C1 默认（无 env）-> local，且 `HuggingFaceEmbeddings` 实例
    C2 `EMBEDDING_BACKEND=http` -> `RemoteEmbeddings` 实例、device='remote'，
       且**进程里没有 import torch**（证明没有加载模型）
    C3 非法值 -> 回落 local
    C4 http 模式下 `EMBEDDING_INFO` 带 url，供 /health 探活
"""

import json
import os
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PY = sys.executable

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


PROBE = r"""
import json, sys
sys.path.insert(0, r'%s')
from rag_kb.llm_models import embeddings_model as m
print('__RESULT__' + json.dumps({
    'cls': type(m.bge_embedding).__name__,
    'backend': m.EMBEDDING_INFO.get('backend'),
    'device': m.EMBEDDING_INFO.get('device'),
    'url': m.EMBEDDING_INFO.get('url'),
    'torch_imported': 'torch' in sys.modules,
}))
""" % (ROOT)


def run_probe(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('EMBEDDING_')}
    env.update(env_extra)
    out = subprocess.run([PY, '-c', PROBE], capture_output=True, text=True, encoding='utf-8', env=env).stdout
    for line in out.splitlines():
        if line.startswith('__RESULT__'):
            return json.loads(line[len('__RESULT__') :])
    raise AssertionError('子进程没有输出结果，stdout=%r' % out[-500:])


print("=" * 96)
print("embedding 后端选择（子进程隔离验证）")
print("=" * 96)

print("\n[C1] 默认 = local")
r = run_probe({})
check("实例类型", r['cls'], 'HuggingFaceEmbeddings')
check("backend", r['backend'], 'local')
check("确实加载了 torch", r['torch_imported'], True)

print("\n[C2] EMBEDDING_BACKEND=http")
r = run_probe({'EMBEDDING_BACKEND': 'http', 'EMBEDDING_URL': 'http://127.0.0.1:8001'})
check("实例类型", r['cls'], 'RemoteEmbeddings')
check("backend", r['backend'], 'http')
check("device 标记为 remote", r['device'], 'remote')
check("url 透传", r['url'], 'http://127.0.0.1:8001')
check("**没有 import torch（即没有加载模型）**", r['torch_imported'], False)

print("\n[C3] 非法值回落 local")
r = run_probe({'EMBEDDING_BACKEND': 'grpc'})
check("仍然 local", r['backend'], 'local')

print("\n[C4] http 模式下的默认 URL")
r = run_probe({'EMBEDDING_BACKEND': 'http'})
check("默认 url", r['url'], 'http://127.0.0.1:8001')

print("\n" + "=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：%s" % failures)
print("=" * 96)
sys.exit(0 if not failures else 1)
