"""验证 P0-2（rowspan 展开）与 P0-4（KB_TENANTS 报错）的修复。

P0-2 的原始用例来自 ROADMAP：
    输入两行（首格 rowspan=2）
    修复前实际  [['合并A','B1','B2'], ['C1','C2','']]      <- 第二行整体左移
    期望        [['合并A','B1','B2'], ['合并A','C1','C2']]
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from documents.table_handler import build_table_chunks, parse_table_html  # noqa: E402

CASES = [
    ("rowspan=2（P0-2 原始用例）",
     '<table><tr><td rowspan="2">合并A</td><td>B1</td><td>B2</td></tr>'
     '<tr><td>C1</td><td>C2</td></tr></table>',
     [['合并A', 'B1', 'B2'], ['合并A', 'C1', 'C2']]),

    ("rowspan=3",
     '<table><tr><td rowspan="3">R</td><td>a1</td></tr>'
     '<tr><td>a2</td></tr><tr><td>a3</td></tr></table>',
     [['R', 'a1'], ['R', 'a2'], ['R', 'a3']]),

    ("colspan=2（回归，不应被本次改动影响）",
     '<table><tr><td colspan="2">宽</td><td>c</td></tr>'
     '<tr><td>x</td><td>y</td><td>z</td></tr></table>',
     [['宽', '宽', 'c'], ['x', 'y', 'z']]),

    ("rowspan + 中间行插新列（保证列不错位）",
     '<table><tr><td rowspan="2">K</td><td>v1</td><td>w1</td></tr>'
     '<tr><td>v2</td><td>w2</td></tr></table>',
     [['K', 'v1', 'w1'], ['K', 'v2', 'w2']]),

    ("两个并排 rowspan",
     '<table><tr><td rowspan="2">A</td><td rowspan="2">B</td><td>x</td></tr>'
     '<tr><td>y</td></tr></table>',
     [['A', 'B', 'x'], ['A', 'B', 'y']]),

    ("rowspan 超出表格行数（不应崩）",
     '<table><tr><td rowspan="5">L</td><td>only</td></tr></table>',
     [['L', 'only']]),
]

print("=" * 96)
print("P0-2 rowspan 展开验证")
print("=" * 96)
n_ok = 0
for name, html, want in CASES:
    got = parse_table_html(html)
    ok = got == want
    n_ok += ok
    print()
    print("  [%s] %s" % ("OK  " if ok else "FAIL", name))
    if not ok:
        print("        期望: %s" % want)
        print("        实际: %s" % got)
print()
print("  => %d/%d 通过" % (n_ok, len(CASES)))

# 顺带确认 build_table_chunks 在含 rowspan 的表上不崩且保留合并值
print()
print("=" * 96)
print("build_table_chunks 端到端（确认合并值进入文本块）")
print("=" * 96)
html = ('<table><tr><th>组</th><th>值</th></tr>'
        '<tr><td rowspan="2">甲组</td><td>1</td></tr>'
        '<tr><td>2</td></tr></table>')
chunks = build_table_chunks(html, table_name="测试表")
for c in chunks:
    print("  ---- 块 ----")
    for line in c.splitlines():
        print("    " + line)
print()
print("  合并值「甲组」出现次数: %d（期望 2——两行都要有组名，否则 LLM 读不出第 2 行属于哪组）"
      % sum(c.count("甲组") for c in chunks))

# ---------------------------------------------------------------------------
# P0-4：KB_TENANTS 非法时必须抛出**带上下文**的错误（且不能是裸 JSONDecodeError）
# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("P0-4 KB_TENANTS 非法配置的处理（子进程，因为 env_utils 在 import 期就读环境变量）")
print("=" * 96)

PROBE = (
    "import sys; sys.path.insert(0, %r)\n"
    "try:\n"
    "    from utils.env_utils import TENANTS\n"
    "    print('FAIL_NO_ERROR', TENANTS)\n"
    "except Exception as e:\n"
    "    print('RAISED', type(e).__name__, '|', str(e)[:150])\n"
) % str(ROOT / "RAG_PROJECT")

for bad in ('{acme: [milvus_docs]}', '["milvus_docs"]', '{"acme": ["no_such_source"]}'):
    env = dict(os.environ, KB_TENANTS=bad, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    line = (r.stdout or "").strip().splitlines()
    print("  KB_TENANTS=%-32s -> %s" % (bad[:32], line[-1] if line else "(无输出)"))
