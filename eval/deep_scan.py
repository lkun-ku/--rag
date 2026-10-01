"""系统性深度扫描（阶段 1）—— 静态层面找潜在缺陷。

## 为什么用 AST 而不是正则

正则会把注释/字符串里的字样也算进来（大量假阳性），而本轮要「找出所有潜在 bug」，
噪声会淹没真问题。AST 能精确判断「`except` 块里到底有没有日志调用」
这类结构性事实。

## 13 类规则

**确定性问题**（通常就是 bug）：

    R1  裸 `except:`              —— 连 KeyboardInterrupt 都吞
    R2  `except: pass` 静默吞异常  —— 无任何日志，出问题无从排查
    R3  可变默认参数               —— def f(x=[]) 的经典陷阱
    R4  同文件同名函数重复定义      —— 后者覆盖前者（P0-7 就是这类）
    R5  生产代码里的 assert        —— python -O 时会被整体移除
    R6  eval / exec / os.system    —— 安全与可维护性

**可疑模式**（需人工判断）：

    R7  `except Exception` 内无任何 log
    R8  TODO / FIXME / XXX / HACK
    R9  硬编码绝对路径（C:\\ / /Users/ / /home/）
    R10 生产代码里的 print(
    R11 未使用的 import
    R12 空函数 / 空类（只有 docstring 或 pass）
    R13 超长函数（> 120 行）

用法：
    .venv\\Scripts\\python.exe eval\\deep_scan.py
    .venv\\Scripts\\python.exe eval\\deep_scan.py --only R1,R2
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
from typing import Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = ROOT / "RAG_PROJECT"
SKIP_DIRS = {".venv", "scratch", "__pycache__", ".git", "node_modules", "logs", "models"}
OUT = HERE / "deep_scan.json"

# 这些文件里的 print 是「CLI 入口的正常输出」，不算调试残留
CLI_OK = {"run.py", "show_trace.py", "try_search.py"}

findings: List[Dict] = []


def add(rule: str, sev: str, path: pathlib.Path, line: int, detail: str) -> None:
    findings.append(
        {
            "rule": rule,
            "severity": sev,
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "line": line,
            "detail": detail,
        }
    )


def _has_log_call(nodes: List[ast.AST]) -> bool:
    """块内是否有日志调用（log.xxx / logger.xxx / print）。"""
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Attribute) and f.attr in (
                    "info",
                    "warning",
                    "error",
                    "debug",
                    "exception",
                    "critical",
                ):
                    return True
                if isinstance(f, ast.Name) and f.id in ("print", "warn"):
                    return True
    return False


class V(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path, src: str) -> None:
        self.path = path
        self.src = src
        self.funcs: Dict[str, int] = {}
        self.imported: Dict[str, int] = {}  # 名字 -> 定义行
        self.used: set = set()

    # ---- R1/R2/R7 ----
    def visit_Try(self, node: ast.Try) -> None:
        for h in node.handlers:
            line = h.lineno
            if h.type is None:
                add("R1", "high", self.path, line, "裸 except:（会吞掉 KeyboardInterrupt/SystemExit）")
            body = h.body
            only_pass = all(isinstance(s, ast.Pass) for s in body)
            if only_pass and h.type is not None:
                tn = ast.unparse(h.type) if h.type else "?"
                add("R2", "high", self.path, line, f"`except {tn}: pass` —— 静默吞异常，无任何日志")
            elif not _has_log_call(body) and h.type is not None:
                tn = ast.unparse(h.type) if h.type else "?"
                # [修正] `except X: raise`（或 `raise ... from`）是**向上传播**，
                # 不是「吞掉」—— 那是正常的错误处理，不该报。
                has_raise = any(isinstance(sub, ast.Raise) for s in body for sub in ast.walk(s))
                if not has_raise:
                    add("R7", "medium", self.path, line, f"`except {tn}` 块内无日志调用（吞掉错误后继续）")
        self.generic_visit(node)

    # ---- R3 ----
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_func(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_func(self, node) -> None:
        # R4：同 scope 同名重复（这里只查同文件顶层与同类内）
        key = getattr(node, "_scope", "") + node.name
        if key in self.funcs:
            add("R4", "high", self.path, node.lineno, f"函数 `{node.name}` 重复定义（上次在第 {self.funcs[key]} 行）")
        else:
            self.funcs[key] = node.lineno

        # R3：可变默认参数
        for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                add(
                    "R3",
                    "high",
                    self.path,
                    node.lineno,
                    f"`{node.name}` 使用可变默认参数（{ast.unparse(d)}）—— 跨调用共享状态",
                )
        # R5：assert
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert):
                add(
                    "R5",
                    "medium",
                    self.path,
                    sub.lineno,
                    f"生产代码里的 assert（python -O 会被移除）: {ast.unparse(sub.test)[:60]}",
                )
        # R12：空函数
        body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not body or all(isinstance(s, ast.Pass) for s in body):
            add("R12", "low", self.path, node.lineno, f"`{node.name}` 是空函数")
        # R13：超长函数
        if node.end_lineno and node.end_lineno - node.lineno > 120:
            add(
                "R13",
                "low",
                self.path,
                node.lineno,
                f"`{node.name}` 长 {node.end_lineno - node.lineno} 行（>120，可维护性）",
            )

    # ---- R6 / R10 ----
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        name = ast.unparse(f)
        if name in ("eval", "exec") or name in ("os.system", "subprocess.call", "subprocess.run"):
            add("R6", "high", self.path, node.lineno, f"使用 `{name}(...)`")
        if name == "print" and self.path.name not in CLI_OK:
            add("R10", "low", self.path, node.lineno, "生产代码里的 print(")
        self.generic_visit(node)

    # ---- R11 ----
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imported[(a.asname or a.name).split(".")[0]] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            if a.name != "*":
                self.imported[a.asname or a.name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.used.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # `log.info` 这种：把最外层名字也记入 used
        n = node
        while isinstance(n, ast.Attribute):
            n = n.value  # type: ignore[assignment]
        if isinstance(n, ast.Name):
            self.used.add(n.id)
        self.generic_visit(node)


TEXT_RULES = [
    ("R8", re.compile(r"\b(TODO|FIXME|XXX|HACK)\b"), "medium", "遗留标记"),
    # 硬编码绝对路径。只扫**非注释行**；且用 `(?<![\w:/])` 排除 URL ——
    # 否则 `http://schemas...` 里的 `p:/` 会被误判成 Windows 盘符（实测假阳性 4 条）。
    ("R9", re.compile(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|/Users/|/home/[a-z])"), "medium", "硬编码绝对路径"),
]


def scan_file(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        add("R0", "high", path, e.lineno or 0, f"**语法错误**: {e.msg}")
        return
    v = V(path, src)
    # 标记 scope（简单区分顶层与类内）
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node._scope = ""  # type: ignore[attr-defined]
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sub._scope = node.name + "."  # type: ignore[attr-defined]
    v.visit(tree)

    _lines = src.splitlines()
    for name, line in v.imported.items():
        if name not in v.used and name != "annotations":
            # [修正] 带 `# noqa` 的 import 是**刻意保留**的（例如 `import tools.retriever_tools
            # # noqa: F401` 是为了触发单例构造的副作用），不该报未使用。
            raw = _lines[line - 1] if 0 < line <= len(_lines) else ""
            if "noqa" in raw:
                continue
            add("R11", "low", path, line, f"未使用的 import: `{name}`")

    for rule, rx, sev, label in TEXT_RULES:
        for i, raw in enumerate(src.splitlines(), 1):
            s = raw.strip()
            if s.startswith("#"):
                continue  # 注释里的字样不算
            if rx.search(raw):
                add(rule, sev, path, i, f"{label}: {s[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="系统性深度静态扫描")
    ap.add_argument("--only", default="", help="只跑指定规则，逗号分隔（如 R1,R2）")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    files = [p for p in PKG.rglob("*.py") if not any(s in p.parts for s in SKIP_DIRS)]
    for p in sorted(files):
        scan_file(p)

    if only:
        findings[:] = [f for f in findings if f["rule"] in only]

    ORDER = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (ORDER.get(f["severity"], 9), f["rule"], f["file"], f["line"]))

    print("=" * 104)
    print("系统性深度静态扫描：%d 个文件，命中 %d 条" % (len(files), len(findings)))
    print("=" * 104)

    by_rule: Dict[str, List[Dict]] = {}
    for f in findings:
        by_rule.setdefault(f["rule"], []).append(f)

    for rule in sorted(by_rule):
        items = by_rule[rule]
        sev = ORDER.get(items[0]["severity"], 9)
        print()
        print("--- %s [%s] 共 %d 条 ---" % (rule, items[0]["severity"], len(items)))
        for it in items[:14]:
            print("  %-52s :%-5d %s" % (it["file"], it["line"], it["detail"][:88]))
        if len(items) > 14:
            print("  ...（还有 %d 条，见 JSON）" % (len(items) - 14))

    OUT.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("  明细 -> eval/deep_scan.json")
    print()
    print(
        "  严重度统计: high=%d  medium=%d  low=%d"
        % (
            sum(1 for f in findings if f["severity"] == "high"),
            sum(1 for f in findings if f["severity"] == "medium"),
            sum(1 for f in findings if f["severity"] == "low"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
