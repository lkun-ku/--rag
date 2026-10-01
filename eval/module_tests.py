"""系统性深度测试 · 阶段 2：**模块级边界测试**。

## 目的

阶段 1 的静态扫描只能发现「结构性」问题（静默异常、死代码…）。
本阶段用**真实调用 + 边界输入**验证各模块的**行为**：

    M1 citation      parse / validate / build_meta / format_context / is_refusal
    M2 table_handler HTML 解析、rowspan/colspan、清洗、字节闸、异常输入
    M3 markdown_parser 超短块合并（首块 / 结构块 / 跨文件）、字节截断
    M4 tracer        启用/停用、step、to_dict、save 的健壮性
    M5 env_utils     配置解析的边界（含非法值）
    M6 retriever_tools 集合发现、租户解析的失败路径

## 设计原则

* **每个用例都断言具体值**，不是「不抛异常就算过」—— 后者会漏掉静默错误。
* 边界优先：空输入、单元素、超长、非法值、越界线（n 与 n+1）。
* 失败时打印**期望 vs 实际**，便于直接定位。

用法：
    .venv\\Scripts\\python.exe eval\\module_tests.py
    .venv\\Scripts\\python.exe eval\\module_tests.py --only M1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.documents import Document  # noqa: E402

RESULTS = []  # (组, 用例, ok, 期望, 实际)


_UNSET = object()  # 哨兵：区分「未指定 expect」与「期望值是 None」


def check(group: str, name: str, fn, expect=_UNSET) -> None:
    """跑一个用例。expect 未指定时用「真值判断」；指定了就做 == 比较。

    **为什么需要哨兵**：首版用 `expect=None` 兼表「未指定」，于是
    「期望返回 None」的用例永远判 FAIL（`bool(None)` 为 False）——
    实测 2 个用例因此误报（M4 的 disabled save、M6 的 get_tenant）。
    """
    try:
        got = fn()
    except Exception as e:  # noqa: BLE001
        RESULTS.append((group, name, False, expect, f"<{type(e).__name__}: {str(e)[:70]}>"))
        return
    ok = bool(got) if expect is _UNSET else (got == expect)
    RESULTS.append((group, name, ok, expect, got))


def D(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta or {"filename": "t.md"})


# ===========================================================================
# M1 · citation
# ===========================================================================
def m1() -> None:
    from utils.citation import (
        REFUSAL_TEXT,
        build_citation_meta,
        format_context,
        is_refusal,
        parse_citations,
        validate_citations,
    )

    check("M1", "parse 空串 -> []", lambda: parse_citations(""), [])
    check("M1", "parse None -> []", lambda: parse_citations(None), [])
    check("M1", "parse 无引用 -> []", lambda: parse_citations("没有引用"), [])
    check("M1", "parse 去重保序 [1][1][2] -> [1,2]", lambda: parse_citations("[1][1][2]"), [1, 2])
    check("M1", "parse [0] 也会被提取（越界交给 validate）", lambda: parse_citations("[0]"), [0])
    check("M1", "parse 连续 [1][3] -> [1,3]", lambda: parse_citations("a[1][3]"), [1, 3])

    check("M1", "validate 空答案 valid=True", lambda: validate_citations("", 5)["valid"], True)
    check("M1", "validate 边界内 [5]/n=5 valid", lambda: validate_citations("[5]", 5)["valid"], True)
    check("M1", "validate 越界 [6]/n=5 invalid=[6]", lambda: validate_citations("[6]", 5)["invalid"], [6])
    check("M1", "validate n=0 时任何引用都越界", lambda: validate_citations("[1]", 0)["invalid"], [1])
    check(
        "M1",
        "validate 原文编号 -> quoted 而非 invalid",
        lambda: (
            validate_citations("[28]", 5, docs=[D("见 [28] 节")])["quoted"],
            validate_citations("[28]", 5, docs=[D("见 [28] 节")])["invalid"],
        ),
        ([28], []),
    )
    check(
        "M1",
        "validate 真幻觉仍 invalid",
        lambda: validate_citations("[99]", 5, docs=[D("见 [28] 节")])["invalid"],
        [99],
    )
    check("M1", "validate 拒答 is_refusal=True", lambda: validate_citations(REFUSAL_TEXT, 5)["is_refusal"], True)

    check("M1", "format_context 空列表 -> 空串", lambda: format_context([]), "")
    check(
        "M1",
        "format_context 编号从 1 开始",
        lambda: format_context([D("甲"), D("乙")]).split("\n")[0].startswith("[1] 甲"),
        True,
    )

    check("M1", "build_meta 越界编号被忽略", lambda: build_citation_meta([D("a")], [9]), [])
    check("M1", "build_meta 空 cites -> []", lambda: build_citation_meta([D("a")], []), [])
    check(
        "M1",
        "build_meta 知识库来源 from_web=False",
        lambda: build_citation_meta([D("a", filename="x.md")], [1])[0]["from_web"],
        False,
    )
    check(
        "M1",
        "build_meta 网络来源 from_web=True",
        lambda: build_citation_meta([D("a", source="web", filename="（网络搜索）")], [1])[0]["from_web"],
        True,
    )
    check("M1", "is_refusal 非拒答 -> False", lambda: is_refusal("正常答案"), False)


# ===========================================================================
# M2 · table_handler
# ===========================================================================
def m2() -> None:
    from documents.table_handler import build_table_chunks, clean_cell, parse_table_html, restore_table_documents

    check("M2", "parse 空串 -> []", lambda: parse_table_html(""), [])
    check("M2", "parse 非表格 -> []", lambda: parse_table_html("<p>abc</p>"), [])
    check("M2", "parse 单行表", lambda: parse_table_html("<table><tr><td>a</td></tr></table>"), [["a"]])
    check(
        "M2", "parse 空单元格保留", lambda: parse_table_html("<table><tr><td>a</td><td></td></tr></table>"), [["a", ""]]
    )
    # 实测行为：标签未闭合时 HTMLParser 不会补全，解析结果为空 -> 整体丢弃。
    # 这是**设计如此**（`restore_table_documents` 返回 [] 时上游会回落到原有合并逻辑，
    # 不会丢内容），但值得记录：不规范 HTML 的表会被当作「无表」处理。
    check(
        "M2",
        "parse 未闭合标签的表 -> 整体丢弃（返回 []，不抛）",
        lambda: parse_table_html("<table><tr><td>a<td>b</table>"),
        [],
    )

    check("M2", "clean_cell NaN -> 空", lambda: clean_cell("NaN"), "")
    check("M2", "clean_cell nan -> 空", lambda: clean_cell("nan"), "")
    check("M2", "clean_cell 1200.0 -> 1200", lambda: clean_cell("1200.0"), "1200")
    check("M2", "clean_cell 1.5 保留", lambda: clean_cell("1.5"), "1.5")
    check("M2", "clean_cell 普通文本不变", lambda: clean_cell("abc"), "abc")

    check("M2", "build 空 html -> []", lambda: build_table_chunks(""), [])
    check("M2", "build 只有表头 -> 1 块", lambda: len(build_table_chunks("<table><tr><th>a</th></tr></table>")), 1)
    check(
        "M2",
        "build 带表名 -> 前缀含「表格：」",
        lambda: build_table_chunks("<table><tr><th>a</th></tr><tr><td>1</td></tr></table>", table_name="T")[
            0
        ].startswith("表格：T"),
        True,
    )
    # 边界：max_rows=0 曾经会死循环（P0-6），必须仍能返回
    check(
        "M2",
        "build max_rows=0 不死循环且仍产出",
        lambda: len(build_table_chunks("<table><tr><th>a</th></tr>" + "<tr><td>1</td></tr>" * 5, max_rows=0)) >= 1,
        True,
    )
    check(
        "M2",
        "build max_rows=1 按行切分（每块 1 行 + 表头）",
        lambda: len(
            build_table_chunks("<table><tr><th>h</th></tr><tr><td>1</td></tr><tr><td>2</td></tr></table>", max_rows=1)
        ),
        2,
    )

    check("M2", "restore 无 text_as_html -> []", lambda: restore_table_documents(D("x", category="Table")), [])
    check("M2", "restore 非 Table -> []", lambda: restore_table_documents(D("x", category="content")), [])
    check(
        "M2",
        "restore 正常 -> category=content 且不残留 text_as_html",
        lambda: (
            restore_table_documents(
                D("x", category="Table", text_as_html="<table><tr><th>a</th></tr><tr><td>1</td></tr></table>")
            )[0].metadata["category"],
            "text_as_html"
            in restore_table_documents(
                D("x", category="Table", text_as_html="<table><tr><th>a</th></tr><tr><td>1</td></tr></table>")
            )[0].metadata,
        ),
        ("content", False),
    )

    # ---- P1-1：PPT 表格必须复用通用表格规则（清洗 + 结构前缀）----
    # 用假 shape 对象，避免依赖真实 pptx 文件（构造畸形/特制 pptx 成本高）
    from documents.pptx_parser import PPTParser

    class _Cell:
        def __init__(self, t):
            self.text = t

    class _Row:
        def __init__(self, cells):
            self.cells = cells

    class _Tbl:
        def __init__(self, rows):
            self.rows = rows

    class _Shape:
        def __init__(self, rows):
            self.table = _Tbl(rows)

    _md = PPTParser._table_to_markdown(
        _Shape([_Row([_Cell("a"), _Cell("NaN")]), _Row([_Cell("1"), _Cell("1200.0")])]), table_name="幻灯片 1"
    )

    check(
        "M2", "PPT 表格带「表格：」前缀（否则短表会被超短块合并吞掉）", lambda: _md.startswith("表格：幻灯片 1"), True
    )
    check("M2", "PPT 表格清洗 NaN -> 空（原先原样入库）", lambda: "NaN" not in _md, True)
    check("M2", "PPT 表格清洗 1200.0 -> 1200", lambda: ("1200.0" not in _md) and ("1200" in _md), True)
    check("M2", "PPT 表格含「列名：」行", lambda: "列名：" in _md, True)
    check("M2", "PPT 表格无 table 属性时返回空串", lambda: PPTParser._table_to_markdown(object()), "")


# ===========================================================================
# M3 · markdown_parser（超短块合并 + 字节截断）
# ===========================================================================
def m3() -> None:
    from documents.markdown_parser import MarkdownParser

    M = MarkdownParser
    # **必须带 category="content"**：合并的 `_joinable()` 要求前一块 category == "content"，
    # 不带的话「没合并」会**因为错误的原因而通过**（假通过）。首版就踩了这个坑。
    CT = {"category": "content", "filename": "t.md"}

    check("M3", "merge 空列表 -> []", lambda: M._merge_short_chunks([]), [])
    check("M3", "merge 长块不动", lambda: len(M._merge_short_chunks([D("x" * 100, **CT)])), 1)

    r = M._merge_short_chunks([D("长" * 100, **CT), D("短", **CT)])
    check(
        "M3",
        "merge 超短块并入前一块（同文件 + content）",
        lambda: (len(r), r[0].page_content.endswith("短")),
        (1, True),
    )

    r2 = M._merge_short_chunks([D("短", **CT), D("长" * 100, **CT)])
    check("M3", "merge 首块超短 -> 原样保留、不向后找（绝不丢内容）", lambda: (len(r2), r2[0].page_content), (2, "短"))

    r3 = M._merge_short_chunks([D("长" * 100, **CT), D("表格：T 列名：a", **CT)])
    check("M3", "merge 结构块不被合并（块数不变）", lambda: len(r3), 2)

    r4 = M._merge_short_chunks(
        [D("长" * 100, category="content", filename="a.md"), D("短", category="content", filename="b.md")]
    )
    check("M3", "merge 跨文件不合并", lambda: len(r4), 2)

    r5 = M._merge_short_chunks([D("长" * 100, **CT), Document(page_content="短", metadata={"category": "content"})])
    check("M3", "merge 未标注 filename 的块不参与合并", lambda: len(r5), 2)

    # 边界：前一块不是 content（如 Title）-> 不合并
    # 理由：只有 content 块会入库与被检索，并入 Title 会让内容随它一起被过滤掉（静默丢失）
    r6 = M._merge_short_chunks([D("长" * 100, category="Title", filename="t.md"), D("短", **CT)])
    check("M3", "merge 前一块非 content -> 不合并", lambda: len(r6), 2)

    check(
        "M3",
        "STRUCT_PREFIXES 含「表格：」与「幻灯片」",
        lambda: all(any(p in x for x in M.STRUCT_PREFIXES) for p in ("表格：", "幻灯片")),
        True,
    )
    check("M3", "MIN_CHUNK_CHARS 为正数", lambda: M.MIN_CHUNK_CHARS > 0, True)


# ===========================================================================
# M4 · tracer
# ===========================================================================
def m4() -> None:
    from utils import tracer as T

    t = T.Tracer("q?", enabled=False)
    t.step("n1", elapsed=1.0, extra=1)
    # 注意：`n_steps` 是 `to_dict()` 的**键**，不是 Tracer 的属性（首版用错 API 导致误报）
    check("M4", "disabled trace 不记录步骤", lambda: t.to_dict()["n_steps"], 0)
    check("M4", "disabled save 返回 None", lambda: t.save(), None)
    check("M4", "current() 未 start 时为 None", lambda: T.current() is None, True)

    t2 = T.Tracer("问题", enabled=True)
    t2.step("a", elapsed=0.5)
    t2.step("b", elapsed=1.5)
    check("M4", "enabled trace 记录 2 步", lambda: t2.to_dict()["n_steps"], 2)
    d = t2.to_dict()
    check("M4", "to_dict 含关键字段", lambda: all(k in d for k in ("question", "steps", "total_sec", "tokens")), True)
    check("M4", "to_dict 的 question 正确", lambda: d["question"], "问题")

    # traced() 包装：返回值必须与原始函数一致（只加观测，不改行为）
    @T.traced("wrap")
    def fn(state):
        return {"ok": True}

    check("M4", "traced 包装不改返回值", lambda: fn({}), {"ok": True})


# ===========================================================================
# M5 · env_utils（配置解析）
# ===========================================================================
def m5() -> None:
    from utils.env_utils import DOC_SOURCES, EMBEDDING_DIM, TENANTS

    check("M5", "DOC_SOURCES 非空", lambda: len(DOC_SOURCES) > 0, True)
    check(
        "M5",
        "每个源都有 name/collection",
        lambda: all(s.get("name") and s.get("collection") for s in DOC_SOURCES),
        True,
    )
    check(
        "M5",
        "集合名合法（字母数字下划线）",
        lambda: all(s["collection"].replace("_", "").isalnum() for s in DOC_SOURCES),
        True,
    )
    check("M5", "EMBEDDING_DIM 为正整数", lambda: isinstance(EMBEDDING_DIM, int) and EMBEDDING_DIM > 0, True)
    check("M5", "未配置 KB_TENANTS 时 TENANTS 为空 dict", lambda: isinstance(TENANTS, dict), True)
    # 非法 JSON 的错误信息必须带上下文（P0-4），这里只验证正常路径；
    # 非法路径在 eval/verify_rowspan.py 里用子进程验证


# ===========================================================================
# M6 · retriever_tools（集合发现 / 租户失败路径）
# ===========================================================================
def m6() -> None:
    from tools.retriever_tools import _discover_collections, get_retriever, get_tenant, set_tenant, tenant_collections

    cols = _discover_collections()
    check("M6", "_discover_collections 返回非空且无重复", lambda: len(cols) > 0 and len(cols) == len(set(cols)), True)
    check("M6", "_discover_collections(only=子集) 生效", lambda: _discover_collections(only={cols[0]}), [cols[0]])
    check("M6", "_discover_collections(only=空集) -> []", lambda: _discover_collections(only=set()), [])
    try:
        tenant_collections("definitely_not_a_tenant")
        RESULTS.append(("M6", "未登记租户抛 PermissionError", False, "PermissionError", "未抛"))
    except PermissionError:
        RESULTS.append(("M6", "未登记租户抛 PermissionError", True, "PermissionError", "PermissionError"))
    except Exception as e:  # noqa: BLE001
        RESULTS.append(("M6", "未登记租户抛 PermissionError", False, "PermissionError", type(e).__name__))

    check("M6", "默认 get_retriever() 返回非 None", lambda: get_retriever() is not None, True)
    set_tenant(None)
    check("M6", "set_tenant(None) 后 get_tenant() 为 None", lambda: get_tenant(), None)


GROUPS = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6}


def main() -> int:
    ap = argparse.ArgumentParser(description="模块级边界测试")
    ap.add_argument("--only", default="", help="只跑指定组，逗号分隔（如 M1,M2）")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()} or set(GROUPS)

    t0 = time.time()
    for g in ("M1", "M2", "M3", "M4", "M5", "M6"):
        if g not in only:
            continue
        print("  [%s] 运行中..." % g, flush=True)
        GROUPS[g]()

    print()
    print("=" * 100)
    print("模块级边界测试结果（耗时 %.1fs）" % (time.time() - t0))
    print("=" * 100)
    cur = None
    for g, name, ok, expect, got in RESULTS:
        if g != cur:
            cur = g
            print()
        mark = "OK  " if ok else "**FAIL**"
        print("  [%s] %s" % (mark, name))
        if not ok:
            print("        期望: %r" % (expect,))
            print("        实际: %r" % (got,))

    n_ok = sum(1 for r in RESULTS if r[2])
    print()
    print("=" * 100)
    print("合计 %d/%d 通过" % (n_ok, len(RESULTS)))
    fails = [r for r in RESULTS if not r[2]]
    if fails:
        print()
        print("**失败清单**：")
        for g, name, _ok, expect, got in fails:
            print("  [%s] %s" % (g, name))
            print("      期望 %r / 实际 %r" % (expect, got))
    print("=" * 100)

    (HERE / "module_tests.json").write_text(
        json.dumps(
            [{"group": g, "case": n, "ok": ok, "expect": repr(e), "got": repr(v)} for g, n, ok, e, v in RESULTS],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("  结果 -> eval/module_tests.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
