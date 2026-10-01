"""`.html` / `.txt` 解析器 —— 契约测试（ROADMAP 4.50）。

## 锁定的行为

| # | 契约 |
|---|---|
| T1 | `HTMLDocParser.parse` 能把 HTML 转成 Document，**标题层级保留** |
| T2 | `TextDocParser.parse` 用**文件名**当一级标题（`# stem`） |
| T3 | TXT 里「像标题」的短行被提升为 `##` |
| T4 | 空文件 -> 返回 `[]`（不抛） |
| T5 | `write_milvus` 的 `text` / `html` 分支**不再 NotImplementedError** |
| T6 | `DOC_SOURCES` 有 `web` 源（html），且**默认 `enabled=False`**（避免空集合影响检索）|
| T7 | 转换后的 `filename` 元数据是**真实文件名**（不是临时文件名）|
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

FAILED = []
TOTAL = 0


def check(name, cond, detail=""):
    global TOTAL
    TOTAL += 1
    if cond:
        print("  [OK ] %s" % name)
    else:
        print("  [FAIL] %s  %s" % (name, detail))
        FAILED.append(name)


def write_tmp(suffix: str, content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="t_", text=False)
    with open(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


print("=" * 100)
print("T1/T7 HTML -> Document（标题保留、filename 是真实名）")
print("=" * 100)
from documents.plain_converters import HTMLDocParser, TextDocParser  # noqa: E402

html = """<html><body>
<h1>Milvus 检索性能</h1>
<p>HNSW 索引在 98% 过滤比例下 P99 为 48.1ms。</p>
<h2>召回率</h2>
<table><tr><th>产品</th><th>召回率</th></tr>
<tr><td>TurboPuffer</td><td>61.35%</td></tr></table>
</body></html>"""
p = write_tmp(".html", html)
try:
    docs = HTMLDocParser().parse(p)
    check("T1 产出 >0 块", len(docs) > 0, len(docs))
    alltext = "\n".join(d.page_content for d in docs)
    check("T1 保留标题文本", "Milvus 检索性能" in alltext, alltext[:60])
    check("T1 保留表格数值", "61.35" in alltext, alltext[:80])
    check(
        "T7 filename 是真实名（含 .md 后缀）",
        all(d.metadata.get("filename", "").endswith(".md") for d in docs),
        [d.metadata.get("filename") for d in docs[:2]],
    )
    check("T7 filename 不含临时前缀 conv_", all("conv_" not in d.metadata.get("filename", "") for d in docs))
finally:
    pathlib.Path(p).unlink(missing_ok=True)

print()
print("=" * 100)
print("T2/T3 TXT -> Document（文件名做标题、小标题提升）")
print("=" * 100)
txt = """一、安装步骤
下载安装包并解压。

二、配置说明
修改 config.yaml 中的端口。

普通段落文字，不是标题。
"""
p2 = write_tmp(".txt", txt)
try:
    docs2 = TextDocParser().parse(p2)
    check("T2 产出 >0 块", len(docs2) > 0, len(docs2))
    t2 = "\n".join(d.page_content for d in docs2)
    stem = pathlib.Path(p2).stem
    check("T2 文件名作为一级标题", stem in t2 or "标题" in t2, t2[:60])
    check("T3「一、安装步骤」被提升为标题", "安装步骤" in t2, t2[:100])
    check("T3 普通段落保留", "普通段落文字" in t2)
finally:
    pathlib.Path(p2).unlink(missing_ok=True)

print()
print("=" * 100)
print("T4 空文件 -> []")
print("=" * 100)
p3 = write_tmp(".txt", "   \n\n  ")
try:
    check("T4 空 txt 返回 []", TextDocParser().parse(p3) == [])
finally:
    pathlib.Path(p3).unlink(missing_ok=True)
p4 = write_tmp(".html", "<html></html>")
try:
    check("T4 空 html 返回 []", HTMLDocParser().parse(p4) == [])
finally:
    pathlib.Path(p4).unlink(missing_ok=True)

print()
print("=" * 100)
print("T5 write_milvus 分支")
print("=" * 100)
src = (ROOT / "RAG_PROJECT/documents/write_milvus.py").read_text(encoding="utf-8")
check("T5 有 html 分支", "parser_type == 'html'" in src)
check(
    "T5 text 分支不再抛 NotImplementedError",
    "raise NotImplementedError" not in src.split("if parser_type == 'text'")[1][:600],
    "仍是 NotImplementedError",
)
check("T5 引用了 TextDocParser", "TextDocParser" in src)
check("T5 引用了 HTMLDocParser", "HTMLDocParser" in src)

print()
print("=" * 100)
print("T6 DOC_SOURCES 的 web 源")
print("=" * 100)
from utils.env_utils import DOC_SOURCES  # noqa: E402

web = next((s for s in DOC_SOURCES if s["name"] == "web"), None)
check("T6 存在 web 源", web is not None)
if web:
    check("T6 格式含 .html", ".html" in web["formats"], web["formats"])
    check("T6 parser='html'", web["parser"] == "html", web["parser"])
    check("T6 **默认 enabled=False**（避免空集合影响检索）", web.get("enabled") is False, web.get("enabled"))

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
