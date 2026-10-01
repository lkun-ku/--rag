"""语义分块封装 —— 契约测试（ROADMAP 4.48）。

## 锁定的行为

| # | 契约 |
|---|---|
| S1 | 默认返回 **SemanticChunker**（`breakpoint_threshold_type="percentile"`）—— **行为与改造前完全一致** |
| S2 | `force_fallback=True` 返回 RecursiveCharacterTextSplitter |
| S3 | 降级时**打 WARNING**（不许静默降级，项目既有原则） |
| S4 | `SEMANTIC_AVAILABLE` 与真实 import 结果一致 |
| S5 | `status()` 返回可用性 + 降级参数 |
| S6 | 两个分块器都能真的分块（不是空壳） |
| S7 | `markdown_parser` 已改为走 `make_splitter`（接线到位） |

**为什么 S1 是最重要的一条**：这层的目的只是「防未来 import 失败」，
**不能顺手改变现有分块行为** —— 否则 523 块要重灌、golden 的 `source_blocks` 全部失效。
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


from langchain_core.documents import Document  # noqa: E402

from rag_kb.tools import semantic_splitter as ss  # noqa: E402

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


SAMPLE = """## 检索性能

Milvus 的 HNSW 索引参数包括 M 和 efConstruction。M 越大内存占用越高。

| 场景 | 产品 | QPS | 召回率 |
|---|---|---|---|
| 静态 | TurboPuffer | 649.88 | 89.93% |
| 带过滤 | TurboPuffer | 382.53 | 61.35% |

### 延迟

P99 延迟在 98% 过滤比例下为 48.1ms。
"""

print("=" * 100)
print("S1/S2 默认语义分块、强制降级")
print("=" * 100)
sp_default = ss.make_splitter(embedding=None)  # 语义分块不需要真的 embedding 才能构造
name_default = type(sp_default).__name__
check("S1 默认返回 SemanticChunker", "Semantic" in name_default, name_default)

sp_fb = ss.make_splitter(embedding=None, force_fallback=True)
name_fb = type(sp_fb).__name__
check("S2 force_fallback -> RecursiveCharacterTextSplitter", "Recursive" in name_fb, name_fb)

print()
print("=" * 100)
print("S4/S5 状态")
print("=" * 100)
check("S4 SEMANTIC_AVAILABLE 与真实导入一致", ss.SEMANTIC_AVAILABLE is True, ss.SEMANTIC_AVAILABLE)
st = ss.status()
check("S5 status 含 semantic_available", st.get("semantic_available") is True)
check(
    "S5 status 含降级参数",
    st.get("fallback_chunk_size") == ss.FALLBACK_CHUNK_SIZE
    and st.get("fallback_chunk_overlap") == ss.FALLBACK_CHUNK_OVERLAP,
    st,
)

print()
print("=" * 100)
print("S6 两个分块器都能真的分块")
print("=" * 100)
docs = [Document(page_content=SAMPLE * 6, metadata={"filename": "t.md", "title": "t"})]
out_fb = sp_fb.split_documents(docs)
# SAMPLE*6 ≈ 1,800 字符 > chunk_size 800 -> 应产出多块
check("降级分块器产出 >1 块", len(out_fb) > 1, len(out_fb))
check("降级分块的块都非空", all(d.page_content.strip() for d in out_fb))
check("降级分块保留表格行", any("61.35" in d.page_content for d in out_fb))

# 语义分块需要真 embedding，这里只验证「有 split_documents 方法」
check("语义分块器有 split_documents", hasattr(sp_default, "split_documents"))

print()
print("=" * 100)
print("S3 降级必须打 WARNING（不许静默）")
print("=" * 100)
import io  # noqa: E402
from loguru import logger  # noqa: E402

buf = io.StringIO()
handler_id = logger.add(buf, level="WARNING", format="{message}")
ss.make_splitter(embedding=None, force_fallback=True)
logger.remove(handler_id)
captured = buf.getvalue()
check("降级时输出 WARNING", "降级" in captured or "WARNING" in captured.upper(), captured[:80].replace("\n", " "))
check("WARNING 提示了「分块结果会变化」", "分块结果会变化" in captured, captured[:80])

print()
print("=" * 100)
print("S7 markdown_parser 的接线")
print("=" * 100)
src = (ROOT / "RAG_PROJECT/rag_kb/documents/markdown_parser.py").read_text(encoding="utf-8")
check("已 import make_splitter", "make_splitter" in src)
# 只检查**代码行**（排除注释）—— 注释里出现 `SemanticChunker(` 字样是正常的说明文字
code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
check(
    "不再直接调用 SemanticChunker(",
    not any("SemanticChunker(" in l for l in code_lines),
    [l.strip()[:60] for l in code_lines if "SemanticChunker(" in l],
)
check("调用的是 make_splitter(", "make_splitter(" in src)

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
