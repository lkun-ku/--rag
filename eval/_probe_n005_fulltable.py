"""对照实验：把 n005 那张表的**完整内容**喂给生成链，看它能否作答。

## 为什么要做这一步

n005 实测 **10/10 拒答**，而答案（`TurboPuffer, 83.52%`）**就在上下文里的两个 golden 块里**。

**我的假设**：表格被按 30 行切成碎片，模型看到的只是表格的一部分，
**无法确认"83.52% 是全局最低"** → 保守拒答。

> **但这个假设没有被验证过。** 业界的主流方案（父块 / 结构化查询）都建立在
> 「补全上下文就能解决」的前提上 —— 如果前提不成立，改了也白改。

## 本实验

把**同一张表的全部分片**（`filename=vector_db_benchmark.xlsx.md`,
`title=检索性能实测`，**包括判分没有保留的那些**）拼成完整表，
直接喂给生成链，问同一个问题。

**判读（决策树）**：

    · 拼全表后 **能作答且答案正确** -> 假设成立 -> **A 路线（父块）有效**
    · 拼全表后 **仍拒答**          -> 假设不成立 -> 问题不在上下文完整性，
                                      需转 B（结构化查询）或 C（切分策略）

## 成本

一次生成调用（约 2s）+ 一次库里扫表（约 1s）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

import metrics  # noqa: E402
from utils.citation import is_refusal  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

QID = os.environ.get("FT_QID", "n005")
TARGET_FILE = "vector_db_benchmark.xlsx.md"
TARGET_TITLE = "检索性能实测"

cases = {
    json.loads(l).get("id"): json.loads(l)
    for l in (HERE / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
}
C = cases[QID]
Q = C["q"]
print("=" * 104)
print("对照实验：完整表 vs 切片碎片（%s）" % QID)
print("=" * 104)
print("  问题      : %s" % Q)
print("  golden 答案: %s" % C.get("answer"))
print("  kind      : %s" % C.get("kind"))
print()

# ---- 1. 取该表的**全部分片**（不筛判分，直接用表名+标题捞）--------------------
from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402

client = MilvusClient(uri=MILVUS_URI)
parts = []
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024", "kb_manuals_bge_large_1024"):
    try:
        rows = client.query(collection_name=coll, filter="", output_fields=["filename", "title", "text"], limit=8000)
    except Exception:  # noqa: BLE001
        continue
    for r in rows:
        if r.get("filename") == TARGET_FILE and (r.get("title") or "") == TARGET_TITLE:
            parts.append(r.get("text") or "")

print("  该表在库里的分片数: %d" % len(parts))
if not parts:
    print("  [**找不到分片**] 请检查 filename/title 是否变了")
    sys.exit(2)

# ---- 2. 拼成完整表 ---------------------------------------------------------
# 分片是「第 N-M 行」的形式，按起始行号排序后拼接；表头重复出现，只保留一次
import re  # noqa: E402


def _start_line(t: str) -> int:
    m = re.search(r"第\s*(\d+)\s*-\s*(\d+)\s*行", t)
    return int(m.group(1)) if m else 10**9


parts.sort(key=_start_line)
full_lines, seen_head = [], set()
for p in parts:
    for l in p.splitlines():
        # 去掉「表格：…（第 N-M 行）」这类元信息行，避免重复
        if l.startswith("表格：") or l.startswith("列名："):
            continue
        if l.startswith("|") and set(l.strip()) <= set("|-: "):
            if "sep" in seen_head:
                continue
            seen_head.add("sep")
        if l.startswith("|") and ("场景" in l or "产品与规格" in l):
            if "hdr" in seen_head:
                continue
            seen_head.add("hdr")
        full_lines.append(l)

full_text = "\n".join(full_lines)
print("  拼接后: %d 行 / %d 字符" % (len(full_lines), len(full_text)))
print()

# ---- 3. 直接喂生成链（**不经过判分**，模拟「父块」策略下的上下文）------------
from graph2.generate_node2 import generate  # noqa: E402

print("=" * 104)
print("A. 完整表（模拟父块策略）")
print("=" * 104)
err_a = err_b = False  # [FIX] 三态判读需要它们
try:
    # 直接用生产节点 —— 它内部走同一个 rag_chain，且已处理「单个/多个 documents」
    st = {
        "question": Q,
        "documents": [Document(page_content=full_text, metadata={"filename": TARGET_FILE, "title": TARGET_TITLE})],
    }
    out_full = generate(dict(st))
    txt_full = (out_full or {}).get("generation") or ""
    print("  上下文: 1 块（完整表，%d 字符）" % len(full_text))
    print("  拒答: %s" % ("**是**" if is_refusal(txt_full) else "否"))
    print("  答案: %s" % txt_full[:400].replace("\n", " "))
    # 是否命中正确值
    low = txt_full.lower()
    hit = ("83.52" in txt_full) and ("turbopuffer" in low)
    print("  **命中正确值（TurboPuffer + 83.52）: %s**" % ("✅ 是" if hit else "❌ 否"))
except Exception as e:  # noqa: BLE001
    print("  失败: %s: %s" % (type(e).__name__, str(e)[:140]))
    hit = False
    err_a = True

# ---- 4. 对照：只给两个 golden 分片（生产实际看到的）-------------------------
print()
print("=" * 104)
print("B. 对照：只给判分后保留的块（生产实际上下文）")
print("=" * 104)
from graph2.graph_2 import run_with_trace  # noqa: E402

try:
    out = run_with_trace(Q)
    gen_prod = (out.get("generation") or "").strip()
    docs = out.get("documents") or []
    print("  上下文块数: %d" % (len(docs) if isinstance(docs, list) else 1))
    print("  拒答: %s" % ("**是**" if is_refusal(gen_prod) else "否"))
    print("  答案: %s" % gen_prod[:300].replace("\n", " "))
except Exception as e:  # noqa: BLE001
    print("  失败: %s: %s" % (type(e).__name__, str(e)[:140]))
    err_b = True

print()
print("=" * 104)
print("判读")
print("=" * 104)
# [FIX] **三态判读** —— 首版把「调用失败」与「答不对」混为一谈，
# 于是 LLM 超时时会输出「假设不成立」这种**没有依据的结论**。
# 这是测量脚本的经典错误（本项目在 4.13 吃过同类：拿旧 trace 当对照）。
if err_a or err_b:
    print(
        "  **本次实验未完成**（A=%s, B=%s）—— **不能据此下任何结论**。"
        % ("失败" if err_a else "成功", "失败" if err_b else "成功")
    )
    print("  需在端点稳定后重跑本脚本。**不要**把「调用失败」读成「假设不成立」。")
elif hit:
    print("  **完整表下答对了** -> 假设成立：问题确实是「上下文不完整」导致的。")
    print("  **A 路线（父块 / 递归检索）有效** —— 值得按 4.35.5 推进：")
    print("    本例完整表 %d 字符（约 %d token），增量可接受。" % (len(full_text), len(full_text) // 2))
else:
    print("  **完整表下仍未答对**（A/B 都成功但没命中正确值）-> 假设不成立。")
    print("  说明问题**不在上下文完整性** —— A 路线（父块）不会解决 n005。")
    print("  应转向：")
    print("    · B 结构化查询（表格转 SQL/DataFrame 后精确求最小值）")
    print("    · 或 C 切分策略（但本次已给全表，C 也难奏效）")
    print("  **这一步提前拦住了一次无效的改造。**")
