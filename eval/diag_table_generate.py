"""3.5.1 前置实测（第三节）：**模型拿到完整表格后，究竟能不能聚合出答案？**

## 为什么要单独测这一节

前一节（`diag_table_parent_child.py`）已用块 ID 精确对照，得到：

  * n005 的**结构化行段全都召回了**（「第 1-30 行」「第 31-60 行」分别在第 15、12 位，
    且 61-85 段也在列表里）—— 即**「父子块要修的检索层缺口」并不存在**；
  * 未召回的是**同一张表的另外两种形态**（扁平平铺版 `94e09c…`、PS 版 `3e7f9a…`），
    它们**不是**父子块能解决的（那是「同表多副本」问题）。

于是只剩一种可能：**模型看到了完整表格，仍聚合不好**。本脚本直接测它 ——
把 n005 相关的表格块**原样喂给项目真实的生成链**（同一个 `CITE_PROMPT`、同一个 LLM），
看模型能否答出「召回率最低的产品及其数值」。

## 三个对照组（一次跑完，避免服务漂移干扰对比）

    A. 只用**结构化行段**（3 段，即父子块方案能给到的东西）
    B. 加入**扁平平铺版**（同表的另一种形态）
    C. **随机对照**：同样块数、但换成无关的表格块（验证「答案确实来自这些表格」）

若 A 就能答对 -> 生成层没问题，**3.5.1 应当否决**；
若 A 答错、B 答对 -> 问题在「形态不完整」，**方向应是「同表多形态合并」而非父子块**。
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
from utils.env_utils import MILVUS_URI  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
               "kb_manuals_bge_large_1024"]

client = MilvusClient(uri=MILVUS_URI)
by_id: dict = {}
for coll in COLLECTIONS:
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "text", "title"], limit=4000):
        by_id[metrics.chunk_id(r.get("text") or "")] = r.get("text") or ""

QUESTION = "参与评测的产品里，召回率最低的是哪个？召回率是多少？"

# n005 的 4 个 golden（实测得到的 ID，见 diag_table_parent_child.json）
G_TABLE_1_30 = "af0fc19664f3"      # 表格：检索性能实测（第 1-30 行）
G_TABLE_31_60 = "f1e2ce7777cf"     # 表格：检索性能实测（第 31-60 行）
G_FLAT = "94e09c6a8006"            # 检索性能实测 场景 数据集 …（扁平平铺版）
G_PS = "3e7f9a39fdb1"              # PS 流式写入下的检索 Cohere …（PS 版）

# 61-85 段（golden 之外的第三段，但属于同一张表）
from tools.retriever_tools import retriever  # noqa: E402

docs = retriever.invoke(QUESTION)
rids = [metrics.chunk_id(d.page_content) for d in docs]
seg_61_85 = None
for d in docs:
    if "第 61-85 行" in d.page_content[:60]:
        seg_61_85 = d.page_content
        break

print("=" * 100)
print("第三节：模型拿到「完整表格」后能否聚合？（直接用项目真实生成链）")
print("=" * 100)
print("  问题: %s" % QUESTION)
print()
print("  本次检索（final_k=24）里，与「检索性能实测」这张表相关的块:")
for i, d in enumerate(docs, 1):
    head = d.page_content[:60].replace("\n", " ")
    if "检索性能实测" in head or "第 61-85 行" in head:
        gid = metrics.chunk_id(d.page_content)
        print("     [%2d] %-12s %s" % (i, gid, head))
print()

GROUPS = {
    "A. 仅结构化行段（1-30 + 31-60）": [G_TABLE_1_30, G_TABLE_31_60],
    "B. 结构化 3 段（+61-85）": [G_TABLE_1_30, G_TABLE_31_60],
    "C. 结构化 2 段 + 扁平平铺版": [G_TABLE_1_30, G_TABLE_31_60, G_FLAT],
}

from langchain_core.documents import Document  # noqa: E402
from graph2.generate_node2 import CITE_PROMPT  # noqa: E402
from utils.citation import format_context  # noqa: E402
from llm_models.all_llm import llm  # noqa: E402   # 实测路径：llm_models/ 而非 graph2/

if seg_61_85:
    GROUPS["B. 结构化 3 段（+61-85）"].append(metrics.chunk_id(seg_61_85))
else:
    GROUPS.pop("B. 结构化 3 段（+61-85）")
    print("  [!] 本次未召回到「第 61-85 行」段，B 组跳过")
print()

results = {}
for name, ids in GROUPS.items():
    texts = [by_id[i] for i in ids if i in by_id]
    missing = [i for i in ids if i not in by_id]
    if missing:
        print("  [%s] 缺块 %s" % (name, missing)); continue
    ctx = format_context([Document(page_content=t, metadata={}) for t in texts])
    prompt = CITE_PROMPT.format(context=ctx, question=QUESTION) if "{context}" in CITE_PROMPT \
        else CITE_PROMPT + "\n\n资料：\n" + ctx + "\n\n问题：" + QUESTION
    try:
        ans = llm.invoke(prompt)
        text = getattr(ans, "content", str(ans))
    except Exception as e:  # noqa: BLE001
        text = "调用失败: %s: %s" % (type(e).__name__, e)
    results[name] = {"n_blocks": len(texts), "answer": text}
    print("  " + "=" * 96)
    print("  [%s]  块数=%d" % (name, len(texts)))
    print("  " + "-" * 96)
    for line in text.strip().splitlines()[:8]:
        print("     %s" % line[:110])
    print()

# 判定：答案里是否出现 golden 的数值 61.35 与产品名 TurboPuffer
print("=" * 100)
print("判定（用 golden 答案的两个要素做客观核对）")
print("=" * 100)
print("  golden 要素: 产品名含 `TurboPuffer`、数值 `61.35`")
for name, r in results.items():
    a = r["answer"]
    ok_p = "TurboPuffer" in a or "Turbo" in a
    ok_n = "61.35" in a
    print("    %-30s 产品名=%-5s 数值=%-5s  拒答=%s" % (
        name[:30], "有" if ok_p else "**无**", "有" if ok_n else "**无**",
        "是" if "未找到依据" in a else "否"))

(HERE / "diag_table_generate.json").write_text(
    json.dumps({"question": QUESTION, "results": results}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print()
print("  证据 -> eval/diag_table_generate.json")
