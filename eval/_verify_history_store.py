"""历史向量检索 —— **真实 Milvus + 真实 embedding** 端到端验证（ROADMAP 4.44）。

契约测试（test_history_store.py）用的是假 store，只锁工程约定。
本脚本回答真正的问题：**它真的能找到「含具体数值的早期轮次」吗？会话隔离真的生效吗？**

流程：
    1. 开 HISTORY_ENABLED=1（import 前设置，因为模块加载时读环境变量）
    2. 往 session-A 写 3 轮**含具体数值**的真实问答
    3. 往 session-B 写 1 轮（测隔离）
    4. 用「指代式」问题检索 A -> 应找回含 184.54 的那一轮
    5. 检索 B -> 看不到 A 的内容
"""
from __future__ import annotations

import os
import pathlib
import sys
import uuid

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

# **必须在 import 前设置**：模块在加载时读 HISTORY_ENABLED
os.environ["HISTORY_ENABLED"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from tools import history_store  # noqa: E402

RUN = uuid.uuid4().hex[:6]
SID_A = "__hist_%s_A" % RUN
SID_B = "__hist_%s_B" % RUN

print("=" * 98)
print("历史向量检索端到端（真实 Milvus + 真实 embedding）")
print("=" * 98)
print("  HISTORY_ENABLED = %s" % history_store.history_enabled())
print("  HISTORY_COLLECTION = %s" % history_store.HISTORY_COLLECTION)
print("  session-A = %s | session-B = %s" % (SID_A, SID_B))

RES = []

# ---- 写入 A 的 3 轮（第 1 轮含关键数值）-------------------------------------
print()
print("【写入】session-A 三轮真实问答")
turns_a = [
    ("TurboPuffer 在带过滤向量检索、98% 过滤比例下的 QPS 是多少？",
     "TurboPuffer 在 Cohere Large 10M-768D 数据集、98.00% 过滤比例下 QPS 为 184.54 [1]。"),
    ("Milvus 支持哪些索引类型？",
     "Milvus 支持 FLAT、IVF 系列（IVF_FLAT/IVF_PQ/IVF_SQ8）、HNSW 系列、DISKANN、SCANN 等 [2]。"),
    ("jieba 分词在哪里配置？",
     "jieba 分析器在 Milvus 集合 schema 的 text 字段上配置 analyzer_params [3]。"),
]
for i, (q, a) in enumerate(turns_a, 1):
    ok = history_store.record_turn(SID_A, i, q, a)
    print("  第%d轮 写入 %s | %s" % (i, "OK" if ok else "**FAIL**", q[:46]))
    RES.append(("A 第%d轮写入成功" % i, ok is True, None))

# ---- 写入 B 的 1 轮（测隔离）------------------------------------------------
print()
print("【写入】session-B 一轮（用于隔离验证）")
ok_b = history_store.record_turn(SID_B, 1, "完全不相干的问题", "另一个会话的答案：返回值是 999 [9]。")
print("  写入 %s" % ("OK" if ok_b else "**FAIL**"))
RES.append(("B 第1轮写入成功", ok_b is True, None))

# ---- 检索 A：指代式问题，应找回第 1 轮（含 184.54）---------------------------
print()
print("【检索】session-A：指代式问题「那个 QPS 对应的产品叫什么？」")
hits_a = history_store.lookup_history(SID_A, "那个 QPS 对应的产品叫什么？", top_k=3)
print("  命中 %d 块" % len(hits_a))
for i, d in enumerate(hits_a, 1):
    md = getattr(d, "metadata", None) or {}
    print("   %d.（%s）%s" % (i, md.get("title", "?"), (d.page_content or "")[:78].replace("\n", " ")))
RES.append(("A 检索到结果", len(hits_a) >= 1, len(hits_a)))
RES.append(("A 找回含 184.54 的第1轮",
            any("184.54" in (d.page_content or "") for d in hits_a),
            [d.page_content[:40] for d in hits_a]))
RES.append(("A 命中的块都属于 session-A",
            all((getattr(d, "metadata", None) or {}).get("filename") == SID_A for d in hits_a)))

# ---- 检索 B：隔离验证 --------------------------------------------------------
print()
print("【隔离】session-B 检索「QPS 是多少」")
hits_b = history_store.lookup_history(SID_B, "QPS 是多少", top_k=3)
print("  命中 %d 块" % len(hits_b))
for i, d in enumerate(hits_b, 1):
    print("   %d. %s" % (i, (d.page_content or "")[:78].replace("\n", " ")))
RES.append(("B 检索不到 A 的内容（隔离生效）",
            not any("184.54" in (d.page_content or "") for d in hits_b),
            [d.page_content[:40] for d in hits_b]))
RES.append(("B 命中的块都属于 session-B",
            all((getattr(d, "metadata", None) or {}).get("filename") == SID_B for d in hits_b)))

print()
print("=" * 98)
print("判读")
print("=" * 98)
ok = sum(1 for _, c, _ in RES if c)
for item in RES:
    name, cond = item[0], item[1]
    val = item[2] if len(item) > 2 else ""
    print("  %s %-40s %s" % ("[OK ]" if cond else "[FAIL]", name,
                             ("实际 %r" % (val,)) if not cond else ""))
print("  通过 %d / %d" % (ok, len(RES)))
sys.exit(0 if ok == len(RES) else 1)
