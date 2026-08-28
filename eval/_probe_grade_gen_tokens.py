"""最小探针：`grade_generation_v_documents_and_question` 的两次判定，token 到底统计到了吗？

## 背景

4.25 给它加了并发 + config 透传，端到端实测 `grade_generation` 耗时 1.42s -> 0.68s（有效），
但 **total token 与改前完全相同（31,465）**，强烈提示那两次 LLM 调用**仍未被统计**。

而前一个探针已经证明「条件边函数能拿到 config/callbacks」——
所以问题要么在 `_sub_config` 的取值上，要么在别处。**直接测这个函数，不绕图。**
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

Q = "Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 是多少？"
DOCS = [Document(page_content="表格：检索性能实测\n| 产品与规格 | QPS |\n"
                              "| Milvus-16c64g-sq4u-fp16-force_merge | 9575.69 |",
                 metadata={"filename": "t.md"})]
GEN = "Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 是 9,575.69 [1]。"
STATE = {"question": Q, "documents": DOCS, "generation": GEN}


class Counting(BaseCallbackHandler):
    def __init__(self):
        self.n = 0

    def on_llm_end(self, response, **kwargs):
        self.n += 1


print("=" * 96)
print("探针：grade_generation 内部两次 LLM 调用的 token 统计")
print("=" * 96)

from graph2.graph_2 import grade_generation_v_documents_and_question as fn  # noqa: E402

for label, cfg in (("传 config（含 callbacks）", None), ("**不传** config（对照）", "SKIP")):
    u, c = UsageMetadataCallbackHandler(), Counting()
    config = {"callbacks": [u, c]} if cfg is None else None
    try:
        decision = fn(dict(STATE), config)
    except Exception as e:  # noqa: BLE001
        print("  [%s] 调用失败: %s: %s" % (label, type(e).__name__, e))
        continue
    meta = dict(getattr(u, "usage_metadata", {}) or {})
    print()
    print("  [%s]" % label)
    print("     决策: %s" % decision)
    print("     on_llm_end 触发: %d 次" % c.n)
    print("     usage_metadata: %r" % (meta or "（空）",))

print()
print("=" * 96)
print("判读")
print("=" * 96)
print("  * 传 config 时 on_llm_end 应 > 0 且 usage_metadata 非空；")
print("  * 不传 config 时两者都应为 0/空（否则说明统计来自别处，结论不成立）。")
