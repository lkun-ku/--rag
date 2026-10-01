"""探针：判分链（`with_structured_output`）的响应里，到底有没有 `usage_metadata`？

## 背景

`utils/tracer.py` 用 `UsageMetadataCallbackHandler` 统计 token —— 它**只在响应的
`usage_metadata` 存在时**才记账。而判分链是：

    structured_llm_grader = llm.with_structured_output(GradeDocuments, method=...)
    retrieval_grader_chain = grade_prompt | structured_llm_grader

**`with_structured_output` 会把结果转成 Pydantic 对象**，很可能让回调收到的
`response` 里没有 `usage_metadata` —— 那样即便 config 透传了，
`UsageMetadataCallbackHandler` 也**记不到 token**（这解释了修复后 total 只 +637）。

## 探针要回答三个问题

1. 用 `UsageMetadataCallbackHandler` 跑判分链，`usage_metadata` 里有东西吗？
2. `on_llm_end` 收到的 `response` 长什么样（有没有 `generations` / `message`）？
3. 对比：**同一模型不用 `with_structured_output`** 时能否被统计（对照组）？
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


class ShapeCB(BaseCallbackHandler):
    """把 on_llm_end 收到的 response 结构打印出来。"""

    def __init__(self):
        self.shapes = []

    def on_llm_end(self, response, **kwargs):
        gens = getattr(response, "generations", None) or []
        info = {"n_gen_groups": len(gens)}
        try:
            g0 = gens[0][0] if gens and gens[0] else None
            info["gen_type"] = type(g0).__name__
            msg = getattr(g0, "message", None)
            info["has_message"] = msg is not None
            info["msg_type"] = type(msg).__name__ if msg is not None else None
            info["has_usage_metadata"] = bool(getattr(msg, "usage_metadata", None)) if msg is not None else False
            info["llm_output_keys"] = list((getattr(response, "llm_output", None) or {}).keys())
        except Exception as e:  # noqa: BLE001
            info["err"] = "%s: %s" % (type(e).__name__, e)
        self.shapes.append(info)


print("=" * 100)
print("探针：判分链的 token 能否被 UsageMetadataCallbackHandler 统计")
print("=" * 100)

from graph2.grader_chain import retrieval_grader_chain  # noqa: E402
from llm_models.all_llm import llm  # noqa: E402

Q = "Milvus-16c64f-sq4u-fp16-force_merge 的 QPS 是多少？"
DOC = "表格：检索性能实测（第 1-30 行）\n| 场景 | 产品与规格 | QPS |\n| 静态向量检索 | Milvus-16c64g-sq4u-fp16-force_merge | 9575.69 |"

# --- ① 判分链（with_structured_output）---
print()
print("[① 判分链 with_structured_output]")
u1, s1 = UsageMetadataCallbackHandler(), ShapeCB()
try:
    r1 = retrieval_grader_chain.invoke({"question": Q, "document": DOC}, config={"callbacks": [u1, s1]})
    print("  返回: %r" % (getattr(r1, "binary_score", r1),))
except Exception as e:  # noqa: BLE001
    print("  调用失败: %s: %s" % (type(e).__name__, e))
print("  usage_metadata: %r" % (getattr(u1, "usage_metadata", None),))
print("  on_llm_end 收到的结构: %r" % (s1.shapes,))

# --- ② 对照组：同一个 llm，不带结构化输出 ---
print()
print("[② 对照组：同一个 llm，直接调用（无 with_structured_output）]")
u2, s2 = UsageMetadataCallbackHandler(), ShapeCB()
try:
    r2 = llm.invoke("只回答一个词：yes 或 no。问题：Milvus 是向量数据库吗？", config={"callbacks": [u2, s2]})
    print("  返回类型: %s" % type(r2).__name__)
except Exception as e:  # noqa: BLE001
    print("  调用失败: %s: %s" % (type(e).__name__, e))
print("  usage_metadata: %r" % (getattr(u2, "usage_metadata", None),))
print("  on_llm_end 收到的结构: %r" % (s2.shapes,))

print()
print("=" * 100)
print("判读")
print("=" * 100)
ok1 = bool(getattr(u1, "usage_metadata", None))
ok2 = bool(getattr(u2, "usage_metadata", None))
print("  判分链可被统计: %s" % ("是" if ok1 else "**否** <- 这就是 total 只 +637 的原因"))
print("  直接调用可被统计: %s" % ("是" if ok2 else "**否**"))
if ok2 and not ok1:
    print()
    print("  -> 根因确认：`with_structured_output` 让回调拿不到 usage_metadata。")
    (print("     修法方向：让判分链内部**多暴露一层**（如改用 tool-calling 并手工取"),)
    print("     `response.usage_metadata`），或在 tracer 侧补一条路径。")
