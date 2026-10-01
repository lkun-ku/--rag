"""企业级精排链路 —— 契约测试（ROADMAP 4.42，TDD 先行）。

## 背景：四个该上而没上的条件

| 条件 | 本项目现状 | 本测试锁定 |
|---|---|---|
| ① 延迟预算撑不住 LLM 判分 | 判分常态在跑，token 占 76% | reranker 可开关，开启后**先精排再判分**（省判分 token） |
| ② 检索深度 k 被迫加大 | `PER_COLLECTION_K=12` 恒定 | 支持**深检索**（按查询类型提 k），reranker 负责收敛 |
| ③ 跨块/跨文档聚合 | 已有同表合并（`merge_table_slices`），但未识别「聚合类」问题 | `is_aggregation_query` + 深检索 |
| ④ 需要硬过滤省 LLM 成本 | 只用 LLM 判分，无硬门槛 | **用 reranker 分数做门槛**（不是裸 cosine） |

## 契约

| # | 契约 | 为什么重要 |
|---|---|---|
| C1 | **默认关闭**，`enterprise_rerank` 原样返回 | 保持 24 个提交的既有行为，回归才可信 |
| C2 | 开启后按 reranker 分数**降序** | 精排的定义 |
| C3 | `top_k` 截断生效 | k 加大后必须收敛 |
| C4 | `score_threshold` **剔除**低于阈值者 | 省 LLM 判分成本的关键 |
| C5 | reranker 抛异常 -> **降级返回原文** | 企业级可观测要求：不因单点故障断链 |
| C6 | 空列表 / None 安全 | 边界 |
| C7 | `is_aggregation_query` 正确识别 | 深检索与强制合并的触发条件 |
| C8 | 深检索 k 常量可读且 > 默认 | 配置驱动 |

**刻意不测**：真实模型的排序质量（那需要标注数据，属于离线评测），
本测试只锁定**工程契约**；A/B 质量对比由 `eval/_ab_rerank.py` 单独做。
"""

from __future__ import annotations

import inspect
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.documents import Document  # noqa: E402

import tools.reranker_tools as rt  # noqa: E402

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


def mk(i, text):
    return Document(page_content="chunk-%d %s" % (i, text), metadata={"filename": "f.md", "title": "t"})


print("=" * 100)
print("C1 默认关闭 -> 原样返回（保持既有行为）")
print("=" * 100)
rt._reset_for_test()  # 回到默认（未启用）状态
docs = [mk(1, "a"), mk(2, "b"), mk(3, "c")]
out = rt.enterprise_rerank("问题", docs)
check("块数不变", len(out) == 3, len(out))
check("顺序不变", [d.page_content for d in out] == [d.page_content for d in docs])

print()
print("=" * 100)
print("C2/C3/C4 开启后：降序 + top_k 截断 + 阈值过滤")
print("=" * 100)


# 用假 reranker 替换真实模型（测试要秒级完成）
class FakeReranker:
    """分数 = 块号（1-based）：chunk-3 得 3.0、chunk-2 得 2.0、chunk-1 得 1.0。

    这样降序后必然反转为 chunk-3, chunk-2, chunk-1 —— 一眼能看出排序是否生效。
    """

    def rank(self, query, docs):
        return [(i, float(i + 1)) for i in range(len(docs))]


rt._reset_for_test(enabled=True, reranker=FakeReranker(), top_k=0, threshold=0)
out = rt.enterprise_rerank("问题", [mk(1, "a"), mk(2, "b"), mk(3, "c")])
check(
    "C2 按分数降序（3,2,1）",
    [d.page_content.split()[0] for d in out] == ["chunk-3", "chunk-2", "chunk-1"],
    [d.page_content.split()[0] for d in out],
)

rt._reset_for_test(enabled=True, reranker=FakeReranker(), top_k=2, threshold=1.5)
out = rt.enterprise_rerank("问题", [mk(1, "a"), mk(2, "b"), mk(3, "c")])
check("C3 top_k=2 截断", len(out) == 2, len(out))
# 分数为 1.0/2.0/3.0；threshold=1.5 应剔掉 1.0，保留 2.0 与 3.0
check("C4 threshold=1.5 剔除 1.0（剩 chunk-3/chunk-2）", len(out) == 2, len(out))

rt._reset_for_test(enabled=True, reranker=FakeReranker(), top_k=0, threshold=2.5)
out = rt.enterprise_rerank("问题", [mk(1, "a"), mk(2, "b"), mk(3, "c")])
check(
    "threshold 生效（只留 chunk-3）",
    len(out) == 1 and out[0].page_content.startswith("chunk-3"),
    [d.page_content.split()[0] for d in out],
)

print()
print("=" * 100)
print("C5 reranker 抛异常 -> 降级返回原文（不阻断主流程）")
print("=" * 100)


class BoomReranker:
    def rank(self, q, docs):
        raise RuntimeError("模型加载失败")


rt._reset_for_test(enabled=True, reranker=BoomReranker(), top_k=2, threshold=0.0)
out = rt.enterprise_rerank("问题", [mk(1, "a"), mk(2, "b"), mk(3, "c")])
check("异常时返回原始 3 块（顺序不变）", len(out) == 3 and out[0].page_content.startswith("chunk-1"), len(out))

print()
print("=" * 100)
print("C6 边界：空列表 / None")
print("=" * 100)
rt._reset_for_test(enabled=True, reranker=FakeReranker())
check("空列表 -> 空列表", rt.enterprise_rerank("q", []) == [])
check("None -> []", rt.enterprise_rerank("q", None) == [])

print()
print("=" * 100)
print("C7 is_aggregation_query")
print("=" * 100)
agg_true = [
    "参与评测的产品里，召回率最低的是哪个？",
    "哪个产品 QPS 最高？",
    "这些产品的平均召回率是多少？",
    "一共有多少个产品参与评测？",
    "对比一下 TurboPuffer 和 Milvus",
]
agg_false = ["Milvus 怎么安装？", "HNSW 索引的原理是什么？", "如何配置 Milvus 集群？", "P99 延迟的定义"]
for q in agg_true:
    check("识别聚合类: %s" % q[:26], rt.is_aggregation_query(q) is True)
for q in agg_false:
    check("非聚合类: %s" % q[:26], rt.is_aggregation_query(q) is False)

print()
print("=" * 100)
print("C8 深检索配置")
print("=" * 100)
from tools.retriever_tools import (
    FINAL_K,
    FINAL_K_DEEP,
    PER_COLLECTION_K,  # noqa: E402
    PER_COLLECTION_K_DEEP,
)

check("PER_COLLECTION_K_DEEP 存在且为 int", isinstance(PER_COLLECTION_K_DEEP, int), PER_COLLECTION_K_DEEP)
check(
    "深检索 k > 默认 k",
    PER_COLLECTION_K_DEEP > PER_COLLECTION_K,
    "deep=%s default=%s" % (PER_COLLECTION_K_DEEP, PER_COLLECTION_K),
)
check("深检索的 final_k 同步放大", FINAL_K_DEEP > FINAL_K, "deep=%s default=%s" % (FINAL_K_DEEP, FINAL_K))

print()
print("=" * 100)
print("C9 配置开关契约")
print("=" * 100)
sig = inspect.signature(rt.enterprise_rerank)
check("enterprise_rerank(query, docs) 签名", list(sig.parameters)[:2] == ["query", "docs"], str(list(sig.parameters)))
check("有 rerank_enabled 供调用方读取", isinstance(rt.rerank_enabled(), bool))

print()
print("=" * 100)
print("C10/C11 深检索检索器（会连 Milvus，约几秒）")
print("=" * 100)
import tools.retriever_tools as rtt  # noqa: E402

sig = inspect.signature(rtt.get_retriever)
check("get_retriever 有 deep 参数", "deep" in sig.parameters, str(list(sig.parameters)))
check("deep 默认 False（非聚合类零差异）", sig.parameters["deep"].default is False)

g_normal = rtt.get_retriever()
g_deep = rtt.get_retriever(deep=True)
check("deep=True 返回**不同**检索器", g_normal is not g_deep)
g_deep2 = rtt.get_retriever(deep=True)
check("get_deep_retriever 单例（不重复连库）", g_deep is g_deep2 and rtt.get_deep_retriever() is g_deep)

print()
print("=" * 100)
print("C12 retrieve 节点的接线")
print("=" * 100)
from graph2.retriever_node import retrieve  # noqa: E402

src = inspect.getsource(retrieve)
check("retrieve 调用了 enterprise_rerank", "enterprise_rerank" in src)
check("retrieve 识别聚合类", "is_aggregation_query" in src)
check("retrieve 深检索开关", "deep=is_agg" in src)
rsig = inspect.signature(retrieve)
check("retrieve 接收 config（不透传会丢 token 统计）", "config" in rsig.parameters, str(list(rsig.parameters)))

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
