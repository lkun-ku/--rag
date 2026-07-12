"""检索工具：跨多集合的混合检索（稠密 + BM25，两阶段 RRF 融合）。

背景（ROADMAP 1.2 的配套改动）：
    入库改成「一源一集合」后，检索侧必须能跨集合查 —— 否则「多源」形同虚设
    （实测：只连默认集合时，问任何问题都只返回 milvus_docs 的内容，
     论文与手册集合虽然数据齐全却永远检索不到）。

检索策略（两阶段 RRF）：

    阶段一（集合内）：每个集合各自做 hybrid_search
        dense(bge-large 1024) + sparse(Milvus 内置 BM25) --RRF--> 该集合 top-k

    阶段二（跨集合）：把各集合的结果按 RRF 再次融合
        score = Σ 1/(rrf_k + rank)   （同内容块分数累加，实现去重）

    为什么用 RRF 而不是分数排序：不同集合的原始分数（IP 内积 / BM25）量纲不同、
    不可直接比较；RRF 只看排名，天然规避这个问题 —— 与项目在集合内做混合检索时
    的选择一致。

接口保持向后兼容：`retriever` 与 `retriever_tool` 名称、类型、调用方式均不变，
因此 graph1 / graph2 / rag_agent 三个入口**零改动**。
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from pymilvus import Function
from pymilvus.client.types import FunctionType

from documents.milvus_db import MilvusVectorSave
from utils.citation import format_context
from utils.env_utils import DOC_SOURCES, MILVUS_RERANKER_STYLE
from utils.log_utils import log

# 单集合召回条数。
PER_COLLECTION_K = 6
# 最终交给下游的条数。
FINAL_K = 18
# 阶段二 RRF 的平滑常数
RRF_K = 60

# [3.0.3 修复] 混合检索的「每路预取数」。
#
# ## 问题（实测定位）
#
# langchain_milvus 的 `_collection_hybrid_search` 里：
#
#     request = AnnSearchRequest(data=[...], anns_field=field,
#                                param=param_dict, limit=fetch_k, expr=expr)
#     ...
#     self.client.hybrid_search(..., ranker=rrf, limit=k, ...)
#
# 其中 `fetch_k` 是**每路（dense / sparse）各自预取多少条**，
# 而它的默认值是 **4**，且 **`k` 完全不影响它**。
#
# 后果：两路各只预取 4 条 -> RRF 融合池最多 8 条 -> 无论 `k` 调多大，
# 单集合最终**最多只能返回 8 块**（去重后更少）。这正是实测「K 调大无效」的原因。
#
# 它同时解释了 1.5 参数扫描里的一组数据：
#     k=6  final_k=18  -> 平均返回 14.8 块
#     k=12 final_k=36  -> 平均返回 16.1 块   （3 集合 × 8 = 24 为上限，去重后即此数）
#
# ## 为什么 `fetch_k` 调大是安全的
#
# `limit=k` 依然生效，所以**返回块数不会变多** —— 变大的只是「参与排序的候选池」，
# 排序结果因此更准。这对 q043 那种「正确块排名靠后」的问题是对症的。
#
# ## 取值 12：由 57 题参数扫描实测确定（不是拍脑袋）
#
#   fetch_k   context_recall   未满分题数   每集合状态
#   ---------------------------------------------------------------
#   默认(4)      0.8801          8 题       候选池太小，q043 捞不到
#   8            0.9181          6 题
#   12           0.9327          5 题       <- 采用
#   16           0.9197          8 题
#   24           0.9060          8 题
#   32           0.8947          9 题
#   50           0.8626          9 题       融合被"两路共识"主导，q008/q013/q015 掉出
#   120          0.8012         13 题
#
# 规律：**太小**（候选池不足，正确块进不了融合）与**太大**（RRF 的
# `1/(rrf_k + rank + 1)` 让「两路都中等」的块压过「单路极强」的块）都不好，
# 12 是平衡点。它恰好等于 2 × PER_COLLECTION_K —— 每路预取 2 倍于最终配额，
# 保证 RRF 融合后仍有足够候选填满 k，同时不引入过多噪声参与排序。
#
# 注：bge-reranker-v2-m3 已验证能显著改善排序（能把第 30 名提到第 1 名），
# 但本机无 GPU，CPU 推理约 1s/块（单题 18 块需 16s），线上不可接受，故暂不接入。
FETCH_K_PER_ROUTE = 12

# ---------------------------------------------------------------------------
# 上面两个参数由**参数扫描实测**确定（45 条 Golden Dataset，见 ROADMAP 1.5）：
#
#   每集合k  final_k   Context Recall   未命中   平均返回块数
#   ------------------------------------------------------------
#     4        6        0.6741          13        6.0      ← 原值
#     4       12        0.8259           7       12.0
#     6       12        0.8259           7       12.0
#     6       18        0.8926           4       14.8      ← 采用
#     8       16        0.8926           4       15.4
#     8       24        0.8926           4       16.1
#    12       36        0.8926           4       16.1
#
# 结论：
#   * 原值 k=4/final_k=6 时 Recall 只有 0.674 —— 3 个集合共 12 条候选里
#     只取 6 条，把各集合第 2~4 名大量挤掉（13/45 题完全未命中）。
#   * k=6/final_k=18 是**达到 Recall 上限的最小代价组合**；
#     再加大 k 只增加返回块数（16.1 块/问），Recall 不再提升 —— 说明瓶颈
#     已不在召回数量。
#   * 代价：每问多约 9 个块进入上下文。下游 `grade_documents_node` 的
#     `grader_chain` 会做相关性过滤，因此实际喂给 LLM 的远少于 18 块。
# ---------------------------------------------------------------------------


def _build_search_kwargs(expr: str = 'category == "content"') -> dict:
    """构造 as_retriever 的 search_kwargs（reranker 写法由配置决定）。"""
    if MILVUS_RERANKER_STYLE == "function":
        # RRF 形式的 rerank Function 有两个易错点（已在真实 Milvus 2.6.25 实测确认）：
        #   1) params 里必须带 "reranker" 键指明是哪一种 rerank
        #   2) RRF 不接受输入字段，input_field_names 必须为空列表
        ranker_kwargs = {
            "reranker": Function(
                name="rrf",
                function_type=FunctionType.RERANK,
                input_field_names=[],
                params={"reranker": "rrf", "k": 100},
            )
        }
    else:
        ranker_kwargs = {"ranker_type": "rrf", "ranker_params": {"k": 100}}

    return {
        "k": PER_COLLECTION_K,
        # [3.0.3 修复] 每路预取数。不传则用 langchain_milvus 的默认值 4，
        # 会让单集合的最终返回被硬顶在 8 条以内（详见 FETCH_K_PER_ROUTE 的说明）。
        # 它经 as_retriever(search_kwargs=...) -> similarity_search(**kwargs)
        # -> similarity_search_with_score(**kwargs) -> _collection_hybrid_search(**kwargs)
        # 透传，最终被 AnnSearchRequest 的 limit 参数消费（已读源码确认这条链路）。
        "fetch_k": FETCH_K_PER_ROUTE,
        # 注意：langchain-milvus 与 langchain-core 均不消费此参数（一直是惰性的），保留原值
        "score_threshold": 0.1,
        **ranker_kwargs,
        # expr 才是 Milvus 真正生效的过滤写法。
        # （原写作 'filter': {...} 会被塞进 **kwargs 透传给 MilvusClient.hybrid_search()，
        #   而该函数没有 filter 形参，因此自项目创建以来从未生效。）
        "expr": expr,
    }


def _fingerprint(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


class MultiCollectionRetriever(BaseRetriever):
    """跨多个 Milvus 集合的混合检索器（两阶段 RRF 融合）。

    只有 1 个集合时行为等价于原来的单集合检索器（多一步无副作用的 RRF）。
    """

    # {collection_name: retriever}
    retrievers: Dict[str, object] = {}
    per_collection_k: int = PER_COLLECTION_K
    final_k: int = FINAL_K
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:  # noqa: D102
        ranked: List[tuple] = []   # (collection, rank, doc)
        errors = []
        for name, r in self.retrievers.items():
            try:
                hits = r.invoke(query)
                for rank, doc in enumerate(hits):
                    ranked.append((name, rank, doc))
            except Exception as e:  # noqa: BLE001
                # 单个集合失败不影响其他集合（例如某集合被 drop 或暂时不可用）
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
                log.warning(f"集合 {name} 检索失败（继续其他集合）: {type(e).__name__}: {e}")

        if not ranked:
            # 不静默：全部集合都失败时明确报错级别日志，便于与「确实没找到」区分
            if errors:
                log.error(f"所有集合检索均失败：{errors}")
            return []

        # 阶段二：跨集合 RRF（按排名融合，规避不同集合分数量纲不一致）
        scores: Dict[str, float] = {}
        docs: Dict[str, Document] = {}
        for _name, rank, doc in ranked:
            key = _fingerprint(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            docs.setdefault(key, doc)  # 同内容块只保留一份

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [docs[k] for k, _ in ordered[: self.final_k]]


def _discover_collections() -> List[str]:
    """从 DOC_SOURCES 收集启用源的目标集合（去重、保持顺序）。"""
    seen, out = set(), []
    for s in DOC_SOURCES:
        c = s.get("collection")
        if s.get("enabled", True) and c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_multi_retriever() -> MultiCollectionRetriever:
    """为每个启用集合各建一个 retriever，组装成跨集合检索器。

    集合不存在时跳过（首次运行还没入库时的正常情况），只记 warning。
    """
    from pymilvus import MilvusClient

    from utils.env_utils import MILVUS_URI

    try:
        existing = set(MilvusClient(uri=MILVUS_URI).list_collections())
    except Exception as e:  # noqa: BLE001
        log.warning(f"无法列出集合（Milvus 可能未启动），将尝试按配置连接: {e}")
        existing = None

    kwargs = _build_search_kwargs()
    built: Dict[str, object] = {}
    for coll in _discover_collections():
        if existing is not None and coll not in existing:
            log.warning(f"集合 {coll} 尚不存在，跳过（入库后重启即可生效）")
            continue
        try:
            mv = MilvusVectorSave(collection_name=coll)
            mv.create_connection()
            built[coll] = mv.vector_store_saved.as_retriever(
                search_type="similarity", search_kwargs=kwargs
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"集合 {coll} 检索器构造失败，跳过: {type(e).__name__}: {e}")

    if not built:
        # 兜底：一个都没建成时，退回默认集合，让错误在真正检索时暴露得更清楚
        log.warning("未构造出任何集合检索器，回退到默认集合 COLLECTION_NAME")
        mv = MilvusVectorSave()
        mv.create_connection()
        built[mv.collection_name] = mv.vector_store_saved.as_retriever(
            search_type="similarity", search_kwargs=kwargs
        )

    log.info(f"跨集合检索器已构造，覆盖集合: {list(built.keys())}")
    return MultiCollectionRetriever(retrievers=built)


retriever = _build_multi_retriever()


@tool('rag_retriever')
def retriever_tool(query: str) -> str:
    """搜索并返回知识库中的信息，覆盖多个来源：Milvus 向量数据库文档（架构与部署、索引类型与指标、性能调优、运维监控、常见问题排查）、学术论文、产品手册与性能测试数据。多来源内容分别存储、统一检索。

    工具描述必须与语料一致：原描述写的是「半导体和芯片」，而 datas/ 下的实际语料是
    Milvus 文档 + 论文 + 手册。描述不符会让 create_agent（rag_agent.py）判定
    「工具与问题无关」而直接拒答、不发起检索（实测会回答「我唯一能用的工具是半导体检索器」）。

    [UPGRADE 2026-10-09 / ROADMAP 2.1] 为什么从 `create_retriever_tool` 改为自定义 tool：
        引用溯源要求上下文中每个块**带编号**（`[1] xxx`），且编号必须与 documents 下标
        严格对应，供下游校验层反查原文。而 graph1 的上下文在 `generate` 节点里已是
        拼接好的字符串（块边界丢失），只能在工具层编号。
        实测 `create_retriever_tool` 的 `document_prompt` 变量**只能取自 document.metadata**，
        传 `{index}` 会抛 ValueError，无法加序号，故改为自定义实现。
        工具名（rag_retriever）、入参（query: str）、描述语义均保持不变，
        `llm.bind_tools([retriever_tool])` 的用法不受影响。
    """
    docs = retriever.invoke(query)
    return format_context(docs)
