"""Parent-Child 检索器：**子块检索 → 父块回查 → 按命中数排序**（ROADMAP 4.52）。

## 为什么不能「直接检索父块」

父块是一个完整章节（几千字），把它整体向量化会**稀释语义**：
一个章节里只要有一句相关，整章向量就被拉向「平均含义」，反而不如子块精准。

**small-to-big 的核心思路**：

    检索用**子块**（小而准）  →  生成用**父块**（大而全）

## 排序为什么不能只看父块相似度

父块很长，向量被稀释。正确做法是**用子块的证据给父块打分**：

    score(parent) = 0.6 * norm(命中子块数) + 0.4 * max(子块相似度)

* **命中子块数**：越多个子块命中同一个父块，说明这个章节越相关（多证据支持）
* **最高子块相似度**：即使只命中 1 个子块，只要它极度相关，父块也值得看

## 与数据无关

检索流程、排序公式、批量回查、降级策略 **全部与语料无关**，
用假 client 即可完整测试。只有 `top_k_children` / `top_n_parents` 两个参数
需要等真实数据调。
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document

from rag_kb.utils.log_utils import log


def _flag(new_name: str, old_name: str, default: str = "0") -> bool:
    """开关读取：**新名优先、旧名兜底** —— 改名前后的 `.env` 都能用。"""
    v = os.getenv(new_name)
    if v is None:
        v = os.getenv(old_name, default)
    return v not in ("", "0", "false", "False")


# 总开关：默认关（开启需要「有 parent_id 的集合」，即必须重建索引）
PARENT_CHILD_ENABLED = _flag("KB_PARENT_CHILD", "PARENT_CHILD_ENABLED", "0")

DEFAULT_TOP_K_CHILDREN = 24
DEFAULT_TOP_N_PARENTS = 6

# 排序权重：命中子块数 vs 最高相似度
_W_HITS = 0.6
_W_SIM = 0.4

# 子块检索的过滤条件 —— **必须只查子块**，否则父块也会被当成子块参与检索
CHILD_FILTER = "is_parent == false"


def build_parent_expr(parent_ids: List[str]) -> str:
    """构造「批量回查父块」的 expr。**一次查完，不做 N+1**。"""
    quoted = ", ".join('"%s"' % p.replace('"', "'") for p in parent_ids)
    return "is_parent == true and parent_id in [%s]" % quoted


def _normalize(values: List[float]) -> List[float]:
    """把数值归一化到 0~1（全相同时返回全 0，避免除零）。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_with_parents(
    question: str,
    child_retriever,
    client,
    collection_name: str,
    top_k_children: int = DEFAULT_TOP_K_CHILDREN,
    top_n_parents: int = DEFAULT_TOP_N_PARENTS,
) -> Optional[List[Document]]:
    """检索子块 -> 批量回查父块 -> 按证据排序 -> 组装。

    Args:
        question: 用户问题。
        child_retriever: 已配置 `expr=CHILD_FILTER` 的子块检索器（有 `.invoke`）。
        client: Milvus 客户端（有 `.query(collection, filter=, output_fields=, limit=)`）。
        collection_name: 集合名。
        top_k_children: 取多少个命中子块参与聚合。
        top_n_parents: 最终返回几个父块。

    Returns:
        list[Document]：**父块**（`page_content` 为父块全文），每个父块的
        `metadata` 里带 `hit_count` / `max_child_sim` / `matched_children`
        （后者是命中的子块文本列表，供生成时标出「答案可能在这里」）。

        **开关关闭时返回 `None`** —— 调用方据此走原有路径；
        正常路径（开关开启但无命中）返回 `[]`。这两者**不能混淆**。
    """
    if not PARENT_CHILD_ENABLED:
        return None

    try:
        kids = list(child_retriever.invoke(question))[:top_k_children]
    except Exception as e:  # noqa: BLE001
        log.warning("[pc] 子块检索失败，降级: %s: %s" % (type(e).__name__, str(e)[:100]))
        return None
    if not kids:
        return []

    # ---- 1. 按 parent_id 聚合证据 ----
    agg: dict[str, dict] = {}
    for d in kids:
        md = getattr(d, "metadata", None) or {}
        pid = md.get("parent_id")
        if not pid:
            continue
        sim = float(md.get("score") or 0.0)
        a = agg.setdefault(pid, {"hits": 0, "max_sim": 0.0, "kids": []})
        a["hits"] += 1
        a["max_sim"] = max(a["max_sim"], sim)
        a["kids"].append(d.page_content)

    if not agg:
        return []

    # ---- 2. 批量回查父块（**一次**，不做 N+1）----
    pids = list(agg.keys())
    try:
        rows = client.query(
            collection_name,
            filter=build_parent_expr(pids),
            output_fields=["text", "parent_id", "heading_path", "filename"],
            limit=len(pids) * 2,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[pc] 父块回查失败，降级: %s: %s" % (type(e).__name__, str(e)[:100]))
        return None

    parents: List[Document] = []
    for r in rows or []:
        pid = r.get("parent_id")
        if pid not in agg:
            continue
        a = agg[pid]
        parents.append(Document(
            page_content=r.get("text") or "",
            metadata={
                "filename": r.get("filename") or "",
                "parent_id": pid,
                "is_parent": True,
                "heading_path": r.get("heading_path") or "",
                "hit_count": a["hits"],
                "max_child_sim": a["max_sim"],
                "matched_children": a["kids"],
            },
        ))
    if not parents:
        return []

    # ---- 3. 按证据排序 ----
    hits_norm = _normalize([float(p.metadata["hit_count"]) for p in parents])
    sims_norm = _normalize([float(p.metadata["max_child_sim"]) for p in parents])
    for p, hn, sn in zip(parents, hits_norm, sims_norm):
        p.metadata["pc_score"] = _W_HITS * hn + _W_SIM * sn
    parents.sort(key=lambda d: d.metadata["pc_score"], reverse=True)

    log.info("[pc] 子块 %d -> 父块 %d（命中 pid 数 %d），取前 %d"
             % (len(kids), len(parents), len(pids), top_n_parents))
    return parents[:top_n_parents]


def make_child_retriever(client, collection_name: str, embedding, top_k: int = 24):
    """构造「只检索子块」的检索器（供 `retrieve_with_parents` 用）。

    为什么单独封装：`expr` 必须固定为 `is_parent == false`，
    交给调用方每次手写容易漏 —— **漏了就会把父块当子块检索，父块体积大、语义稀释，
    反而拉低精度**。这里固定写死。
    """
    from pymilvus import MilvusClient  # noqa: F401  (仅用于类型提示的场景)

    from rag_kb.documents.milvus_db import MilvusVectorSave

    mv = MilvusVectorSave(collection_name=collection_name)
    mv.create_connection()
    return mv.store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, "expr": CHILD_FILTER},
    )
