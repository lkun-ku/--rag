"""企业级精排链路：reranker 精排 + **reranker 分数门槛** + 聚合类识别（ROADMAP 4.42）。

## 为什么现在做（四个该上而没上的条件）

| 条件 | 现状 | 本模块的应对 |
|---|---|---|
| ① 延迟预算撑不住 LLM 判分 | 判分占总成本 76%，每块都要 LLM 过 | **先用 reranker 收敛**，只把高分的块送判分 -> 省 token |
| ② 检索深度 k 被迫加大（召回优先） | `PER_COLLECTION_K=12` 恒定 | 加深检索到 `PER_COLLECTION_K_DEEP`，**由 reranker 负责收敛** |
| ③ 跨块/跨文档聚合 | 已有 `merge_table_slices`，但未识别「聚合类」 | `is_aggregation_query()` + 深检索 |
| ④ 需要硬过滤省 LLM 成本 | 只有 LLM 判分这一道 | **用 reranker 分数做门槛**（不是裸 cosine —— 后者跨查询不可比） |

## 三条设计约束

1. **默认关闭**：`RERANK_ENABLED=0`（默认）时 `enterprise_rerank` 原样返回，
   **24 个提交的既有行为与全部指标零变化**；企业级部署时开 `RERANK_ENABLED=1`。
2. **失败降级**：模型加载失败 / 推理抛异常 -> 记 warning 并**返回原文**，
   绝不让单点故障断掉整个问答（ROADMAP 1.3.6「不得静默降级」的反面是
   **可降级但要留痕**）。
3. **分数门槛用 reranker 分数**：cosine/IP 分数跨查询不可比（同一阈值对不同查询
   一松一紧，ROADMAP 4.10 实测伤召回）；cross-encoder 分数更接近真实相关度。
   `RERANK_SCORE_THRESHOLD=0` 表示不过滤，只为「硬过滤省成本」准备。

## 与 `measure_rerank_gain.py` 的关系

那个脚本测的是**质量**（reranker 能否把 30 名提到 1 名），结论是「本项目天花板只剩 9%，
增益 0」。本模块不推翻那个结论——**它提供的是「开关」而不是「默认开启」**：
一旦业务侧需要加大 k 或提高精度，这个开关就是现成的，不必再改代码。
"""
from __future__ import annotations

import os
import pathlib
import re

from utils.log_utils import log

# ---------------------------------------------------------------------------
# 配置（环境变量驱动，默认关闭）
# ---------------------------------------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") not in ("", "0", "false", "False")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "2000"))   # 单块送去打分的截断长度

_HERE = pathlib.Path(__file__).resolve().parent
_MODEL_ROOT = _HERE.parent / "models"
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    str(_MODEL_ROOT / "bge-reranker-v2-m3"),      # **默认本地路径**，不走网络下载
)

# 聚合类问题的特征词：命中说明用户要「跨行比较/统计」，需要看完整表
_AGG_PATTERN = re.compile(
    r"最低|最高|最大|最小|最多|最少|平均|中位|一共|总共|总计|对比|比较|排名|"
    r"哪个最|哪些最|第几|多少最|相差|差距|占比|百分比是多少"
)

# ---------------------------------------------------------------------------
# 内部状态（惰性单例 + 测试注入点）
# ---------------------------------------------------------------------------
_reranker = None
_reranker_failed = False          # 加载失败后不再重试（避免每次问答都卡一遍）
_override: dict = {}              # 仅供测试注入，生产永远为空


def _reset_for_test(enabled=None, reranker=None, top_k=None, threshold=None):
    """**仅供测试**：重置模块状态并注入假 reranker。

    生产代码不应调用它。测试用它（a）开开关、（b）注入秒级的假模型，
    从而不加载真实模型也能锁定工程契约。
    """
    global RERANK_ENABLED, _reranker, _reranker_failed
    if enabled is not None:
        RERANK_ENABLED = enabled
    if top_k is not None:
        _override["top_k"] = top_k
    elif "top_k" in _override:
        del _override["top_k"]
    if threshold is not None:
        _override["threshold"] = threshold
    elif "threshold" in _override:
        del _override["threshold"]
    _reranker = reranker
    _reranker_failed = reranker is None and enabled is True and "injected" not in _override
    _override["injected"] = reranker is not None


def rerank_enabled() -> bool:
    """是否启用精排。调用方（如 retriever_node / 诊断脚本）用它做分支。"""
    return bool(RERANK_ENABLED)


class _CrossEncoderWrapper:
    """把 `CrossEncoder` 包成本模块约定的 `.rank(query, docs)` 接口。"""

    def __init__(self, model):
        self.model = model

    def rank(self, query: str, docs):
        pairs = [(query, (d.page_content or "")[:RERANK_MAX_CHARS]) for d in docs]
        scores = self.model.predict(pairs)
        return list(enumerate(scores))          # [(idx, score), ...]


def _get_reranker():
    """惰性加载 CrossEncoder；失败返回 None（调用方降级）。"""
    global _reranker, _reranker_failed
    if _override.get("injected"):
        return _reranker                          # 测试注入的假模型
    if _reranker is not None:
        return _reranker
    if _reranker_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder

        if not pathlib.Path(RERANK_MODEL).exists():
            raise FileNotFoundError("模型目录不存在: %s" % RERANK_MODEL)
        _reranker = _CrossEncoderWrapper(
            CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
        )
        log.info("[rerank] 模型已加载: %s" % RERANK_MODEL)
    except Exception as e:  # noqa: BLE001
        _reranker_failed = True                  # 不重试，避免每次问答都卡
        log.warning("[rerank] 模型加载失败，本次降级为不精排: %s: %s"
                    % (type(e).__name__, str(e)[:120]))
        return None
    return _reranker


def is_aggregation_query(question: str) -> bool:
    """是否是「跨行比较/统计」类问题。

    为什么需要它：这类问题（「哪个产品召回率最低」）需要看到**完整表**，
    而表格被 `VARCHAR max_length=6000` 物理切碎（ROADMAP 4.40 的 n005 案例）。
    命中时调用方应（a）用深检索多召回（b）合并同表碎片再生成。
    """
    if not question:
        return False
    return bool(_AGG_PATTERN.search(question))


def _cfg_top_k(explicit):
    """优先级：显式参数 > 测试注入 > 配置默认。"""
    if explicit is not None:
        return explicit
    if "top_k" in _override:
        return _override["top_k"]
    return RERANK_TOP_K


def _cfg_threshold(explicit):
    """优先级同上。注意 0 是「合法值」（不过滤），所以用 `is not None` 判断。"""
    if explicit is not None:
        return explicit
    if "threshold" in _override:
        return _override["threshold"]
    return RERANK_SCORE_THRESHOLD


def enterprise_rerank(query: str, docs, top_k=None, threshold=None):
    """精排 + 截断 + 分数门槛。**任何失败都降级为返回原文**。

    Args:
        query: 用户问题。
        docs: `list[Document]`。
        top_k: 精排后保留几块（None 用 `RERANK_TOP_K`）。
        threshold: reranker 分数下限（None 用 `RERANK_SCORE_THRESHOLD`；0 = 不过滤）。

    Returns:
        list[Document]：可能被重排/截断/过滤后的列表；失败或未启用时**原样返回**。
    """
    if not docs:
        return []
    if not RERANK_ENABLED:
        return docs

    rr = _get_reranker()
    if rr is None:
        return docs

    try:
        ranked = rr.rank(query, docs)
    except Exception as e:  # noqa: BLE001
        log.warning("[rerank] 推理失败，降级为原文: %s: %s" % (type(e).__name__, str(e)[:120]))
        return docs

    # ranked: [(idx, score)] -> 按分数降序
    ordered = sorted(ranked, key=lambda x: float(x[1]), reverse=True)
    result = [docs[i] for i, _ in ordered]

    # 分数门槛（用 reranker 分数，不用 cosine）
    th = _cfg_threshold(threshold)
    if th and th > 0:
        kept = [(i, s) for i, s in ordered if float(s) >= th]
        n_drop = len(ordered) - len(kept)
        if n_drop:
            log.info("[rerank] 分数门槛 %.3f 剔除 %d 块（剩 %d）"
                     % (th, n_drop, len(kept)))
        result = [docs[i] for i, _ in kept]

    # top_k 截断
    k = _cfg_top_k(top_k)
    if k and k > 0 and len(result) > k:
        log.info("[rerank] top_k=%d 截断 %d -> %d" % (k, len(result), k))
        result = result[:k]

    return result
