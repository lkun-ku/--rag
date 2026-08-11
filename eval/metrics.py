"""评测指标：分层实现（确定性优先，LLM 其次）。

设计依据（ROADMAP 1.5.2）：

    Layer 0 —— 检索层：**纯集合运算，0 LLM 调用、0 抖动、0 成本**
        Context Recall    = |golden ∩ retrieved| / |golden|
        Context Precision = |golden ∩ retrieved| / |retrieved|

        为什么能这么做：Golden Dataset 直接标注了「应该命中哪些块」（`source_blocks`），
        而 RAGAS 的对应指标需要 LLM 去判断「参考答案的每个论断能否被上下文支持」——
        块 ID 已知时，集合运算比让 LLM 反推**更精确**（无歧义），且完全可复现。

    Layer 1 —— 生成层：复用项目**已有的判分器**（它们本来就在工作流里用）
        Faithfulness      ≈ graph2.grade_hallucinations_chain
        Answer Relevancy  ≈ graph2.grade_answer_chain

        这层依赖外部 LLM，因此配了「缓存 + 失效保护」两道措施。

**重要**：本模块顶层**不 import 任何 LLM 相关组件** —— Layer 0 必须能在
LLM 不可用（余额耗尽/断网）时独立运行，否则“抗抖动”就失去了意义。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
CACHE_FILE = HERE / "cache.jsonl"


# ===========================================================================
# Layer 0：确定性指标（无需 LLM）
# ===========================================================================
def chunk_id(text: str) -> str:
    """内容指纹。必须与 eval/export_chunks.py 的算法**完全一致**，
    否则 golden 里标注的块 ID 与检索结果对不上。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def score_retrieval(golden_blocks: Sequence[str],
                    retrieved_blocks: Sequence[str]) -> Dict:
    """检索层指标（纯集合运算）。

    Args:
        golden_blocks: Golden Dataset 标注的「应命中块」ID。
        retrieved_blocks: 检索实际返回的块 ID（按返回顺序）。

    Returns:
        {recall, precision, hit_ids, missed_ids, extra_ids}
    """
    g = set(golden_blocks)
    r = set(retrieved_blocks)
    hit = g & r
    return {
        "recall": len(hit) / len(g) if g else 0.0,
        "precision": len(hit) / len(r) if r else 0.0,
        # hit：是否「至少命中一个」必需块。宽召回架构下这个指标比 precision 更有意义 ——
        # precision 天然低（返回 18 条只为命中 1 条），但它低不代表检索差；
        # 而 hit=False 意味着「完全没找到该找的东西」，是真正需要关注的失败。
        "hit": 1.0 if hit else 0.0,
        "hit_ids": sorted(hit),
        "missed_ids": sorted(g - r),
        "extra_ids": sorted(r - g),
    }


def aggregate_layer0(per_case: List[Dict]) -> Dict[str, float]:
    """汇总 Layer 0：**宏平均**（每题的权重相同）。

    为什么用宏平均而非微平均：不同题标注的块数不同（1~3 个），
    微平均会让「多块题」主导总分。门禁关心的是「平均每题表现如何」。
    """
    if not per_case:
        return {"context_recall": 0.0, "context_precision": 0.0, "hit_rate": 0.0}
    return {
        "context_recall": sum(c["recall"] for c in per_case) / len(per_case),
        "context_precision": sum(c["precision"] for c in per_case) / len(per_case),
        "hit_rate": sum(c.get("hit", 0.0) for c in per_case) / len(per_case),
    }


# ===========================================================================
# 判定缓存（回归对比时命中缓存 -> 0 抖动、0 成本）
# ===========================================================================
class JudgeCache:
    """LLM 判定结果缓存。

    key = sha256(kind + question + generation + contexts) 的十六进制摘要。
    同一 (问题, 答案, 上下文) 组合重复评测时直接复用结果 —— 这消除了
    「同样输入、不同分数」的抖动，也让回归对比的结果可复现。
    """

    def __init__(self, path: pathlib.Path = CACHE_FILE, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._data: Dict[str, Dict] = {}
        self.hits = 0
        self.misses = 0
        if enabled:
            self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                self._data[rec["key"]] = rec
            except Exception:  # noqa: BLE001
                continue

    @staticmethod
    def make_key(kind: str, question: str, generation: str, contexts: str = "") -> str:
        # 对上下文**排序后再算 key**：实测发现跨集合 RRF 融合后块的出现顺序
        # 可能有细微变化（同样的块、不同的先后），若直接拼原始字符串，
        # 会造成「内容没变但 key 失配」的假性未命中（实测命中率只有 32/90）。
        # 判定看的是内容而非顺序，因此排序不影响语义。
        ctx = "\n\n".join(sorted(contexts.split("\n\n"))) if contexts else ""
        raw = f"{kind}\x00{question}\x00{generation}\x00{ctx}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        rec = self._data.get(key)
        if rec is not None:
            self.hits += 1
        return rec

    def put(self, key: str, value: Dict):
        if not self.enabled:
            return
        rec = {"key": key, **value}
        self._data[key] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.misses += 1


# ===========================================================================
# Layer 1：LLM 判定（延迟 import + 缓存 + 失效保护）
# ===========================================================================
class LLMUnavailable(RuntimeError):
    """LLM 不可用（余额/网络/服务故障）——与「判定为否」严格区分。"""


def llm_health_check() -> Tuple[bool, str]:
    """评估前探活：确认 LLM 能正常响应。

    为什么必须做：实测遇到过 DeepSeek 返回 `402 Insufficient Balance`，
    此时若把失败当成「得 0 分」，门禁会把**环境故障误报为质量崩塌**。
    先探活就能明确区分这两种情况。

    Returns:
        (可用, 说明)
    """
    try:
        from llm_models.all_llm import llm

        resp = llm.invoke("回复一个字：好")
        text = getattr(resp, "content", str(resp))
        return True, f"OK ({str(text)[:20]!r})"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def _is_llm_env_error(e: Exception) -> bool:
    """区分「LLM 环境不可用」与「代码 bug」。"""
    txt = f"{type(e).__name__} {e}".lower()
    return any(k in txt for k in (
        "insufficient balance", "402", "401", "429", "403",
        "connection", "timeout", "unreachable", "refused", "dns",
        "api key", "apikey", "credential", "unavailable", "rate limit",
    ))


def judge_faithfulness(question: str, generation: str, contexts: str,
                       cache: Optional[JudgeCache] = None) -> Dict:
    """Faithfulness（抗幻觉）：答案是否**只**基于给定上下文。

    复用项目现有的幻觉判分器（graph2/grade_hallucinations_chain.py），
    它本来就在 CRAG 工作流里跑，因此这里评估的是**真实链路**的判断标准。

    Returns:
        {"value": 1.0/0.0, "raw": "yes"/"no", "cached": bool}
    Raises:
        LLMUnavailable: LLM 不可用时抛出（**不返回 0 分**，避免误报质量回退）
    """
    key = JudgeCache.make_key("faith", question, generation, contexts)
    if cache:
        rec = cache.get(key)
        if rec:
            return {"value": rec["value"], "raw": rec["raw"], "cached": True}

    try:
        from graph2.grade_hallucinations_chain import hallucination_grader_chain

        score = hallucination_grader_chain.invoke(
            {"documents": contexts, "generation": generation}
        )
        raw = str(getattr(score, "binary_score", "")).strip().lower()
        value = 1.0 if raw == "yes" else 0.0
    except Exception as e:  # noqa: BLE001
        if _is_llm_env_error(e):
            raise LLMUnavailable(f"Faithfulness 判定失败: {type(e).__name__}: {str(e)[:160]}") from e
        raise

    if cache:
        cache.put(key, {"kind": "faith", "value": value, "raw": raw,
                        "question": question[:120]})
    return {"value": value, "raw": raw, "cached": False}


def judge_answer_relevancy(question: str, generation: str,
                           cache: Optional[JudgeCache] = None) -> Dict:
    """Answer Relevancy：答案是否**真的回答了**问题。

    复用 graph2/grade_answer_chain.py。

    Raises:
        LLMUnavailable: 同上，绝不用 0 分掩盖环境故障
    """
    key = JudgeCache.make_key("relev", question, generation, "")
    if cache:
        rec = cache.get(key)
        if rec:
            return {"value": rec["value"], "raw": rec["raw"], "cached": True}

    try:
        from graph2.grade_answer_chain import answer_grader_chain

        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        raw = str(getattr(score, "binary_score", "")).strip().lower()
        value = 1.0 if raw == "yes" else 0.0
    except Exception as e:  # noqa: BLE001
        if _is_llm_env_error(e):
            raise LLMUnavailable(f"Answer Relevancy 判定失败: {type(e).__name__}: {str(e)[:160]}") from e
        raise

    if cache:
        cache.put(key, {"kind": "relev", "value": value, "raw": raw,
                        "question": question[:120]})
    return {"value": value, "raw": raw, "cached": False}


def generate_answer(question: str, docs: Sequence,
                    cache: Optional[JudgeCache] = None) -> Tuple[str, bool]:
    """用项目自身的生成链产出答案（复用 graph2/generate_node2.generate）。

    为什么直接调 generate() 而不是跑完整工作流：
      评测需要**控制变量** —— 上下文就是我们刚检索到的那批块。
      跑完整工作流会引入路由决策、web_search 降级等分支，
      让「检索质量」与「生成质量」纠缠在一起，难以归因。

    为什么生成结果也要缓存：
      即使 `LLM_TEMPERATURE=0`，同一输入的生成结果仍可能有细微差异（服务端实现所致），
      这会让下游判定的 key 失配、并让「回归对比」引入与代码无关的波动。
      key 基于**上下文内容**，因此上下文一变就会重新生成，不会掩盖真实变化。

    Returns:
        (答案, 是否命中缓存)
    """
    ctx = "\n\n".join(d.page_content for d in docs)
    key = JudgeCache.make_key("gen", question, "", ctx)
    if cache:
        rec = cache.get(key)
        if rec and rec.get("text"):
            return rec["text"], True

    from graph2.generate_node2 import generate

    result = generate({"question": question, "documents": list(docs)})
    ans = result["generation"]

    if cache:
        cache.put(key, {"kind": "gen", "text": ans, "question": question[:120]})
    return ans, False


# ===========================================================================
# 门禁判定
# ===========================================================================
# 阈值**由实测基线推导**，不是预先拍脑袋定的（ROADMAP 初稿里的 0.40 就不成立，
# 见下方 context_precision 的说明）。
#
#   context_recall     检索层核心指标：必需块有没有被召回。
#                      实测基线 0.8926（k=6/final_k=18），故硬门禁 0.60、告警 0.80。
#   hit_rate           「至少命中一个」的比例。宽召回架构下比 precision 更能反映
#                      「有没有完全找不到」。实测基线 0.911，硬门禁 0.75、告警 0.85。
#   context_precision  在「宽召回 + 严判分」架构下**天然很低**：为命中 1 个必需块
#                      要返回约 15 个块，理论上限就只有 1/15 ≈ 0.067。
#                      它低**不代表检索差**，因此只作监控（阈值设在基线附近，
#                      用于发现「检索严重发散」这一类异常），不作为主要门禁。
#   生成层两项         LLM 判定，阈值放宽（判定本身有抖动）。
THRESHOLDS = {
    "context_recall": {"fail": 0.60, "warn": 0.80},
    "hit_rate": {"fail": 0.75, "warn": 0.85},
    "context_precision": {"fail": 0.03, "warn": 0.05},
    "faithfulness": {"fail": 0.60, "warn": 0.80},
    "answer_relevancy": {"fail": 0.60, "warn": 0.80},
}


def gate(scores: Dict[str, Optional[float]],
         gen_requested: bool = False) -> Tuple[str, List[str]]:
    """门禁判定。

    关键设计（ROADMAP 1.5.3）：**区分「未完成」与「不合格」**
        - 生成层缺失（LLM 不可用）-> INCONCLUSIVE（退出码 3），**不计为 FAIL**
        - Layer 0 未达标           -> FAIL（检索层确定性，可放心判定）

    Args:
        scores: {指标名: 分数或 None}。None 表示该指标未完成。

    Returns:
        (verdict, reasons)，verdict ∈ {"PASS","WARN","FAIL","INCONCLUSIVE"}
    """
    reasons: List[str] = []
    failed = False
    warned = False

    for name in ("context_recall", "hit_rate", "context_precision"):
        v = scores.get(name)
        # hit_rate 是后加的指标，旧记录里可能没有 —— 缺失时跳过而非判失败
        if v is None:
            if name == "hit_rate":
                continue
            reasons.append(f"{name}: 未计算（Layer 0 应始终可用，请检查）")
            failed = True
            continue
        th = THRESHOLDS[name]
        if v < th["fail"]:
            reasons.append(f"{name}: {v:.4f} < 硬门禁 {th['fail']}")
            failed = True
        elif v < th["warn"]:
            reasons.append(f"{name}: {v:.4f} 处于告警区间 [{th['fail']}, {th['warn']})")
            warned = True

    llm_skipped = [n for n in ("faithfulness", "answer_relevancy") if scores.get(n) is None]
    for name in ("faithfulness", "answer_relevancy"):
        v = scores.get(name)
        if v is None:
            continue
        th = THRESHOLDS[name]
        if v < th["fail"]:
            reasons.append(f"{name}: {v:.4f} < 硬门禁 {th['fail']}")
            failed = True
        elif v < th["warn"]:
            reasons.append(f"{name}: {v:.4f} 处于告警区间 [{th['fail']}, {th['warn']})")
            warned = True

    if failed:
        return "FAIL", reasons
    if llm_skipped:
        # [FIX 2026-10-15] 「生成层缺失」有两种原因，必须区分 —— 原先一律写成
        # 「LLM 不可用导致」，实测误导过：我跑 `run_eval.py`（**没传 --with-generation**）
        # 时看到这句，第一反应是「LLM 挂了」，于是去查端点，浪费了排查时间。
        #   * gen_requested=True  -> LLM 探活失败（余额/网络/服务故障）
        #   * gen_requested=False -> 本次根本没启用生成层评测
        why = ("LLM 不可用导致" if gen_requested
               else "本次未启用生成层评测（未传 --with-generation）")
        return "INCONCLUSIVE", reasons + [
            f"生成层指标缺失（{', '.join(llm_skipped)}）—— {why}，"
            f"**不得据此判定质量回退**"
        ]
    if warned:
        return "WARN", reasons
    return "PASS", reasons
