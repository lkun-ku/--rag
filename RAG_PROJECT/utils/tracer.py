"""检索与生成的可观测性（ROADMAP 3.1）。

## 为什么需要

阶段一升到 1024 维后编码耗时 ×14.6，阶段二又引入了引用校验、熔断重试、
web_search 降级等多个分支 —— **一次问答实际走了哪条路径、每步花了多久、
检索到几块、判分结果如何，全都只散落在 INFO 日志里**，无法按次回看。

对应业界经验「70% 的 RAG 应用因检索不可观测而失效」：不是功能坏掉，
而是**出问题时无法定位是哪一环**。

## 设计要点

**为什么用 ContextVar 而不是显式传参**：
    LangGraph 的节点函数签名是固定的 `fn(state)`，无法追加 tracer 参数。
    若改成闭包或全局变量，会污染节点实现、且并发时不安全。
    `ContextVar` 是 asyncio 上下文安全的，天然适配「一次问答一个 trace」。

**为什么用 LangChain callback 统计 token**：
    `UsageMetadataCallbackHandler` 会被 `graph.stream(config={"callbacks": ...})`
    自动下传到所有 LLM 调用（含判分链、路由链），无需逐个包装。
    实测 langchain-core 1.6.7 可用。

**为什么 trace 落盘为 JSONL**：
    可直接 `grep`/`jq`，也能被 `scripts/show_trace.py` 读取；
    追加写不清空历史，便于对比「同一问题在不同版本下的表现」。
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from utils.env_utils import PKG_ROOT

# trace 落盘位置：与 eval/ 同级放在项目根下的 logs/ 里
TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR") or (pathlib.Path(PKG_ROOT).parent / "logs"))
TRACE_FILE = TRACE_DIR / "trace.jsonl"


class Tracer:
    """一次问答的完整执行轨迹。

    Attributes:
        question: 本次的用户问题。
        steps: 按执行顺序记录的节点步骤。
        usage: LangChain 的 token 统计回调（会被下传给所有 LLM 调用）。
    """

    def __init__(self, question: str, enabled: bool = True):
        self.question = question
        self.enabled = enabled
        self.t0 = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.result: Optional[str] = None
        self._usage = None

    # ---- token 统计 ----
    @property
    def usage(self):
        """惰性构造：langchain_core 未提供该回调时退化为 None（不阻断主流程）。

        [FIX / 代码审查] **失败必须留痕**。原实现只写了
        `self._usage = False` 就默默退化为「没有统计」—— 那正是本轮 4.22
        挖了半天的「成本不可见」的**第二条隐形路径**：langchain 一升级、
        这个 import 一变，token 就**静默归零**，而监控里看不出任何异常。
        现在补一条 warning（项目纪律：不得静默降级）。
        """
        if self._usage is None:
            try:
                from langchain_core.callbacks import UsageMetadataCallbackHandler

                self._usage = UsageMetadataCallbackHandler()
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "tracer: UsageMetadataCallbackHandler 不可用（%s: %s）—— "
                    "**本次 trace 不会有 token 统计**，成本监控会缺失，"
                    "请检查 langchain_core 版本。",
                    type(e).__name__,
                    str(e)[:80],
                )
                self._usage = False  # 标记为不可用
        return self._usage or None

    @property
    def callbacks(self) -> List[Any]:
        """传给 `graph.stream(config={"callbacks": ...})` 的回调列表。"""
        u = self.usage
        return [u] if u is not None else []

    def tokens(self) -> Dict[str, Any]:
        """汇总各模型的 token 用量。"""
        u = self.usage
        if u is None:
            return {"available": False}
        try:
            meta = dict(getattr(u, "usage_metadata", {}) or {})
        except Exception:  # noqa: BLE001
            return {"available": False}
        total_in = sum(int(v.get("input_tokens", 0) or 0) for v in meta.values())
        total_out = sum(int(v.get("output_tokens", 0) or 0) for v in meta.values())
        return {
            "available": True,
            "by_model": {k: dict(v) for k, v in meta.items()},
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
        }

    # ---- 记录 ----
    def step(self, node: str, elapsed: float = 0.0, **info: Any):
        """记录一个节点的执行。"""
        if not self.enabled:
            return
        rec = {"node": node, "t": round(time.time() - self.t0, 3), "elapsed_sec": round(elapsed, 3)}
        rec.update({k: v for k, v in info.items() if v is not None})
        self.steps.append(rec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "total_sec": round(time.time() - self.t0, 3),
            "n_steps": len(self.steps),
            "steps": self.steps,
            "tokens": self.tokens(),
            "result_preview": (self.result or "")[:200],
        }

    def save(self) -> Optional[pathlib.Path]:
        """追加写入 trace.jsonl。失败不抛（观测不应影响主流程）。"""
        if not self.enabled:
            return None
        try:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
            return TRACE_FILE
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# 当前 trace（ContextVar：并发安全，且不改动任何节点函数签名）
# ---------------------------------------------------------------------------
_CURRENT: ContextVar[Optional[Tracer]] = ContextVar("rag_tracer", default=None)


def start(question: str, enabled: bool = True) -> Tracer:
    """开始一次追踪，并设为「当前 trace」。"""
    t = Tracer(question, enabled=enabled)
    _CURRENT.set(t)
    return t


def current() -> Optional[Tracer]:
    """取当前 trace（无则返回 None，调用方须容忍）。"""
    return _CURRENT.get()


# [删除  / P1-9] 这里原先有个 `mark(node, **info)` 辅助函数，
# **全项目没有任何调用方** —— 它是个「通用打点 API」，但实际打点由
# `graph2._traced_router` 包装条件边后直接调 `t.step()` 完成。
# 留着它只会造成「两套打点方式」，而文档还指着它（见下），属于误导。


def traced(name: str):
    """节点包装器：自动记录耗时 + 状态摘要。

    只记录「可由 state 推断」的信息（如 documents 数量、generation 长度）；
    更细的语义（**路由决策、判分结论**）由 `graph2._traced_router` 包装条件边后
    直接调 `tracer.current().step()` 补充 —— 这样无需改动任何节点实现文件。

    ## [FIX / ROADMAP 4.20.7] 必须把 `config` 透传给被包装的节点

    原先 `wrapper(state)` 只接 state，**把 LangGraph 传下来的 config 丢掉了**。
    后果是一条静默的链路断裂：

        LangGraph 传 config -> wrapper 丢弃 -> 节点收到 config=None
        -> 节点内手动 `.invoke(...)` 拿不到 callbacks -> **token 不被统计**

    实测代价：`grade_documents` 是**最大**的 LLM 调用方（`FINAL_K=36` 时 36 次/题），
    修好这一处之前，它的消耗在生产监控里**完全不可见**
    （修复前后 `trace.jsonl` 的 total 只差 637，而 36 次判分本应带来 ~17K）。

    为什么不在 `add_node` 处改、而在包装器改：包装器是所有节点进出的**唯一必经点**，
    改一处即可覆盖全部节点，且不需要每个节点实现文件都改签名。
    """

    def deco(fn):
        # 签名判定：本项目多数节点仍是 `fn(state)`（retrieve / web_search），
        # **统一多传一个参数会直接 TypeError**。所以按签名决定怎么调。
        #
        # [FIX / 代码审查] 首版只看「**位置参数**个数」——
        # 于是 `def fn(state, *, config=None)`（仅关键字 config）会被判成
        # 「不接受 config」，**静默丢弃、token 归零、且无任何告警**。
        # 那正是本轮在修的「静默断链」，判据却是启发式 —— 现在改为
        # **同时识别仅关键字形态**，三种签名都覆盖：
        #     fn(state)                     -> 直接调
        #     fn(state, config)             -> 位置传
        #     fn(state, *, config=None)     -> 关键字传
        try:
            _params = list(inspect.signature(fn).parameters.values())
            _n_pos = len([p for p in _params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
            _kwonly_cfg = any(p.name == "config" and p.kind == p.KEYWORD_ONLY for p in _params)
        except (TypeError, ValueError):  # 内建/无签名可读时按「只接 state」处理
            _n_pos, _kwonly_cfg = 1, False

        def wrapper(state, config=None):
            t = _CURRENT.get()
            t0 = time.time()
            if _n_pos >= 2:
                out = fn(state, config)
            elif _kwonly_cfg:
                out = fn(state, config=config)
            else:
                out = fn(state)
            if t is not None:
                info: Dict[str, Any] = {}
                try:
                    docs = (out or state).get("documents")
                    if isinstance(docs, list):
                        info["documents"] = len(docs)
                    gen = (out or state).get("generation")
                    if gen:
                        info["generation_chars"] = len(str(gen))
                except Exception:  # noqa: BLE001
                    pass
                t.step(name, elapsed=time.time() - t0, **info)
            return out

        wrapper.__name__ = getattr(fn, "__name__", name)
        return wrapper

    return deco
