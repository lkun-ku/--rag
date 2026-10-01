"""会话历史的向量存储与检索（ROADMAP 4.44）。

## 它解决什么不可替代的问题

「最近 13 轮原文」覆盖**近期指代**，「摘要」覆盖**主题脉络**，
但**都找不回「含具体数值的那一轮」**：

    第 3 轮：用户问「TurboPuffer 的 QPS 是多少」-> 答 184.54
    第 30 轮：用户问「刚才那个 QPS 对应的产品叫什么」
      · 最近 13 轮：第 3 轮已过期
      · 摘要：只留「讨论过 QPS」，不留 184.54
      · **本模块：按语义召回第 3 轮原文**

大数据量/长会话下这是最有价值的一层，也是「多轮记忆」三件套
（原文 + 摘要 + 向量检索）里最后一层。

## 为什么独立集合，不复用知识库集合

    ① 混进来会污染知识库检索 —— 用户提问可能召回「自己上一轮的答案」
    ② 生命周期不同 —— 知识库是长期资产，会话历史要能按 session 清理
    ③ 隔离维度不同 —— 知识库按集合/租户，历史按 session_id

## 零 schema 改动（复用现有 MilvusVectorSave 的字段）

    text      = "问：…\\n答：…"   **一个完整轮次**，保留具体数值
    category  = "history"          -> expr 过滤，绝不与知识库混
    filename  = session_id         -> 隔离键
    title     = "第 N 轮"          -> 轮次标识（也可溯源）
    source    = "session_history"

该集合**不进** `DOC_SOURCES`，所以 `_discover_collections()` 不会发现它 ——
知识库检索永远看不到历史块。

## 三条设计约束

1. **默认关闭**（`HISTORY_ENABLED=0`）：单轮与未启用时完全不读写，
   对既有指标零影响。大数据量部署时开 `HISTORY_ENABLED=1`。
2. **失败降级**：Milvus 不可用 / 写入失败 -> `record_turn` 返回 False、
   `lookup_history` 返回 `[]`，**都不抛异常**（不因记忆层断掉问答主链）。
3. **无 session_id 不动作**：单轮调用（无 `thread_id`）直接 no-op。
"""

from __future__ import annotations

import datetime
import os

from langchain_core.documents import Document
from utils.log_utils import log

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
HISTORY_COLLECTION = os.getenv("HISTORY_COLLECTION", "kb_session_history_bge_large_1024")
# [ROADMAP 4.44] **默认开启**（用户要求，为大数据量/长会话准备基础设施）。
#
# 影响面是**可控的**：只有「显式传 session_id 的请求」才会读写历史
# （`run_with_trace(thread_id=...)` 才走记忆路径）。
# 评测 / 回归 / 单轮 CLI 都不传 session_id -> 完全不触发，对既有指标零影响。
HISTORY_ENABLED = os.getenv("HISTORY_ENABLED", "1") not in ("", "0", "false", "False")
HISTORY_TOP_K = int(os.getenv("HISTORY_TOP_K", "3"))
# 单轮 text 的字符上限。表 schema 的 VARCHAR 上限是 6000，留余量。
# 轮次要**保留完整数值**，所以这里比历史的单条上限（500）宽松得多。
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "1500"))
# 历史保留月数（TTL）：早于 N 个月前的**整月桶**会被 `purge_expired` 清掉。
# 按月而非按天：Milvus 的 VARCHAR 无法按「字符串里的日期」做范围比较，
# 按月桶 + `like` 后缀匹配是零 schema 改动下最务实的 TTL 实现
# （要按天/按小时精确 TTL，需给 schema 加 INT64 时间字段 —— 见 ROADMAP 4.44.6）。
HISTORY_KEEP_MONTHS = int(os.getenv("HISTORY_KEEP_MONTHS", "3"))

_impl = None  # 底层 store（真实或测试注入）
_impl_failed = False  # 真实 store 初始化失败后不再重试
_override: dict = {}


def _reset_for_test(enabled=None, store=None):
    """**仅供测试**：重置开关并注入假 store。"""
    global HISTORY_ENABLED, _impl, _impl_failed
    if enabled is not None:
        HISTORY_ENABLED = enabled
    _impl = store
    _impl_failed = False
    _override["injected"] = store is not None


def history_enabled() -> bool:
    return bool(HISTORY_ENABLED)


# ---------------------------------------------------------------------------
# 底层 store
# ---------------------------------------------------------------------------
class _MilvusHistoryStore:
    """基于 `MilvusVectorSave` 的真实读写。

    与知识库共用同一套 Milvus 连接与 embedding 模型（项目里已加载），
    不引入新依赖。
    """

    def _mv(self):
        from documents.milvus_db import MilvusVectorSave

        return MilvusVectorSave(collection_name=HISTORY_COLLECTION)

    def _ensure(self, mv):
        """**只在集合不存在时创建**；已存在则原样复用。

        [FIX / ROADMAP 4.44] 首版无条件调 `create_collection()`，
        而它会 **drop 后重建** —— 于是每写一轮就把整个历史库清空一次。
        实测痕迹：4 轮「写入成功」后 `num_entities` 只有 1（只剩最后那轮）。

        因此这里必须先 `list_collections()` 判断，已存在就**什么都不做**。
        """
        mv.create_connection()
        from pymilvus import MilvusClient
        from utils.env_utils import MILVUS_URI

        existing = MilvusClient(uri=MILVUS_URI).list_collections()
        if HISTORY_COLLECTION not in existing:
            log.info("[history] 集合 %s 不存在，创建之" % HISTORY_COLLECTION)
            mv.create_collection()

    def write(self, session_id: str, turn: int, question: str, answer: str) -> bool:
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return False
        text = "问：%s\n答：%s" % (q, a)
        # `title` 带**月份桶**（"第 N 轮 @YYYY-MM"）：这是 TTL 的清理依据。
        # 不加 schema 字段就能按月归档删除（见 HISTORY_KEEP_MONTHS 的取舍说明）。
        _bucket = datetime.datetime.now().strftime("%Y-%m")
        doc = Document(
            page_content=text[:HISTORY_MAX_CHARS],
            metadata={
                "category": "history",
                "source": "session_history",
                "filename": str(session_id),
                "title": "第 %d 轮 @%s" % (int(turn), _bucket),
                "filetype": "session",
                # schema 里的 INT64 字段，**不提供会被 Milvus 拒收**
                # （实测报 DataNotMatchException: missed an field `category_depth`）。
                # 历史块没有标题层级，恒为 0。
                "category_depth": 0,
            },
        )
        mv = self._mv()
        self._ensure(mv)
        # [FIX / ROADMAP 4.44] **必须调 `mv.add_documents`，不能调
        # `mv.vector_store_saved.add_documents`**：
        #   后者是 langchain-milvus 的原始方法，只把数据写进 **WAL**，
        #   不 flush -> `get_collection_stats` 返回 0、检索查不到
        #   （实测：4 轮「写入成功」里只有 1 轮偶然可见，另 3 轮彻底丢失）。
        #   前者是项目自己的包装，内部会 `client.flush(self.collection_name)`。
        mv.add_documents([doc])
        return True

    def search(self, session_id: str, query: str, top_k: int, filter_expr: str = "") -> list:
        mv = self._mv()
        mv.create_connection()
        kwargs = {"k": int(top_k)}
        if filter_expr:
            kwargs["expr"] = filter_expr
        retriever = mv.vector_store_saved.as_retriever(search_type="similarity", search_kwargs=kwargs)
        return list(retriever.invoke(query or ""))

    def purge(self, filter_expr: str) -> int:
        """按过滤条件删除历史块，返回删除条数。"""
        from pymilvus import MilvusClient
        from utils.env_utils import MILVUS_URI

        cli = MilvusClient(uri=MILVUS_URI)
        return cli.delete(HISTORY_COLLECTION, filter=filter_expr)


def _get_impl():
    global _impl, _impl_failed
    if _override.get("injected"):
        return _impl
    if _impl is not None:
        return _impl
    if _impl_failed:
        return None
    try:
        _impl = _MilvusHistoryStore()
    except Exception as e:  # noqa: BLE001
        _impl_failed = True
        log.warning("[history] 存储层初始化失败，记忆检索将降级为无: %s: %s" % (type(e).__name__, str(e)[:120]))
        return None
    return _impl


# ---------------------------------------------------------------------------
# 门面（供 generate 调用）
# ---------------------------------------------------------------------------
def build_filter_expr(session_id: str) -> str:
    """构造「只取该会话历史」的 Milvus 过滤表达式。

    **`filename == session_id` 是隔离的关键** —— 漏了它就会跨会话串台
    （把别人的历史召回进当前 prompt）。测试 V3 锁定这一点。
    """
    sid = str(session_id).replace('"', "'")
    return 'category == "history" and filename == "%s"' % sid


def record_turn(session_id: str, turn: int, question: str, answer: str) -> bool:
    """把一轮问答写入历史集合。失败返回 False，**不抛异常**。

    无 session_id / 未启用 -> 直接 no-op 返回 False。
    """
    if not HISTORY_ENABLED or not session_id:
        return False
    impl = _get_impl()
    if impl is None:
        return False
    try:
        return bool(impl.write(session_id, turn, question, answer))
    except Exception as e:  # noqa: BLE001
        log.warning("[history] 写入失败（不影响本次问答）: %s: %s" % (type(e).__name__, str(e)[:120]))
        return False


def lookup_history(session_id: str, query: str, top_k: int | None = None) -> list:
    """按语义检索该会话的历史轮次。失败返回 `[]`，**不抛异常**。

    无 session_id / 未启用 -> 直接返回 `[]`（单轮零影响）。
    """
    if not HISTORY_ENABLED or not session_id:
        return []
    impl = _get_impl()
    if impl is None:
        return []
    k = HISTORY_TOP_K if top_k is None else top_k
    try:
        return list(impl.search(session_id, query or "", k, filter_expr=build_filter_expr(session_id)))
    except Exception as e:  # noqa: BLE001
        log.warning("[history] 检索失败，降级为无历史: %s: %s" % (type(e).__name__, str(e)[:120]))
        return []


# ---------------------------------------------------------------------------
# 清理（TTL）
# ---------------------------------------------------------------------------
def purge_session(session_id: str) -> int:
    """删除某个会话的**全部**历史。返回删除条数；失败返回 0 且**不抛**。

    用途：会话结束 / 用户要求删除记忆 / GDPR 式「被遗忘权」。
    无 session_id / 未启用 -> 直接返回 0。
    """
    if not HISTORY_ENABLED or not session_id:
        return 0
    impl = _get_impl()
    if impl is None:
        return 0
    try:
        return int(impl.purge(build_filter_expr(session_id)))
    except Exception as e:  # noqa: BLE001
        log.warning("[history] 清理会话失败: %s: %s" % (type(e).__name__, str(e)[:120]))
        return 0


def purge_expired(keep_months: int | None = None) -> int:
    """删除 `keep_months` 个月之前的**整月桶**历史。返回删除条数；失败返回 0。

    按月桶的原因：Milvus 的 VARCHAR 无法对「字符串里的日期」做范围比较，
    零 schema 改动下只能靠 `title like "%@YYYY-MM"` 做后缀匹配（见
    `HISTORY_KEEP_MONTHS` 的取舍说明）。要按天精确 TTL，需给 schema 加
    INT64 时间字段 —— 那会改动**所有**集合的写入路径，风险不成比例。
    """
    if not HISTORY_ENABLED:
        return 0
    keep = HISTORY_KEEP_MONTHS if keep_months is None else keep_months
    impl = _get_impl()
    if impl is None:
        return 0

    today = datetime.datetime.now()
    # 需要删掉的月份桶：从当月往前推 keep+1 个月开始，一直到最早
    total = 0
    for back in range(keep + 1, keep + 25):  # 最多回看 2 年
        y, m = today.year, today.month - back
        while m <= 0:
            m += 12
            y -= 1
        bucket = "%04d-%02d" % (y, m)
        try:
            total += int(impl.purge('category == "history" and title like "%%@%s"' % bucket))
        except Exception as e:  # noqa: BLE001
            log.warning("[history] 清理桶 %s 失败: %s: %s" % (bucket, type(e).__name__, str(e)[:100]))
            break
    return total
