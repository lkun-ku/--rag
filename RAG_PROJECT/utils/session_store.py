"""会话元数据管理 + **滑动 TTL**（ROADMAP 4.45）。

## 为什么需要它

「新开对话 / 会话列表 / 自动清理」都要求服务端知道**有哪些会话、何时活跃**，
而 LangGraph 的 checkpointer 只按 `thread_id` 存状态，不提供「会话目录」。
本模块补上这一层，**复用同一个 `checkpoints.db`**（零新依赖、零新文件）。

## 为什么 TTL 用 `last_active` 而不是「创建时间」

「创建后 6 小时删」会**误删活跃会话**：

    用户 9:00 开始问，持续到 15:30（同一会话跨 6.5h）
    -> 9:00 写入的历史在 15:00 全部过期
    -> 用户 15:30 问「刚才那个 QPS」-> 历史已被清   ❌

滑动 TTL（每次问答更新 `last_active`）才是「6 小时**不活跃**才清」：

    同一会话持续工作 -> last_active 一直刷新 -> 永不过期 ✅
    搁置超过 6 小时   -> 下次清理任务命中          ✅

## 为什么不动 Milvus schema

按小时删历史块需要 INT64 时间字段，而 `create_collection()` 是**所有集合共用**的，
加字段会让 `MilvusVectorSave.add_documents` 必须补默认值，否则知识库入库全线
`DataNotMatchException`。**本方案不需要**：会话级清理走已实现的
`history_store.purge_session(sid)`，TTL 判定走 SQLite（毫秒级）。

## 删除一个会话时会清掉的三处

    ① history_store.purge_session(sid)   历史向量集合里该会话的块
    ② checkpoints / writes 表里 thread_id = sid 的行（LangGraph 状态）
    ③ sessions 表里的元数据行
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid

from utils.env_utils import CHECKPOINT_DB
from utils.log_utils import log

# 默认 6 小时（用户确认）。语义是「6 小时不活跃」。
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "21600"))
# 后台清理线程的轮询间隔（秒）。默认 30 分钟 —— 远小于 TTL，
# 所以「过期」与「被清理」之间的延迟对用户不可感知。
SESSION_JANITOR_INTERVAL = int(os.getenv("SESSION_JANITOR_INTERVAL", "1800"))

_conn = None  # 进程级单例连接
_override_db = None  # 仅供测试注入（用临时 DB，避免污染真实数据）


def _reset_for_test(db_path=None):
    """**仅供测试**：切到临时 DB（或恢复默认）。"""
    global _conn, _override_db
    _conn = None
    _override_db = db_path


def _db_path() -> str:
    return _override_db or CHECKPOINT_DB


def _get_conn():
    global _conn
    if _conn is None:
        c = sqlite3.connect(_db_path(), check_same_thread=False)
        c.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "  id           TEXT PRIMARY KEY,"
            "  created_at   INTEGER NOT NULL,"
            "  last_active  INTEGER NOT NULL,"
            "  title        TEXT NOT NULL DEFAULT ''"
            ")"
        )
        c.commit()
        _conn = c
    return _conn


def _row_to_dict(r) -> dict:
    return {"id": r[0], "created_at": r[1], "last_active": r[2], "title": r[3]}


def create_session(title: str = "") -> str:
    """新建会话，返回服务端生成的 `session_id`。

    `id` 用 UUID（去横线）—— 足够唯一且短；不暴露任何内部信息。
    """
    sid = uuid.uuid4().hex
    now = int(time.time())
    c = _get_conn()
    c.execute(
        "INSERT INTO sessions (id, created_at, last_active, title) VALUES (?, ?, ?, ?)",
        (sid, now, now, (title or "")[:200]),
    )
    c.commit()
    log.info("[session] 新建会话 %s" % sid[:8])
    return sid


def touch_session(sid: str) -> bool:
    """刷新 `last_active`（滑动 TTL 的核心）。不存在返回 False。"""
    if not sid:
        return False
    c = _get_conn()
    cur = c.execute("UPDATE sessions SET last_active = ? WHERE id = ?", (int(time.time()), sid))
    c.commit()
    return cur.rowcount > 0


def ensure_session(sid: str, title: str = "") -> str:
    """确保会话存在：不存在则建档，存在则刷新 `last_active`。

    为什么需要它：`session_id` 名义上由服务端 `POST /sessions/new` 发放，
    但**不能假设客户端一定守约**（自生成 UUID、旧版本客户端、直接调 API 等）。
    用 upsert 语义既宽容又不丢会话目录；若严格「没建档就拒绝」，
    这些请求的记忆会**静默不持久化**，更难排查。
    """
    if not sid:
        return ""
    if not touch_session(sid):
        now = int(time.time())
        c = _get_conn()
        c.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at, last_active, title) VALUES (?, ?, ?, ?)",
            (sid, now, now, (title or "")[:200]),
        )
        c.commit()
    return sid


def get_session(sid: str):
    cur = _get_conn().execute("SELECT id, created_at, last_active, title FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    return _row_to_dict(r) if r else None


def list_sessions(limit: int = 50) -> list:
    """按最后活跃倒序返回会话列表（最近聊的在前）。"""
    cur = _get_conn().execute(
        "SELECT id, created_at, last_active, title FROM sessions ORDER BY last_active DESC LIMIT ?", (int(limit),)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def delete_session(sid: str) -> bool:
    """彻底删除一个会话：历史块 + LangGraph 检查点 + 元数据。

    任一步失败都**不抛**（返回 False），但会记 warning ——
    删不干净比删不掉更危险（用户以为没了，数据还在）。
    """
    if not sid:
        return False
    c = _get_conn()
    try:
        # ① 历史向量集合
        try:
            from tools import history_store

            history_store.purge_session(sid)
        except Exception as e:  # noqa: BLE001
            log.warning("[session] 历史集合清理失败（继续删检查点）: %s" % str(e)[:100])
        # ② LangGraph 检查点（SqliteSaver 无公开的按 thread 删除 API，直接 SQL）
        c.execute("DELETE FROM checkpoints WHERE thread_id = ?", (sid,))
        try:
            c.execute("DELETE FROM writes WHERE thread_id = ?", (sid,))
        except Exception:  # noqa: BLE001
            pass  # 老版本可能没有 writes 表
        # ③ 元数据
        cur = c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        c.commit()
        log.info("[session] 已删除会话 %s" % sid[:8])
        return cur.rowcount > 0
    except Exception as e:  # noqa: BLE001
        log.warning("[session] 删除会话失败: %s: %s" % (type(e).__name__, str(e)[:120]))
        return False


def purge_expired_sessions(ttl_seconds: int | None = None) -> list:
    """删除「超过 TTL 不活跃」的会话，返回被删的 session_id 列表。

    滑动语义：判据是 `now - last_active > ttl`，与创建时间无关。
    """
    ttl = SESSION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    now = int(time.time())
    cur = _get_conn().execute("SELECT id, last_active FROM sessions WHERE ? - last_active > ?", (now, int(ttl)))
    stale = cur.fetchall()
    removed = []
    for sid, last in stale:
        if delete_session(sid):
            removed.append(sid)
    if removed:
        log.info("[session] TTL 清理：删除 %d 个过期会话（TTL=%ds）" % (len(removed), ttl))
    return removed


def run_janitor_once(ttl_seconds: int | None = None) -> list:
    """跑一次过期会话清理，返回被删的 session_id 列表。**绝不让异常逃出**。

    为什么单独抽这个函数（而不是把逻辑塞进后台线程）：
    后台线程的循环体极难单测（要起线程、要 sleep、要信号同步），
    而「清理」本身是纯逻辑。抽出来后测试可以直接调用，
    线程只负责「每隔 N 秒调一次」这一件 dumb 的事。

    多进程部署时会重复跑：**幂等无害** —— `purge_expired_sessions`
    只删已过期的，第二次跑自然为空。
    """
    try:
        return purge_expired_sessions(ttl_seconds)
    except Exception as e:  # noqa: BLE001
        log.warning("[session] 清理任务异常（不影响服务）: %s: %s" % (type(e).__name__, str(e)[:120]))
        return []


def janitor_loop(interval: int | None = None, stop_event=None):
    """后台清理循环（由 API 的 startup 钩子以 daemon 线程启动）。"""
    gap = SESSION_JANITOR_INTERVAL if interval is None else interval
    log.info("[session] 清理线程启动：每 %ds 轮询一次，TTL=%ds" % (gap, SESSION_TTL_SECONDS))
    while True:
        if stop_event is not None and stop_event.is_set():
            log.info("[session] 清理线程收到停止信号，退出")
            return
        try:
            removed = run_janitor_once()
            if removed:
                log.info("[session] 本轮清理 %d 个过期会话" % len(removed))
        except Exception as e:  # noqa: BLE001
            # 双保险：即使 run_janitor_once 漏了什么，也不能让线程死掉
            log.warning("[session] 清理线程异常（继续运行）: %s" % str(e)[:120])
        if stop_event is not None:
            stop_event.wait(gap)
        else:
            time.sleep(gap)
    """删除「超过 TTL 不活跃」的会话，返回被删的 session_id 列表。

    滑动语义：判据是 `now - last_active > ttl`，与创建时间无关。
    """
    ttl = SESSION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    now = int(time.time())
    cur = _get_conn().execute("SELECT id, last_active FROM sessions WHERE ? - last_active > ?", (now, int(ttl)))
    stale = cur.fetchall()
    removed = []
    for sid, last in stale:
        if delete_session(sid):
            removed.append(sid)
    if removed:
        log.info("[session] TTL 清理：删除 %d 个过期会话（TTL=%ds）" % (len(removed), ttl))
    return removed
