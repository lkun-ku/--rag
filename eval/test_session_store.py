"""会话管理 + 滑动 TTL —— 契约测试（ROADMAP 4.45）。

**全部用临时 DB**：不碰真实 `checkpoints.db`，也不连 Milvus
（`history_store` 被关掉，`purge_session` 直接返回 0）。

## 契约

| # | 契约 |
|---|---|
| S1 | `create_session` 返回唯一 sid，且可 `get_session` 读回 |
| S2 | `touch_session` 刷新 `last_active`（**滑动 TTL 的核心**）|
| S3 | `list_sessions` 按 `last_active` **倒序** |
| S4 | `delete_session` 删掉元数据行 |
| S5 | **滑动语义**：TTL=0 时，刚 touch 过的会话**不删**，陈旧的才删 |
| S6 | 默认 TTL == 21600（6 小时）|
| S7 | touch / delete 不存在的 sid 都返回 False，不抛 |
| S8 | title 截断到 200 字 |
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from tools import history_store  # noqa: E402
from utils import session_store as ss  # noqa: E402

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


# ---- 隔离环境：临时 DB + 建空表 + 关掉历史集合 ------------------------------
_tmpdir = tempfile.mkdtemp(prefix="sess_test_")
_db = os.path.join(_tmpdir, "test.db")
_c = sqlite3.connect(_db)
_c.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, checkpoint BLOB, metadata BLOB)")
_c.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_id TEXT, task_id TEXT, channel TEXT, value BLOB)")
_c.commit()
_c.close()

ss._reset_for_test(db_path=_db)
history_store._reset_for_test(enabled=False)  # 不连 Milvus

print("=" * 100)
print("S1/S7/S8 创建与边界")
print("=" * 100)
a = ss.create_session("第一个会话")
b = ss.create_session("第二个会话")
check("S1 返回 sid", bool(a) and len(a) >= 8, a)
check("S1 sid 唯一", a != b, "%s vs %s" % (a, b))
ga = ss.get_session(a)
check("S1 get_session 读回", ga and ga["id"] == a, str(ga))
check("S1 title 正确", ga and ga["title"] == "第一个会话", str(ga))
check(
    "S8 title 截断 200",
    ss.create_session("标" * 500)
    and ss.get_session([s["id"] for s in ss.list_sessions() if s["title"].startswith("标")][0])["title"].__len__()
    == 200,
)

check("S7 touch 不存在的 sid -> False", ss.touch_session("nope") is False)
check("S7 delete 不存在的 sid -> False", ss.delete_session("nope") is False)
check("S7 get 不存在的 -> None", ss.get_session("nope") is None)

print()
print("=" * 100)
print("S2/S3 touch 与列表排序")
print("=" * 100)
old_last = ga["last_active"]
time.sleep(1.1)
ss.touch_session(a)
ga2 = ss.get_session(a)
check("S2 touch 后 last_active 变大", ga2["last_active"] > old_last, "%s -> %s" % (old_last, ga2["last_active"]))
lst = ss.list_sessions()
check("S3 列表按 last_active 倒序", len(lst) >= 2 and lst[0]["id"] == a, [x["id"][:6] for x in lst[:3]])
check("S3 limit 生效", len(ss.list_sessions(limit=1)) == 1)

print()
print("=" * 100)
print("S4/S5 删除与**滑动** TTL")
print("=" * 100)
# 造一个「陈旧的」会话：直接把 last_active 改到很久以前
conn = sqlite3.connect(_db)
conn.execute("UPDATE sessions SET last_active = ? WHERE id = ?", (int(time.time()) - 100000, b))
conn.commit()
conn.close()

check("S4 delete_session 删除元数据", ss.delete_session(b) is True)
check("S4 删后 get 返回 None", ss.get_session(b) is None)

# 滑动语义：TTL=5s 时，刚 touch 的 a **不该**被删；陈旧的会被删。
# 注意不能用 TTL=0：秒级精度下「创建后过了 2 秒」也会被判过期（S8 的会话就是），
# 那样测不出「滑动」与「普通超时」的区别。
c_stale = ss.create_session("陈旧会话")
conn = sqlite3.connect(_db)
# 把**除 a 之外**的所有会话都设为陈旧，确保结果集只剩 c_stale 等
for s in ss.list_sessions():
    if s["id"] != a:
        conn.execute("UPDATE sessions SET last_active = ? WHERE id = ?", (int(time.time()) - 100000, s["id"]))
conn.commit()
conn.close()

removed = ss.purge_expired_sessions(ttl_seconds=5)
check("S5 陈旧会话被清", c_stale in removed, str(removed))
check("S5 **刚 touch 的会话被保留**（滑动 TTL 不误删）", ss.get_session(a) is not None, "a 被误删了！")
check("S5 只删陈旧的（a 未在删除列表里）", a not in removed, str(removed))

print()
print("=" * 100)
print("S6 默认 TTL")
print("=" * 100)
check("S6 SESSION_TTL_SECONDS == 21600（6 小时）", ss.SESSION_TTL_SECONDS == 21600, ss.SESSION_TTL_SECONDS)

print()
print("=" * 100)
print("J1~J4 后台清理线程（janitor）")
print("=" * 100)
import threading  # noqa: E402

check("J1 轮询间隔默认 1800s（30 分钟，远小于 TTL）", ss.SESSION_JANITOR_INTERVAL == 1800, ss.SESSION_JANITOR_INTERVAL)

# J2：run_janitor_once 删除过期会话
d_stale = ss.create_session("待清理")
conn = sqlite3.connect(_db)
conn.execute("UPDATE sessions SET last_active = ? WHERE id = ?", (int(time.time()) - 100000, d_stale))
conn.commit()
conn.close()
removed2 = ss.run_janitor_once(ttl_seconds=5)
check("J2 run_janitor_once 清掉过期会话", d_stale in removed2, str(removed2))
check("J2 活跃会话仍在", ss.get_session(a) is not None)

# J3：非法入参不抛（后台线程最怕这个）
try:
    r = ss.run_janitor_once(ttl_seconds="abc")  # type: ignore[arg-type]
    check("J3 非法 ttl 不抛，返回 list", isinstance(r, list), repr(r))
except Exception as e:  # noqa: BLE001
    check("J3 非法 ttl 不抛", False, "%s: %s" % (type(e).__name__, e))

# J4：stop_event 能让循环退出（不起真线程跑满间隔）
stop = threading.Event()
stop.set()
ss.janitor_loop(interval=1, stop_event=stop)  # 应立即返回
check("J4 stop_event 可立即停止循环（不挂住）", True)

print()
print("=" * 100)
if FAILED:
    print("失败 %d 项：%s" % (len(FAILED), FAILED))
    sys.exit(1)
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
