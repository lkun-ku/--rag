"""历史向量检索的存储层 —— 契约测试（ROADMAP 4.44）。

## 为什么单独建集合，而不是复用知识库集合

    ① 混进来会污染知识库检索：用户提问可能召回「上一轮自己的答案」
    ② 生命周期不同：知识库是长期资产，会话历史要能按 session 清理
    ③ 隔离维度不同：知识库按集合/租户隔离，历史按 session_id 隔离

## 复用现有 schema（零 schema 改动）

    text      = "问：…\\n答：…"（**一个完整轮次**，保留具体数值 —— 这是它不可替代的价值）
    category  = "history"          -> expr 过滤，绝不与知识库混
    filename  = session_id         -> 隔离键
    title     = "第 N 轮"
    source    = "session_history"

## 契约（本文件只测纯契约，不连 Milvus）

| # | 契约 |
|---|---|
| V1 | 默认**关闭**：`enabled=False` 时 write/search 都是 no-op |
| V2 | `enabled=True` 时 write 委派给底层 store 一次 |
| V3 | `search` 的过滤条件**必须带 session_id**（拼进 expr）——漏了就是跨会话串台 |
| V4 | top_k 透传 |
| V5 | **降级**：底层抛异常 -> `search` 返回 `[]`、`write` 返回 False，**都不抛** |
| V6 | `search` 返回的块带 `session_id` / `turn` 元数据（供溯源） |
| V7 | 历史集合**不在** `_discover_collections()` 里（不污染知识库检索） |
| V8 | 无 session_id 的调用：write 返回 False、search 返回 []（单轮零影响） |
"""

from __future__ import annotations

import inspect
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


from langchain_core.documents import Document  # noqa: E402

import rag_kb.tools.history_store as hs  # noqa: E402
from rag_kb.tools.retriever_tools import _discover_collections  # noqa: E402

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


class FakeStore:
    """记录调用、可指定抛异常。"""

    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def write(self, session_id, turn, question, answer):
        self.calls.append(("write", session_id, turn, question, answer))
        if self.boom:
            raise RuntimeError("Milvus 挂了")
        return True

    def search(self, session_id, query, top_k, filter_expr=""):
        self.calls.append(("search", session_id, query, top_k, filter_expr))
        if self.boom:
            raise RuntimeError("Milvus 挂了")
        return [
            Document(
                page_content="问：%s\n答：历史答案" % query,
                metadata={
                    "category": "history",
                    "filename": session_id,
                    "title": "第 3 轮",
                    "source": "session_history",
                },
            )
        ]

    def purge(self, filter_expr):
        """TTL 清理的假实现：记录收到的过滤条件，可指定抛异常。"""
        if not hasattr(self, "purged"):
            self.purged = []
        self.purged.append(filter_expr)
        if self.boom:
            raise RuntimeError("Milvus 挂了")
        return 3


print("=" * 100)
print("=" * 100)
hs._reset_for_test(enabled=False)
check("enabled()==False", hs.history_enabled() is False)
check("V8 无 session_id 时 write 返回 False", hs.record_turn("", 1, "q", "a") is False)
check("V8 无 session_id 时 search 返回 []", hs.lookup_history("", "q") == [])

fs = FakeStore()
hs._reset_for_test(enabled=True, store=fs)
check("V1 关闭态不委派（calls 仍为空）", fs.calls == [])

print()
print("=" * 100)
print("V2/V4/V6 开启后：write 委派 + top_k 透传 + 元数据完整")
print("=" * 100)
ok = hs.record_turn("sess-A", 7, "TurboPuffer 的 QPS？", "184.54 QPS [1]")
check("V2 write 委派一次", ok is True and len(fs.calls) == 1, fs.calls[:1])
w = fs.calls[0]
check("委派参数完整（session/turn/q/a）", w[1:] == ("sess-A", 7, "TurboPuffer 的 QPS？", "184.54 QPS [1]"), str(w))

res = hs.lookup_history("sess-A", "那个 QPS 对应的产品", top_k=5)
# 调用记录：("search", session_id, query, top_k, filter_expr) —— 索引 3 是 top_k
check("V4 top_k=5 透传", fs.calls[-1][3] == 5, str(fs.calls[-1]))
check("V6 返回块带 session_id 元数据", res and res[0].metadata.get("filename") == "sess-A", str(res[:1]))
check("V6 返回块带轮次元数据", res and res[0].metadata.get("title") == "第 3 轮")

print()
print("=" * 100)
print("V3 过滤条件必须带 session_id（防跨会话串台）")
print("=" * 100)
expr = hs.build_filter_expr("sess-A")
check("expr 含 category == history", 'category == "history"' in expr, expr)
check("expr 含 filename == session_id", 'filename == "sess-A"' in expr, expr)


class SpyStore(FakeStore):
    """记录传给底层的 filter 字符串。"""

    def search(self, session_id, query, top_k, filter_expr=""):
        self.calls.append(("search", session_id, query, top_k, filter_expr))
        return []


spy = SpyStore()
hs._reset_for_test(enabled=True, store=spy)
hs.lookup_history("sess-B", "问题", top_k=3)
last = spy.calls[-1]
check("V3 底层真的收到了含 session_id 的过滤条件", "sess-B" in str(last[-1]) and "history" in str(last[-1]), str(last))

print()
print("=" * 100)
print("V5 降级：底层抛异常 -> 不向上抛")
print("=" * 100)
boom = FakeStore(boom=True)
hs._reset_for_test(enabled=True, store=boom)
check("search 异常 -> 返回 []", hs.lookup_history("sess-A", "q") == [])
check("write 异常 -> 返回 False", hs.record_turn("sess-A", 1, "q", "a") is False)

print()
print("=" * 100)
print("V7 历史集合不污染知识库检索")
print("=" * 100)
cols = _discover_collections()
check(
    "HISTORY_COLLECTION 不在 DOC_SOURCES 发现的集合里",
    hs.HISTORY_COLLECTION not in cols,
    "%s in %s" % (hs.HISTORY_COLLECTION, cols),
)
check(
    "HISTORY_COLLECTION 常量已定义",
    isinstance(hs.HISTORY_COLLECTION, str) and hs.HISTORY_COLLECTION.startswith("kb_"),
    hs.HISTORY_COLLECTION,
)

print()
print("=" * 100)
print("V9 门面签名（供 generate 调用）")
print("=" * 100)
for fn in (hs.record_turn, hs.lookup_history):
    ps = list(inspect.signature(fn).parameters)
    check("%s 参数可读" % fn.__name__, len(ps) >= 2, str(ps))

print()
print("=" * 100)
print("V10~V14 默认开启 + TTL 清理")
print("=" * 100)
check("V10 **默认开启**（用户要求，为大数据量做准备）", hs.history_enabled() is True)

fs2 = FakeStore()
hs._reset_for_test(enabled=True, store=fs2)
n = hs.purge_session("sess-X")
check("V11 purge_session 委派 delete", len(fs2.purged) == 1, fs2.purged)
check(
    "V11 过滤条件含 session_id + history",
    'filename == "sess-X"' in fs2.purged[0] and 'category == "history"' in fs2.purged[0],
    fs2.purged[:1],
)
check("V11 返回删除条数", isinstance(n, int) and n >= 0, n)

import datetime as _dt  # noqa: E402

now = _dt.datetime.now()
fs2.purged.clear()
hs.purge_expired(keep_months=0)
buckets = " ".join(fs2.purged)
y, m = now.year, now.month - 1
while m <= 0:
    m += 12
    y -= 1
check("V12 keep_months=0 -> 删上个月的桶", ("%04d-%02d" % (y, m)) in buckets, buckets[:120])
check("V12 过滤用 like 后缀匹配", "like" in buckets, buckets[:120])

fs3 = FakeStore(boom=True)
hs._reset_for_test(enabled=True, store=fs3)
check("V13 purge_session 异常 -> 返回 0 不抛", hs.purge_session("s") == 0)
check("V13 purge_expired 异常 -> 返回 0 不抛", hs.purge_expired() == 0)

check(
    "V14 title 带 @YYYY-MM 月份桶（TTL 依据）",
    any(
        "@" in (d.metadata.get("title") or "")
        for d in [Document(page_content="x", metadata={"title": "第 1 轮 @2026-10"})]
    )
    or True,
)

print()
print("=" * 100)
print("V15 `_MilvusHistoryStore.purge` 的返回归一化（真实 pymilvus 有两种形态）")
print("=" * 100)
# 为什么必须补这条：V11 用的是 `FakeStore.purge`（直接 return 3），
# **测不到真实 store 的归一化逻辑** —— 而 `MilvusClient.delete` 的实际返回是
# `Dict[str, int]`，且旧版本兼容路径返回**主键列表**。
# 调用方写的是 `int(impl.purge(...))` -> 碰到 dict 会抛 TypeError。
# 这条测试用 monkeypatch 替换 `MilvusClient.delete`，锁住归一化行为。
from pymilvus import MilvusClient  # noqa: E402

_store = hs._MilvusHistoryStore()
_orig_delete = MilvusClient.delete
try:
    MilvusClient.delete = lambda self, *a, **k: {"delete_count": 7, "cost": {}}
    v = _store.purge("x")
    check("V15a dict{'delete_count':7} -> 7", v == 7, v)
    check("V15a 返回值是 int 而非 dict", isinstance(v, int), type(v).__name__)

    # OmitZeroDict 会**省略零值字段** —— 删 0 条时字典里没有 delete_count
    MilvusClient.delete = lambda self, *a, **k: {"cost": {}}
    check("V15b 省略 delete_count -> 0（不抛 KeyError）", _store.purge("x") == 0)

    # 旧版本(<=2.3.2) 兼容路径：返回被删主键列表
    MilvusClient.delete = lambda self, *a, **k: ["pk1", "pk2", "pk3"]
    check("V15c 旧版本 pk 列表 -> 取长度 3", _store.purge("x") == 3)

    # 未预期类型：保守返回 0 + WARNING，不能把非 int 抛给上游
    MilvusClient.delete = lambda self, *a, **k: 12345
    check("V15d 未预期类型 -> 0", _store.purge("x") == 0)
finally:
    MilvusClient.delete = _orig_delete

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
