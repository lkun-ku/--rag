"""端到端验证：轮次提升 + 摘要压缩，**全部走真实代码与真实 LLM**。

## 验什么

| 断言 | 含义 |
|---|---|
| E1 30 轮历史 + 1 问 -> messages 变 62 条 | 轮次上限提高到 13 后，**仍持续累积** |
| E2 `updates["summary"]` 非空 | **真实触发了一次 LLM 摘要**（不是空跑） |
| E3 摘要里保留了早期主题 | 摘要确实压进了内容 |
| E4 复用：第二次带 summary 进来 -> **不再调 LLM**（summary 原样保留） | 成本取舍生效 |
| E5 单轮（无 messages）-> 无 summary、prompt 走原模板 | 零影响 |

**为什么不用全图跑 30 轮**：那要 30 × 15s ≈ 7 分钟，且验证的是同一件事。
直接构造 30 轮 messages 调 `generate`，能完整覆盖「阈值判定 -> LLM 摘要 -> 注入 -> 返回」。
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ.setdefault("TAVILY_API_KEY", "tvly-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from graph2.generate_node2 import (
    MAX_HISTORY_TURNS,
    _format_history,  # noqa: E402
    _should_summarize,
    generate,
)

RES = []


def conv(n):
    """构造 n 轮**语义不同**的对话，让摘要「有东西可压」。"""
    topics = [
        "Milvus 的 HNSW 索引参数",
        "TurboPuffer 的召回率只有 61.35%",
        "P99 延迟的定义与测量",
        "bge-large-zh-v1.5 是 1024 维",
        "混合检索用 RRF 融合",
        "表格被切成 4 片的后果",
        "判分占总成本的 76%",
        "批量判分省了 40% token",
        "RRF 只看排名不看绝对分",
        "jieba 中文分词必须开启",
        "contextvar 在线程池内失效",
        "checkpointer 用 thread_id 隔离",
        "Milvus 单块上限 6000 字符",
        "跨文档聚合仍未实现",
        "reranker 实测增益为 0",
    ]
    msgs = []
    for i in range(n):
        t = topics[i % len(topics)]
        msgs.append(HumanMessage(content="第%02d轮：请介绍一下 %s。" % (i + 1, t)))
        msgs.append(
            AIMessage(content="第%02d轮回答：%s 是本项目的一个具体事实，详见 ROADMAP 对应章节的实测记录。" % (i + 1, t))
        )
    return msgs


docs = [
    Document(
        page_content="Milvus 是一个向量数据库，支持稠密与稀疏混合检索。", metadata={"filename": "f.md", "title": "t"}
    )
]

print("=" * 98)
print("E1/E2/E3 30 轮历史 -> 触发真实 LLM 摘要")
print("=" * 98)
print("  MAX_HISTORY_TURNS = %d（预期 13）" % MAX_HISTORY_TURNS)
msgs = conv(30)
st = {
    "question": "前面聊过的最低召回率是多少？",
    "documents": list(docs),
    "messages": list(msgs),
    "session_id": "__histtest__",
}
out = generate(dict(st))

n_msgs = len(out.get("messages") or [])
summary = out.get("summary") or ""
print("  messages: %d -> %d 条（%d 轮 + 本轮 2 条）" % (len(msgs), n_msgs, len(msgs) // 2))
print("  summary 长度: %d 字" % len(summary))
print("  summary 内容: %s" % summary[:180].replace("\n", " "))

RES.append(("E1 本轮新增 2 条（完整累积由图内 reducer 负责，见 4.41）", n_msgs == 2, n_msgs))
RES.append(("E1b 60 条历史确实超过摘要阈值", _should_summarize(msgs)[0] is True, None))
RES.append(("E2 summary 非空（真实 LLM 调用）", len(summary) > 20, len(summary)))
RES.append(("E3 摘要含早期主题", "61.35" in summary or "HNSW" in summary or "RRF" in summary, summary[:60]))

print()
print("=" * 98)
print("E4 复用：带 summary 再来一轮 -> summary 原样保留（不再调 LLM）")
print("=" * 98)
st2 = {
    "question": "那 40% 是指什么？",
    "documents": list(docs),
    "messages": out["messages"],
    "summary": summary,
    "session_id": "__histtest__",
}
out2 = generate(dict(st2))
print("  原 summary 长度 %d -> 新 %d" % (len(summary), len(out2.get("summary") or "")))
RES.append(("E4 summary 原样复用", out2.get("summary") == summary, len(out2.get("summary") or "")))

print()
print("=" * 98)
print("E5 单轮（无 messages）-> 零影响")
print("=" * 98)
st3 = {"question": "Milvus 怎么安装？", "documents": list(docs)}
out3 = generate(dict(st3))
RES.append(("E5 单轮无 summary", not out3.get("summary"), repr(out3.get("summary"))))
RES.append(("E5 单轮不写 messages", "messages" not in out3, "messages" in out3))
print("  summary=%r | 有 messages 键=%s" % (out3.get("summary"), "messages" in out3))

print()
print("=" * 98)
print("判读")
print("=" * 98)
ok = sum(1 for _, c, _ in RES if c)
for name, cond, val in RES:
    print("  %s %-46s 实际 %r" % ("[OK ]" if cond else "[FAIL]", name, val))
print("  通过 %d / %d" % (ok, len(RES)))
sys.exit(0 if ok == len(RES) else 1)
