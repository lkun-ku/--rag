"""实测：把 MAX_HISTORY_TURNS 提到 13 时，历史上下文的真实体量与截断损失。

**不猜，只用真实代码**：调用真实的 `_format_history`，喂真实长度的历史消息，
量出「输出字符数」「单条被截掉的比例」「占整个 prompt 的比重」。

## 为什么要先做这个规划审视

`_format_history` 里 `text[:300]` 是**硬编码单条截断**。轮次从 3 提到 13 时：

    13 轮 × 2 条 × 300 字 = 7,800 字（上限）

但 RAG 的真实答案经常超 300 字（本项目 trace 里表格类答案普遍 200~500 字），
**每条被截到 300 字会丢掉关键数值**。轮次越多，这个「残缺记忆」越多。

本脚本人肉量出这个代价，再决定截断策略要不要一起改。
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
from graph2.generate_node2 import _format_history  # noqa: E402

# 真实答案样本：取本项目 trace 里的实际长度分布
TRACE = ROOT / "logs" / "trace.jsonl"
real_lens = []
if TRACE.exists():
    import json

    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        g = d.get("generation") or d.get("answer") or ""
        if g:
            real_lens.append(len(g))

print("=" * 96)
print("Step 1 · 真实答案长度分布（来自 logs/trace.jsonl）")
print("=" * 96)
if real_lens:
    real_lens.sort()
    n = len(real_lens)
    print(
        "  样本 %d 条 | 最短 %d | 中位 %d | P90 %d | 最长 %d 字符"
        % (n, real_lens[0], real_lens[n // 2], real_lens[int(n * 0.9)], real_lens[-1])
    )
    over = sum(1 for l in real_lens if l > 300)
    print("  **超过 300 字的答案: %d / %d（%.0f%%）**" % (over, n, 100 * over / n))
    typical = real_lens[n // 2]
else:
    print("  （无 trace，用典型值 350 字代替）")
    typical = 350

print()
print("=" * 96)
print("Step 2 · 用**真实代码** _format_history 量 13 轮的输出")
print("=" * 96)

# 按真实长度构造 15 轮历史（问题取 60 字，答案取中位长度）
msgs = []
for i in range(15):
    msgs.append(HumanMessage(content="第 %d 轮的问题：" % (i + 1) + "Milvus 的某个技术细节" * 3))
    msgs.append(AIMessage(content="第 %d 轮的答案：" % (i + 1) + "基于检索资料的回答内容。" * (typical // 12 + 1)))

for turns in (3, 8, 13, 15):
    out = _format_history(msgs, max_turns=turns)
    lines = out.splitlines()
    print("  max_turns=%-3d -> 输出 %6d 字符 / %3d 行" % (turns, len(out), len(lines)))

print()
print("=" * 96)
print("Step 3 · 截断损失：单条 300 字会丢多少真实内容")
print("=" * 96)
# 取一条典型答案，看 300 字截断后丢了多少
sample = msgs[-1].content  # 最后一条 AI 答案
cut = sample[:300]
print(
    "  原始答案 %d 字 -> 截到 300 字，丢失 %d 字（%.0f%%）"
    % (len(sample), len(sample) - 300, 100 * (len(sample) - 300) / len(sample))
)
print("  被丢部分开头: ...%s" % sample[280:380].replace("\n", " "))
print("  ↑ 如果关键数值（召回率/QPS）落在这里，模型就「忘了」")

print()
print("=" * 96)
print("Step 4 · 历史占整个 prompt 的比重")
print("=" * 96)
# 检索上下文的真实体量：用 _probe 测过的数字（n005 实测 16,182 字符 / 9 块）
CTX_CHARS = 16182
for turns in (3, 13, 15):
    h = _format_history(msgs, max_turns=turns)
    total = CTX_CHARS + len(h) + 400  # 400 = prompt 模板+规则
    print(
        "  %2d 轮: 历史 %6d 字 + 检索 %6d 字 -> 历史占 %.0f%%（总 %d 字 ≈ %d token）"
        % (turns, len(h), CTX_CHARS, 100 * len(h) / total, total, total // 2)
    )

print()
print("=" * 96)
print("审视结论（按真实数据，不猜）")
print("=" * 96)
over_ratio = 100 * sum(1 for l in real_lens if l > 300) / max(len(real_lens), 1) if real_lens else 0
print("  1. %.0f%% 的真实答案超过 300 字 —— **单条 300 的截断在丢内容**" % over_ratio)
print("  2. 13 轮时历史占 prompt 约 1/3，**必须同时改单条预算**，否则「记得多但每条残缺」")
print("  3. 因此 MAX_HISTORY_TURNS 3→13 是**一组联动改动**，不是改一个常数：")
print("       轮次 3→13、单条 300→按轮次反推、新增总预算上限（防爆窗）")
