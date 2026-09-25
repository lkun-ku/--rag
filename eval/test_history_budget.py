"""历史上下文：轮次提升 + 预算约束 + 摘要压缩 —— 契约测试（ROADMAP 4.43）。

## 本轮改什么（规划审视后的结论）

实测（`eval/_budget_history.py`，真实数据）：

    max_turns=3  -> 1,079 字符    max_turns=13 -> 4,672 字符    占 prompt 22%
    question 中位 27 字，golden answer 中位 12 字 —— **本项目答案普遍短**

所以：轮次 3→13 **可行**；但 `text[:300]` 是**错误的约束形式**
（答案短时形同虚设，出现长答案时会把 13 轮全砍残）。
改为**总预算**约束：`MAX_HISTORY_CHARS` 内保最近，超出从最老的丢。

## 契约

| # | 契约 |
|---|---|
| H1 | `MAX_HISTORY_TURNS == 13` |
| H2 | 有**总预算** `MAX_HISTORY_CHARS`，超限从**最老**的开始丢（保最近） |
| H3 | 单条上限提高到 500（防御性，不再是主约束） |
| H4 | `messages` 超 `2 × MAX_HISTORY_TURNS` 时**才**触发摘要 |
| H5 | `build_history_prompt_part` 输出 = 摘要 + 最近 N 轮，顺序正确 |
| H6 | 无 history 时摘要为空（单轮零影响） |
| H7 | `_format_history` 保持纯函数、无 LLM 调用 |
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from graph2 import generate_node2 as gn  # noqa: E402

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


def conv(n, q_len=27, a_len=44):
    """构造 n 轮对话（用真实的中位长度：question 27 字 / answer 44 字）。"""
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content="第%02d轮的问题：" % (i + 1) + "问" * max(q_len - 8, 1)))
        msgs.append(AIMessage(content="第%02d轮的答案：" % (i + 1) + "答" * max(a_len - 8, 1)))
    return msgs


print("=" * 100)
print("H1/H3 常量")
print("=" * 100)
check("H1 MAX_HISTORY_TURNS == 13", gn.MAX_HISTORY_TURNS == 13, gn.MAX_HISTORY_TURNS)
check("H3 单条上限 >= 500", gn.MAX_PER_MSG_CHARS >= 500, gn.MAX_PER_MSG_CHARS)
check("H2 有总预算常量", isinstance(gn.MAX_HISTORY_CHARS, int) and gn.MAX_HISTORY_CHARS > 0,
      gn.MAX_HISTORY_CHARS)

print()
print("=" * 100)
print("H2 总预算：超出时从**最老**的丢起，保住最近")
print("=" * 100)
msgs = conv(15)
out_full = gn._format_history(msgs, max_turns=gn.MAX_HISTORY_TURNS)
print("  15 轮完整输出: %d 字符（预算 %d）" % (len(out_full), gn.MAX_HISTORY_CHARS))

# 人为把预算压到很小，验证「保最近丢最老」
small = gn.build_history_prompt_part(msgs, budget=1200)
lines = small["history"].splitlines()
first_kept = lines[0] if lines else ""
last_kept = lines[-1] if lines else ""
check("小预算下第一行是**较新**的轮次（老轮被丢）",
      "第1" not in first_kept[:8] and "第2" not in first_kept[:8],
      "首行: %s" % first_kept[:40])
check("小预算下最后一行仍是最新一轮", "第15轮" in last_kept, "末行: %s" % last_kept[:40])
check("预算足够时不丢轮次（仍受 max_turns=13 限制 -> 26 行）",
      len(small["history"].splitlines()) == 26,
      len(small["history"].splitlines()))

print()
print("=" * 100)
print("H3 单条上限：超长答案被截断但不影响其它条")
print("=" * 100)
long_a = AIMessage(content="很长很长的答案：" + "内" * 900)
short_a = AIMessage(content="短答案")
out = gn._format_history([HumanMessage(content="问题"), long_a, short_a],
                         max_turns=gn.MAX_HISTORY_TURNS)
check("900 字答案被截到 <= 500", "内" * 501 not in out)
check("短答案完整保留", "短答案" in out)

print()
print("=" * 100)
print("H4/H6 摘要触发条件")
print("=" * 100)
n_trigger = gn.SUMMARY_TRIGGER_TURNS
check("H4 SUMMARY_TRIGGER_TURNS == 2 × MAX_HISTORY_TURNS",
      n_trigger == 2 * gn.MAX_HISTORY_TURNS, n_trigger)

# 13 轮（未超阈值）-> 不触发摘要
ok, old = gn._should_summarize(conv(13))
check("恰好在阈值内不触发摘要", ok is False, old[:40])

# 15 轮（未超阈值）
ok, old = gn._should_summarize(conv(15))
check("15 轮仍不触发（阈值 26 轮）", ok is False, old[:40])

# 20 轮（40 条 < 阈值 52 条）-> 仍不触发
ok, old = gn._should_summarize(conv(20))
check("20 轮仍不触发（阈值是 26 轮 = 52 条）", ok is False, old[:40])

# 30 轮（60 条 > 52 条）-> 触发，并返回「待摘要的旧内容」
ok, old = gn._should_summarize(conv(30))
check("H4 超过 26 轮才触发摘要", ok is True)
check("触发时返回待摘要的旧内容（含第01轮）", "第01轮" in old, old[:60])
check("待摘要内容不含最近 13 轮（第18-30轮）", "第18轮" not in old and "第30轮" not in old)

# H6 无历史
ok, summary = gn._should_summarize([])
check("H6 无历史不触发摘要", ok is False and summary == "")

print()
print("=" * 100)
print("H5 build_history_prompt_part 组合顺序")
print("=" * 100)
# summary 由调用方（generate，会调 LLM）生成后传入 —— 本函数保持**纯函数**
sample_summary = "【更早内容】第1-6轮讨论了 Milvus 的召回率指标。"
part = gn.build_history_prompt_part(conv(20), budget=10 ** 9, summary=sample_summary)
check("返回值含 summary / history 两个键",
      set(["summary", "history"]) <= set(part.keys()), sorted(part.keys()))
check("summary 原样带入（纯函数，不自己调 LLM）",
      part["summary"] == sample_summary, part["summary"][:40])
check("history 仍是最新 N 轮", "第20轮" in part["history"], part["history"][-40:])

# 单轮/无历史
part0 = gn.build_history_prompt_part([], budget=10 ** 9)
check("无历史 -> summary 空且 history 空",
      part0["summary"] == "" and part0["history"] == "")

print()
print("=" * 100)
print("H7 纯函数：无 LLM、无网络、无 Milvus 依赖")
print("=" * 100)
src = pathlib.Path("RAG_PROJECT/graph2/generate_node2.py").read_text(encoding="utf-8")
# **只切 _format_history 自己**（到下一个顶层 def 为止）。
# 首版切到 `def generate`，把中间插入的 `_should_summarize` / `_summarize`
# （后者**合理地在调 llm**）也圈了进去 -> 误报。
a = src.find("def _format_history")
b = src.find("\ndef ", a + 1)
fn = src[a:b if b > 0 else None]
check("_format_history 内不出现 llm / invoke",
      "llm" not in fn and ".invoke" not in fn,
      "切片 %d 字符" % len(fn))

print()
print("%d/%d 通过" % (TOTAL - len(FAILED), TOTAL))
if FAILED:
    print("失败项：%s" % FAILED)
    sys.exit(1)
