"""修正 n005 的 golden 答案（**golden 标错**，非模型问题）。

## 证据链

1. **模型实测**（`_probe_n005_fulltable.py`）：完整表与生产上下文下**都作答**，
   答案均为「**TurboPuffer，61.35%**」；
2. **库内核对**（85 条召回率记录）：**全表最低 = 61.35%**
   （TurboPuffer / 带过滤向量检索 / 过滤 98.00%）；
3. **golden 原标 83.52%** 取的是「TurboPuffer / 静态向量检索 / 0% 过滤」那一行，
   **不是全表最低**；
4. **61.35% 与 83.52% 在同一个 chunk `f1e2ce7777cf` 内** ——
   所以 **`source_blocks` 无需改动**，只有 `answer` 错了。

## 为什么这个问题会标错（值得记）

问句「参与评测的产品里，召回率最低的是哪个？」**本身有歧义**：
每个产品在表里**出现多行**（不同场景 / 过滤比例），
所以「产品的召回率」不是一个唯一值。

原标注按「**静态向量检索 / 0% 过滤**」这一子集取的，而我们后来按
「全表取最小」核对 —— 两种解读不同，**标注时没写明按哪种**。

**本次修正同时把解读写进 note**，避免下次再歧义。
"""
from __future__ import annotations

import json
import pathlib

P = pathlib.Path(__file__).parent / "golden_dataset.jsonl"
lines = [l for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]

NOTE = (
    "跨行比较 + 取最小（**2026-10-17 修正**）：原标 83.52% 取的是"
    "「静态向量检索 / 0% 过滤」那一行，**不是全表最低**。"
    "按「全表 85 条召回率记录取最小」的口径，真实最低是 "
    "「带过滤向量检索 / 98% 过滤」下的 61.35%（TurboPuffer）。"
    "两者同在 chunk f1e2ce7777cf 内，故 source_blocks 不变。"
    "模型实测答出 61.35% —— **是对的**。"
)

out, n = [], 0
for l in lines:
    d = json.loads(l)
    if d.get("id") == "n005":
        old = d.get("answer")
        d["answer"] = "TurboPuffer，61.35%"
        d["note"] = NOTE
        d["answer_fixed_at"] = "2026-10-17"
        d["answer_old"] = old
        n += 1
        print("修正 n005:")
        print("  answer : %s  ->  %s" % (old, d["answer"]))
        print("  note   : 已更新（写明口径与修正原因）")
    out.append(json.dumps(d, ensure_ascii=False))

P.write_text("\n".join(out) + "\n", encoding="utf-8")
print()
print("共修正 %d 条" % n)

d = [json.loads(x) for x in P.read_text(encoding="utf-8").splitlines() if x.strip()]
c = next(x for x in d if x["id"] == "n005")
print("复核 n005: answer=%r" % c["answer"])
print("         source_blocks=%s（未改动）" % c["source_blocks"])
print("         kind=%s" % c["kind"])
print("全库题数 %d" % len(d))
