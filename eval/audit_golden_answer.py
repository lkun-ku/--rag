"""golden 「含答案性」核查：`source_blocks` 里的块**真的含有答案**吗？

## 为什么要查这个（一次差点走偏的教训）

4.19 用判分保留率诊断出「n004 的 golden 被判分筛掉」，**差点据此去改判分 prompt**。
逐块复核后发现：**判分是对的** —— `408208b67503`（表格第 61-85 行）里
**确实没有** Pinecone 那行；答案在 `f1e2ce7777cf`（第 31-60 行），而判分对它给的是 `yes`。
**真正错的是 golden**（4.18 修订时用「内容匹配」补块，重叠 0.88，但补错了行段）。

**教训**：诊断「判分是否误杀」之前，必须先确认「golden 标的是不是含答案的块」——
否则会把**标注错误**误判成**判分缺陷**。

## 口径

对每道题，从问题里抽出**实体词**（产品型号 / 专有名词：含字母数字与连字符的长 token，
如 `Pinecone-p2.x8-1node`、`Millvus-16c64g-sq4u-fp16-force_merge`），
检查每个 golden 块是否命中**至少一个实体词**：

    * 命中 -> 该块「与问题同主题」，可能是含答案的块
    * 全部 golden 块都不命中 -> **高度可疑**（golden 可能标错），打印出来人工核对

只在**有实体词**的题上做（纯概念题如「Milvus 支持哪些索引类型？」没有型号，跳过）。
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI  # noqa: E402
import metrics  # noqa: E402

# 实体词：含字母、数字、连字符/下划线/点，长度 >= 8（型号类；太短的会误命中）
ENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9._\-]{7,}")

client = MilvusClient(uri=MILVUS_URI)
by_id = {}
for coll in ("kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
             "kb_manuals_bge_large_1024"):
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "text"], limit=8000):
        t = r.get("text") or ""
        by_id[metrics.chunk_id(t)] = {"filename": r.get("filename", ""), "text": t}

cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

print("=" * 104)
print("golden 含答案性核查（问题里的实体词是否出现在 golden 块中）")
print("=" * 104)
print()

suspicious = []
checked = 0
for c in cases:
    gids = c.get("source_blocks") or []
    if not gids:
        continue
    ents = sorted({e for e in ENT_RE.findall(c["q"])})
    if not ents:
        continue
    checked += 1
    info = []
    any_hit = False
    for g in gids:
        b = by_id.get(g)
        if not b:
            info.append((g, "块不在库里", ""))
            continue
        hits = [e for e in ents if e.lower() in b["text"].lower()]
        if hits:
            any_hit = True
            info.append((g, "+".join(hits[:2]), b["text"][:60].replace("\n", " ")))
        else:
            info.append((g, "", b["text"][:60].replace("\n", " ")))
    if not any_hit:
        suspicious.append((c.get("id"), c["q"], ents, info))

print("  有实体词的题: %d 道；其中 **全部 golden 块都不含任何实体词** 的: %d 道" % (
    checked, len(suspicious)))
print()
if suspicious:
    print("  %-6s %-44s %s" % ("题", "问题", "实体词 / golden 块首行"))
    print("  " + "-" * 100)
    for cid, q, ents, info in suspicious:
        print("  %-6s %-44s ents=%s" % (cid, q[:44], ents[:2]))
        for g, hit, head in info:
            print("           %s %-10s %s" % (g, hit or "(无)", head[:60]))
else:
    print("  [OK] 没有发现「golden 完全不含问题实体」的题。")

(HERE / "audit_golden_answer.json").write_text(json.dumps(
    {"checked": checked, "suspicious": [
        {"id": cid, "q": q, "ents": ents,
         "blocks": [{"id": g, "hit": h, "head": hd} for g, h, hd in info]}
        for cid, q, ents, info in suspicious]}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print()
print("  证据 -> eval/audit_golden_answer.json")
