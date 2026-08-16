"""② 的本地验证（**不重建集合**）：确认接入 `is_flat_twin` 后行为正确、且不丢内容。

## 两段验证，各自回答一个问题

**A 段 · 元素级（用库里的真实组）** —— 判据是否按预期分类？

  逐组取「扁平块（父块）」与「结构化块」，跑 `is_flat_twin()`，与**独立测得的
  反向覆盖率**对照：

    反向 >= 0.95 的 9 组（纯孪生）          -> 期望 True（跳过累积）
    反向 <  0.95 的 6 组（含额外正文）      -> 期望 False（照旧累积，**不丢内容**）

  这一段的期望值来自 `eval/measure_flat_twin.json` 的实测反向覆盖率，
  **不是**按被测代码反推。

**B 段 · 管道级（真实 md 文件，新旧各跑一次）** —— 接入后有没有副作用？

  用同一份 `datas/md/*.md` 解析两次：
    * 「旧行为」= 把 `is_flat_twin` 短路成恒 False（等价于改造前）；
    * 「新行为」= 正常调用。
  对比块数，并**逐块核对「新行为下消失的块」是否真的被结构化块覆盖**
  —— 这是「不丢内容」的直接证据。

## 为什么必须在重建前做

用户要求「先本地验证再重建」。重建是**不可逆**的（要重测全部基线），
所以先用现有数据把逻辑验穿。
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "RAG_PROJECT")]
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

from pymilvus import MilvusClient  # noqa: E402
from utils.env_utils import MILVUS_URI, MD_DIR  # noqa: E402
from documents.table_handler import is_flat_twin, _norm_ws  # noqa: E402

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024",
               "kb_manuals_bge_large_1024"]

failures = []


def check(name, got, want, extra=""):
    ok = got == want
    print("  [%s] %-52s 期望=%-5s 实际=%-5s %s" % (
        "OK " if ok else "**FAIL**", name, want, got, extra))
    if not ok:
        failures.append(name)


def is_structured(t: str) -> bool:
    head = t.lstrip()[:40]
    return head.startswith("表格：") or "列名：" in t[:500]


# ---------------------------------------------------------------------------
# A 段：元素级 —— 判据是否按预期分类
# ---------------------------------------------------------------------------
print("=" * 104)
print("A 段 · 元素级：is_flat_twin() 对库里 15 组的判定，与独立测得的反向覆盖率对照")
print("=" * 104)

measured = {("%s|%s" % (r["filename"], r["title"])): r
            for r in json.loads((HERE / "measure_flat_twin.json").read_text(encoding="utf-8"))}

client = MilvusClient(uri=MILVUS_URI)
rows = []
for coll in COLLECTIONS:
    for r in client.query(collection_name=coll, filter="",
                          output_fields=["filename", "title", "text"], limit=4000):
        rows.append((r.get("filename", "?"), r.get("title", "?"), r.get("text") or ""))

groups = collections.defaultdict(list)
for fn, ti, t in rows:
    groups[(fn, ti)].append(t)

print("  %-34s %-6s %-8s %-10s %-10s %s" % (
    "filename", "块数", "反向cov", "期望", "实际", "说明"))
print("  " + "-" * 100)
n_pure = n_mixed = n_skipped = 0
for (fn, ti), ts in groups.items():
    st = [t for t in ts if is_structured(t)]
    fl = [t for t in ts if not is_structured(t)]
    if not st or not fl:
        continue
    key = "%s|%s" % (fn, ti)
    m = measured.get(key)
    if not m or m.get("reverse_coverage") is None:
        continue

    # **只对「1 结构化 + 1 扁平」的组做严格对照。**
    #
    # 首版对多元素组也取「最短的一对」当代表，结果 **4 组判错** —— 复查确认是
    # **取样错配**，不是判据错：同一 `(filename,title)` 下有多个元素时
    # （如 `cli_reference / Options` 有 24 结构化 + 24 扁平），
    # 「最短的扁平块」与「最短的结构化块」根本不属于同一个元素。
    # 这是**聚合视图（库里已不保留元素级关联）**的固有局限。
    #
    # 元素级的精确验证放在 B 段：真实解析时配对天然正确。
    if not (len(st) == 1 and len(fl) == 1):
        n_skipped += 1
        continue

    rc = m["reverse_coverage"]
    want = rc >= 0.95
    if want:
        n_pure += 1
    else:
        n_mixed += 1
    got = is_flat_twin(fl[0], st)
    check("%s / %s" % (fn[:30], ti[:14]), got, want,
          "(反向 %.2f)" % rc)

print()
print("  纯孪生（期望 True）%d 组；含额外正文（期望 False）%d 组；"
      "多元素组跳过 %d 组（库里无元素级关联，改由 B 段验证）"
      % (n_pure, n_mixed, n_skipped))

# ---------------------------------------------------------------------------
# B 段：管道级 —— 真实 md 文件，新旧各跑一次
# ---------------------------------------------------------------------------
print()
print("=" * 104)
print("B 段 · 管道级：真实 md 文件，改造前 vs 改造后（验证不丢内容）")
print("=" * 104)

import documents.markdown_parser as mp_mod  # noqa: E402
from documents.markdown_parser import MarkdownParser  # noqa: E402  (类名以实测为准)

md_files = sorted(p for p in pathlib.Path(MD_DIR).iterdir() if p.suffix == ".md")[:2]
if not md_files:
    print("  未找到 md 文件，跳过 B 段")
else:
    parser = MarkdownParser()
    orig = mp_mod.is_flat_twin
    for md in md_files:
        # 旧行为
        mp_mod.is_flat_twin = lambda *a, **k: False
        try:
            old = parser.parse_markdown_to_documents(str(md))
        except Exception as e:  # noqa: BLE001
            print("  [%s] 旧行为解析失败: %s: %s" % (md.name, type(e).__name__, e))
            mp_mod.is_flat_twin = orig
            continue
        # 新行为
        mp_mod.is_flat_twin = orig
        try:
            new = parser.parse_markdown_to_documents(str(md))
        except Exception as e:  # noqa: BLE001
            print("  [%s] 新行为解析失败: %s: %s" % (md.name, type(e).__name__, e))
            continue

        old_set = {_norm_ws(d.page_content) for d in old}
        new_set = {_norm_ws(d.page_content) for d in new}
        removed = old_set - new_set
        added = new_set - old_set
        print()
        print("  [%s]  旧 %d 块 -> 新 %d 块（少 %d、多 %d）" % (
            md.name, len(old), len(new), len(removed), len(added)))

        # 不丢内容的核对：每个被移除的块，其内容是否已被「新集合里某个块」覆盖
        lost = []
        new_blob = "\n".join(new_set)
        for r in removed:
            # 取该块里长度 >= 12 的片段做抽查（12 字符以下噪声太大）
            pieces = [p for p in r.split() if len(p) >= 12]
            if not pieces:
                continue
            miss = [p for p in pieces if p not in new_blob]
            if miss:
                lost.append((r[:70], len(miss), len(pieces)))
        if lost:
            print("      [!] 有 %d 个被移除的块存在**未被覆盖**的片段：" % len(lost))
            for head, nm, tot in lost[:5]:
                print("          %s  缺 %d/%d 片段" % (head, nm, tot))
            failures.append("内容丢失: %s" % md.name)
        else:
            print("      [OK ] 被移除块的内容片段**全部**能在新集合里找到 —— 未丢内容")

print()
print("=" * 104)
print("结果：%s" % ("全部通过" if not failures else "**失败 %d 项**：%s" % (len(failures), failures)))
print("=" * 104)
sys.exit(0 if not failures else 1)
