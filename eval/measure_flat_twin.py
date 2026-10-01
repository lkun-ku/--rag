"""② 的 fixture 测量：为 `is_flat_twin()` 标定判据阈值（**先测再写测试**）。

## 为什么必须先测

用户已定判据方向：**严格向** —— 「结构化块的每个值都必须在扁平文本里找到，
**且长度近似**；宁可漏判（维持现状）也不误删内容」。

但「长度近似」**不能凭感觉定阈值**：
* `milvus_cli_reference/Options`：结构化 729 vs 扁平 762 -> 近似；
* `vector_db_benchmark/检索性能实测`：结构化 3 段合计约 8900 vs 扁平 4912 -> **差近一倍**。

后者是因为 **Markdown 表格天然比扁平文本长**（有 `|`、`---`、每段重复表头）。
所以阈值必须由**实测分布**定，否则会把真孪生判成非孪生（漏判）或反之（误删）。

## 测量口径

对每个 `(filename, title)` 组：分出「结构化块」（`表格：` 前缀 / `列名：`）与「扁平块」，
计算：
  1. 结构化块的**单元格值集合** —— 扁平文本对它的**覆盖率**（值是否都在扁平里出现）；
  2. **长度比** = 扁平总长 / 结构化总长（以及 / 结构化最长块）；
  3. 反向覆盖率（扁平里的词是否都在结构化里）—— 判断是否扁平**含结构化没有的内容**。

输出按「覆盖率」排序，用于**定阈值**与**挑 fixture**。
"""

from __future__ import annotations

import collections
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

COLLECTIONS = ["kb_milvus_bge_large_1024", "kb_papers_bge_large_1024", "kb_manuals_bge_large_1024"]


def is_structured(t: str) -> bool:
    head = t.lstrip()[:40]
    return head.startswith("表格：") or "列名：" in t[:500]


def cell_values(chunks):
    """结构化块的「单元格值集合」—— 取 Markdown 表格行里 `|` 分隔的长值。

    只取长度 >= 2 的值，并过滤 `---` 分隔行与纯数字（数字太短，覆盖率无判别力，
    且容易因格式差异（`61.35%` vs `61.35 %`）误判）。
    """
    vals = set()
    for c in chunks:
        for line in c.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if set(line) <= set("|- "):
                continue
            for v in line.strip("|").split("|"):
                v = v.strip()
                if len(v) >= 4 and not re.fullmatch(r"[\d.,%\s/-]+", v):
                    vals.add(v)
    return vals


client = MilvusClient(uri=MILVUS_URI)
rows = []
for coll in COLLECTIONS:
    for r in client.query(collection_name=coll, filter="", output_fields=["filename", "title", "text"], limit=4000):
        rows.append((r.get("filename", "?"), r.get("title", "?"), r.get("text") or ""))

groups = collections.defaultdict(list)
for fn, ti, t in rows:
    groups[(fn, ti)].append(t)

out = []
for (fn, ti), ts in groups.items():
    st = [t for t in ts if is_structured(t)]
    fl = [t for t in ts if not is_structured(t)]
    if not st or not fl:
        continue
    vals = cell_values(st)
    flat_all = "\n".join(fl)
    flat_norm = re.sub(r"\s+", " ", flat_all)
    if vals:
        hit = sum(1 for v in vals if re.sub(r"\s+", " ", v) in flat_norm)
        cov = hit / len(vals)
    else:
        cov = float("nan")
    len_st = sum(len(t) for t in st)
    len_fl = sum(len(t) for t in fl)

    # [新增 ] **反向覆盖** —— 扁平文本里有没有「结构化块完全没有」的内容。
    #
    # 为什么必须加这一维：取真实样本时发现两类形态，**正向覆盖率都 1.00，但性质不同**：
    #   样本 B（milvus_capability_matrix / 来源说明）扁平 = 结构化去掉 `|`/`---`  -> 纯孪生；
    #   样本 A（milvus_index_spec / 6.2 依赖组件版本）扁平里多了
    #       「6 部署环境要求 Before installing a Milvus Standalone instance, check…」
    #       「出处：milvus.io v2.6.x — Requirements…」
    #   —— 说明该扁平块**已被父标题合并逻辑并入了周围正文**，**直接删会丢内容**。
    #
    # 判据：取扁平文本里的「长词」（>= 6 字符，含中文），看有多少能在结构化块里找到。
    # 纯孪生应接近 1.00；含额外正文的会明显偏低。
    st_all = "\n".join(st)
    words = set(re.findall(r"[A-Za-z\u4e00-\u9fff0-9./_-]{6,}", flat_all))
    # 只保留「像内容词」的：排除纯数字/纯标点
    words = {w for w in words if re.search(r"[A-Za-z\u4e00-\u9fff]", w)}
    rev_cov = (sum(1 for w in words if w in st_all) / len(words)) if words else float("nan")

    out.append(
        {
            "filename": fn,
            "title": ti,
            "n_structured": len(st),
            "n_flat": len(fl),
            "len_structured": len_st,
            "len_flat": len_fl,
            "len_max_structured": max(len(t) for t in st),
            "ratio_flat_over_structured": round(len_fl / max(len_st, 1), 3),
            "ratio_flat_over_max_structured": round(len_fl / max(max(len(t) for t in st), 1), 3),
            "value_coverage": round(cov, 4) if cov == cov else None,
            "reverse_coverage": round(rev_cov, 4) if rev_cov == rev_cov else None,
            "n_words_flat": len(words),
            "n_values": len(vals),
        }
    )

out.sort(key=lambda r: (-(r["value_coverage"] or 0), -r["n_structured"]))
print("=" * 112)
print("② fixture 测量：同时含「结构化块」与「扁平块」的组（共 %d 组）" % len(out))
print("=" * 112)
print(
    "  %-38s %-18s %-6s %-6s %-8s %-8s %-8s %s"
    % ("filename", "title", "结构", "扁平", "覆盖率", "长度比", "最长比", "值数")
)
print("  " + "-" * 108)
for r in out[:26]:
    print(
        "  %-38s %-18s %-6d %-6d %-8s %-8.2f %-8.2f %d"
        % (
            r["filename"][:38],
            r["title"][:18],
            r["n_structured"],
            r["n_flat"],
            ("%.2f" % r["value_coverage"]) if r["value_coverage"] is not None else "n/a",
            r["ratio_flat_over_structured"],
            r["ratio_flat_over_max_structured"],
            r["n_values"],
        )
    )

# 分布统计：用于定阈值
covs = [r["value_coverage"] for r in out if r["value_coverage"] is not None]
print()
print("  覆盖率分布：")
if covs:
    hi = [c for c in covs if c >= 0.99]
    mid = [c for c in covs if 0.5 <= c < 0.99]
    lo = [c for c in covs if c < 0.5]
    print("     >= 0.99（值全在扁平里）: %d 组  <- 强孪生候选" % len(hi))
    print("     0.50~0.99            : %d 组" % len(mid))
    print("     <  0.50              : %d 组  <- 扁平里找不到，非孪生" % len(lo))
print()
print("  **判读**：覆盖率 >= 0.99 的组才是「扁平孪生」的强候选；")
print("          低于此值说明扁平文本并不含全部结构化值（可能被截断或是另一张表）。")

(HERE / "measure_flat_twin.json").write_text(
    json.dumps(
        {
            "measurement_granularity": "group",  # 见下方说明
            "groups": out,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 112)
print("  **粒度限制（[FIX / 二次筛查] 必须显式标注）**")
print("=" * 112)
print("  本脚本的阈值是**组级**测量的：按 `(filename, title)` 把块聚合成一组再算")
print("  「正向覆盖 / 反向覆盖 / 长度比」。而生产里的 `is_flat_twin(flat_text,")
print("  table_chunks)` 是**元素级**的（同一个表格元素的扁平文本 + 它自己的结构化块）。")
print()
print("  **两者不等价**，原因：")
print("    * 库里**不保留元素级关联**（入库时没有写入元素 id 一类的字段），")
print("      所以**无法**从现有数据还原「哪个扁平块对应哪几个结构化块」；")
print("    * 同一 `(filename,title)` 下若有多组元素，组级统计是它们的**汇总**，")
print("      与「单个元素」的分布可以差很远（4.17.5 记录过：组级与元素级曾给出")
print("      4 组不同的判定）。")
print()
print("  **所以这些阈值是「近似上界」，不是元素级真值。**")
print("  元素级的精确验证在**生产路径**上做：`eval/verify_flat_twin_pipeline.py` 的")
print("  B 段（真实 md 文件跑解析，配对天然正确）—— 那里验证了「未丢内容」。")
print()
print("  **若要元素级阈值**，正确做法是**在解析流程里插桩**：")
print("  在 `markdown_parser.merge_documents` 调用 `is_flat_twin` 的位置，")
print("  把 `(扁平文本, 结构化块)` 实际长度对输出到日志，再离线统计 ——")
print("  那才是与生产完全同源的口径。**本轮未做**（改动侵入解析主流程，风险不对等）。")
print()
print("  证据 -> eval/measure_flat_twin.json（结构含 measurement_granularity 字段）")
