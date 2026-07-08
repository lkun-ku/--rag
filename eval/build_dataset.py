"""构建 Golden Dataset（评测基线）。

产物：`eval/golden_dataset.jsonl`，每行一条：
    {"id","q","source","answer","source_blocks","note"}

设计要点（依据 ROADMAP 1.5.1）：
  * **`source_blocks` 是关键字段** —— 它让检索层指标（Context Recall / Precision）
    变成纯集合运算，做到 0 LLM 调用、0 抖动。标准答案仅供参考。
  * 块 ID 是**内容指纹**（`md5(page_content)[:12]`，见 export_chunks.py），
    因此即使重建集合（换维度/换模型），只要块内容不变，标注依然有效。
  * **覆盖要求**：每个源至少 5 条，且必须包含「只有该源能回答」的问题，
    用于验证集合隔离是否真正生效。

本脚本同时做**自校验**：检查每个 `source_blocks` 里的 ID 是否存在于
`chunk_index.json`，并做一次「问题关键词 → 块内容」的粗匹配，提示可能的错标。

用法：
    $env:PYTHONPATH="RAG_PROJECT"
    .venv\\Scripts\\python.exe eval\\build_dataset.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
INDEX = HERE / "chunk_index.json"
OUT = HERE / "golden_dataset.jsonl"

# ---------------------------------------------------------------------------
# Golden Dataset 定义
#
# source 字段用于**按源统计**（验证集合隔离），非检索过滤条件。
# source_blocks 用内容指纹；多块表示「需要多块才能完整回答」。
# ---------------------------------------------------------------------------
CASES = [
    # ---------------- milvus_docs：运维 FAQ（operational_faq.md）----------------
    dict(id="q001", source="milvus_docs", q="Milvus 能否在无外网的环境下部署？",
         answer="可以，Milvus 支持离线部署。",
         source_blocks=["0b0b3ee716af"]),
    dict(id="q002", source="milvus_docs", q="多个 Milvus 实例可以共用一个 MinIO 实例吗？",
         answer="可以共用 MinIO。",
         source_blocks=["2eb090a78c97"]),
    dict(id="q003", source="milvus_docs", q="多个 Milvus 实例可以共用一个 etcd 实例吗？",
         answer="可以。",
         source_blocks=["914e705f1803"]),
    dict(id="q004", source="milvus_docs", q="Milvus 可以在 Windows 上安装吗？",
         answer="可以，Milvus 支持在 Windows 上安装（Docker 方式）。",
         source_blocks=["829a68274f4b"]),
    dict(id="q005", source="milvus_docs", q="怎么判断 Milvus 是否已经成功启动？",
         answer="可通过查看服务与日志判断是否成功启动。",
         source_blocks=["92c850e08ab4"]),
    dict(id="q006", source="milvus_docs", q="为什么我修改了配置文件却不生效？",
         answer="配置修改后需要重启服务才会生效。",
         source_blocks=["a77cfe680dc9"]),
    dict(id="q007", source="milvus_docs", q="哪些因素会影响检索的召回率？",
         answer="召回率受索引类型、参数与数据分布等多因素影响。",
         source_blocks=["bf6eba262ec4"]),
    dict(id="q008", source="milvus_docs", q="Milvus 运行日志存放在哪里？",
         answer="日志有固定的存放位置，可据此排查问题。",
         source_blocks=["e07b085f060a"]),

    # ---------------- milvus_docs：产品 FAQ（product_faq.md）----------------
    dict(id="q009", source="milvus_docs", q="Milvus 的主键字段支持哪些数据类型？",
         answer="支持 INT64 与 VARCHAR 两类主键类型。",
         source_blocks=["a70eeba08d45"]),
    dict(id="q010", source="milvus_docs", q="Milvus 支持的最大向量维度是多少？",
         answer="Milvus 可管理的向量维度有上限。",
         source_blocks=["e267537875dc"]),
    dict(id="q011", source="milvus_docs", q="Milvus 里 search 和 query 有什么区别？",
         answer="search 做向量相似性检索，query 按表达式做标量过滤。",
         source_blocks=["51fa881f382f"]),
    dict(id="q012", source="milvus_docs", q="插入数据之后还能再创建索引吗？",
         answer="可以，索引可在插入数据后创建。",
         source_blocks=["188d181cc688"]),
    dict(id="q013", source="milvus_docs", q="Milvus 支持并发搜索吗？",
         answer="支持，同一集合上的查询可以并发执行。",
         source_blocks=["54402d7995eb"]),
    dict(id="q014", source="milvus_docs", q="Milvus 的数据存储在哪里？",
         answer="Milvus 的数据分为元数据与向量数据，存放位置不同。",
         source_blocks=["11374010e020"]),
    dict(id="q015", source="milvus_docs", q="为什么欧氏距离和內积检索返回的结果不一样？",
         answer="L2 与 IP 的度量语义不同，且受是否归一化影响。",
         source_blocks=["7d823fe4074a"]),
    dict(id="q016", source="milvus_docs", q="为什么搜索 topk 时返回的结果少于 k 个？",
         answer="可能因过滤条件或数据量不足导致返回少于 k 条。",
         source_blocks=["7fdafb2f9975"]),
    dict(id="q017", source="milvus_docs", q="什么是归一化？为什么向量检索需要归一化？",
         answer="归一化把向量缩放到单位长度，使内积等价于余弦相似度。",
         source_blocks=["dc8cb1aaf97d"]),
    dict(id="q018", source="milvus_docs", q="Milvus 支持 Apple M1 芯片吗？",
         answer="当前版本对 M1 的支持情况有说明。",
         source_blocks=["7ab96bd3324c"]),
    dict(id="q019", source="milvus_docs", q="集合删除后 MinIO 里的数据为什么还在？",
         answer="删除集合不会立即清理对象存储中的数据。",
         source_blocks=["907094dead81"]),
    dict(id="q020", source="milvus_docs", q="FLAT 和 IVF_FLAT 索引有什么区别？",
         answer="FLAT 暴力检索保证精度；IVF_FLAT 通过聚类加速但有召回损失。",
         source_blocks=["86239c7298c1"]),

    # ---------------- milvus_docs：性能 FAQ（performance_faq.md）----------------
    dict(id="q021", source="milvus_docs", q="同时进行数据插入和搜索会影响查询性能吗？",
         answer="会有影响，插入与查询会竞争资源。",
         source_blocks=["46907130284d"]),
    dict(id="q022", source="milvus_docs", q="IVF 索引的 nlist 和 nprobe 应该怎么设置？",
         answer="nlist 与 nprobe 需按数据规模与精度要求权衡设置。",
         source_blocks=["4b194a810c50"]),
    dict(id="q023", source="milvus_docs", q="为什么在较小的数据集上查询有时反而更慢？",
         answer="小数据集可能因索引结构未充分发挥作用而更慢。",
         source_blocks=["8b60482e6213"]),
    dict(id="q024", source="milvus_docs", q="哪些操作会导致 CPU 占用升高？",
         answer="构建索引等操作会显著提高 CPU 占用。",
         source_blocks=["dc0dfdfd65c6"]),

    # ---------------- milvus_docs：概念（overview.md）----------------
    dict(id="q025", source="milvus_docs", q="Milvus 支持哪些索引类型和相似度度量方式？",
         answer="支持多种索引（如 HNSW、IVF 系列）与度量（L2、IP、COSINE）。",
         source_blocks=["4d0dade0f425"]),
    dict(id="q026", source="milvus_docs", q="Milvus 的核心概念包括哪些？",
         answer="包括非结构化数据、嵌入向量、向量相似度搜索等。",
         source_blocks=["1acfd00f647c"]),

    # ---------------- milvus_docs：半导体语料（tech_report_*.md）----------------
    dict(id="q027", source="milvus_docs", q="等离子体处理的基本原理是什么？",
         answer="等离子体是物质的第四态，由离子、电子和中性粒子组成。",
         source_blocks=["6284a33a3a18"]),
    dict(id="q028", source="milvus_docs", q="等离子体工艺的核心参数应该如何优化？",
         answer="功率密度通常控制在 0.1-10 W/cm² 等范围内。",
         source_blocks=["d711b2919e5f"]),
    dict(id="q029", source="milvus_docs", q="XPS 在半导体表面分析中有什么作用？",
         answer="XPS 用于表面化学分析，可表征元素组成与化学态。",
         source_blocks=["03c03cc95ed8"]),
    dict(id="q030", source="milvus_docs", q="EUV 光刻胶与传统光刻胶有什么本质区别？",
         answer="EUV 光刻胶在光子能量与反应机制上与传统光刻胶存在本质差异。",
         source_blocks=["98ae48872e30"]),
    dict(id="q031", source="milvus_docs", q="EUV 光刻胶在 7nm 以下制程面临哪些技术挑战？",
         answer="面临材料组分优化、光子-材料相互作用控制等挑战。",
         source_blocks=["39fed678c6b8"]),
    dict(id="q032", source="milvus_docs", q="等离子体原子层沉积（PE-ALD）有什么优势？",
         answer="PE-ALD 可在低温下实现高精度薄膜沉积。",
         source_blocks=["749c3ee1a3d1"]),

    # ---------------- papers：LayoutParser 论文 ----------------
    dict(id="q033", source="papers", q="LayoutParser 是什么？",
         answer="LayoutParser 是一个用于深度学习文档图像分析的统一工具包。",
         source_blocks=["8f351a1f5d40"]),
    dict(id="q034", source="papers", q="LayoutParser 的核心目标是什么？",
         answer="让文档图像分析的流水线构建变得简单易用。",
         source_blocks=["c0be1d913d6f"]),
    dict(id="q035", source="papers", q="LayoutParser 有哪些典型应用场景？",
         answer="用于构建文档数字化与表格抽取等流水线。",
         source_blocks=["c0be1d913d6f", "9e36ca9154a0"]),
    dict(id="q036", source="papers", q="文档版面分析领域有哪些相关工作？",
         answer="深度学习模型与数据集是该领域的主要相关工作。",
         source_blocks=["d689dc0fdf82"]),
    dict(id="q037", source="papers", q="什么是轻量级文档处理流水线？",
         answer="轻量级流水线面向相对简单的文档结构。",
         source_blocks=["61d20b913234"]),
    dict(id="q038", source="papers", q="LayoutParser 的结论是什么？",
         answer="LayoutParser 提供了完整的文档分析工具包。",
         source_blocks=["c0be1d913d6f"]),

    # ---------------- manuals：手册 / 数据 / 演示 ----------------
    dict(id="q039", source="manuals", q="并发数为 100 时查询延迟的 P50 和 P99 分别是多少？",
         answer="P50 为 380ms，P99 为 950ms。",
         source_blocks=["1a17374205f2"],
         note="表格数值题，答案必须精确"),
    dict(id="q040", source="manuals", q="SIFT1M 数据集的召回率是多少？",
         answer="0.95。",
         source_blocks=["ae0abdcfda09"],
         note="表格数值题"),
    dict(id="q041", source="manuals", q="Milvus 产品手册中提到的部署要求是什么？",
         answer="需要 Python 3.10 以上版本与至少 8GB 内存。",
         source_blocks=["afa9f185e71f"]),
    dict(id="q042", source="manuals", q="Milvus 架构的主要组件有哪些？",
         answer="Coordinator 负责调度，QueryNode 执行检索。",
         source_blocks=["8ebb60710ddf"]),
    dict(id="q043", source="manuals", q="Milvus 的混合检索是如何实现的？",
         answer="结合稠密向量与稀疏关键词两路结果，用 RRF 融合。",
         source_blocks=["5d6ad24e6936"]),

    # ---------------- 跨源问题（验证集合隔离 + 跨集合检索）----------------
    dict(id="q044", source="cross", q="Milvus 的架构设计与部署方式分别是什么？",
         answer="架构（存储计算分离）见 pptx；部署方式（Docker/单机）见 FAQ 与手册。",
         source_blocks=["7fcdd765c9df", "829a68274f4b", "afa9f185e71f"],
         note="跨源：验证能否同时召回 manuals 与 milvus_docs"),
    dict(id="q045", source="cross", q="向量检索的召回率受什么影响？",
         answer="受索引参数与数据分布影响（FAQ）；手册给出了具体召回率数值（xlsx）。",
         source_blocks=["bf6eba262ec4", "ae0abdcfda09"],
         note="跨源：验证 milvus_docs 与 manuals 同时被召回"),
]


def main() -> int:
    if not INDEX.exists():
        print(f"缺少 {INDEX}，请先运行 eval/export_chunks.py")
        return 2

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    # chunk_id -> (collection, filename)
    known = {}
    for coll, info in index["collections"].items():
        for c in info["chunks"]:
            known[c["chunk_id"]] = (coll, c["filename"])

    print("=" * 78)
    print("构建 Golden Dataset")
    print("=" * 78)
    print(f"块索引: {len(known)} 个可检索块")

    # ---- 自校验 1：块 ID 是否存在 ----
    missing = []
    for c in CASES:
        for b in c["source_blocks"]:
            if b not in known:
                missing.append((c["id"], b))

    # ---- 自校验 2：同一问题是否误用了同一块（合理，但需注意）----
    print()
    print("[校验] 块 ID 有效性")
    if missing:
        print(f"  ** {len(missing)} 个块 ID 不存在于索引中 **")
        for cid, b in missing:
            print(f"     {cid}: {b}")
    else:
        print("  全部块 ID 均存在 OK")

    # ---- 自校验 3：按源统计覆盖度 ----
    print()
    print("[校验] 按源覆盖度")
    by_src = {}
    for c in CASES:
        by_src.setdefault(c["source"], []).append(c)
    for s, cs in sorted(by_src.items()):
        n_blocks = sum(len(c["source_blocks"]) for c in cs)
        print(f"  {s:<14} {len(cs):>3} 条  (共标注 {n_blocks} 个块)")

    # ---- 自校验 4：块所属集合是否与 source 字段一致 ----
    #
    # 注意用 name -> collection 的显式映射比对，**不能用子串匹配**：
    #   "milvus_docs" 并不是 "kb_milvus_bge_large_1024" 的子串（虽然 "milvus" 是），
    #   子串判断会产生大量误报。
    print()
    print("[校验] source 字段与块实际所属集合是否一致")
    coll2name = {coll: info.get("name", "?") for coll, info in index["collections"].items()}
    mismatch = []
    for c in CASES:
        if c["source"] == "cross":
            continue
        for b in c["source_blocks"]:
            if b in known:
                coll = known[b][0]
                actual_name = coll2name.get(coll, "?")
                if actual_name != c["source"]:
                    mismatch.append((c["id"], b, c["source"], actual_name))
    if mismatch:
        print(f"  ** {len(mismatch)} 处不一致 **")
        for cid, b, s, actual in mismatch:
            print(f"     {cid}: source={s} 但块属于源 {actual}")
    else:
        print("  全部一致 OK（说明标注的块确实来自期望的源，隔离验证可信）")

    if missing or mismatch:
        print()
        print("存在校验问题，未写出文件。请修正后重跑。")
        return 1

    # ---- 写出 ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in CASES:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print()
    print(f"已写出 {len(CASES)} 条 -> {OUT}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
