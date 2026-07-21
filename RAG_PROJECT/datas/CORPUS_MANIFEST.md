# 语料溯源清单（CORPUS MANIFEST）

> 生成日期：2026-10-10 · 对应 `DATASET_SPEC.md`（v2.6 主题：**向量数据库 / 混合检索 / RAG**）
> 原则：**不捏造**。md / pdf 均为网上下载的真实文件；docx / xlsx / pptx 的**每一个数字与每一段引文均可追溯到公开来源**（见下表），
> 仅由脚本做「排版封装」，不改写事实、不新增数据。

## 一、主题设计（三源互证）

```
源1 论文（原理）    milvus_vectordb_paper / hnsw_anns_paper / rag_failure_modes_paper
源2 产品文档（实现）datas/md/ 6 个官方文档 + datas/pdf/ 白皮书
源3 数据（效果）    datas/office/ 3 个 xlsx（VectorDBBench 实测 + 官方基准）+ pptx 图表
```

## 二、逐文件溯源

### datas/md/（6 个，均为官方文档原文）

| 文件 | 原始出处（GitHub: milvus-io/milvus-docs @ v2.6.x） | 处理方式 |
|---|---|---|
| `milvus_product_overview.md` | `site/en/about/overview.md` | 原文（0 改动） |
| `milvus_faq.md` | `site/en/faq/product_faq.md` | 仅把 `#### ` 归一化为 `## `，文字逐字保留（含原有的双空格不规范标题，用于健壮性测试） |
| `milvus_troubleshooting.md` | `site/en/userGuide/tools/birdwatcher_usage_guides.md` | 原文（0 改动），排障工具使用指南，标题全部顶格 |
| `milvus_cli_reference.md` | `site/en/userGuide/tools/cli_commands.md` | 截取前 16672 字节（到 `## alter collection_properties` 节边界），108 行表格、29 个短节、36 段代码 |
| `milvus_vs_alternatives.md` | `site/en/about/comparison.md` | 原文（0 改动），3 张对比表 |
| `milvus_glossary.md` | `site/en/reference/glossary.md` | 原文（0 改动），51 条术语 |

官网同源链接：`https://milvus.io/docs/{overview,product_faq,birdwatcher_usage_guides,cli_commands,comparison,glossary}.md`

### datas/pdf/（5 个）

| 文件 | 类型 | 出处 | 说明 |
|---|---|---|---|
| `milvus_vectordb_paper.pdf` | 双栏论文 | SIGMOD 2021，Zilliz 官方镜像 `https://assets.zilliz.com/Milvus_A_Purpose_Built_Vector_Data_Management_System_0dca420d64.pdf` | 14 页，12 张表、8 张图，**跨页表格**；原样下载 |
| `hnsw_anns_paper.pdf` | 单栏论文 | arXiv `1603.09320` | 18 页，17 张表；原样下载 |
| `rag_failure_modes_paper.pdf` | 单栏论文 | arXiv `2401.05856`（Seven Failure Points When Engineering a RAG System） | 6 页；原样下载 |
| `milvus_hybrid_search_whitepaper.pdf` | 白皮书（自建排版） | 正文逐字引文：milvus.io v2.6.x 的 Rerankers Overview / RRF Ranker / Metric Types / Benchmark；数据表：Zilliz VectorDBBench 榜单 + Milvus 2.2 官方基准 | 6 页，单栏 + 目录 + 大表格；**仅排版，内容全部为官方原文与实测值** |
| `scanned_performance_faq.pdf` | **扫描件** | 内容为官方 `site/en/faq/performance_faq.md` 原文 | 渲染为位图后嵌入 PDF，**无任何文本层**（选不中文字），2 页；模拟打印后扫描 |

### datas/office/（7 个，全部由真实公开数据封装）

| 文件 | 数据/内容来源 | 关键特征 |
|---|---|---|
| `milvus_index_spec.docx` | milvus.io v2.6.x：In-memory Index / HNSW / IVF_FLAT / DISKANN / SCANN / IVF_RABITQ / Metric Types / 安装要求 | 多级标题；**3 张合并单元格表**（分类列纵向合并、「默认度量 COSINE」合并 4 行、IVF 示例值合并）；含官方基准表 |
| `milvus_deployment_requirements.docx` | Zilliz 官方中文博客《十倍成本优化，Milvus 2.5到2.6升级官方手把手教程》（zilliz.com.cn）；milvus.io 安装要求；官方架构文档 | **6 个 ≥200 字中文单元格**（最长 354 字），均为官方中文原文引文 |
| `vector_db_benchmark.xlsx` | **Zilliz VectorDBBench 榜单页**（`https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch`，页面内嵌真实测量结果） | 85 行 × 8 列；QPS 千分位 `#,##0.00`；召回率写成文本 `93.83%`；带空格子；含小数；含第二 sheet「数据来源」 |
| `milvus_spec_summary.xlsx` | 官方索引文档 / 度量文档 / 安装要求 / `release_notes.md` / `metrics_dashboard.md` | 5 个 sheet：索引类型总览、相似度度量、部署要求、版本与SDK（25 个版本真实对应关系）、监控面板指标（99 行） |
| `milvus_capability_matrix.xlsx` | 官方《官宣 Milvus 2.6 正式开源》中文公告（modb.pro/db/1934464950274109440）+ Zilliz 中文博客 | 30 行 × **12 列**；「说明」列为官方中文原文引文（每格 ≥50 字，最长 290 字） |
| `milvus_product_intro.pptx` | 官方文档 + VectorDBBench 实测 + 官方 release notes；SmartArt 版式为 Microsoft 官方 *Chevron Block Process* 版式（取自 BrandonStudio/SmartArtTemplates，源自 Microsoft Designer） | **14 页**；2 个**原生图表对象**（柱状图=各系统 QPS、饼图=10M 吞吐份额，均带数据标签）；**SmartArt 流程图**（4 个 diagram 部件）；**每页演讲备注**；**1 处组合形状**（3 个圆角矩形编组） |
| `milvus_architecture.pptx` | milvus.io v2.6.x — Architecture Overview（逐字引文） | 10 页；文字密集页 + 1 张组件职责表；每页演讲备注 |

## 三、与 SPEC 的偏差说明（3 处，均有理由）

1. **PDF 为 5 个**（SPEC 写 3~4 个）：因为 `papers` 源要求 2~3 篇论文，加上白皮书与扫描件后为 5 个；文件总数 18 仍落在 18~22 区间内。
2. **`milvus_cli_reference.md` / `milvus_vs_alternatives.md`** 分别承担 SPEC 中「changelog（30+ 行表格）」与「版本列表」的角色：官方 release notes 单文件 152 KB 过大，故改用同为官方来源、表格更密集的 CLI 命令参考（108 行表格）与版本对比文档；如需要「版本列表 + 30 行表格」的专门文件，可另行从 release notes 节选。
3. **`milvus_product_overview.md`** 未含代码块与表格（原文如此），但它承担「长文切片 + 标题层级」检验；代码块与表格样本由 `milvus_cli_reference.md`、`milvus_troubleshooting.md` 覆盖。
4. **docx / xlsx / pptx / 白皮书 / 扫描件**为脚本封装产物——但所有数字、引文均可按上表逐条核对到原始 URL（已确认用户同意此方式）。

## 四、建议的跨文档评测题（需综合 ≥2 个文档）

1. **「混合检索的融合原理是什么？Milvus 里怎么配置参数？实测吞吐如何？」** → RRF 公式（白皮书 §3 / md 术语表）＋ `milvus_product_intro.pptx` 第 8 页 ＋ `vector_db_benchmark.xlsx`。
2. **「HNSW 的 M / efConstruction / ef 分别控制什么？」** → `milvus_index_spec.docx` §4.1 ＋ `hnsw_anns_paper.pdf`。
3. **「10M 数据集上各系统的 QPS 与召回率分别是多少？ZillizCloud 排第几？」** → `vector_db_benchmark.xlsx`（过滤比例=0% 且 10M 的行）＋ 白皮书 §5.1。
4. **「Milvus 2.6 相比 2.5 简化了哪些组件？」** → `milvus_deployment_requirements.docx` 需求表 REQ-004 ＋ `milvus_capability_matrix.xlsx`「MixCoord 架构合并」行。
5. **「scan 扫描件里提到了哪些性能调优建议？」** → `scanned_performance_faq.pdf`（必须走 OCR 路径）。
6. **「BM25 支持哪些字段类型？默认度量是什么？」** → `milvus_glossary.md` ＋ `milvus_index_spec.docx` §5.1 合并单元格表 ＋ 白皮书 §4。
7. **「Streaming Node 的职责是什么？写入链路经过哪些组件？」** → `milvus_architecture.pptx` 第 4/5/7 页 ＋ `milvus_troubleshooting.md`。

## 五、自查结果（按 SPEC 第四章脚本校验，全部通过）

```
总数 18（md 6 / pdf 5 / office 7）；扫描件文本层=0；论文双栏+12 表；
docx 合并区 54 处、超长单元格 6 格(最长 354 字)；xlsx 85 行大表 + 真 % + 千分位 + 空格；
宽表 30 行 × 12 列（30/30 行中文长文本）；pptx 14 页/2 图表/4 SmartArt 部件/14 条备注/1 组合形状；
FAQ 40 条、术语 51 条、表格 108 行；5 个文件中英混排。
```
