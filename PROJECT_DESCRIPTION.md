# 项目架构说明

> 本文只描述**当前代码**：系统由什么组成、数据怎么流、关键设计为什么这么做。
> 迭代过程与实测证据见 [`ROADMAP.md`](ROADMAP.md)；对外简历口径见
> [`docs/RESUME_PROJECT.md`](docs/RESUME_PROJECT.md)；入口与快速开始见 [`README.md`](README.md)。

---

## 一、概览

### 1.1 它是什么

一个**面向技术资料的可溯源 RAG 问答系统**（对外定位：TechDocs Copilot）：

* 把 Markdown / PDF / Office 文档解析、切片、向量化，按语义域写入 **3 个 Milvus 集合**；
* 用户提问后，先做**问题路由**，再做**跨集合混合检索**（稠密 + BM25，两阶段 RRF）；
* 检索结果经过**文档相关性判分**，再生成带 `[n]` 引用编号的答案；
* 生成后做**确定性引用校验** + **幻觉判分** + **问题匹配度判分**，不合格则自纠错重试；
* 通过 FastAPI + SSE 提供同步/流式问答，支持集合级多租户隔离；
* 支持**多轮记忆**（最近 13 轮原文 + 摘要压缩 + 历史向量检索三层）与会话管理
  （`POST /sessions/new` 等 3 个端点，6 小时滑动 TTL）；
* 配套 `eval/` 评测、门禁、探针与回归入口，所有关键结论都有实测数字。

### 1.2 规模

| 项 | 当前值 |
|---|---|
| 业务代码 | **8,099 行 / 52 个 `.py`** |
| 评测与验证代码 | **12,961 行 / 90 个 `.py`** |
| 根目录 `scripts/` | **1,166 行 / 4 个 `.py`** |
| 语料 | **18 文件 / 454 块 / 3 集合**（6 Markdown + 5 PDF + 7 Office） |
| golden 集 | 32 题 |

> 454 是当前 Milvus 三个集合的 `row_count` 合计（216 + 160 + 78，已实测）；
> `eval/chunk_index.json` 的 496 是历史快照，不代表当前行数。

业务代码分布：

| 目录 | 行数 | 职责 |
|---|---:|---|
| `documents/` | 2,605 | 解析（含 **HTML/TXT 转换器**）、表格、建表、入库 |
| `graph2/` | 1,782 | CRAG 主工作流（含多轮记忆 + **按章节聚合**） |
| `tools/` | 1,374 | 混合检索、reranker、历史向量检索、**语义分块封装**、**按章节聚合** |
| `utils/` | 1,027 | 配置、日志、引用、轨迹、检查点、会话管理 |
| `api/` | 527 | FastAPI + SSE（含会话管理端点） |
| `graph/` | 311 | 简化 ReAct 工作流 |
| `scripts/`（包内） | 332 | 轨迹查看、检索试验 |
| `agent/` | 77 | 1.x 标准 Agent |
| `llm_models/` | 64 | LLM 与本地嵌入模型 |

### 1.3 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangChain 1.4.3 / LangGraph 1.2.14 / langchain-core 1.6.7 |
| 向量库 | Milvus 3.0.1（服务端）+ langchain-milvus 0.4.0 + pymilvus 3.0.2 |
| 嵌入 | `BAAI/bge-large-zh-v1.5`，**1024 维**，本地 CPU，L2 归一化 |
| 生成 | DeepSeek `deepseek-flash`（OpenAI 兼容协议） |
| 文档解析 | unstructured 0.27.16 / pymupdf4llm 1.28.2 / markitdown 0.1.8 |
| 结构化输出 | Pydantic v2 + `with_structured_output(method="function_calling")` |
| 服务化 | FastAPI 0.143.0 + uvicorn 0.54.0，SSE 流式 |
| 会话记忆 | `langgraph-checkpoint-sqlite` 3.1.1（`graph2` / `graph1` / `agent` 均支持，按 `thread_id` 隔离） |
| 精排（可选） | `sentence-transformers` CrossEncoder + `BAAI/bge-reranker-v2-m3`（本地，**默认关**） |
| 日志与追踪 | loguru + 自建 `utils/tracer.py` |
| 环境 | Python 3.12.7（Windows / Linux） |

---

## 二、目录结构与职责

```
RAG_PROJECT/
├── agent/
│   ├── __init__.py
│   └── rag_agent.py              create_agent + SqliteSaver 的标准 Agent
├── api/                          （无 __init__.py，按 PYTHONPATH 导入）
│   ├── main.py                   FastAPI：/ /health /ask /ask/stream /ingest
│   │                                      + /sessions/new /sessions /sessions/{id}
│   └── run.py                    服务启动脚本
├── documents/
│   ├── __init__.py
│   ├── markdown_parser.py        元素解析 + 标题合并 + 分级切片
│   ├── pdf_parser.py             PDF → Markdown（pymupdf4llm，可选 OCR）
│   ├── office_parser.py          docx / xlsx → Markdown；pptx 优先 PPTXParser
│   ├── pptx_parser.py            PPTX 逐页结构化提取（备注/SmartArt/图表）
│   ├── table_handler.py          表格结构恢复 + 行切 + 表头注入
│   ├── milvus_db.py              集合 schema、索引、写入、维度自检
│   └── write_milvus.py           多源多进程入库管线
├── graph/
│   ├── __init__.py
│   ├── graph1.py                 简化 ReAct 工作流（4 显式节点 + 2 条件边）
│   ├── graph_state1.py
│   ├── agent_node.py
│   ├── generate_node.py
│   ├── rewrite_node.py
│   ├── get_human_message.py
│   └── graph_rag1-2.png          工作流示意图
├── graph2/
│   ├── __init__.py
│   ├── graph_2.py                CRAG 主工作流 + 熔断 + run_with_trace
│   ├── graph_state2.py           GraphState + 判分用 Pydantic 模型
│   ├── query_route_chain.py      问题路由（查库 / 联网）
│   ├── retriever_node.py         跨集合检索节点
│   ├── web_search_node.py        Tavily 检索节点
│   ├── grade_documents_node.py   文档相关性判分（支持批量）
│   ├── transform_query_node.py   问题改写
│   ├── generate_node2.py         生成 + 引用 + 同表切片合并
│   ├── grader_chain.py           文档相关性判分器
│   ├── grade_hallucinations_chain.py
│   └── grade_answer_chain.py
├── llm_models/
│   ├── __init__.py
│   ├── all_llm.py                 ChatOpenAI（配置驱动）+ Tavily 工具
│   └── embeddings_model.py        本地 BGE 嵌入
├── tools/
│   ├── __init__.py
│   ├── retriever_tools.py         跨集合混合检索 + 编号上下文 + 租户作用域
│   ├── reranker_tools.py          reranker 精排 + 分数门槛 + 聚合类问题识别
│   └── history_store.py           会话历史的向量存储与检索（独立集合，按 session 隔离）
├── utils/
│   ├── __init__.py
│   ├── env_utils.py               主要配置与 DOC_SOURCES / TENANTS
│   ├── citation.py                引用解析、校验、拒答识别
│   ├── tracer.py                  节点/边级轨迹与 token 统计
│   ├── checkpointer.py            SqliteSaver 工厂
│   ├── session_store.py           会话元数据表 + 6h 滑动 TTL + 后台清理
│   ├── log_utils.py
│   └── print_utils.py
├── scripts/                      （无 __init__.py）
│   ├── show_trace.py              查看 logs/trace.jsonl
│   └── try_search.py
├── datas/
│   ├── CORPUS_MANIFEST.md         语料来源与清洗记录
│   ├── md/                        6 个 Markdown 语料
│   ├── pdf/                       5 个 PDF 语料
│   └── office/                    7 个 Office 语料
├── models/                       本地嵌入模型（gitignored，可重建）
├── logs/                         运行轨迹（gitignored）
├── graph_rag1.png                工作流示意图
└── graph_rag2.png
```

> `__init__.py` 为 0 字节；`api/` 与 `scripts/` 没有该文件（按 `PYTHONPATH=RAG_PROJECT`
> 以命名空间包导入）。`.idea/`、`__pycache__/`、`.venv/` 等本地/生成物未列出。

工作区根另有：`eval/`（评测与回归）、`scripts/`（**端到端冒烟** / 模型下载 / Milvus 探测 / 迁移验证）、
`docs/`、`scratch/`（临时验证，未入库）、`.venv/`、`requirements*.txt`、`.env.example`、
`pyrightconfig.json`、本地保留的迁移前快照（不随公开仓库发布）。

---

## 三、数据流

### 3.1 入库链路

```
DOC_SOURCES（utils/env_utils.py，一源一集合）
├─ milvus_docs : datas/md/*.md        → MarkdownParser → kb_milvus_bge_large_1024
├─ papers      : datas/pdf/*.pdf      → PDFParser      → kb_papers_bge_large_1024
└─ manuals     : datas/office/*       → OfficeParser   → kb_manuals_bge_large_1024
   （notes/.txt 源默认关闭：需先人工整理为 Markdown）
        │
        ▼
① 解析：Unstructured elements / pymupdf4llm / markitdown / PPTX 提取
② 标题合并：merge_title_content() → “A -> B -> 正文”面包屑
③ 表格处理：text_as_html 恢复结构 → 按行切分 + 每块重复表头
④ 短块合并：< 50 字符碎片并入同文件相邻正文块
⑤ 语义切分：仅对 > 5000 字符的块用 SemanticChunker 二次切分
⑥ 字段兜底：category / source / filename / filetype / title / category_depth
⑦ 嵌入：bge-large-zh-v1.5 → 1024 维归一化向量
⑧ 写入：按源写入各自集合；失败显式记录，不静默丢弃
```

**集合 schema（10 个字段）**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 | 自增主键 |
| `text` | VARCHAR(6000) | 块正文；`enable_analyzer=true` + jieba |
| `dense` | FLOAT_VECTOR(1024) | HNSW（M=16 / efConstruction=64），IP 度量 |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25（k1=1.2 / b=0.75），倒排索引 |
| `category` | VARCHAR(1000) | `content` 为正文块，其余为标题/噪声 |
| `title` | VARCHAR(1000) | 标题面包屑 |
| `category_depth` | INT64 | 元素嵌套层级 |
| `source` / `filename` / `filetype` | VARCHAR(1000) | 来源信息 |

**两个关键约束**：

* Milvus 的 `text` 字段按**字节**计（`VARCHAR(6000)`），中文约 3 字节/字；
  切块按字节判定并留余量，否则整批写入会被拒；
* 集合名只允许字母/数字/下划线；Milvus Lite 比真实服务宽松，
  因此客户端在建表前统一校验，避免“Lite 通过、真实服务失败”。

入库管线是**解析与写入分离 + 有界队列背压**；真实 Milvus 走多进程，
Milvus Lite 因文件锁自动降级为单进程。

### 3.2 查询链路（`graph2`，CRAG 完整版）

```
用户问题
  │
  ├─ route_question（闸门①：查库 or 联网）
  │
  ├─ vectorstore 路径                        web_search 路径
  │     │                                      │
  │     ▼                                      ▼
  │  retrieve（跨集合混合检索）             Tavily 搜索
  │     │   · 聚合类查询走深检索（per_k=20 / final_k=60）
  │     │   · reranker 精排 + 分数门槛（RERANK_ENABLED=1 时，默认不做）
  │     ▼                                      │
  │  grade_documents（闸门②：逐块相关性）      │
  │     ├─ 有相关块 ───────────────────────┐   │
  │     └─ 全不相关                          │ │
  │          ├─ transform_count < 2 → transform_query（改写问题）
  │          └─ transform_count ≥ 2 → web_search
  │                                          ▼   ▼
  │                                       generate
  │                              · 同表切片合并（n005 类跨行统计）
  │                              · 上下文编号 [1]..[n]
  │                              · **三层历史注入**（仅 session_id 模式）：
  │                                  最近 13 轮 + 摘要 + 历史向量检索 top-3
  │                              · 强约束引用 prompt
  │                                          │
  │                                          ▼
  │                              grade_generation（闸门③④）
  │                              · 先做确定性引用校验
  │                              · 拒答 → useful（不重试）
  │                              · 越界引用 → not supported（重试）
  │                              · 幻觉 + 匹配度并发判分
  │                                          │
  │                        useful → END      │
  │                        not supported → generate（受熔断）
  │                        not useful → transform_query
  └──────────────────────────────────────────┘
```

**拓扑**：5 个显式节点（`retrieve` / `web_search` / `grade_documents` /
`transform_query` / `generate`）+ `START` / `END`；3 处条件边。

**四道闸门**：

| 闸门 | 位置 | 判断什么 | 不通过时 |
|---|---|---|---|
| ① 问题路由 | `route_question` | 该查库还是该联网 | 走另一条通道 |
| ② 文档相关性 | `grade_documents` | 每个召回块是否切题 | 过滤掉；全空则改写或转联网 |
| ③ 幻觉检测 | `grade_generation` | 答案是否基于检索块 | 重新生成（受熔断） |
| ④ 问题匹配度 | `grade_generation` | 答案是否真的回答问题 | 改写问题重试 |

**生成层细节**：

* `CITE_PROMPT` 要求每个事实句标注 `[n]`，只允许使用给定资料；
* 资料不足时固定回复 `知识库中未找到依据。`，由 `is_refusal()` 精确识别；
* `merge_table_slices()` 在构建上下文前，把**同一张表**的多切片拼回完整表
  （按 `(filename, title)` 分组、按行号升序、表头去重、超过 20,000 字符放弃）；
* 引用校验先于 LLM 判分：越界编号是“幻觉引用”，拒答是“正确行为”，
  复述原文自带的参考文献编号不算幻觉（P0-3）。

### 3.3 查询链路（`graph1` / `rag_agent`）

| 入口 | 构建方式 | 拓扑 | 记忆 | 特点 |
|---|---|---|---|---|
| `graph2` | 手写 `StateGraph` | 5 显式节点 + 3 条件边 | `SqliteSaver`（`get_memory_graph()`，**三层记忆**） | 路由 + 联网 + 四道闸门 + 熔断 |
| `graph1` | 手写 `StateGraph` | 4 个显式节点（含 START/END 共 6 个）+ 2 处条件边（7 条分支） | SqliteSaver | ReAct 循环 + 相关性 + 改写 |
| `rag_agent` | `create_agent()` | `model` / `tools` | SqliteSaver | LangChain 1.x 标准写法 |

三者共用同一个 `retriever` / `retriever_tool`，检索配置完全一致。

### 3.4 HTTP 服务

| 端点 | 说明 |
|---|---|
| `GET /` | 服务元信息：版本、文档入口、端点清单 |
| `GET /health` | 分别探测 Milvus、检索器、LLM、租户配置，不把失败糊成一个 `false` |
| `POST /ask` | 同步问答：答案 + 引用溯源；`with_trace=true` 时在响应里附带轨迹，但**不落盘** |
| `POST /ask/stream` | SSE 流式：`event: step` 逐节点进度，`event: answer` 最终答案，异常时 `event: error` |
| `POST /ingest` | 按 `DOC_SOURCES` 源名入库；`reset=true` 才清空集合，默认增量追加 |
| `POST /sessions/new` | **新开对话**：服务端生成 `session_id`（UUID）并建档，返回 sid 与 ttl |
| `GET /sessions` | 会话列表，按 `last_active` 倒序 |
| `DELETE /sessions/{id}` | 删会话：历史向量块 + LangGraph 检查点 + 元数据，三处一起清 |

* 多租户：`X-Tenant-Id` 请求头；配置了 `KB_TENANTS` 时缺失返回 400，未知租户返回 403；
* 未配置 `KB_TENANTS` 时忽略该头（单租户模式，可访问全部集合）；
* SSE 的生成器在另一个上下文执行，租户切换用 `set + finally`，不能用 `ContextVar.reset(token)`；
* **只有 `/ask/stream` 会把轨迹写入 `logs/trace.jsonl`**；`/ask` 只用 `run_with_trace(save=False)`；
* `/ingest` 对未知源、`enabled=False` 的源返回 400；入库异常返回 500，不返回“假成功”。

---

## 四、关键设计

### 4.1 混合检索与两阶段 RRF

**四种检索原语**：

| # | 方式 | 本项目用法 |
|---|---|---|
| 1 | 稠密向量（HNSW / IP） | 混合检索的一路 |
| 2 | 稀疏向量（BM25 / 倒排） | 混合检索的另一路 |
| 3 | 混合检索 + RRF 融合 | 集合内主路径 |
| 4 | 标量过滤（`expr`） | 叠加在 1/2/3 之上 |

**两阶段 RRF**：

```
阶段一（集合内）：每个集合各做 hybrid_search
    dense(bge-large 1024) + sparse(BM25) --RRF(k=100)--> 集合 top-12

阶段二（跨集合）：把各集合结果按块内容去重后再次 RRF
    score = Σ 1/(rrf_k + rank)   （rrf_k=60）
    --> 最终 top-36
```

参数：

| 参数 | 值 | 说明 |
|---|---:|---|
| `FETCH_K_PER_ROUTE` | 24 | 每集合、每路预取条数（必须 ≥ 最终配额） |
| `PER_COLLECTION_K` | 12 | 每集合最终保留条数 |
| `FINAL_K` | 36 | 跨集合融合后交给下游的条数 |
| `RRF_K` | 60 | 跨集合二次 RRF 的平滑常数 |
| `expr` | `category == "content"` | 排除纯标题块 |

**为什么用 RRF 而不是分数相加**：IP 内积与 BM25 量纲不同，无法直接比较；
RRF 只看排名，天然规避量纲问题。

**查询侧的两条路径**：dense 路由客户端 `embed_query()` 编码为 1024 维；
sparse 路把**原始 query** 交给 Milvus 服务端，用与入库相同的 jieba 分析器算 BM25。
两端分词口径必须一致，否则关键词路会失效。

**reranker 的定位**（ROADMAP 4.42）：bge-reranker-v2-m3 在本项目 8 题上实测**增益为 0**
（基线已拿到大部分可召回块，天花板有限），因此**默认关闭**；
但 `tools/reranker_tools.py` 已提供开关（`RERANK_ENABLED=1`）+ reranker 分数门槛，
供**加大检索深度**或**更高精度要求**的企业级场景启用 —— 届时由它负责把粗排的多余候选收敛掉。

**HNSW vs FLAT**：FLAT 暴力遍历、召回 100% 但线性变慢；HNSW 用 `M` /
`efConstruction` 换速度，适合百万级向量 + 毫秒延迟的企业知识库场景。

### 4.2 中文 BM25

BM25 依赖分词。中文不分词时，“光刻”与“光刻机”会被当成无关词。
本项目在 **Milvus 服务端**启用 jieba：

```python
analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]}
```

**远端 Milvus** 需在服务端安装 jieba（官方镜像已内置，自定义镜像需自行安装）；
**Milvus Lite** 的分词在客户端进程内执行，因此 `requirements.txt` 把 `jieba` 钉为客户端依赖
（缺失时建表即报 `JiebaAnalyzer requires the 'jieba' package`）。

### 4.3 解析、切片与表格

| 阶段 | 做什么 | 为什么 |
|---|---|---|
| 元素解析 | `mode="elements"`，保留 `category` / `element_id` / `parent_id` | 标题与正文的父子关系是后续合并的依据 |
| PPTX 逐页 | 优先 `PPTXParser`（逐页一块，提取备注/SmartArt/图表/组合形状）；失败回退 markitdown | markitdown 会把多页并成 1 块且丢结构 |
| 标题合并 | 子元素拼进父标题，`category` 改为 `content` | 检索时带层级语境，且能用 `expr` 排除纯标题块 |
| 表格恢复 | 从 `text_as_html` 还原 Markdown 表格，加“表格：/列名：”前缀 | Unstructured 会吞掉 `|` 分隔符 |
| 表格切分 | **行数与字节双重闸门**：两者都未超限才整表一块，任一超限即按行切（按字节上限收缩），每块重复完整表头 + 行范围 | 行边界天然对齐记录，避免一条记录跨块 |
| **表格孪生去重** | `is_flat_twin()` 三维判据（前向 0.98 / 反向 0.95 / 长度比 0.5~2.0），在标题合并阶段剔除与结构化表格块重复的扁平文本 | 避免同一张表以「表格版 + 压平版」两份进入检索 |
| 短块合并 | < 50 字符碎片并入同文件相邻正文块 | 超短块会挤占检索配额 |
| 语义切分 | 仅 > 5000 字符的块走 `SemanticChunker` | 语义切分要调 embedding，成本高，只对大块用 |
| 同表合并 | 生成前把同表碎片拼回完整表 | 跨行统计类问题（如“最低值”）需要看到整表 |

### 4.4 引用与拒答

`utils/citation.py` 提供四件事：

1. `format_context()`：把文档编号成 `[1] ... [2] ...`，编号与下标严格对应；
2. `parse_citations()`：从答案提取编号（保序去重）；
3. `validate_citations()`：区分越界引用、原文自带编号、设计内拒答；
4. `is_refusal()` / `build_citation_meta()`：拒答识别与溯源映射。

拒答文案固定为 `知识库中未找到依据。`——固定字符串让下游能精确区分
“正确拒答”与“幻觉”，拒答不触发熔断重试。

### 4.5 四道闸门与批量判分

四个判分器都用“LLM + Pydantic 结构化输出”把主观判断转成可路由结论：

| 判分器 | Schema | 判断什么 |
|---|---|---|
| `query_route_chain` | `RouteQuery.datasource` | 查库 / 联网 |
| `grader_chain` | `GradeDocuments.binary_score` | 单个文档是否切题 |
| `grade_hallucinations_chain` | `GradeHallucinations.binary_score` | 答案是否基于文档 |
| `grade_answer_chain` | `GradeAnswer.binary_score` | 答案是否回答问题 |

**批量判分**：`grade_documents` 默认开启 `USE_BATCH_GRADE=True`，
一次调用判 `BATCH_SIZE=8` 块，批间最多 8 并发。实测：

| 指标 | 逐块 | 批量 |
|---|---:|---:|
| 判分 token | 173,265 | **104,378（−40%）** |
| 判分调用 | 215 | **30（−86%）** |
| 端到端耗时 | 8.1s | **3.8 ~ 5.4s** |
| 判定一致率 | — | **91.6%（197/215，18 个分歧）** |

> 产物：`eval/measure_grade_batch_saving.json`。ROADMAP 4.23.3 另一次记录为
> 199/215（92.6%），两次差 2 块，属 LLM 判定抖动；golden 块的逐块核对见该节。

### 4.6 可观测性

`utils/tracer.py` 用 LangChain callback 收集每次运行的：

* 节点耗时与决策（路由结果、检索块数、判分保留数、引用校验结果）；
* token 用量（按模型、输入/输出、缓存命中）；
* 最终答案预览。

轨迹追加写入 `logs/trace.jsonl`（`graph2` / HTTP 路径；`graph1` 与 `rag_agent` 不接 tracer），
用 `scripts/show_trace.py` 查看。`graph2` 的所有节点与条件边都经过 tracer 包装，
**只加观测、不改行为**。

### 4.7 多租户

`KB_TENANTS` 用 JSON 配置“租户 → 允许访问的源名列表”：

```powershell
$env:KB_TENANTS='{"acme":["milvus_docs","manuals"],"beta":["papers"]}'
```

* 隔离在**集合级**：租户只能检索授权源对应的集合；
* 不加 schema 字段、不重建集合，零迁移成本；
* API 层在进入检索前校验，越权快速失败；
* 只做授权隔离，**不含身份认证**。

### 4.8 会话记忆

**`graph2` 自 ROADMAP 4.41 起已支持多轮记忆**，与 `graph1` / `rag_agent` 并列，
三者都按 `thread_id` 隔离会话；区别在**注入 prompt 的层数**：

| 入口 | 持久化 | 注入 prompt 的历史 |
|---|---|---|
| `graph2`（生产） | `SqliteSaver`（`get_memory_graph()` 惰性单例）| **三层**：最近 13 轮原文 + 摘要 + 历史向量检索 top-3 |
| `graph1` / `rag_agent` | `SqliteSaver` | 无（state 里有 `messages`，但 prompt 不注入）|

**三层记忆的分工**（ROADMAP 4.43 / 4.44）：

```
最近 13 轮原文   近期指代（「它」「那个」）      有界：MAX_HISTORY_CHARS=6000
摘要（`state["summary"]`）  主题脉络（做过什么）        有界：500 字；超 26 轮才生成，只生成一次
历史向量检索      **含具体数值的早期轮次**        kb_session_history_* 集合，按 session_id 隔离
```

第三层是前两层替代不了的：第 3 轮问「TurboPuffer 的 QPS」答 184.54，
第 30 轮问「刚才那个 QPS 对应的产品」—— 前两层都答不了。

**触发条件**：只有 `run_with_trace(thread_id=...)` / `/ask` 带 `session_id` 才走记忆路径；
单轮调用（不传）行为与改造前**完全一致**。

**会话管理**（ROADMAP 4.45）：`sessions` 表（复用 `checkpoints.db`）记录
`id / created_at / last_active / title`，支撑 `POST /sessions/new`、
`GET /sessions`、`DELETE /sessions/{id}` 三个端点；
`DELETE` 会清三处——历史向量块 + `checkpoints`/`writes` 行 + 元数据行。

**TTL 是滑动的**（`now - last_active > 6h`），不是创建时间：
否则持续 6.5 小时的会话会被误删。后台 janitor 线程每 30 分钟清一次。

`SqliteSaver.setup()` 不需要手动调用，首次读写时自动建表；
真正上生产应换 `PostgresSaver`。

### 4.9 熔断与重试

* `MAX_GENERATION_RETRY = 2`：`not supported → generate` 是无条件自环，
  达到上限后改走 `not useful`（受 `transform_count` 保护），循环有界；
* `transform_count ≥ 2`：全不相关时不再改写，转 `web_search` 降级；
* 判分异常按保守方向处理（无法确认则走 `not useful`），不抛异常中断整次问答。

### 4.10 配置

配置统一从 `.env` 读取，但**分散在多个模块**（不是在 `env_utils.py` 一处）：

| 变量组 | 实际读取位置 |
|---|---|
| LLM / Milvus / 嵌入 / `DOC_SOURCES` / `KB_TENANTS` / `CHECKPOINT_DB` | `utils/env_utils.py` |
| `PDF_OCR_FALLBACK` / `PDF_USE_OCR` | `documents/write_milvus.py` |
| `TRACE_DIR` | `utils/tracer.py` |
| `RERANK_*`（开关 / top_k / 阈值 / 模型 / 截断） | `tools/reranker_tools.py` |
| `PER_COLLECTION_K_DEEP` / `FINAL_K_DEEP` | `tools/retriever_tools.py` |
| `HISTORY_*`（开关 / 集合名 / top_k / 上限 / 保留月数） | `tools/history_store.py` |
| `MAX_HISTORY_TURNS` / `MAX_HISTORY_CHARS` / `MAX_PER_MSG_CHARS` | `graph2/generate_node2.py` |
| `SESSION_TTL_SECONDS` / `SESSION_JANITOR_INTERVAL` | `utils/session_store.py` |

| 变量 | 默认 / 推荐 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | **必填** | OpenAI 兼容端点的 key（DeepSeek 也用它） |
| `TAVILY_API_KEY` | 可选 | `web_search` 分支需要 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 地址 |
| `MILVUS_LITE_PATH` | 空 | 设置后走 Milvus Lite（单进程） |
| `EMBEDDING_MODEL` | 项目内 `models/bge-large-zh-v1.5` | 缺失时退回 HF repo id |
| `EMBEDDING_DIM` | `1024` | 必须与模型输出一致 |
| `COLLECTION_NAME` | `kb_milvus_bge_large_1024` | 默认集合（兼容旧脚本） |
| `KB_COLLECTION_MD/PDF/OFFICE` | 见 `DOC_SOURCES` | 每源集合名 |
| `SRC_DIR_MD/PDF/OFFICE` | `datas/md` 等 | 每源目录 |
| `LLM_BASE_URL` | 代码默认 `https://xiaoai.plus/v1`；DeepSeek 填 `https://api.deepseek.com` | OpenAI 兼容端点 |
| `LLM_MODEL` | 代码默认 `gpt-4o-mini`；DeepSeek 填 `deepseek-flash` | 模型名 |
| `LLM_TEMPERATURE` | `0` | 生成温度 |
| `LLM_STRUCTURED_METHOD` | `function_calling` | DeepSeek 不支持 `json_schema` |
| `LLM_DISABLE_THINKING` | `1` | thinking 与强制 `tool_choice` 冲突 |
| `MILVUS_RERANKER_STYLE` | `function` | `function` 需 Milvus ≥ 2.6；Lite 用 `legacy` |
| `KB_TENANTS` | 空 | 多租户 JSON；空 = 单租户 |
| `CHECKPOINT_DB` | `checkpoints.db` | 会话记忆文件 |
| `PDF_OCR_FALLBACK` | `0` | 扫描件走带页码 OCR（在 `write_milvus.py` 读取） |
| `PDF_USE_OCR` | `1` | pymupdf4llm 图片 OCR 开关（在 `write_milvus.py` 读取） |
| `RERANK_ENABLED` | `0` | reranker 精排开关；`1` 才启用（本项目天花板仅 9%，故默认关） |
| `RERANK_TOP_K` | `20` | 精排后保留几块 |
| `RERANK_SCORE_THRESHOLD` | `0` | reranker 分数门槛；`0`=不过滤，`>0` 时低于阈值的块不送 LLM 判分 |
| `RERANK_MODEL` | 空 | 空 = 项目内 `models/bge-reranker-v2-m3`（本地） |
| `RERANK_MAX_CHARS` | `2000` | 单块送去精排的截断长度 |
| `PER_COLLECTION_K_DEEP` | `20` | 聚合类查询的每集合召回数（默认 12） |
| `FINAL_K_DEEP` | `60` | 聚合类查询的最终块数（默认 36） |
| `MAX_HISTORY_TURNS` | `13` | 注入 prompt 的最近轮数 |
| `MAX_HISTORY_CHARS` | `6000` | 历史**总预算**；超出从最老的丢起，保住最近 |
| `MAX_PER_MSG_CHARS` | `500` | 单条历史上限（防御性，已不是主约束） |
| `HISTORY_ENABLED` | `1` | 历史向量检索开关（三层记忆第三层） |
| `HISTORY_COLLECTION` | `kb_session_history_bge_large_1024` | 历史独立集合（**不进** `DOC_SOURCES`） |
| `HISTORY_TOP_K` | `3` | 每次召回几轮历史 |
| `HISTORY_MAX_CHARS` | `1500` | 单轮 text 上限（schema VARCHAR 6000，留余量） |
| `HISTORY_KEEP_MONTHS` | `3` | 历史保留月数（按月桶 TTL） |
| `SESSION_TTL_SECONDS` | `21600` | 会话 6 小时**不活跃**才清（滑动，非创建时间） |
| `SESSION_JANITOR_INTERVAL` | `1800` | 后台清理线程轮询间隔（30 分钟） |

> 另有 `DATAS_DIR` / `MODELS_DIR` / `SRC_DIR_TXT` / `KB_COLLECTION_TXT` 等路径配置，
> 见 `env_utils.py` 的 `DOC_SOURCES`；`TRACE_DIR` 见 `utils/tracer.py`。

---

## 五、运行方式

### 5.1 准备

```powershell
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env      # 填 key、Milvus 地址、LLM 端点
.venv\Scripts\python.exe scripts\fetch_models.py
```

### 5.2 入库

```powershell
# 按 DOC_SOURCES 逐源入库（真实 Milvus 多进程；Lite 自动单进程）
.venv\Scripts\python.exe -m RAG_PROJECT.documents.write_milvus
```

> ⚠️ CLI 默认 `recreate=True`：**每个源先 `create_collection()`（会 drop 重建、清空集合）**。
> 增量追加必须加 `--no-recreate`，否则是「全量重灌」。

### 5.3 三个 CLI 入口

```powershell
.venv\Scripts\python.exe -m RAG_PROJECT.graph2.graph_2      # CRAG 完整链路
.venv\Scripts\python.exe -m RAG_PROJECT.graph.graph1        # 简化 ReAct
.venv\Scripts\python.exe -m RAG_PROJECT.agent.rag_agent      # 1.x 标准 Agent
```

### 5.4 HTTP 服务

```powershell
.venv\Scripts\python.exe RAG_PROJECT\api\run.py
# 或
$env:PYTHONPATH="RAG_PROJECT"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5.5 评测

```powershell
.venv\Scripts\python.exe eval\run_eval.py                 # 仅检索层（秒级）
.venv\Scripts\python.exe eval\run_eval.py --with-generation
.venv\Scripts\python.exe eval\run_all_checks.py --full    # 全量回归
```

---

## 六、评测与门禁

### 6.1 指标分层

| 层 | 指标 | 计算方式 | 抖动 |
|---|---|---|---|
| 检索层 | `context_recall` / `hit_rate` / `context_precision` | 块 ID 集合运算 | **0** |
| 生成层 | `faithfulness` / `answer_relevancy` | LLM 判分 + 判定缓存 | 受控 |

`context_precision` 天然偏低，且随配额放大必然更低，**只作硬门禁 / 监控，
不参与告警**。

### 6.2 门禁阈值

| 指标 | FAIL | WARN |
|---|---:|---:|
| `context_recall` | < 0.60 | < 0.80 |
| `hit_rate` | < 0.75 | < 0.85 |
| `context_precision` | < 0.03 | 不告警 |
| `faithfulness` | < 0.60 | < 0.80 |
| `answer_relevancy` | < 0.60 | < 0.80 |

LLM 不可用时返回 `INCONCLUSIVE`（退出码 3），**不得判为质量回退**。

### 6.3 当前结果

| 指标 | 值 |
|---|---:|
| `context_recall` | **0.8664** |
| `hit_rate` | **0.9375** |
| `context_precision` | 0.0477（仅监控） |
| `faithfulness` | **1.0000** |
| `answer_relevancy` | **0.8750** |
| 门禁判定 | **PASS** |

### 6.4 回归与并发

* `run_all_checks.py --full`：**20/20 判定通过**（含 API+SSE、多租户联动）；
* 并发实测（1/2/4/8）：墙钟 **4.4 / 4.3 / 5.9 / 8.1s**，全部成功（1/1、2/2、4/4、8/8）；
  8 较 4 为 **+37%**，说明开始排队；
* 容量建议：**生产并发上限取 4~6**；更高并发与 HTTP 层压测未做。

---

## 七、边界与关键坑

### 7.1 能力边界

| 项 | 现状 |
|---|---|
| reranker | **质量层面**：当前语料实测增益 0（天花板仅 9%），故**默认关闭**；**工程层面**：已提供开关 `RERANK_ENABLED=1` + 分数门槛，为大数据量/深检索场景预留（ROADMAP 4.42） |
| 表格父子块 | 对查询类无增益；跨行统计改用「同表切片合并」（`merge_table_slices`，生成前拼回完整表，**标题块不再阻断合并**）|
| **单表内聚合** | **已解决** —— `tools/table_query.py` 用代码算 `min/max/mean/sum/count`，结果作为事实注入。实测 **0/3 → 3/3**（仅加 prompt 引导是 1/3）|
| **跨表 / 跨文档聚合** | **仍未做** —— 需跨表联合 + 多列条件筛选；`table_query` 只处理单块单列，完整解法是 text-to-SQL |
| **支持的文档格式** | ✅ **`.md` / `.pdf` / `.docx` / `.xlsx` / `.pptx` / `.html` / `.txt`**（后两者经 `plain_converters` 转 Markdown 后复用同一管道）<br>❌ **`.doc` / `.xls`（旧格式）、`.csv`、图片**（无独立入库路径、无图像向量）|
| **HTML/TXT 的启用方式** | 解析器已实现，但 `DOC_SOURCES` 的 `web` / `notes` 源**默认 `enabled=False`**（目录为空时开启会让检索连到不存在的集合）。接入语料后改 `True` |
| **TXT 的固有局限** | 无结构，靠**文件名当标题**；若文件名无意义（如「新建文本文档」），切片质量会差 |
| **分块策略** | 入库仍是 `SemanticChunker(percentile)`（**未换**）；已加封装降级 + 可选「按章节聚合」（`SECTION_MERGE_ENABLED`，默认关）。换策略需重建索引 + 重标 golden，**尚未做** |
| 多轮记忆 | 三层（13 轮原文 + 摘要 + 历史向量检索）；**6 小时不活跃即清**，跨天不继承；单块历史上限 1500 字 |
| 历史集合 TTL | 按**月桶**删（`HISTORY_KEEP_MONTHS`）；要按小时/天精确删需给 schema 加 INT64 字段，未做 |
| 会话清理触发 | 后台 janitor 每 30 分钟一次；**无手动触发端点** |
| 并发 | 4 安全；8 劣化；16/32 未测 |
| HTTP 层并发 | 功能已验证，未做压测 |
| 扫描件 OCR | 默认关闭；开启后约 2.5s/页 |
| 多租户 | 集合级授权，不含身份认证 |
| `score_threshold` | `search_type="similarity"` 下不被消费，属惰性参数（可靠门槛用 reranker 分数，见 4.42） |

### 7.2 关键坑（均已处理）

1. **结构化输出必须显式 `function_calling`**：`langchain-openai` 1.x 默认改成
   `json_schema`，DeepSeek 不支持；项目内 **6 处**调用点全部显式指定。
2. **config 必须透传**：线程池 / 条件边包装器不继承 contextvar，
   不透传 `config` 会导致 LLM 调用的 token **静默归零**。
3. **集合名不能含连字符**：Milvus Lite 能接受，真实服务报 `code=1100`；
   客户端已前置校验。
4. **向量维度不可变**：换模型必须重建集合或换集合名；写入前做维度自检。
5. **`add_documents` 只写 WAL**：写完立即查可能 `row_count=0`，需显式 `flush`。
6. **SSE 不能用 `ContextVar.reset(token)`**：生成器跨上下文执行会抛
   `Token was created in a different Context`；用 `set + finally` 归零。
7. **`filter` 不生效，必须用 `expr`**：`langchain-milvus` 的 hybrid 路径只认 `expr`。
8. **Milvus Lite 只适合验证逻辑**：不支持 Function 形式 reranker，
   集合名校验更宽松，且文件锁只允许单进程。
9. **`create_collection()` 会 drop 重建**：不能放在「每次写入前」调用，
   否则每写一轮就把整个集合清空（ROADMAP 4.44 实测：4 轮写入成功却只剩 1 块）。
   正确做法是先 `list_collections()` 判断，已存在就什么都不做。
10. **`stream()` 产出的是节点增量，不是完整 state**：带 reducer 的字段
    （如 `messages`）在图内累积到 4 条，但单次节点增量只有 2 条；
    用 `dict.update()` 拼会把累积结果覆盖掉。记忆模式下应读
    `graph.get_state(config).values`（ROADMAP 4.41）。
11. **LangGraph 会把 TypedDict 里声明的 `messages` 初始化为 `[]`**：
    所以「是否处于记忆模式」不能用 `state.get("messages") is not None` 判断
    （恒为真），要用一个只有记忆模式才注入的字段（如 `session_id`）。
12. **`history_store` 必须走项目的 `add_documents` 包装**：直接调
    `vector_store_saved.add_documents` 只写 WAL 不 flush，
    表现为「写入成功但 `row_count=0`、检索不到」（ROADMAP 4.44 bug ①）。

---

## 八、文档

| 文档 | 用途 |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | 迭代日志与实测证据（按时间追加，**主文档**） |
| [`CHANGELOG.md`](CHANGELOG.md) | 对外时间线与当前状态 |
| [`docs/RESUME_PROJECT.md`](docs/RESUME_PROJECT.md) | 简历 / 面试口径 |
| [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md) | 面试问答（含如何体现独立能力） |
| [`docs/archive/HISTORY.md`](docs/archive/HISTORY.md) | LangChain 1.x 迁移历史（已完成） |
