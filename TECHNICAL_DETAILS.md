# RAG 企业知识库项目 · 技术实现详解

> ## ⚠️ 时效提示（ 核对）
>
> **本文最后更新（阶段一之前）**，以下内容已过时：
>
> | 本文中的说法 | 当前实际 | 权威出处 |
> |---|---|---|
> | 嵌入 `bge-small-zh-v1.5`（512 维） | **`bge-large-zh-v1.5`（1024 维）** | ROADMAP 阶段一 |
> | 语料 7 个 Markdown / 约 102 块 | **18 文件 / 523 块 / 3 集合** | ROADMAP 4.7 |
> | `graph2` 的「7 节点 / 9 边」 | 已增 `web_search`、判分并发等改动 | ROADMAP 4.25 |
> | 无评测体系 | **33 脚本 / 10,113 行** | ROADMAP 4.36 |
>
> **仍然成立的部分**（推理与机制类）：为什么 HNSW 而非 FLAT、为什么用 `expr` 而非
> `filter`、`score_threshold` 为何无效、RRF 为何不直接相加分数、分级切片策略、
> jieba 中文分词的必要性 —— 这些**机制层面的解释没有变**。
>
> **→ 建议用法：当「概念词典」查机制；所有数字以
> [`ROADMAP.md`](ROADMAP.md) 为准。** 见 [`docs/DOC_INDEX.md`](docs/DOC_INDEX.md)。

> 本文按 9 个维度结构化梳理项目的技术实现，全部结论基于源码阅读 + 真实环境实测
> （Milvus 3.0.1 + DeepSeek + BGE-small-zh + Tavily；依赖已于  全量更新至当时最新）。
> 概览与目录职责见 [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md)；
> 升级过程见 [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md)。

---

## 一、支持解析的文件类型

### 1.1 当前实际支持：**只有 Markdown（`.md`）**

三处代码共同锁定了这个范围：

| 位置 | 代码 | 作用 |
|---|---|---|
| `documents/markdown_parser.py` | 只构造 `UnstructuredMarkdownLoader` | 没有 PDF / Word / HTML / Excel loader |
| `documents/write_milvus.py:21` | `if f.endswith('.md')` | 批量管道只扫 `.md` |
| `documents/milvus_db.py` | `file_path` 指向 `.md` | 单文件演示同样是 Markdown |

现有语料：`datas/md/` 下 **7 个 .md，689 行，43 KB**。

```
operational_faq.md     114 行   运维 FAQ
overview.md            144 行   Milvus 产品总览
performance_faq.md      51 行   性能 FAQ
product_faq.md         183 行   产品 FAQ
tech_report_0tfhhamx.md  80 行   技术报告
tech_report_0ui655n3.md  42 行   技术报告
troubleshooting.md     75 行   排障手册
```

### 1.2 仓库里存在的其它文件（非 Milvus 数据）

| 文件/目录 | 说明 |
|---|---|
| `datas/layout-parser-paper.pdf` | 4.68 MB 的 PDF 论文（LayoutParser 论文） |
| `datas/output/*.json` | **187 个** PDF 解析产物（Title 17 / NarrativeText 126 / Table 2 / CodeSnippet 2，覆盖 12+ 页） |

⚠️ 这 187 个 JSON **没有任何代码消费**，也不是本项目代码产出的：
它们的元数据带 `coordinates` / `PixelSpace`，属于 Unstructured **hi_res** 策略的签名，
而本项目只使用 `strategy='fast'`（不产坐标）。它们来自项目外部，保留作资料。

### 1.3 PDF 技术上可行，但需补三层依赖（实测结论）

`UnstructuredPDFLoader` 在 `langchain_community` 中**可用**，实测加载项目自带 PDF 时逐层报缺：

```
strategy="fast"  → ModuleNotFoundError: pdfminer              （纯 Python，已装）
                 → ModuleNotFoundError: pi_heif               （纯 Python，已装）
                 → ModuleNotFoundError: unstructured_inference
```

最后一步会拉入 15 个包，含 `torch 2.14.1`、`torchvision`、`onnxruntime 1.30.0`、
`opencv-python 5.0`、`accelerate`、`timm`、`matplotlib`、`onnx`。

> 即：**做 PDF 是新增功能，不是恢复原有意图**（原始 143 行 freeze 中无任何 PDF 专用包）。


---

## 二、文档载入策略与切片（分段）策略

### 2.1 载入策略：`mode='elements'`（元素级）

```python
loader = UnstructuredMarkdownLoader(
    file_path=md_file,
    mode="elements",       # 产出「原子元素」而非整篇文档
    strategy="fast"        # 快速解析，不做版面检测、不产坐标
)
```

`mode='elements'` 是一切的基础：把 Markdown 拆成带**父子关系**的最小单元
（`Title` / `NarrativeText` / `ListItem` / `Table` / `Image` ...），
每个元素带 `category`、`element_id`、`parent_id`、`category_depth` 等元数据。

`strategy='fast'` 只做文本切分，**不产出坐标**（所以本项目的产物里没有坐标）。

### 2.2 标题合并（项目自研，核心技巧）

朴素做法按字符数硬切会把标题和内容切散。这里利用元素的父子关系合并：

```
原始 elements：                      合并后（merge_title_content）：
  Title      "封装技术"(id=1)          ┌─ doc1  category='content'
  ListItem   "…芯片封装…"(parent=1)     │  page_content = "封装技术 -> …芯片封装…"
  ListItem   "…测试方法…"(parent=1)     │  title = "封装技术"
  Title      "测试技术"(id=2)           ┌─ doc2  category='content'
  ListItem   "…测试流程…"(parent=2)       │  "测试技术 -> …测试流程…"
  NarrativeText（无父节点，前言）         │  title = "测试技术"
                                        └─ doc3  category='NarrativeText'（顶层正文）
```

三条规则：
1. **Title 元素**存入 `parent_dict` 作为「父文档」，并记录 `title`
2. **子元素**（`parent_id` 命中）的 `page_content` 拼接到父文档
3. 父文档的 `category` 从 `Title` **改写为 `content`**（第六节的过滤依赖这一步）

若元素没有 `parent_id`（`strategy='fast'` 的实际情况），则按
`category == 'NarrativeText' and parent_id is None` 判定顶层正文块，单独保留并补 `title`。

### 2.3 切片（分段）策略：**分级切片**

```python
if len(d.page_content) > 5000:          # 仅大块做二次切分
    new_docs.extend(self.text_splitter.split_documents([d]))
    continue
new_docs.append(d)                      # 小块直接放行
```

`SemanticChunker(breakpoint_threshold_type="percentile")` 需要调用 embedding
计算语义相似度定位语义边界，**成本高**。因此只在块确实过大时才用 ——
这是**效果与成本的折中**。本项目 102 个块里绝大多数走「直接放行」分支。

### 2.4 字段兜底

```python
REQUIRED_FIELD_DEFAULTS = {
    'category': '', 'source': '', 'filename': '',
    'filetype': '', 'title': '', 'category_depth': 0,
}
```

`unstructured` 并不对所有元素都产出这些键（如顶层正文块没有 `category_depth`），
而 Milvus schema 中它们非空，缺字段会报 `DataNotMatchException`。
故统一 `setdefault` 兜底（**只补 metadata，不动正文**）。

### 2.5 入库时的 flush

`langchain-milvus` 的 `add_documents` 只写 WAL 不主动 flush，写完立即查会得到
`row_count = 0`、检索报 `nq [0] is invalid`。故 `MilvusVectorSave.add_documents()`
末尾显式 `client.flush(COLLECTION_NAME)`，失败只告警不中断。

---

## 三、Embedding 模型与嵌入方式

### 3.1 使用的模型

| 项 | 值 |
|---|---|
| 模型 | **BAAI/bge-small-zh-v1.5** |
| 维度 | **输出 512 维**的小模型 |
| 运行设备 | **CPU**（`model_kwargs={"device": "cpu"}`） |
| 归一化 | `encode_kwargs={"normalize_embeddings": True}` → L2 归一化 |
| 封装 | `langchain_huggingface.HuggingFaceEmbeddings`（底层 `sentence-transformers`） |
| 首次加载 | 从 HuggingFace Hub 下载约 95 MB，之后可 `HF_HUB_OFFLINE=1` 离线加载 |

### 3.2 嵌入方式

**文档侧**（`MilvusVectorSave.add_documents`）：

```python
# langchain-milvus 内部
dense_vectors  = embedding_func.embed_documents([d.page_content for d in docs])
sparse_vectors = ...   # 由服务端的 BM25 内置函数自动生成，不是客户端算的
```

- **稠密向量**：本地模型算出 512 维浮点向量，写入 `dense` 字段
- **稀疏向量**：客户端只声明一个 BM25 内置函数，由 **Milvus 服务端**在写入时
  自动对 `text` 分词并生成，写入 `sparse` 字段：

```python
schema.add_function(Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
))
```

### 3.3 中文分词（稀疏向量的前提）

BM25 依赖分词。中文不分词时「光刻」与「光刻机」是无关的两个词，关键词检索会失效。
本项目在 **Milvus 服务端**启用 jieba：

```python
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=6000,
    enable_analyzer=True,
    analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
)
```

`cnalphanumonly` 过滤数字与符号，让 BM25 统计更干净。

⚠️ **jieba 必须装在 Milvus 服务端**（客户端装了无用）。官方 `milvusdb/milvus` 镜像已内置。

### 3.4 为什么「归一化 + IP 度量」

`normalize_embeddings=True` 把向量 L2 归一化（模长=1），此时**内积（IP）在数值上等于
余弦相似度**。好处：① 检索用 IP 索引比余弦索引更快（无需运行时归一化）② 分数有统一含义。

### 3.5 生成模型（与嵌入模型分离）

| 项 | 值 |
|---|---|
| 服务 | DeepSeek 官方 `https://api.deepseek.com` |
| 模型 | `deepseek-flash`（备选 `deepseek-v4-pro`） |
| 温度 | `0`（判分/路由需可复现） |
| 协议 | OpenAI 兼容（`langchain-openai.ChatOpenAI`） |

> 嵌入模型（本地 BGE）与生成模型（云端 DeepSeek）是两套独立配置。

---

## 四、向量数据库实现方案

### 4.1 三层依赖

| 层 | 组件 | 版本 | 作用 |
|---|---|---|---|
| 服务端 | **Milvus** | 3.0.1 | 真正存数据、算相似度、执行 BM25 |
| 客户端 SDK | `pymilvus` | 3.0.2 | 建表、建索引、schema 原生访问 |
| LangChain 适配器 | `langchain-milvus` | 0.4.0 | 把 Milvus 包装成 `VectorStore` |

### 4.2 部署形态（实测环境，更新）

Windows 11 → WSL2 → Docker Desktop 29.8.1：

```
milvus-standalone   milvusdb/milvus:v3.0.1      端口 19530 / 9091  (healthy)
milvus-etcd         quay.io/coreos/etcd:v3.5.25
milvus-minio        milvusdb/minio:RELEASE.2024-12-18
attu                zilliz/attu                  端口 3000（图形界面）
```

> Milvus **没有 Windows 原生版**，通过 Docker + WSL2 运行 Linux 镜像。
>  由 v2.6.25 升级到 v3.0.1（跨大版本），实测 2.6 的数据可被 3.0.1 直接读取，无需重新导入。
> 客户端（`pymilvus`）是纯 Python，Windows 原生可用。另有纯 Python 的
> **Milvus Lite**（嵌入式，无需服务），但不支持 Function 形式 reranker、hybrid 稳定性也差，
> 只能验证「能否跑通」，验证不了「跑得好不好」。

### 4.3 集合 schema（实测 10 个字段）

| 字段 | 类型 | 关键配置 |
|---|---|---|
| `id` | INT64 | 自增主键 |
| `text` | VARCHAR(6000) | 块正文，`enable_analyzer=true`，jieba + cnalphanumonly |
| `dense` | FLOAT_VECTOR(512) | HNSW 索引，度量 IP，`M=16`, `efConstruction=64` |
| `sparse` | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED_INDEX，度量 BM25，`k1=1.2`, `b=0.75` |
| `category` | VARCHAR(500) | `content`=正文块，`Title`/`NarrativeText`=其它 |
| `title` | VARCHAR(1000) | 标题面包屑 |
| `category_depth` | INT64 | 元素嵌套层级 |
| `source` / `filename` / `filetype` | VARCHAR | 来源信息 |

### 4.4 两种 reranker 写法（按服务端版本切换）

```python
# MILVUS_RERANKER_STYLE=function（默认，官方新写法，需服务端 ≥ 2.6）
"reranker": Function(
    name="rrf",
    function_type=FunctionType.RERANK,
    input_field_names=[],                    # ① RRF 不接受输入字段，必须为空
    params={"reranker": "rrf", "k": 100},   # ② params 里必须有 reranker 判别键
)

# MILVUS_RERANKER_STYLE=legacy（旧写法，已 deprecated，兼容 Lite 与旧版服务端）
"ranker_type": "rrf",
"ranker_params": {"k": 100},
```

> 这两个易错点是**在真实 Milvus 2.6.25 上试出来的**，写错服务端报
> `code=1100 unsupported rerank function: []`。两者语义一致，返回结果与距离完全相同。

---

## 五、面向向量库的检索方式

项目用到的检索原语有 **4 种**，它们在 Milvus 里是不同层级的操作：

| # | 检索方式 | Milvus API | 本项目用在哪 |
|---|---|---|---|
| 1 | **稠密向量检索**（ANN） | `dense` 字段 + HNSW/IP | 混合检索的其中一路 |
| 2 | **稀疏向量检索**（BM25） | `sparse` 字段 + SPARSE_INVERTED_INDEX/BM25 | 混合检索的另外一路 |
| 3 | **混合检索 + 融合** | `hybrid_search(reqs, ranker=RRF)` | `retriever.invoke()` 实际走的就是这条 |
| 4 | **标量过滤** | `expr` 表达式 | 作为 1/3 的**前置约束**叠加使用 |

### 5.1 检索参数（`tools/retriever_tools.py`）

```python
search_kwargs = {
    "k": 4,                              # 最终返回 4 条
    "score_threshold": 0.1,              # ⚠️ 惰性参数，不被消费（见 5.4）
    "expr": 'category == "content"',     # 标量过滤
    "reranker": Function(... RRF k=100), # 或 ranker_type/ranker_params
}
```

### 5.2 混合检索的内部构造

`langchain-milvus` 的 `_collection_hybrid_search` 做的事：

```python
for field in self._vector_fields:                  # ['dense', 'sparse']
    if field in self._vector_fields_from_embedding: # 哪些字段需要客户端嵌入
        search_data = embedding_func.embed_query(query)   # dense → 向量
    else:
        search_data = query                              # sparse → 原始文本
    request = AnnSearchRequest(data=[search_data], anns_field=field,
                               param={"metric_type": "IP" if field=="dense" else "BM25"},
                               limit=2)
result = client.hybrid_search(collection_name, [req_dense, req_sparse],
                               ranker=reranker_obj, limit=4)
```

**两路各自 `limit=2`（top-2），融合后取 `limit=4`**。这是典型的「多路召回 → 融合 → 截断」结构。

### 5.3 为什么标量过滤用 `expr` 而不是 `filter`

langchain-milvus 的混合检索路径只用 `expr=expr` 接收标量过滤，
其余 search_kwargs 进 `**kwargs` 透传给 pymilvus。而：

- `MilvusClient.hybrid_search()` **没有 `filter` 形参**（`milvus_client.py:359-368`）
- `AnnSearchRequest` 本身也没有 `filter` 形参

所以写 `filter` 会被**静默丢弃**。正确写法是 `expr`。本项目原先就写错了
（`filter={"category":"content"}` 从未生效），现已改正。

### 5.4 `score_threshold` 为什么无效

`score_threshold` 在 `langchain-milvus` 与 `langchain-core` 中**都没有被消费**：
- `search_type='similarity'` 走 `similarity_search()`，该参数被 `**kwargs` 吞掉
- 只有 `search_type='similarity_score_threshold'` 才会走
  `similarity_search_with_relevance_scores()` 并真正按阈值过滤

本项目用的是 `search_type='similarity'`，所以该参数**一直是惰性的**（新旧版本都一样）。
要启用需改 `search_type`。

---

## 六、用户问答如何进行向量化

这是混合检索里最容易混淆的一点 —— **查询走两条完全不同的路径**：

```
用户问题（原始文本）
      │
      ├──→ [客户端] embedding_func.embed_query(query)
      │         └→ BGE-small-zh 本地推理 → 512 维归一化向量
      │         → 作为 dense 路���的 search_data
      │
      └──→ [服务端] search_data = query（原始字符串，不嵌入）
                → Milvus 服务端用 jieba 分词
                → 按 BM25 公式打分得稀疏向量
                → 作为 sparse 路线的 search_data
      │
      ▼
  hybrid_search([req_dense, req_sparse], ranker=RRF, limit=4)
```

| 维度 | 稠密路（dense） | 稀疏路（sparse） |
|---|---|---|
| 谁嵌入 | **客户端**（`embed_query`） | **不嵌入**，原文交给服务端 |
| 编码器 | BGE-small-zh（神经网络） | jieba 分词 + BM25 统计 |
| 产出 | 512 维稠密向量 | 稀疏索引项权重 |
| 擅长 | 同义改写、自然语言提问 | 专有名词、型号、精确关键词 |
| 延迟 | 需一次本地模型推理（CPU 毫秒级） | 无需模型，索引内计算 |

**一个重要设计后果**：因为 sparse 路用**原文**而非向量，检索词与入库词的
分词必须一致 —— 这正是服务端启用 jieba 的原因（`fast` 策略下入库不分词，
所以 query 与 text 都靠服务端同一套 jieba 分析器，保证分词口径一致）。

---

## 七、向量相似度匹配

### 7.1 两路各自的匹配算法

| 通道 | 索引结构 | 匹配算法 | 含义 |
|---|---|---|---|
| `dense` | **HNSW**（`M=16`, `efConstruction=64`） | **内积 IP** | 向量已 L2 归一化 → IP 等价于**余弦相似度**，取值 `[-1, 1]`，越大越相似 |
| `sparse` | **倒排索引** SPARSE_INVERTED_INDEX | **BM25**（`k1=1.2`, `b=0.75`） | 词频/文档频率统计打分，无固定值域 |
| `+` expr | — | 标量比较 | `category == "content"`，**先过滤再算相似度** |

### 7.2 为什么 HNSW 而不是全量扫描（FLAT）

- **FLAT**：暴力遍历全部向量，**召回 100%**，但数据量大时线性变慢
- **HNSW**：分层可导航小世界图，**近似最近邻（ANN）**，用 `M`（每节点邻居数）
  与 `efConstruction`（建索引候选队列）换速度，牺牲少量召回

本项目选 HNSW，是「百万级向量 + 毫秒延迟」这类企业知识库典型场景的常规选择。

### 7.3 RRF 融合：两路分数**不直接相加**

两路分数不可比：IP 值域 `[-1,1]`，BM25 值域随语料变化（可能 0~30）。
RRF（Reciprocal Rank Fusion）**只用排名、忽略分值**：

```
RRF_score(d) = Σ  1 / (k + rank_i(d))        # k = 100
                i ∈ {dense, sparse}
```

- `rank_i(d)` = 文档 d 在第 i 路结果中的排名（从 1 开始）
- 未命中的路不计入；某路排得越靠前贡献越大；两路都靠前的文档得分最高

**好处**：完全规避量纲问题，无需人工调权重即可融合异构检索器。
**代价**：丢失分数的绝对强度信息（不能据此设阈值）。

### 7.4 完整匹配时序

```
用户问题
  │
  ├─ ① expr 预过滤：category == "content"      ← 先砍掉非正文块
  │
  ├─ ② dense 路：HNSW + IP  →  top-2（按余弦相似度降序）
  │
  ├─ ③ sparse 路：BM25 倒排  →  top-2（按 BM25 分降序）
  │
  └─ ④ RRF 融合（k=100）→ 取 top-4
        ↓
   4 个 Document（带完整 metadata，供生成阶段使用）
```

**实测验证**：检索返回 4 条，且 4 条的 `category` **全部**为 `'content'` —— 证明 expr 过滤真实生效。

---

## 八、共有几种向量检索方式，如何搭配组合

### 8.1 检索方式清单（共 4 种）

| # | 方式 | 独立可用 | 本项目组合 |
|---|---|---|---|
| 1 | 稠密向量检索（HNSW + IP） | ✅ | 与 2 融合 |
| 2 | 稀疏向量检索（BM25 倒排） | ✅ | 与 1 融合 |
| 3 | 混合检索 + RRF 融合 | ✅（本项目主用） | 1 + 2 + 4 |
| 4 | 标量过滤（expr） | ✅ | 叠加在 1/2/3 之上 |

### 8.2 组合方式

```
                      ┌─ ① 稠密检索（HNSW / IP）─┐
用户问题 ──expr 过滤──┤                            ├─ RRF 融合 ──► top-4
                      └─ ② 稀疏检索（BM25）──────┘
                                        ▲
              reranker = Function(RERANK, params={"reranker":"rrf","k":100})
```

这是**唯一一条检索路径**。`retriever.invoke(query)` 内部一次性完成
「过滤 → 双路召回 → 融合 → 截断」，对上层是单次调用。

### 8.3 为什么不用「向量检索 + 关键词检索」两段式

业界常见另一种做法：先向量检索 top-20，再用交叉编码器重排。
本项目用的是**融合召回（fusion）**而非**重排（rerank）**：

| | 融合（本项目） | 重排（另一种） |
|---|---|---|
| 何时融合 | 检索阶段，两路结果合并 | 检索后，对候选集重新打分 |
| 需要模型 | 否（RRF 是纯公式） | 需要 cross-encoder / bge-reranker |
| 延迟 | 低 | 高 |
| 适合 | 需要关键词+语义双覆盖 | 需要极致精度、可接受延迟 |

选融合的原因：企业知识库专有名词多，必须保住关键词一路；
而 `bge-reranker` 之类交叉编码器需额外模型和 GPU，对本项目（102 块、CPU 推理）不划算。

### 8.4 项目里没有其它检索模式

`graph1.py` 与 `rag_agent.py` 调用的也是同一个 `retriever`，走的同样是混合检索。
**项目里没有任何地方配置纯稠密或纯稀疏检索** —— `expr` + 混合 + RRF 是全项目统一配置。

---

## 九、项目中有哪些 Agent，各自承担什么功能

严格说项目有 **2 个真正的 Agent** + **1 个 CRAG 工作流** + **4 个判分器（自评角色）**。

### 9.1 三个可运行入口

| # | 文件 | 类型 | 拓扑 | 职责 |
|---|---|---|---|---|
| 1 | `graph2/graph_2.py` | **CRAG 工作流**（手写 StateGraph） | **7 节点 / 9 边** | 完整检索问答：双通道 + 四道闸门 + 自纠错 |
| 2 | `graph/graph1.py` | **手写 ReAct Agent** | **6 节点 / 7 边** | 简化版：模型↔工具循环 + 文档相关性与问题改写 |
| 3 | `agent/rag_agent.py` | **LangChain 1.x 标准 Agent** | 4 节点 `__start__/model/tools/__end__` | 由 `create_agent()` 生成，官方标准写法 |

### 9.2 三个入口的对比

| 维度 | `rag_agent` | `graph1` | `graph2` |
|---|---|---|---|
| 构建方式 | `create_agent()`（官方封装） | 手写 `StateGraph` | 手写 `StateGraph` |
| 模型↔工具循环 | 内部自动 | `tools_condition` 条件边 | 不走工具，直接检索 |
| 检索时机 | 模型自主决定 | 固定在 `agent` 之后 | 固定在路由之后 |
| 网络搜索 | 无 | 无 | **有**（Tavily） |
| 问题路由 ① | 无 | 无 | **有** |
| 文档相关性 ② | 无 | 有 | **有** |
| 幻觉检查 ③ | 无 | 无 | **有** |
| 匹配度检查 ④ | 无 | 无 | **有** |
| 问题改写 | 无 | 有（`rewrite`） | 有（`transform_query`） |
| 生成熔断 | 不适用 | 无 | **有**（`MAX_GENERATION_RETRY=2`） |
| 会话记忆 | ✅ SqliteSaver | ✅ SqliteSaver | ✅（LangGraph 状态） |
| 适合场景 | 快速问答、模型自主决策 | 学习 LangGraph 编排 | 生产问答、要质量控制 |

### 9.3 `graph2` 的 7 个节点与 9 条边

```
节点：__start__ / retrieve / web_search / grade_documents /
      transform_query / generate / __end__

边（9 条）：
  __start__       ⇢ retrieve              （路由判定 = vectorstore）
  __start__       ⇢ web_search            （路由判定 = websearch）
  retrieve        →  grade_documents
  transform_query →  retrieve
  web_search      →  generate
  grade_documents ⇢ transform_query        （不相关/空，transform_count < 2）
  grade_documents ⇢ web_search             （transform_count ≥ 2，降级网络搜索）
  grade_documents →  __end__                （无相关文档且已走过网络搜索）
  generate        ⇢ generate               （not supported，受熔断）
  generate        →  __end__                （useful）
  generate        ⇢ transform_query        （not useful）
```

### 9.4 四个判分器（自评角色，不是 Agent）

| 判分器 | Schema | 判断什么 | 不通过时 |
|---|---|---|---|
| `query_route_chain` | `RouteQuery.datasource` | 查库还是上网 | 走另一条通道 |
| `grader_chain` | `GradeDocuments.binary_score` | 单个文档是否切题 | 过滤掉该文档 |
| `grade_hallucinations_chain` | `GradeHallucinations.binary_score` | 回答是否基于文档 | 重新生成（受熔断） |
| `grade_answer_chain` | `GradeAnswer.binary_score` | 回答是否回答了问题 | 改写问题重试 |

四者都用「LLM + Pydantic 结构化输出」把主观判断转成可路由的结论，调用方式：

```python
llm.with_structured_output(GradeDocuments, method="function_calling")
```

> `langchain-openai` 1.x 把默认方法从 `function_calling` 改成了 `json_schema`，
> 而 DeepSeek 不支持 `json_schema`（报 `This response_format type is unavailable now`）。
> 项目 5 处调用点全部显式指定。

### 9.5 工具（Tool）清单

| 工具 | 定义位置 | 用途 |
|---|---|---|
| `rag_retriever` | `tools/retriever_tools.py` | Milvus 混合检索（k=4，expr 过滤） |
| `TavilySearchResults` | `llm_models/all_llm.py` | 网络搜索（`max_results=2`） |

> 工具描述必须与语料主题一致。原先 `rag_retriever` 的描述写的是「半导体和芯片」
> （原项目遗留），而语料是 Milvus 文档，导致 `rag_agent` 依据描述判断
> 「工具与问题无关」而**直接拒答**。现已改为「Milvus 向量数据库」主题。

---

## 十、数据流速查

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| **入库** | `datas/md/*.md` | 元素化 → 标题合并 → 分级切片 → 字段兜底 → BGE 嵌入 + 服务端 BM25 | Milvus 集合（102 块） |
| **检索** | 用户问题（文本） | expr 过滤 → dense(embed_query) + sparse(原文) → RRF 融合 | top-4 Document |
| **生成** | top-4 块 | 拼上下文 + 提示词 → DeepSeek 生成 | 回答文本 |
| **自检** | 回答文本 | ③ 幻觉 + ④ 匹配度 | 通过 / 重生成 / 改写问题 |

## 十一、关键配置速查

| 配置项 | 默认 | 影响 |
|---|---|---|
| `LLM_MODEL` / `LLM_BASE_URL` | gpt-4o-mini @ xiaoai.plus | 生成与判分模型（实测用 deepseek-flash @ api.deepseek.com） |
| `LLM_STRUCTURED_METHOD` | `function_calling` | 判分器的结构化输出方式 |
| `LLM_DISABLE_THINKING` | `1` | DeepSeek thinking 会阻塞 `tool_choice` |
| `MILVUS_URI` | 失效的旧公网地址 | **必须改** |
| `MILVUS_RERANKER_STYLE` | `function` | 需服务端 ≥ 2.6（3.0.1 实测可用）；Lite/旧服务用 `legacy` |
| `COLLECTION_NAME` | `t_collection01` | 目标集合 |
