# RAG 企业知识库项目 · 详细说明

> 本文档是对代码的完整梳理：架构、数据流、模块职责、关键原理、配置与已知限制。
> 描述均基于源码实际阅读 + 真实 Milvus 2.6.25 / DeepSeek / Tavily 环境下的实测。
> 升级细节见 [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md)，验证结果见 [`迁移验证报告.md`](迁移验证报告.md)。

---

## 一、项目概览

### 1.1 它是什么

一个**企业知识库检索问答系统**（RAG）。把企业内部 Markdown 文档切分、向量化后存入向量库；
用户提问时先检索相关片段，再让大模型基于这些片段生成答案，并通过一系列判分节点**自检与纠错**。

它不是"能跑通就行"的 Demo，有明确工程考量：

| 考量 | 体现在哪 |
|---|---|
| 混合检索 | 稠密向量（语义）+ BM25 稀疏（关键词）双路召回，RRF 融合 |
| 中文分词 | Milvus 服务端启用 jieba 分析器，否则中文 BM25 失效 |
| 检索准确 | 表达式过滤（`expr`）排除纯标题块，只保留正文 |
| 答案可信 | 四道判分闸门：问题路由 / 文档相关性 / 答案幻觉 / 问题匹配度 |
| 自纠错 | 检索不到就改写问题重试，生成不合格就重新生成，且有熔断防死循环 |
| 会话记忆 | LangGraph 检查点（SqliteSaver）持久化，支持多轮与跨重启 |
| 双通道 | 知识库检索 + 网络搜索，由路由器自动选择 |

### 1.2 规模

| 项 | 数量 |
|---|---|
| 业务代码 | **1692 行**（38 个 `.py`，含空 `__init__.py`） |
| 核心工作流 | 3 个入口（CRAG 完整版 / 简化 ReAct / 标准 Agent） |
| 判分链 | 4 个 |
| 语料 | 7 个 Markdown，689 行，43 KB |
| 检索单元 | 入库后约 102 块 |
| 依赖 | 31 个精确 pin |

代码量分布（不含空文件）：

| 目录 | 行数 | 职责 |
|---|---|---|
| `graph2/` | 563 | CRAG 完整工作流（核心） |
| `documents/` | 383 | 解析、建表入库、批量写入 |
| `graph/` | 280 | 简化版 ReAct 工作流 |
| `utils/` | 176 | 配置、日志、检查点工厂、打印 |
| `tools/` | 155 | 检索器与工具封装（含 1 个 WIP 文件） |
| `agent/` | 67 | LangChain 1.x 标准 Agent |
| `llm_models/` | 40 | LLM 与嵌入模型 |

### 1.3 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangChain 1.4.3 / LangGraph 1.2.14 |
| 向量库 | Milvus 3.0.1 + langchain-milvus 0.4.0 + pymilvus 3.0.2 |
| 嵌入 | BAAI/bge-small-zh-v1.5（512 维，本地 CPU） |
| 生成 | DeepSeek `deepseek-flash`（OpenAI 兼容协议） |
| 网络搜索 | Tavily |
| 文档解析 | Unstructured 0.27.10（`strategy='fast'`） |
| 结构化输出 | Pydantic v2 + `with_structured_output(method='function_calling')` |
| 会话记忆 | langgraph-checkpoint-sqlite 3.1.1 |
| 日志 | loguru |
| 环境 | Python 3.12.7（Windows / Linux 均可） |

---

## 二、目录结构与逐文件职责

```
RAG_PROJECT/                ← Python 包根
├── agent/
│   ├── rag_agent.py            67  create_agent + SqliteSaver 检查点
│   └── __init__.py              0
├── documents/
│   ├── markdown_parser.py     124  解析 + 标题合并 + 切片
│   ├── milvus_db.py           152  Milvus schema 建表 + 向量库封装
│   ├── write_milvus.py        107  多进程批量入库管道
│   └── __init__.py              0
├── graph/
│   ├── graph1.py             154  StateGraph 构图 + create_graph 工厂 + main 入口
│   ├── graph_state1.py        17  AgentState / Grade
│   ├── agent_node.py          25  模型 + 工具绑定节点
│   ├── generate_node.py       37  RAG 生成节点
│   ├── rewrite_node.py        36  问题改写节点
│   ├── get_human_message.py   11  取出最后一条用户消息
│   └── __init__.py              0
├── graph2/                CRAG 完整工作流（核心）
│   ├── graph_2.py            207  构图（7 节点/9 边）+ 熔断 + 工厂 + 入口
│   ├── graph_state2.py        22  GraphState 与 4 个判分用Pydantic 模型
│   ├── query_route_chain.py   48  路由器：向量库 or 网络搜索
│   ├── retriever_node.py      18  向量库检索节点
│   ├── web_search_node.py     45  Tavily 检索节点
│   ├── grade_documents_node.py 32  文档相关性闸门 + 是否改写问题的决策
│   ├── transform_query_node.py 46  信息不足时改写问题
│   ├── generate_node2.py      44  RAG 生成 + 幻觉检查 + 答案判分
│   ├── grader_chain.py        33  文档相关性判分器
│   ├── grade_hallucinations_chain.py 34  幻觉判分器
│   ├── grade_answer_chain.py  34  答案匹配度判分器
│   └── __init__.py              0
├── llm_models/
│   ├── all_llm.py             25  ChatOpenAI + Tavily 工具（配置驱动）
│   ├── embeddings_model.py    15  BGE-small-zh（512 维，CPU）
│   └── __init__.py              0
├── tools/
│   ├── retriever_tools.py     59  Milvus 检索器 + 工具封装
│   ├── text_search.py         96  ⚠️ WIP 实验文件，非生产代码（见 8.3）
│   └── __init__.py              0
├── utils/
│   ├── env_utils.py           65  全部配置（密钥/服务/路径/LLM/检索）
│   ├── log_utils.py           60  loguru 日志封装
│   ├── checkpointer.py        29  SqliteSaver 工厂
│   ├── print_utils.py         22  流式事件去重打印
│   └── __init__.py              0
├── datas/
│   ├── md/                    7 个 Markdown 语料
│   ├── output/               187 个 PDF 解析产物（非 Milvus 数据，见 8.4）
│   └── layout-parser-paper.pdf
├── main.py                    16  ⚠️ PyCharm 模板残留，无真实入口
├── draw_png.py                12  导出工作流 mermaid 图
└── logs/                          日志目录

工作区根另有：requirements*.txt、UPGRADE_NOTES.md、迁移验证报告.md、
scripts/（验证脚本）、.env / .env.example、.gitignore
```

---

## 三、数据流全景

### 3.1 入库链路（一次性）

```
datas/md/*.md（7 个文件，689 行）
      │
      │ ① UnstructuredMarkdownLoader(mode='elements', strategy='fast')
      │    产出「原子元素」：Title / NarrativeText / ListItem / Table ...
      │    每元素带 metadata：category、element_id、parent_id、
      │                          filename、filetype、category_depth ...
      ▼
   元素列表
      │
      │ ② merge_title_content()      ← 项目自研的核心技巧
      │    · 顶层正文块（NarrativeText 且无 parent_id）→ 直接保留，补 title
      │    · Title 元素 → 作为「父文档」存入 parent_dict，并记录自身标题
      │    · 子元素（ListItem/Table 等）→ 内容拼进父文档，
      │      父文档 category 改为 'content'
      │    效果：标题与正文合并成「A -> B -> 正文」的面包屑结构
      ▼
   合并后的 Document[]
      │
      │ ③ SemanticChunker(breakpoint_threshold_type='percentile')
      │    仅对 > 5000 字符的块做二次语义切分
      │    （分级策略：小块直接放行，省embedding 成本）
      ▼
   chunk 列表（本次 102 块）
      │
      │ ④ _ensure_required_fields()
      │    补齐 schema 必需但 unstructured 未必产出的字段：
      │    category / source / filename / filetype / title / category_depth
      ▼
   就绪的 Document[]
      │
      │ ⑤ 嵌入：bge_embedding.embed_documents() → 512 维稠密向量
      │    BM25 内置函数在服务端把 text 转成稀疏向量
      ▼
   Milvus 集合
```

**集合 schema**（`documents/milvus_db.py`，实测已在 Milvus 2.6.25 建成）：

实际共 **10 个字段**（下表最后一行是 3 个字段合并展示），实测输出：
`id`(INT64) / `text`(VARCHAR, analyzer=on) / `category` / `source` / `filename` /
`filetype` / `title` / `category_depth`(INT64) / `sparse`(SPARSE_FLOAT_VECTOR) /
`dense`(FLOAT_VECTOR, dim=512)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 | 自增主键 |
| `text` | VARCHAR(6000) | 块正文，`enable_analyzer=true`，`analyzer_params={"tokenizer":"jieba","filter":["cnalphanumonly"]}` |
| `dense` | FLOAT_VECTOR(512) | 稠密向量，HNSW（`M=16`, `efConstruction=64`），度量 IP |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量，SPARSE_INVERTED_INDEX（`k1=1.2`, `b=0.75`） |
| `category` | VARCHAR(500) | `content` 表示正文块，`Title` 表示纯标题块 |
| `title` | VARCHAR(1000) | 标题面包屑（由 merge 阶段生成） |
| `category_depth` | INT64 | 元素嵌套层级 |
| `source` / `filename` / `filetype` | VARCHAR | 来源信息 |

入库有两条入口：
- `documents/write_milvus.py`：多进程批量管道（扫描目录 → 解析 → 有界队列 → 批量写入）
- `documents/milvus_db.py` 的 `__main__`：单文件演示

### 3.2 查询链路（`graph2`，CRAG 完整版）

```
用户问题
   │
   ├─▶ ① route_question【闸门①问题路由】
   │      LLM + Pydantic(RouteQuery) 判定：vectorstore or websearch
   │      ↓
   ├─▶ vectorstore 路径                    websearch 路径
   │      │                                 │
   │      ▼                                 ▼
   │   retriever（混合检索 k=4）         Tavily 搜索
   │      · dense 512 维 + sparse BM25      │
   │      · RRF 融合（k=100）               │
   │      · expr: category == "content"     │
   │      │                                 │
   │      ▼                                 │
   │   grade_documents【闸门②文档相关性】    │
   │      LLM + Pydantic(GradeDocuments)    │
   │      │                                 │
   │      ├─ 有相关文档 ──────────────┐      │
   │      │                                │      │
   │      └─ 全不相关/为空            │      │
   │           transform_query_count<2      │      │
   │           → transform_query（改写问题）  │      │
   │                                     │      │
   │                                     ▼      ▼
   │◀─────────────────────────────────────── generate【RAG 生成 / Web 答案】
   │                                          · 有文档 → 拼上下文 + 提示词
   │                                          · 无文档 → 直接用网页内容
   │                                          │
   │                                          ▼
   │                                    grade_generation【闸门③幻觉 + 闸门④匹配度】
   │                                          ③ 生成内容是否基于参考文档？
   │                                          ④ 回答是否真的回答了问题？
   │      ┌───────────────────────────────────┼────────────────────┐
   │      │ yes │ no（且重试未超限）           │ yes                 │ no
   │      │      ▼                            ▼                     ▼
   │      │  "not supported" 回到 generate  "useful"            "not useful"
   │      │  （受 MAX_GENERATION_RETRY 熔断）    │                     │
   │      └──────────────────► END ◄────────┘                     │
   │                                                                │
   │                                          "not useful" → transform_query → 重试
   └────────────────────────────────────────────────────────────────┘
```

**拓扑结构**：`graph2` 共 **7 节点 / 9 边**（`__start__`、`retrieve`、`web_search`、
`grade_documents`、`transform_query`、`generate`、`__end__`），
与升级前导出的 `graph_rag2.png` 完全一致。

**四道闸门的作用**：

| 闸门 | 位置 | 判别对象 | 不通过时 |
|---|---|---|---|
| ① 问题路由 | `route_question` | 该查库还是该上网 | 走另一条通道 |
| ② 文档相关性 | `grade_documents` | 召回文档是否切题 | 改写问题重检索（≥2 次转网络搜索） |
| ③ 幻觉检测 | `grade_generation` | 回答是否基于检索到的文档 | 重新生成（受熔断保护） |
| ④匹配度 | `grade_generation` | 回答是否真的回答了问题 | 改写问题重试 |

### 3.3 查询链路（`graph1`，简化版）

```
START → agent ⇄ retrieve → grade_documents → generate → END
                ↑                        │
                └──── rewrite ◄──────────┘
```

没有路由、没有网络搜索、没有幻觉检测，`rewrite` 后直接回 `agent`重来。
它的价值是**结构清晰、便于对照理解** LangGraph 的基本用法。

### 3.4 查询链路（`rag_agent`，1.x 标准 Agent）

```
START → model ⇄ tools → END
```

由 `create_agent()` 生成，内部自动处理模型↔工具循环；会话记忆交给 `checkpointer` + `thread_id`。
这是 LangChain 1.x 官方推荐的标准写法，`graph1`/`graph2` 是手写编排的对照版本。

---

## 四、关键原理与技术要点

### 4.1 混合检索（Hybrid Search）

单靠向量检索在企业知识库里有明显短板：型号、编号、人名、缩写这类**专有名词**的语义向量区分度差。
本项目采用双路召回：

| 通道 | 字段 | 索引 | 擅长 |
|---|---|---|---|
| 稠密（语义） | `dense` 512 维 | HNSW（IP 度量） | 同义表达、自然语言提问 |
| 稀疏（关键词） | `sparse` | SPARSE_INVERTED_INDEX（BM25） | 专有名词、型号、精确关键词 |

两路结果用 **RRF（Reciprocal Rank Fusion）** 融合，参数 `k=100`。
RRF 只看**排名**不看分数，规避了两路分数量纲不一致的问题。

**写法上有两种可选**（由 `MILVUS_RERANKER_STYLE` 控制）：

```python
# function（默认，官方新写法，需服务端 Milvus 2.6+）
"reranker": Function(
    name="rrf",
    function_type=FunctionType.RERANK,
    input_field_names=[],                # ① RRF 不接受输入字段，必须为空
    params={"reranker": "rrf", "k": 100} # ② params 里必须有 reranker 判别键
)

# legacy（旧写法，已 deprecated 但兼容面最广，Milvus Lite 只能用这个）
"ranker_type": "rrf",
"ranker_params": {"k": 100}
```

> 这两个易错点是**在真实 Milvus 2.6.25 上实测出来的**，服务端会报
> `code=1100 unsupported rerank function: []`。两者语义一致，返回结果与距离完全相同。

### 4.2 中文分词（jieba）

BM25 依赖分词。中文不分词时，「光刻」和「光刻机」会被当成两个无关词，导致关键词检索失效。
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

`cnalphanumonly` 用于过滤数字和符号，让 BM25 统计更干净。

⚠️ **jieba 必须装在 Milvus 服务端**（客户端装了没用）。官方 `milvusdb/milvus` 镜像已内置；
自定义镜像需自行 `docker exec -it <milvus> pip install jieba`。

### 4.3 结构感知切片（merge_title_content）

这是本项目最有价值的一段代码（`documents/markdown_parser.py`）。

朴素做法是按字符数硬切，会把标题和内容切散。这里利用 Unstructured 的
`mode='elements'` 产出**带父子关系**的原子元素，再做合并：

```
原始 elements：                      合并后：
  Title    "封装技术"（element_id=1）    ┌─ doc1: category='content'
  ListItem "…芯片封装…"（parent=1）      │  "封装技术 -> …芯片封装…"
  ListItem "…测试方法…"（parent=1）      └─ title='封装技术'
  Title    "测试技术"（element_id=2）    ┌─ doc2: category='content'
  ListItem "…测试流程…"（parent=2）      │  "测试技术 -> …测试流程…"
  NarrativeText（无父节点，前言）        └─ title='测试技术'
                                       doc3: category='NarrativeText'（顶层正文，补 title）
```

三个要点：
1. **标题下沉**：子元素内容拼进父标题文档，形成「A -> B -> 正文」面包屑，检索时带层级语境
2. **category 改写**：合并后文档的 category 从 `Title` 改为 `content`
3. **过滤配合**：检索时用 `expr: category == "content"` 排除纯标题块

> ⚠️ 这里踩过一个坑：顶层正文块（无父节点）走单独分支，不会被赋予 `title`，
> 而 schema 中 `title` 非空 → 写入必然报 `DataNotMatchException`。
> 现已由 `_ensure_required_fields()` 统一兜底。

### 4.4 分级切片策略

```python
if len(d.page_content) > 5000:      # 只有大块才做二次语义切分
    new_docs.extend(self.text_splitter.split_documents([d]))
    continue
new_docs.append(d)                  # 小块直接放行
```

`SemanticChunker` 需调用 embedding 做语义边界检测，成本高。
只在块确实过大时才用，是**效果与成本的折中**。

### 4.5 判分链与结构化输出

四道闸门都用「大模型 + Pydantic 模型」把主观判断转成**可路由的结构化结论**：

| 判分器 | Schema | 输出 |
|---|---|---|
| 路由器 | `RouteQuery.datasource: Literal["vectorstore","websearch"]` | 二选一，决定走哪条通道 |
| 文档相关性 | `GradeDocuments.binary_score: Literal["yes","no"]` | 该文档是否切题 |
| 幻觉检测 | `GradeHallucinations.binary_score: Literal["yes","no"]` | 回答是否基于文档 |
| 答案匹配度 | `GradeAnswer.binary_score: Literal["yes","no"]` | 回答是否回答了问题 |

这样 LangGraph 才能用返回值做条件边分流：

```python
llm.with_structured_output(GradeDocuments, method="function_calling")
```

> ⚠️ `method` 必须显式指定。`langchain-openai` 1.x 把默认值从 `function_calling`
> 改成了 `json_schema`，而 DeepSeek 官方 API 不支持 `json_schema`，会报
> `This response_format type is unavailable now`。项目里共 5 处调用点，都已显式指定。

配套的开关（`utils/env_utils.py`）：

| 配置 | 默认 | 作用 |
|---|---|---|
| `LLM_STRUCTURED_METHOD` | `function_calling` | 结构化输出方式；换到支持 `json_schema` 的服务商可改|
| `LLM_DISABLE_THINKING` | `1` | DeepSeek 开启 thinking 时不支持强制 `tool_choice`，故默认关闭 |

### 4.6 会话记忆（Checkpointer）

```python
# utils/checkpointer.py
conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
saver = SqliteSaver(conn)
saver.setup()
```

**为什么不用内存版**：官方文档指出 `MemorySaver` / `InMemorySaver` 只把检查点存在内存，
进程重启即全部丢失，官方建议生产改用持久化实现。

**选型依据**：
- 官方把 `PostgresSaver` 定位为生产首选，`SqliteSaver` 为本地开发/小型项目
- 本项目是**同步、轻量**的 RAG 应用，`SqliteSaver` 的 docstring 自述适用于
  "lightweight, synchronous use cases (demos and small projects)"
- 上生产时把 `create_checkpointer()` 换成 `PostgresSaver.from_conn_string(...)` 即可

> `SqliteSaver.setup()` **不要手动调用** —— 官方 docstring 明确
> "It is called automatically when needed and should not be called directly by the user"。

### 4.7 熔断（MAX_GENERATION_RETRY）

`"not supported" → generate` 是一条**无条件自环**。若模型反复产生幻觉，工作流会无限循环。
`transform_count` 只保护了 `transform_query` 分支，管不到这里。

```python
MAX_GENERATION_RETRY = 2

# 判分函数中
generation_count = state.get("generation_count", 0)
if generation_count >= MAX_GENERATION_RETRY:
    return "not useful"     # 改走受 transform_count 保护的分支，循环有界
```

实测中确实观察到 web_search 与 vectorstore 之间多次振荡，
若无此熔断就是死循环。

### 4.8 数据入库的 flush 语义

`langchain-milvus` 的 `add_documents` 只把数据写进 **WAL**，不主动 flush。
实测表现为：写完立刻查 `row_count = 0`、检索报 `nq [0] is invalid`（查到空集合）。
现已在 `MilvusVectorSave.add_documents()` 末尾显式 `client.flush(COLLECTION_NAME)`，
失败只告警不中断（数据仍在 WAL，会被自动 flush）。

### 4.9 模块结构：工厂函数 + main 入口

三个工作流文件都遵循同一结构：

```python
def create_graph():
    """构图并编译，返回已编译的图对象。"""
    ...
    return workflow.compile(checkpointer=checkpointer)

graph = create_graph()        # 模块级导出，保持原有 import 用法不变

def main():
    ...  # 交互循环

if __name__ == '__main__':
    main()
```

依据是 LangGraph 官方《Application structure》：`langgraph.json` 的 `graphs` 入口
既可以是「已编译的图对象」，也可以是「一个用于创建图的函数」。

**改造前**这些文件在模块顶层就执行 `while True: input()`，import 即阻塞、无法复用。
现在三个入口都可以被安全 import。

### 4.10 检索器的绑定时机（易踩）

`tools/retriever_tools.py` 在**模块 import 时**就执行 `as_retriever(...)`，
此时使用的 `COLLECTION_NAME` 被固化。后果：

- 运行中修改 `COLLECTION_NAME` 对已导入的检索器**无效**，需重启进程或 `importlib.reload`
- 检索器指向不存在的集合时，检索返回空并报
  `nq [0] is invalid, nq should be in range [1, 16384], but got 0`
  —— 这个报错**不指向真实原因**，容易误判为数据问题

---

## 五、配置清单

所有配置集中在 `utils/env_utils.py`，从 `.env` 读取（**不要硬编码**）。

### 5.1 密钥

| 变量 | 必需 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | 同时用于 `ChatOpenAI` 与 `OpenAIEmbeddings`。**两者在模块顶层构造**，缺失会导致 import 直接失败 |
| `TAVILY_API_KEY` | ❌ | `web_search` 分支需要。缺失时该分支降级（工具返回错误字符串，已被优雅处理） |
| `DEEPSEEK_API_KEY` | ❌ | 备用配置，默认已注释 |

> ⚠️ `load_dotenv(override=True)` 意味着 **`.env` 优先级高于系统环境变量**。
> 若在 `.env` 里写 `OPENAI_API_KEY=`（空值），会把你系统里已设的变量**覆盖成空**。
> 要么填真实值，要么整行注释掉，**不要留空赋值**。

### 5.2 服务与路径

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_URI` | `http://<old-ip>:19530` | Milvus 服务地址。**默认值已失效，必须改** |
| `COLLECTION_NAME` | `t_collection01` | 目标集合名 |
| `DATAS_DIR` | `<包根>/datas` | 语料目录 |
| `CHECKPOINT_DB` | `<工作区根>/checkpoints.db` | 会话记忆持久化文件 |

### 5.3 LLM 服务商

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://xiaoai.plus/v1` | OpenAI 兼容端点。换服务商改这里 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `LLM_TEMPERATURE` | `0` | 生成温度 |
| `LLM_STRUCTURED_METHOD` | `function_calling` | 结构化输出方式 |
| `LLM_DISABLE_THINKING` | `1` | 关闭 DeepSeek thinking 模式（与 `tool_choice` 冲突） |

**DeepSeek 配置示例**：
```
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_TEMPERATURE=0
```
> DeepSeek 官方当前可用模型为 `deepseek-flash` 与 `deepseek-v4-pro`（**没有** `deepseek-chat`）。

### 5.4 检索

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_RERANKER_STYLE` | `function` | `function`（官方新写法，需2.6+）/ `legacy`（旧写法，兼容 Lite 与旧版服务端） |

---

## 六、运行方式

### 6.1 准备

```powershell
# 1. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 2. 配置密钥与服务地址
Copy-Item .env.example .env      # 然后填入真实值

# 3. 确认 Milvus 已启动且版本 >= 2.6（若要用 function 写法）
docker ps                # 应看到 milvus-standalone healthy
.venv\Scripts\python.exe scripts\milvus_probe.py     # 只读探测：版本/集合/兼容性判断
```

> 首次使用建议先跑 `milvus_probe.py`，它会明确告诉你服务端版本、是否需要鉴权、
> 以及当前 `reranker` 写法是否兼容。

### 6.2 四个入口

```powershell
# A) 批量入库（首次必做）
.venv\Scripts\python.exe -m RAG_PROJECT.documents.write_milvus

# B) 完整 CRAG 工作流（双通道 + 四道闸门 + 自纠错）
.venv\Scripts\python.exe -m RAG_PROJECT.graph2.graph_2

# C) 简化版 ReAct 工作流
.venv\Scripts\python.exe -m RAG_PROJECT.graph.graph1

# D) LangChain 1.x 标准 Agent
.venv\Scripts\python.exe -m RAG_PROJECT.agent.rag_agent
```

退出方式：输入 `q` / `exit` / `quit`。

### 6.3 建议的 Milvus 部署方式（本地开发）

```yaml
# docker-compose.yml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    volumes: [etcd_data:/etcd]
    command: etcd -advertise-client-urls=http://127.0.0.1:2379
             -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment: {MINIO_ACCESS_KEY: minioadmin, MINIO_SECRET_KEY: minioadmin}
    volumes: [minio_data:/minio_data]
    command: minio server /minio_data
  standalone:
    image: milvusdb/milvus:v3.0.1      # 3.0+；2.6 以上才支持 Function 形式 reranker
    command: ["milvus", "run", "standalone"]
    environment: {ETCD_ENDPOINTS: etcd:2379, MINIO_ADDRESS: minio:9000}
    volumes: [milvus_data:/var/lib/milvus]
    ports: ["19530:19530", "9091:9091"]
    depends_on: [etcd, minio]
volumes: {etcd_data: {}, minio_data: {}, milvus_data: {}}
```

```powershell
docker compose up -d
docker exec -it standalone pip install jieba   # 若镜像未内置
```

可选装 `zilliz/attu`（Milvus 图形界面，3000 端口）。
> 若 Attu 与 Milvus 不在同一个 Docker 网络，Attu 里要填
> `http://<milvus容器名>:19530` 而不是 `localhost:19530`（容器内 `localhost` 指向自己）。

### 6.4 在 AutoDL / 云服务器上跑

- Milvus 需要 **≥4GB内存**（官方最低要求，推荐 8GB+），因为要同时跑 Milvus + etcd + MinIO
- AutoDL 容器通常**不允许** Docker 起 Milvus；可考虑用平台的**自定义服务**做端口转发
- 项目本身在云上跑的好处：模型权重、语料都在那边，不用重复下载

---

## 七、测试与验证

### 7.1 迁移验证脚本（不碰真实服务）

```powershell
.venv\Scripts\python.exe scripts\verify_migration.py --with-embeddings
.venv\Scripts\python.exe scripts\verify_migration.py --report 迁移验证报告.md
```

8 组 21 项闸门：

| 闸门 | 内容 |
|---|---|
| G1 编译 | 38 个 `.py` 全部可编译 |
| G2 import | AST 静态校验第三方符号（约 70 个 / 28 个文件） |
| G3 探针 | 8 项高风险 API：标准 Agent API、Tavily、Loader、Chunker、pymilvus枚举、Milvus 适配器签名、LC核心 API、HF |
| G4 切片 | elements metadata 键兼容性 + 切片产物统计 |
| G5 拓扑 | 两张图**以升级前快照为基线做断言**（6节点7边 / 7节点9边） |
| G6 向量 | `dim=512` + L2 归一化 |
| G7 1.x写法 | `create_agent` +检查点 + 会话隔离（用假模型，不发网络请求） |
| G8 熔断 | `MAX_GENERATION_RETRY` 的返回值序列 |

### 7.2 端到端冒烟测试（连真实服务）

```powershell
$env:HF_HUB_OFFLINE = "1"     # 模型已缓存时避免联网
.venv\Scripts\python.exe scripts\e2e_smoke_test.py
```

覆盖：解析 → 建表 → 写入 → 混合检索 → `expr` 过滤验证 → 清理临时集合。

> 脚本会自动注入占位密钥（仅构造对象用，不发请求）。
> 想跑真实的 LLM 链路，直接执行四个入口之一即可。

---

## 八、已知限制与注意事项

### 8.1 会影响使用的限制

| 项 | 说明 |
|---|---|
| **`MILVUS_URI` 默认值已失效** | 原项目硬编码的 `http://<old-ip>:19530` 是已失效的公网地址，必须改为自己的实例 |
| **`score_threshold` 是惰性参数** | `langchain-milvus` 与 `langchain-core` 均不消费它，一直是透传后被忽略。要启用阈值过滤需改 `search_type='similarity_score_threshold'` |
| **`langchain-community` 已停止维护** | 导入即有 DeprecationWarning，1.0 将移除。`UnstructuredMarkdownLoader` 目前只在该包提供 |
| **`TavilySearchResults` 已弃用** | 标记 `removal='1.0'`，将来需迁 `langchain_tavily.TavilySearch`（注意其默认返回**字符串**而非 dict 列表） |
| **Milvus Lite 功能精简** | 支持建表（BM25+稀疏+512维）但**不支持 Function 形式 reranker**（报 `code=6`），且 `hybrid_search` 不稳定。只能验证"能否跑通"，验证不了"跑得好不好" |
| **`InMemorySaver` 已被替换** | 官方指出内存版重启即丢检查点；本项目已改为 `SqliteSaver`。生产环境建议再换 `PostgresSaver` |
| **`main.py` 无真实入口** | 仍是 PyCharm 模板残留（`print_hi('PyCharm')`），请用四个入口之一 |
| **`graph1`/`graph_2` 依赖 `input()`** | 交互式脚本，不适合自动化/服务端部署 |

### 8.2 `langchain-community` 弃用后的迁移方向

| 现用 | 目标 | 说明 |
|---|---|---|
| `UnstructuredMarkdownLoader` | `langchain_unstructured` | 该包只导出 `UnstructuredLoader`，**无同名替换**，暂无 drop-in |
| `ChatMessageHistory` | `langchain_classic` | 改造后已不再使用（改用 checkpointer） |
| `TavilySearchResults` | `langchain_tavily.TavilySearch` | 返回类型不同，需同步改取值逻辑 |

### 8.3 Attu 图形界面的两个坑

用 `zilliz/attu`（Milvus 官方 Web 界面）时实测遇到：

| 现象 | 原因与解法 |
|---|---|
| 连接始终失败 | 地址不能填 `localhost`。Attu 是**服务端**连 Milvus，`localhost` 指向 Attu 容器自己。同一 Docker 网络内用容器名：地址填 `milvus-standalone:19530`；跨网络用 `host.docker.internal:19530`。另外 Milvus 未开鉴权时，认证方式要选「**没有**」而不是「密码」 |
| 集合列表正常，但点进 `schema` 页报「出了点问题」 | **Attu v3.0.1 前端 bug，非数据问题**。同一份 schema 用 SDK `describe_collection()` 读取完全正常 |

第二个问题已排除内存因素：四个容器 `restarts=0`、`OOMKilled=false`，Docker 配额
7.37 GB 而总占用仅约 520 MB（7%）。若真为内存不足，容器会被 kill 且重启次数大于 0。

**结论**：Attu 的集合列表、实体数量（`t_collection01`=102、`demo`=52）都可信；
schema 页加载失败时改用 SDK 查看，不要据此判断数据有问题。

### 8.4 `scratch/text_search.py` 是 WIP 文件（已隔离）

96 行的**个人实验稿**，不是生产代码。它尝试用不同方式建表/检索做对比实验，
**存在多处未定义名称**（`DataType`、`Function`、`FunctionType`、`Milvus`、
`bge_embedding`、`COLLECTION_NAME`、`res` 等），调用即报 `NameError`。

它原本在 `RAG_PROJECT/tools/text_search.py`，没有任何模块 import 它，
但会在 IDE 里持续产生 12 个真实报错、并干扰 `compileall`。
现已移至 **`scratch/text_search.py`**，并在 `pyrightconfig.json` 的 `exclude` 中排除。
实验内容保留，需要时可参考 `RAG_PROJECT/tools/retriever_tools.py` 的正式实现。

### 8.4a 静态检查（IDE）配置说明

项目以「包根」为导入起点（`from utils.env_utils import ...`），但 `RAG_PROJECT/`
没有 `__init__.py`，IDE 默认会把 `from utils.xxx` 判成 PEP 328 已废弃的
**隐式相对导入**，从而产生大量误报（实测 67 个）。

根目录的 **`pyrightconfig.json`** 已声明：

| 配置项 | 作用 |
|---|---|
| `extraPaths: ["./RAG_PROJECT"]` | 把包根声明为分析根，**消除 67 个隐式相对导入报错 + 7 个找不到模块** |
| `reportImplicitRelativeImport: none` | 同上，二者任一即可生效 |
| `venvPath` / `venv` | 指向 `.venv`，避免分析器用错解释器 |
| `exclude` | 排除 `.venv`、`scratch`、本地 Milvus 库文件等非生产内容 |
| 5 条 `report*` 设为 `none` | 见下 |

实测效果（basedpyright 1.40.2）：**error 117 → 0**。

被关掉的 5 条规则全部经过逐条核实，**均为类型严格度误报**，不是 bug：

| 规则 | 数量 | 说明 |
|---|---|---|
| `reportArgumentType` | 19 | 传给 SDK 的 dict 类型与注解不精确匹配，运行正常 |
| `reportAttributeAccessIssue` | 7 | `scored_result.binary_score` 静态推断成 `dict`，实际是 Pydantic 模型 |
| `reportCallIssue` | 2 | `OpenAIEmbeddings(openai_api_key=...)` 运行时**有效**，但 `langchain-openai` 类型存根用 `**kwargs` 未声明 |
| `reportOperatorIssue` | 2 | `milvus_db.py:67` 等处的存根类型差异 |
| `reportGeneralTypeIssues` | 1 | 泛型推断噪音 |

> 若某天需要重新启用其中某条，把对应的 `"规则名": "none"` 删掉即可。

### 8.5 `datas/output/` 的 187 个 JSON

它们是 `datas/layout-parser-paper.pdf` 的解析产物（Title 17 / NarrativeText 126 /
Table 2 / CodeSnippet 2，覆盖 12+ 页），**不是 Milvus 数据**，也没有任何代码消费它们。

- 它们带 `coordinates` / `PixelSpace` 元数据，是 Unstructured **hi_res** 策略的签名，
  而本项目从来只用 `strategy='fast'` → **来自项目外部**
- 仓库里**没有 PDF 解析脚本**，删掉无法用现有代码重建 → **保留**
- 已做的修正：把失效的 `metadata.source`（`E:\my_project\...`）改为相对路径
  `datas/layout-parser-paper.pdf`，`page_content` 零改动

### 8.6 硬编码的既有路径

`utils/env_utils.py` 里 `MILVUS_URI` 的**默认值**仍是旧公网地址（保留是为了行为不变）。
首次使用务必在 `.env` 中覆盖。

---

## 九、实测结论（真实环境）

环境：Docker Milvus 2.6.25 + DeepSeek `deepseek-flash` + 本地 BGE-small-zh + Tavily，
语料 102 块（7 个 md）。

### 9.1 三个入口全部跑通

| 入口 | 结果 |
|---|---|
| `graph2`（CRAG 完整链路） | 路由 / 检索 / 判分 / 生成 / 幻觉检查 / 答案判分 全通 |
| `graph1` | 6 节点 7 边；`SqliteSaver` 生效；多轮记忆注入正常（2 → 4 条消息） |
| `rag_agent` | 1.x 标准 `create_agent`（节点 `model`/`tools`）；多轮记忆正常 |
| `web_search` 分支 | Tavily 官方 key 下返回真实网页内容（约 3.1k 字符） |

### 9.2 问答质量（与 Milvus 官方文档核对一致）

```
Q: What index types does Milvus support?
A: HNSW（基于图，适合对搜索效率要求高的场景）、FLAT（百万级小数据集上追求完全
   精确搜索）、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN、DiskIndex ...

Q: Does Milvus support non-x86 architectures?
A: Milvus 不支持非 x86 架构…CPU 必须支持 SSE4.2、AVX、AVX2 或 AVX512 等 x86 专用
   SIMD 指令集。当前版本也不支持 Apple M1 CPU。

Q: How does Milvus handle index building?
A: 插入向量后自动建索引，但仅在新增向量填满一个 segment 时才建；阈值由
   rootcoord.minSegmentSizeToEnableIndex（默认 1024 行）控制，在此之前走暴力检索。
```

### 9.3 观察到的 CRAG 振荡路径（设计使然，非缺陷）

```
web_search -> generate -> transform_query -> retrieve -> grade_documents
-> transform_query -> retrieve -> grade_documents -> web_search -> generate
-> transform_query -> retrieve -> grade_documents -> web_search -> generate  ← 终止
```

问题在「向量库检索」与「网络搜索」之间来回切换，最终由 `MAX_GENERATION_RETRY`
熔断保证终止 —— **若无该熔断，此处会无限循环**。

### 9.4 踩过的坑（供后续参考）

| 现象 | 根因 | 解法 |
|---|---|---|
| `code=1100 unsupported rerank function: []` | Function reranker缺 `reranker` 键、多了 `input_field_names` | 见 4.1 |
| `nq [0] is invalid` | ① 检索器指向不存在的集合；② 入库未 flush | 见 4.8 / 4.10 |
| `This response_format type is unavailable now` | `with_structured_output` 1.x 默认改用 `json_schema` | 显式指定 `method` |
| `Thinking mode does not support this tool_choice` | DeepSeek thinking 与 `tool_choice` 冲突 | 关闭 thinking |
| `Insert missed an field 'title'` | 顶层正文块未被赋予 `title` | `_ensure_required_fields()` 兜底 |
| `JiebaAnalyzer requires the 'jieba' package` | 服务端未装 jieba | 装在 Milvus 容器里 |
| `TypeError: string indices must be integers` | Tavily 工具出错时返回**字符串** | `web_search_node` 兼容三种形态 |
| `Invalid token`（401） | 密钥与 `base_url` 不匹配 | 密钥须与服务商对应 |
| 检索结果不相关 / 拒答 | 工具描述写的是「半导体和芯片」而语料是 Milvus | 描述已改为与语料一致 |

---

## 十、依赖清单

完整列表见 `requirements.txt`（31 个精确 pin）与 `requirements-lock.txt`（全量 freeze）。
`requirements-legacy-freeze.txt` 是升级前的原始基线，用于回滚对照。

主要依赖：

```
langchain==1.4.3                langchain-core==1.6.6          langchain-openai==1.6.7
langgraph==1.2.12               langgraph-checkpoint==4.2.0langgraph-prebuilt==1.1.0
langchain-community==0.4.2      langchain-classic==1.0.8       langchain-experimental==0.4.2
langchain-milvus==0.4.0         pymilvus==3.0.2langchain-unstructured==1.0.1
langchain-huggingface==1.2.2    transformers==4.57.6          torch==2.7.1
sentence-transformers==5.7.0    unstructured==0.27.10         jieba==0.42.1
Markdown==3.11                   tavily-python==0.8.4          langgraph-checkpoint-sqlite==3.1.1
loguru==0.7.3pydantic==2.13.5                numpy==2.5.3                  openai==3.24.0
```

> `langchain-classic` 虽未被源码引用，但它是 `langchain-community 0.4.2` 的**硬依赖**，
> 删除会导致 `pip check` 报破损依赖，不可移除。

---

## 十一、配套文档

| 文档 | 内容 |
|---|---|
| `UPGRADE_NOTES.md` | 升级报告：版本对照、改动理由、硬约束清单、全部实测结论与踩坑记录 |
| `迁移验证报告.md` | 验证脚本自动生成的闸门结果快照（勿手工编辑） |
| `scripts/verify_migration.py` | 迁移验证脚本，可重复运行，不修改项目任何文件 |
| `scripts/milvus_probe.py` | 真实 Milvus 只读探测（版本/鉴权/集合/兼容性判断） |
| `scripts/e2e_smoke_test.py` | 端到端冒烟测试（建表→写入→检索→过滤→清理） |
