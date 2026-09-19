# 历史归档：LangChain 1.x 迁移

> **状态**：已完成的一次性工程，以下内容只作历史记录。
> **当前架构**见 [`../../PROJECT_DESCRIPTION.md`](../../PROJECT_DESCRIPTION.md)；
> **当前依赖版本**见 `requirements.txt`（后续阶段又升级过，本文表格是迁移时点）。
> **当前指标**见 [`../../README.md`](../../README.md)。

---

## 一、背景

项目的基础版写于 2025-04，技术栈停留在 LangChain 0.3.x / LangGraph 0.3.x。
2026-06 对基础版做了一次**只改生态、不改业务行为**的迁移：

* 保留：prompt 原文、Milvus schema 与索引参数、检索参数、两张图的拓扑与节点逻辑；
* 允许：import 路径、API 调用形式、依赖版本 pin、补缺失依赖；
* 目标：迁到 LangChain 1.x 生态，跑通全部入口与验证闸门。

---

## 二、结果摘要

| 项 | 结果 |
|---|---|
| 源码改动 | **11 个文件改动 + 1 个新增**（`utils/checkpointer.py`） |
| 未改动 | 快照中 36 个 `.py` 里 **25 个逐字节完全一致** |
| 行为回归 | **0 处**（3 个疑似点经新旧源码比对确认为既有缺陷） |
| 验证闸门 | **24 PASS / 0 WARN / 0 SKIP / 0 FAIL** |
| 修掉的既有缺陷 | 3 个：`filter` 从未生效、`web_search` 返回类型不匹配、generate 自环无熔断 |

一句话：**这是一次“换生态、保行为”的迁移，不是重写。**

---

## 三、版本对照（迁移时点）

### 主生态

| 包 | 升级前 | 升级后 |
|---|---|---|
| `langchain` | 0.3.23 | **1.4.3** |
| `langchain-core` | 0.3.51 | **1.6.6** |
| `langchain-openai` | 0.3.12 | **1.6.7** |
| `langgraph` | 0.3.30 | **1.2.12** |
| `langgraph-checkpoint` | 2.0.24 | **4.2.0** |
| `langgraph-prebuilt` | 0.1.8 | **1.1.0** |
| `langchain-community` | 0.3.21 | **0.4.2**（已停止维护） |
| `langchain-classic` | — | **1.0.8**（community 的硬依赖，保留） |

### 解析 / 向量库 / ML

| 包 | 升级前 | 升级后 |
|---|---|---|
| `langchain-milvus` | 0.1.9 | **0.4.0** |
| `pymilvus` | 2.5.6 | **3.0.2** |
| `unstructured` | 0.17.2 | **0.27.10** |
| `langchain-unstructured` | 0.1.6 | **1.0.1** |
| `langchain-experimental` | 0.3.4 | **0.4.2** |
| `langchain-huggingface` | 0.1.2 | **1.2.2** |
| `transformers` | 4.51.1 | **4.57.6** |
| `torch` | 2.6.0 | **2.7.1** |
| `sentence-transformers` | 4.0.2 | **5.7.0** |
| `openai` | 1.70.0 | **3.24.0** |
| `pydantic` | 2.11.2 | **2.13.5** |

> 当前 `requirements.txt` 又前进过（例如 `langchain-core 1.6.7`、
> `langgraph 1.2.14`、`unstructured 0.27.16`、`torch 2.14.1`）——
> 以它为准，本表只用于说明迁移跨度。

---

## 四、改了什么

1. **`agent/rag_agent.py`**：改用 1.x 标准 `create_agent()`，
   取代 `create_tool_calling_agent()` + `AgentExecutor()`；
   多轮记忆由 `RunnableWithMessageHistory` 改为 checkpointer + `thread_id`。
2. **`graph/graph1.py` / `graph2/graph_2.py`**：拆出 `create_graph()` 工厂与
   `main()` 入口，import 不再阻塞；`graph` 模块级导出保持原用法。
3. **配置外置**：`MILVUS_URI` / `COLLECTION_NAME` / LLM 的 base_url / model /
   temperature 从源码移到环境变量；新增 `utils/checkpointer.py`。
4. **路径配置化**：`documents/*` 里的硬编码绝对路径改为基于 `PKG_ROOT`。
5. **补依赖**：`tavily-python`、`Markdown`；移除废弃的 `dotenv` 包。

---

## 五、既有缺陷：3 个修复 + 1 个记录

迁移中确认了 4 个既有缺陷：3 个一并修复，`score_threshold` 只记录、未改行为。

### 5.1 `filter` 从未生效（影响最大）

`tools/retriever_tools.py` 原先传 `filter={"category": "content"}`，
但 `langchain-milvus` 的 hybrid 路径只认 `expr`，`filter` 被 `**kwargs` 吞掉；
`pymilvus` 的 `hybrid_search()` 也没有 `filter` 形参。

**修复**：改为 `expr='category == "content"'`。
这是唯一会改变召回结果的改动——纯标题块不再进入上下文。

### 5.2 `score_threshold` 从未生效

`search_type="similarity"` 路径不消费 `score_threshold`；
只有 `similarity_score_threshold` 才会真正过滤。

**结论**：保留现状并记录，未改 `search_type`。当前文档已把它标为惰性参数。

### 5.3 `web_search` 返回类型不匹配

`TavilySearchResults` 默认 `response_format="content_and_artifact"`，
`invoke()` 返回 `(list, raw)` 元组，而原代码按 list 遍历取 `d["content"]`，
会抛 `TypeError: list indices must be integers`。

**修复**：加元组解包 `results = docs[0] if isinstance(docs, tuple) else docs`。

### 5.4 generate 自环无熔断

`not supported → generate` 是无条件自环，幻觉检测持续失败会无限循环。

**修复**：新增 `generation_count` 与 `MAX_GENERATION_RETRY=2`；
达到上限改走 `not useful`（受 `transform_count` 保护）。

---

## 六、验证闸门

迁移脚本 `scripts/verify_migration.py` 覆盖 8 组 24 项：

| 闸门 | 内容 | 结果 |
|---|---|---|
| G1 | 全部 `.py` 语法编译 | PASS |
| G2 | AST 静态校验第三方符号 | PASS |
| G3（8 项） | Agent API / Tavily / Loader / Chunker / pymilvus 枚举 / Milvus 签名 / LC 核心 / HF | 全 PASS |
| G4 | elements metadata 键 + `merge_title_content` 产物 | PASS |
| G5 | `graph1`（6 节点含 START/END / 7 条分支）、`graph2`（7 节点含 START/END）拓扑 | PASS |
| G6 | 向量维度（当时 512）+ L2 归一化 | PASS |
| G7 | `create_agent` + 检查点 + 会话隔离 | PASS |
| G8 | 熔断返回值序列（1 / 2 / 7） | PASS |

合计：**24 PASS / 0 FAIL**。

> G6 的 512 维是迁移时点的状态；后续阶段已升级为 bge-large / 1024 维。
> 迁移时的 graph2 是“7 节点 / 9 边”口径（含 START/END），当前为
> 5 个显式节点 + 3 处条件边——**以当前代码为准**。

### 当时无法验证的项目

迁移环境没有可用的 Milvus / LLM / Tavily key，以下只做了构造与静态验证：

* Milvus 端到端读写；
* LLM 判分链与路由链的真实输出；
* Tavily 联网；
* HuggingFace 模型下载（当时用临时缓存离线完成 G6）。

这些在后续阶段都已补测（见 ROADMAP 的实测记录）。

---

## 七、回滚方式

1. 依赖：`pip install -r requirements-legacy-freeze.txt`；
2. 源码：使用本地保留的迁移前快照（不随公开仓库发布）；
3. 数据：迁移未改 schema / 索引参数 / 集合名，旧集合可直接复用。

---

## 八、留下的三条教训

1. **“升级后坏了”不等于“升级造成的”**：三个疑似回归经新旧源码逐行比对，
   全部是既有缺陷；先比对再下结论。
2. **配置必须外置**：硬编码的地址 / 模型 / 路径，换机即失败。
3. **Lite 通过 ≠ 真实服务通过**：集合名校验、Function reranker、文件锁，
   三者都是 Lite 与真实 Milvus 的差异点；服务端规则必须上真实服务验证。
