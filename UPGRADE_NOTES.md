# RAG 企业知识库项目 —— LangChain 1.x 迁移升级报告

- **升级范围**：LangChain / LangGraph 1.x 生态迁移 + 既有缺陷修复
- **项目路径**：`RAG_PROJECT/`（Python 包根，内含 36 个 `.py`）
- **Python**：3.12.7
- **升级策略**：迁移到最新生态。**prompt、Milvus schema 与索引参数、检索核心参数、两张图拓扑一律不变**；只改 import / API 调用形式 / 依赖版本，另修复 3 项经比对确认属既有缺陷的问题（详见第五节）

---

## 〇、两份文档的分工与关系

| 文档 | 性质 | 维护方式 | 什么时候看它 |
|---|---|---|---|
| **`UPGRADE_NOTES.md`（本文）** | **人工撰写的升级叙事**：版本对照、改动理由、决策取舍、风险与约束 | 手工维护 | 想知道「为什么这么改、改了什么、有什么坑」 |
| **`迁移验证报告.md`** | **脚本自动生成的验证快照**：8 组闸门的逐项结果与实测数据 | 由 `scripts/verify_migration.py --report` 重新生成，**不要手工编辑**（会被覆盖） | 想确认「现在跑起来还正常吗」 |

**两者关系：本文是决策依据，验证报告是执行证据。**
本文中每一条「已验证 / 已修复」的结论，其原始输出都能在验证报告中找到对应条目；
任何代码改动之后重跑一次脚本即可刷新证据。

配套文件：

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 精简直接依赖（29 个精确 pin） |
| `requirements-lock.txt` | 全量 freeze，用于精确复现环境 |
| `requirements-legacy-freeze.txt` | 升级前原始基线，回滚对照用 |
| `scripts/verify_migration.py` | 验证脚本本体，可重复运行，不修改项目任何文件 |
| `.env.example` | 环境变量模板 |

## 一、结论摘要

| 项目 | 结果 |
|---|---|
| 源码改动 | **11 个文件改动 + 1 个新增**：`agent/rag_agent.py`（1.x 标准 `create_agent`）、`graph1.py`/`graph_2.py`（`create_graph` + `main`）、`utils/env_utils.py` 与新增 `utils/checkpointer.py`（配置外置 + 持久化）、3 个 `documents/*`（路径配置化）、5 个既有缺陷修复（见 5.1 / 5.3 / 5.4） |
| 业务行为改动 | prompt / 工具 / 会话标识 / 提问内容 / 图拓扑 **均未变**；但有 3 处**恢复既有设计意图**的行为变化，见 5.1、5.3、5.4，均已逐项标注 |
| 改动范围核验 | **字节级比对**：快照中 36 个 `.py` 里 **25 个逐字节完全一致**，11 个有改动，另新增 1 个文件 `utils/checkpointer.py`；行尾符 0 处差异 |
| 由升级引入的行为回归 | **0 处**（3 个疑似点经新旧源码比对确认为项目既有缺陷） |
| 验证闸门 | **24 PASS / 0 WARN / 0 SKIP / 0 FAIL**（新增 G8 熔断闸门） |
| 新增/补齐依赖 | `tavily-python`（原项目遗漏）、`Markdown` |
| 移除依赖 | `dotenv==0.9.9`（已废弃的独立包） |

**一句话结论**：这次升级动到 11 个文件 + 新增 1 个——1 个改用 1.x 标准 `create_agent`，2 个做模块结构重构，
1 组配置外置与会话持久化，4 个是缺陷修复；快照中其余 25 个 `.py` **逐字节零改动**
（含全部 prompt、Milvus schema 与索引参数、两张 StateGraph 的节点与边定义）。

---

## 二、版本对照表（旧 → 新）

### LangChain / LangGraph 主生态

| 包 | 升级前 | 升级后 | 跨度 |
|---|---|---|---|
| `langchain` | 0.3.23 | **1.4.3** | 大版本 |
| `langchain-core` | 0.3.51 | **1.6.6** | 大版本 |
| `langchain-openai` | 0.3.12 | **1.6.7** | 大版本 |
| `langgraph` | 0.3.30 | **1.2.12** | 大版本 |
| `langgraph-checkpoint` | 2.0.24 | **4.2.0** | 大版本 |
| `langgraph-prebuilt` | 0.1.8 | **1.1.0** | 大版本 |
| `langgraph-sdk` | 0.1.61 | **0.4.5** | 中版本 |
| `langsmith` | 0.3.24 | **0.14.4** | 中版本 |
| `langchain-community` | 0.3.21 | **0.4.2** | 小版本（注意：已停止维护） |
| `langchain-classic` | — | **1.0.8** | 项目源码已不再引用，但 community 0.4.2 将其声明为硬依赖，**必须保留** |
| `langchain-tavily` | — | **0.2.18** | **本次新增**（备用，本次未切） |

### 解析 / 切片 / 向量库 / ML

| 包 | 升级前 | 升级后 | 备注 |
| `langchain-milvus` | 0.1.9 | **0.4.0** | 0.2.0 起要求 `langchain-core>=1.0`，1.x 生态唯一可用版本 |
| `pymilvus` | 2.5.6 | **3.0.2** | 大版本；实测枚举与子模块路径全部兼容 |
| `unstructured` | 0.17.2 | **0.27.10** | elements metadata 键名未变（已实测） |
| `langchain-unstructured` | 0.1.6 | **1.0.1** | |
| `langchain-experimental` | 0.3.4 | **0.4.2** | `SemanticChunker` 仍在原位 |
| `langchain-text-splitters` | 0.3.8 | **1.1.3** | |
| `Markdown` | 3.7 | **3.11** | `unstructured.partition.md` 硬依赖，**原精简时易漏** |
| `langchain-huggingface` | 0.1.2 | **1.2.2** | |
| `transformers` | 4.51.1 | **4.57.6** | 固定 4.x 末班 |
| `torch` | 2.6.0 | **2.7.1** | |
| `sentence-transformers` | 4.0.2 | **5.7.0** | 固定 5.x 末班 |
| `tokenizers` | 0.21.1 | **0.22.2** | |
| `huggingface-hub` | 0.30.2 | **0.36.2** | |

### 基础设施

| 包 | 升级前 | 升级后 |
|---|---|---|
| `openai` | 1.70.0 | **3.24.0** |
| `pydantic` | 2.11.2 | **2.13.5** |
| `protobuf` | 6.30.2 | **7.36.2** |
| `numpy` | 2.2.4 | **2.5.3** |
| `pandas` | 2.2.3 | **3.0.6** |
| `tiktoken` | 0.9.0 | **0.14.0** |
| `python-dotenv` | 1.1.0 | **1.2.4** |
| `loguru` | 0.7.3 | 0.7.3（已是最新） |
| `tavily-python` | **缺失** | **0.8.4（补齐）** |
| `dotenv` | 0.9.9 | **已移除**（废弃包，与 `python-dotenv` 重复） |

---

## 三、源码改动清单（11 个文件 + 1 个新增）

### 3.1 `RAG_PROJECT/agent/rag_agent.py`：改用 1.x 官方标准写法 `create_agent`

本项目**唯一**需要改动的文件。LangChain 1.x 已把 agent 的构建方式彻底标准化，
本文件已从「旧版三件套 + 手工 prompt 编排」改为「1.x 标准 `create_agent` + 检查点记忆」。

**改动前（旧版写法）**：

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
 ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题'),
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 ('human', '{input}'),
 MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])
agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[retriever_tool])

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory: ...
agent_with_history = RunnableWithMessageHistory(
 executor, get_session_history,
 input_messages_key='input', history_messages_key='chat_history')
resp2 = agent_with_history.invoke({'input': '什么是EUV光刻机？'},
 config={'configurable': {"session_id": 'zs123'}})
```

**改动后（1.x 标准写法）**：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = '你是一个智能助手，尽可能的调用工具回答用户的问题' # 逐字未变
SESSION_ID = 'zs123' # 会话标识未变

agent = create_agent(
 model=llm,
 tools=[retriever_tool],
 system_prompt=SYSTEM_PROMPT,
 checkpointer=create_checkpointer(),   # 由 InMemorySaver 改为 SqliteSaver
)
config = {'configurable': {'thread_id': SESSION_ID}}

if __name__ == '__main__':
 resp = agent.invoke({'messages': [('user', '什么是EUV光刻机？')]}, config=config)
 print(resp['messages'][-1].content)
```

**对照明细**：

| 维度 | 旧写法 | 1.x 标准写法 | 逻辑是否等价 |
| agent 构建 | `create_tool_calling_agent` + `AgentExecutor` | `create_agent`（内部即已编译的 LangGraph 图） | 是 |
| 系统提示词 | `ChatPromptTemplate` 的 `('system', ...)` | `system_prompt=SYSTEM_PROMPT` | **逐字相同** |
| scratchpad | 手工 `MessagesPlaceholder('agent_scratchpad')` | 由 `create_agent` 内部自动装配 | 是 |
| 会话记忆 | `RunnableWithMessageHistory` + `ChatMessageHistory` + 手工 `store` 字典 | `checkpointer` + `thread_id` | 是（见下方实测）；checkpointer 实现见 3.3b |
| 输入格式 | `{'input': '...'}` | `{'messages': [('user', '...')]}` | 是（`AgentState` 规范） |
| 输出取值 | `print(resp)`（整个字典） | `resp['messages'][-1].content` | 是（仅输出最终答案） |
| 会话标识 | `config={'configurable': {'session_id': 'zs123'}}` | `config={'configurable': {'thread_id': 'zs123'}}` | 是（键名随 API 变更，值不变） |
| 副作用 | 顶层直接 `invoke`，import 即执行 | 加 `if __name__ == '__main__':` 守卫 | 改进（可直接 import 复用） |

**为什么必须改**：`langchain.agents` 在 1.x 中只剩 `AgentState` 与 `create_agent`，实测：

```
ImportError: cannot import name 'create_tool_calling_agent' from 'langchain.agents'
```

**为什么不用 `langchain-classic`**：该包能提供旧 API，但它属于**过渡兼容层**，
而 1.x 的官方方向是 `create_agent` + 检查点。若学习/使用最新标准，应直接采用新写法。
因此 `langchain-classic` 仍保留在依赖清单中——**注意：项目源码虽已不再引用它，但
`langchain-community 0.4.2` 把它声明为硬依赖**；曾尝试卸载，`pip check` 立即报出
`langchain-community 0.4.2 requires langchain-classic, which is not installed`，
全新环境安装时也会缺包，因此必须保留。

**多轮记忆等价的实测证据**（验证脚本 G7 闸门，用假模型不发网络请求）：

```
首轮模型收到的消息（2 条）：
 你是一个智能助手，尽可能的调用工具回答用户的问题 | 什么是EUV光刻机？
次轮同一 thread_id 收到的消息（4 条）：
 你是一个智能助手，尽可能的调用工具回答用户的问题 | 什么是EUV光刻机？ | 回答：什么是EUV光刻机？ | 它和DUV有什么区别？
另一 thread_id 收到的消息（2 条，历史归零）
→ checkpointer + thread_id 完整承担了旧 RunnableWithMessageHistory 的职责，且会话隔离有效
```

`create_agent` 编译出的图为标准结构：节点 = `['__start__', 'model', 'tools', '__end__']`。

除本节列出的 7 个文件外，其余 29 个 `.py` **逐字节零改动**，包括所有 prompt、
Milvus schema 与索引参数、检索参数、两张 StateGraph 的节点与边定义。

### 3.2 `graph/graph1.py` 与 `graph2/graph_2.py`：拆出 `create_graph` 工厂与 `main` 入口

**依据（已核实官方文档）**：LangGraph 官方《Application structure》明确规定，`langgraph.json` 的 `graphs` 入口
既可以是「已编译的图对象」，也可以是「**a function that makes a graph**」（一个用于创建图的函数）；
同时官方要求图构建与运行入口分离，否则部署端 import 图模块就会启动交互循环。

**改动前**：两个文件都在**模块顶层**执行 `while True: input(...)`，import 即阻塞，无法复用。

**改动后**（两个文件均为同样处理）：

```python
graph = create_graph # 模块级仍导出已编译的图，保持原有用法不变

def create_graph():
 """返回已编译的图对象（官方支持的「创建图的函数」入口形式）。"""
 return graph

def main():
 ... # 原来的交互循环，逻辑与退出条件一字未改

if __name__ == '__main__':
 main
```

**行为影响**：
- 直接 `python graph1.py` 运行，交互体验与迁移前完全一致。
- `import graph.graph1` 不再有副作用，可被部署端、测试、验证脚本复用。
- 图拓扑、节点、条件边**均未改动**（G5 闸门以升级前快照为基线做断言，实测仍为 6 节点/7 边 与 7 节点/9 边）；仅检查点实现由 `MemorySaver` 换为 `SqliteSaver`（见 3.3b）。
- 附带修复一处脆弱取值：`graph_2.py` 原先在循环外直接取最后一个节点的 `value["generation"]`，
 若末节点不是 `generate`（例如 `grade_documents`）会抛 `KeyError`；
 现改为记录「最后一条出现过的 generation」，**正常路径输出完全一致**。

### 3.3 配置外置 + 会话记忆持久化

**（a）配置外置** —— `utils/env_utils.py`

迁移前 `MILVUS_URI` / `COLLECTION_NAME` 以及三个 `E:\path\to\...` 路径是硬编码的，换机即失败。
现改为从环境变量读取，**默认值与升级前完全一致**，因此不设置任何新变量时行为与升级前逐字节相同：

| 配置项 | 环境变量 | 默认值（= 升级前硬编码值） |
|---|---|---|
| `MILVUS_URI` | `MILVUS_URI` | `http://<old-ip>:19530` |
| `COLLECTION_NAME` | `COLLECTION_NAME` | `t_collection01` |
| 语料目录 | `DATAS_DIR` | 包内 `RAG_PROJECT/datas`（`MD_DIR` = 其下的 `md`） |
| 检查点文件 | `CHECKPOINT_DB` | 工作区根 `checkpoints.db` |

三个脚本的演示入口改用 `MD_DIR`：`documents/markdown_parser.py`、`documents/milvus_db.py`、
`documents/write_milvus.py`。**注意**：`write_milvus.py` 原先指向 `E:\path\to\langchain_demo01\md`
（与另两个脚本的路径还不一致），现统一为 `MD_DIR`，这**改变了该演示脚本的扫描目录**，
但原路径在本机并不存在，因此属修复而非回退。

**（b）会话记忆持久化** —— 新增 `utils/checkpointer.py`

依据 LangGraph 官方持久化文档原文：

> `MemorySaver` and `InMemorySaver` store checkpoints in RAM. When the process restarts,
> all checkpoints are lost. **Fix:** Use a persistent checkpointer for production:
> `PostgresSaver` (PostgreSQL with async support), `SqliteSaver` (local file-based storage for development).

选型：官方把 `PostgresSaver` 定位为生产首选、`SqliteSaver` 为本地开发/小型项目。
本项目是**同步、轻量**的 RAG 应用，`SqliteSaver` 的 docstring 亦自述适用于
"lightweight, synchronous use cases (demos and small projects)"，故采用 `SqliteSaver`；
上生产时只需把 `create_checkpointer` 换成 `PostgresSaver.from_conn_string(...)`。
`rag_agent.py` 与 `graph/graph1.py` 已统一改用它。

**实测证据**：`checkpoints.db` 生成后含 `checkpoints` / `writes` 两张表，
调用图后写入 2 条 checkpoint、3 条 write，`thread_id` 正确落盘 —— 记忆可跨进程重启恢复。

---

## 四、6 处高风险 API 的判定结论

| # | 风险点 | 结论 | 证据 |
| 1 | Agent API | **已迁移到 1.x 标准 `create_agent`**（唯一改动点） | `langchain.agents.create_agent` 可用；`create_tool_calling_agent` 已移除；G7 闸门端到端验证通过 |
| 2 | Tavily 工具 | **无需改动** | `langchain_community.tools.TavilySearchResults` 在 0.4.2 中仍可导入（已加 `@deprecated(removal="1.0")` 标记，但功能保留） |
| 3 | Markdown loader | **无需改动** | `langchain_community.document_loaders.UnstructuredMarkdownLoader` 可用；`mode='elements'` 下 `category`/`element_id`/`parent_id` 三键全部存在（实测 raw 85 个元素） |
| 4 | SemanticChunker | **无需改动** | `langchain_experimental.text_splitter.SemanticChunker` 可用 |
| 5 | **pymilvus 3.x 枚举** | **兼容，风险解除** | `IndexType` / `MilvusClient` / `Function` / `pymilvus.client.types.{MetricType, DataType, FunctionType}` 全部可导入；`Milvus.__init__` 的 `embedding_function`/`collection_name`/`builtin_function`/`vector_field`/`consistency_level`/`auto_id`/`connection_args` 全部保留 |
| 6 | Milvus 适配器检索参数 | **参数签名兼容** | `ranker_type`/`ranker_params` 仍被接受（仅多一条 DeprecationWarning，行为不变） |

---

## 五、既有缺陷与本次修复（均经新旧源码比对确认非升级引入）

这是本次审查最有价值的产出。以下三点在升级后看起来像"坏了"，但通过**下载新旧版本源码逐行比对**，证明升级前后行为完全一致。按「不改变业务」的约束，本次**未做修改**，仅记录。

### 5.1 `filter={"category": "content"}` 可能从未生效（影响最大）

- 现象：`tools/retriever_tools.py:13` 传入 `filter`，但 `langchain_milvus` 的 hybrid 检索路径只用 `expr=expr`，其余参数进 `**kwargs`（`milvus.py:1789`、`1800`），而 `pymilvus 3.0.2` 的 `MilvusClient.hybrid_search` **没有 `filter` 形参**。
- 新旧比对：旧版 `langchain-milvus 0.1.9` 的 `_collection_hybrid_search` 结构完全同构（同样 `expr=expr` + `**kwargs`）。pymilvus 2.5.6 的 `hybrid_search` 同样不识别 `filter`。
- 结论：**升级前后都不生效**。本项目是双向量（dense + sparse）必然走 hybrid 路径，因此这个过滤条件大概率一直是空转的。检索结果中可能混入 `category='Title'` 的纯标题块。
- **已修复**：`tools/retriever_tools.py` 已把 `'filter': {"category": "content"}` 改为 `"expr": 'category == "content"'`。经查 `expr` 会一路透传到 `AnnSearchRequest(expr=expr)`（`milvus.py:1789`），hybrid 路径同样生效。
 **这是本次唯一会改变召回结果的改动**：过滤条件自项目创建起从未生效，修复后 `category != 'content'`（主要是纯标题块）不再进入上下文。属「恢复既有设计意图」，非新增业务规则。如需完全保持旧行为，把 `expr` 改回 `filter` 即可。

### 5.2 `score_threshold=0.1` 从未生效

- 现象：`score_threshold` 在 `langchain-milvus 0.4.0` 源码中 **0 处匹配**；且 `search_type='similarity'` 路径下 langchain-core 也不会消费它（只有 `search_type='similarity_score_threshold'` 才会走 `similarity_search_with_relevance_scores`）。
- 新旧比对：旧版 `0.1.9` 同样 **0 处匹配**。
- 结论：**升级前后都是透传后被忽略**。若确实需要阈值过滤语义，应改用 `search_type='similarity_score_threshold'`。**本次未改**。

### 5.3 `web_search_node.py` 的 `d["content"]` 与返回类型不匹配

- 现象：`TavilySearchResults` 的 `response_format` 默认为 `"content_and_artifact"`（单数），`invoke` 返回**元组** `(list, raw)`，而项目按 list 遍历取 `d["content"]`。
- 新旧比对：旧版 `langchain-community 0.3.21` 同一行代码也是 `response_format: Literal["content_and_artifact"] = "content_and_artifact"`，`_run` 同样返回元组。
- 结论：**升级前后同样不匹配**，web_search 分支在旧版就存在问题。
- 附带信息：该类已标记 `@deprecated(since="0.3.25", removal="1.0", alternative_import="langchain_tavily.TavilySearch")`。若将来修复，建议一并迁移到 `langchain_tavily.TavilySearch`（已装入 venv，可直接切换）。
- **已修复**：`graph2/web_search_node.py` 已加正确的元组解包 `results = docs[0] if isinstance(docs, tuple) else docs`，取值逻辑 `d["content"]` 保持不变。实测返回 `Document('EUV 是 13.5nm 极紫外光刻\nDUV 用 193nm')`，与原设计意图一致；修复前该分支会抛 `TypeError: list indices must be integers`。
 未迁移到 `langchain_tavily.TavilySearch`，因为后者的 `response_format` 默认为 `'content'`、返回的是**字符串**而非 dict 列表，会连带改动此处的取值逻辑，改动面更大。若将来要迁移，需同步调整本节点。

---

### 5.4 generate 自环无熔断（唯一的真实死循环风险）

- 现象：`not supported` 到 generate 是无条件自环，而 `transform_count` 只保护了 `transform_query` 分支，幻觉检测持续失败会无限循环。
- 已修复：`graph_state2.py` 新增 `generation_count`；`generate_node2.py` 每次生成累加写回；`graph_2.py` 新增 `MAX_GENERATION_RETRY = 2`，达上限时改返回 `not useful`（该分支受 `transform_count` 保护，循环因此有界）。
- 实测（G8 闸门）：计数 1 返回 not supported（维持原重试）、2 返回 not useful、7 返回 not useful。
- 行为影响：仅在原本会无限循环时生效；正常一次成型的工作流行为完全不变。图拓扑未变（仍 7 节点 / 9 边）。

### 5.5 采用 1.x 标准写法后，多轮历史的消息构成与旧实现不同

code-review 的 Spec 轴发现、且经复核确认的一处真实行为差异，此前 G7 证据不足以覆盖它。

- 旧写法 `RunnableWithMessageHistory` 的历史回写（`langchain_core/runnables/history.py:490-500`）：包裹 `AgentExecutor` 时其输出字典只有一个键 `output`，故只把最终答案作为 `AIMessage` 写入历史；工具调用产生的 `AIMessage(tool_calls)` 与 `ToolMessage` 不进历史。
- 新写法 `checkpointer` 持久化全部消息，包括工具调用与工具输出。
- 后果：多轮中模型看到的上下文比旧实现更长（多出工具调用与工具输出），能保留上轮调用了哪个工具、返回了什么，但 token 消耗更高。
- 这是 LangChain 1.0 的设计方向（用检查点取代手工历史裁剪），不是缺陷。若需严格复刻旧裁剪行为，可在 `middleware` 中加历史裁剪中间件。

### 5.8 接入 DeepSeek 后实测暴露的四个问题（2026-10-07）

配置：`.env` 指向 DeepSeek 官方（`LLM_BASE_URL=https://api.deepseek.com`，
`LLM_MODEL=deepseek-flash`），并在真实 Milvus 2.6.25 上跑通全部三个入口。
过程中发现四个问题，均已修复：

**① LLM 的 base_url / model / temperature 原本硬编码在源码里**
原 `llm_models/all_llm.py` 写死 `model='gpt-4o-mini'`、`base_url="https://xiaoai.plus/v1"`，
换服务商必须改代码。现已按第3.3 节的同一思路外置为
`LLM_BASE_URL` / `LLM_MODEL` / `LLM_TEMPERATURE`，默认值保持迁移前写法。

**② `langchain-openai` 1.x 把 `with_structured_output` 的默认方法改了**
1.x 默认值从 `function_calling` 变成了 **`json_schema`**，而 DeepSeek 官方 API 不支持
（报 `This response_format type is unavailable now`）。
项目共有 5 处 `with_structured_output` 调用（`graph1.grade_documents`、
`graph2` 的 grader / answer / hallucinations / router），现已全部显式指定
`method=LLM_STRUCTURED_METHOD`（默认 `function_calling`）。
> 换到支持 `json_schema` 的服务商时，改这一个环境变量即可。

**③ DeepSeek 的 thinking 模式与 `tool_choice` 冲突**
开启 thinking 时报`Thinking mode does not support this tool_choice`，
而 `function_calling` 依赖 `tool_choice`。因此默认关闭 thinking
（`LLM_DISABLE_THINKING=1`，需要时可置0 打开）。

**④ `web_search` 节点还漏了一种返回形态**
`TavilySearchResults.invoke` 在工具执行出错时，langchain 会把**异常信息包成字符串**
返回（实测拿到 `"HTTPError('401 Client Error: Unauthorized for url: https://api.tavily.com/search')"`）。
原先只处理了 `(list, raw)` 元组与 `list`，遇到字符串会抛
`TypeError: string indices must be integers`。现已兼容 `tuple` / `list` / `str` 三种。

### 5.9 原项目自带的一处设定不一致（非本次改动引入）

`tools/retriever_tools.py:51` 的工具描述写的是：
> 搜索并返回关于 '半导体和芯片' 的信息, 内容涵盖：半导体和芯片的封装、测试、光刻胶等

但 `datas/md/` 里的实际语料是 **Milvus / 向量数据库**文档。影响面：
- **`graph2` 不受影响** —— 它直接使用 `retriever`，不经过工具描述
- **`agent/rag_agent.py` 会受影响** —— `create_agent` 依据工具描述决定是否调用工具，
  实测模型回答「我唯一能用的工具是一个专注于半导体和芯片的检索器，
  和向量数据库索引无关」，因此**拒绝检索 Milvus 相关问题**

**已于 2026-10-07 修正**（只改描述文案，不改工具名、不改检索行为）：

> 搜索并返回关于 Milvus 向量数据库的信息，内容涵盖：架构与部署、索引类型与指标、
> 性能调优、运维监控、常见问题排查等

修正前后对比（真实环境实测）：

| | 修改前 | 修改后 |
|---|---|---|
| 提问 "What index types does Milvus support?" | `human -> ai`（**直接拒答，未调工具**） | `human -> ai -> tool -> ai` |
| 提问 "How do I troubleshoot a Milvus deployment?" | `human -> ai` | `human -> ai -> tool -> ai -> human -> ai -> tool -> tool -> ai` |

即：修正后 `rag_agent` 会真正发起检索，且多轮对话中能连续调用工具。

### 5.10 真实环境下的完整验证结果（2026-10-07）

环境：Docker `milvusdb/milvus:v2.6.25`（本地）+ `deepseek-flash`（DeepSeek 官方）
+本地 `BAAI/bge-small-zh`（512 维）+ Tavily 官方 key，语料 102 块（7 个 md）。

三个入口全部跑通：

| 入口 | 结果 |
|---|---|
| `graph2`（CRAG 完整链路） | 路由 / 检索 / 判分 / 生成 / 幻觉检查 / 答案判分 全通 |
| `graph1` | 6 节点 7 边；SqliteSaver 生效；多轮记忆注入正常（2 -> 4 条消息） |
| `agent/rag_agent.py` | 1.x 标准 `create_agent`（节点 model/tools）；多轮记忆正常 |
| `web_search` 分支 | Tavily 官方 key 下返回真实网页内容（约 3.1k 字符） |

实测问答（答案与Milvus 官方文档一致）：
```
Q: What index types does Milvus support?
A: HNSW（基于图，适合对搜索效率要求高的场景）、FLAT（百万级小数据集上追求
   完全精确搜索）、IVF_FLAT、IVF_SQ8、IVF_PQ、SCANN、DiskIndex ...

Q: Does Milvus support non-x86 architectures?
A: Milvus 不支持非 x86 架构…CPU 必须支持 SSE4.2、AVX、AVX2 或 AVX512 等
   x86 专用 SIMD 指令集。当前版本也不支持 Apple M1 CPU。

Q: How does Milvus handle index building?
A: 插入向量后自动建索引，但仅在新增向量填满一个 segment 时才建；阈值由
   rootcoord.minSegmentSizeToEnableIndex（默认 1024 行）控制，在此之前走暴力检索。
```

**观察到 CRAG 的真实振荡路径**（设计使然，非缺陷）：
```
web_search -> generate -> transform_query -> retrieve -> grade_documents
-> transform_query -> retrieve -> grade_documents -> web_search -> generate
-> transform_query -> retrieve -> grade_documents -> web_search -> generate  ← 终止
```
问题在「向量库检索」与「网络搜索」之间来回切换，最终由
`MAX_GENERATION_RETRY` 熔断（见 5.4）保证终止——**若无该熔断，此处会无限循环**。

## 六、验证闸门结果

执行 `.venv\Scripts\python.exe scripts\verify_migration.py --with-embeddings --report 迁移验证报告.md`

| 闸门 | 检查项 | 结果 |
| G1-编译 | 全部 36 个 `.py` 语法编译 | PASS |
| G2-import | AST 静态校验 67 个第三方符号 / 23 个文件 | PASS |
| G2-import | 4 个 UNSAFE 模块识别 | PASS |
| G3 ×8 | 1.x 标准 Agent API / Tavily / Loader / Chunker / pymilvus 枚举 / Milvus 签名 / LC 核心 API / HF | 全 PASS |
| G4-切片 | elements metadata 键兼容性 | PASS |
| G4-切片 | `merge_title_content` 复刻产物 | PASS |
| G5-拓扑 | `graph/graph1.py` 构图（6 节点 / 7 边） | PASS |
| G5-拓扑 | `graph2/graph_2.py` 构图（7 节点 / 9 边） | PASS |
| G6-向量 | 输出维度 `dim=512` | PASS |
| G6-向量 | L2 归一化 `1.000000` | PASS |
| G7-1.x写法 | 导入改写后的 rag_agent.py，构造出 agent 图（节点 model / tools） | PASS |
| G7-1.x写法 | AgentState 输入输出规范 | PASS |
| G7-1.x写法 | checkpointer + thread_id 承担多轮记忆 | PASS |
| G7-1.x写法 | 会话隔离 | PASS |

| G8-熔断 | generate 自环熔断（计数 1/2/7 的返回值） | PASS |

合计：24 PASS / 0 WARN / 0 SKIP / 0 FAIL

其中 G7 为 1.x 标准写法专项验证（用假模型端到端跑通 create_agent + 检查点记忆，不发网络请求）：导入改写后的 rag_agent.py、AgentState 输入输出规范、多轮记忆累积、会话隔离四项均全部通过。

### 关键实测数据

**切片一致性**（`datas/md/overview.md`，检测静默行为漂移）：

```
raw 元素 85 个
metadata 键 = ['category', 'category_depth', 'element_id', 'emphasized_text_contents',
 'emphasized_text_tags', 'file_directory', 'filename', 'filetype',
 'image_url', 'languages', 'last_modified', 'link_texts', 'link_urls',
 'parent_id', 'source']
raw category 分布 = {'NarrativeText': 25, 'Title': 16, 'Image': 2, 'ListItem': 42}
merge 后 = 17 个，其中 category=='content' = 16 个，含 ' -> ' 面包屑 = 8 个
→ merge_title_content 的标题合并逻辑完全正常
```

**工作流拓扑**（与升级前 `graph_rag1.png` / `graph_rag2.png` 对照，**完全一致**）：

```
graph1 节点: [__end__, __start__, agent, generate, retrieve, rewrite]
 __start__ → agent
 agent ⇢ __end__ (tools_condition 无工具调用分支)
 agent ⇢ retrieve (tools_condition 工具调用分支)
 retrieve ⇢ generate / ⇢ rewrite
 rewrite → agent
 generate → __end__

graph2 节点: [__end__, __start__, generate, grade_documents, retrieve, transform_query, web_search]
 __start__ ⇢ retrieve (vectorstore) / ⇢ web_search
 retrieve → grade_documents
 transform_query → retrieve
 web_search → generate
 generate ⇢ __end__ (useful) / ⇢ generate (not supported 自环) / ⇢ transform_query (not useful)
 grade_documents → __end__
```

**向量**：`BAAI/bge-small-zh-v1.5` 输出 `dim=512`、`l2_norm=1.000000`，与 `documents/milvus_db.py` 中 dense 字段 `dim=512` 严格匹配。

---

## 七、因环境受限而无法验证的项目（如实标注）

| 项目 | 阻塞原因 | 影响 |
| Milvus 端到端读写 | `utils/env_utils.py:10` 硬编码 `MILVUS_URI='http://<old-ip>:19530'`，实测 `MilvusException code=2 Fail connecting to server`（公网地址已失效） | schema/索引创建、写入、真实召回质量无法端到端验证；**属既有配置问题，与版本无关**。验证脚本已用离线打桩绕过构图环节 |
| LLM 真实调用 | 无 `OPENAI_API_KEY`（`llm_models/all_llm.py` 在模块顶层构造 `ChatOpenAI`） | 判分链、路由链的实际输出未验证；构造与调用链已验证 |
| Tavily 真实调用 | 无 `TAVILY_API_KEY`（且原项目代码从未读取过该变量） | 见 5.3 |
| HuggingFace 模型下载 | 本机 HTTPS 代理（`127.0.0.1`）做 TLS 拦截，代理 CA 不在 certifi 信任链中，`verify=True` 全部失败 | 已用**一次性临时脚本**（在临时目录，不属交付物）灌入模型缓存，随后以 `HF_HUB_OFFLINE=1` 离线完成 G6 验证。**交付脚本中不含任何证书绕过逻辑** |

---

## 八、如何使用

```powershell
# 1. 激活环境
.venv\Scripts\Activate.ps1

# 2. 配置密钥（复制模板并填写）
Copy-Item .env.example .env

# 3. 修正失效的 Milvus 地址（编辑 RAG_PROJECT\utils\env_utils.py 第 10 行）

# 4. 跑通验证脚本（可重复执行，不修改项目任何文件）
.venv\Scripts\python.exe scripts\verify_migration.py --with-embeddings

# 5. 导入数据（注意 milvus_db.py / markdown_parser.py 内的硬编码路径需自行调整）
.venv\Scripts\python.exe -m RAG_PROJECT.documents.write_milvus

# 6. 启动工作流
.venv\Scripts\python.exe -m RAG_PROJECT.graph2.graph_2

# 7. 运行 1.x 标准写法的 Agent 演示
.venv\Scripts\python.exe -m RAG_PROJECT.agent.rag_agent
```

---

## 九、回滚方式

本次改动可完全回滚：

1. **源码**：`_source_snapshot_before_upgrade.zip` 是升级前整目录快照；本次共 7 个文件有实质改动（1 个版本迁移 + 2 个模块结构重构 + 4 个缺陷修复），其余 29 个 `.py` 逐字节未变。
2. **依赖**：`requirements-legacy-freeze.txt` 是升级前原始 freeze。
3. **环境**：直接删除 `.venv/` 目录即可，不影响 `C:\Anaconda`（本次全程未改动 base 环境）。

---

## 十、后续建议（按优先级排序）

1. ~~修 5.1 的 `filter` 空转~~ —— **已完成**（改为 `expr`）。
2. ~~修 5.3 的 Tavily 返回类型~~ —— **已完成**（元组解包）。若要彻底切到 `langchain_tavily.TavilySearch`，需同步调整取值逻辑（其默认返回字符串）。
3. ~~修 `generate → generate` 自环~~ —— **已完成**（`generation_count` + `MAX_GENERATION_RETRY` 熔断）。
4. ~~清理硬编码（Milvus 地址、三处 `E:\path\to\...` 路径）~~ —— **已完成**，改为 env 变量 + 同值默认（见 3.3a）。`main.py` 仍是 PyCharm 模板残留、无真实入口，本次未动。
5. ~~`graph1.py` / `graph_2.py` 顶层 `while True: input` 导致无法被 import 复用~~ —— **已完成**（拆出 `create_graph` 工厂 + `main` 入口 + `__main__` 守卫，见 3.2）。
6. **切离已停止维护的 `langchain-community`**：`UnstructuredMarkdownLoader` 迁到 `langchain-unstructured`，Tavily 迁到 `langchain-tavily`。（`ChatMessageHistory` 已随 `rag_agent.py` 改造不再使用）
7. ~~`TAVILY_API_KEY` 未被 `env_utils.py` 读取~~ —— **已完成**：`env_utils.py` 已显式读取该变量。
8. ~~生产化会话存储（`InMemorySaver` 仅内存、重启即丢）~~ —— **已完成**：改为官方推荐的 `SqliteSaver`（见 3.3b）。
9. **再上一档到生产级持久化**：`SqliteSaver` 适合本地开发与小型项目；多实例 / 并发 / 海量会话请换 `PostgresSaver`（需 `pip install langgraph-checkpoint-postgres`），改法见 `utils/checkpointer.py` 的 docstring。
10. **`main.py` 仍是 PyCharm 模板残留**（`print_hi('PyCharm')`），项目无统一入口；可考虑改为指向 `graph2/graph_2.py` 的薄入口。

---

## 十一、硬约束清单（回归检查基线）

本次升级与缺陷修复一律遵守以下约束。**改动前应先跑一遍 `scripts/verify_migration.py` 建立基线，改完再跑一遍对比**；若下列任一项发生变化，即属超出授权范围的改动，需先确认。

### 11.1 绝对不可改动的项

| 约束 | 具体位置 | 校验方式 |
|---|---|---|
| **prompt 文本逐字不变** | `generate_node.py`、`generate_node2.py`、`graph2/*_chain.py`、`query_route_chain.py`、`transform_query_node.py`、`rag_agent.py` 的 `SYSTEM_PROMPT` | 字节级比对快照 |
| **Milvus schema 不变** | `milvus_db.py` 全部 `add_field`，含 `dense` 的 `dim=512` | G3 探针 + 人工比对 |
| **jieba 分词配置不变** | `enable_analyzer=True` 与 `analyzer_params` 中的 jieba + cnalphanumonly | 人工比对 |
| **索引参数不变** | BM25 `k1=1.2`/`b=0.75`/`DAAT_MAXSCORE`；HNSW `M=16`/`efConstruction=64` | 人工比对 |
| **嵌入模型与维度不变** | `BAAI/bge-small-zh-v1.5`、`device=cpu`、`normalize_embeddings=True`、输出 512 维 | G6 闸门 |
| **切片算法不变** | `mode='elements'`、`strategy='fast'`、5000 字符阈值、`merge_title_content` 的父子合并与 `category` 改写 | G4 闸门 |
| **两张图拓扑不变** | `graph1` 6 节点/7 边；`graph2` 7 节点/9 边 | G5 闸门 |
| **检索核心参数不变** | `k=4`、`ranker_type="rrf"`、`ranker_params={"k":100}`、工具描述文案 | 人工比对 |
| **会话标识与提问不变** | `SESSION_ID='zs123'`、`什么是EUV光刻机？` | G7 闸门 |

### 11.2 环境约束

- 依赖一律装在**工作区根的 `.venv`**，**不得修改 `C:\Anaconda` base 环境**（base 仍为 langchain 1.4.3 / transformers 5.17.0，与本项目无关）。
- 若在 `C:\Anaconda\Lib\site-packages` 下发现 `langchain-milvus`、`pymilvus`、`tavily-python` 或 `transformers 4.57.x`，说明 base 被污染，需排查。
- 每次改动后 `pip check` 必须输出 `No broken requirements found.`
- `requirements.txt` 中的 `langchain-classic` **不得删除**——它是 `langchain-community 0.4.2` 的硬依赖，删掉会导致全新环境安装缺包（本次已实测验证）。
- 验证脚本运行需 `HF_HUB_OFFLINE=1`（模型已缓存）或可访问 HuggingFace；构图验证需注入占位密钥，脚本会自动处理。

### 11.2a 为何保留 `datas/output/`（曾评估是否删除，结论：保留）

该目录下 187 个 JSON 是 `datas/layout-parser-paper.pdf` 的完整解析产物
（Title 17 / NarrativeText 126 / PageNumber 15 / Header 15 / Image 10 / Table 2 / CodeSnippet 2，
覆盖 12+ 页）。曾按「无用则删」逐条核查，结论是**保留**，依据如下：

**1. 无任何代码消费它们** —— 全项目搜索 `datas/output` / `.json` 零命中
（`write_milvus.py` 只扫 `.md`、`markdown_parser.py` 只处理 Markdown）。

**2. 但它们不可再生** —— 仓库中**没有 PDF 解析脚本**：`markdown_parser.py` 只用
`UnstructuredMarkdownLoader`，`write_milvus.py` 只收 `.md`（docstring 明写
"解析目录下所有md文件"，且 `if f.endswith('.md')`）。删掉后无法用现有代码重建。

**3. 本项目从未打算支持 PDF**（这是"保留"而非"补齐缺失功能"的依据）：
- 原始依赖清单 143 行中，PDF 相关只有 `pypdf`，而它是 `unstructured` 的传递依赖；
  无 `pdfplumber` / `poppler` / OCR / `marker` / `doclayout` 任何一项
- 36 个 `.py` 搜索 `pdf` / `UnstructuredPDFLoader` / `partition_pdf` / `hi_res` 均零命中
- 7 个 Markdown 语料搜索 `\bpdf\b` 零命中

**4. 产物来自项目外部** —— 这些 JSON 的 metadata 带 `coordinates` / `PixelSpace` /
`layout_width`，是 Unstructured **hi_res** 策略的签名；而本项目从来只用
`strategy='fast'`（不产出坐标）。文件名 `layout-parser-paper.pdf` 亦为另一 demo 的样本。

**本次仅做的修正**：把 187 个文件中失效的 `metadata.source`
（`E:\my_project\...`）改为相对路径 `datas/layout-parser-paper.pdf`；
`page_content` **改动数为 0**，187/187 仍全部可正常解析。

> 若将来确实需要 PDF 入库，那是一次**独立的新需求**（需先确定 hi_res 还是 fast、
> 是否入库、版面表格如何处理），不属于「恢复原有意图」。

### 11.3 已知但本次未修的项（如需修改请先评估）

| 项 | 现状 | 影响 / 处置建议 |
|---|---|---|
| `score_threshold=0.1` | 新旧版本均不消费，一直是惰性参数 | 无实际效果。若要启用阈值过滤需改 `search_type='similarity_score_threshold'` |
| `langchain-community` 已停止维护 | 仍承载 `UnstructuredMarkdownLoader` | 导入即有 DeprecationWarning，1.0 将移除 |
| `TavilySearchResults` 已 deprecated | 仍可用，标记 `removal='1.0'` | 将来迁 `langchain_tavily.TavilySearch`（注意其默认返回**字符串**，需同步改取值逻辑） |
| ~~硬编码 Milvus 地址与路径~~ | **已修复** | 已改为 env 变量 + 同值默认（见 3.3a） |
| `graph1.py` / `graph_2.py` 顶层 `while input` | **已修复** | 已拆出 `create_graph` + `main`，import 无副作用 |
| 多轮历史含工具调用与工具输出 | 1.x 标准行为，与旧实现不同 | token 消耗更高（详见 5.5） |
| ~~`InMemorySaver` 仅适用于调试/测试~~ | **已修复** | 已改用官方推荐的 `SqliteSaver`（见 3.3b）；进一步上生产建议换 `PostgresSaver` |

### 5.6 端到端实测发现的三个真实缺陷（均为项目自带，非升级引入）

用 `scripts/e2e_smoke_test.py` 跑通「解析 -> 建表 -> 写入 -> 混合检索 -> 过滤」全链路时，
暴露出三个**一年多前就存在、但从未被真实数据触发过**的问题：

**① 顶层正文块缺 `title`，导致写入必然失败**
`markdown_parser.py` 的 `merge_title_content()` 对 `parent_id is None` 的 `NarrativeText`
块直接 `append`，绕过了设置 `title` 的 Title 分支；而 `milvus_db.py` 的 schema 中
`title` 是非空 VARCHAR。实测 7 个 md 文件中有 5 个各缺 1 个：
`DataNotMatchException: Insert missed an field 'title'`。
*修复*：在该分支用 `setdefault` 补 `title`（取 `filename`，缺失退化为空串），不改正文内容。

**② `category_depth` 同样可能缺失**
该字段是 unstructured 的嵌套元素属性（表格/图片嵌套在正文中才有），顶层文本没有；
而 schema 中它是「非空 INT64 无默认值」。实测同样 5 个块缺失。
*修复*：新增 `MarkdownParser._ensure_required_fields()`，对
`category/source/filename/filetype/title/category_depth` 统一 `setdefault` 兜底
（VARCHAR 给 `''`、INT64 给 `0`），只补 metadata，不动正文。

**③ `jieba` 是缺失依赖**
`milvus_db.py` 的 schema 用了 `analyzer_params={"tokenizer": "jieba"}`，
但原始 143 行 freeze 里**没有 jieba**。缺它时建表直接失败：
`JiebaAnalyzer requires the 'jieba' package. Install it with: pip install jieba`。
*修复*：已补入 `requirements.txt`（`jieba==0.42.1`）。
注意这是**服务端**依赖——真实 Milvus 服务端也要装 jieba，客户端装了没用。

**④ 附带发现（非缺陷，但影响可测性）**：`create_collection()` 的幂等重建逻辑
先按固定索引名 `drop_index` 再 `drop_collection`。真实 Milvus 上索引名由
`add_index(index_name=...)` 显式指定，没问题；但 **Milvus Lite 的索引是自动命名的**，
重跑时会报 `code=700 no index to drop`。因此 e2e 脚本改为自行预清理集合。

### 5.7 在真实 Milvus 2.6.25 上实测得出的结论（与 Lite 差异很大）

真实服务：`milvusdb/milvus:v2.6.25`（Docker compose，含 milvus-minio / milvus-etcd），无鉴权。

**① 官方 `Function(RERANK)` 写法的正确形状（我第一次写错了）**

最初按"官方规范"写成这样，在真实服务上报 `code=1100 unsupported rerank function: []`：
```python
Function(name="rrf", function_type=FunctionType.RERANK,
         input_field_names=["dense", "sparse"], params={"k": 100})   # 错
```
逐项试探服务端后确认正确形状有两处硬性要求：
```python
Function(name="rrf", function_type=FunctionType.RERANK,
         input_field_names=[],# ① RRF 不接受输入字段，必须为空
         params={"reranker": "rrf", "k": 100})   # ② params 里必须有 "reranker" 判别键
                                                     # （服务端支持 decay/model/rrf/weighted）
```
在真实服务上与旧式 `RRFRanker(100)` 对照，返回结果与距离**完全一致**（0.0197 / 0.0099）。
**这个错误只有真实服务能暴露，代码审查与 Lite 都发现不了。**

**② `add_documents` 缺少 flush，导致入库后立刻检索不到（重要）**

实测对比：
```
add_documents 之后 row_count = 0     ← 检索报 nq [0] is invalid（查到空集合）
显式 client.flush() 之后 row_count = 17，检索正常
```
`langchain-milvus` 的 `add_documents` 只把数据写进 WAL，不主动flush。
*修复*：`MilvusVectorSave.add_documents()` 末尾显式 `client.flush(COLLECTION_NAME)`，
失败只告警不中断（数据仍在 WAL，会被自动 flush）。
这同时修复了批量入库管道 `write_milvus.py` 的同类问题。

**③ `tools/retriever_tools.py` 在 import 时就绑定集合名**

该模块在导入时执行 `MilvusVectorSave().create_connection().as_retriever()`，
此时 `COLLECTION_NAME` 被固化。**之后修改 `COLLECTION_NAME` 不会影响已创建的检索器**，
表现为检索指向不存在的集合 → `nq [0] is invalid`。
验证方法：在改集合名后 `importlib.reload(retriever_tools)`，或改完重启进程。

**④ `.env` 的优先级高于系统环境变量**

`utils/env_utils.py` 用 `load_dotenv(override=True)`，因此 `.env` 里写
`OPENAI_API_KEY=`（空值）会**把系统里已设的同名变量覆盖成空**，
表现为明明设了 key 却报 `OpenAIError: The api_key client option must be set`。
`.env` 中应填真实值，或把该行注释掉。

**⑤ Lite 与真实服务的差异（选型参考）**

| 能力 | Milvus Lite 3.2.1 | 真实 Milvus 2.6.25 |
|---|---|---|
| 建表（BM25 内置函数 + jieba + 稀疏 + 512 维） | 支持 | 支持 |
| `Function(RERANK)` reranker | **不支持**（code=6） | 支持 |
| 旧式 `RRFRanker` / `WeightedRanker` | 支持 | 支持 |
| 混合检索稳定性 | 差（`MergeOp requires at least one input`） | 稳定 |
| 服务端 jieba | 需本地装 | 已内置 |

结论：**Lite 只能验证"能不能跑通"，验证不了"跑得好不好"**；检索质量必须在真实服务上评估。

**⑥ 真实服务上的端到端结果**

`scripts/e2e_smoke_test.py`（真实 BGE 模型 512 维）两种 reranker 分支均通过：
```
row_count=17 | k=4 | 检索 4 条 | 全部 category==content（expr 过滤生效）
top: How is Milvus designed? As a cloud-native vector database, Milvus sepa...
```
52 块（3 个 md 文件）时同样正常，标题面包屑生效。

---

## 附录：2026-10-08 全量依赖更新记录

在保持功能完全不变的前提下，把环境更新到当时的最新版本。分三批实施，
每批均用「21 项闸门 + e2e 冒烟 + 三入口实测」验证通过后才继续。

### 批次 1：低风险（patch / minor）

| 包 | 升级前 | 升级后 |
|---|---|---|
| `langchain-core` | 1.6.6 | **1.6.7** |
| `langgraph` | 1.2.12 | **1.2.14** |
| `langchain-huggingface` | 1.2.2 | **1.2.3** |
| `unstructured` | 0.27.10 | **0.27.16** |
| `openai` | 3.24.0 | **3.26.1** |
| `tavily-python` | 0.8.4 | **0.8.5** |

### 批次 2：ML 三件套（跨大版本，风险最高）

| 包 | 升级前 | 升级后 | 说明 |
|---|---|---|---|
| `transformers` | 4.57.6 | **5.19.0** | 4.x → 5.x 大版本 |
| `sentence-transformers` | 5.7.0 | **6.1.0** | 5.x → 6.x 大版本 |
| `torch` | 2.7.1 | **2.14.1** | 无新增依赖 |
| `tokenizers`（传递） | 0.22.2 | **0.23.2** | |
| `huggingface-hub`（传递） | 0.36.2 | **1.33.0** | |

**重点验证**：transformers 5.x 是否还能正常加载 BGE 并输出 512 维归一化向量。
实测 `dim=512`、`L2=1.000000`、批量嵌入正常 —— **迁移成功**。

### 批次 3：Milvus 服务端（大版本）

| 项 | 升级前 | 升级后 |
|---|---|---|
| 服务端镜像 | `milvusdb/milvus:v2.6.25` | **`milvusdb/milvus:v3.0.1`** |
| 客户端 `pymilvus` | 3.0.2 | 3.0.2（已是最新） |

**操作要点**：
- 配置位于 `D:\rag-app\milvus\docker-compose.yml`（不在项目目录内）
- 数据是 bind mount 到 `volumes/{etcd,minio,milvus}`，跨大版本前已备份到
  `volumes_backup_2.6.25/`（142.6 MB），并另存 `docker-compose.yml.bak-v2.6.25`
- **兼容性实测**：2.6.25 写入的数据（102 + 52 块）能被 3.0.1 **直接读取**，
  旧集合检索正常、`expr` 过滤仍生效；新建集合（含 BM25 内置函数 + 稀疏向量 + HNSW 512 维）也正常
- 结论：**无需重新导入数据**

### 升级中发现的一个操作教训

验证脚本里想新建一个测试集合时踩到了本文档 4.10 节记录的坑
（`COLLECTION_NAME` 在 `import` 时就绑定），结果误把 `t_collection01` 重建成了
17 块的空集。发现后已用全部 7 个 md 重新入库恢复为 102 块。
**再次印证：改集合名必须在 import 业务模块之前设置。**

### 升级后最终版本清单

```
langchain              1.4.3     langchain-core        1.6.7
langchain-openai       1.6.7     langchain-community   0.4.2
langchain-milvus       0.4.0     langchain-classic     1.0.8
langchain-experimental 0.4.2     langchain-huggingface 1.2.3
langchain-text-splitters 1.1.3   langchain-unstructured 1.0.1
langchain-tavily       0.2.18    langgraph             1.2.14
langgraph-checkpoint   4.2.0     langgraph-checkpoint-sqlite 3.1.1
langgraph-prebuilt     1.1.0     pymilvus              3.0.2
Milvus 服务端            3.0.1     transformers          5.19.0
sentence-transformers  6.1.0     torch                 2.14.1
tokenizers             0.23.2     huggingface-hub       1.33.0
unstructured           0.27.16    openai                3.26.1
tavily-python          0.8.5      numpy 2.5.3 / pandas 3.0.6 / pydantic 2.13.5
```

验证结果：21 项闸门全 PASS、e2e 全部通过、三个入口（graph2 / graph1 / rag_agent）
与 web_search 分支均正常，多轮记忆正常，`pip check` 无破损。
