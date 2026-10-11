# RAG 企业知识库

基于 Milvus 的多格式文档知识库问答系统。文档按主题目录组织，解析成块后写入对应集合，检索使用稠密向量与 BM25 混合、两级 RRF 融合；问答支持查库、联网、直接回答和降级作答四条路径，答案带引用来源。

系统自带评测与回归测试：检索层指标不消耗 LLM 调用，换语料后三行命令即可重建评测集。

## 特性

- **多格式解析** — md / pdf / docx / xlsx / pptx / html / txt，转换为 Markdown 后走同一分块管道
- **目录即集合** — `datas/` 下每个子目录自动注册为一个源和一个 Milvus 集合，目录内可混放格式、可嵌套子目录
- **混合检索** — 稠密 HNSW 与 BM25 稀疏检索，集合内 RRF、跨集合二次 RRF
- **四条应答路径** — 查库 / 联网 / 直接回答 / 降级作答，响应中标注 `source_type` 与 `grounded`
- **SSE 流式** — 节点进度 + 生成阶段逐 token 下发（`event: token`），整包 `answer` 事件保留
- **多轮记忆** — 最近 13 轮原文、超长对话摘要、历史语义索引三层
- **可评测** — 自动出题 + 要点匹配，检索层指标 0 次 LLM 调用
- **回归测试** — 23 项判定，覆盖解析、检索、判分、记忆、服务

## 技术栈

| 环节 | 选型 |
|---|---|
| 嵌入模型 | `bge-large-zh-v1.5`，1024 维，本地 GPU（无 CUDA 时回落 CPU） |
| 生成模型 | 任意 OpenAI 兼容端点（示例配置为 DeepSeek） |
| 向量库 | Milvus（dense HNSW/IP + sparse BM25） |
| 编排 | LangGraph、LangChain 1.x |
| 服务 | FastAPI + SSE |

## 快速开始

### 1. 环境

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 是必需的：包内模块使用绝对导入（`from rag_kb.utils...`），装成可编辑包后才能在任意目录解析。

### 2. 配置

```powershell
Copy-Item .env.example .env
```

`.env` 中至少填写 `OPENAI_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`MILVUS_URI`。

### 3. 准备模型与自检

```powershell
python scripts\fetch_models.py
python main.py doctor
```

`doctor` 会检查 Python 版本、GPU、嵌入模型、Milvus 连接与增量清单。

### 4. 入库

把文档放进 `datas/` 下的任意主题目录，目录名即源名：

```
datas/
├── docs/                  纯文本文档
│   ├── getting-started.md
│   └── configuration.md
└── manuals/               二进制文档（pdf / docx / pptx）
    └── user-guide.pdf
```

目录名会作为源名和路由提示词里的主题描述，建议按语义域命名；目录内格式可混放，也支持嵌套子目录。

```powershell
python main.py ingest --list      # 查看识别到的源与文件数
python main.py ingest             # 全量重建
python main.py ingest --no-recreate   # 增量：只处理内容变化的文件
```

### 5. 启动服务

```powershell
python main.py serve              # http://127.0.0.1:8000
python main.py chat               # 或直接在终端对话
```

多 worker 场景先启动独立的 embedding 服务，避免每个进程各加载一份模型：

```powershell
python main.py embed-server --port 8001
python main.py serve --workers 4 --embed-url http://127.0.0.1:8001
```

### 6. 调用

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType 'application/json' `
  -Body '{"question":"支持哪些配置项？"}'
```

响应包含 `answer`、`citations`、`grounded`、`source_type`。多轮对话在请求中带 `session_id`。

## 目录结构

```
rag_kb/
├── agent/          LangChain 1.x 标准 Agent
├── api/            FastAPI + SSE
├── cli/            检索试验与轨迹查看
├── documents/      解析与入库
├── graph_crag/     CRAG 主工作流（路由、判分、生成、降级）
├── graph_react/    简化版 ReAct 工作流
├── llm_models/     LLM 与嵌入模型
├── tools/          检索器、历史向量库、精排
└── utils/          配置、日志、引用校验、检查点

eval/               评测工具链：块导出、自动出题、指标与门禁
tests/              回归测试，按流水线阶段分组
scripts/            模型下载、Milvus 探测
datas/              语料目录，每个子目录一个集合
docs/               架构与运转逻辑
```

## 工作流

```
问题
 ├─ 路由 ──┬─ vectorstore  检索 → 逐块相关性判分 → 生成 → 引用校验 → 幻觉/匹配度判分
 │         │                 不通过则改写重试，额度用尽后终止
 │         ├─ web_search   联网检索 → 生成
 │         └─ direct       不检索，直接作答
 └─ 检索无依据 ── fallback  通用知识作答，答案标注非知识库内容
```

四条路径通过 `source_type` 区分：`knowledge_base`、`web`、`model_knowledge`、`model_direct`。

## 评测

评测集由语料自动生成，判据是答案要点是否出现在召回块中，因此不依赖分块方式，换语料后重新生成即可。

```powershell
python eval\export_chunks.py                    # 导出块清单
python eval\gen_testset.py --per-collection 5   # LLM 出题 -> golden_auto.jsonl
python eval\run_eval.py                         # 检索层，0 次 LLM 调用
python eval\run_eval.py --with-generation --verbose   # 追加生成层指标，逐题打印问答
```

指标与阈值：

| 指标 | 含义 | 阈值文件 |
|---|---|---|
| `context_recall` | 答案要点被召回的比例 | `eval/thresholds.json` |
| `hit_rate` | 至少命中一个要点的题目比例 | 同上 |
| `faithfulness` | 答案中的事实能否由资料支持 | 同上 |
| `answer_relevancy` | 答案是否切题 | 同上 |

回归测试：

```powershell
python tests\run_all_checks.py          # 23 项
python tests\run_all_checks.py --full   # 追加 API/SSE 集成
```

## 配置项

| 变量 | 说明 |
|---|---|
| `MILVUS_URI` | Milvus 地址，默认 `http://127.0.0.1:19530` |
| `LLM_BASE_URL` / `LLM_MODEL` | OpenAI 兼容端点与模型名 |
| `TAVILY_API_KEY` | 联网检索，未配置时联网分支不可用 |
| `RERANK_ENABLED` / `RERANK_SCORE_THRESHOLD` | 精排开关与分数门槛 |
| `MAX_HISTORY_TURNS` / `MAX_HISTORY_CHARS` | 注入 prompt 的历史轮数与字符预算 |
| `HISTORY_ENABLED` | 历史语义索引开关，短会话可关闭 |
| `SESSION_TTL_SECONDS` | 会话过期时间，默认 6 小时 |
| `GRAPH_RECURSION_LIMIT` | 工作流最大步数，默认 20 |
| `PDF_OCR_FALLBACK` | 扫描件 OCR，默认关闭 |
| `TOKEN_STREAM_ENABLED` | SSE 是否逐 token 下发，默认开启 |

## 已知限制

- **扫描件 PDF** 默认不做 OCR，需要时设置 `PDF_OCR_FALLBACK=1` 并安装 Tesseract
- **HTML/TXT** 需转换为 Markdown 后入库；`.doc`、`.xls`、`.csv` 与图片不支持
- **大集合召回** 受每集合召回数限制，块数上万的集合中，部分问题的目标块可能进不了最终上下文
- **reranker** 在本项目语料上未观察到增益，默认关闭

## 文档

架构设计、模块依赖、数据流与改动指引见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
