# RAG 企业知识库

> **对外定位：技术资料 Copilot（TechDocs Copilot）** —— 同一套通用 RAG 底座，
> 面向论文 / 官方文档 / 产品手册的可溯源技术问答。
>
> 多格式文档摄入 → 三集合混合检索（稠密 + BM25 / 两阶段 RRF）→ 四道判分闸门 + 自纠错 → FastAPI + SSE。

本项目是一个可运行、可评测的企业知识库问答系统。它不只回答“能不能跑通”，
还回答“回答得准不准、贵不贵、换一种改法会不会更好”——所有关键结论都有
`eval/` 下的实测产物支撑。

---

## 一、当前状态

| 项 | 当前值 |
|---|---|
| 语料 | **18 文件 / 454 块 / 3 集合**（6 Markdown + 5 PDF + 7 Office） |
| 支持格式 | `.md` · `.pdf` · `.docx` · `.xlsx` · `.pptx` · **`.html`** · **`.txt`**<br>（html/txt 转 Markdown 后复用同一管道；不支持 `.doc`/`.xls`/`.csv`/图片）|
| 嵌入 | `BAAI/bge-large-zh-v1.5`，**1024 维**，本地 CPU |
| 生成 | DeepSeek `deepseek-flash`（OpenAI 兼容端点） |
| 检索 | 稠密 HNSW/IP + BM25/jieba；集合内 RRF，跨集合二次 RRF |
| 质量门禁 | **PASS** — recall 0.8664 · hit_rate 0.9375 · faithfulness **1.0000** · answer_relevancy 0.8750 |
| 回归 | `run_all_checks.py --full` **20/20 判定通过**（含 API+SSE、多租户联动） |
| 业务代码 | **7,416 行 / 48 个 `.py`** |
| 评测代码 | **11,999 行 / 81 个 `.py`**（评测代码量超过业务代码） |
| golden 集 | 32 题 |

> 454 是当前 Milvus 三个集合的 `row_count` 合计（216 + 160 + 78）；
> `eval/chunk_index.json` 的 496 是**重灌前**的历史快照，不代表当前行数。

---

## 二、文档地图（日常只看这 4 个）

| 文档 | 一句话 | 什么时候看 |
|---|---|---|
| **`README.md`** | 总览、当前状态、快速开始、文档地图 | 第一次进入项目 |
| **`ROADMAP.md`** | **迭代日志**：按时间追加的改造过程与实测证据 | 想知道“某个数字/结论怎么来的” |
| **`PROJECT_DESCRIPTION.md`** | **当前架构**：数据流、模块职责、关键设计、配置 | 想知道“系统现在怎么运转” |
| **`docs/RESUME_PROJECT.md`** | 简历/面试口径：5 项核心亮点 + 可讲述的决策 | 对外介绍 |

历史归档：

| 文档 | 说明 |
|---|---|
| `docs/archive/HISTORY.md` | LangChain 1.x 迁移（已完成的一次性工程），只在追溯迁移时看 |

> `HISTORY.md` 是精简版；迁移的逐项验证输出可用
> `scripts/verify_migration.py --report` 重新生成，不再单独保留旧报告。

以下 `.md` **不是项目文档**，不要当说明读：

* `RAG_PROJECT/datas/md/*.md` —— 知识库语料源文件（改它们 = 改语料 = 需重建索引）；
* `RAG_PROJECT/datas/CORPUS_MANIFEST.md` —— 语料来源与清洗记录；
* `RAG_PROJECT/models/*/README.md` —— 模型自带说明；
* `scratch/` —— 临时验证目录（未入库，不是交付文档）。

---

## 三、目录结构

```
RAG_PROJECT/
├── rag_kb/                    ← 代码全部在这个命名空间下
│   ├── agent/                 LangChain 1.x 标准 Agent（create_agent）
│   ├── api/                   FastAPI + SSE 服务（/health /ask /ask/stream /ingest）
│   ├── documents/             解析与入库（Markdown / PDF / Office / PPTX / 表格）
│   ├── graph/                 简化版 ReAct 工作流
│   ├── graph2/                CRAG 主工作流（路由 + 四道闸门 + 自纠错）
│   ├── llm_models/            LLM 与本地嵌入模型
│   ├── tools/                 跨集合混合检索器
│   ├── utils/                 配置 / 日志 / 引用校验 / 轨迹 / 检查点
│   └── scripts/               检索试验与轨迹查看
├── datas/                     语料：md / pdf / office（数据在包外）
├── models/                    本地嵌入模型（不入库）
└── logs/                      运行日志与 trace

eval/                 评测、探针、测量、回归（run_eval / run_all_checks）
scripts/              模型下载、Milvus 探测、迁移验证、E2E
docs/                 RESUME + 历史归档
```

包名统一带 `rag_kb` 前缀（`from rag_kb.utils.env_utils import MD_DIR`），
避免 `utils` / `tools` 这类通用顶层名与第三方包撞名。

---

## 四、核心链路

```
datas/{md,pdf,office}
   │  解析 → 标题合并 → 表格结构恢复/切分 → 短块合并 → 语义切分
   ▼
bge-large-zh-v1.5 (1024) ──► 3 个 Milvus 集合
                              dense: HNSW/IP
                              sparse: BM25 + jieba

question
   │
   ├─ 路由：查库 / 联网
   ├─ 跨集合混合检索（集合内 RRF → 跨集合 RRF）
   ├─ 文档相关性闸门（逐块 yes/no）
   ├─ 生成（上下文编号 [1]..[n]，强约束引用）
   ├─ 确定性引用校验（越界/拒答/原文编号）
   └─ 幻觉 + 问题匹配度并发判分 → 自纠错 / 熔断
```

---

## 五、快速开始

```powershell
# 1. 激活环境
.venv\Scripts\Activate.ps1

# 2. 安装依赖（首次），并把本项目以「可编辑模式」装进环境
#    最后一步是关键：包内一律用绝对导入（from utils... / from documents...），
#    装成包之后这些导入在任何工作目录下都能解析，模块里无需再手工改 sys.path。
#    源码仍就地修改，改完不用重装。详见 pyproject.toml 顶部说明。
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

# 3. 配置（必填：OPENAI_API_KEY、Milvus 地址、LLM 端点与模型）
Copy-Item .env.example .env

# 4. 准备本地模型（首次）
.venv\Scripts\python.exe scripts\fetch_models.py

# 5. 入库（按 DOC_SOURCES 逐源写入）
.venv\Scripts\python.exe -m rag_kb.documents.write_milvus

# 6. 启动 HTTP 服务
.venv\Scripts\python.exe RAG_PROJECT\api\run.py
```

问答：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType 'application/json' `
  -Body '{"question":"Milvus 支持哪些索引类型？"}'
```

命令行入口：

```powershell
.venv\Scripts\python.exe -m rag_kb.graph2.graph_2        # CRAG 完整链路
.venv\Scripts\python.exe -m rag_kb.graph.graph1          # 简化 ReAct
.venv\Scripts\python.exe -m rag_kb.agent.rag_agent       # 1.x 标准 Agent
.venv\Scripts\python.exe -m rag_kb.scripts.try_search    # 交互式试检索
```

评测：

```powershell
# 检索层（秒级、0 LLM 调用）
.venv\Scripts\python.exe eval\run_eval.py

# 检索 + 生成层（需要可用 LLM）
.venv\Scripts\python.exe eval\run_eval.py --with-generation

# 全量回归（判定项 20/20，含 API/SSE 与多租户）
.venv\Scripts\python.exe eval\run_all_checks.py --full
```

---

## 六、已知边界

* **reranker 已实测否决**：bge-reranker-v2-m3 在本项目 8 题上增益为 0；
* **表格“跨行统计”**：已用同表切片合并解决；未引入 text-to-SQL；
* **并发**：1/2 基本无劣化；4 并发 4/4 成功（推荐上限 4~6）；8 并发仍 8/8 成功但墙钟 +37%；更高并发与 HTTP 层压测未做；
* **扫描件 OCR**：默认关闭（`PDF_OCR_FALLBACK=0`）；
* **多租户**：只做集合级授权隔离，不含身份认证；
* **`context_precision`**：天然偏低，只作硬门禁 / 监控，不参与告警。

更多“为什么这么定”的证据，见 [`ROADMAP.md`](ROADMAP.md)；
当前架构细节见 [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md)。
