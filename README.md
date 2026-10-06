# RAG 企业知识库

> **对外定位：技术资料 Copilot（TechDocs Copilot）** —— 同一套通用 RAG 底座，
> 面向论文 / 官方文档 / 产品手册的可溯源技术问答。
>
> 多格式文档摄入 → 按主题分集合的混合检索（稠密 + BM25 / 两阶段 RRF）
> → 四分支应答（查库 / 联网 / 直接回答 / 降级）+ 四道判分闸门与自纠错 → FastAPI + SSE。

本项目是一个可运行、可评测的企业知识库问答系统。它不只回答"能不能跑通"，
还回答"回答得准不准、贵不贵、换一种改法会不会更好"——质量由 `eval/` 的**自动题集**
与指标门禁衡量，改动由 `tests/` 的回归门禁兜底。

> 📐 **开发者先读**：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —— 一页纸看清
> 分层依赖、入库/问答/检索/评测四条链路的图与落点、六个不可破坏的契约、
> 以及「想改某个功能该动哪些文件」的索引。

---

## 一、当前状态

| 项 | 当前值 |
|---|---|
| 语料 | **22 文件 / 3,662 块 / 2 集合**（自动源：`datas/` 下每个目录 = 一个集合） |
| 支持格式 | `.md` · `.pdf` · `.docx` · `.xlsx` · `.pptx` · **`.html`** · **`.txt`**<br>（html/txt 转 Markdown 后复用同一管道；不支持 `.doc`/`.xls`/`.csv`/图片）|
| 嵌入 | `bge-large-zh-v1.5`，**1024 维**，本地 **GPU（CUDA，不可用时自动回落 CPU）**；多 worker 可切共享 embed server |
| 生成 | DeepSeek `deepseek-flash`（OpenAI 兼容端点） |
| 检索 | 稠密 HNSW/IP + BM25/jieba；集合内 RRF，跨集合二次 RRF |
| 应答分支 | **四条**：查库（vectorstore）· 联网（web_search）· 直接回答（direct）· 降级作答（fallback） |
| 来源契约 | `grounded` + `source_type`：`knowledge_base` / `web` / `model_knowledge` / `model_direct` |
| 多轮记忆 | 三层：最近 13 轮原文（SQLite）+ 摘要（>26 轮触发）+ 历史语义索引（Milvus，可 `KB_HISTORY=0` 关闭） |
| 质量门禁 | **PASS** — recall 0.9832 · hit_rate 0.9874 · faithfulness 1.0000 · answer_relevancy 0.8750<br>（2026-11-04：检索层 398 题自动题集，生成层 50 题抽样；阈值见 `eval/thresholds.json`） |
| 回归 | `run_all_checks.py` **23/23 判定通过**（API+SSE 与多租户联动需 `--full`） |
| 业务代码 | **10,108 行 / 63 个 `.py`**（`rag_kb/`） |
| 评测与回归代码 | **7,827 行 / 46 个 `.py`**（`eval/` + `tests/`） |
| 评测集 | **自动生成**（`gen_testset.py` 出题，当前 398 题；要点匹配，与语料解耦） |

---

## 二、目录结构

项目根 = 包根 = 工作区根，只有 `rag_kb/` 一个 Python 包。

```
├── rag_kb/                    业务代码（唯一命名空间）
│   ├── agent/                 LangChain 1.x 标准 Agent（create_agent）
│   ├── api/                   FastAPI + SSE 服务（app.py / run.py / embed_server.py）
│   ├── cli/                   检索试验与轨迹查看（try_search / show_trace）
│   ├── documents/             解析与入库（Markdown / PDF / Office / HTML / 表格）
│   ├── graph_react/           简化版 ReAct 工作流
│   ├── graph_crag/            CRAG 主工作流（三路路由 + 四道闸门 + 降级作答）
│   ├── llm_models/            LLM 与本地嵌入模型
│   ├── tools/                 跨集合混合检索器 / 历史向量库
│   ├── utils/                 配置 / 日志 / 引用校验 / 轨迹 / 检查点
│   └── doctor.py              环境自检
├── eval/                      评测域
│   ├── export_chunks.py       导出块清单（换语料第一步）
│   ├── gen_testset.py         自动出题（LLM 出题 + 原文要点抽取 + 本地校验）
│   ├── run_eval.py            评测运行器（检索层 / 生成层）
│   ├── metrics.py             指标与门禁（要点匹配，与语料解耦）
│   ├── thresholds.json        阈值（换语料后按新基线调整，不改代码）
│   └── venv_guard.py          解释器护栏（裸 python 自动切到 .venv）
├── tests/                     回归域（按流水线阶段分组）
│   ├── run_all_checks.py      统一回归入口（一键跑完全部）
│   ├── ingest/                解析 / 切分 / 表格
│   ├── retrieval/             召回 / 父子回查 / 精排
│   ├── workflow/              判分 / 引用 / 记忆 / 端到端
│   ├── service/               CLI / API / 会话 / embedding
│   ├── quality/               门禁 / 静态扫描 / 审计
│   └── eval/ metrics/         评测自身的契约测试
├── docs/                      开发者文档（ARCHITECTURE.md：架构与运转逻辑）
├── scripts/                   运维脚本：模型下载、Milvus 探测
├── models/                    本地模型权重（不入库）
├── datas/                     语料：**每个子目录 = 一个源 = 一个集合**
├── main.py                    统一 CLI 入口（速查表在 `python main.py`）
└── pyproject.toml             editable 安装配置
```

包名统一带 `rag_kb` 前缀（`from rag_kb.utils.env_utils import DATAS_DIR`），
避免 `utils` / `tools` 这类通用顶层名与第三方包撞名。

---

## 三、核心链路

```
datas/<主题目录>/              一个目录 = 一个源 = 一个集合
   │                          格式可混放（md/pdf/docx/xlsx/pptx/html/txt），支持子目录
   │  解析 → 标题面包屑 → 表格结构恢复/切片 → 短块合并 → 语义切分
   ▼
bge-large-zh-v1.5 (1024)
   ▼
Milvus（每目录一个集合）        dense: HNSW/IP     sparse: BM25 + jieba
   ▼
question
   │
   ├─ 路由（LLM 结构化输出，三选一）
   │    ├─ vectorstore → 跨集合混合检索（集合内 RRF → 跨集合 RRF）
   │    │                 → 逐块相关性闸门 → 生成 → 引用校验 → 幻觉/匹配度判分
   │    │                 → 不合格则改写重试；两侧额度用尽 → 终止（有明确出边）
   │    ├─ web_search  → 联网（引用带真实 URL）
   │    └─ direct       → 不检索，直接用模型知识回答（闲聊 / 追问上文 / 写作）
   │
   └─ 检索彻底无依据 → 降级作答（fallback）：答案开头强制标注「非知识库资料」
```

**四分支的边界**（问答接口返回 `source_type` 标明）：

| source_type | grounded | 含义 |
|---|---|---|
| `knowledge_base` | true | 有知识库依据，答案带 `[1]..[n]` 引用 |
| `web` | true | 联网结果，引用带 URL |
| `model_knowledge` | false | **降级作答**：检索无依据，用通用知识回答并强制标注 |
| `model_direct` | false | 主动闲聊/追问/写作，未检索 |

---

## 四、快速开始

```powershell
# 1. 激活环境
.venv\Scripts\Activate.ps1

# 2. 安装依赖（首次），并把本项目以「可编辑模式」装进环境
#    包内一律用绝对导入（from rag_kb.utils...），装成包之后任何工作目录都能解析。
#    源码仍就地修改，改完不用重装。详见 pyproject.toml 顶部说明。
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

# 3. 配置（必填：OPENAI_API_KEY、Milvus 地址、LLM 端点与模型）
Copy-Item .env.example .env

# 4. 准备本地模型（首次）
.venv\Scripts\python.exe scripts\fetch_models.py

# 5. 环境自检
.venv\Scripts\python.exe main.py doctor

# 6. 入库：把文档放进 datas/ 下的一个主题目录，然后
.venv\Scripts\python.exe main.py ingest --list          # 看有哪些源、各自文件数
.venv\Scripts\python.exe main.py ingest                 # 全量重建（--no-recreate 为增量）
.venv\Scripts\python.exe main.py ingest --sources 前端复习笔记

# 7. 启动 HTTP 服务
.venv\Scripts\python.exe main.py serve                  # 127.0.0.1:8000（单 worker，本地 GPU 模型）

# 多 worker：先起共享 embedding 服务（模型只占 1 份显存），再起 API
.venv\Scripts\python.exe main.py embed-server --port 8001
.venv\Scripts\python.exe main.py serve --workers 4 --embed-url http://127.0.0.1:8001

# 8. 终端连续问答（进程内跑，或 --url 复用已启动的服务）
.venv\Scripts\python.exe main.py chat
.venv\Scripts\python.exe main.py chat --url http://127.0.0.1:8000
```

> **统一入口是项目根目录的 `main.py`**（`doctor / serve / embed-server / ingest / eval / version`）。
> 原有入口仍可用，行为不变：`python -m rag_kb.documents.write_milvus`、`rag_kb/api/run.py`、
> `python -m rag_kb.doctor`。
>
> 直接用 `python main.py ...` 也可以：检测到当前解释器不是项目 `.venv` 时会**自动切换**过去
> （打印一行提示；设 `KB_NO_REEXEC=1` 可关闭）。`eval/` 下的工具同样有护栏（`EVAL_NO_REEXEC=1`）。

问答（示例用当前语料里的话题）：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType 'application/json' `
  -Body '{"question":"CSS 标准盒模型和怪异盒模型的区别是什么？"}'
```

命令行入口：

```powershell
.venv\Scripts\python.exe -m rag_kb.graph_crag.crag_graph     # CRAG 完整链路（四分支）
.venv\Scripts\python.exe -m rag_kb.graph_react.react_graph    # 简化 ReAct
.venv\Scripts\python.exe -m rag_kb.agent.rag_agent           # 1.x 标准 Agent
.venv\Scripts\python.exe -m rag_kb.cli.try_search            # 交互式试检索
```

评测（换语料后三步重建，零代码）：

```powershell
python eval\export_chunks.py                     # 导出块清单（缺失/落后于向量库时会拒绝出题）
python eval\gen_testset.py --per-collection 5    # LLM 自动出题 -> golden_auto.jsonl

# 检索层（秒级、0 LLM 调用；默认优先使用 golden_auto.jsonl）
.venv\Scripts\python.exe eval\run_eval.py

# 检索 + 生成层（需要可用 LLM；产物按题集派生：baseline_auto.jsonl 等）
.venv\Scripts\python.exe eval\run_eval.py --golden eval\golden_auto.jsonl --with-generation --verbose

# 全量回归（判定项 23/23，含 API/SSE 与多租户需 --full）
.venv\Scripts\python.exe tests\run_all_checks.py --full
```

---

## 五、已知边界

* **reranker 已实测否决**：bge-reranker-v2-m3 在早期 8 题上增益为 0；
* **表格"跨行统计"**：已用同表切片合并解决；未引入 text-to-SQL；
* **自动题的召回率特性**：题目从块生成，大集合（数千块）中出题块未必进 top-36，
  recall 会低于人工精选题。这是「检索容量」的真实反映，不是评测缺陷；
* **HNSW 搜索参数**：`ef` 在代码内固定为 128（langchain_milvus 默认 10 会被 pymilvus 3.x
  以 `ef > k` 校验拒绝，导致新集合检索全空）；
* **新集合首次入库后需 load**：已由 `ingest` 自动完成（未 load 的集合查询会超时失败）；
* **并发**：1/2 基本无劣化；4 并发 4/4 成功（推荐上限 4~6）；8 并发 8/8 成功但墙钟 +37%；
* **扫描件 OCR**：默认关闭（`PDF_OCR_FALLBACK=0`）；
* **多租户**：只做集合级授权隔离，不含身份认证；
* **`context_precision`**：要点口径下不适用（返回 N/A），块 ID 口径下天然偏低，只作监控。
