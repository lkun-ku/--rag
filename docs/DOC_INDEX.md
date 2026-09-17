# 文档索引 —— 每个文档干什么用、时效如何

> **最后核对：2026-10-17**
> 项目里有 **17 个 `.md`** —— 但其中 11 个是**语料源文件**或**模型自带 README**，
> 真正的「项目文档」只有 **6 个**。本文把它们说清楚，避免读错版本。

---

## 一、项目文档（4 个有效 + 2 个已归档）—— 用途与时效

### 当前有效（4 个）

| # | 文档 | 用途 | 权威度 | 最后更新 | **时效** |
|---|---|---|---|---|---|
| 1 | **`ROADMAP.md`** | **主文档**：所有改造过程 + **实测证据**（P0~P3 问题清单、每项改动的数字与依据） | ⭐⭐⭐ **唯一权威** | **2026-10-17** | ✅ **最新** |
| 2 | **`PROJECT_DESCRIPTION.md`** | **架构与逐模块职责梳理**（技术栈 / 目录职责 / 数据流 / 关键原理 / 配置 / 已知限制），63 个标题 | ⭐⭐ 权威（架构层面） | **2026-10-17** | ⚠️ **正文过时，但已加「修订说明」对照表** |
| 3 | **`docs/RESUME_PROJECT.md`** | **简历/面试用**：5 项核心亮点 + 技术栈 + 可讲述的决策 | ⭐⭐ 面向对外 | **2026-10-17** | ✅ **最新** |
| 4 | `TECHNICAL_DETAILS.md` | **技术原理深挖**（问答式：「共有几种检索方式」「有哪些 Agent」），含解析/切片/Embedding/RRF 的细节 | ⭐ 参考 | **2026-10-08** | ⚠️ **停在阶段一之前**（见下方说明） |

### 已归档（2 个，移入 `docs/archive/`）

**它们的结论已全部并入 ROADMAP / PROJECT_DESCRIPTION —— 日常不需要看。**

| 文档 | 是什么 | 结论已并入 | 归档理由 |
|---|---|---|---|
| `docs/archive/UPGRADE_NOTES.md` | LangChain 1.x 迁移报告（783 行） | `PROJECT_DESCRIPTION` 技术栈 | **迁移已完成**，属一次性工作记录 |
| `docs/archive/迁移验证报告.md` | 迁移验证输出（自动生成，124 行） | `ROADMAP` 阶段零决策 | 同上，且是**生成物** |

> **归档 ≠ 废弃** —— 它们记录了「当时为什么这么做」，是**证据链的一部分**。
> 需要追问「为什么从 LangChain 0.x 升到 1.x」「迁移时验证了什么」时再翻。
> 详见 [`docs/archive/README.md`](archive/README.md)。

### 关于 1 与 2 的关系（最容易搞混的一对）

```
ROADMAP.md            = **过程与证据**（"我们改了什么、实测数字是多少、为什么这么定"）
                        6,194 行 —— 回答「这个结论凭什么成立」
PROJECT_DESCRIPTION.md = **结构与原理**（"系统由什么组成、每个模块干什么"）
                        861 行 —— 回答「这套系统长什么样」
```

**两者互补，不重复**。要查「某个结论怎么来的」→ ROADMAP；
要查「某个模块/字段是什么」→ PROJECT_DESCRIPTION。

### 关于 4 的时效（建议这样用它）

`TECHNICAL_DETAILS.md` 的**组织形式独特**（问答式，适合快速定位某个概念），
但**内容停在阶段一之前**，以下字段已过时：

| 该文档里的说法 | 当前实际 | 权威出处 |
|---|---|---|
| 嵌入 `bge-small-zh-v1.5`（512 维） | **`bge-large-zh-v1.5`（1024 维）** | ROADMAP 阶段一 |
| 语料 7 个 Markdown / 约 102 块 | **18 文件 / 523 块 / 3 集合** | ROADMAP 4.7 |
| `graph2` 的「7 个节点 / 9 条边」 | 已增 `web_search`、判分并发等改动 | ROADMAP 4.25 |
| `score_threshold` 无效 | 仍成立 ✅ | ROADMAP 4.10 |

**→ 建议用法：当「概念词典」查，数字一律以 ROADMAP 为准。**

---

## 二、非项目文档（11 个，不要当文档读）

| 类别 | 文件 | 说明 |
|---|---|---|
| **语料源文件**（6） | `RAG_PROJECT/datas/md/*.md`（`milvus_cli_reference` / `milvus_faq` / `milvus_glossary` / `milvus_product_overview` / `milvus_troubleshooting` / `milvus_vs_alternatives`） | **这是知识库的输入数据**，不是项目文档。改动它们 = 改语料 = 要重建索引 |
| **语料溯源清单**（1） | `RAG_PROJECT/datas/CORPUS_MANIFEST.md` | 语料的来源与清洗记录（**数据凭据**，有价值） |
| **模型自带 README**（2） | `RAG_PROJECT/models/bge-large-zh-v1.5/README.md`、`models/bge-reranker-v2-m3/README.md` | 下载模型时附带的（HuggingFace 原文），**不是本项目写的** |
| **约定说明**（1） | `scratch/README.md` | `scratch/` 是「非生产代码存放区」的约定 |
| **AI 会话计划**（1） | `.codebuddy/plans/*.md` | 工具生成的计划文件（**不属于项目交付物**） |

---

## 三、什么时候更新哪个文档

| 场景 | 更新哪里 |
|---|---|
| **做了任何改造 / 得出任何实测结论** | **`ROADMAP.md`**（**唯一必更**）—— 附数字与证据 |
| 改了架构 / 新增模块 / 改了字段 | `PROJECT_DESCRIPTION.md`（并同步顶部的「修订说明」） |
| 对外材料（简历、面试、介绍）需要同步 | `docs/RESUME_PROJECT.md` |
| 语料换了一批 | `datas/CORPUS_MANIFEST.md` + 重跑入库 |
| 概念解释需要补充 | `TECHNICAL_DETAILS.md`（但**先确认它说的数字还成立**） |

**一句话原则**：**`ROADMAP.md` 是唯一权威 —— 其它文档的数字与它冲突时，以它为准。**

---

## 四、为什么不删掉过时的文档

`TECHNICAL_DETAILS.md` / `UPGRADE_NOTES.md` / `迁移验证报告.md` 都已过时或已归档，
**但都保留**，理由与项目一贯做法一致：

* **`UPGRADE_NOTES.md` + `迁移验证报告.md`**：是**已完成工作的证据**（迁移前 vs 后的对照），
  删了就再也说不清「当时为什么这么迁移」；
* **`TECHNICAL_DETAILS.md`**：问答式组织**无可替代**（`PROJECT_DESCRIPTION.md` 是叙述式），
  且它的「为什么 HNSW 而不是 FLAT」「为什么用 `expr` 而不是 `filter`」等推理**仍然成立**；
* 本项目的既定做法就是**保留旧值 + 标注当前值**，而不是覆盖（如 `golden` 修正保留
  `answer_old`、`PROJECT_DESCRIPTION` 用「修订说明」对照表、`measure_grade_batch_saving.json`
  记录 QIDS）。

> **改写历史会让证据链断裂** —— 而本项目所有结论都建立在「能追溯到当时实测」之上。
