# 架构与运转逻辑

> 给开发者的一页纸：**数据从哪来、经过谁、被谁消费**。
> 每个箭头都对应真实函数；不确定时先看第 1 节的总图，再看对应章节。

---

## 1. 一图看懂

```
 ┌──────────────────────────── 写入侧（离线） ────────────────────────────┐
 │                                                                        │
 │  datas/<主题目录>/          一个目录 = 一个源 = 一个 Milvus 集合         │
 │      │   （md / pdf / docx / xlsx / pptx / html / txt 可混放，可递归）   │
 │      ▼                                                                 │
 │  env_utils.discover_auto_sources()   扫 datas/ 子目录 → 生成源清单       │
 │      ▼                                                                 │
 │  file_scan.list_source_files()      递归收集 + 过滤锁文件/隐藏文件        │
 │      ▼                                                                 │
 │  write_milvus.ingest_source()       每源：全量重建 or 增量              │
 │      │  build_parser('auto') → _auto_parse()  按扩展名分派解析器        │
 │      │  *_parser → MarkdownParser：标题面包屑→表格恢复→语义切分→短块合并 │
 │      ▼                                                                 │
 │  milvus_db.MilvusVectorSave         建表 + 写入                        │
 │      │  schema: text / category / source / filename / title /          │
 │      │          filetype / category_depth / dense / sparse(BM25)       │
 │      └─ flush() ──► load_collection()   ← 不做这一步，集合检索全空       │
 └────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────── 读取侧（在线） ────────────────────────────┐
 │                                                                        │
 │  question ──► graph_crag.run_with_trace()                               │
 │                  │                                                      │
 │                  ├─ route_question()  LLM 结构化输出三选一               │
 │                  │     ├─ vectorstore ─► 检索 ─► 逐块判分 ─► 生成 ─► 判分│
 │                  │     ├─ web_search  ─► 联网 ─► 生成 ─► 判分           │
 │                  │     └─ direct      ─► 直接回答（不检索）              │
 │                  │                                                      │
 │                  └─ 检索彻底无依据 ─► fallback（降级作答，强制标注）      │
 │                                                                        │
 │  输出：answer + citations[{n, filename, title, url, from_web}]          │
 │        + grounded（是否有依据）+ source_type（四态）                     │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 分层与依赖方向

依赖**只能从上往下**（已用 grep 验证无反向 import）：

```
┌─ L3 接口层 ────────────────────────────────────────────────┐
│  main.py（CLI）      api/app.py（HTTP+SSE）   api/run.py     │
│  rag_kb/doctor.py    cli/try_search · show_trace            │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌─ L2 编排层 ────────────────────────────────────────────────┐
│  graph_crag/（主链路，12 文件）  graph_react/（简化 ReAct）  │
│  agent/rag_agent.py（LangChain 1.x 标准 Agent）             │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌─ L1 数据/能力层 ───────────────────────────────────────────┐
│  documents/  解析 + 入库（9 文件）                           │
│  tools/      检索器 / 历史向量库 / 精排 / 切分辅助（7 文件）  │
│  llm_models/ LLM 与 embedding（3 文件）                     │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌─ L0 基础设施 ──────────────────────────────────────────────┐
│  utils/  配置 / 日志 / 引用校验 / 轨迹 / 检查点 / 会话（9 文件）│
└────────────────────────────────────────────────────────────┘
```

**为什么要这个方向**：`documents` 和 `tools` 不知道自己会被谁调用，才能被 CLI、API、评测、测试四方复用。任何一层反向 import 上层，都会让评测无法单独加载它。

---

## 3. 入库链路（写入侧）

```
main.py ingest [--sources X] [--no-recreate] [--list]
  │
  ▼
env_utils.DOC_SOURCES                      = 显式源(_EXPLICIT_DOC_SOURCES) + 自动源(discover_auto_sources)
  │   自动源规则：datas/ 的直接子目录；集合名 kb_auto_<md5(目录名)[:8]>_bge_large_1024
  │               主题描述 = 目录里的 _topic.txt 首行，否则「<目录名>目录下的文档」
  ▼
file_scan.list_source_files(source)         os.walk 递归；跳过 . / _ 开头目录与 ~$ 锁文件
  ▼
write_milvus.ingest_source(dir, files, collection, parser, ...)
  ├─ 全量：recreate()  → drop + create（schema 见 milvus_db）
  ├─ 增量：按 sha1 比对 logs/ingest_manifest，只解析变化的文件
  │
  ├─ build_parser(parser)                  parser='auto' → _auto_parse
  │     _EXT_TO_PARSER: .md→markdown  .pdf→pdf  .docx/.xlsx/.pptx→office  .html→html  .txt→text
  │
  ├─ 每文件：parse_fn(path) → List[Document]
  │     markdown : 标题面包屑 → 表格结构恢复/切片 → 语义切分 → 短块合并 → 字节上限兜底
  │     pdf      : pymupdf4llm 转 Markdown（可选 OCR）→ 同上
  │     office   : markitdown 转 Markdown → 同上
  │     html/txt : plain_converters 转 Markdown → 同上
  │
  ├─ _iter_file_batches() 流式产出 → milvus_db.add_documents()
  └─ flush() → load_collection()          ← 缺 load 则该集合检索返回空（静默）
  ▼
Milvus 集合（每源一个）
```

---

## 4. 问答链路（读取侧，`graph_crag`）

```
run_with_trace(question, thread_id?, save?)
  │
  ▼
┌──────────────┐
│ START        │  state = {question, original_question, messages, session_id}
└──────┬───────┘
       ▼
  route_question ──── query_route_chain：LLM + function_calling，三选一
       │               （知识库主题清单从 DOC_SOURCES 的 topic 动态生成）
       │
       ├── "vectorstore" ──► retrieve ──► grade_documents（逐块 yes/no）
       │                        ▲              │
       │                        │              ├─ 有保留 ──► generate
       │                        │              │                 │
       │                        │              │                 ▼
       │                        │              │        grade_generation（幻觉 + 匹配度）
       │                        │              │            │      │        │
       │                        │              │     useful     not       give_up
       │                        │              │      END    supported      │
       │                        │              │                 │         ▼
       │                        │              │                 └─► generate（重试）   END
       │                        │              │            not useful
       │                        │              │                 │
       │                        │              │                 ▼
       │                        │              │          transform_query（改写，≤2 次）
       │                        │              └─────────────────┘
       │                        │
       │                   transform_query（≤2 次）
       │                        │
       │                   ┌────┴──────────────┐
       │                   │ 改写额度用尽仍 0 保留 │
       │                   ▼                   │
       │            generate_fallback ─► END   │
       │            （强制加「非知识库」抬头）   │
       │                                      │
       ├── "web_search" ──► web_search ──► generate ──► grade_generation
       │                        （引用带真实 URL）         │
       │                                          useful → END
       │                                          失败   → give_up → END（不再转查库）
       │
       └── "direct" ──► generate_direct ──► END（不检索、不判分）

终止保护：give_up 边 + recursion_limit=20（防环）
```

**四条分支的判据**（都在 `query_route_chain` 的 system prompt 里）：

| 分支 | 判据 | 耗时量级 |
|---|---|---|
| `vectorstore` | 命中知识库主题（Milvus 类 / 论文类 / 手册类 / 各主题目录名） | 检索 + 5~9 次 LLM |
| `web_search` | 实时信息（天气、股价、新闻） | 1 次联网 + 2 次 LLM |
| `direct` | 寒暄 / 追问上文 / 写作翻译 / 通用概念 | 2 次 LLM |
| `fallback` | **不是路由结果**，是 vectorstore 分支失败后的兜底 | 2~9 次 LLM |

---

## 5. 检索链路（跨集合融合）

```
retriever_tools.retriever           ← 模块级单例：进程启动时构建一次
  │  构建时读 DOC_SOURCES，检查集合是否存在（不存在则跳过并告警）
  ▼
MultiCollectionRetriever
  ├─ 对每个集合：MilvusVectorSave.as_retriever(
  │        k=PER_COLLECTION_K(12), fetch_k=FETCH_K_PER_ROUTE(24),
  │        ranker=RRFRanker(), expr='category == "content"')
  │      └─ 每集合内部：dense(HNSW/IP, ef=128) + sparse(BM25) → 集合内 RRF
  ├─ 汇总所有集合结果 → 跨集合 RRF → 取前 FINAL_K(36)
  └─ 单集合失败**不影响**其它集合（try/except + warning）
```

两个**参数陷阱**（都已在代码里处理）：

* `ef` 必须 ≥ `fetch_k`：langchain_milvus 默认 `ef=10`，pymilvus 3.x 会以
  `ef(10) should be larger than k(24)` 拒绝搜索 → `milvus_db.connect()` 覆盖为 **128**；
* 集合必须 `load`：新集合入库后未加载，查询会超时/失败 → `ingest` 结尾自动 load。

---

## 6. 多轮记忆（三层，各管一段）

```
第 N 轮对话
   │
   ├─ ① 最近 13 轮原文      checkpoints.db（SqliteSaver，按 thread_id 隔离）
   │                        generate_node._format_history() → 注入 prompt
   ├─ ② 更早内容摘要         state["summary"]（>26 轮触发一次 LLM 摘要）
   └─ ③ 历史语义索引         kb_session_history_bge_large_1024
                            lookup_history(session_id, question) → 找回掉出窗口的具体数值
                            record_turn() 在后台单线程队列写入（不阻塞返回）
```

**只存对话不存文档**：三层记忆都只承载"说过的话"，文档语料走另一条线（第 3 节）。
`KB_HISTORY=0` 可关闭第 ③ 层（短会话场景零开销）。

---

## 7. 评测链路

```
Milvus 集合
   │  eval/export_chunks.py                 导出块清单（含一致性防呆）
   ▼
eval/chunk_index.json  [{id, text, filename, title, source, collection}]
   │  eval/gen_testset.py --per-collection N
   │    每块 → LLM 出题 + 抽「原文要点」→ 本地校验要点必须能在原文匹配
   ▼
eval/golden_auto.jsonl  [{id, q, answer, key_points, source_blocks, expect_doc}]
   │  eval/run_eval.py [--with-generation] [--verbose] [--golden P]
   ▼
eval/metrics.py
   ├─ Layer 0（确定性，0 LLM）：score_case → 要点命中 → aggregate_layer0 → gate
   │      recall / hit_rate / （precision 仅块 ID 口径）
   ├─ Layer 1（LLM，可选）：generate_answer → judge_faithfulness / judge_answer_relevancy
   └─ 门禁：PASS / WARN / FAIL / INCONCLUSIVE（未完成 ≠ 不合格，且不落盘）
   ▼
eval/baseline_auto.json + history_auto.jsonl（按题集派生，不同题集互不覆盖）
```

---

## 8. 六个不可破坏的契约

| # | 契约 | 破坏后果 |
|---|---|---|
| 1 | **一个目录 = 一个源 = 一个集合** | 混装导致跨域召回污染；按格式拆目录则与人的归档习惯冲突 |
| 2 | `parser='auto'` 按扩展名逐文件分派 | 写死单一 parser 会让混放目录里的其它格式全部漏收 |
| 3 | HNSW `ef=128`（不要用默认 10） | `ef > k` 校验失败 → 该集合检索全空，且异常被容错吞掉 |
| 4 | 入库后必须 `load_collection` | 新集合查询超时/失败，表现为「该集合的题全部 0 召回」 |
| 5 | 输出必须有 `grounded` + `source_type`，fallback 抬头由**代码**强制加 | 用户分不清「知识库内容」与「模型通用知识」，等于误导 |
| 6 | 评测判据用 `key_points`，不用块 ID；拒答题 `recall=None` | 换语料后评测全灭（块指纹失效）；拒答题把 recall 压低 6 个百分点 |

---

## 9. 改哪里（功能 → 文件）

| 想改 | 动这些文件 |
|---|---|
| 新增文档格式 | `documents/*_parser.py` + `write_milvus._EXT_TO_PARSER` + `env_utils.AUTO_FORMATS` |
| 切分粒度 | `documents/markdown_parser.py`（阈值常量）+ `tools/semantic_splitter.py` |
| 表格处理 | `documents/table_handler.py`、`tools/section_merge.py`、`tools/table_query.py` |
| 检索召回量 / 融合 | `tools/retriever_tools.py`（`PER_COLLECTION_K` / `FINAL_K` / ranker） |
| 向量索引参数 | `documents/milvus_db.py`（HNSW 参数、`ef`、schema、维度自检） |
| 精排 | `tools/reranker_tools.py` |
| 路由策略 | `graph_crag/query_route_chain.py`（prompt + Literal）+ `crag_graph.route_question` |
| 题块相关性闸门 | `graph_crag/grade_documents_node.py` |
| 幻觉 / 匹配度判分 | `graph_crag/grade_hallucinations_chain.py`、`grade_answer_chain.py` |
| 答案风格 / 引用约束 | `graph_crag/generate_node.py`（`CITE_PROMPT`） |
| 降级作答话术 | `graph_crag/direct_node.py`（`FALLBACK_NOTICE` / `_SYSTEM_FALLBACK`） |
| 记忆与摘要 | `utils/checkpointer.py`、`utils/session_store.py`、`tools/history_store.py` |
| 引用解析与校验 | `utils/citation.py` |
| 评测指标 / 门禁 | `eval/metrics.py`；阈值 `eval/thresholds.json` |
| 自动出题 | `eval/gen_testset.py` |
| HTTP 契约 | `rag_kb/api/app.py`（`AskResponse` 的字段就是对外契约） |
| CLI 命令 | `main.py`（顶部 docstring 与 `EPILOG` 是同一份速查表，改一处要改两处） |

---

## 10. 启动时序（排查"为什么慢/为什么起不来"）

[2026-11] 重资源已改为**惰性单例**：import 只定义函数，首次使用才构造。

```
import main                                   耗时（实测）
  ├─ rag_kb.utils.env_utils                   0.02s   读 .env；解析 DOC_SOURCES
  ├─ rag_kb.llm_models.all_llm                1.55s   构造 LLM 客户端（不联网）
  ├─ rag_kb.tools.retriever_tools             1.35s   **不加载模型**
  └─ rag_kb.graph_crag.crag_graph             2.73s   **不编译图、不加载模型**

显式初始化（首次用到时才发生）
  get_embedding_info()      import torch 探测设备              ~3s
  get_global_retriever()    加载 bge-large-zh-v1.5 + 连 Milvus  ~25s
  get_graph()               编译工作流图                        ~0.1s

预热入口
  api lifespan 的 _warmup() 显式调用上述三者（启动时完成，不转嫁给首个请求）
  CLI                       由首次使用自动触发（chat / ask / ingest）

耗时热点：模型加载 25s（进程级一次）> 判分 13s/题（36 块）> 检索 0.1~0.5s
```

> 惰性化的动机：原来 `import` 就加载模型（17.9~18.5s），`--workers N` 时每个进程
> 各加载一份（N × 1.3GB 显存），任何脚本只要 import 到相关模块都白等 20s。
