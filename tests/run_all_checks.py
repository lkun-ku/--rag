"""统一回归入口 —— 一键跑完全部检查。

## 为什么需要它

SSE 的教训：SSE 端点曾长期**没有任何测试覆盖**，它坏到 P1-2 才被发现
（`answer` 事件从未发出）。而当时并不是「没有验证脚本」，而是**没有一个地方能把
所有验证串起来跑** —— 于是新加的路径没人测、改动的路径靠记忆去测。

本脚本把 `eval/` 下的全部检查按「快 / 慢」两档串起来，一次跑完并给统一结论。

## 两档

    --quick   只跑**秒级**的检查（静态扫描 / 模块边界 / 纯逻辑验证），默认档
    --full    加上需要 LLM 或起服务的检查（API+SSE 集成），约 3~5 分钟

## 判定方式

每个检查自己输出「`X/Y 通过`」或 `**FAIL**`；本脚本解析这两者，不重复实现断言。
`deep_scan` 属**报告型**（它给的是清单而非通过/失败），单列为「报告」不参与判定。

用法：
    .venv\\Scripts\\python.exe eval\\run_all_checks.py            # 快速档
    .venv\\Scripts\\python.exe eval\\run_all_checks.py --full      # 完整档
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not pathlib.Path(PY).exists():
    PY = sys.executable

# (名称, 脚本, 档位, 说明)
CHECKS = [
    # **前置门禁**：LLM 端点不可达时，后面所有需要 LLM 的检查都会失败。
    # A/B 那次是跑了 8 分钟才在日志里发现全是超时 —— 这条能在十几秒内给出结论。
    ("LLM 端点", "tests/quality/check_llm.py", "quick", "前置门禁：走 SDK 真实调用（非裸 socket）"),
    ("静态扫描", "tests/quality/deep_scan.py", "report", "AST 13 类规则（报告型，不看通过率）"),
    # 标准评测：32 题检索层指标 + 门禁判定。**回归时必须 `--no-save`** ——
    # 基线 `baseline.json` 只在「正式评测」时更新，回归跑不该改它（否则会掩盖回退）。
    ("标准评测", "eval/run_eval.py", "quick", "32 题检索层指标 + 门禁（回归用 --no-save）"),
    ("模块边界", "tests/quality/module_tests.py", "quick", "66 个模块级边界用例"),
    ("表格处理", "tests/ingest/verify_rowspan.py", "quick", "rowspan/colspan/合并单元格 + 配置边界"),
    ("表格孪生去重", "tests/ingest/test_flat_twin.py", "quick", "is_flat_twin 三维判据（正负例均取自真实样本）"),
    # [ROADMAP 4.40] 同表切片合并：n005「召回率最低」原本 3/3 稳定拒答
    # （表格被切成 4 片，模型看不到完整表），合并后 3/3 答对。
    ("同表切片合并", "tests/ingest/test_table_merge.py", "quick", "同表多切片合并（表头去重/乱序/超限/非表格）"),
    # [ROADMAP 4.48] 语义分块封装：隔离 langchain-experimental sunset 风险
    ("语义分块封装", "tests/ingest/test_semantic_splitter.py", "quick", "SemanticChunker 封装 + 降级告警"),
    # [4.x] 面包屑 / 归属契约：parent_id 缺失时的标题合并正确性
    ("面包屑归属", "tests/ingest/test_markdown_breadcrumb.py", "quick", "标题合并 + 父子归属 + 表块命名"),
    # [ROADMAP 4.49] 按章节聚合（Parent-Child 检索后轻量版，默认关）
    ("按章节聚合", "tests/ingest/test_section_merge.py", "quick", "同章节父子块聚合 + 超限拆分 + 默认关"),
    # [ROADMAP 4.51/4.52] Parent-Child：结构感知切分 + 子块检索父块回查（默认关）
    ("父子块切分", "tests/ingest/test_parent_child_splitter.py", "quick", "按标题切父块 + 段落/表格切子块"),
    ("父子块检索", "tests/retrieval/test_parent_child_retriever.py", "quick", "子块检索 + 父块批量回查 + 命中数排序"),
    # [ROADMAP 4.50] html / txt 解析器（统一转 Markdown 后复用同一管道）
    ("html/txt 解析", "tests/ingest/test_plain_converters.py", "quick", "HTML/TXT -> Markdown + 真实文件名元数据"),
    # [ROADMAP 4.47] 表格结构化查询：聚合类问题交给 pandas 算，不交给 LLM 猜
    ("表格结构化查询", "tests/ingest/test_table_query.py", "quick", "markdown 表解析 + min/max/mean + 降级"),
    # [ROADMAP 4.42] 企业级精排链路：reranker（默认关）+ 分数门槛 + 聚合类深检索
    ("企业级精排", "tests/retrieval/test_rerank.py", "quick", "reranker 精排/门槛/降级 + 聚合类深检索"),
    # [ROADMAP 4.41] 多轮记忆：默认无 checkpointer（零破坏）+ 历史格式化 + 单例
    ("多轮记忆", "tests/workflow/test_memory.py", "quick", "graph_crag 记忆契约（判据/截断/单例/签名）"),
    # [ROADMAP 4.43] 轮次提升到 13 + 总预算约束 + 摘要压缩阈值
    ("历史预算与摘要", "tests/service/test_history_budget.py", "quick", "13 轮/总预算/单条上限/摘要触发"),
    # [ROADMAP 4.44] 会话历史的向量存储与检索（默认开启）
    ("历史向量检索", "tests/service/test_history_store.py", "quick", "历史集合/隔离 expr/TTL 清理/降级"),
    # [ROADMAP 4.45] 会话管理 + 滑动 TTL（新开对话/列表/删除）
    ("会话管理", "tests/service/test_session_store.py", "quick", "sessions 表 CRUD + 6h 滑动 TTL"),
    # [FIX / 代码审查] 这两个脚本此前**没有注册** —— 写了测试但回归里不跑，
    # 等于「有测试等于没测试」（review 的 Standards 轴 A4 发现）。
    ("批量判分契约", "tests/workflow/test_grade_batch.py", "quick", "grade_batch 结构/语义/容错/并发顺序"),
    ("审查修复回归", "tests/quality/test_review_fixes.py", "quick", "代码审查发现的修复"),
    ("documents 类型一致", "tests/workflow/test_web_search_docs_type.py", "quick", "web_search 返回 list（与其它节点一致）"),
    # [FIX / 二次筛查] **这一层此前完全缺失** —— 回归里所有项要么只走检索层
    # （run_eval），要么只测单个函数（test_*.py），**没有任何一项端到端跑图**。
    # 于是「grade_generation 里 `_cfg` 未定义」这种**只有走完整链路才触发**的 P0
    # 能安然通过 10/10 的回归。本项补上这层覆盖（跑 1 题，约 5s）。
    ("完整工作流冒烟", "tests/workflow/test_pipeline_smoke.py", "quick", "端到端跑图（补上缺失的那层覆盖）"),
    ("判分 token 埋点", "tests/workflow/test_grade_callback.py", "quick", "grade_documents 的 config 透传（含 tracer 包装链路）"),
    ("门禁告警集合", "tests/quality/test_gate_precision.py", "quick", "context_precision 只作监控、不参与告警（保留硬门禁）"),
    ("引用校验", "tests/workflow/verify_citations.py", "quick", "越界 / 原文编号 / 幻觉的区分"),
    ("API + SSE", "tests/service/verify_p1_batch.py", "slow", "SSE 事件类型、trace 落盘、disabled 源"),
]


def run_one(name: str, script: str) -> tuple:
    """跑一个检查，返回 (ok_or_None, 摘要)。None 表示「报告型」。"""
    path = ROOT / script
    if not path.exists():
        return False, "**脚本不存在**"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    # 回归用 `--no-save`：`run_eval.py` 默认会把结果写进 `baseline.json`，
    # 而**基线只能在正式评测时更新** —— 回归顺手改基线会掩盖真实回退。
    extra = ["--no-save"] if path.name == "run_eval.py" else []

    t0 = time.time()
    r = subprocess.run(
        [PY, "-u", str(path), *extra],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    out = (r.stdout or "") + (r.stderr or "")
    el = time.time() - t0

    # check_llm.py 的判定（它给的是「可用 / 不可用」而非通过率）
    if "**可用**" in out:
        return True, "可用（%.1fs）" % el
    if "**不可用**" in out:
        return False, "**不可用**（%.1fs）—— 需 LLM 的检查会全部失败" % el

    # run_eval.py 的判定（它给的是「门禁判定: PASS/WARN/FAIL/INCONCLUSIVE」）
    #
    # **INCONCLUSIVE 不是失败** —— 按项目设计（ROADMAP 1.5.3），它表示「**未完成**」
    # 而非「不合格」：生成层指标缺失（未传 `--with-generation` 或 LLM 不可用）时，
    # 即便检索层全达标也会返回它，**不得计为质量回退**。
    # 首版把它和 FAIL 一起判成 False，于是回归报了「失败项：标准评测」——
    # 而那次唯一的实际内容只是「recall 0.7993 落在告警区间」，属设计内的提示。
    # 这里返回 None（报告型），让它不参与判定，但把原因显式打出来。
    mg = re.search(r"门禁判定[:：]\s*(\w+)", out)
    if mg:
        v = mg.group(1)
        tail = re.search(r"门禁判定[:：][^\n]*\n\s*-\s*(.+)", out)
        detail = ("（%s）" % tail.group(1).strip()[:64]) if tail else ""
        if v == "INCONCLUSIVE":
            return None, "门禁 INCONCLUSIVE —— 生成层缺失，非回退%s（%.1fs）" % (detail, el)
        return (v in ("PASS", "WARN")), "门禁 %s%s（%.1fs）" % (v, detail, el)

    m = re.search(r"(\d+)\s*/\s*(\d+)\s*通过", out)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ok = a == b
        return ok, "%d/%d 通过（%.1fs）" % (a, b, el)
    if "**FAIL**" in out or "FAIL:" in out:
        return False, "输出含 FAIL（%.1fs）" % el
    return None, "已运行，无通过率可解析（%.1fs）" % el


def main() -> int:
    ap = argparse.ArgumentParser(description="统一回归入口")
    ap.add_argument("--full", action="store_true", help="含慢检查（LLM / 起服务）")
    args = ap.parse_args()

    print("=" * 100)
    print("统一回归检查（%s档）" % ("完整" if args.full else "快速"))
    print("=" * 100)
    print()

    results = []
    for name, script, tier, desc in CHECKS:
        if tier == "slow" and not args.full:
            results.append((name, None, "跳过（需 --full）"))
            print("  [跳过] %-12s %s" % (name, desc))
            continue
        print("  [运行] %-12s %s ..." % (name, desc), flush=True)
        ok, summary = run_one(name, script)
        if tier == "report":
            ok = None
        results.append((name, ok, summary))
        mark = "OK  " if ok else ("报告" if ok is None else "**FAIL**")
        print("         [%s] %s" % (mark, summary))

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    judged = [(n, o, s) for n, o, s in results if o is not None]
    n_ok = sum(1 for _n, o, _s in judged if o)
    for name, ok, summary in results:
        if ok is None:
            print("  [报告] %-14s %s" % (name, summary))
        else:
            print("  [%s] %-14s %s" % ("OK  " if ok else "FAIL", name, summary))
    print()
    if judged:
        print("  判定项合计 %d/%d 通过" % (n_ok, len(judged)))
    print("=" * 100)

    fails = [n for n, o, _s in judged if not o]
    if fails:
        print()
        print("  **失败项**: %s" % ", ".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
