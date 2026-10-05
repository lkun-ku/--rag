"""从语料块**自动生成评测集** —— 换材料后不用再手写题。

## 为什么需要它

原 `build_dataset.py` 里的 32 道题是**手写死**的（`source_blocks=["0b0b3ee716af"]`），
换一批文档后：题目与新领域无关，块 ID 也全对不上，评测直接归零。

本脚本把「出题」变成数据驱动的：

    块清单（export_chunks.py）-> 采样 -> LLM 出题 + 抽要点 -> golden_auto.jsonl

## 关键设计：为什么要抽「原文要点」

评测的命中判据改成 `key_points`（见 `metrics.score_case`）：判断「答案里的那个数字/
结论有没有出现在召回块中」，而不是「是不是某一块」。

所以要点必须**能在块文本里被字符串匹配到**（数字、专名、短语），不能是模型的概括。
prompt 里对这点做了硬性要求，并在写盘前**本地校验**：匹配不上的要点直接丢弃，
丢光了就整题作废。这样产出的评测集不会因为「要点写得太抽象」而必然低分。

## 用法

    # 每个集合出 3 题（默认），写 golden_auto.jsonl
    .venv\\Scripts\\python.exe eval\\gen_testset.py

    # 每集合 8 题，指定输出
    .venv\\Scripts\\python.exe eval\\gen_testset.py --per-collection 8 --out eval/golden_auto.jsonl

## 与手工题的关系

产物是**独立文件**，不覆盖 `golden_dataset.jsonl`。可以：
  * 只用自动题（换材料场景）；
  * 手工题 + 自动题合并（把两份 jsonl 拼起来即可，`run_eval` 支持 mixed）。
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sy

# 解释器护栏：裸 python（Anaconda）跑本脚本会 ModuleNotFoundError，自动切到项目 .venv
_HERE = _pl.Path(__file__).resolve().parent
if str(_HERE) not in _sy.path:
    _sy.path.insert(0, str(_HERE))
import venv_guard  # noqa: E402

venv_guard.ensure_venv()

import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
INDEX = HERE / "chunk_index.json"
OUT = HERE / "golden_auto.jsonl"
sys.path.insert(0, str(HERE))

# 太短的块出不了题；纯表格块容易产出「跨行比较」类问题，与手工题型重复
MIN_CHARS = 120
MAX_TABLE_RATIO = 0.5


class SynthCase:
    """结构化输出的字段定义（延迟导入 pydantic，避免 `--help` 也要装依赖）。"""


def _build_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from rag_kb.llm_models.all_llm import llm
    from rag_kb.utils.env_utils import LLM_STRUCTURED_METHOD

    class SynthCase(BaseModel):  # noqa: D101
        empty: bool = Field(False, description="资料信息不足、出不了好问题时置 true")
        q: str = Field("", description="一个只能靠该资料回答的具体问题")
        answer: str = Field("", description="答案，不超过 60 字")
        key_points: list[str] = Field(default_factory=list, description="2~3 个**资料原文中出现**的片段")

    system = """你是评测集生成器。给定一段资料，出 1 道「只能靠这段资料回答」的题。

要求：
1. 问题要**具体**（问到某个数字、条件、结论、对比），不要问「这段讲了什么」；
2. `key_points` 给 2~3 个，**必须逐字出现在资料原文里**（数字、专有名词、结论短语），
   不要改写、不要概括 —— 它们会被当作字符串做匹配打分；
3. `answer` 简洁，不超过 60 字；
4. 这段资料信息太少、或出不了客观题时，把 `empty` 置为 true。"""

    human = """文件名：{filename}
章节：{title}

资料：
{text}"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm.with_structured_output(SynthCase, method=LLM_STRUCTURED_METHOD)


def _table_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln.strip().startswith("|")) / len(lines)


def _pick(blocks: list, n: int, rng: random.Random) -> list:
    """均匀采样：先打散再按顺序取，避免全落在同一章节。"""
    pool = list(blocks)
    rng.shuffle(pool)
    return pool[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="从块清单自动生成评测集")
    ap.add_argument("--per-collection", type=int, default=3, help="每个集合生成多少题")
    ap.add_argument(
        "--collection", default="", help="只给指定集合出题（如 kb_frontend_bge_large_1024）"
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--seed", type=int, default=20261103, help="采样随机种子（保证可复现）")
    ap.add_argument("--force", action="store_true", help="跳过「索引 vs 向量库」一致性检查")
    args = ap.parse_args()

    idx_path = pathlib.Path(args.index)
    if not idx_path.exists():
        print(f"缺少 {idx_path}，请先运行：python eval/export_chunks.py")
        return 2

    blocks = json.loads(idx_path.read_text(encoding="utf-8"))

    # [2026-11] 防呆：块索引是 export_chunks.py 导出时的**快照**。
    # 入库新材料后忘记重新导出，就会拿旧索引出题 —— 新集合的块一道题都不会有
    # （实测：AI-agent 入库 2654 块后直接出题，200 题全部来自另一个集合）。
    # 这里连 Milvus 对比集合清单，不一致就拒绝出题；--force 可跳过检查。
    if not args.force:
        try:
            from pymilvus import MilvusClient

            from rag_kb.tools.history_store import HISTORY_COLLECTION
            from rag_kb.utils.env_utils import MILVUS_URI

            # 会话历史集合是记忆层，不参与出题，从对比中排除
            live = set(MilvusClient(uri=MILVUS_URI).list_collections()) - {HISTORY_COLLECTION}
            indexed = {b.get("collection") for b in blocks}
            missing = sorted(c for c in live - indexed if c)
            if missing:
                print("[阻塞] 块索引落后于向量库：以下集合尚未收录 -> %s" % missing)
                print("       先运行：python eval/export_chunks.py")
                print("       确认要用旧索引时加 --force")
                return 2
        except Exception:
            pass  # Milvus 不可用（离线重跑场景）时不阻塞

    if args.collection:
        blocks = [b for b in blocks if b.get("collection") == args.collection]
    groups: dict = {}
    for b in blocks:
        text = b.get("text") or ""
        if len(text) < MIN_CHARS or _table_ratio(text) > MAX_TABLE_RATIO:
            continue
        groups.setdefault(b.get("collection") or "unknown", []).append(b)

    print("=" * 84)
    print("自动生成评测集")
    print("=" * 84)
    for c, rows in groups.items():
        print("  %-40s 可用 %d 块" % (c, len(rows)))
    print()

    chain = _build_chain()
    rng = random.Random(args.seed)
    out_rows, n_empty, n_bad_points = [], 0, 0

    for coll, rows in groups.items():
        picked = _pick(rows, args.per_collection, rng)
        for b in picked:
            text = b["text"]
            try:
                res = chain.invoke({"filename": b.get("filename", ""), "title": b.get("title", ""), "text": text})
            except Exception as e:  # noqa: BLE001
                print("  [跳过] 生成失败 %s: %s" % (type(e).__name__, str(e)[:80]))
                continue

            if getattr(res, "empty", False) or not (res.q or "").strip():
                n_empty += 1
                continue

            # 本地校验：要点必须能在原文里匹配到，否则这道题必然判低分
            points = [p.strip() for p in (res.key_points or []) if p and p.strip() and p.strip() in text]
            if not points:
                n_bad_points += 1
                continue

            out_rows.append(
                {
                    "id": "auto-%03d" % (len(out_rows) + 1),
                    "source": coll.replace("kb_", "").replace("_bge_large_1024", ""),
                    "q": res.q.strip(),
                    "answer": (res.answer or "").strip(),
                    "key_points": points,
                    # 保留块 ID 仅用于诊断（口径以 key_points 为准）
                    "source_blocks": [b["id"]],
                    "expect_doc": b.get("filename", ""),
                    "note": "自动生成 | %s" % (b.get("title") or "")[:60],
                    "kind": "自动",
                }
            )
            print("  [OK] %-12s %s" % (coll[:12], res.q.strip()[:52]))

    out = pathlib.Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print()
    print("生成 %d 题 -> %s" % (len(out_rows), out))
    print("丢弃：资料不足 %d 道 / 要点无法匹配 %d 道" % (n_empty, n_bad_points))
    print("下一步：python eval/run_eval.py --no-save   （先看真实水平，再调 thresholds.json）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
