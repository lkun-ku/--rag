"""可行性验证 1：检索排名 与 LLM 判分结果的关系（判分分级方案的核心依据）。

问题：方案 2.2 提出「高分块直接放行、低分块直接丢弃、只对灰区做 LLM 判分」。
     但 RRF 分数是题内相对值（1/60 量级），跨题不可比，因此真正可用的信号是
     **排名位置**。本脚本测：排名第 N 位的块被判为相关的概率是多少。

方法：取 N 道自动题 → 检索 36 块（顺序即 RRF 排名）→ 逐块 LLM 判分 →
     按排名分桶统计 yes 比例。若首尾两端的分布足够纯，分级才可行。

用法：python _probe_grading.py [题目数]
"""

import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

GOLD = ROOT / "eval" / "golden_auto.jsonl"
BUCKETS = [(1, 5), (6, 10), (11, 18), (19, 28), (29, 36)]


def bucket_of(rank: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= rank <= hi:
            return f"{lo}-{hi}"
    return "other"


def main() -> int:
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    all_cases = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 只取当前已入库集合的题（Milvus 重建后仅剩前端集合 eec008a3）
    pool = [c for c in all_cases if "eec008a3" in (c.get("source") or "")] or all_cases
    cases = pool[:n_cases]
    print("选取 %d 题（候选池 %d）" % (len(cases), len(pool)))

    from rag_kb.graph_crag.grade_documents_node import _grade_one
    from rag_kb.tools.retriever_tools import retriever

    stat = {f"{lo}-{hi}": {"yes": 0, "n": 0} for lo, hi in BUCKETS}
    per_case = []
    t_all = time.time()

    for i, case in enumerate(cases, 1):
        q = case["q"]
        docs = retriever.invoke(q)  # 顺序 = 跨集合 RRF 排名
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            flags = list(ex.map(lambda d: _grade_one(q, d)[1], docs))
        cost = time.time() - t0

        for rank, flag in enumerate(flags, 1):
            b = bucket_of(rank)
            stat[b]["n"] += 1
            stat[b]["yes"] += 1 if flag else 0

        kept = sum(flags)
        per_case.append((i, len(docs), kept, cost, q[:40]))
        print("  [%d] 召回 %2d 块，保留 %2d 块，判分 %.1fs | %s" % (i, len(docs), kept, cost, q))

    print()
    print("=" * 78)
    print("按排名分桶的相关性分布（%d 题 / %d 块判分，总耗时 %.1fs）" % (n_cases, sum(v["n"] for v in stat.values()), time.time() - t_all))
    print("=" * 78)
    print("%-10s %8s %8s %8s" % ("排名段", "块数", "判yes", "yes率"))
    for lo, hi in BUCKETS:
        k = f"{lo}-{hi}"
        v = stat[k]
        rate = v["yes"] / v["n"] * 100 if v["n"] else 0
        print("%-10s %8d %8d %7.1f%%" % (k, v["n"], v["yes"], rate))

    total_n = sum(v["n"] for v in stat.values())
    total_yes = sum(v["yes"] for v in stat.values())
    print()
    print("整体保留率: %d/%d = %.1f%%" % (total_yes, total_n, total_yes / total_n * 100))

    # 方案预估：首段放行 + 末段丢弃，只判中间
    head_n = stat["1-5"]["n"]
    head_yes = stat["1-5"]["yes"]
    tail_n = stat["29-36"]["n"]
    mid_n = total_n - head_n - tail_n
    print("若「1-5 名放行 + 29-36 名丢弃 + 中间判分」：")
    print("  放行段误丢: %d 块（%s）" % (head_n - head_yes, "无" if head_n == head_yes else "存在"))
    print("  丢弃段误丢: %d 块（正确判为相关的比例 %.1f%%）" % (stat["29-36"]["yes"], stat["29-36"]["yes"] / tail_n * 100 if tail_n else 0))
    print("  判分调用: %d -> %d，节省 %.0f%%" % (total_n, mid_n, (total_n - mid_n) / total_n * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
