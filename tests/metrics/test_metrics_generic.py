"""评测指标通用化 —— 契约测试（不依赖任何语料内容）。

## 为什么需要它

原评测把「判据」和「具体材料」焊死了：

    source_blocks = md5(page_content)[:12]

块内容改一个字，标注就失效；换一批文档，58 个标注全废，recall 直接归零。
本测试用**纯构造数据**验证新的匹配接口——它必须与语料内容无关。

## 契约

    C1 有 `key_points` 时按**要点命中**算 recall（不看块 ID）
    C2 无 `key_points` 时退回 `source_blocks`（兼容旧 golden）
    C3 两者都为空（拒答题）→ `recall is None`，不参与聚合
    C4 `expect_doc` 命中 → `doc_hit == 1`；未命中 → 0
    C5 `aggregate_layer0` **跳过** `recall is None` 的题（分母不含拒答题）

## 运行

    python tests/metrics/test_metrics_generic.py
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import metrics  # noqa: E402

failures = []
passed = 0


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%r  实际=%r" % (name, want, got))


def recalled(*items):
    """构造检索结果：[{"id","text","filename"}]。"""
    return [{"id": i, "text": t, "filename": f} for i, t, f in items]


print("=" * 96)
print("评测指标通用化")
print("=" * 96)

# ---------------------------------------------------------------------------
# C1 要点命中：判据是「答案里的那个数/结论在不在召回块里」，与块 ID 无关
# ---------------------------------------------------------------------------
print()
print("[C1] 有 key_points 时按要点命中（与语料内容无关）")
case = {
    "q": "某配置的 QPS 是多少",
    "answer": "9575.69",
    "key_points": ["9575.69", "force_merge"],
    "source_blocks": ["will-never-match"],  # 故意给一个必然对不上的旧 ID
}
# 召回的块里包含这两个要点（内容完全可以是新语料）
r = metrics.score_case(
    case,
    recalled(
        ("aaa", "Milvus-16c64g-sq4u-fp16-force_merge 的 QPS 为 9575.69", "新文档.md"),
        ("bbb", "无关内容", "别的.md"),
    ),
)
check("recall == 1.0（两个要点都命中）", r["recall"], 1.0)
check("basis == 'key_points'", r["basis"], "key_points")
check("旧 source_blocks 不再参与判定", r["recall"] == 1.0, True)

r2 = metrics.score_case(case, recalled(("aaa", "只有 9575.69 这个数", "新文档.md")))
check("只命中 1/2 要点 -> recall == 0.5", r2["recall"], 0.5)
check("missed_points 记录漏掉的要点", r2["missed_points"], ["force_merge"])

r3 = metrics.score_case(case, recalled(("ccc", "完全无关", "x.md")))
check("一个都没命中 -> recall == 0.0", r3["recall"], 0.0)
check("hit == 0", r3["hit"], 0)

# ---------------------------------------------------------------------------
# C2 无 key_points 时退回旧逻辑（保证历史 golden 仍可跑）
# ---------------------------------------------------------------------------
print()
print("[C2] 无 key_points 时退回 source_blocks")
legacy = {"q": "旧题", "source_blocks": ["id-1", "id-2"]}
r4 = metrics.score_case(legacy, recalled(("id-1", "正文", "a.md"), ("id-9", "正文", "b.md")))
check("recall == 0.5（命中 1/2 块）", r4["recall"], 0.5)
check("basis == 'source_blocks'", r4["basis"], "source_blocks")

# ---------------------------------------------------------------------------
# C3 拒答题：两者都空 -> recall 为 None（关键：不是 0.0）
# ---------------------------------------------------------------------------
print()
print("[C3] 拒答题不参与召回统计")
refusal = {"q": "Milvus 的订阅价格是多少？", "key_points": [], "source_blocks": []}
r5 = metrics.score_case(refusal, recalled(("zzz", "随便什么", "x.md")))
check("recall is None（而非 0.0）", r5["recall"], None)
check("is_refusal_case 标记为真", r5.get("is_refusal_case"), True)

# ---------------------------------------------------------------------------
# C4 文档级命中：换材料后仍有效（文件还在就能判）
# ---------------------------------------------------------------------------
print()
print("[C4] expect_doc 文档级命中")
case_doc = {
    "q": "能力矩阵里列了哪些版本",
    "key_points": ["2.6"],
    "expect_doc": "milvus_capability_matrix.xlsx",
}
r6 = metrics.score_case(
    case_doc, recalled(("aaa", "2.6 版本新增分区键", "milvus_capability_matrix.xlsx"))
)
check("doc_hit == 1", r6["doc_hit"], 1)
r7 = metrics.score_case(case_doc, recalled(("aaa", "2.6 版本新增分区键", "other.md")))
check("文件名不符 -> doc_hit == 0", r7["doc_hit"], 0)
check("无 expect_doc 时 doc_hit is None", metrics.score_case(case, recalled())["doc_hit"], None)

# ---------------------------------------------------------------------------
# C5 聚合跳过拒答题（否则 2 道必错题会拉低 6 个百分点）
# ---------------------------------------------------------------------------
print()
print("[C5] aggregate_layer0 跳过 recall=None")
per_case = [
    {"recall": 1.0, "precision": 0.1, "hit": 1.0},
    {"recall": 0.5, "precision": 0.05, "hit": 1.0},
    {"recall": None, "precision": 0.0, "hit": 0.0},  # 拒答题
]
agg = metrics.aggregate_layer0(per_case)
check("context_recall == 0.75（只算 2 题）", agg["context_recall"], 0.75)
check("hit_rate == 1.0", agg["hit_rate"], 1.0)
check("refusal_cases 统计到 1 题", agg.get("refusal_cases"), 1)

# 全是拒答题时不应除零
agg2 = metrics.aggregate_layer0([{"recall": None, "precision": 0.0, "hit": 0.0}])
check("全为拒答题不除零", agg2["context_recall"], 0.0)

print()
print("=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
