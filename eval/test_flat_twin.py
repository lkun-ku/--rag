"""TDD 测试：`is_flat_twin(flat_text, table_chunks) -> bool`

判断「一段扁平文本是否为『已产出结构化块的表格』的孪生」——
若是，调用方可以安全地丢弃该扁平文本（同一张表不必以两种形态同时入库）。

## 期望值的来源（避免同义反复 —— TDD 的硬要求）

**两个核心用例的文本都是库里的真实样本**（`eval/measure_flat_twin.json`），
期望值是我**独立判断**的，不是按被测代码反推：

| 用例 | 结构化 | 扁平 | 正向cov | 反向cov | 期望 | 为什么 |
|---|---|---|---|---|---|---|
| 正例 `capability_matrix/来源说明` | 338 | 277 | 1.00 | **1.00** | **True** | 扁平 = 结构化去掉 `\\|`/`---`，纯孪生 |
| 负例 `index_spec/6.2 依赖组件版本` | 265 | 389 | 1.00 | **0.41** | **False** | 扁平里多了「6 部署环境要求 Before installing…」「出处：milvus.io…」 |

**这一对是本次最关键的发现**：两者**正向覆盖率都是 1.00**（结构化块的值都在扁平里），
但负例的扁平块**已被父标题合并逻辑并入了周围正文** —— 直接删会丢内容。
所以判据必须有**反向覆盖**这一维。

## 阈值依据（实测，不是拍的）

`eval/measure_flat_twin.py` 对库里 15 组的实测：

    正向覆盖率（结构化长值 -> 扁平）  min 0.98  中位 1.00  max 1.00
    反向覆盖率（扁平长词 -> 结构化）  min 0.41  中位 1.00  max 1.00
    长度比（扁平 / 结构化总长）        min 0.74  中位 0.84  max 1.47

取 **正向 >= 0.98**、**反向 >= 0.95**、**长度比 ∈ [0.5, 2.0]**。
反向阈值 0.95 落在实测的空隙里（纯孪生全为 1.00，含额外正文最高 0.69）。

> **不能**用「扁平 / 结构化最长块」：该比值实测跨 0.74~7.13
> （一个表格元素会被切成多个行段，分母随切段数变化）。
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
os.environ["HF_HUB_OFFLINE"] = "1"

failures = []
passed = 0


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%s  实际=%s" % (name, want, got))


# ---------------------------------------------------------------------------
# 真实样本（取自库；见文件头表格）
# ---------------------------------------------------------------------------
# 正例：milvus_capability_matrix.xlsx.md / 来源说明
POS_STRUCT = (
    "表格：来源说明\n"
    "列名：字段 | 说明\n"
    "\n"
    "| 字段 | 说明 |\n"
    "|---|---|\n"
    "| 说明列 | 全部为官方中文原文逐字引文，未改写；来源见“出处”列 |\n"
    "| 出处 | 墨天轮转载·Milvus 官方《官宣 Milvus 2.6 正式开源》（modb.pro/db/1934464950274109440） |\n"
    "| 出处 | Zilliz 官方中文博客《十倍成本优化，Milvus 2.5到2.6升级官方手把手教程》"
    "（zilliz.com.cn/blog/Milvus-2.5-to-2.6-tenfold-cost-optimization-guide） |\n"
    "| 能力项/类别/引入版本 | 依据上述官方发布说明整理 |\n"
    "| 官方指标 | 官方公布的实测数字，逐字保留 |"
)
POS_FLAT = (
    "来源说明 字段 说明 说明列 全部为官方中文原文逐字引文，未改写；来源见“出处”列 "
    "出处 墨天轮转载·Milvus 官方《官宣 Milvus 2.6 正式开源》（modb.pro/db/1934464950274109440） "
    "出处 Zilliz 官方中文博客《十倍成本优化，Milvus 2.5到2.6升级官方手把手教程》"
    "（zilliz.com.cn/blog/Milvus-2.5-to-2.6-tenfold-cost-optimization-guide） "
    "能力项/类别/引入版本 依据上述官方发布说明整理 官方指标 官方公布的实测数字，逐字保留"
)

# 负例：milvus_index_spec.docx.md / 6.2 依赖组件版本
NEG_STRUCT = (
    "表格：6.2 依赖组件版本\n"
    "列名：Software | Version | Note\n"
    "\n"
    "| Software | Version | Note |\n"
    "|---|---|---|\n"
    "| etcd | 3.5.0 | Disk performance is critical to etcd. It is highly recommended "
    "that you use local NVMe SSDs. |\n"
    "| MinIO | RELEASE.2024-12-18T13-15-44Z |  |\n"
    "| Pulsar | 2.8.2 |  |"
)
NEG_FLAT = (
    "6 部署环境要求 Before installing a Milvus Standalone instance, check your hardware "
    "and software to see if they meet the requirements. -> 6.2 依赖组件版本 "
    "Software Version Note etcd 3.5.0 Disk performance is critical to etcd. It is highly "
    "recommended that you use local NVMe SSDs. MinIO RELEASE.2024-12-18T13-15-44Z "
    "Pulsar 2.8.2 出处：milvus.io v2.6.x — Requirements for Installing Milvus Standalone"
    "（逐字引文）"
)

print("=" * 96)
print("is_flat_twin() 测试  ——  正例/负例文本均取自库中真实样本")
print("=" * 96)
print()

try:
    from rag_kb.documents.table_handler import is_flat_twin
except ImportError as e:
    print("  [**FAIL**] 无法 import is_flat_twin（尚未实现）: %s" % e)
    print()
    print("Red 阶段符合预期：函数不存在，测试无法运行。")
    sys.exit(1)

print("[正例] 纯孪生（反向覆盖 1.00）-> 期望 True")
check("capability_matrix/来源说明", is_flat_twin(POS_FLAT, [POS_STRUCT]), True)

print()
print("[负例·核心] 扁平里含结构化没有的正文（反向覆盖 0.41）-> 期望 False")
check("index_spec/6.2 依赖组件版本（含「6 部署环境要求…」）", is_flat_twin(NEG_FLAT, [NEG_STRUCT]), False)

print()
print("[负例] 扁平被截断（缺结构化块里的长值）-> 期望 False")
check("缺一个长值", is_flat_twin(POS_FLAT.replace("官方公布的实测数字，逐字保留", ""), [POS_STRUCT]), False)
check("只剩表头", is_flat_twin("来源说明 字段 说明", [POS_STRUCT]), False)

print()
print("[负例] 长度比超出 [0.5, 2.0] -> 期望 False")
check("扁平极短", is_flat_twin("逐字保留", [POS_STRUCT]), False)
check("扁平极长（同内容重复多遍）", is_flat_twin(POS_FLAT + " " + POS_FLAT + " " + POS_FLAT, [POS_STRUCT]), False)

print()
print("[负例] 边界输入 -> 期望 False")
check("扁平为空", is_flat_twin("", [POS_STRUCT]), False)
check("结构化为空", is_flat_twin(POS_FLAT, []), False)
check("都为空", is_flat_twin("", []), False)
check(
    "结构化块没有可提取的长值",
    is_flat_twin("来源说明 字段 说明", ["表格：来源说明\n列名：字段 | 说明\n\n| 字段 | 说明 |\n|---|---|"]),
    False,
)
check("扁平与结构化是两份无关内容", is_flat_twin("Milvus 支持混合检索与稀疏向量，性能优异。", [POS_STRUCT]), False)

print()
print("=" * 96)
# 输出格式与项目其他测试保持一致（`N/M 通过`）—— `run_all_checks.py` 用它解析通过率。
# 首版写成「结果：11 通过，0 失败」，回归里因此显示「无通过率可解析」（报告型）。
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：")
    for f in failures:
        print("   - %s" % f)
print("=" * 96)
sys.exit(0 if not failures else 1)
