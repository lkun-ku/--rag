"""TDD：embedding 设备选择与批大小（P0 的配置层）。

## 为什么需要这道测

把 embedding 从 CPU 换到 GPU 后，最大的运维风险不是慢，而是**静默回落**：
驱动更新 / 远程桌面 / 显存被占时 `torch.cuda.is_available()` 会变 False，
若代码默默用 CPU，性能掉 40x 却没有任何信号。所以「设备如何决定」必须是
一个**可测的纯函数**，而不是散落在模块级的三元表达式里。

## 契约

    C1 未设置 EMBEDDING_DEVICE：cuda 可用 -> cuda，否则 cpu
    C2 显式 cpu：即使 cuda 可用也用 cpu
    C3 显式 cuda 但不可用：**回落 cpu**（不允许直接崩）
    C4 大小写/空格容错（' CUDA ' -> cuda）
    C5 非法值（'tpu'）：按「未设置」处理，不抛异常
    C6 批大小跟设备走：GPU=16、CPU=20（实测：GPU 批 32 比 16 慢 1.5x；CPU 批 32 慢 35%）
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

passed = 0
failures = []


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print("  [OK ] %s" % name)
    else:
        failures.append(name)
        print("  [**FAIL**] %s  期望=%r 实际=%r" % (name, want, got))


print("=" * 96)
print("embedding 设备选择（P0 配置层）")
print("=" * 96)

from rag_kb.llm_models.embeddings_model import batch_size_for, resolve_device  # noqa: E402

print("\n[C1] 未设置 -> 按 cuda 可用性")
check("cuda 可用 -> cuda", resolve_device(None, True), "cuda")
check("cuda 不可用 -> cpu", resolve_device(None, False), "cpu")

print("\n[C2] 显式 cpu 优先")
check("显式 cpu（cuda 可用）", resolve_device("cpu", True), "cpu")

print("\n[C3] 显式 cuda 但不可用 -> 回落")
check("cuda 不可用时回落 cpu", resolve_device("cuda", False), "cpu")

print("\n[C4] 容错")
check("' CUDA ' 视为 cuda", resolve_device(" CUDA ", True), "cuda")
check("'Cpu' 视为 cpu", resolve_device("Cpu", True), "cpu")

print("\n[C5] 非法值按自动处理")
check("'tpu' + cuda 可用 -> cuda", resolve_device("tpu", True), "cuda")
check("'tpu' + cuda 不可用 -> cpu", resolve_device("tpu", False), "cpu")

print("\n[C6] 批大小跟设备走")
check("GPU 批大小", batch_size_for("cuda"), 16)
check("CPU 批大小", batch_size_for("cpu"), 20)

print("\n" + "=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：%s" % failures)
print("=" * 96)
sys.exit(0 if not failures else 1)
