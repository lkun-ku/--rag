"""TDD：增量入库的「变更计划」（P1 的核心判定，纯函数）。

## 为什么先测这个

增量入库的正确性完全取决于一件事：**哪些文件需要重做、哪些要删掉**。
判错的两个方向都很难受：
  * 漏判变更 -> 库里是旧内容（用户以为更新了）；
  * 多判变更 -> 每次全量重算（等于没做增量）。
所以把「比较文件指纹 + 与上次清单核对」抽成纯函数，用固定输入断言输出。

## 契约

    C1 全部是新文件（清单为空）-> 全部待处理，且没有要删的
    C2 内容未变 -> 全部跳过
    C3 单个文件内容变化 -> 只有它待处理
    C4 文件从磁盘消失 -> 进 removed（要删库里的旧块）
    C5 只改 mtime、内容不变 -> 仍视为未变（按内容哈希，不看时间）
    C6 路径与清单键都按 basename 对齐（库里的 filename 字段存的就是 basename）
"""

import hashlib
import pathlib
import sys
import tempfile

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
print("增量入库的变更计划")
print("=" * 96)

from rag_kb.documents.write_milvus import file_fingerprint, plan_incremental  # noqa: E402

tmp = pathlib.Path(tempfile.mkdtemp(prefix="inc_plan_"))
a, b = tmp / "a.md", tmp / "b.md"
a.write_text("# A\n\n内容 A\n", encoding="utf-8")
b.write_text("# B\n\n内容 B\n", encoding="utf-8")
files = [str(a), str(b)]

print("\n[C1] 首次运行（清单为空）")
to_ingest, unchanged, removed = plan_incremental(files, {})
check("全部待处理", sorted(pathlib.Path(p).name for p in to_ingest), ["a.md", "b.md"])
check("没有跳过", unchanged, [])
check("没有要删的", removed, [])

known = {pathlib.Path(p).name: file_fingerprint(p) for p in files}

print("\n[C2] 内容未变")
to_ingest, unchanged, removed = plan_incremental(files, known)
check("无需处理", to_ingest, [])
check("全部跳过", sorted(pathlib.Path(p).name for p in unchanged), ["a.md", "b.md"])
check("没有要删的", removed, [])

print("\n[C3] 只有一个文件变化")
b.write_text("# B\n\n内容 B 改过了\n", encoding="utf-8")
to_ingest, unchanged, removed = plan_incremental(files, known)
check("只处理 b.md", [pathlib.Path(p).name for p in to_ingest], ["b.md"])
check("a.md 跳过", [pathlib.Path(p).name for p in unchanged], ["a.md"])
check("没有要删的", removed, [])

print("\n[C4] 文件被删除")
b.unlink()
to_ingest, unchanged, removed = plan_incremental([str(a)], known)
check("无需处理", to_ingest, [])
check("a.md 跳过", [pathlib.Path(p).name for p in unchanged], ["a.md"])
check("b.md 需要从库里删掉", removed, ["b.md"])

print("\n[C5] 只改 mtime，内容不变")
import os
import time

time.sleep(0.01)
os.utime(a, None)
check("指纹不变", file_fingerprint(str(a)), known["a.md"])
to_ingest, _, _ = plan_incremental([str(a)], {"a.md": known["a.md"]})
check("仍视为未变", to_ingest, [])

print("\n[C6] 清单键 = basename")
check("指纹是 40 位 sha1", len(known["a.md"]), 40)
check("同一内容指纹稳定", hashlib.sha1(a.read_bytes()).hexdigest(), known["a.md"])

print("\n" + "=" * 96)
print("结果：%d/%d 通过" % (passed, passed + len(failures)))
if failures:
    print("失败项：%s" % failures)
print("=" * 96)
sys.exit(0 if not failures else 1)
