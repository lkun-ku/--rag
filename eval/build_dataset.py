r"""构建 Golden Dataset（评测基线）—— **自动出题已由 gen_testset.py 接管**。

[2026-11] 本脚本原本内置 32 道手工题（`source_blocks` 绑定旧语料的块内容指纹）。
自动出题机制（`gen_testset.py`：LLM 出题 + 原文要点抽取 + 本地可检索性校验）
上线后，这批题对新语料完全失效，硬编码模板随之退役 —— 保留它们只会让
「换语料」这件事看起来还需要手工维护题目。

当前职责：**块索引概览与流程指引**。

换语料后的标准流程：

    python main.py ingest
    python eval\export_chunks.py
    python eval\gen_testset.py --per-collection 5
    python eval\run_eval.py --golden eval\golden_auto.jsonl --no-save

出题要点（为什么 gen_testset 抽取的是「原文要点」而不是块 ID）：
  * 块 ID = `md5(内容)[:12]`，语料重切片/改一个字就失效；
  * 要点是「答案中的数字/结论/专名」，只要关键事实还在语料里就能命中，
    与分块方式、文件重排无关 —— 这才是与语料解耦的判据。
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
INDEX = HERE / "chunk_index.json"


def main() -> int:
    if not INDEX.exists():
        print(f"缺少 {INDEX}，请先运行 eval/export_chunks.py")
        return 2

    index = json.loads(INDEX.read_text(encoding="utf-8"))

    # 新版块索引（export_chunks.py 的平铺列表）-> 打印概览并指引自动出题
    if isinstance(index, list):
        by_coll: dict = {}
        for b in index:
            by_coll[b.get("collection", "?")] = by_coll.get(b.get("collection", "?"), 0) + 1
        print(f"块索引共 {len(index)} 块：")
        for c, cnt in sorted(by_coll.items()):
            print(f"    {c:<42} {cnt} 块")
        print()
        print("自动出题请使用：")
        print("    python eval/gen_testset.py --per-collection 5")
        print("    python eval/run_eval.py --golden eval/golden_auto.jsonl --no-save")
        return 0

    # 旧版字典格式（历史产物）：提示重新导出
    print("检测到旧版块索引格式（{\"collections\": {...}}）。")
    print("请重新运行 eval/export_chunks.py 生成新版块索引，再使用 gen_testset.py 自动出题。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
