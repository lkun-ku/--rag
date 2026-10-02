"""源目录扫描（**轻量模块，不 import 解析器/嵌入模型**）。

为什么单独拆出来：`documents/write_milvus.py` 在 import 期就会加载 1.24GB 的 bge 模型
（解析器依赖 embeddings_model）。而「列出有哪些源、各多少文件」这种只读操作不该付这个代价——
实测 `main.py ingest --list` 因此要多花 ~5s 并占 1.24GB 显存。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


def list_source_files(source: Dict[str, Any]) -> List[str]:
    """列出源目录下匹配 formats 的文件；扩展名不区分大小写。

    [2026-11] 改为**递归扫描**（`os.walk`）：自动源模式下，`datas/` 的每个直接
    子目录就是一个「资料包」，用户在资料包里建子目录组织内容是正常操作，
    不应该被静默忽略（原实现只扫一层，曾导致 17 个 md 全部漏收）。

    同时过滤：
    * 隐藏文件（`.` 开头）；
    * Office 打开时的锁文件（`~$` 开头）；
    * 评测/配置类的私有文件（`_` 开头，如 `_topic.txt`）。
    """
    dir_path = source['dir']
    formats = tuple(f.lower() for f in source['formats'])
    if not os.path.isdir(dir_path):
        return []

    out: List[str] = []
    for root, dirs, files in os.walk(dir_path):
        # 子目录里以 . / _ 开头的不递归（放私有素材用）
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        for f in files:
            if f.startswith(('~$', '.', '_')):
                continue
            if f.lower().endswith(formats):
                out.append(os.path.join(root, f))
    return sorted(out)


def format_source_table(sources: List[Dict[str, Any]]) -> str:
    """CLI `--list` 用的表格文本（与原先 write_milvus 的输出保持一致）。"""
    lines = ['%-14s %-34s %-9s %5s %s' % ('源', '集合', 'parser', '文件数', '备注'), '-' * 78]
    for s in sources:
        lines.append(
            '%-14s %-34s %-9s %5d %s'
            % (
                s['name'],
                s['collection'],
                s['parser'],
                len(list_source_files(s)),
                '' if s.get('enabled', True) else '(disabled)',
            )
        )
    return '\n'.join(lines)
