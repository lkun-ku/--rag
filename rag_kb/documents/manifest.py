"""增量清单：文件指纹与 per-collection manifest 的读写。

从 `write_milvus.py` 拆出（原文件 650 行同时承担解析分派、增量判定、
批次写入与 CLI 汇总）。本模块只依赖 hashlib/json/os 与日志，可单独测试，
见 `tests/service/test_incremental_plan.py`。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

from rag_kb.documents.milvus_db import MilvusVectorSave
from rag_kb.utils.env_utils import PKG_ROOT
from rag_kb.utils.log_utils import log

MANIFEST_DIR = os.path.join(PKG_ROOT, 'logs', 'ingest_manifest')
MANIFEST_VERSION = 1


def _fmt_err(e: BaseException) -> str:
    """异常压成一行短文本（汇总表要打印全部失败项，完整栈留在日志里）。"""
    return f'{type(e).__name__}: {str(e)[:120]}'


def file_fingerprint(path: str) -> str:
    """文件内容 sha1（分块读）。不用 mtime：解压/拷贝会改时间但不改内容。"""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def plan_incremental(files: List[str], known: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """算出 ``(to_ingest, unchanged, removed)``：待重做的路径 / 未变的路径 / 已消失的 basename。"""
    to_ingest: List[str] = []
    unchanged: List[str] = []
    present = set()
    for path in files:
        name = os.path.basename(path)
        present.add(name)
        (unchanged if known.get(name) == file_fingerprint(path) else to_ingest).append(path)
    removed = sorted(name for name in known if name not in present)
    return to_ingest, unchanged, removed


def _manifest_path(collection: str) -> str:
    return os.path.join(MANIFEST_DIR, f'{collection}.json')


def load_manifest(collection: str) -> Dict[str, Any]:
    """读清单；缺失/损坏/版本不符都返回空清单（按首次入库处理，不抛异常）。"""
    path = _manifest_path(collection)
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if data.get('version') == MANIFEST_VERSION and isinstance(data.get('files'), dict):
            return data
        log.warning(f'清单 {path} 版本不符，按首次入库处理（本次重做全部文件）')
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning(f'清单 {path} 读取失败，按首次入库处理：{_fmt_err(e)}')
    return {'version': MANIFEST_VERSION, 'collection': collection, 'files': {}}


def save_manifest(collection: str, files_map: Dict[str, str]) -> None:
    """写清单：先写临时文件再 replace，避免崩溃留下半个 JSON。"""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = _manifest_path(collection)
    tmp = path + '.tmp'
    data = {'version': MANIFEST_VERSION, 'collection': collection, 'files': files_map}
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _delete_filenames(mv: MilvusVectorSave, collection: str, names: List[str]) -> None:
    """按 filename 删旧块。真实 Milvus 的删除异步可见，必须删完立即 flush。"""
    if not names:
        return
    quoted = ','.join("'%s'" % n.replace("'", "\\'") for n in sorted(set(names)))
    mv.store.client.delete(collection_name=collection, filter=f'filename in [{quoted}]')
    mv.flush()
