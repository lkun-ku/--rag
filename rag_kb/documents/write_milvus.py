"""多源、多格式入库管线：解析 ``DOC_SOURCES`` 里的文件，写进各自的 Milvus 集合。

数据流（全在一个进程内）：

    逐文件解析 ──批次──► MilvusVectorSave ──► 该源自己的集合
                        （统计对象在同进程内存里）

为什么解析与写入都留在主进程（2026-10-24 在 GPU 上复测，不要再拆）：
  1. ``bge_embedding`` 是 1.24GB 的模块级单例（现跑在 GPU），拆进程会各占一份显存；
  2. 切片与入库**用的是同一个模型**：解析侧的切片 embedding 也在 GPU 上，
     与入库 embedding 不是「CPU vs GPU」的正交关系，拆开只会抢同一块卡；
  3. GPU 之后写入侧已经很轻，能重叠的部分很小：
     md 源 216 块 = 解析 5.4s + embedding 1.7s + flush 7.7s；
     PDF 源 160 块 = 解析 84.3s + 写入 0.7s（重叠上限仅 1%）。
     （CPU 时代测过 413s vs 415.4s / 0.6%，结论一致，但当时的机制说法已过时。）
  4. 单向数据流，不存在「解析等 EOF、写入等批次」；Lite 的文件锁是进程级，单进程天然满足。

> 真要「正交」，正确做法不是拆解析/写入，而是把模型移出 API 进程：
> ``EMBEDDING_BACKEND=http`` + ``rag_kb.api.embed_server``（worker 不加载模型，见 ROADMAP 4.51.4）。

几条硬约束（都踩过坑，别改）：
  * 失败必须显式记录（ROADMAP 1.3.6），单文件/单批失败只跳过，不毁整批；
  * 连续写失败即熔断本源，避免 Milvus 不可达时继续烧 CPU；
  * 同一集合必须串行（并发 recreate 会静默丢数据）；
  * 整批只 flush 一次（单次 flush 实测 ~7s，按批 flush 会慢数倍）；
  * ``recreate=False`` 默认走增量：按文件 sha1 与清单比对，未变跳过、变更先删后插。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from rag_kb.documents.manifest import (
    _delete_filenames,
    _fmt_err,
    file_fingerprint,
    load_manifest,
    plan_incremental,
    save_manifest,
)
from rag_kb.documents.milvus_db import MilvusVectorSave
from rag_kb.documents.parser_factory import ParseFn, build_parser
from rag_kb.utils.env_utils import DOC_SOURCES, EMBEDDING_DIM
from rag_kb.utils.log_utils import log

if TYPE_CHECKING:  # 仅类型标注
    from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 20  # 每批交给写入侧的块数
DEFAULT_MAX_WRITE_FAILURES = 1  # 连续写失败多少次后熔断本源

OnError = Callable[[str], None]


# ---------------------------------------------------------------------------
# 集合级互斥
# ---------------------------------------------------------------------------
_COLLECTION_LOCKS: Dict[str, threading.Lock] = {}
_COLLECTION_LOCKS_GUARD = threading.Lock()


def _collection_lock(collection: str) -> threading.Lock:
    """集合级互斥锁：同集合的入库串行，不同集合仍可并行。"""
    with _COLLECTION_LOCKS_GUARD:
        lock = _COLLECTION_LOCKS.get(collection)
        if lock is None:
            lock = threading.Lock()
            _COLLECTION_LOCKS[collection] = lock
        return lock


# 增量判定、清单读写见 `rag_kb/documents/manifest.py`；
# 解析器分派见 `rag_kb/documents/parser_factory.py`（本模块只保留入库编排）。


# ---------------------------------------------------------------------------
# 源扫描
# ---------------------------------------------------------------------------
# 扫描实现放在轻量模块（不加载模型）：这里的 import 只为保持 `write_milvus.list_source_files`
# 这个历史入口可用。
from rag_kb.utils.file_scan import list_source_files  # noqa: E402  (保持向后兼容的重导出)


# ---------------------------------------------------------------------------
# 统计对象
# ---------------------------------------------------------------------------
@dataclass
class SourceStats:
    """单个源的结果。``blocks``=交给写入侧的块数，``written``=写成功的块数，差值即可见差异。"""

    source: str
    collection: str
    files: int = 0
    blocks: int = 0
    written: int = 0
    errors: List[str] = field(default_factory=list)
    skipped: Optional[str] = None
    aborted: bool = False
    reused: int = 0  # 增量：未变而跳过的文件数
    removed: int = 0  # 增量：从库里清理掉旧块的文件数

    def to_dict(self) -> Dict[str, Any]:
        """与历史 stats dict 字段一致，仅在有内容时追加 skipped/aborted/reused/removed。"""
        data: Dict[str, Any] = {
            'source': self.source,
            'collection': self.collection,
            'files': self.files,
            'blocks': self.blocks,
            'written': self.written,
            'errors': list(self.errors),
        }
        for key in ('skipped', 'aborted', 'reused', 'removed'):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data


# ---------------------------------------------------------------------------
# 解析与写入
# ---------------------------------------------------------------------------
def _parse_one(file_path: str, parse_fn: ParseFn, on_error: OnError) -> Optional[List['Document']]:
    """解析单个文件；失败只记录并返回 None（单文件失败不毁整批）。"""
    try:
        return parse_fn(file_path)
    except Exception as e:  # noqa: BLE001
        basename = os.path.basename(file_path)
        log.error(f'[解析] 失败 {basename}: {type(e).__name__}: {e}')
        on_error(f'{basename}: {_fmt_err(e)}')
        return None


def _iter_file_batches(
    files: List[str], parse_fn: ParseFn, batch_size: int, on_error: OnError
) -> Iterator[List['Document']]:
    """逐文件解析、攒够 batch_size 出一批（批次按文件边界触发，可能略超 batch_size）。"""
    batch: List['Document'] = []
    for file_path in files:
        docs = _parse_one(file_path, parse_fn, on_error)
        if not docs:
            continue
        batch.extend(docs)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_batches(
    batches: Iterable[List['Document']],
    mv: MilvusVectorSave,
    stats: 'SourceStats',
    max_consecutive_failures: int = DEFAULT_MAX_WRITE_FAILURES,
) -> None:
    """批量写入；单批失败继续，连续失败达阈值即熔断本源（aborted=True）。"""
    streak = 0
    for batch in batches:
        stats.blocks += len(batch)
        try:
            mv.add_documents(batch, flush=False)  # 整批结束再 flush 一次
            stats.written += len(batch)
            streak = 0
            log.info(f'[写入] {stats.collection} 累计已写入: {stats.written} 个块')
        except Exception as e:  # noqa: BLE001
            log.exception(f'[写入] 失败（{len(batch)} 个块）')
            stats.errors.append(_fmt_err(e))
            streak += 1
            if streak >= max_consecutive_failures:
                stats.aborted = True
                stats.errors.append(f'写入连续失败 {streak} 次，中止本源：本次 {len(batch)} 块及其后续块未入库')
                return


def _ingest_files_individually(
    mv: MilvusVectorSave,
    files: List[str],
    parse_fn: ParseFn,
    stats: 'SourceStats',
    on_success: Callable[[str], None],
    max_consecutive_failures: int = DEFAULT_MAX_WRITE_FAILURES,
) -> None:
    """逐文件解析+写入，并按文件回调成功——增量需要精确到文件地更新指纹。"""
    for file_path in files:
        docs = _parse_one(file_path, parse_fn, stats.errors.append)
        if docs is None:
            continue
        if not docs:
            on_success(file_path)  # 解析成功但无块：记为已处理，否则每次增量都重试
            continue
        written_before = stats.written
        _write_batches([docs], mv, stats, max_consecutive_failures=max_consecutive_failures)
        if stats.written > written_before:
            on_success(file_path)
        if stats.aborted:
            return


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def _plan_changes(
    name: str, files: List[str], collection: str, recreate: bool, incremental: bool, stats: 'SourceStats'
) -> Optional[Tuple[List[str], List[str], Dict[str, Any]]]:
    """算出本次要处理什么；返回 (待处理文件, 需删除的 basename, 旧清单)，无需处理时返回 None。"""
    manifest = load_manifest(collection)
    if recreate or not incremental:
        if not recreate:
            log.warning(f'源 {name} 为追加模式（incremental=False）：同一文件可能产生重复块')
        return list(files), [], manifest

    to_ingest, unchanged, removed = plan_incremental(files, manifest['files'])
    stats.reused = len(unchanged)
    if not to_ingest and not removed:
        log.info(f'源 {name} 无变更：{len(unchanged)} 个文件内容未变，跳过入库（未连接 Milvus）')
        return None
    log.info(f'源 {name} 增量：待重做 {len(to_ingest)} / 未变 {len(unchanged)} / 待删除 {len(removed)}')
    return to_ingest, removed, manifest


def _prepare_collection(mv: MilvusVectorSave, collection: str, recreate: bool, stats: 'SourceStats') -> bool:
    """重建集合，或确认增量目标存在。返回 False 表示该源应跳过。"""
    if recreate:
        mv.recreate()  # 维度自检在 connect() 里，重建不会与旧维度冲突
        return True
    try:
        exists = mv.exists()
    except Exception as e:  # noqa: BLE001
        log.warning(f'无法确认集合 {collection} 是否存在（不阻断）：{_fmt_err(e)}')
        return True
    if exists:
        return True
    # langchain-milvus 对不存在的集合会静默新建 —— 集合名拼错就会「成功但检索不到」
    stats.errors.append(f'集合 {collection} 不存在，且 recreate=False（增量模式）：已跳过，避免静默新建集合')
    log.error(f'源 {stats.source} 跳过：{stats.errors[-1]}')
    return False


def _persist_manifest(
    collection: str,
    recreate: bool,
    incremental: bool,
    manifest: Dict[str, Any],
    removed: List[str],
    succeeded: List[str],
) -> None:
    """写入成功后才更新清单；失败的源保留旧清单，下次自然重试。"""
    if recreate:
        save_manifest(collection, {os.path.basename(p): file_fingerprint(p) for p in succeeded})
    elif incremental:
        files_map = dict(manifest['files'])
        for name in removed:
            files_map.pop(name, None)
        for path in succeeded:
            files_map[os.path.basename(path)] = file_fingerprint(path)
        save_manifest(collection, files_map)


def ingest_source(
    source: Dict[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    dim: Optional[int] = None,
    single_process: Optional[bool] = None,
    max_write_failures: int = DEFAULT_MAX_WRITE_FAILURES,
    incremental: bool = True,
) -> SourceStats:
    """处理单个源：校验 parser -> 建/重建集合 -> 解析并写入。返回该源统计。

    Args:
        recreate: True 为全量重灌（**会清空集合**）；False 默认按文件指纹增量。
        incremental: ``recreate=False`` 时是否增量；False 则纯追加（可能重复）。
        single_process: 已废弃，接受但忽略（解析与写入本来就在同一进程）。
        max_write_failures: 连续写失败多少次后熔断本源。
    """
    name, collection = source['name'], source['collection']
    dim = int(dim or EMBEDDING_DIM)
    if single_process is not None:
        log.warning('ingest_source(single_process=...) 已废弃且被忽略：解析与写入本来就在同一进程内')

    log.info(f'===== 开始处理源 {name} -> 集合 {collection} =====')
    files = list_source_files(source)
    if not files:
        log.warning(f'源 {name} 目录无匹配文件，跳过（目录: {source["dir"]}）')
        return SourceStats(source=name, collection=collection, skipped='目录无匹配文件')

    _, parse_fn = build_parser(source['parser'])  # 先校验配置，避免删空集合后才发现写错

    stats = SourceStats(source=name, collection=collection, files=len(files))
    plan = _plan_changes(name, files, collection, recreate, incremental, stats)
    if plan is None:
        return stats
    to_ingest, removed, manifest = plan

    lock = _collection_lock(collection)
    if lock.locked():
        log.warning(f'源 {name} 等待同一集合 {collection} 上的另一次入库完成…')
    succeeded: List[str] = []
    with lock:
        try:
            with MilvusVectorSave(collection_name=collection, dim=dim) as mv:
                if not _prepare_collection(mv, collection, recreate, stats):
                    return stats
                mv.connect()

                if not recreate and incremental:
                    # 先删后插：否则中途失败会同时留下新旧两份块（检索重复）
                    _delete_filenames(mv, collection, removed + [os.path.basename(p) for p in to_ingest])
                    stats.removed = len(removed)

                if recreate or incremental:
                    _ingest_files_individually(mv, to_ingest, parse_fn, stats, succeeded.append, max_write_failures)
                else:
                    batches = _iter_file_batches(to_ingest, parse_fn, batch_size, stats.errors.append)
                    _write_batches(batches, mv, stats, max_consecutive_failures=max_write_failures)

                if stats.written:
                    mv.flush()
                    # [FIX 2026-11] **必须显式 load**：新建/重灌的集合若不加载到内存，
                    # 查询会超时或静默失败 —— 实测：全新集合入库 2654 块后未 load，
                    # 评测时该集合 30 题全部 0 召回、每题检索 17s；手动 load 后立即恢复。
                    # 对已加载的集合重复 load 是幂等操作，无副作用。
                    try:
                        mv.store.client.load_collection(collection)
                    except Exception as le:  # noqa: BLE001
                        log.warning(f"load_collection 失败（可能已加载）: {type(le).__name__}: {le}")

            if not stats.aborted:
                _persist_manifest(collection, recreate, incremental, manifest, removed, succeeded)
        except Exception as e:  # noqa: BLE001
            # 建集合/建连接/写入等 Milvus 侧故障：记录后正常返回，不影响后续源
            log.exception(f'源 {name} 因 Milvus 侧错误中止（集合 {collection}）')
            stats.errors.append(_fmt_err(e))

    if stats.blocks == 0 and not (stats.errors or stats.removed or stats.reused):
        log.warning(f'源 {name} 未产出任何块（文件 {stats.files} 个）：请检查语料是否为空/格式是否正确')
    log.info(
        f'===== 源 {name} 完成：文件 {stats.files} / 块 {stats.blocks} / '
        f'写入 {stats.written} / 失败 {len(stats.errors)}{"（已熔断）" if stats.aborted else ""} ====='
    )
    return stats


def ingest_many(
    sources: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    max_write_failures: int = DEFAULT_MAX_WRITE_FAILURES,
    incremental: bool = True,
) -> List[SourceStats]:
    """逐个源入库并返回统计。**纯函数：不打印、不 exit。**

    ``sources=None`` 表示 DOC_SOURCES 中全部 enabled 的源；空列表表示什么都不做。
    """
    targets = [s for s in (DOC_SOURCES if sources is None else sources) if s.get('enabled', True)]
    all_stats: List[SourceStats] = []
    owners: Dict[str, str] = {}  # 集合 -> 已占用它的源名

    for source in targets:
        collection = source.get('collection')
        if collection and collection in owners:
            # 两个源写同一集合：后者的 recreate 会把前者的数据整体删掉
            owner = owners[collection]
            log.error(f'源 {source["name"]} 与源 {owner} 指向同一集合 {collection}，已跳过（避免互相覆盖）')
            all_stats.append(
                SourceStats(
                    source=source['name'],
                    collection=collection,
                    errors=[f'集合配置冲突：与源 {owner} 指向同一集合 {collection}，本批次内不能重复写入'],
                )
            )
            continue
        if collection:
            owners[collection] = source['name']

        try:
            all_stats.append(
                ingest_source(
                    source,
                    batch_size=batch_size,
                    recreate=recreate,
                    max_write_failures=max_write_failures,
                    incremental=incremental,
                )
            )
        except NotImplementedError as e:
            log.warning(f'源 {source["name"]} 跳过: {e}')  # 早期未实现的 parser 视为跳过
        except Exception as e:  # noqa: BLE001
            log.exception(f'源 {source["name"]} 处理失败')
            all_stats.append(SourceStats(source=source['name'], collection=collection or '', errors=[_fmt_err(e)]))
    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(
    sources: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
) -> List[Dict[str, Any]]:
    """CLI 入口：入库并打印汇总表。服务端应调 ``ingest_many``（不打印）。"""
    stats = ingest_many(sources, batch_size=batch_size, recreate=recreate)
    as_dicts = [s.to_dict() for s in stats]
    _print_summary(as_dicts)
    return as_dicts


def _summary_flag(s: Dict[str, Any]) -> str:
    """汇总表里的状态标记（熔断 / 跳过 / 无变更 / 增量统计）。"""
    if s.get('aborted'):
        return ' [熔断]'
    if s.get('skipped'):
        return ' [跳过]'
    if s.get('reused') and not s.get('written'):
        return ' [无变更]'
    if s.get('reused') or s.get('removed'):
        return ' [增量:%d改/%d删/%d未变]' % (
            s.get('files', 0) - s.get('reused', 0),
            s.get('removed', 0),
            s.get('reused', 0),
        )
    return ''


def _print_summary(all_stats: List[Dict[str, Any]]) -> None:
    """CLI 汇总表：每源文件/块/写入数 + 失败明细。"""
    print()
    print('=' * 78)
    print('入库汇总')
    print('=' * 78)
    print('%-16s %-34s %8s %8s %8s' % ('源', '集合', '文件', '块', '写入'))
    print('-' * 78)
    for s in all_stats:
        print(
            '%-16s %-34s %8s %8s %8s%s'
            % (
                s['source'],
                s['collection'],
                s.get('files', 0),
                s.get('blocks', 0),
                s.get('written', 0),
                _summary_flag(s),
            )
        )
    print('-' * 78)
    print(f'总计失败项: {sum(len(s.get("errors", [])) for s in all_stats)}')
    for s in all_stats:
        for err in s.get('errors', []):
            print(f'  [{s["source"]}] {err}')
    print('=' * 78)


def _print_source_table(sources: List[Dict[str, Any]]) -> None:
    """CLI `--list`：打印源 / 集合 / parser / 文件数。"""
    print('%-14s %-34s %-9s %5s %s' % ('源', '集合', 'parser', '文件数', '备注'))
    print('-' * 78)
    for s in sources:
        print(
            '%-14s %-34s %-9s %5d %s'
            % (
                s['name'],
                s['collection'],
                s['parser'],
                len(list_source_files(s)),
                '' if s.get('enabled', True) else '(disabled)',
            )
        )


def _pick_sources(names: Optional[List[str]]) -> List[Dict[str, Any]]:
    """按源名过滤 DOC_SOURCES；None=全部，[]=一个都不选（两者语义必须分开）。"""
    if names is None:
        return DOC_SOURCES
    if not names:
        print('[警告] 未指定任何源名（--sources 为空），不做任何处理')
        return []
    wanted = set(names)
    selected = [s for s in DOC_SOURCES if s['name'] in wanted]
    missing = wanted - {s['name'] for s in selected}
    if missing:
        print(f'[警告] 未找到的源名: {sorted(missing)}')
    return selected


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='多源多格式入库管线')
    ap.add_argument('--sources', nargs='*', default=None, help='只处理指定源名（默认全部 enabled 的源）')
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='每批写入块数（默认 %(default)s）')
    ap.add_argument('--no-recreate', action='store_true', help='不重建集合，走增量（只处理变化的文件）')
    ap.add_argument('--list', action='store_true', help='只列出源与文件数，不入库')
    args = ap.parse_args()

    if args.list:
        _print_source_table(DOC_SOURCES)
    else:
        main(sources=_pick_sources(args.sources), batch_size=args.batch_size, recreate=not args.no_recreate)
