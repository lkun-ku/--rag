"""多源、多格式入库管线。

架构（沿用原设计的「解析与写入分离 + 有界队列背压」）：

    ┌──────────────── 源清单 DOC_SOURCES（utils/env_utils.py）────────────────┐
    │  每个源独立处理：                                                        │
    │                                                                         │
    │   解析进程 ──docs_queue(maxsize=20)──► 写入进程 ──► 该源自己的集合       │
    │   （按 parser 类型分派）                （MilvusVectorSave）             │
    │        │                                        │                       │
    │        └────────── stats_queue ─────────────────┘                       │
    │                          │                                              │
    └──────────────────────────┴──► 主进程汇总统计（文件数 / 块数 / 失败清单）│

设计要点：
  1. **一源一集合**：不同语义域写入不同集合，避免内容相互干扰（ROADMAP 1.2）。
  2. **按 parser 分派**：markdown / pdf / office 各用各的解析器，
     但都产出同构的 Document 块，因此写入与检索逻辑完全共用。
  3. **有界队列背压**：maxsize 让解析在写入变慢时自动阻塞，内存不会被打爆。
  4. **失败显式记录**：单个文件失败只记录错误并继续，**但结束时汇总失败清单** ——
     既不因一个坏文件毁掉整批，也不会让失败静默消失（ROADMAP 1.3.6）。
"""

import multiprocessing
import os
from multiprocessing import Queue

from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from documents.office_parser import OfficeParser
from documents.pdf_parser import PDFParser
from utils.env_utils import DOC_SOURCES, EMBEDDING_DIM, MILVUS_LITE_PATH
from utils.log_utils import log

# 队列消息类型标记
_MSG_EOF = '__eof__'

# Milvus Lite 用**文件锁**保护数据目录，第二个进程打开同一个 .db 会抛
#   milvus_lite.exceptions.DataDirLockedError: another process holds the lock
# 因此 Lite 下必须单进程执行（解析与写入顺序进行）。
# 真实 Milvus 是网络服务，没有这个限制，仍走多进程 + 有界队列背压。
_SINGLE_PROCESS = bool(MILVUS_LITE_PATH)


def build_parser(parser_type: str):
    """按 parser 类型返回 (parser 实例, 单文件解析函数)。

    所有返回的解析函数签名统一为 `fn(file_path) -> List[Document]`，
    且都保证产出与原生 Markdown 路径同构的块。

    [FIX / P1-4] PDF 的 OCR 开关此前**在入库链路上不可达** ——
        `PDFParser()` 无参构造 -> `ocr_fallback=False`，而 `PDFParser.to_markdown`
        的 `use_ocr` 也没有任何入口能传进来。结果是 ROADMAP 2.3 实现的
        「按文本层判定扫描件 -> 走带页码的整页 OCR」这条路径**从未在正式入库时生效过**
        （扫描件实际靠 `pymupdf4llm` 自带的 OCR 兜过去，但产出**不带页码**，
        不满足引用粒度对页码的要求）。

        现在通过环境变量开放两个开关（**默认值保持与修复前完全一致**，不改变既有行为）：
            PDF_OCR_FALLBACK=1   扫描件走本项目 RapidOCR（带 `## 第 N 页`），慢（约 2.5s/页）
            PDF_USE_OCR=0        关闭 pymupdf4llm 的图片 OCR（与阶段一基线对齐时用）
    """
    if parser_type == 'markdown':
        p = MarkdownParser()
        return p, p.parse_markdown_to_documents
    if parser_type == 'pdf':
        ocr_fallback = os.getenv('PDF_OCR_FALLBACK', '0') not in ('', '0', 'false', 'False')
        use_ocr = os.getenv('PDF_USE_OCR', '1') not in ('', '0', 'false', 'False')
        p = PDFParser(ocr_fallback=ocr_fallback, use_ocr=use_ocr)
        return p, p.parse_pdf_to_documents
    if parser_type == 'office':
        p = OfficeParser()
        return p, p.parse_office_to_documents
    if parser_type == 'text':
        # 按决策：txt 需先人工整理为 Markdown（ROADMAP 1.3.3），此处不支持自动摄入
        raise NotImplementedError(
            'txt 源需先人工整理为 Markdown 再入库（见 ROADMAP 1.3.3）；'
            '若确需自动处理，请单独立项做专用解析器。'
        )
    raise ValueError(f'未知的 parser 类型: {parser_type!r}')


def list_source_files(source: dict):
    """列出源目录下符合格式的文件（返回绝对路径列表）。"""
    dir_path = source['dir']
    formats = tuple(f.lower() for f in source['formats'])
    if not os.path.isdir(dir_path):
        return []
    return [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if f.lower().endswith(formats) and os.path.isfile(os.path.join(dir_path, f))
    ]


def source_parser_process(source: dict, output_queue: Queue, stats_queue: Queue, batch_size: int = 20):
    """解析进程：扫描单个源目录，按格式分派解析器，分批送进队列。

    结束时向 output_queue 发送 (_MSG_EOF, None) 终止信号，
    并向 stats_queue 汇报统计（文件数 / 块数 / 失败清单）。
    """
    name = source['name']
    log.info(f'[解析进程] 源 {name} 开始扫描: {source["dir"]}')

    files = list_source_files(source)
    stats = {'source': name, 'files': len(files), 'blocks': 0, 'errors': []}

    if not files:
        log.warning(f'[解析进程] 源 {name} 未找到任何匹配文件（格式 {source["formats"]}）')
        output_queue.put((_MSG_EOF, None))
        stats_queue.put(stats)
        return

    parser, parse_fn = build_parser(source['parser'])
    doc_batch = []
    for file_path in files:
        try:
            docs = parse_fn(file_path)
            if docs:
                doc_batch.extend(docs)
                stats['blocks'] += len(docs)

            if len(doc_batch) >= batch_size:
                output_queue.put((docs and doc_batch.copy()) or [])
                doc_batch.clear()
        except Exception as e:  # noqa: BLE001
            # 单个文件失败不中断整批，但必须记录（不静默）
            basename = os.path.basename(file_path)
            log.error(f'[解析进程] 解析失败 {basename}: {type(e).__name__}: {e}')
            stats['errors'].append(f'{basename}: {type(e).__name__}: {str(e)[:120]}')

    if doc_batch:
        output_queue.put(doc_batch)

    output_queue.put((_MSG_EOF, None))
    stats_queue.put(stats)
    log.info(f'[解析进程] 源 {name} 完成：{stats["files"]} 个文件 / {stats["blocks"]} 个块 / '
             f'{len(stats["errors"])} 个失败')


def milvus_writer_process(input_queue: Queue, stats_queue: Queue, collection_name: str, dim: int):
    """写入进程：从队列读取并写入 **指定集合**。"""
    log.info(f'[写入进程] 启动，目标集合 {collection_name}（dim={dim}）')

    mv = MilvusVectorSave(collection_name=collection_name, dim=dim)
    mv.create_connection()
    total_count = 0
    write_errors = []

    while True:
        try:
            msg = input_queue.get()  # 阻塞
            if isinstance(msg, tuple) and msg and msg[0] == _MSG_EOF:
                break

            datas = msg
            if isinstance(datas, list) and datas:
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f'[写入进程] {collection_name} 累计已写入: {total_count} 个块')
        except Exception as e:  # noqa: BLE001
            log.error(f'[写入进程] 写入失败: {type(e).__name__}: {e}')
            log.exception(e)
            write_errors.append(f'{type(e).__name__}: {str(e)[:120]}')

    stats_queue.put({'source': collection_name, 'written': total_count,
                     'write_errors': write_errors})
    log.info(f'[写入进程] 结束，集合 {collection_name} 总计写入 {total_count} 个块')


def _run_single_process(source: dict, mv, files: list, collection: str) -> dict:
    """单进程模式：解析与写入在同一进程内顺序执行。

    为什么需要这个模式：Milvus Lite 用**文件锁**保护数据目录，第二个进程
    打开同一个 .db 会抛 `DataDirLockedError: another process holds the lock`。
    因此 Lite 下无法使用「解析进程 + 写入进程」的两进程结构。
    真实 Milvus 是网络服务，没有该限制，仍走多进程以获得有界队列背压。
    """
    name = source['name']
    _, parse_fn = build_parser(source['parser'])
    total_blocks = written = 0
    errors = []
    for fp in files:
        basename = os.path.basename(fp)
        try:
            docs = parse_fn(fp)
        except Exception as e:  # noqa: BLE001
            log.error(f'[单进程] 解析失败 {basename}: {type(e).__name__}: {e}')
            errors.append(f'{basename}: {type(e).__name__}: {str(e)[:120]}')
            continue
        if not docs:
            continue
        total_blocks += len(docs)
        try:
            mv.add_documents(docs)
            written += len(docs)
        except Exception as e:  # noqa: BLE001
            log.error(f'[单进程] 写入失败 {basename}: {type(e).__name__}: {e}')
            errors.append(f'{basename}(写入): {type(e).__name__}: {str(e)[:120]}')
    log.info(f'[单进程] 源 {name} 完成：{len(files)} 文件 / {total_blocks} 块 / 写入 {written}')
    return {'source': name, 'collection': collection, 'files': len(files),
            'blocks': total_blocks, 'written': written, 'errors': errors}


def ingest_source(source: dict, batch_size: int = 20, recreate: bool = True,
                  dim: int = None, single_process: bool = None) -> dict:
    """处理单个源：建集合 -> 多进程解析 -> 写入。返回统计 dict。

    Args:
        source: DOC_SOURCES 中的一项。
        batch_size: 每批写入的块数。
        recreate: 是否先删除并重建集合（默认 True，语义为「全量重灌」）。
            **注意**：recreate=True 会清空该集合现有数据。
        dim: 稠密向量维度，默认取配置 EMBEDDING_DIM。
    """
    name = source['name']
    collection = source['collection']
    dim = int(dim or EMBEDDING_DIM)
    queue_maxsize = 20  # 队列最大容量（背压，防止内存溢出）
    # 单进程判定：显式参数优先，否则按「是否使用 Milvus Lite」自动决定
    use_single = _SINGLE_PROCESS if single_process is None else bool(single_process)

    log.info(f'===== 开始处理源 {name} -> 集合 {collection} =====')

    files = list_source_files(source)
    if not files:
        log.warning(f'源 {name} 目录无匹配文件，跳过（目录: {source["dir"]}）')
        return {'source': name, 'collection': collection, 'files': 0,
                'blocks': 0, 'written': 0, 'errors': [], 'skipped': '目录无匹配文件'}

    mv = MilvusVectorSave(collection_name=collection, dim=dim)
    if recreate:
        mv.create_collection()
    else:
        log.info(f'源 {name} recreate=False，沿用现有集合 {collection}')

    if use_single:
        log.info(f'源 {name} 使用单进程模式'
                 f'（Milvus Lite 用文件锁保护数据目录，不允许多进程同时打开）')
        # 单进程下连接由本进程建立（多进程时由写入子进程各自建立，连接不可跨进程传递）
        mv.create_connection()
        return _run_single_process(source, mv, files, collection)

    docs_queue = Queue(maxsize=queue_maxsize)
    stats_queue = Queue()

    parser_proc = multiprocessing.Process(
        target=source_parser_process,
        args=(source, docs_queue, stats_queue, batch_size),
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue, stats_queue, collection, dim),
    )

    parser_proc.start()
    writer_proc.start()
    parser_proc.join()
    writer_proc.join()

    # 汇总两个子进程的统计
    results = []
    while not stats_queue.empty():
        results.append(stats_queue.get())

    parse_stats = next((r for r in results if r.get('source') == name), {})
    write_stats = next((r for r in results if r.get('source') == collection), {})

    summary = {
        'source': name,
        'collection': collection,
        'files': parse_stats.get('files', len(files)),
        'blocks': parse_stats.get('blocks', 0),
        'written': write_stats.get('written', 0),
        'errors': parse_stats.get('errors', []) + write_stats.get('write_errors', []),
    }
    log.info(f'===== 源 {name} 完成：文件 {summary["files"]} / 块 {summary["blocks"]} / '
             f'写入 {summary["written"]} / 失败 {len(summary["errors"])} =====')
    return summary


def main(sources=None, batch_size: int = 20, recreate: bool = True):
    """遍历 DOC_SOURCES，逐个源摄入。返回全部统计。"""
    targets = [s for s in (sources or DOC_SOURCES) if s.get('enabled', True)]
    all_stats = []
    for source in targets:
        try:
            all_stats.append(ingest_source(source, batch_size=batch_size, recreate=recreate))
        except NotImplementedError as e:
            log.warning(f'源 {source["name"]} 跳过: {e}')
        except Exception as e:  # noqa: BLE001
            log.error(f'源 {source["name"]} 处理失败: {type(e).__name__}: {e}')
            log.exception(e)
            all_stats.append({'source': source['name'], 'collection': source.get('collection'),
                              'files': 0, 'blocks': 0, 'written': 0,
                              'errors': [f'{type(e).__name__}: {str(e)[:120]}']})

    print()
    print('=' * 78)
    print('入库汇总')
    print('=' * 78)
    print('%-16s %-34s %8s %8s %8s' % ('源', '集合', '文件', '块', '写入'))
    print('-' * 78)
    for s in all_stats:
        print('%-16s %-34s %8s %8s %8s' % (
            s['source'], s['collection'], s.get('files', 0), s.get('blocks', 0), s.get('written', 0)))
    total_errors = sum(len(s.get('errors', [])) for s in all_stats)
    print('-' * 78)
    print(f'总计失败项: {total_errors}')
    for s in all_stats:
        for err in s.get('errors', []):
            print(f'  [{s["source"]}] {err}')
    print('=' * 78)
    return all_stats


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='多源多格式入库管线')
    ap.add_argument('--sources', nargs='*', default=None,
                    help='只处理指定的源名（默认处理 DOC_SOURCES 中所有 enabled 的源）')
    ap.add_argument('--batch-size', type=int, default=20, help='每批写入的块数')
    ap.add_argument('--no-recreate', action='store_true',
                    help='不重建集合（默认会先删除重建，即全量重灌）')
    ap.add_argument('--list', action='store_true', help='只列出源与文件数，不入库')
    args = ap.parse_args()

    if args.list:
        print('%-14s %-34s %-9s %s' % ('源', '集合', 'parser', '文件数'))
        print('-' * 78)
        for s in DOC_SOURCES:
            n = len(list_source_files(s))
            print('%-14s %-34s %-9s %-5d %s' % (
                s['name'], s['collection'], s['parser'], n,
                '' if s.get('enabled', True) else '(disabled)'))
    else:
        selected = DOC_SOURCES
        if args.sources:
            wanted = set(args.sources)
            selected = [s for s in DOC_SOURCES if s['name'] in wanted]
            missing = wanted - {s['name'] for s in selected}
            if missing:
                print(f'[警告] 未找到的源名: {sorted(missing)}')
        main(sources=selected, batch_size=args.batch_size, recreate=not args.no_recreate)
