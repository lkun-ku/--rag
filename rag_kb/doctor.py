"""环境自检：一条命令看清「能不能跑 / 跑在哪」。

覆盖：Python 与虚拟环境、关键依赖版本、torch/CUDA、embedding（设备/后端/真实试算一次）、
Milvus（连通性与各源集合行数）、增量清单、可选的 LLM 连通性。

设计约束：
  * 只读：不写入、不建集合、不改清单；
  * 分层失败：某一项失败不影响其它项的汇报，最终 ``ok`` 只由「硬失败」决定
    （embedding 不可用、Milvus 不可达）；LLM 只作为可选检查，默认不调用（有费用）。
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from typing import Any, Dict, List

from rag_kb.utils.env_utils import DOC_SOURCES, EMBEDDING_DIM, MILVUS_URI
from rag_kb.utils.log_utils import log

DEPENDENCIES = (
    'torch',
    'transformers',
    'sentence_transformers',
    'pymilvus',
    'langchain_milvus',
    'fastapi',
    'uvicorn',
    'httpx',
)


def collect(include_llm: bool = False) -> Dict[str, Any]:
    """执行全部检查，返回 ``{'ok': bool, 'checks': {...}, 'problems': [...]}``。"""
    checks: Dict[str, Any] = {}
    problems: List[str] = []

    # ---- Python 与虚拟环境 ----
    checks['python'] = {
        'version': platform.python_version(),
        'executable': sys.executable,
        'venv': sys.prefix != sys.base_prefix,
        'prefix': sys.prefix,
    }
    if not checks['python']['venv']:
        problems.append('当前不是虚拟环境（建议用项目内 .venv 运行）')

    # ---- 依赖版本 ----
    deps: Dict[str, Any] = {}
    for name in DEPENDENCIES:
        try:
            mod = importlib.import_module(name)
            deps[name] = getattr(mod, '__version__', 'unknown')
        except Exception as e:  # noqa: BLE001
            deps[name] = f'MISSING({type(e).__name__})'
    checks['deps'] = deps
    for name, ver in deps.items():
        if str(ver).startswith('MISSING'):
            problems.append(f'缺少依赖 {name}（pip install -r requirements.txt）')

    # ---- torch / CUDA ----
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        checks['torch'] = {
            'version': torch.__version__,
            'cuda_build': torch.version.cuda,
            'cuda_available': cuda,
            'device': torch.cuda.get_device_name(0) if cuda else None,
            'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if cuda else None,
        }
    except Exception as e:  # noqa: BLE001
        checks['torch'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:120]}'}
        problems.append('torch 不可用')

    # ---- embedding（会真的算一次，验证模型能加载）----
    try:
        from rag_kb.llm_models.embeddings_model import EMBEDDING_INFO, bge_embedding

        t0 = time.time()
        vec = bge_embedding.embed_query('自检')
        checks['embedding'] = {
            'ok': len(vec) == EMBEDDING_DIM,
            'model': EMBEDDING_INFO.get('model'),
            'dim': len(vec),
            'expected_dim': EMBEDDING_DIM,
            'backend': EMBEDDING_INFO.get('backend'),
            'device': EMBEDDING_INFO.get('device'),
            'batch_size': EMBEDDING_INFO.get('batch_size'),
            'query_ms': round((time.time() - t0) * 1000),
            **({'url': EMBEDDING_INFO['url']} if EMBEDDING_INFO.get('url') else {}),
        }
        if not checks['embedding']['ok']:
            problems.append(f'embedding 维度不符：返回 {len(vec)}，期望 {EMBEDDING_DIM}')
        if checks['embedding']['backend'] == 'http':
            checks['embedding']['server'] = _probe_embed_server(EMBEDDING_INFO.get('url'))
            if not checks['embedding']['server'].get('ok'):
                problems.append('embed 服务不可达（EMBEDDING_URL 指向的服务没起来）')
    except Exception as e:  # noqa: BLE001
        checks['embedding'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:160]}'}
        problems.append('embedding 不可用')

    # ---- Milvus ----
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=MILVUS_URI)
        try:
            existing = set(client.list_collections())
            wanted = [s['collection'] for s in DOC_SOURCES]
            rows = {}
            for coll in dict.fromkeys(wanted):
                if coll in existing:
                    # 注意：真实 Milvus 的 row_count 含已删除实体，这里只作规模参考
                    rows[coll] = client.get_collection_stats(collection_name=coll).get('row_count')
                else:
                    rows[coll] = None
            checks['milvus'] = {'ok': True, 'uri': MILVUS_URI, 'collections': sorted(existing), 'source_rows': rows}
            missing = [
                c
                for c, n in rows.items()
                if n is None and any(s['collection'] == c and s.get('enabled', True) for s in DOC_SOURCES)
            ]
            if missing:
                problems.append(f'以下集合尚未创建：{missing}（先跑 `main.py ingest`）')
        finally:
            client.close()
    except Exception as e:  # noqa: BLE001
        checks['milvus'] = {'ok': False, 'uri': MILVUS_URI, 'error': f'{type(e).__name__}: {str(e)[:160]}'}
        problems.append('Milvus 不可达')

    # ---- 增量清单 ----
    try:
        from rag_kb.documents.manifest import MANIFEST_DIR

        files = sorted(p.name for p in __import__('pathlib').Path(MANIFEST_DIR).glob('*.json'))
        checks['manifest'] = {'dir': MANIFEST_DIR, 'count': len(files), 'files': files[:10]}
    except Exception as e:  # noqa: BLE001
        checks['manifest'] = {'dir': None, 'error': f'{type(e).__name__}: {str(e)[:120]}'}

    # ---- LLM（可选，有费用）----
    if include_llm:
        try:
            from rag_kb.llm_models.all_llm import llm

            t0 = time.time()
            reply = llm.invoke('只回复一个字：好')
            checks['llm'] = {
                'ok': True,
                'model': getattr(llm, 'model_name', None) or getattr(llm, 'model', None),
                'latency_ms': round((time.time() - t0) * 1000),
                'reply': str(getattr(reply, 'content', reply))[:20],
            }
        except Exception as e:  # noqa: BLE001
            checks['llm'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:160]}'}
            # LLM 只影响问答，不影响入库，因此不算硬失败
            log.warning(f'LLM 不可用（不影响入库）：{type(e).__name__}: {e}')

    hard_fail = not checks.get('embedding', {}).get('ok') or not checks.get('milvus', {}).get('ok')
    return {'ok': not hard_fail and not problems, 'problems': problems, 'checks': checks}


def _probe_embed_server(url: Any) -> Dict[str, Any]:
    """探活独立 embedding 服务。"""
    if not url:
        return {'ok': False, 'error': 'EMBEDDING_BACKEND=http 但未设置 EMBEDDING_URL'}
    try:
        from rag_kb.utils.http_local import make_client

        with make_client(str(url), timeout=3) as client:
            resp = client.get(str(url).rstrip('/') + '/health')
            data = resp.json() if resp.status_code == 200 else {}
            return {'ok': resp.status_code == 200 and bool(data.get('ok')), **data}
    except Exception as e:  # noqa: BLE001
        return {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:120]}'}


def format_report(report: Dict[str, Any]) -> str:
    """把结果排成给人看的短表格（JSON 走 --json）。"""
    c = report['checks']
    lines = ['=' * 72, '环境自检', '=' * 72]

    py, torch, emb, mil = c.get('python', {}), c.get('torch', {}), c.get('embedding', {}), c.get('milvus', {})
    lines.append(f"Python      : {py.get('version')} | venv={py.get('venv')} | {py.get('executable')}")
    lines.append(
        f"torch       : {torch.get('version')} | cuda={torch.get('cuda_available')} "
        f"| {torch.get('device') or '-'} | {torch.get('vram_gb') or '-'}GB"
    )
    lines.append(
        f"embedding   : backend={emb.get('backend')} device={emb.get('device')} dim={emb.get('dim')} "
        f"batch={emb.get('batch_size')} query={emb.get('query_ms')}ms"
    )
    if emb.get('server'):
        lines.append(
            f"  └ server  : ok={emb['server'].get('ok')} pid={emb['server'].get('pid')} "
            f"device={emb['server'].get('device')}"
        )
    if not mil.get('ok'):
        lines.append(f"Milvus      : 不可达（{mil.get('error')}）")
    else:
        lines.append(f"Milvus      : {mil.get('uri')} | {len(mil.get('collections', []))} 个集合")
        for coll, n in (mil.get('source_rows') or {}).items():
            lines.append(f"  └ {coll:<34} {'未创建' if n is None else str(n) + ' 行'}")
    mf = c.get('manifest', {})
    lines.append(f"增量清单    : {mf.get('count', 0)} 个（{mf.get('dir')}）")
    if 'llm' in c:
        lines.append(f"LLM         : ok={c['llm'].get('ok')} latency={c['llm'].get('latency_ms')}ms")

    lines.append('-' * 72)
    if report['ok']:
        lines.append('结论: OK')
    else:
        lines.append('结论: 有问题')
        for p in report['problems']:
            lines.append(f'  - {p}')
    lines.append('=' * 72)
    return '\n'.join(lines)


def main(argv: List[str] | None = None) -> int:
    """独立运行：``python -m rag_kb.doctor [--json] [--with-llm]``。"""
    import argparse
    import json

    ap = argparse.ArgumentParser(description='环境自检')
    ap.add_argument('--json', action='store_true', help='输出 JSON')
    ap.add_argument('--with-llm', action='store_true', help='额外测一次 LLM（有费用）')
    args = ap.parse_args(argv)

    if os.getenv('HF_HUB_OFFLINE') is None:
        os.environ['HF_HUB_OFFLINE'] = '1'  # 模型在项目内，禁止联网重试

    report = collect(include_llm=args.with_llm)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
