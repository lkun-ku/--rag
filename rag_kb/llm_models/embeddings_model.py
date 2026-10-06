import os
from typing import Optional

from rag_kb.utils.env_utils import EMBEDDING_DIM, EMBEDDING_MODEL
from rag_kb.utils.log_utils import log

# ---------------------------------------------------------------------------
# 本地嵌入模型（唯一嵌入来源）
#
# 演进过程：
#   1. 原项目：分块用**远程** OpenAI embedding，入库用**本地** bge
#      => 分块语义与检索向量不同源；远程 key 失效会中断入库；文本发往第三方
#   2. 修坑①：分块与入库统一到本地 bge（small，512 维）
#   3. 升级：模型升到 bge-large-zh-v1.5（1024 维），并**移除 OpenAI embedding**
#   4. [P0] 设备可配置：默认优先 GPU（CUDA），不可用时自动回落 CPU 并告警
#   5. [P3] 后端可切换：`EMBEDDING_BACKEND=http` 时不加载本地模型，改为调用
#      独立进程（`rag_kb.api.embed_server`）—— 多 worker 部署时共用一份模型。
#
# 为什么彻底移除 OpenAI（决策见 ROADMAP 1.4）：
#   - 已无关键路径依赖（分块与入库都由本地模型承担）
#   - 若只给 md 保留 OpenAI 分块，会造成同一知识库内**两种切片标准**，
#     块粒度不一致，与「多格式统一管道」的架构直接冲突
#   - 消除隐私外发、远程成本与 key 失效风险
#
# 模型与维度全部配置化（rag_kb/utils/env_utils.py）：
#   默认 bge-large-zh-v1.5 / 1024 维，指向项目内 models/ 目录（离线可用）
#   做 512 维 A/B 对照时改 EMBEDDING_MODEL / EMBEDDING_DIM 即可，无需改代码
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_URL = 'http://127.0.0.1:8001'


def resolve_device(env_value: Optional[str] = None, cuda_available: Optional[bool] = None) -> str:
    """决定 embedding 跑在哪个设备：显式配置优先，其次是「CUDA 可用就用 CUDA」。

    为什么单独抽成纯函数：换到 GPU 后最大的运维风险是**静默回落** ——
    驱动更新 / 远程桌面 / 显存被占时 `torch.cuda.is_available()` 会变 False，
    若代码默默降级到 CPU，吞吐掉约 40x 却没有任何信号。把判定做成可注入的纯函数，
    既能被测试覆盖，也方便 /health 把真实设备暴露出来。

    Args:
        env_value: ``EMBEDDING_DEVICE`` 的原始值；None 表示读环境变量。
        cuda_available: CUDA 是否可用；None 表示读 torch 的真实状态（此时才 import torch）。

    Returns:
        ``'cuda'`` 或 ``'cpu'``。显式要求 cuda 但不可用时**回落 cpu**（不抛异常）。
    """
    raw = (os.getenv('EMBEDDING_DEVICE', '') if env_value is None else env_value).strip().lower()
    if cuda_available is None:
        import torch  # 局部导入：远程后端模式下完全没有必要加载 torch

        cuda_available = torch.cuda.is_available()

    if raw in ('cpu', 'cuda'):
        if raw == 'cuda' and not cuda_available:
            log.warning('EMBEDDING_DEVICE=cuda 但当前 CUDA 不可用，已回落 CPU（吞吐约下降 40x）')
            return 'cpu'
        return raw

    if raw:
        log.warning(f'未知的 EMBEDDING_DEVICE={raw!r}（只认 cpu / cuda），按自动规则处理')
    return 'cuda' if cuda_available else 'cpu'


def batch_size_for(device: str) -> int:
    """按设备返回 embedding 的批大小（均为实测最优，不要凭直觉调大）。

    实测（16 线程 CPU / 8 GB 显存 GPU）：
      * CPU：批 20 -> 1.76 块/秒；批 32 -> 1.11（慢 35%），批 64 同样更慢；
      * GPU：批 16 -> 70 块/秒；批 32 -> 45；批 64/128 -> 24。
    批越大越慢的主要原因是块长不均导致的 padding 浪费。
    """
    return 16 if device == 'cuda' else 20


def resolve_backend(env_value: Optional[str] = None) -> str:
    """返回 ``'local'`` 或 ``'http'``。只认这两个值，其它值按 local 处理并告警。"""
    raw = (os.getenv('EMBEDDING_BACKEND', '') if env_value is None else env_value).strip().lower()
    if raw in ('local', 'http'):
        return raw
    if raw:
        log.warning(f'未知的 EMBEDDING_BACKEND={raw!r}（只认 local / http），按 local 处理')
    return 'local'


EMBEDDING_BACKEND = resolve_backend()

if EMBEDDING_BACKEND == 'http':
    # 远程模式：**绝不加载本地模型**（否则多 worker 又会各占一份显存，等于没做）
    EMBEDDING_URL = (os.getenv('EMBEDDING_URL') or DEFAULT_EMBEDDING_URL).rstrip('/')
    EMBEDDING_DEVICE = 'remote'
    EMBEDDING_BATCH_SIZE = 16  # 客户端分批粒度；服务端仍会按自己的最优批再分批

    from rag_kb.llm_models.remote_embeddings import build_remote_embeddings

    bge_embedding = build_remote_embeddings(EMBEDDING_URL, batch_size=EMBEDDING_BATCH_SIZE)
    log.info(f'embedding 后端: http（{EMBEDDING_URL}），本进程不加载模型；model={EMBEDDING_MODEL} dim={EMBEDDING_DIM}')
    EMBEDDING_INFO = {
        'model': EMBEDDING_MODEL,
        'dim': EMBEDDING_DIM,
        'normalize': True,
        'backend': 'http',
        'device': 'remote',
        'url': EMBEDDING_URL,
        'batch_size': EMBEDDING_BATCH_SIZE,
        'cuda_available': None,  # 本进程不知道（也不加载 torch），以服务端 /health 为准
    }
else:
    from langchain_huggingface import HuggingFaceEmbeddings

    EMBEDDING_URL = None
    EMBEDDING_DEVICE = resolve_device()
    EMBEDDING_BATCH_SIZE = batch_size_for(EMBEDDING_DEVICE)

    # bge_embedding 保持原名：markdown_parser（分块）与 milvus_db（入库）都引用它，
    # 保证「分块语义」与「入库向量」始终同源。
    bge_embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE},
    )

    import torch  # 仅本地模式需要

    log.info(
        f'embedding 后端: local；已加载 model={EMBEDDING_MODEL} dim={EMBEDDING_DIM} '
        f'device={EMBEDDING_DEVICE} batch={EMBEDDING_BATCH_SIZE} (cuda_available={torch.cuda.is_available()})'
    )
    EMBEDDING_INFO = {
        'model': EMBEDDING_MODEL,
        'dim': EMBEDDING_DIM,
        'normalize': True,
        'backend': 'local',
        'device': EMBEDDING_DEVICE,
        'batch_size': EMBEDDING_BATCH_SIZE,
        'cuda_available': torch.cuda.is_available(),
    }
