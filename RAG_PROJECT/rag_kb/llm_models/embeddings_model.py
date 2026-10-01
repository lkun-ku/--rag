from langchain_huggingface import HuggingFaceEmbeddings

from utils.env_utils import EMBEDDING_DIM, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# 本地嵌入模型（唯一嵌入来源）
#
# 演进过程：
#   1. 原项目：分块用**远程** OpenAI embedding，入库用**本地** bge
#      => 分块语义与检索向量不同源；远程 key 失效会中断入库；文本发往第三方
#   2. 修坑①：分块与入库统一到本地 bge（small，512 维）
#   3. 本次升级：模型升到 bge-large-zh-v1.5（1024 维），并**移除 OpenAI embedding**
#
# 为什么彻底移除 OpenAI（决策见 ROADMAP 1.4）：
#   - 已无关键路径依赖（分块与入库都由本地模型承担）
#   - 若只给 md 保留 OpenAI 分块，会造成同一知识库内**两种切片标准**，
#     块粒度不一致，与「多格式统一管道」的架构直接冲突
#   - 消除隐私外发、远程成本与 key 失效风险
#
# 模型与维度全部配置化（utils/env_utils.py）：
#   默认 bge-large-zh-v1.5 / 1024 维，指向项目内 models/ 目录（离线可用）
#   做 512 维 A/B 对照时改 EMBEDDING_MODEL / EMBEDDING_DIM 即可，无需改代码
# ---------------------------------------------------------------------------

# bge_embedding 保持原名：markdown_parser（分块）与 milvus_db（入库）都引用它，
# 保证「分块语义」与「入库向量」始终同源。
bge_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 供启动自检与日志使用（不参与向量计算）
EMBEDDING_INFO = {
    "model": EMBEDDING_MODEL,
    "dim": EMBEDDING_DIM,
    "normalize": True,
    "device": "cpu",
}
