from langchain_core.tools import create_retriever_tool
from pymilvus import Function
from pymilvus.client.types import FunctionType
from documents.milvus_db import MilvusVectorSave
from utils.env_utils import MILVUS_RERANKER_STYLE

mv = MilvusVectorSave()
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',  # 仅返回相似度超过阈值的文档
    search_kwargs={
        "k": 4,
        # 注意：langchain-milvus 与 langchain-core 均不消费此参数（一直是惰性的），保留原值
        "score_threshold": 0.1,
        # reranker 写法由 MILVUS_RERANKER_STYLE 决定（见 utils/env_utils.py 说明）：
        #   function：官方新写法 reranker=Function(RERANK)，需服务端 Milvus 2.6+
        #   legacy  ：ranker_type + ranker_params，已 deprecated 但兼容面最广
        # 两者语义一致（RRF，k=100）。
        **({
            # 注意：RRF 形式的 rerank Function 有两个易错点（已在真实
            # Milvus 2.6.25 上实测确认，否则服务端报 code=1100）：
            #   1) params 里必须带 "reranker" 键指明是哪一种 rerank
            #      （服务端支持 decay / model / rrf / weighted）
            #   2) RRF 不接受输入字段，input_field_names 必须为空列表
            "reranker": Function(
                name="rrf",
                function_type=FunctionType.RERANK,
                input_field_names=[],
                params={"reranker": "rrf", "k": 100},
            )
        } if MILVUS_RERANKER_STYLE == "function" else {
            "ranker_type": "rrf",
            "ranker_params": {"k": 100},
        }),
        # 修复：原写作 'filter': {"category": "content"}，但 langchain-milvus 的
        # hybrid 检索路径只用 expr=expr，其余参数进 **kwargs 透传给
        # MilvusClient.hybrid_search()，而该函数没有 filter 形参
        # （langchain_milvus/vectorstores/milvus.py:1789、1800；
        #   pymilvus/milvus_client/milvus_client.py:359-368），
        # 因此这个过滤条件自项目创建以来从未生效。Milvus 的正确写法是 expr。
        # 注意：修复后过滤真正生效，召回结果不再包含 category != 'content' 的块，
        #       属「恢复既有设计意图」，而非新增业务规则。
        "expr": 'category == "content"',
    }
)


retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    # [UPGRADE 2026-10] 工具描述原为「搜索并返回关于'半导体和芯片'的信息…封装、测试、光刻胶等」，
    # 但本项目 datas/md/ 下的实际语料是 Milvus 向量数据库的文档（overview / product_faq /
    # performance_faq / operational_faq / troubleshooting / tech_report）。
    # 描述与语料不一致会让 create_agent（rag_agent.py）依据描述判断"工具与问题无关"而
    # 直接拒答、不发起检索（实测会回答"我唯一能用的工具是半导体检索器"）。
    # 这里只改描述文案，不改工具名、不改检索行为。
    '搜索并返回关于 Milvus 向量数据库的信息，内容涵盖：架构与部署、索引类型与指标、'
    '性能调优、运维监控、常见问题排查等'
)