# [FIX] 允许「直接 python xxx.py」运行：包内绝对导入需要把 RAG_PROJECT/ 加入 sys.path。
# 用 `python -m` 运行时该路径已在 path 中，此段自动跳过（幂等）。
import pathlib as _pathlib
import sys as _sys

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME, MD_DIR, EMBEDDING_DIM, EMBEDDING_MODEL
from utils.log_utils import log


# Milvus 集合名规则：只允许字母/数字/下划线，首字符必须是字母或下划线，长度 <= 255。
#
# 为什么要在客户端提前校验：实测发现 **Milvus Lite 与真实 Milvus 的校验宽严不一致** ——
#   `kb_milvus_bge-large_1024`（含连字符）在 Lite 上能正常建表，
#   在真实 Milvus 3.0.1 上却被拒绝：
#       MilvusException (code=1100): Invalid collection name: kb_milvus_bge-large_1024
# 若不拦住，就会「Lite 上看着没问题、换到真实服务才炸」。
_COLLECTION_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,254}$')


def _validate_collection_name(name: str) -> None:
    if not _COLLECTION_NAME_RE.match(name or ''):
        raise ValueError(
            f"非法集合名: {name!r}。Milvus 只允许字母/数字/下划线，且必须以字母或下划线开头"
            f"（连字符 `-` 会被真实 Milvus 拒绝，虽然 Milvus Lite 能接受）。"
            f"建议用下划线形式，例如 kb_milvus_bge_large_1024。"
        )


class MilvusVectorSave:
    """把新的 document 数据插入到 Milvus 中。

    支持「一源一集合」：集合名与向量维度都可注入，
    便于按 DOC_SOURCES 分集合写入（避免不同语义域内容互相干扰）。
    不传参时回落到配置默认值，因此原有 `MilvusVectorSave()` 调用方式不变。
    """

    def __init__(self, collection_name: str = None, dim: int = None) -> None:
        """自定义 collection 的索引。

        Args:
            collection_name: 目标集合名。默认取配置 COLLECTION_NAME。
                规范：kb_{domain}_bge-large_{dim}（见 ROADMAP 1.2）。
            dim: 稠密向量维度。**必须与实际嵌入模型输出一致**，
                否则写入会失败。默认取配置 EMBEDDING_DIM。
        """
        self.collection_name = collection_name or COLLECTION_NAME
        _validate_collection_name(self.collection_name)
        self.dim = int(dim or EMBEDDING_DIM)
        # 维度与模型一致性自检：Milvus 的 dense 字段维度固定且不可变，
        # 写错维度只会在写入时报错，且是「集合已建好才发现」——代价很高，故前置检查。
        if self.dim <= 0:
            raise ValueError(f"非法向量维度: {self.dim}")
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            field_name='text',
            datatype=DataType.VARCHAR,
            max_length=6000,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        )
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 维度来自配置：升维（512 -> 1024）只需改 EMBEDDING_DIM，无需改代码。
        # 注意：Milvus 的向量维度不可变，改维度 = 必须重建集合。
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64},  # M :邻接节点数, efConstruction: 搜索范围
        )

        if self.collection_name in client.list_collections():
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=self.collection_name)
            client.drop_index(collection_name=self.collection_name, index_name='sparse_inverted_index')
            client.drop_index(collection_name=self.collection_name, index_name='dense_inverted_index')
            client.drop_collection(collection_name=self.collection_name)

        client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)

    def check_dim_consistency(self):
        """若目标集合已存在，校验其 dense 维度与当前嵌入模型是否一致。

        为什么必须前置检查：Milvus 的向量维度**不可变**。用 1024 维模型去连
        512 维的旧集合，错误只会在真正写入时才暴露，且报出的往往是很底层的
        Milvus 异常（难以定位到「维度配错了」这个根因）。
        这里提前把话说清楚，并提示正确的处置方式。

        网络不可达时只告警不阻断（可能只是 Milvus 没启动，交给后续流程报错）。
        """
        try:
            client = MilvusClient(uri=MILVUS_URI)
            if self.collection_name not in client.list_collections():
                return  # 集合不存在，无冲突（后续 create_collection 会新建）
            desc = client.describe_collection(collection_name=self.collection_name)
            dims = [f.get('params', {}).get('dim') for f in desc.get('fields', []) if f.get('name') == 'dense']
            existing = dims[0] if dims else None
            if existing and int(existing) != self.dim:
                raise ValueError(
                    f"维度不匹配：集合 {self.collection_name} 的 dense 字段是 "
                    f"{int(existing)} 维，而当前嵌入模型配置为 {self.dim} 维。\n"
                    f"  当前模型: {EMBEDDING_MODEL}\n"
                    f"  处置方式（三选一）：\n"
                    f"    1) 换回匹配的模型：设置 EMBEDDING_DIM={int(existing)} 并选对应模型\n"
                    f"    2) 让管线重建集合：write_milvus.ingest_source(recreate=True)（会清空该集合）\n"
                    f"    3) 换一个集合名：设 KB_COLLECTION_* 指向新集合（推荐，可保留旧集合做 A/B 对照）"
                )
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning(f"维度自检未能完成（不阻断，可能是 Milvus 未启动）：{type(e).__name__}: {e}")

    def create_connection(self):
        """创建一个 Connection：milvus + langchain。pip install langchain-milvus"""
        self.check_dim_consistency()
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=self.collection_name,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI},
        )

    def add_documents(self, datas: List[Document]):
        """把新的 document 保存到 Milvus 中

        [UPGRADE ] 写入后显式 flush。
        实测（真实 Milvus 2.6.25）：langchain-milvus 的 add_documents 只把数据写进
        WAL，不会立刻 flush，此时 get_collection_stats 返回 0、检索报
        "nq [0] is invalid"（因为查到的是空集合）。显式 flush 后 row_count
        立刻变为实际条数、检索正常。仅影响数据可见时机，不改变写入的数据本身。
        """
        if not datas:
            # 不静默：空输入往往意味着上游解析出了问题，必须显式暴露（见 ROADMAP 1.3.6）
            raise ValueError("add_documents 收到空列表：上游解析可能失败，请检查解析器日志")

        self.vector_store_saved.add_documents(datas)
        try:
            self.vector_store_saved.client.flush(self.collection_name)
            log.info(f"flush 完成，当前集合 {self.collection_name} 数据已可检索")
        except Exception as e:  # noqa: BLE001
            # flush 失败不应中断入库（数据仍在 WAL 中，会被自动 flush）
            log.warning(f"flush 未成功（不影响数据本身）：{type(e).__name__}: {e}")


if __name__ == '__main__':
    # 解析文件内容
    # 路径改为配置化：默认取包内 datas/md，可用 DATAS_DIR 覆盖
    file_path = os.path.join(MD_DIR, 'tech_report_0tfhhamx.md')
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入 Milvus 数据库
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(collection_name=mv.collection_name)
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(collection_name=mv.collection_name)
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(collection_name=mv.collection_name, index_name=i)
            print(desc_index)

    result = client.query(
        collection_name=mv.collection_name,
        filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
        output_fields=['text', 'category', 'filename'],  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)
