import os
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from rag_kb.documents.markdown_parser import MarkdownParser
from rag_kb.llm_models.embeddings_model import get_embedding
from rag_kb.utils.env_utils import MILVUS_URI, COLLECTION_NAME, MD_DIR, EMBEDDING_DIM, EMBEDDING_MODEL
from rag_kb.utils.log_utils import log


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

    def __init__(self, collection_name: Optional[str] = None, dim: Optional[int] = None) -> None:
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
        # None = 尚未建立连接（`connect()` / `create_connection()` 之后才有值）
        self.vector_store_saved: Optional[Milvus] = None

    @property
    def store(self) -> Milvus:
        """已建立的 Milvus 连接。

        未连接时显式抛错，而不是让调用方在 None 上取属性得到
        难以定位的 AttributeError。
        """
        if self.vector_store_saved is None:
            raise RuntimeError("Milvus 连接尚未建立，请先调用 connect() / create_connection()")
        return self.vector_store_saved

    # ---- 生命周期 ----------------------------------------------------------
    # 为什么提供 with：入库管线会在一个进程里「建集合 -> 写入 -> 收尾」多次执行，
    # 每源若不关闭 client，长跑进程（FastAPI）会累积连接/句柄。用 with 让
    # 「打开就一定会关」成为类型强制，而不是靠调用方记得。
    def __enter__(self) -> 'MilvusVectorSave':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """关闭底层 Milvus 连接（幂等；未连接时是空操作）。"""
        store = self.vector_store_saved
        if store is None:
            return
        client = getattr(store, 'client', None)
        try:
            if client is not None:
                client.close()
        except Exception as e:  # noqa: BLE001
            # 关闭失败不影响已写入的数据，只告警（下次 GC 也会回收）
            log.warning(f"关闭 Milvus 连接未成功（不影响已写入数据）：{type(e).__name__}: {e}")
        finally:
            self.vector_store_saved = None

    def recreate(self):
        """删除并重建目标集合（**会清空该集合现有数据**）。"""
        client = MilvusClient(uri=MILVUS_URI)
        try:
            self._recreate_with(client)
        finally:
            # [FIX] 原实现每个源新建一个 MilvusClient 且从不 close：
            # 在反复入库的长跑进程里会持续泄漏连接/句柄。
            client.close()

    def _recreate_with(self, client: MilvusClient) -> None:
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
            self._drop_existing(client)

        client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)

    def _drop_existing(self, client: MilvusClient) -> None:
        """删除已存在的同名集合（先释放、再按**实际索引名**删索引、最后删集合）。

        [FIX] 原实现写死 `sparse_inverted_index` / `dense_inverted_index` 两个索引名，
        实测在 **Milvus Lite** 上必然失败：Lite 会忽略传入的 `index_name`，
        用**字段名**建索引（`list_indexes()` 返回 `['sparse', 'dense']`），于是
            MilvusException (code=700): no index to drop
        使得「对已存在集合做全量重灌」（`recreate=True`，也是 CLI 默认值）直接失败。
        改为遍历 `list_indexes()` 的真实名字，并容忍单个索引删除失败 ——
        删集合本身会连带删掉索引，这里保留显式删除只是为了沿用原意图。
        """
        try:
            client.release_collection(collection_name=self.collection_name)
        except Exception as e:  # noqa: BLE001
            # 集合从未 load 过时释放会报错，属正常情况
            log.debug(f"释放集合 {self.collection_name} 未成功（可忽略）：{type(e).__name__}: {e}")

        try:
            for index_name in client.list_indexes(collection_name=self.collection_name):
                try:
                    client.drop_index(collection_name=self.collection_name, index_name=index_name)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"删除索引 {index_name} 未成功（可忽略）：{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"列出集合 {self.collection_name} 的索引未成功（可忽略）：{type(e).__name__}: {e}")

        client.drop_collection(collection_name=self.collection_name)

    def create_collection(self):
        """兼容别名，等价于 `recreate()`（保留给旧脚本与外部调用）。"""
        return self.recreate()

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
            try:
                if self.collection_name not in client.list_collections():
                    return  # 集合不存在，无冲突（后续 recreate 会新建）
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
            finally:
                client.close()  # 自检用的临时连接必须关掉，否则每源泄漏一个 client
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning(f"维度自检未能完成（不阻断，可能是 Milvus 未启动）：{type(e).__name__}: {e}")

    def connect(self):
        """建立 langchain Milvus 连接（内部先做维度自检）。"""
        self.check_dim_consistency()
        self.vector_store_saved = Milvus(
            # 惰性：连接集合时才加载嵌入模型（import 期不加载）
            embedding_function=get_embedding(),
            collection_name=self.collection_name,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI},
        )
        # [FIX 2026-11] langchain_milvus 对 HNSW 集合的内置搜索参数是 **ef=10**，
        # 而本项目的检索预取 fetch_k=24、RRF k 最大 36 —— pymilvus 3.x 会直接拒绝搜索：
        #   `ef(10) should be larger than k(24)`
        # 实测后果：**新入库的集合**（首次 recreate）检索必然失败，且被
        # MultiCollectionRetriever 的单集合容错**静默吞掉** —— 表现为该集合的
        # 评测题全部 0 召回、每题检索耗时十几秒（超时等待）。
        # HNSW 官方建议 ef ∈ [topK, 500]，取 128 是召回与延迟的均衡点。
        #
        # 注意要覆盖**两处**：`default_search_params`（源头模板）和
        # `search_params`（构造期 _load() 已从模板拷贝出的实例参数）——
        # 只改前者无效，因为搜索用的是后者。
        _vs = self.store

        def _bump_ef(obj) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "ef":
                        obj[k] = 128
                    else:
                        _bump_ef(v)
            elif isinstance(obj, list):
                for v in obj:
                    _bump_ef(v)

        # default_search_params 在类型存根里是 Collection[str]（只读视图），
        # 运行时是嵌套 dict；用 getattr 取得 Any 再收窄，避免与存根类型冲突。
        base_params = getattr(_vs, "default_search_params", None)
        if isinstance(base_params, dict):
            base_params["HNSW"]["params"]["ef"] = 128
        else:
            # 不静默：该模板决定**后续新建**集合的搜索参数，跳过会让新集合回到
            # ef=10、检索被 pymilvus 拒绝（历史上正是这条静默路径导致的 0 召回）。
            log.warning(
                "default_search_params 结构不符合预期（%s），未能预置 ef=128；"
                "新建集合的检索参数可能仍是 ef=10，请检查 langchain_milvus 版本",
                type(base_params).__name__,
            )
        _bump_ef(getattr(_vs, "search_params", None))

    def create_connection(self):
        """兼容别名，等价于 `connect()`（保留给旧脚本与检索侧调用）。"""
        return self.connect()

    def add_documents(self, datas: List[Document], flush: bool = True):
        """把新的 document 保存到 Milvus 中。

        Args:
            datas: 待写入的块。
            flush: 是否**立刻**把 WAL 刷成可检索状态（默认 True，保持旧行为）。
                批量入库时应传 False，由调用方在整批结束时调用一次 `flush()`：
                实测（真实 Milvus）单次 flush 约 7s —— 按批（20 块）flush 会让
                「5 批 × 20 块」从 7.9s 变成 43.4s（+450%），是批量入库里最容易被
                忽略的开销之一（端到端 md 源 216 块：129s -> 114s）。
                不 flush 不代表丢数据：数据仍在 WAL 中，会被自动刷盘，只是可见时机延后。

        Raises:
            ValueError: datas 为空（不静默：往往意味着上游解析出了问题，见 ROADMAP 1.3.6）。
        """
        if not datas:
            raise ValueError("add_documents 收到空列表：上游解析可能失败，请检查解析器日志")

        self.store.add_documents(datas)
        if flush:
            self.flush()

    def flush(self) -> None:
        """把 WAL 中的数据刷到可检索状态（真实 Milvus 实测约 7s/次，务必按批调用）。"""
        try:
            self.store.client.flush(self.collection_name)
            log.info(f"flush 完成，当前集合 {self.collection_name} 数据已可检索")
        except Exception as e:  # noqa: BLE001
            # flush 失败不应中断入库（数据仍在 WAL 中，会被自动 flush）
            log.warning(f"flush 未成功（不影响数据本身）：{type(e).__name__}: {e}")

    def exists(self) -> bool:
        """目标集合是否存在（用于 `recreate=False` 时拦截「集合名拼错被静默新建」）。"""
        client = MilvusClient(uri=MILVUS_URI)
        try:
            return self.collection_name in client.list_collections()
        finally:
            client.close()


if __name__ == '__main__':
    # 解析文件内容
    # 路径改为配置化：默认取包内 datas/md，可用 DATAS_DIR 覆盖
    file_path = os.path.join(MD_DIR, 'tech_report_0tfhhamx.md')
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入 Milvus 数据库
    mv = MilvusVectorSave()
    mv.recreate()
    mv.connect()
    mv.add_documents(docs)

    client = mv.store.client
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
