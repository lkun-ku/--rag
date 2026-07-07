
from pydoc import doc
from unittest import result

from pymilvus import MilvusClient

from RAG_PROJECT.documents.markdown_parser import MarkdownParser
from RAG_PROJECT.documents.milvus_db import MilvusVectorSave
from RAG_PROJECT.utils.env_utils import MILVUS_URI


if __name__ == "__main__":
    mv=MilvusVectorSave()
    mv.create_connection()  
    result=mv.vector_store_saved.similarity_search_with_score(  # pyright: ignore[reportUnknownMemberType]
        query="半导体和芯片的封装、测试、光刻胶等",
        k=3,
        expr='category == "content"'
    )
    # similarity_search_with_score 返回的是一个元组，第一个是文档id，第二个是相似度分数
    for doc in result:
        print(doc)


def text2():
    """创建一个新的collection"""
    client = MilvusClient(uri=MILVUS_URI)

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VECTOR, max_length=6000,enable_analyzer=True,
                    analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    schema.add_field(field_name="category", datatype=DataType.VECTOR, max_length=6000,enable_lazy=True)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        name="text_bm25_emb",  # Function name
        input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse"],
        # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,  # Set to `BM25`
    )
    schema.add_function(bm25_function)

def text3():
    """往collection中插入数据"""
    vector_store = Milvus(
        embedding_function=None,
    )

    #解析文件内容
    file_path =r'<local-user>\Desktop\<project-dir>\RAG_PROJECT\documents\md_files\半导体和芯片的封装、测试、光刻胶等.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    
    vector_store.add_documents(docs)

def text4():
    """全文搜索测试"""
    vector_store_saved = Milvus(
        embedding_function=bge_embedding,
        collection_name=COLLECTION_NAME,
        builtin_function=BM25BuiltInFunction(),
        vector_field=['dense', 'sparse'],
        consistency_level="Strong",
        auto_id=True,
        connection_args={"uri": MILVUS_URI}
    )

    res = vector_store_saved.similarity_search_with_score(  # pyright: ignore[reportUnknownMemberType]
        query="活性氧原子",
        k=3,
        expr='category == "content"'
    )
    for doc in res:
        print(doc)

def text5():
    """采用PyMilvus的库来进行检索"""
    client = MilvusClient(uri=MILVUS_URI)
    client.search(
        collection_name='demo',
        data=['半导体表面特征改善'],
        anns_field='sparse',
        output_fields=['text','category','id'],
        search_params={"params":{"metric_type": "L2"}},
        limit=3,
    )
    for item in res[0]:
        print(item)

if  __name__ == "__main__":
    text2() #创建一个新的collection表结构
    text3() #插入数据
    text4() #全文搜索测试

