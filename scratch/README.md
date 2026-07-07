# scratch —— 非生产代码存放区

本目录下的文件**不参与项目运行**，也没有任何模块 import 它们。
放在这里是为了让编辑器的静态检查跳过它们，避免持续报错污染项目代码视图。

## text_search.py

个人实验稿：用不同方式建表 / 检索做对比测试的草稿。
存在多处未定义名称（`DataType`、`Function`、`FunctionType`、`Milvus`、
`bge_embedding`、`COLLECTION_NAME`、`res` 等），**直接调用会报 NameError**。

原位置：`RAG_PROJECT/tools/text_search.py`
移出原因：它与生产检索层无关，却会在 IDE 里产生 12 个真实报错。
参考实现请看 `RAG_PROJECT/tools/retriever_tools.py` 与
`RAG_PROJECT/documents/milvus_db.py`。
