"""Checkpointer 工厂（LangGraph 官方推荐的持久化实现）。

依据 LangGraph 官方持久化文档：InMemorySaver / MemorySaver 只把检查点存在内存中，
进程重启即全部丢失；官方建议生产环境改用持久化实现（PostgresSaver 为生产首选，
SqliteSaver 面向本地开发与小型项目）。

本项目为同步、轻量的 RAG 应用，故采用 SqliteSaver（其 docstring 定位为
"lightweight, synchronous use cases"）。真正上生产请换成 PostgresSaver。
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from utils.env_utils import CHECKPOINT_DB


def create_checkpointer() -> SqliteSaver:
    """创建 SqliteSaver（进程级单例）。

    建表不需显式触发：SqliteSaver.setup() 的官方 docstring 明确
    "It is called automatically when needed and should not be called directly
    by the user"，首次读写检查点时内部会自动建表。

    连接参数与官方 SqliteSaver.from_conn_string 保持一致（check_same_thread=False），
    但直接持有连接而非使用 with，因为本工厂返回的是进程级单例、需活到进程结束。
    """
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)
