"""真实 Milvus 端到端冒烟测试。

验证链路：Markdown 解析 -> 建集合（BM25 稀疏 + 稠密 512 维）-> 写入 -> 混合检索
        -> expr 过滤是否真正生效 -> 清理

与 verify_migration.py 的区别：那个不碰真实服务、只做构造级验证；这个会真的
连你指定的 Milvus，建一个临时集合、写入真实数据、检索、然后删掉该集合。

用法（在服务器上，需先激活 .venv）：
    # 1) 只探测，不写数据
    .venv/bin/python scripts/milvus_probe.py

    # 2) 端到端（会写入临时集合，结束自动删除）
    .venv/bin/python scripts/e2e_smoke_test.py

    # 3) 指定服务与鉴权（也可用 .env）
    MILVUS_URI=127.0.0.1:19530 MILVUS_USER=root MILVUS_PASSWORD=xxx \\
        .venv/bin/python scripts/e2e_smoke_test.py

环境变量：
    MILVUS_URI / MILVUS_USER / MILVUS_PASSWORD / MILVUS_DB
    SMOKE_COLLECTION   临时集合名（默认 smoke_test）
    SMOKE_KEEP=1       保留临时集合不删除（调试用）

嵌入向量说明：
    默认优先用项目真实的本地模型 BAAI/bge-small-zh-v1.5（512 维）。
    若服务器无法访问 HuggingFace（国内常见），会自动降级为确定性伪向量
    （dim=512），此时可验证 schema / 过滤 / 检索链路，但**不验证语义召回质量**。
    脚本会明确打印用的是哪种。
"""

from __future__ import annotations

import os
import sys
import traceback
from hashlib import sha256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RAG_PROJECT"))

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass

COLLECTION = os.getenv("SMOKE_COLLECTION", "smoke_test")
MD_FILE = os.path.join(ROOT, "RAG_PROJECT", "datas", "md", "overview.md")

# 注意：MILVUS_URI 是 pymilvus 自己的保留环境变量，它在 import 阶段就会解析，
# 且只接受 http[s]:// 形式（给成 ./x.db 会让 pymilvus 直接抛
# ConnectionConfigException: Illegal uri）。因此嵌入式 Milvus Lite 的本地库路径
# 不能通过 MILVUS_URI 传入，必须用下面的 SMOKE_MILVUS_URI 在进程内覆盖。
# 对真实服务仍建议沿用 MILVUS_URI=http://host:19530（两种含义一致）。
LITE_URI = os.getenv("SMOKE_MILVUS_URI", "")


def apply_config():
    """在导入业务模块之前覆盖配置（模块顶层会把常量绑定到自己的命名空间）。"""
    import utils.env_utils as env_utils

    if LITE_URI:
        env_utils.MILVUS_URI = LITE_URI
        print(f"[配置] Milvus 地址覆盖为: {LITE_URI}")
        if not LITE_URI.endswith(".db"):
            print("[配置] 非 .db 结尾，按真实服务处理")
        else:
            print("[配置] 嵌入式 Milvus Lite 模式（无需启动服务）")
    env_utils.COLLECTION_NAME = COLLECTION
    print(f"[配置] 集合名 = {COLLECTION}")
    print(f"[配置] jieba 分词器: " + ("已安装" if _has_jieba() else "**未安装**（建表会失败）"))
    return env_utils


def _has_jieba():
    try:
        import jieba  # noqa: F401

        return True
    except Exception:
        return False


def step(n, title):
    print()
    print("-" * 72)
    print(f"[{n}] {title}")
    print("-" * 72)


def make_fallback_embeddings():
    """确定性伪嵌入：同一文本永远得到同一向量，dim=512。

    仅用于在无法下载真实模型时打通链路，不代表语义效果。
    """
    from langchain_core.embeddings import Embeddings

    class HashEmbeddings(Embeddings):
        dim = 512

        def _vec(self, text):
            h = sha256(text.encode("utf-8")).digest()
            out = [((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]
            n = sum(x * x for x in out) ** 0.5 or 1.0
            return [x / n for x in out]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    return HashEmbeddings()


def build_embeddings():
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        v = emb.embed_query("维度自检")
        print(f"  使用真实模型 BAAI/bge-small-zh-v1.5，dim={len(v)}")
        return emb, "real"
    except Exception as e:  # noqa: BLE001
        print(f"  真实模型不可用（{type(e).__name__}: {str(e)[:110]}）")
        print("  -> 降级为确定性伪向量（dim=512）。可验证链路，不验证语义召回质量。")
        return make_fallback_embeddings(), "fake"


def parse_docs():
    from documents.markdown_parser import MarkdownParser

    print(f"  源文件: {MD_FILE}")
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(MD_FILE)
    cats = {}
    for d in docs:
        k = d.metadata.get("category")
        cats[k] = cats.get(k, 0) + 1
    print(f"  解析得到 {len(docs)} 个文档块，category 分布: {cats}")
    return docs


def main():
    # 必须在导入任何业务模块之前应用配置：业务模块在 import 时就把
    # utils.env_utils 的常量绑定到自己的命名空间（from ... import MILVUS_URI）。
    apply_config()

    from documents.milvus_db import MilvusVectorSave

    step(1, "准备嵌入向量")
    emb, mode = build_embeddings()

    step(2, "解析 Markdown（项目自身的 MarkdownParser）")
    docs = parse_docs()

    step(3, f"重建集合 {COLLECTION}（使用项目自身的 schema 与索引参数）")
    import utils.env_utils as env_utils

    env_utils.COLLECTION_NAME = COLLECTION
    os.environ["COLLECTION_NAME"] = COLLECTION
    if LITE_URI:
        env_utils.MILVUS_URI = LITE_URI
    # 预先清掉同名集合。项目的 create_collection() 是「先按固定索引名 drop_index
    # 再 drop_collection」的幂等重建逻辑，而 Milvus Lite 的索引是自动命名，
    # 直接调用会报 code=700 no index to drop。这里绕开它，保证测试可重复运行。
    try:
        from pymilvus import MilvusClient as _MC

        _cli = _MC(uri=LITE_URI or os.environ.get("MILVUS_URI", ""))
        if COLLECTION in _cli.list_collections():
            _cli.drop_collection(COLLECTION)
            print(f"  已清理上一次遗留的集合 {COLLECTION}")
        _cli.close()
    except Exception as e:  # noqa: BLE001
        print(f"  （预清理跳过：{type(e).__name__}: {str(e)[:70]}）")

    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    print(f"  集合 {COLLECTION} 已创建")

    step(4, "写入数据")
    mv.add_documents(docs)
    print(f"  已写入 {len(docs)} 个文档块")

    step(5, "构造检索器（项目自身的 tools/retriever_tools.py 路径）")
    import importlib

    import tools.retriever_tools as rt

    importlib.reload(rt)
    print(f"  检索器: {rt.retriever_tool.name}")
    print("  search_kwargs:")
    sk = rt.retriever.search_kwargs
    for k, v in sk.items():
        print(f"    {k} = {repr(v)[:96]}")

    step(6, "执行检索（k=4，expr 过滤 category=='content'）")
    hits = rt.retriever.invoke("光刻机是怎么工作的")
    print(f"  返回 {len(hits)} 条")
    ok = True
    if len(hits) > 4:
        print("  [!] 返回条数超过 k=4，检索参数可能未生效")
        ok = False
    bad = [h for h in hits if h.metadata.get("category") != "content"]
    for i, h in enumerate(hits, 1):
        t = (h.page_content or "")[:60].replace("\n", " ")
        print(f"   {i}. category={h.metadata.get('category')!r} len={len(h.page_content)} | {t}")
    if bad:
        print(f"  [!] 有 {len(bad)} 条结果的 category != 'content'，说明 expr 过滤未生效")
        ok = False
    else:
        print("  [OK] 全部结果的 category 均为 'content' —— expr 过滤真正生效")

    step(7, "清理")
    if os.getenv("SMOKE_KEEP") == "1":
        print(f"  SMOKE_KEEP=1，保留集合 {COLLECTION}")
    else:
        from pymilvus import MilvusClient as _MC2

        _cli2 = _MC2(uri=LITE_URI or os.environ.get("MILVUS_URI", ""))
        _cli2.drop_collection(COLLECTION)
        _cli2.close()
        print(f"  已删除临时集合 {COLLECTION}")

    print()
    print("=" * 72)
    print(f"端到端冒烟测试结束：{'全部通过' if ok else '存在问题，见上面 [!] 标记'}")
    print(f"（嵌入向量模式：{mode}）")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print()
        print("[致命错误] 测试中断：")
        traceback.print_exc()
        sys.exit(3)
