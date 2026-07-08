"""Milvus 服务端只读探测脚本（不写入任何数据）。

用途：在真实 Milvus 上确认版本、鉴权方式、现有集合，以及本项目的 schema 能否被
该版本支持。这是端到端测试前的必要一步——尤其 reranker 的两种写法对服务端版本
有要求（见输出 [4]）。

在 AutoDL 服务器上运行（需先激活本项目 .venv 并配好 .env）：
    .venv/bin/python scripts/milvus_probe.py

可用的环境变量（也可用 .env）：
    MILVUS_URI      服务地址，如 127.0.0.1:19530
    MILVUS_USER     用户名（无鉴权则留空）
    MILVUS_PASSWORD 密码（无鉴权则留空）
    MILVUS_DB       数据库名（默认 default）
    COLLECTION_NAME 要检查的集合名，默认 t_collection01

本脚本只做读操作：list / describe / get_server_version，不创建或修改任何集合。
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RAG_PROJECT"))

try:  # 允许在没有项目完整环境时独立运行（只需 pymilvus）
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(override=True)
except Exception:  # noqa: BLE001
    pass


def line(char="-", n=74):
    print(char * n)


def probe_client():
    from pymilvus import MilvusClient

    uri = os.getenv("MILVUS_URI", "")
    user = os.getenv("MILVUS_USER", "") or ""
    pwd = os.getenv("MILVUS_PASSWORD", "") or ""
    db = os.getenv("MILVUS_DB", "default")

    line("=")
    print("Milvus 服务端只读探测")
    line("=")
    print("MILVUS_URI      = " + (uri or "(未设置)"))
    print("MILVUS_USER     = " + (user or "(无)"))
    print("MILVUS_PASSWORD = " + ("(已设置)" if pwd else "(无)"))
    print("MILVUS_DB       = " + db)
    line()

    if not uri:
        print("[结果] 未设置 MILVUS_URI，无法连接。请在 .env 中配置。")
        return None

    kwargs = {"uri": uri}
    if user:
        kwargs["user"] = user
        kwargs["password"] = pwd
    if db:
        kwargs["db_name"] = db

    try:
        client = MilvusClient(**kwargs)
        print("[连接] 成功")
        return client
    except Exception as e:
        print("[连接] 失败：" + type(e).__name__ + ": " + str(e)[:300])
        print()
        print("排查建议：")
        print("  1) 服务是否已启动：ss -lntp | grep 19530  或  docker ps | grep milvus")
        print("  2) 容器内访问用 127.0.0.1；跨机需用内网/公网 IP 并放行端口")
        print("  3) 若启用鉴权，必须同时提供 MILVUS_USER / MILVUS_PASSWORD")
        return None


def report_version(client):
    line()
    print("[1] 服务端版本")
    for fn in ("get_server_version", "get_version"):
        try:
            v = getattr(client, fn)()
            print("  " + fn + "() = " + str(v))
            return str(v)
        except Exception as e:
            print("  " + fn + "() 不可用：" + type(e).__name__ + ": " + str(e)[:80])
    return None


def report_collections(client):
    line()
    print("[2] 现有集合")
    try:
        cols = client.list_collections()
        print("  共 " + str(len(cols)) + " 个：" + str(cols))
        return cols
    except Exception as e:
        print("  读取失败：" + type(e).__name__ + ": " + str(e)[:120])
        return []


def report_project_collection(client, cols):
    name = os.getenv("COLLECTION_NAME", "t_collection01")
    line()
    print("[3] 本项目集合 '" + name + "' 的结构")
    if name not in cols:
        print("  集合不存在（首次运行时由 milvus_db.py 创建）")
        return
    try:
        d = client.describe_collection(collection_name=name)
    except Exception as e:
        print("  describe 失败：" + type(e).__name__ + ": " + str(e)[:150])
        return
    print("  字段数：" + str(len(d.get("fields", []))))
    for f in d.get("fields", []):
        params = f.get("params") or {}
        extra = ""
        if params.get("enable_analyzer"):
            extra = "  enable_analyzer=true"
            ap = params.get("analyzer_params") or {}
            # Milvus 3.0.1 的 describe_collection 把 analyzer_params 返回为
            # **JSON 字符串**（2.6 返回 dict），两者都要兼容。
            if isinstance(ap, str):
                try:
                    ap = json.loads(ap)
                except Exception:  # noqa: BLE001
                    ap = {}
            if isinstance(ap, dict) and ap:
                extra += " tokenizer=" + str(ap.get("tokenizer"))
            elif ap:
                extra += " analyzer_params=" + str(ap)[:60]
        if f.get("type") in ("FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR"):
            extra += "  dim=" + str(params.get("dim", "-"))
        if params.get("function"):
            extra += "  function=" + str(params.get("function"))
        print("    - " + str(f.get("name")).ljust(16) + " " + str(f.get("type")) + extra)
    try:
        for idx in client.list_indexes(collection_name=name):
            info = client.describe_index(collection_name=name, index_name=idx)
            print("    索引 " + idx + ": " + str(info.get("index_type"))
                  + " metric=" + str(info.get("metric_type"))
                  + " params=" + str(info.get("params")))
    except Exception as e:
        print("    索引读取失败：" + type(e).__name__ + ": " + str(e)[:100])


def judge_reranker_support(version):
    line()
    print("[4] 结论：本项目 reranker 写法的兼容性")
    print("  项目现用：reranker=Function(function_type=RERANK, ...)")
    print("  langchain-milvus 源码明确：Function reranker 需要服务端 Milvus 2.6+")
    if not version:
        print("  版本未知 -> 直接跑 e2e_smoke_test.py 验证；若报 rerank 相关错误，")
        print("  把 tools/retriever_tools.py 改回 ranker_type='rrf' + ranker_params={'k':100}")
        return
    digits = [int(x) for x in re.findall(r"\d+", version)[:2]]
    major = digits[0] if digits else 0
    minor = digits[1] if len(digits) > 1 else 0
    if (major, minor) >= (2, 6):
        print("  服务端 " + version + " >= 2.6 -> 当前 reranker=Function 写法可用")
    else:
        print("  服务端 " + version + " < 2.6 -> Function reranker 可能不被支持，需要回退为：")
        print('      "ranker_type": "rrf",')
        print('      "ranker_params": {"k": 100},')
        print("  （pymilvus 3.0.2 中 RRFRanker 仍存在，该旧写法可用，只是有弃用告警）")


def main():
    client = probe_client()
    if client is None:
        return 2
    version = report_version(client)
    cols = report_collections(client)
    report_project_collection(client, cols)
    judge_reranker_support(version)
    line()
    print("探测完成。本脚本未做任何写操作。")
    line("=")
    return 0


if __name__ == "__main__":
    sys.exit(main())
