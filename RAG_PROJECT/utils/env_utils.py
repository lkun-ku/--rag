import os

from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================================
# 配置外置
#
# 迁移前 MILVUS_URI / COLLECTION_NAME 以及各脚本里的 E:\path\to\... 路径是
# 硬编码的，换机即失败。改为一律从环境变量读取、并保留与迁移前完全相同的默认值，
# 因此在未设置任何新环境变量时，行为与迁移前完全一致。
#
# 密钥沿用本项目原有的 os.getenv + python-dotenv 风格（与 LangChain 自身依赖的
# langchain_core.utils.env 属于同一类 env 读取约定）。
# ============================================================================

# ---- 密钥（必需，缺失会导致 import 期即失败）----
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
# 补充读取：此前代码从未显式读取过该变量，完全依赖 Tavily SDK 自行从环境读取。
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# ---- 向量库：可用 .env 覆盖，默认值与迁移前一致 ----
MILVUS_URI = os.getenv('MILVUS_URI', 'http://<old-ip>:19530')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 't_collection01')

# ---- LLM 服务商配置（换服务商/换模型不用改代码）----
# 原项目把 base_url 与 model 硬编码在 llm_models/all_llm.py 里，换服务商必须改代码。
# 现改为环境变量读取，默认值保持迁移前的写法。
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://xiaoai.plus/v1')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0'))

# 结构化输出方式。langchain-openai 1.x 的 with_structured_output 默认值已从
# 'function_calling' 改成 'json_schema'，而 DeepSeek 官方 API 不支持 json_schema
# （报 "This response_format type is unavailable now"），
# 因此这里显式指定为 function_calling。换到支持 json_schema 的服务商可改这里。
LLM_STRUCTURED_METHOD = os.getenv('LLM_STRUCTURED_METHOD', 'function_calling')

# 是否关闭 DeepSeek 的 thinking 模式。thinking 开启时不支持强制 tool_choice，
# 而 function_calling 依赖 tool_choice，因此默认关闭。
LLM_DISABLE_THINKING = os.getenv('LLM_DISABLE_THINKING', '1') not in ('0', 'false', 'False')

# ---- 路径：消除硬编码 ----
# PKG_ROOT 即 Python 包根（RAG_PROJECT/RAG_PROJECT），其下有 datas/ 等目录
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT_DIR 是工作区根（.venv、requirements.txt 所在层级）
ROOT_DIR = os.path.dirname(PKG_ROOT)

# 语料目录：默认取包内 datas/md，可用 DATAS_DIR 覆盖
DATAS_DIR = os.getenv('DATAS_DIR') or os.path.join(PKG_ROOT, 'datas')
MD_DIR = os.path.join(DATAS_DIR, 'md')

# 混合检索的 reranker 写法（二选一）：
#   function —— 官方新写法 reranker=Function(RERANK)，语义最准，但要求服务端 Milvus 2.6+
#   legacy   —— 旧写法 ranker_type='rrf' + ranker_params，已 deprecated 但兼容面最广
#              （Milvus Lite、以及 2.6 以下的服务端只支持这种）
# 实测：Milvus Lite 3.2.1 对 function 形式报
#       MilvusException code=6 "Unsupported Hybrid FunctionScore reranker"，
#       对 legacy 形式正常返回。
MILVUS_RERANKER_STYLE = (os.getenv('MILVUS_RERANKER_STYLE', 'function') or 'function').strip().lower()

# LangGraph 会话记忆持久化文件（SqliteSaver），可用 CHECKPOINT_DB 覆盖
CHECKPOINT_DB = os.getenv('CHECKPOINT_DB') or os.path.join(ROOT_DIR, 'checkpoints.db')
