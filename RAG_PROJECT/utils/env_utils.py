import json
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

# Milvus Lite（嵌入式）库路径，与 MILVUS_URI 二选一。
#
# 为什么不直接用 MILVUS_URI 传本地路径：
#   MILVUS_URI 是 **pymilvus 自己的保留环境变量**，它在 import 阶段就被解析，
#   且只接受 unix/http/tcp/grpc/local 前缀的形式。传 `./x.db` 会让 pymilvus
#   在 import 时直接抛 ConnectionConfigException（实测）。
#
# 为什么需要单独的环境变量（而不是进程内覆盖）：
#   write_milvus 走多进程，子进程会重新 import 本模块 —— 进程内 monkey-patch
#   不会被继承。所以 Lite 路径必须能通过环境变量传递。
#
# 用法（PowerShell）：
#   $env:MILVUS_LITE_PATH="./_test_lite/milvus.db"
MILVUS_LITE_PATH = (os.getenv('MILVUS_LITE_PATH') or '').strip()
if MILVUS_LITE_PATH:
    MILVUS_URI = MILVUS_LITE_PATH
    # Milvus Lite **不会自动创建父目录**，路径上一层不存在时会直接报
    #   ConnectionConfigException: Open local milvus failed, dir: xxx not exists
    # 这里预创建，避免用户每次都要手动 mkdir。
    _lite_parent = os.path.dirname(os.path.abspath(MILVUS_LITE_PATH))
    if _lite_parent:
        os.makedirs(_lite_parent, exist_ok=True)

# ---- 路径：消除硬编码 ----
# PKG_ROOT 即 Python 包根（RAG_PROJECT/RAG_PROJECT），其下有 datas/ 等目录
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT_DIR 是工作区根（.venv、requirements.txt 所在层级）
ROOT_DIR = os.path.dirname(PKG_ROOT)

# 语料目录：默认取包内 datas，可用 DATAS_DIR 覆盖
DATAS_DIR = os.getenv('DATAS_DIR') or os.path.join(PKG_ROOT, 'datas')
MD_DIR = os.path.join(DATAS_DIR, 'md')

# ============================================================================
# 嵌入模型（配置化：升维 / 换模型 / A/B 对照都只改 env，不动代码）
#
# 规范：模型文件随项目放在 MODELS_DIR 下（不进 git，见 .gitignore）。
#       scripts/fetch_models.py 可重建。
#
# 默认指向项目内的 bge-large-zh-v1.5（1024 维）。
# 做 512 维 A/B 对照时切回小模型即可：
#     EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
#     EMBEDDING_DIM=512
#     KB_COLLECTION=kb_milvus_bge-small_512
# 依据见 ROADMAP 1.1.6：小模型必须保留，否则失去对照能力。
# ============================================================================
MODELS_DIR = os.getenv('MODELS_DIR') or os.path.join(PKG_ROOT, 'models')

_DEFAULT_MODEL_DIR = os.path.join(MODELS_DIR, 'bge-large-zh-v1.5')
# 项目内目录存在就用本地路径（离线可加载）；否则退回 HF repo id（需要联网或缓存）
_DEFAULT_EMBEDDING = _DEFAULT_MODEL_DIR if os.path.isdir(_DEFAULT_MODEL_DIR) else 'BAAI/bge-large-zh-v1.5'

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or _DEFAULT_EMBEDDING
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM') or 1024)

# ============================================================================
# 集合命名（规范：kb_{domain}_bge-large_{dim}）
#
# COLLECTION_NAME 是「默认集合」，向后兼容旧脚本与旧配置；
# 正式的多源摄入按下方 DOC_SOURCES 里各自的 collection 分集合写入，
# 以避免不同语义域的内容相互干扰（依据见 ROADMAP 1.2）。
# ============================================================================
COLLECTION_NAME = os.getenv('COLLECTION_NAME') or 'kb_milvus_bge_large_1024'

# ============================================================================
# 多文档源清单（一源目录 ↔ 一集合）
#
# 核心是「语义域隔离」，不是「格式隔离」：同一语义域的文档放同一目录
#（可含多种格式），不同语义域用不同目录 → 不同集合。
#
# parser 取值：
#   markdown —— 原生 .md，直接走 MarkdownParser
#   pdf      —— pymupdf4llm 转 Markdown 后走 MarkdownParser
#   office   —— markitdown 转 Markdown 后走 MarkdownParser（docx / xlsx / pptx）
#   text     —— .txt，**默认关闭**：按决策需先人工整理为 md（见 ROADMAP 1.3.3）
#
# dir 不存在时该源会被自动跳过（不报错），因此新目录可先建配置后补文件。
# ============================================================================
DOC_SOURCES = [
    {
        'name': 'milvus_docs',
        'dir': os.getenv('SRC_DIR_MD') or MD_DIR,
        'collection': os.getenv('KB_COLLECTION_MD') or 'kb_milvus_bge_large_1024',
        'formats': ('.md',),
        'parser': 'markdown',
        'enabled': True,
    },
    {
        'name': 'papers',
        'dir': os.getenv('SRC_DIR_PDF') or os.path.join(DATAS_DIR, 'pdf'),
        'collection': os.getenv('KB_COLLECTION_PDF') or 'kb_papers_bge_large_1024',
        'formats': ('.pdf',),
        'parser': 'pdf',
        'enabled': True,
    },
    {
        'name': 'manuals',
        'dir': os.getenv('SRC_DIR_OFFICE') or os.path.join(DATAS_DIR, 'office'),
        'collection': os.getenv('KB_COLLECTION_OFFICE') or 'kb_manuals_bge_large_1024',
        'formats': ('.docx', '.xlsx', '.pptx'),
        'parser': 'office',
        'enabled': True,
    },
    {
        'name': 'notes',
        'dir': os.getenv('SRC_DIR_TXT') or os.path.join(DATAS_DIR, 'txt'),
        'collection': os.getenv('KB_COLLECTION_TXT') or 'kb_notes_bge_large_1024',
        'formats': ('.txt',),
        'parser': 'text',
        # 按决策关闭：txt 需先人工整理为 Markdown 再入库（ROADMAP 1.3.3）
        'enabled': False,
    },
]

# ============================================================================
# 多租户配置（ROADMAP 3.3）
# ============================================================================
# 租户 -> 允许访问的**源名**列表（源名见上方 DOC_SOURCES）。
#
# 为什么用「源名」而不是「集合名」：DOC_SOURCES 已经定义了「源 -> 集合」的映射，
# 再配一遍集合名容易出现两处不一致（改了 collection 忘了改租户表）。
#
# 为什么**不给 schema 加 tenant 字段**（ROADMAP 3.3 的原始设想）：
#   加字段意味着**重建所有集合**，而当前语料里并没有真实的租户数据 ——
#   加了也无法验证，属于「造出没人用的功能」。
#   而阶段一的「一源一集合」本身就是天然的隔离边界：不同语义域的文档
#   本来就在不同集合里，租户只需在这条边界上做**集合子集映射**。
#   这样零 schema 改动、零迁移成本，**且能被测试证明**（租户 A 检索不到租户 B 的块）。
#
# 若将来真需要「同一集合内多租户」（例如两个部门共用 milvus_docs 集合），
# 那就必须加 schema 字段并重建 —— 届时先跑评测基线再动。
#
# 格式（.env 或环境变量）：
#   KB_TENANTS={"acme": ["milvus_docs", "manuals"], "beta": ["papers"]}
KB_TENANTS_JSON = os.getenv('KB_TENANTS')
# 未配置时为空 dict -> 退化为「单租户模式」（可访问全部 enabled 源）。
# 这是**刻意的默认**：不设置该环境变量时，行为与引入本功能前完全一致。
TENANTS = json.loads(KB_TENANTS_JSON) if KB_TENANTS_JSON else {}

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

# 结构化输出方式。langchain-openai 1.x 的 with_structured_output 默认值已从
# 'function_calling' 改成 'json_schema'，而 DeepSeek 官方 API 不支持 json_schema
# （报 "This response_format type is unavailable now"），
# 因此这里显式指定为 function_calling。换到支持 json_schema 的服务商可改这里。
LLM_STRUCTURED_METHOD = os.getenv('LLM_STRUCTURED_METHOD', 'function_calling')

# 是否关闭 DeepSeek 的 thinking 模式。thinking 开启时不支持强制 tool_choice，
# 而 function_calling 依赖 tool_choice，因此默认关闭。
LLM_DISABLE_THINKING = os.getenv('LLM_DISABLE_THINKING', '1') not in ('0', 'false', 'False')

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
