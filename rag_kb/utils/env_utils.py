import hashlib
import os
import pathlib
from typing import Any, Dict, List

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

# ---- 向量库：可用 .env 覆盖 ----
# [FIX / 深度扫描 R9] 默认值原为 `<旧公网 IP>`（一个**公网 IP**）——
# 那是迁移前的遗留写法。隐患：`.env` 缺失时**不会「连不上」**，而是静默去连一台外部
# 服务器 —— 既可能读到别人的数据，也让报错方向完全被误导（以为是本地 Milvus 的问题）。
# 改为本地地址，与 `.env.example` 保持一致。（本机 `.env` 已显式配置，故实际零影响。）
MILVUS_URI = os.getenv('MILVUS_URI', 'http://127.0.0.1:19530')

# Milvus Lite（嵌入式）库路径，与 MILVUS_URI 二选一。
#
# 为什么不直接用 MILVUS_URI 传本地路径：
#   MILVUS_URI 是 **pymilvus 自己的保留环境变量**，它在 import 阶段就被解析，
#   且只接受 unix/http/tcp/grpc/local 前缀的形式。传 `./x.db` 会让 pymilvus
#   在 import 时直接抛 ConnectionConfigException（实测）。
#
# 为什么需要单独的环境变量（而不是进程内覆盖）：
#   部分场景会在**子进程**里重新 import 本模块（write_milvus 的可选解析子进程、
#   迁移/评测脚本的多进程写法）—— 进程内 monkey-patch 不会被继承。
#   所以 Lite 路径必须能通过环境变量传递。
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
# PKG_ROOT 是「项目根」：含 datas/ models/ logs/ 与 rag_kb/ 的那一层，
# 即 **Python 包 `rag_kb/` 的上一层** —— 代码在包里，数据在包外，二者刻意分离。
# 层级：rag_kb/utils/env_utils.py -> rag_kb/utils -> rag_kb -> 项目根
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 项目根即工作区根（.venv、requirements.txt、checkpoints.db 所在层级）
ROOT_DIR = PKG_ROOT

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
# 多文档源：**自动发现** + 显式配置（可选）
#
# [2026-11 通用化] 核心变化：`datas/` 下的每个**直接子目录**自动成为一个源，
# 不再需要手工维护源清单。
#
#   datas/
#   └── docs/      ← 一个目录 = 一个主题包 = 一个 Milvus 集合
#       ├── 01-HTML5.md
#       ├── 某论文.pdf      ← 格式混放：按扩展名自动选解析器（parser='auto'）
#       └── 某手册.docx
#
# 规则：
#   1. 目录名 = 源名（`main.py ingest --sources docs`）；
#   2. 集合名自动生成：kb_auto_<md5(目录名)[:8]>_bge_large_1024（稳定且唯一）；
#   3. **支持递归**：子目录里的文件也会被收进来；
#   4. 目录内可以混放 md / pdf / docx / xlsx / pptx / html / txt；
#   5. 主题描述默认取目录名（会拼进路由提示词）。想写得更精确，
#      在目录里放一个 `_topic.txt`，首行即主题描述。
#
# 需要精细控制（指定集合名 / 特殊解析器 / 独立目录）时，往 `_EXPLICIT_DOC_SOURCES`
# 里加一条即可（结构见下方注释），显式源优先于自动源（同目录不重复发现）。
# ============================================================================
AUTO_FORMATS = (
    '.md', '.markdown', '.pdf', '.docx', '.xlsx', '.pptx', '.html', '.htm', '.txt',
)

# 显式源：**默认为空** —— 自动发现已覆盖「建目录 + 放文件」的场景。
# 需要指定集合名等精细控制时，按这个结构添加（dir 不存在时该源自动跳过）：
_EXPLICIT_DOC_SOURCES: List[Dict[str, Any]] = [
    # {
    #     'name': '公司制度',
    #     'dir': os.path.join(DATAS_DIR, '公司制度'),
    #     'collection': 'kb_policy_bge_large_1024',
    #     'formats': AUTO_FORMATS,
    #     'parser': 'auto',
    #     'enabled': True,
    #     'topic': '公司内部制度与流程',
    # },
]


def discover_auto_sources(datas_dir: str) -> List[Dict[str, Any]]:
    """扫描 `datas/` 的直接子目录，每个生成一个「混合格式」源。

    * 目录名 -> 源名；集合名 = kb_auto_<md5(目录名)[:8]>_bge_large_1024（稳定唯一）；
    * 被显式源占用的目录不重复发现；
    * `.` / `_` 开头的目录跳过（放私有素材用）；
    * 子目录里有 `_topic.txt` 时，首行作为该源的路由主题描述。
    """
    root = pathlib.Path(datas_dir)
    if not root.is_dir():
        return []
    explicit_dirs = {
        os.path.normcase(os.path.abspath(s['dir'])) for s in _EXPLICIT_DOC_SOURCES if s.get('dir')
    }
    out: List[Dict[str, Any]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(('.', '_')):
            continue
        if os.path.normcase(os.path.abspath(str(d))) in explicit_dirs:
            continue

        cid = hashlib.md5(d.name.encode('utf-8')).hexdigest()[:8]
        topic = ''
        topic_file = d / '_topic.txt'
        if topic_file.is_file():
            topic = topic_file.read_text(encoding='utf-8').strip().splitlines()[0][:120]
        if not topic:
            topic = f'「{d.name}」目录下的文档'

        out.append(
            {
                'name': d.name,
                'dir': str(d),
                'collection': f'kb_auto_{cid}_bge_large_1024',
                'formats': AUTO_FORMATS,
                'parser': 'auto',
                'enabled': True,
                'topic': topic,
                'auto': True,
            }
        )
    return out


DOC_SOURCES: List[Dict[str, Any]] = [
    *_EXPLICIT_DOC_SOURCES,
    *discover_auto_sources(DATAS_DIR),
]

# 路由提示词用：把 enabled 源的 topic 拼成编号列表（无 topic 的源自动忽略）
KB_TOPIC_BLOCK = '\n'.join(
    '%d. %s' % (i, s['topic']) for i, s in enumerate([x for x in DOC_SOURCES if x.get('enabled', True) and x.get('topic')], 1)
)

# ---- LLM 服务商配置（换服务商/换模型不用改代码）----
# 原项目把 base_url 与 model 硬编码在 rag_kb/llm_models/all_llm.py 里，换服务商必须改代码。
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

# [FIX / P0-7] 上方这两个配置项曾**被整段复制了一遍**（含注释），
# 由阶段二首个提交 e8c506d 引入。两份取值相同故当时无功能差异，但
# `LLM_STRUCTURED_METHOD` / `LLM_DISABLE_THINKING` 被 all_llm.py、grader_chain.py、
# query_route_chain.py 同时读取 —— **改一处忘另一处就会静默行为漂移**，故删除重复的一份。

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
