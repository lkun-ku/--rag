from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_kb.utils.env_utils import (
    OPENAI_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_DISABLE_THINKING,
)

# [UPGRADE 2026-10] base_url / model / temperature 原先硬编码在这里，换服务商必须改代码；
# 现改为从环境变量读取（见 rag_kb/utils/env_utils.py），默认值与迁移前一致。
# 例：DeepSeek 官方
#   LLM_BASE_URL=https://api.deepseek.com
#   LLM_MODEL=deepseek-flash
# DeepSeek 的 thinking 模式不支持强制 tool_choice，而本项目的结构化输出判分链
# （with_structured_output(method='function_calling')）依赖 tool_choice，
# 因此默认关闭 thinking（可用 LLM_DISABLE_THINKING=0 打开）。
_EXTRA_BODY = {"thinking": {"type": "disabled"}} if LLM_DISABLE_THINKING else None

# 本机系统代理会对 HTTPS 做 TLS 中间人，而 Python 用 certifi 证书库、不认其根证书 ——
# 表现为 `OpenAIConnectionError: Connection error` / `SSLCertVerificationError`
# （Tavily 搜索同理）。把这两个域名排除出代理即可；公司代理环境设 KB_KEEP_PROXY=1 关闭。
from rag_kb.utils.http_local import bypass_proxy, host_of  # noqa: E402

bypass_proxy(host_of(LLM_BASE_URL), 'api.tavily.com')

llm = ChatOpenAI(  # openai的
    temperature=LLM_TEMPERATURE,
    model=LLM_MODEL,
    # ChatOpenAI 的 api_key 只接受 SecretStr（或可调用对象），不能直接传裸 str；
    # 未配置时保持 None，让 SDK 在调用时报出与之前一致的错误。
    api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
    base_url=LLM_BASE_URL,
    extra_body=_EXTRA_BODY,
)


# [2026-10-28] `langchain_community.tools.TavilySearchResults` 已在 LangChain 0.3.25 弃用、
# 1.0 将移除，改用官方独立包 `langchain-tavily` 的 `TavilySearch`。
#
# 成功时返回原始响应 dict（`{'results': [{url, content, ...}]}`）；
# 失败时返回 `{'error': ...}`，两种形态都由 `web_search_node` 兼容。
web_search_tool = TavilySearch(max_results=2)
