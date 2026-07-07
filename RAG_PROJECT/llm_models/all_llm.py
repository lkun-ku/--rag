from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import (OPENAI_API_KEY, DEEPSEEK_API_KEY, LLM_BASE_URL,
                             LLM_MODEL, LLM_TEMPERATURE, LLM_DISABLE_THINKING)

# [UPGRADE 2026-10] base_url / model / temperature 原先硬编码在这里，换服务商必须改代码；
# 现改为从环境变量读取（见 utils/env_utils.py），默认值与迁移前一致。
# 例：DeepSeek 官方
#   LLM_BASE_URL=https://api.deepseek.com
#   LLM_MODEL=deepseek-flash
# DeepSeek 的 thinking 模式不支持强制 tool_choice，而本项目的结构化输出判分链
# （with_structured_output(method='function_calling')）依赖 tool_choice，
# 因此默认关闭 thinking（可用 LLM_DISABLE_THINKING=0 打开）。
_EXTRA_BODY = {"thinking": {"type": "disabled"}} if LLM_DISABLE_THINKING else None

llm = ChatOpenAI(  # openai的
   temperature=LLM_TEMPERATURE,
   model=LLM_MODEL,
   api_key=OPENAI_API_KEY,
   base_url=LLM_BASE_URL,
   extra_body=_EXTRA_BODY)


web_search_tool = TavilySearchResults(max_results=2)
