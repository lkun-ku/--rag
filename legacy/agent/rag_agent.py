# =============================================================================
# 本文件已迁移到 LangChain 1.x 官方标准写法。
#
# 迁移前（仅作对照，已删除）：
#   from langchain.agents import create_tool_calling_agent, AgentExecutor
#   prompt = ChatPromptTemplate.from_messages([
#       ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题'),
#       MessagesPlaceholder(variable_name='chat_history', optional=True),
#       ('human', '{input}'),
#       MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
#   ])
#   agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
#   executor = AgentExecutor(agent=agent, tools=[retriever_tool])
#   agent_with_history = RunnableWithMessageHistory(
#       executor, get_session_history,
#       input_messages_key='input', history_messages_key='chat_history')
#   agent_with_history.invoke({'input': '什么是EUV光刻机？'},
#                             config={'configurable': {'session_id': 'zs123'}})
#
# 1.x 标准写法：
#   * create_agent() 取代 create_tool_calling_agent() + AgentExecutor()
#     —— 1.x 的 agent 本身就是一个已编译的 LangGraph 图，AgentExecutor 已不再需要。
#   * system_prompt 直接传字符串；chat_history 与 agent_scratchpad 两个占位符
#     由 create_agent 内部自动装配，无需手工编排 ChatPromptTemplate。
#   * 多轮会话记忆由 checkpointer（检查点）+ thread_id 承担，这是 1.x 的标准做法，
#     取代旧的 RunnableWithMessageHistory + ChatMessageHistory。
#   * 输入输出遵循 AgentState 规范：输入 {'messages': [...]}，
#     输出 {'messages': [...]}，最终答案取最后一条消息的 content。
#
# 业务逻辑保持不变：系统提示词原文、所用工具、会话标识 'zs123'、提问内容均未改动。
# =============================================================================
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_kb.llm_models.all_llm import llm
from rag_kb.tools.retriever_tools import retriever_tool
from rag_kb.utils.checkpointer import create_checkpointer

# 系统提示词：与迁移前逐字一致
SYSTEM_PROMPT = '你是一个智能助手，尽可能的调用工具回答用户的问题'

# 会话标识：与迁移前的 session_id 'zs123' 对应，1.x 中改由 thread_id 承载
SESSION_ID = 'zs123'

# 检查点：1.x 中承担「按会话隔离的多轮对话记忆」
# 由 InMemorySaver 改为官方推荐的 SqliteSaver（文件持久化），会话记忆在进程重启后
# 仍可恢复；选型说明见 rag_kb/utils/checkpointer.py。
checkpointer = create_checkpointer()

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# 1.x 的 config 用 thread_id 指定会话
config: RunnableConfig = {'configurable': {'thread_id': SESSION_ID}}

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."

if __name__ == '__main__':
    # 单轮提问：与迁移前完全相同的问题"reportArgumentType": "none"    ← 删掉这行，reportArgumentType 的检查就回来了

    # 声明为 InputAgentState 期望的元素类型：list 不协变，传 list[HumanMessage]
    # 或 list[AnyMessage] 都不满足 list[AnyMessage | dict[str, Any]]。
    messages: list[AnyMessage | dict[str, Any]] = [HumanMessage(content='什么是EUV光刻机？')]
    resp = agent.invoke({'messages': messages}, config=config)

    # 输出最终答案（最后一条消息）
    print(resp['messages'][-1].content)
