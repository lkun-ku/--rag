from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.env_utils import KB_TOPIC_BLOCK, LLM_STRUCTURED_METHOD


# 查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）


# 数据模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的处理方式"""

    datasource: Literal["vectorstore", "web_search", "direct"] = Field(
        ...,
        description=(
            "vectorstore=查知识库（有资料、需溯源）；"
            "web_search=联网（实时信息）；"
            "direct=不检索、直接回答（闲聊/追问上文/写作翻译等通用请求）"
        ),
    )


# 带函数调用的LLM
structured_llm_router = llm.with_structured_output(RouteQuery, method=LLM_STRUCTURED_METHOD)

# 提示词模板
#
# [UPGRADE  / ROADMAP 3.1] 原 prompt 写的是「向量知识库包含与半导体材料，
# 芯片制造，光刻技术相关的文档」，而 datas/ 下的**实际语料是** Milvus 向量数据库文档、
# 版面分析论文、产品手册与性能测试数据。描述与实际语料不符的后果是**致命**的：
#
#   实测 7/7 个库内问题**全部**被路由到 web_search（100% 误判），真实用户走
#   graph_crag 时永远拿不到知识库的答案，最终表现为「知识库中未找到依据」的拒答。
#
# 这个缺陷**评测层完全没覆盖** —— 因为 `eval/run_eval.py` 与 `refusal_audit.py`
# 都直接调 `graph_crag/generate_node.generate`（为了控制变量，绕过路由）。
# 是 3.1 新增的 tracer 在跑完整工作流时暴露出来的。
#
# [2026-11] 同类问题**再次发生**：新增 frontend 语料后，prompt 里仍只写着 Milvus /
# 论文 / 手册，于是「前端有哪些内容」被判成「与知识库无关的编程问题」→ 走联网搜索，
# 1008 个已入库的块永远轮不到检索。
#
# 根治办法：主题清单不再硬编码，改为从 DOC_SOURCES 的 `topic` 字段动态生成
# （见 `env_utils.KB_TOPIC_BLOCK`）。新增语料只要填 topic，这里自动同步。
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。

向量知识库收录以下主题的文档：
%s

判断规则：
- 问题涉及上述主题时，使用**向量知识库**（vectorstore）。
- 问题属于实时信息或与上述主题完全无关时（天气、新闻、股价、体育赛事、生活常识等），
  使用**网络搜索**（web_search）。
- 若问题提到上述任一主题的关键词，一律视为知识库主题，**不要**因为「看起来像常识」
  或「问题太宽泛」就改走联网搜索。
- 想了解「知识库有哪些内容」时，也应查向量知识库，而非联网。
- 以下情况使用**直接回答**（direct）—— 不检索、不联网，模型自己答：
  * 寒暄、闲聊、感谢、告别（「你好」「谢谢」）；
  * **追问本次对话上文**（「我刚才问了什么」「上一个问题是什么」「上面那条再展开一点」），
    这类问题看对话历史就能回答，检索必然找不到；
  * 纯写作 / 改写 / 翻译 / 起名 / 写代码等**通用能力请求**；
  * 与上述主题无关、且不依赖实时信息的一般性概念提问（如「什么是递归」）。
- 拿不准时**优先 vectorstore** —— 宁可多查一次，也不要漏掉知识库里的依据。""" % KB_TOPIC_BLOCK
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统提示词
        ("human", "{question}"),  # 用户问题占位符
    ]
)

# 创建问题路由器链
question_router_chain = route_prompt | structured_llm_router


# 测试路由器
# print(  # 测试非技术问题（应路由到网络搜索）
#     question_router_chain.invoke(
#         {"question": "什么是EUV光刻技术?"}
#     )
# )
# print(  # 测试技术问题（应路由到向量数据库）
#     question_router_chain.invoke({"question": "今天，长沙的天气怎么样?"})
# )
