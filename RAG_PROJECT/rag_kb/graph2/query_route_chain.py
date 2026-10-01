from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_kb.llm_models.all_llm import llm
from rag_kb.utils.env_utils import LLM_STRUCTURED_METHOD


# 查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）


# 数据模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
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
#   graph2 时永远拿不到知识库的答案，最终表现为「知识库中未找到依据」的拒答。
#
# 这个缺陷**评测层完全没覆盖** —— 因为 `eval/run_eval.py` 与 `refusal_audit.py`
# 都直接调 `graph2/generate_node2.generate`（为了控制变量，绕过路由）。
# 是 3.1 新增的 tracer 在跑完整工作流时暴露出来的。
#
# 与 `tools/retriever_tools.py` 里 `retriever_tool` 的描述属于**同一类问题**
# （那次修了工具描述，但漏了这条路由 prompt）。
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。

向量知识库收录以下主题的文档：
1. Milvus 向量数据库（架构与部署、索引类型与相似度指标、性能调优、运维监控、常见问题排查）
2. 文档解析与版面分析相关的研究论文（如 LayoutParser）
3. 产品手册与性能测试数据（接口延迟、吞吐量 QPS、错误率、召回率等）
4. 表格处理与文档切片的技术说明

判断规则：
- 问题涉及上述主题时，使用**向量知识库**（vectorstore）。
- 问题属于实时信息或与上述主题无关时（天气、新闻、股价、体育赛事、生活常识、
  与知识库无关的编程问题等），使用**网络搜索**（web_search）。
- 若问题提到 Milvus、向量检索、索引、召回率、文档解析、性能测试等关键词，
  一律视为知识库主题。"""
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
