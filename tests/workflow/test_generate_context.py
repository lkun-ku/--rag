"""生成阶段的上下文预算与拒答过滤（2026-10-28 修的两个缺陷）。

## 背景（实测）

同一问题「milvus有哪些内容」、同一批检索结果，只改喂给模型的数量：

    6 块 / 4,993 字符   -> 正常回答
    12 块 / 12,089 字符 -> 正常回答
    20 块 / 17,385 字符 -> 正常回答
    36 块 / 29,063 字符 -> **拒答**「知识库中未找到依据」

即「资料明明够、块数一多就拒答」。另外，同一会话里上一轮拒答会**锚定**后续回答
（第二轮 35/36 块判分通过仍拒答），除了最近 N 轮 messages，历史向量库那条通道也会注入旧轮次。

## 契约

    C1 `_truncate_context` 低于预算时原样返回（不复制、不改顺序）
    C2 超预算时按顺序截断，且累计字符不超过预算
    C3 至少保留 1 块（单块超预算也不返回空）
    C4 `_looks_like_refusal` 能识别「带前缀的拒答」（历史库里的块是标题+正文拼接）
    C5 `_format_history` 把「拒答轮」整对剔除，不给后续回答留锚点
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from rag_kb.graph_crag.generate_node import (  # noqa: E402
    _format_history,
    _looks_like_refusal,
    _truncate_context,
)

passed = 0
failures = []


def check(name, got, want):
    global passed
    if got == want:
        passed += 1
        print('  [OK ] %s' % name)
    else:
        failures.append(name)
        print('  [**FAIL**] %s  期望=%r 实际=%r' % (name, want, got))


def docs(*sizes):
    return [Document(page_content='x' * n, metadata={'i': i}) for i, n in enumerate(sizes)]


print('=' * 96)
print('生成上下文预算 / 拒答过滤')
print('=' * 96)

print('\n[C1/C2/C3] _truncate_context')
d = docs(100, 100, 100)
out = _truncate_context(d, budget=1000)
check('未超预算时原样返回', out is d, True)
out = _truncate_context(d, budget=250)
check('按顺序截断到 2 块', [x.metadata['i'] for x in out], [0, 1])
check('累计字符不超过预算', sum(len(x.page_content) for x in out), 200)
check('至少保留 1 块', len(_truncate_context(docs(999), budget=10)), 1)
check('空列表安全', _truncate_context([], budget=10), [])

print('\n[C4] _looks_like_refusal')
check('精确拒答', _looks_like_refusal('知识库中未找到依据。'), True)
check('带标题前缀的拒答', _looks_like_refusal('【更早轮次】知识库中未找到依据。'), True)
check('正常答案不算拒答', _looks_like_refusal('Milvus 是向量数据库 [1]。'), False)
check('空串不算拒答', _looks_like_refusal(''), False)

print('\n[C5] _format_history 剔除拒答轮')
msgs = [
    HumanMessage(content='你可以回答哪些内容'),
    AIMessage(content='知识库中未找到依据。'),
    HumanMessage(content='什么是 Milvus？'),
    AIMessage(content='Milvus 是向量数据库 [1]。'),
]
hist = _format_history(msgs)
check('拒答轮被剔除', '你可以回答哪些内容' not in hist, True)
check('正常轮保留', '什么是 Milvus' in hist, True)
check('没有残留拒答文案', '未找到依据' in hist, False)

print('\n' + '=' * 96)
print('结果：%d/%d 通过' % (passed, passed + len(failures)))
if failures:
    print('失败项：%s' % failures)
print('=' * 96)
sys.exit(0 if not failures else 1)
