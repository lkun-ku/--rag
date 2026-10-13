"""引用解析与校验的纯逻辑单测（不依赖模型 / Milvus / LLM）。"""

from langchain_core.documents import Document

from rag_kb.utils.citation import format_context, parse_citations, validate_citations


def test_parse_citations_keeps_first_seen_order_and_dedups():
    assert parse_citations("先看[2]，再看[1]，最后又回到[2]。") == [2, 1]


def test_parse_citations_empty_answer():
    assert parse_citations("") == []


def test_format_context_numbers_follow_index():
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert format_context(docs) == "[1] A\n\n[2] B"


def test_validate_in_range_is_valid():
    result = validate_citations("答案[1][2]", n_docs=2)
    assert result["valid"] is True
    assert result["invalid"] == []
    assert result["cites"] == [1, 2]


def test_validate_out_of_range_is_invalid():
    result = validate_citations("答案[3]", n_docs=2)
    assert result["valid"] is False
    assert result["invalid"] == [3]


def test_validate_quoted_number_is_not_hallucination():
    """块正文里本来就有的 [28] 属于「复述原文编号」，不算幻觉引用。"""
    docs = [Document(page_content="……见参考文献[28]。")]
    result = validate_citations("如上所述[28]", n_docs=1, docs=docs)
    assert result["quoted"] == [28]
    assert result["invalid"] == []
    assert result["valid"] is True
