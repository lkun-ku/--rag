"""章节聚合的纯逻辑单测：同章节合并、顺序保留、关闭时原样返回。"""

from langchain_core.documents import Document

from rag_kb.tools import section_merge


def _doc(text: str, filename: str = "f.md", title: str = "同一节") -> Document:
    return Document(page_content=text, metadata={"filename": filename, "title": title})


def test_disabled_returns_same_list(monkeypatch):
    monkeypatch.setattr(section_merge, "SECTION_MERGE_ENABLED", False)
    docs = [_doc("a"), _doc("b")]
    assert section_merge.merge_section_blocks(docs) is docs


def test_merges_same_section_in_order(monkeypatch):
    monkeypatch.setattr(section_merge, "SECTION_MERGE_ENABLED", True)
    docs = [_doc("a"), _doc("b"), _doc("c")]
    out = section_merge.merge_section_blocks(docs, max_chars=1000)
    assert len(out) == 1
    assert out[0].page_content == "a\n\nb\n\nc"
    assert out[0].metadata["merged_from"] == 3


def test_different_sections_are_not_merged(monkeypatch):
    monkeypatch.setattr(section_merge, "SECTION_MERGE_ENABLED", True)
    docs = [_doc("a", title="一"), _doc("b", title="二")]
    out = section_merge.merge_section_blocks(docs, max_chars=1000)
    assert [d.page_content for d in out] == ["a", "b"]


def test_budget_splits_into_multiple_blocks(monkeypatch):
    monkeypatch.setattr(section_merge, "SECTION_MERGE_ENABLED", True)
    docs = [_doc("x" * 40), _doc("y" * 40), _doc("z" * 40)]
    out = section_merge.merge_section_blocks(docs, max_chars=50)
    assert len(out) >= 2
