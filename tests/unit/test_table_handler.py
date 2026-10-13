"""表格解析的纯逻辑单测：rowspan 展开、整表/切片、超限切分。"""

from rag_kb.documents.table_handler import build_table_chunks, parse_table_html

ROWSPAN_HTML = (
    '<table><tr><td rowspan="2">甲组</td><td>B1</td><td>B2</td></tr>'
    "<tr><td>C1</td><td>C2</td></tr></table>"
)


def test_parse_plain_table():
    rows = parse_table_html("<table><tr><td>a</td><td>b</td></tr></table>")
    assert rows == [["a", "b"]]


def test_rowspan_is_expanded_to_every_row():
    rows = parse_table_html(ROWSPAN_HTML)
    assert rows[0] == ["甲组", "B1", "B2"]
    # 第二行的首列必须继承合并值，否则整行左移
    assert rows[1] == ["甲组", "C1", "C2"]


def test_build_table_chunks_keeps_merged_value():
    chunks = build_table_chunks(ROWSPAN_HTML, table_name="组别")
    assert chunks
    text = "\n".join(chunks)
    # 两行数据都要带上合并值（表头同样会包含，因此按行内容判断而不是计数）
    assert "甲组 | B1 | B2" in text
    assert "甲组 | C1 | C2" in text


def test_small_table_is_one_chunk():
    chunks = build_table_chunks("<table><tr><td>h</td></tr><tr><td>v</td></tr></table>")
    assert len(chunks) == 1
    assert "h" in chunks[0] and "v" in chunks[0]


def test_large_table_splits_by_rows_and_repeats_header():
    rows = "".join(f"<tr><td>r{i}</td><td>{i}</td></tr>" for i in range(30))
    chunks = build_table_chunks(f"<table><tr><td>name</td><td>value</td></tr>{rows}</table>", max_rows=10)
    assert len(chunks) > 1
    # 每个切片都要带表头，否则模型读不出列含义
    assert all("name" in c for c in chunks)


def test_non_positive_max_rows_does_not_hang():
    """max_rows<=0 曾被防御性修正为 1，不能出现死循环。"""
    chunks = build_table_chunks("<table><tr><td>h</td></tr><tr><td>v</td></tr></table>", max_rows=0)
    assert chunks
