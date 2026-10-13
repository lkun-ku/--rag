"""配置发现逻辑单测：自动源扫描（不依赖 .env 内容与外部服务）。"""

from rag_kb.utils.env_utils import discover_auto_sources


def test_discover_one_source_per_subdirectory(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("x", encoding="utf-8")
    (docs / "_topic.txt").write_text("产品文档与配置说明\n第二行忽略", encoding="utf-8")

    sources = discover_auto_sources(str(tmp_path))

    assert [s["name"] for s in sources] == ["docs"]
    source = sources[0]
    assert source["parser"] == "auto"
    assert source["collection"].startswith("kb_auto_")
    assert source["collection"].endswith("_bge_large_1024")
    assert source["topic"] == "产品文档与配置说明"
    assert source["enabled"] is True


def test_hidden_and_underscore_dirs_are_skipped(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "_private").mkdir()
    (tmp_path / "papers").mkdir()
    sources = discover_auto_sources(str(tmp_path))
    assert [s["name"] for s in sources] == ["papers"]


def test_collection_name_is_stable(tmp_path):
    (tmp_path / "docs").mkdir()
    first = discover_auto_sources(str(tmp_path))[0]["collection"]
    second = discover_auto_sources(str(tmp_path))[0]["collection"]
    assert first == second


def test_missing_directory_returns_empty():
    assert discover_auto_sources("definitely/not/exists") == []
