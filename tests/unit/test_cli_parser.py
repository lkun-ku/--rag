"""CLI 参数解析单测（只构造 parser，不执行任何子命令）。"""

from rag_kb.cli.main import build_parser


def test_no_subcommand_means_help():
    args = build_parser().parse_args([])
    assert args.command is None


def test_version_command_registered():
    args = build_parser().parse_args(["version"])
    assert args.command == "version"
    assert callable(args.func)


def test_ingest_accepts_multiple_sources_and_incremental_flag():
    args = build_parser().parse_args(["ingest", "--sources", "docs", "manuals", "--no-recreate", "--batch-size", "50"])
    assert args.sources == ["docs", "manuals"]
    assert args.no_recreate is True
    assert args.batch_size == 50


def test_serve_parses_numeric_port():
    args = build_parser().parse_args(["serve", "--port", "9000", "--workers", "4"])
    assert args.port == 9000
    assert args.workers == 4


def test_ask_requires_question():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask"])
