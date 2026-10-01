"""HTTP 服务层（FastAPI）。

有 `__init__.py` 才是标准包 —— setuptools 的 `find` 只识别这种形式，
否则 `pip install -e .` 后 `import api.main` 会失败。
"""
