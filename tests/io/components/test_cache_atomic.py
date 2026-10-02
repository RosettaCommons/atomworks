"""Regression coverage for interrupted parser cache writes."""

from pathlib import Path

import pytest

import atomworks.io.parser as parser
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from tests.conftest import TEST_DATA_DIR


def test_cache_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted write leaves neither a cache entry nor a temporary file behind."""

    def failing_to_pickle(obj, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise KeyboardInterrupt("interrupted while writing the cache")

    monkeypatch.setattr(parser.pd, "to_pickle", failing_to_pickle)
    config = ParseConfig(cache_dir=str(tmp_path), save_to_cache=True, load_from_cache=False)
    with pytest.raises(KeyboardInterrupt, match="interrupted while writing the cache"):
        parse(TEST_DATA_DIR / "io" / "2hhb.cif.gz", config=config)

    assert not any(
        path.is_file() for path in tmp_path.rglob("*")
    ), "an interrupted write left a cache or temporary file"
