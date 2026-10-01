import io
import os
import tempfile
import time
from unittest.mock import patch

import pytest

import atomworks.io.parser as parser
from atomworks.io.config import ParseConfig
from atomworks.io.parser import _build_cache_file_path, parse
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from tests.conftest import TEST_DATA_DIR
from tests.io.conftest import get_pdb_path


# Skip this test when running with pytest-xdist (multiprocessing)
# Timing-based tests are unreliable with parallel execution due to variable system load
def pytest_configure(config):
    """Register custom marker for xdist detection."""
    config.addinivalue_line("markers", "no_xdist: skip test when running with pytest-xdist")


def is_xdist_worker(request):
    """Check if test is running in a pytest-xdist worker process."""
    return hasattr(request.config, "workerinput")


TEST_CASES = [
    "4NDZ",  # 29K atoms, large enough to test caching without too much variance
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_caching(pdb_id: str, tmp_path, request):
    # Skip when running with pytest-xdist (multiprocessing)
    # Timing assertions are unreliable with parallel execution
    if is_xdist_worker(request):
        pytest.skip("Caching test disabled in multiprocessing mode (timing-based assertions are unreliable)")

    path = get_pdb_path(pdb_id)

    # First, we load normally, tracking how long it takes
    def normal_parse():
        return parse(
            # Caching arguments
            load_from_cache=False,
            save_to_cache=False,
            cache_dir=None,
            # Standard arguments
            filename=path,
            build_assembly="all",
        )

    # Warmup
    _ = normal_parse()

    start_time = time.time()
    normal_result = normal_parse()
    normal_elapsed_time = time.time() - start_time
    assert normal_result is not None  # Check if processing runs through

    # Load from CIF, saving to the cache
    _ = parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=True,
        cache_dir=tmp_path,
        # Standard arguments
        filename=path,
        build_assembly="all",
    )

    # Load from the cache, and keep track of how long it takes
    def cached_parse():
        return parse(
            # Caching arguments
            load_from_cache=True,
            save_to_cache=False,
            cache_dir=tmp_path,
            # Standard arguments
            filename=path,
            build_assembly="all",
        )

    start_time = time.time()
    cached_result = cached_parse()
    cached_elapsed_time = time.time() - start_time

    # Check that metadata fields are present and correct
    assert "metadata" in cached_result
    assert "parse_arguments" in cached_result["metadata"]
    assert "atomworks.version" in cached_result["metadata"]
    assert isinstance(cached_result["metadata"]["atomworks.version"], str)

    # Load with different parsing arguments
    def different_args_parse():
        return parse(
            # Caching arguments
            load_from_cache=True,
            save_to_cache=False,
            cache_dir=tmp_path,
            # Standard arguments
            filename=path,
            build_assembly="all",
            fix_ligands_at_symmetry_centers=False,
        )

    start_time = time.time()
    _ = different_args_parse()
    different_args_elapsed_time = time.time() - start_time

    # Assert that the assembly data is the same
    annotations_to_compare = ["chain_id", "res_name", "res_id", "atom_name", "chain_iid", "pn_unit_id", "pn_unit_iid"]
    for assembly_id in normal_result["assemblies"]:
        assert_same_atom_array_or_stack(
            normal_result["assemblies"][assembly_id], cached_result["assemblies"][assembly_id], annotations_to_compare
        )

    # Assert that the cached result is at least 4x faster than the normal result
    assert cached_elapsed_time < normal_elapsed_time / 4

    # Assert that the result with different arguments is similar to the normal elapsed time
    assert abs(different_args_elapsed_time - normal_elapsed_time) < normal_elapsed_time * 0.8


def test_cache_key_disambiguates_files_with_same_stem(tmp_path):
    """Two distinct files with the same stem must map to distinct cache files."""
    cfg = ParseConfig(cache_dir=str(tmp_path), save_to_cache=True, load_from_cache=True)

    file_a = tmp_path / "source_a" / "system.cif"
    file_b = tmp_path / "source_b" / "system.cif"
    file_c = tmp_path / "source_c" / "protein.cif.gz"

    p1 = _build_cache_file_path(tmp_path, file_a, cfg)
    p2 = _build_cache_file_path(tmp_path, file_b, cfg)
    p1_again = _build_cache_file_path(tmp_path, file_a, cfg)

    assert p1 != p2, "same stem but different paths must produce different cache keys"
    assert p1 == p1_again, "same path must hash to the same cache key"

    # Unique-stem case must also remain idempotent.
    q1 = _build_cache_file_path(tmp_path, file_c, cfg)
    q2 = _build_cache_file_path(tmp_path, file_c, cfg)
    assert q1 == q2


def test_cache_dir_defaults_to_tempdir_when_enabled():
    """Enabling caching without a path defaults cache_dir to the system tempdir; off otherwise."""
    assert ParseConfig().cache_dir is None
    cfg = ParseConfig(save_to_cache=True, load_from_cache=True)
    assert cfg.cache_dir == os.path.join(tempfile.gettempdir(), "atomworks_parse_cache")


@pytest.mark.parametrize(
    "filename,buffer_type", [("101m_arginine_nh1nh2_swapped.cif", io.StringIO), ("6lyz.bcif", io.BytesIO)]
)
def test_buffer_cache(filename, buffer_type, tmp_path):
    path = TEST_DATA_DIR / "io" / filename
    content = path.read_text() if buffer_type is io.StringIO else path.read_bytes()
    config = ParseConfig.from_preset("minimal", cache_dir=str(tmp_path), save_to_cache=True, load_from_cache=False)
    source = buffer_type(content)
    original = parse(source, config=config)
    position = source.tell()
    cache_path = next(tmp_path.rglob("*.pkl.zst"))
    assert cache_path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    cached_bytes = cache_path.read_bytes()
    with (
        patch.object(parser.os, "replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        parse(buffer_type(content), config=config)
    assert cache_path.read_bytes() == cached_bytes
    assert not list(tmp_path.rglob("*.tmp.*"))
    parser.pd.to_pickle(parser.pd.read_pickle(cache_path), cache_path.with_suffix(".gz"))
    config = config.replace(save_to_cache=False, load_from_cache=True)
    with patch.object(parser, "load_cif", side_effect=AssertionError("cache miss")):
        source = buffer_type(content)
        cached = parse(source, config=config)
        assert source.tell() == position
        assert cached.keys() == original.keys()
        assert cached["metadata"] == original["metadata"]
        assert_same_atom_array_or_stack(cached["asym_unit"], original["asym_unit"])
        assert cached["assemblies"].keys() == original["assemblies"].keys()
        for name in original["assemblies"]:
            assert_same_atom_array_or_stack(cached["assemblies"][name], original["assemblies"][name])
        cached["asym_unit"].coord[:] = 0
        fresh = parse(buffer_type(content), config=config)
        assert_same_atom_array_or_stack(fresh["asym_unit"], original["asym_unit"])
        cache_path.unlink()
        with pytest.raises(AssertionError, match="cache miss"):
            parse(buffer_type(content), config=config)


if __name__ == "__main__":
    pytest.main([__file__])
