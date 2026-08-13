import os
import socket
import time
from pathlib import Path

import pandas as pd
import pytest

from atomworks.io.parser import parse
from atomworks.io.utils.testing import assert_same_atom_array
from tests.io.conftest import TEST_DATA_IO, get_pdb_path

# A small structure that ships with the test data, so the cache tests below do not depend
# on a local PDB mirror.
STRUCTURE = TEST_DATA_IO / "2hhb.cif.gz"

TEST_CASES = [
    "4NDZ",  # 29K atoms, large enough to test caching without too much variance
]


@pytest.mark.xdist_group(name="test_caching")  # Ensure if running tests in parallel, they are run in the same group
@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_caching(pdb_id: str, tmp_path):
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
        assert_same_atom_array(
            normal_result["assemblies"][assembly_id], cached_result["assemblies"][assembly_id], annotations_to_compare
        )

    # Assert that the cached result is at least 4x faster than the normal result
    assert cached_elapsed_time < normal_elapsed_time / 4

    # Assert that the result with different arguments is similar to the normal elapsed time
    assert abs(different_args_elapsed_time - normal_elapsed_time) < normal_elapsed_time * 0.8


def _cache_files(cache_dir: Path) -> list[Path]:
    """Return the cache entries below `cache_dir`, ignoring temporary write files."""
    return [p for p in cache_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]


def test_cached_entry_is_gzip_compressed(tmp_path: Path) -> None:
    """The stored format is unchanged: entries are gzip compressed, as their name says.

    Cache files are named `.pkl.gz` and pandas infers the compression from that name. Writing
    through a temporary file would lose the inference, since the temporary name does not carry
    the suffix, so the compression has to be passed explicitly. This test pins the resulting
    format down.
    """
    parse(STRUCTURE, cache_dir=tmp_path, save_to_cache=True)
    (entry,) = _cache_files(tmp_path)
    assert entry.name.endswith(".pkl.gz")
    gzip_magic = bytes.fromhex("1f8b")
    assert entry.read_bytes()[:2] == gzip_magic, "cache entry is not gzip compressed"

    # ...and it is still readable, i.e. the format matches what the reader expects.
    result = parse(STRUCTURE, cache_dir=tmp_path, load_from_cache=True)
    assert result["asym_unit"].array_length() > 0


def test_cache_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted write leaves no cache entry behind.

    Without an atomic write, a process killed while serialising would leave a truncated file
    that later runs would treat as a valid cache entry. The write is made to fail part way
    through; afterwards the cache directory must contain neither an entry nor a leftover
    temporary file.
    """
    real_to_pickle = pd.to_pickle

    def failing_to_pickle(obj, path, *args, **kwargs):
        # Write a partial file first, so the test would fail if the target path were written
        # to directly instead of via a temporary file.
        Path(path).write_bytes(b"partial")
        raise KeyboardInterrupt("interrupted while writing the cache")

    monkeypatch.setattr(pd, "to_pickle", failing_to_pickle)
    with pytest.raises(KeyboardInterrupt):
        parse(STRUCTURE, cache_dir=tmp_path, save_to_cache=True)
    monkeypatch.setattr(pd, "to_pickle", real_to_pickle)

    assert not _cache_files(tmp_path), "an interrupted write left a cache entry behind"
    assert not list(tmp_path.rglob("*.tmp")), "an interrupted write left a temporary file behind"


def test_cache_write_tolerates_a_destination_created_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the finished file into place works even if the entry appeared meanwhile.

    An existing entry is normally not rewritten, but two workers can pass that check at the
    same time and both go on to write, so the move of the second one finds its destination
    occupied. `Path.replace` overwrites it; `Path.rename` would raise `FileExistsError` on
    Windows and leave the worker's temporary file behind. The race is reproduced here by
    creating the destination while the temporary file is being written.

    Note that the distinction between the two only shows on Windows: POSIX `rename` replaces
    an existing destination silently, so on Linux and macOS this test passes either way and
    covers only that the entry ends up complete and no temporary file is stranded.
    """
    real_to_pickle = pd.to_pickle
    # Rebuild the suffix the implementation appends, so the destination can be derived from
    # the temporary path without assuming anything about the host name.
    suffix = f".{socket.gethostname().replace(os.sep, '_')}.{os.getpid()}.tmp"

    def to_pickle_and_simulate_other_worker(obj, path, *args, **kwargs):
        real_to_pickle(obj, path, *args, **kwargs)
        tmp = Path(path)
        assert tmp.name.endswith(suffix), "cache write no longer uses the expected temporary name"
        tmp.with_name(tmp.name[: -len(suffix)]).write_bytes(b"written by another worker")

    monkeypatch.setattr(pd, "to_pickle", to_pickle_and_simulate_other_worker)
    parse(STRUCTURE, cache_dir=tmp_path, save_to_cache=True)
    monkeypatch.setattr(pd, "to_pickle", real_to_pickle)

    assert len(_cache_files(tmp_path)) == 1, "the concurrent write left more than one entry"
    assert not list(tmp_path.rglob("*.tmp")), "the move left a temporary file behind"

    # The entry is the one this worker wrote, not the placeholder, and it is readable.
    result = parse(STRUCTURE, cache_dir=tmp_path, load_from_cache=True)
    assert result["asym_unit"].array_length() > 0


if __name__ == "__main__":
    pytest.main([__file__])
