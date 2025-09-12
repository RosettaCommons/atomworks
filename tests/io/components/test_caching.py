import time

import pytest

from atomworks.io.parser import parse
from atomworks.io.utils.testing import assert_same_atom_array
from tests.io.conftest import get_pdb_path

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

    # Assert that the cached result is at least 3x faster than the normal result
    assert cached_elapsed_time < normal_elapsed_time / 3

    # Assert that the result with different arguments is similar to the normal elapsed time
    assert abs(different_args_elapsed_time - normal_elapsed_time) < normal_elapsed_time * 0.5


if __name__ == "__main__":
    pytest.main([__file__])
