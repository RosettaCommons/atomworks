import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    {"pdb_id": "4oji", "pH": [8.0, 8.0]},  # _exptl_crystal_grow.pdbx_pH_range   pH8.0
    {"pdb_id": "5hs6", "pH": [7.0, 8.0]},  # _exptl_crystal_grow.pdbx_pH_range   7.0-8.0
    {"pdb_id": "5a93", "pH": [5.9, 6.1]},  # 1 ? ? ? 5.9 ? ?  2 ? ? ? 6.1 ? ? in a loop
    {"pdb_id": "4o8v", "pH": [5.5, 5.5]},  # _exptl_crystal_grow.pdbx_details  '... 0.1 M Bis-Tris pH 5.5, 25% ...'
    {"pdb_id": "1xj9", "pH": None},  # Test error handling (invalid pH range)
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_get_ph(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=False,
        remove_ccds=[],
    )
    assert result["metadata"]["crystallization_details"]["pH"] == test_case["pH"]


if __name__ == "__main__":
    pytest.main([__file__])
