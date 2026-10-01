"""End-to-end test for PDB preprocessing pipeline.

Tests the full workflow: structure parsing -> PN unit records -> interfaces.
"""

from typing import Any

import pytest

from atomworks.ml.preprocessing.preprocess import preprocess
from atomworks.ml.preprocessing.records import Assembly, Interface, PNUnit
from atomworks.ml.utils.testing import get_pdb_mirror_path

PDB_PROCESSING_TEST_CASES = [
    {
        # Protein-ligand with LOI
        "pdb_id": "6wjc",
        "require_matching_interfaces": True,  # Whether we are enumerating all interfaces in `expected_interfaces` or only relevant ones
        "num_pn_units": 8,
        "expected_interfaces": [
            # Protein-protein
            {"pn_unit_1": "A_1", "pn_unit_2": "B_1"},
            # Protein-ligand
            {"pn_unit_1": "A_1", "pn_unit_2": "C_1", "involves_covalent_modification": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "D_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "E_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "F_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "G_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "H_1", "involves_loi": True},
            # Ligand-ligand
            {"pn_unit_1": "F_1", "pn_unit_2": "G_1", "involves_loi": True},
        ],
    },
    {
        # Covalent modifications
        "pdb_id": "1ivo",
        "require_matching_interfaces": False,
        "num_pn_units": 13,
        "expected_interfaces": [
            {"pn_unit_1": "A_1", "pn_unit_2": "B_1"},
            {"pn_unit_1": "B_1", "pn_unit_2": "K_1", "involves_covalent_modification": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "J_1", "involves_covalent_modification": True},
        ],
    },
    {
        # Homomeric symmetry
        "pdb_id": "1a8o",
        "require_matching_interfaces": True,
        "num_pn_units": 2,
        "expected_interfaces": [
            {"pn_unit_1": "A_1", "pn_unit_2": "A_2"},
        ],
    },
    # TODO: Add additional test cases, including clashes, DNA, RNA, multi-chain ligands, etc.
]

# Cache for preprocessing results to avoid re-running for each test
_preprocessing_cache: dict[str, tuple[list[Assembly], list[PNUnit], list[Interface]]] = {}


def _get_cached_results(pdb_id: str) -> tuple[list[Assembly], list[PNUnit], list[Interface]]:
    """Get preprocessing results, cached to avoid re-running."""
    if pdb_id not in _preprocessing_cache:
        path = get_pdb_mirror_path(pdb_id)
        _preprocessing_cache[pdb_id] = preprocess(path)
    return _preprocessing_cache[pdb_id]


@pytest.mark.parametrize("test_case", PDB_PROCESSING_TEST_CASES)
def test_pn_units_counts(test_case: dict[str, Any]):
    """Test that each PDB has correct number of PN units."""
    pdb_id = test_case["pdb_id"]
    expected_num = test_case["num_pn_units"]

    _, pn_units, _ = _get_cached_results(pdb_id)

    assert len(pn_units) == expected_num, f"Number of PN units for {pdb_id} is incorrect"


@pytest.mark.parametrize("test_case", PDB_PROCESSING_TEST_CASES)
def test_interfaces(test_case: dict[str, Any]):
    """Test interface generation and properties."""
    pdb_id = test_case["pdb_id"]
    expected_interfaces = test_case["expected_interfaces"]

    _, _, interfaces = _get_cached_results(pdb_id)

    if test_case["require_matching_interfaces"]:
        assert len(interfaces) == len(expected_interfaces), f"{pdb_id}: Number of interfaces is incorrect"

    for expected in expected_interfaces:
        pn_unit_1 = expected["pn_unit_1"]
        pn_unit_2 = expected["pn_unit_2"]
        involves_loi = expected.get("involves_loi", False)
        involves_covalent_modification = expected.get("involves_covalent_modification", False)
        involves_metal = expected.get("involves_metal", False)

        # Find matching interface by direct property access
        matching = [i for i in interfaces if i.pn_unit_1_iid == pn_unit_1 and i.pn_unit_2_iid == pn_unit_2]

        assert len(matching) == 1, f"{pdb_id}: Interface {pn_unit_1}-{pn_unit_2} not found or multiple found"

        interface = matching[0]
        assert (
            interface.involves_loi == involves_loi
        ), f"{pdb_id}: LOI involvement for interface {pn_unit_1}-{pn_unit_2} is incorrect"
        assert (
            interface.involves_covalent_modification == involves_covalent_modification
        ), f"{pdb_id}: Covalent modification for interface {pn_unit_1}-{pn_unit_2} is incorrect"
        assert (
            interface.involves_metal == involves_metal
        ), f"{pdb_id}: Metal involvement for interface {pn_unit_1}-{pn_unit_2} is incorrect"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
