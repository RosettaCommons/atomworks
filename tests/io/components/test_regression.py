"""Regression tests for complex cases to ensure consistent behavior."""

import pickle
from pathlib import Path  # noqa: F401

import numpy as np
import pytest

from atomworks.constants import CRYSTALLIZATION_AIDS
from atomworks.io.parser import parse
from atomworks.io.transforms import atom_array as ta
from atomworks.io.utils.io_utils import to_cif_file  # noqa: F401
from atomworks.io.utils.testing import assert_same_atom_array
from tests.io.conftest import TEST_DATA_IO, get_pdb_path

TEST_CASES = [
    "6mub",  # Symmetry center clash
    "1j8z",  # Contains misordered atoms in a residue
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "1twr",  # Residue name not in biotite's CCD
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids; also contains various unusual ligands and NCAA's
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_regression_against_stored_result(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        remove_ccds=CRYSTALLIZATION_AIDS,
        fix_ligands_at_symmetry_centers=True,
        build_assembly="all",
        fix_arginines=True,
        convert_mse_to_met=True,
        hydrogen_policy="keep",
        model=None,
    )
    assert result is not None  # Check if processing runs through

    regression_dir = TEST_DATA_IO / "regression_tests"
    regression_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = regression_dir / f"{pdb_id}.pkl"

    # Uncomment the following lines to create the pickle file
    # with pickle_path.open("wb") as f:
    #     import atomworks.io

    #     result["atomworks.version"] = atomworks.__version__
    #     pickle.dump(result, f)

    with pickle_path.open("rb") as f:
        expected_result = pickle.load(f)

    expected_result["asym_unit"] = ta.remove_hydrogens(expected_result["asym_unit"])
    result["asym_unit"] = ta.remove_hydrogens(result["asym_unit"])

    # Save output into test_output
    # to_cif_file(
    #     result["asym_unit"][0],
    #     f"{Path(__file__).parents[1]}/test_outputs/{pdb_id}_new.cif",
    #     id=f"{pdb_id}_new",
    #     date=True,
    #     time=True,
    # )
    # to_cif_file(
    #     expected_result["asym_unit"][0],
    #     f"{Path(__file__).parents[1]}/test_outputs/{pdb_id}_old.cif",
    #     id=f"{pdb_id}_old",
    #     date=True,
    #     time=True,
    # )

    # ## FOR DEBUGGING REGRESSION TESTS UNCOMMENT:
    from atomworks.common import sum_string_arrays

    def get_atom_identifiers(atom_array):
        return sum_string_arrays(
            atom_array.chain_id, "-", atom_array.res_name, "-", atom_array.res_id.astype(str), "-", atom_array.atom_name
        )

    # a = result["asym_unit"][0]
    # b = expected_result["asym_unit"][0]
    a_id = get_atom_identifiers(result["asym_unit"][0])
    b_id = get_atom_identifiers(expected_result["asym_unit"][0])
    # In a but not in b
    print("in new but not in old:", np.setdiff1d(a_id, b_id))
    # In b but not in a
    print("in old but not in new:", np.setdiff1d(b_id, a_id))
    ###

    # Check the asymmetric unit...
    assert_same_atom_array(
        result["asym_unit"],
        expected_result["asym_unit"],
        annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name", "charge"],
        compare_coords=True,
        compare_bonds=True,
    )

    # ... the assemblies
    for assembly_id in result["assemblies"]:
        result["assemblies"][assembly_id] = ta.remove_hydrogens(result["assemblies"][assembly_id])
        expected_result["assemblies"][assembly_id] = ta.remove_hydrogens(expected_result["assemblies"][assembly_id])
        assert_same_atom_array(
            result["assemblies"][assembly_id],
            expected_result["assemblies"][assembly_id],
            annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name", "charge"],
            compare_coords=True,
            compare_bonds=True,
        )

    # ... the ligand of interest information
    assert result["ligand_info"] == expected_result["ligand_info"]

    # ... the chain information
    assert set(result["chain_info"].keys()) == set(expected_result["chain_info"].keys())
    for chain in result["chain_info"]:
        got = result["chain_info"][chain]["chain_type"]
        expected = expected_result["chain_info"][chain]["chain_type"]
        assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["res_name"]
        expected = expected_result["chain_info"][chain]["res_name"]
        assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["res_id"]
        expected = expected_result["chain_info"][chain]["res_id"]
        assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["is_polymer"]
        expected = expected_result["chain_info"][chain]["is_polymer"]
        assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

    # ... the extra information
    assert result["extra_info"] == expected_result["extra_info"]

    # ... the metadata
    for key in expected_result.get("metadata", {}):
        assert key in result["metadata"], f"Missing metadata key: {key}"
        assert result["metadata"][key] == expected_result["metadata"][key]


if __name__ == "__main__":
    pytest.main([__file__])
