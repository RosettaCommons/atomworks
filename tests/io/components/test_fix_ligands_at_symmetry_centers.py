import numpy as np
import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

LIGAND_AT_SYMMETRY_CENTER_TEST_CASES = [
    {
        "pdbid": "7mub",  # metal ion at symmetry center
        "chain_iids_to_include": ["D_1", "E_1"],
        "chain_iids_to_exclude": ["D_2", "D_3", "D_4", "E_2", "E_3", "E_4"],
    },
    {
        "pdbid": "1xan",  # symmetric ligand at symmetry center
        "chain_iids_to_include": ["C_1"],
        "chain_iids_to_exclude": ["C_2"],
    },
]


@pytest.mark.parametrize("test_case", LIGAND_AT_SYMMETRY_CENTER_TEST_CASES)
def test_patch_symmetry_centers(test_case: dict):
    # unpack test case
    pdbid = test_case["pdbid"]

    # Parse the file
    filename = get_pdb_path(pdbid)
    out = parse(
        filename, config=ParseConfig(build_assembly="first", remove_waters=True, fix_ligands_at_symmetry_centers=True)
    )
    chain_iids = np.unique(out["assemblies"]["1"][0].chain_iid.astype(str)).tolist()

    # Ensure that we excluded clashing chains
    assert set(chain_iids).intersection(set(test_case["chain_iids_to_exclude"])) == set()

    # Ensure that we included the correct chains
    assert set(chain_iids).intersection(set(test_case["chain_iids_to_include"])) == set(
        test_case["chain_iids_to_include"]
    )


if __name__ == "__main__":
    pytest.main([__file__])
