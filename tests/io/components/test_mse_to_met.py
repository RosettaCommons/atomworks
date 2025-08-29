import os

import numpy as np
import pytest

from atomworks.common import not_isin
from atomworks.constants import CCD_MIRROR_PATH, HYDROGEN_LIKE_SYMBOLS
from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import mse_to_met
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.io.utils.testing import assert_same_atom_array
from tests.io.conftest import get_pdb_path

TEST_CASES = ["1aqc"]


@pytest.mark.parametrize("ccd_mirror_path", [CCD_MIRROR_PATH, None])
def test_mse_to_met_residue(ccd_mirror_path: os.PathLike | None):
    # Test with local CCD data
    mse = atom_array_from_ccd_code("MSE", ccd_mirror_path=ccd_mirror_path)
    met = atom_array_from_ccd_code("MET", ccd_mirror_path=ccd_mirror_path)
    # Set coordinates to `nan` to avoid comparing coordinates
    mse.coord[:] = np.nan
    met.coord[:] = np.nan
    is_heavy = lambda x: not_isin(x.element, HYDROGEN_LIKE_SYMBOLS)
    mse_converted = mse_to_met(mse)
    assert_same_atom_array(
        mse_converted[is_heavy(mse_converted)],
        met[is_heavy(met)],
        annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name", "element"],
        compare_bonds=False,
    )


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_mse_to_met_pdb(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        build_assembly="all",
        fix_ligands_at_symmetry_centers=True,
        fix_arginines=True,
        convert_mse_to_met=True,
    )
    assert result is not None  # Check if processing runs through
    assert "MSE" not in result["asym_unit"].res_name

    result_unconverted = parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        build_assembly="all",
        fix_ligands_at_symmetry_centers=True,
        fix_arginines=True,
        convert_mse_to_met=False,
    )
    assert "MSE" in result_unconverted["asym_unit"].res_name
