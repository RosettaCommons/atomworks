import numpy as np
import pytest
from biotite.structure import BondType

from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.io_utils import read_any
from tests.conftest import TEST_DATA_DIR


def test_structure_with_non_ccd_ligand(tmp_path):
    """Authored ligand bonds survive absent altlocs; incomplete bond tables raise instead of falling back."""
    source = TEST_DATA_DIR / "io" / "9cox_with_unknown_ccd.cif"
    cif = read_any(source)
    del cif.block["atom_site"]["label_alt_id"]
    path = tmp_path / "ligand.cif"
    cif.write(path)
    config = ParseConfig.from_preset("minimal", altloc="random_clash_aware")
    atoms = parse(path, config=config)["asym_unit"][0]
    ligand = atoms[atoms.res_name == "UNKNOWN_CCD"]
    assert len(ligand) == 43
    i, j = (np.flatnonzero(ligand.atom_name == name).item() for name in ("C26", "C25"))
    neighbors, orders = ligand.bonds.get_bonds(i)
    assert orders[neighbors == j].item() == BondType.AROMATIC_DOUBLE

    for column in ("value_order", "pdbx_aromatic_flag"):
        malformed = read_any(source)
        del malformed.block["chem_comp_bond"][column]
        for category in (None, "chem_comp_atom"):
            if category is not None:
                del malformed.block[category]
            malformed.write(path)
            with pytest.raises(KeyError, match=column):
                parse(path, config=config)
