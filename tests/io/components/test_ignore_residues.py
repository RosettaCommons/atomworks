import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import CRYSTALLIZATION_AIDS
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array import remove_components
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from tests.io.conftest import get_pdb_path

CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX", "5XAG"]


@pytest.mark.parametrize("pdbid", CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST)
def test_remove_crystallization_aids(pdbid: str):
    """Remove aid-only molecules while preserving atoms and bonds of molecules containing non-aids."""
    original = parse(get_pdb_path(pdbid), config=ParseConfig(remove_ccds=[]))["asym_unit"]
    is_aid = np.isin(original.res_name, CRYSTALLIZATION_AIDS)
    assert is_aid.any(), "No crystallization aids found when not excluding."

    # Use unfiltered covalent molecules so deleting every aid cannot satisfy the test.
    bonds = original.bonds.as_array()
    covalent = struc.BondList(original.array_length(), bonds[bonds[:, 2] != struc.BondType.COORDINATION])
    keep = np.zeros(original.array_length(), dtype=bool)
    for molecule in struc.get_molecule_indices(covalent):
        keep[molecule] = not is_aid[molecule].all()

    result = parse(get_pdb_path(pdbid), config=ParseConfig(remove_ccds=CRYSTALLIZATION_AIDS))
    assert_same_atom_array_or_stack(
        result["asym_unit"],
        original[:, keep],
        annotations_to_compare=["chain_id", "res_id", "res_name", "atom_name", "element", "charge"],
    )
    assert set(result["asym_unit"].chain_id) == set(result["chain_info"])


@pytest.mark.parametrize("bond_type", [struc.BondType.SINGLE, struc.BondType.COORDINATION])
@pytest.mark.parametrize("stack", [False, True])
def test_remove_aids_by_covalent_connectivity(bond_type, stack):
    """Preserve whole, transitively linked residues; remove free and aid-only molecules, ignoring coordination."""
    atoms = struc.AtomArray(12)
    atoms.res_name = np.repeat(["IMD", "GOL", "EDO", "GOL", "EDO", "GOL"], 2)
    atoms.res_id = np.repeat(np.arange(6), 2)
    # Only inter-residue bonds: IMD–GOL–EDO, GOL–EDO, and free GOL.
    atoms.bonds = struc.BondList(12, np.array([[0, 2, bond_type], [3, 4, bond_type], [6, 8, bond_type]]))
    if stack:
        atoms = struc.stack([atoms, atoms])

    filtered = remove_components(atoms, remove_ccds=("GOL", "EDO"))
    expected = ["IMD", "GOL", "EDO"] if bond_type == struc.BondType.SINGLE else ["IMD"]
    assert filtered.res_name.tolist() == np.repeat(expected, 2).tolist()
    assert atoms.array_length() == 12
    assert atoms.bonds.get_bond_count() == 3
