"""Hydrogen addition preserves heavy atoms, bonds and residue annotations."""

import biotite.structure as struc
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.protonation import _remove_clashing_hydrogens, ensure_hydrogens


def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    atoms = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=True)
    atoms.res_name[:], atoms.chain_id[:], atoms.res_id[:] = "LIG", "A", 1
    atoms.set_annotation("pn_unit_iid", np.zeros(len(atoms), dtype=int))
    atoms.set_annotation("nhyd", np.array([atom.GetTotalNumHs() for atom in mol.GetAtoms()], dtype=np.int8))
    return atoms


@pytest.mark.parametrize("smiles", ["CCN", "NCC(=O)NCC(=O)O"])
def test_hydrogens_inherit_residue_annotations_and_keep_heavy_geometry(smiles):
    atoms = molecule(smiles)
    if smiles.startswith("NCC"):
        atoms.res_name[:] = "GLY"
        atoms.res_id[4:] = 2
        atoms.set_annotation("is_polymer", np.ones(len(atoms), dtype=bool))
    atoms.set_annotation("molecule_iid", np.full(len(atoms), 7))
    atoms.set_annotation("occupancy", np.full(len(atoms), 0.8))
    atoms.set_annotation("b_factor", np.full(len(atoms), 12.0))
    atoms.set_annotation("custom_atom_score", np.arange(len(atoms)))
    original = atoms.copy()

    result = ensure_hydrogens(atoms)
    heavy = result.element != "H"
    np.testing.assert_array_equal(result.coord[heavy], original.coord)
    np.testing.assert_array_equal(result.res_id[heavy], original.res_id)
    np.testing.assert_array_equal(result[heavy].bonds.as_array(), original.bonds.as_array())
    np.testing.assert_array_equal(atoms.coord, original.coord)
    np.testing.assert_array_equal(atoms.charge, original.charge)
    assert (~heavy).any()
    assert "custom_atom_score" not in result.get_annotation_categories()
    assert np.all(result.molecule_iid == 7)
    np.testing.assert_array_equal(result.occupancy[heavy], original.occupancy)
    assert np.all(result.occupancy[~heavy] == 1)
    assert np.all(np.isnan(result.b_factor[~heavy]))
    for h in np.flatnonzero(~heavy):
        (parent,), _ = result.bonds.get_bonds(int(h))
        assert result.element[parent] != "H"
        assert result.res_id[h] == result.res_id[parent]
    for residue in struc.residue_iter(result):
        assert len(set(residue.atom_name)) == len(residue)
        assert np.all(np.diff((residue.element == "H").astype(int)) >= 0)
    if smiles.startswith("NCC"):
        amide_n = np.flatnonzero((result.res_id == 2) & (result.element == "N"))[0]
        neighbors, _ = result.bonds.get_bonds(int(amide_n))
        assert (result.element[neighbors] == "H").sum() == 1
        assert result.charge[amide_n] == 0


@pytest.mark.parametrize("smiles", ["[Na+]", "[Cl-]"])
def test_ions_are_retained_without_hydrogens(smiles):
    atoms = molecule(smiles)
    result = ensure_hydrogens(atoms)
    assert len(result) == len(atoms) == 1
    np.testing.assert_array_equal(result.coord, atoms.coord)
    np.testing.assert_array_equal(result.charge, atoms.charge)


def test_disordered_atoms_keep_coordinates_and_implicit_hydrogens():
    atoms = molecule("CC")
    atoms.coord[:] = np.nan
    atoms.set_annotation("nhyd", np.array([3, 3], dtype=np.int8))
    result = ensure_hydrogens(atoms)
    assert len(result) == 2
    assert np.isnan(result.coord).all()
    np.testing.assert_array_equal(result.nhyd, [3, 3])


@pytest.mark.parametrize("missing", ["bonds", "charge"])
def test_missing_chemical_information_is_rejected(missing):
    atoms = molecule("CC")
    if missing == "bonds":
        atoms.bonds = None
    else:
        atoms.del_annotation("charge")
    with pytest.raises(ValueError, match=missing):
        ensure_hydrogens(atoms)


def test_stacks_require_explicit_model_selection():
    with pytest.raises(TypeError, match="single model"):
        ensure_hydrogens(struc.stack([molecule("CC")]))


@pytest.mark.parametrize("obstacle", ["C", "H"])
def test_pruning_a_clashing_proton_updates_charge_and_remaps_bonds(obstacle):
    atoms = struc.AtomArray(6)
    atoms.element = np.array(["N", "H", "H", "H", "H", obstacle])
    atoms.coord = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [1.5, 0, 0]])
    atoms.atom_name = np.array(["N", "H1", "H2", "H3", "H4", "obstacle"])
    atoms.set_annotation("charge", np.array([1, 0, 0, 0, 0, 0]))
    atoms.bonds = struc.BondList(6, np.array([[0, i, 1] for i in range(1, 5)]))
    original = atoms.copy()
    result = _remove_clashing_hydrogens(atoms)
    assert "H1" not in result.atom_name
    assert result.charge[0] == 0
    assert result.atom_name[:4].tolist() == ["N", "H2", "H3", "H4"]
    np.testing.assert_array_equal(result.bonds.as_array(), [[0, 1, 1], [0, 2, 1], [0, 3, 1]])
    np.testing.assert_array_equal(result.coord[:4], original.coord[[0, 2, 3, 4]])


def test_existing_explicit_hydrogens_are_rebuilt_once():
    first = ensure_hydrogens(molecule("CCO"))
    second = ensure_hydrogens(first)
    assert len(first) == len(second)
    np.testing.assert_array_equal(first.element, second.element)
    np.testing.assert_array_equal(first.charge, second.charge)
    np.testing.assert_array_equal(first.bonds.as_array(), second.bonds.as_array())
