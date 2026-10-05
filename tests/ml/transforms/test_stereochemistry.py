"""Atom and bond readouts from one RDKit 3D assignment."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.ccd import custom_ccd_residues
from atomworks.ml.transforms.rdkit_utils import get_chiral_centers, get_stereochemistry
from atomworks.ml.utils.annotator import ensure_annotations


def molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=9) == 0
    for atom in mol.GetAtoms():
        atom.SetProp("atom_name", f"A{atom.GetIdx()}")
    return mol


@pytest.mark.parametrize(
    "smiles,count",
    [
        ("C/C=C/C", 1),
        (r"C/C=C\C", 1),
        ("CC=CC", 1),
        ("CC=NC", 1),
        ("CC=C(C)C", 0),
        ("CC/C(CC)=C(/C)CC", 0),
        ("CC=O", 0),
        ("C1=CCCCC1", 0),
        ("FC=C=CCl", 0),
        ("C/C=C/C=C/C", 2),
        ("[2H]C([H])=C(F)Cl", 1),
        ("C/C=C([13CH3])C", 1),
    ],
)
@pytest.mark.parametrize("explicit_hydrogens", [True, False])
def test_assigned_double_bonds(smiles, count, explicit_hydrogens):
    mol = molecule(smiles)
    if not explicit_hydrogens:
        mol = Chem.RemoveHs(mol)
    result = get_stereochemistry(mol)
    assert len(result["double_bonds"]) == count
    for record in result["double_bonds"]:
        left, right, a, b = record["atom_idxs"]
        bond = mol.GetBondBetweenAtoms(left, right)
        assert list(bond.GetStereoAtoms()) == [a, b]
        assert record["stereo"] == bond.GetStereo()
        assert record["atom_names"] == tuple(mol.GetAtomWithIdx(i).GetProp("atom_name") for i in (left, right, a, b))


@pytest.mark.parametrize("right_y,expected", [(-1, Chem.BondStereo.STEREOE), (1, Chem.BondStereo.STEREOZ)])
def test_stereo_comes_from_coordinates_and_follows_renumbering(right_y, expected):
    mol = Chem.MolFromSmiles("CC=CC")
    conf = Chem.Conformer(4)
    for i, coord in enumerate([[-1, 1, 0], [0, 0, 0], [1, 0, 0], [2, right_y, 0]]):
        conf.SetAtomPosition(i, coord)
        mol.GetAtomWithIdx(i).SetProp("atom_name", f"A{i}")
    mol.AddConformer(conf)
    mol = Chem.RenumberAtoms(mol, [3, 1, 0, 2])
    [record] = get_stereochemistry(mol)["double_bonds"]
    assert record == {"atom_idxs": (1, 3, 2, 0), "atom_names": ("A1", "A2", "A0", "A3"), "stereo": expected}


def test_center_contract_and_tetrahedral_filter():
    mol = Chem.RemoveHs(molecule("N[C@@](C)(O)/C=C/F"))
    result = get_stereochemistry(mol)
    assert get_chiral_centers(mol) == result["chiral_centers"]
    [center] = result["chiral_centers"]
    assert center["chiral_center_idx"] == 1
    assert center["bonded_explicit_atom_idxs"] == [0, 2, 3, 4]
    assert center["chiral_type"].is_tetrahedral()
    assert center["chiral_center_atom_name"] == "A1"
    for i, coord in enumerate(mol.GetConformer().GetPositions() * [-1, 1, 1]):
        mol.GetConformer().SetAtomPosition(i, coord)
    reflected = get_stereochemistry(mol, tetrahedral_only=True)
    assert reflected["chiral_centers"][0]["chiral_type"] != center["chiral_type"]
    assert reflected["double_bonds"] == result["double_bonds"]


@pytest.mark.parametrize("smiles,expected", [("C/C=C/C", [False, True, True, False]), ("CC=C(C)C", [False] * 5)])
def test_stereogenic_bond_annotation_marks_only_endpoints(smiles, expected):
    atoms = atom_array_from_rdkit(molecule(smiles), remove_hydrogens=True)
    atoms.res_name[:] = "EZ1"
    with custom_ccd_residues({"EZ1": atoms}):
        order = np.roll(np.arange(len(atoms)), 1)
        reordered = atoms[order]
        ensure_annotations(reordered, "stereogenic_bond_atoms")
        np.testing.assert_array_equal(reordered.stereogenic_bond_atoms, np.array(expected)[order])
        endpoints = atoms[1:3]
        ensure_annotations(endpoints, "stereogenic_bond_atoms")
        np.testing.assert_array_equal(endpoints.stereogenic_bond_atoms, expected[1:3])
