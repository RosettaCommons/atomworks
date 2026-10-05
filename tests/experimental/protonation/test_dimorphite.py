"""Regression tests for Dimorphite molecule outputs."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from atomworks.experimental.protonation.dimorphite import protonate_at_ph


def test_protonation_removes_hydrogens_on_non_tetrahedral_centers() -> None:
    mol = Chem.MolFromSmiles("Cc1cn([C@H]2C[C@H](O)[C@@H](CO[P@SP2H](=O)O)O2)c(=O)[nH]c1=O")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    AllChem.Compute2DCoords(mol)

    product = protonate_at_ph(mol, 7.4)

    assert product is not None
    assert [atom.GetAtomicNum() for atom in product.GetAtoms()] == [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert [atom.GetAtomMapNum() for atom in product.GetAtoms()] == list(range(1, mol.GetNumAtoms() + 1))
    np.testing.assert_array_equal(product.GetConformer().GetPositions(), mol.GetConformer().GetPositions())
