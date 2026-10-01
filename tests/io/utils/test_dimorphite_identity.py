"""Regression tests for preserving RDKit identity during protonation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from atomworks.external.dimorphite_dl import dimorphite_dl as engine
from atomworks.external.dimorphite_dl import protonate_mol_variants
from atomworks.external.dimorphite_dl.dimorphite_dl import run_with_mol_list
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.protonation import ensure_hydrogens


def test_protonation_preserves_identity_order_and_ownership():
    """Preserve molecular identity and deprotonate phosphate monoesters at physiological pH."""
    mol = Chem.MolFromSmiles("CC(=O)[O-].CC[NH3+].CCN=[N+]=[N-]")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 17)
        atom.SetProp("source_name", str(atom.GetIdx()))
    mol.SetProp("source_label", "source")
    AllChem.Compute2DCoords(mol)
    source_smiles = Chem.MolToSmiles(mol)
    source_positions = mol.GetConformer().GetPositions().copy()

    product = run_with_mol_list([mol], min_ph=2.0, max_ph=2.0, pka_precision=0.1)[0]

    assert Chem.MolToSmiles(mol) == source_smiles
    assert [atom.GetAtomMapNum() for atom in product.GetAtoms()] == list(range(17, 17 + mol.GetNumAtoms()))
    assert [atom.GetProp("source_name") for atom in product.GetAtoms()] == [str(i) for i in range(mol.GetNumAtoms())]
    assert product.GetProp("source_label") == "source"
    np.testing.assert_array_equal(product.GetConformer().GetPositions(), source_positions)

    variants = protonate_mol_variants(Chem.MolFromSmiles("CC(=O)O"), min_ph=0.0, max_ph=14.0)
    assert [Chem.GetFormalCharge(variant) for variant in variants] == [-1, 0]
    variants[0].GetAtomWithIdx(0).SetAtomMapNum(999)
    assert variants[1].GetAtomWithIdx(0).GetAtomMapNum() != 999

    phosphate = Chem.MolFromSmiles("COP(=O)(O)O")
    AllChem.Compute2DCoords(phosphate)
    atoms = atom_array_from_rdkit(phosphate, set_coord_if_available=True, remove_hydrogens=True)
    atoms.res_name[:], atoms.chain_id[:], atoms.res_id[:] = "LIG", "A", 1
    atoms.set_annotation("pn_unit_iid", np.zeros(len(atoms), dtype=int))
    protonated = ensure_hydrogens(atoms, ph=7.0, silence_rdkit_warnings=True)
    assert sorted(protonated.charge[protonated.element == "O"]) == [-1, -1, 0, 0]
    assert all(set(protonated.element[bond[:2]]) != {"O", "H"} for bond in protonated.bonds.as_array())


def test_protonation_handles_hydrogens_azides_and_invalid_products(monkeypatch):
    """Handle explicit/implicit H and legacy azides; fall back after an invalid product."""
    explicit_h = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    explicit_h.AddConformer(Chem.Conformer(explicit_h.GetNumAtoms()))
    product = protonate_mol_variants(explicit_h, min_ph=7.0, max_ph=7.0)[0]
    assert product.GetNumAtoms() == product.GetConformer().GetNumAtoms() == 3

    amine = protonate_mol_variants(Chem.MolFromSmiles("CCN"), min_ph=2.0, max_ph=2.0)[0]
    assert Chem.AddHs(amine).GetAtomWithIdx(2).GetDegree() == 4

    for ph, expected in [(2.0, "CN=[N+]=N"), (12.0, "CN=[N+]=[N-]")]:
        azide = Chem.MolFromSmiles("CNN#N", sanitize=False)
        azide.UpdatePropertyCache(strict=False)
        product = protonate_mol_variants(azide, min_ph=ph, max_ph=ph, pka_precision=0.1)[0]
        assert Chem.MolToSmiles(product) == expected

    mol = Chem.MolFromSmiles("OC(=O)CCN")
    AllChem.Compute2DCoords(mol)
    invalid = Chem.MolFromSmiles("[CH5]", sanitize=False)
    monkeypatch.setattr(engine.ProtSubstructFuncs, "protonate_site", lambda *args: [invalid])
    fallback = engine.protonate_mol_variants(mol, min_ph=7.4, max_ph=7.4, pka_precision=0.1)[0]
    np.testing.assert_array_equal(fallback.GetConformer().GetPositions(), mol.GetConformer().GetPositions())
