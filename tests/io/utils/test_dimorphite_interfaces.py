"""Exercise the vendored engine's molecule and SMILES interfaces."""

import sys

import pytest
from rdkit import Chem

from atomworks.external.dimorphite_dl import dimorphite_dl as engine


@pytest.mark.parametrize("mode", ["list", "file", "log"])
def test_smiles_input_preserves_labels_and_skips_invalid_records(mode, tmp_path, monkeypatch, caplog):
    source = tmp_path / "input.smi"
    source.write_text("\ninvalid broken\nCC(=O)[O-] acid\nCC[NH3+] amine\nC[C@H](O)C carbon\n")
    output = tmp_path / "output.smi"
    monkeypatch.setattr(sys, "argv", ["dimorphite", "--silent"])
    options = {"smiles_file": str(source), "min_ph": 7.4, "max_ph": 7.4, "pka_precision": 0, "silent": False}
    with caplog.at_level("INFO", logger=engine.logger.name):
        result = engine.main(
            {**options, "return_as_list": mode == "list", "output_file": str(output) if mode == "file" else None}
        )
    if mode == "list":
        lines = result
    elif mode == "file":
        lines = output.read_text().splitlines()
    else:
        lines = [record.message for record in caplog.records if "\t" in record.message]
    assert {tuple(line.split()) for line in lines} == {
        ("CC(=O)[O-]", "acid"),
        ("CC[NH3+]", "amine"),
        ("CC(C)O", "carbon"),
    }
    assert "Skipping poorly formed SMILES" in caplog.text


@pytest.mark.parametrize("ph,expected", [(0.0, {"CC(=O)O"}), (14.0, {"CC(=O)[O-]"})])
def test_smiles_and_molecule_interfaces_agree_on_acid_states(ph, expected):
    options = {"min_ph": ph, "max_ph": ph, "pka_precision": 0}
    records = list(engine.Protonate({"smiles": "CC(=O)O", "label_states": True, **options}))
    assert {line.split()[0] for line in records} == expected
    assert all(line.split()[1] == ("PROTONATED" if ph == 0 else "DEPROTONATED") for line in records)
    products = engine.protonate_mol_variants(Chem.MolFromSmiles("CC(=O)O"), **options)
    assert {Chem.MolToSmiles(mol) for mol in products} == expected


def test_variant_limit_and_nonionizable_molecules():
    options = {"min_ph": 0, "max_ph": 14, "max_variants": 1, "silent": False}
    records = list(engine.Protonate({"smiles": "OC(=O)CCN", **options}))
    products = engine.protonate_mol_variants(Chem.MolFromSmiles("OC(=O)CCN"), **options)
    assert len(records) == len(products) == 1
    assert Chem.MolFromSmiles(records[0].split()[0]) is not None
    for smiles in ["CC", "[Na+]", "C[C@H](F)Cl"]:
        products = engine.protonate_mol_variants(Chem.MolFromSmiles(smiles))
        assert [Chem.MolToSmiles(mol) for mol in products] == [smiles]
        assert [line.strip() for line in engine.Protonate({"smiles": smiles})] == [smiles]


def test_molecule_list_preserves_typed_properties():
    mol = Chem.MolFromSmiles("CC(=O)[O-].CC[NH3+]")
    setters = {
        "count": (3, "SetIntProp"),
        "weight": (1.5, "SetDoubleProp"),
        "flag": (True, "SetBoolProp"),
        "label": ("input", "SetProp"),
    }
    for key, (value, setter) in setters.items():
        getattr(mol, setter)(key, value)
        getattr(mol.GetAtomWithIdx(3), setter)(key, value)
    original = Chem.MolToSmiles(mol)
    products = engine.run_with_mol_list([mol], min_ph=0, max_ph=14)
    assert len(products) > 1
    assert Chem.MolToSmiles(mol) == original
    for product in products:
        for key, (value, _) in setters.items():
            assert product.GetPropsAsDict()[key] == value
            assert product.GetAtomWithIdx(3).GetPropsAsDict()[key] == value


@pytest.mark.parametrize("argument", ["smiles", "smiles_file", "output_file", "test"])
def test_molecule_list_rejects_file_interface_arguments(argument):
    with pytest.raises(Exception, match=argument):
        engine.run_with_mol_list([], **{argument: "unused"})


def test_cli_reports_missing_input_and_invalid_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dimorphite"])
    with pytest.raises(Exception, match="No SMILES"):
        engine.main()
    monkeypatch.setattr(sys, "argv", ["dimorphite", "--not-an-option"])
    with pytest.raises(Exception, match="unrecognized arguments"):
        engine.main()
    assert "--smiles_file" in capsys.readouterr().out


def test_invalid_smiles_do_not_create_products():
    for value in [None, 7, "not-a-smiles"]:
        assert engine.UtilFuncs.convert_smiles_str_to_mol(value) is None
    sites, mol = engine.ProtSubstructFuncs.get_prot_sites_and_target_states("not-a-smiles", [])
    assert sites == [] and mol is None
