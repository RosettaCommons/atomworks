import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray
from rdkit import Chem

from atomworks.constants import STANDARD_AA
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.tools.rdkit import (
    atom_array_from_rdkit,
    atom_array_to_rdkit,
    ccd_code_to_rdkit,
    fix_charge_based_on_valence,
    get_morgan_fingerprint_from_rdkit_mol,
    sdf_to_rdkit,
    smiles_to_rdkit,
)
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.io.utils.io_utils import load_any
from tests.io.conftest import TEST_DATA_IO

try:
    # Settings for debugging & interactive tests
    from rdkit.Chem.Draw import IPythonConsole

    IPythonConsole.kekulizeStructures = False
    IPythonConsole.drawOptions.addAtomIndices = True
    IPythonConsole.ipython_3d = True
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.drawOptions.addStereoAnnotation = True
    IPythonConsole.molSize = 600, 300
except ImportError:
    pass

TEST_SMILES = [
    "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O)N",  # Adenosine
    "C1=CC=CC=C1",  # Benzene
    "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@@H](O3)n4cnc5c4ncnc5N)O)O)O)O)C(=O)N",  # NAD
]
TEST_ATOM_ARRAYS = [struc.info.residue("ALA"), struc.info.residue("NAD")]


@pytest.mark.parametrize("conformer_id", [None, 17, 41])
def test_rdkit_conversion_selects_conformer_ids_without_mutating_source(conformer_id):
    """Select native conformer IDs, or the first conformer, without sharing coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    positions = np.arange(9, dtype=float).reshape(3, 3)
    for native_id in (17, 41):
        conformer = Chem.Conformer(3)
        conformer.SetId(native_id)
        conformer.SetPositions(positions + native_id)
        mol.AddConformer(conformer, assignId=False)
    source_smiles = Chem.MolToSmiles(mol)

    result = atom_array_from_rdkit(mol, conformer_id=conformer_id)

    expected_id = 17 if conformer_id is None else conformer_id
    np.testing.assert_array_equal(result.coord, positions + expected_id)
    assert result.element.tolist() == ["C", "C", "O"]
    np.testing.assert_array_equal(result.bonds.as_array(), [[0, 1, 1], [1, 2, 1]])
    result.coord[0] = 99
    assert Chem.MolToSmiles(mol) == source_smiles
    assert [conformer.GetId() for conformer in mol.GetConformers()] == [17, 41]
    for conformer in mol.GetConformers():
        np.testing.assert_array_equal(conformer.GetPositions(), positions + conformer.GetId())


def test_rdkit_conversion_looks_up_ids_only_when_coordinates_are_requested():
    """Missing IDs fail native lookup, but disabled or unavailable coordinates remain NaN."""
    mol = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(3)
    conformer.SetId(17)
    mol.AddConformer(conformer, assignId=False)

    with pytest.raises(ValueError, match="Bad Conformer Id"):
        atom_array_from_rdkit(mol, conformer_id=0)
    without_coords = atom_array_from_rdkit(mol, conformer_id=99, set_coord_if_available=False)
    assert np.isnan(without_coords.coord).all()
    assert mol.GetConformer(17).GetId() == 17
    no_conformers = atom_array_from_rdkit(Chem.MolFromSmiles("CCO"), conformer_id=99)
    assert np.isnan(no_conformers.coord).all()


@pytest.mark.parametrize("metadata", ["absent", "unmatched", "partial", "matched"])
@pytest.mark.parametrize("remove_hydrogens", [False, True])
@pytest.mark.parametrize("remove_inferred_atoms", [False, True])
def test_rdkit_output_filters_are_independent_of_annotation_matches(metadata, remove_hydrogens, remove_inferred_atoms):
    """Filter authored and inferred atoms independently of carried annotation matches."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    coordinates = np.arange(18, dtype=float).reshape(6, 3)
    conformer = Chem.Conformer(6)
    conformer.SetPositions(coordinates)
    mol.AddConformer(conformer)
    authored_ids = np.array([-1, 20, 30, -1, -1, -1])
    for index in (1, 2):
        mol.GetAtomWithIdx(index).SetIntProp("rdkit_atom_id", int(authored_ids[index]))
    annotations = {}
    if metadata != "absent":
        annotations = {
            "rdkit_atom_id": np.array({"unmatched": [98, 99], "partial": [20, 99], "matched": [20, 30]}[metadata]),
            "custom_tag": np.array([7, 8], dtype=np.uint8),
        }
        mol._annotations = {name: values.copy() for name, values in annotations.items()}
    original = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    result = atom_array_from_rdkit(mol, remove_hydrogens=remove_hydrogens, remove_inferred_atoms=remove_inferred_atoms)

    keep = np.ones(6, dtype=bool)
    if remove_hydrogens:
        keep &= np.array([6, 8, 1, 1, 1, 1]) != 1
    if remove_inferred_atoms:
        keep &= authored_ids != -1
    np.testing.assert_array_equal(result.coord, coordinates[keep])
    np.testing.assert_array_equal(result.element, np.array(["C", "O", "H", "H", "H", "H"])[keep])
    np.testing.assert_array_equal(result.rdkit_atom_id, authored_ids[keep])
    bonds = np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1]])
    bonds = bonds[keep[bonds[:, :2]].all(axis=1)]
    new_indices = np.cumsum(keep) - 1
    bonds[:, :2] = new_indices[bonds[:, :2]]
    np.testing.assert_array_equal(result.bonds.as_array(), bonds)
    if remove_hydrogens:
        np.testing.assert_array_equal(result.nhyd, np.array([3, 1, 0, 0, 0, 0])[keep])
    else:
        assert "nhyd" not in result.get_annotation_categories()
    if metadata in ("absent", "unmatched"):
        assert "custom_tag" not in result.get_annotation_categories()
    else:
        tags = np.array([255, 7, 8 if metadata == "matched" else 255, 255, 255, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result.custom_tag, tags[keep])
        assert result.custom_tag.dtype == np.uint8
        result.custom_tag[:] = 0
    result.coord[:] = 99
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == original
    assert hasattr(mol, "_annotations") == (metadata != "absent")
    if annotations:
        assert set(mol._annotations) == set(annotations)
        for name, values in annotations.items():
            np.testing.assert_array_equal(mol._annotations[name], values)


def test_rdkit_inferred_atom_filter_can_remove_all_unmatched_atoms():
    """Unmatched carried metadata does not retain atoms lacking authored identities."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol._annotations = {"rdkit_atom_id": np.array([99]), "custom_tag": np.array([7])}

    result = atom_array_from_rdkit(mol, remove_inferred_atoms=True)

    assert len(result) == 0
    assert result.bonds.get_bond_count() == 0
    assert "custom_tag" not in result.get_annotation_categories()


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_smiles_to_rdkit_to_atom_array(smiles):
    mol = smiles_to_rdkit(smiles)

    # remove the inferred hydrogens again
    mol = Chem.RemoveHs(mol)

    atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=True)

    # Add extra annotations
    atom_array.res_name = ["UNL"] * atom_array.array_length()
    atom_array.chain_id = ["A"] * atom_array.array_length()
    atom_array.set_annotation(
        "atom_name", atom_array.element.astype(object) + np.arange(atom_array.array_length()).astype(str)
    )

    assert isinstance(atom_array, AtomArray)
    assert atom_array.array_length() == mol.GetNumAtoms()


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_smiles_to_atom_array_to_rdkit(smiles):
    inputs = []
    inputs.append(
        {
            "smiles": smiles,
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "A",
        }
    )
    atom_array = components_to_atom_array(inputs)
    mol = atom_array_to_rdkit(atom_array, set_coord=True, hydrogen_policy="keep")
    new_atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=False)
    assert new_atom_array.array_length() == atom_array.array_length()
    for annotation in ["chain_id", "res_id", "res_name", "atom_name"]:
        assert np.array_equal(new_atom_array.get_annotation(annotation), atom_array.get_annotation(annotation))


@pytest.mark.parametrize("test_atom_array", TEST_ATOM_ARRAYS)
def test_atom_array_rdkit_interconversion(test_atom_array):
    test_atom_array.chain_id = ["A"] * test_atom_array.array_length()

    # Convert AtomArray to RDKit Mol
    mol = atom_array_to_rdkit(test_atom_array, set_coord=True, hydrogen_policy="keep")

    # Convert back to AtomArray
    new_atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=False)

    # Check if the number of atoms is preserved
    assert new_atom_array.array_length() == test_atom_array.array_length()

    # Check if annotations are preserved
    for annotation in ["chain_id", "res_id", "res_name", "atom_name"]:
        assert np.array_equal(new_atom_array.get_annotation(annotation), test_atom_array.get_annotation(annotation))
    assert np.allclose(new_atom_array.coord, test_atom_array.coord)
    mol._annotations["atom_name"][0] = ""
    assert test_atom_array.atom_name[0] != ""


@pytest.mark.parametrize("set_coord", [None, False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_rdkit_coordinates_respect_explicit_choice(set_coord, missing):
    """Infer coordinate inclusion only when the caller leaves it unspecified."""
    atoms = struc.info.residue("ALA")
    if missing:
        atoms.coord[0] = np.nan

    mol = atom_array_to_rdkit(atoms, set_coord=set_coord)

    expected = not missing if set_coord is None else set_coord
    assert mol.GetNumConformers() == int(expected)
    if expected:
        np.testing.assert_allclose(mol.GetConformer().GetPositions(), atoms.coord, equal_nan=True)


def test_fixing_molecules():
    bad = Chem.MolFromSmiles("[O-]=P([O-])([O-])[O-].C[N+](C)(C)C", sanitize=False)
    bad.UpdatePropertyCache(strict=False)
    expected = Chem.MolFromSmiles("O=P([O-])([O-])[O-].C[N+](C)(C)C")
    repaired = fix_charge_based_on_valence(bad)
    Chem.SanitizeMol(repaired)
    assert Chem.MolToSmiles(repaired) == Chem.MolToSmiles(expected)
    assert fix_charge_based_on_valence(repaired) is repaired

    unrepairable = Chem.MolFromSmiles("C(C)(C)(C)(C)C.O=P([O-])([O-])[O-]", sanitize=False)
    charges = [atom.GetFormalCharge() for atom in unrepairable.GetAtoms()]
    assert fix_charge_based_on_valence(unrepairable) is unrepairable
    assert [atom.GetFormalCharge() for atom in unrepairable.GetAtoms()] == charges

    smi = "c1cc(c[n](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N"
    smi_correct = "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N"

    # Check that loading and sanitizing `smi` fails
    with pytest.raises(Chem.MolSanitizeException):
        smiles_to_rdkit(smi)

    mol = smiles_to_rdkit(smi, sanitize=False, generate_conformers=False)  # noqa: F841
    mol_correct = smiles_to_rdkit(smi_correct)  # noqa: F841

    # TODO: Currently this cannot be fixed by our `fix_mol` function. Revisit this test once we implemented the remaining `TODO`s in `fix_mol`.
    # mol = fix_mol(
    #     mol,
    #     attempt_fix_by_normalizing_like_chembl=True,
    #     attempt_fix_by_normalizing_like_rdkit=True,
    #     attempt_fix_valence_by_changing_formal_charge=True,
    #     in_place=True,
    # )
    # assert Chem.MolToInchi(mol) == Chem.MolToInchi(mol_correct)


@pytest.fixture(scope="module")
def molecules():
    # Create RDKit molecules for some amino acids and small molecules
    mols = {
        "Leucine": ccd_code_to_rdkit("LEU", hydrogen_policy="remove"),
        "Isoleucine": ccd_code_to_rdkit("ILE", hydrogen_policy="remove"),
        "Glycine": ccd_code_to_rdkit("GLY", hydrogen_policy="remove"),
        "HEM": ccd_code_to_rdkit("HEM", hydrogen_policy="remove"),
        "NAG": ccd_code_to_rdkit("NAG", hydrogen_policy="remove"),
        "BMA": ccd_code_to_rdkit("BMA", hydrogen_policy="remove"),
        "CustomUNL": atom_array_to_rdkit(
            load_any(TEST_DATA_IO / "test_unl_ligand_with_bonds.cif", model=1), set_coord=False, hydrogen_policy="keep"
        ),
    }
    return mols


def test_fingerprints(molecules):
    # Generate fingerprints for each molecule
    fingerprints = {name: get_morgan_fingerprint_from_rdkit_mol(mol) for name, mol in molecules.items()}

    # Calculate similarities and check if similar molecules have higher similarity scores
    sim_leu_ile = Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Isoleucine"])
    sim_leu_gly = Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Glycine"])
    sim_nag_bma = Chem.DataStructs.TanimotoSimilarity(fingerprints["NAG"], fingerprints["BMA"])
    sim_nag_hem = Chem.DataStructs.TanimotoSimilarity(fingerprints["NAG"], fingerprints["HEM"])
    # Assert that leucine is more similar to isoleucine than to glycine
    assert sim_leu_ile > sim_leu_gly, "Leucine should be more similar to Isoleucine than to Glycine"

    # Asser that sugars (NAG and BMA) are more similar to each other than to HEM by at least a factor of 5
    assert sim_nag_bma > 5 * sim_nag_hem, "Sugars should be more similar to each other than to HEM"

    # Residues should have a similarity of 1.0 with themselves
    assert Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Leucine"]) == 1.0

    # Lycine and [NAG, BMA, HEM] should be less similar than 0.3 (very different)
    assert Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["NAG"]) < 0.3
    assert Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["BMA"]) < 0.3
    assert Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["HEM"]) < 0.3
    assert Chem.DataStructs.TanimotoSimilarity(fingerprints["CustomUNL"], fingerprints["Leucine"]) < 0.3


def test_chirality_detection_from_ccd():
    # Check the 20 natural amino acids for the correct stereochemistry
    for aa in STANDARD_AA:
        if aa == "GLY":
            assert Chem.FindMolChiralCenters(ccd_code_to_rdkit(aa)) == []
        elif aa == "CYS":
            # NOTE: For cystine the L-amino acid corresponds to na R-configuration
            #  around the CA atom
            assert Chem.FindMolChiralCenters(ccd_code_to_rdkit(aa)) == [(1, "R")]
        elif aa == "ILE":
            assert Chem.FindMolChiralCenters(ccd_code_to_rdkit(aa)) == [(1, "S"), (4, "S")]
        elif aa == "THR":
            assert Chem.FindMolChiralCenters(ccd_code_to_rdkit(aa)) == [(1, "S"), (4, "R")]
        else:
            assert Chem.FindMolChiralCenters(ccd_code_to_rdkit(aa)) == [(1, "S")]

    # Check a handful of non-standard amino acids
    assert Chem.FindMolChiralCenters(ccd_code_to_rdkit("DAL")) == [
        (1, "R")
    ], "D-alanine should have a R configuration at the CA atom"
    assert Chem.FindMolChiralCenters(ccd_code_to_rdkit("DCY")) == [
        (1, "S")
    ], "D-cystine should have a S configuration at the CA atom"
    assert Chem.FindMolChiralCenters(ccd_code_to_rdkit("DTH")) == [
        (1, "R"),
        (2, "S"),
    ], "D-threonine should have a R configuration at the CA atom and a S configuration at the CB atom"


def test_chirality_detection_from_smiles():
    # Alanine (ALA)
    mol = smiles_to_rdkit("C[C@@H](C(=O)O)N")
    assert Chem.FindMolChiralCenters(mol) == [(1, "S")], "Alanine should have a S configuration at the CA atom"

    # D-cystine (DCY)
    mol = smiles_to_rdkit("C([C@H](C(=O)O)N)S")
    assert Chem.FindMolChiralCenters(mol) == [(1, "S")], "D-cystine should have a S configuration at the CA atom"


def test_chriality_in_spoofed_rdkit_molecules():
    # fmt: off
    dal_coord = np.array(
      [[-1.564, -0.992,  0.101],
       [-0.724,  0.176,  0.402],
       [-1.205,  1.374, -0.42 ],
       [ 0.709, -0.132,  0.051],
       [ 1.001, -1.213, -0.403],
       [ 1.66 ,  0.795,  0.243],
       [-1.281, -1.723,  0.736],
       [-2.509, -0.741,  0.351],
       [-0.796,  0.411,  1.464],
       [-1.133,  1.139, -1.481],
       [-2.241,  1.597, -0.166],
       [-0.582,  2.24 , -0.197],
       [ 2.58 ,  0.598,  0.018]]
    )
    dal_coord[[2,3,4]] = dal_coord[[3,4,2]]  # ... adjust order to be C, O, CB (not CB, C, O as in CCD from where these coords are from)
    # fmt: on

    # Get ALA from the CCD
    atom_array = struc.info.residue("ALA")
    mol_ala = atom_array_to_rdkit(atom_array)
    assert Chem.FindMolChiralCenters(mol_ala) == [(1, "S")]

    # ... spoof the coordinates with DAL_COORD (which have inverted chirality)
    atom_array.coord = dal_coord
    mol_ala_inverted = atom_array_to_rdkit(atom_array)
    assert Chem.FindMolChiralCenters(mol_ala_inverted) == [(1, "R")]


@pytest.mark.parametrize(
    "ccd_code",
    [
        "35N",  # Simple TMC with copper
        "GDP",  # Guanosine diphosphate (simpler than ATP)
        "ATP",  # Adenosine triphosphate (several phosphate groups)
        "SAM",  # S-adenosyl methionine (sulfur chemistry)
        "FMN",  # Flavin mononucleotide
    ],
)
def test_atom_array_to_rdkit_bond_inference(ccd_code: str):
    """Test that inferring bonds from coordinates produces same results as using original bonds."""

    original = atom_array_from_ccd_code(ccd_code)

    # Path 1: Normal conversion using original CCD bonds
    mol_with_bonds = atom_array_to_rdkit(
        original,
        infer_bonds=False,
        hydrogen_policy="keep",  # Keep all hydrogens
        sanitize=True,
    )
    result_with_bonds = atom_array_from_rdkit(
        mol_with_bonds,
        remove_hydrogens=False,  # Keep hydrogens for comparison
    )

    # Path 2: Conversion with bond inference from 3D coordinates
    original_no_bonds = original.copy()
    original_no_bonds.bonds = None  # Remove bonds to force inference

    mol_inferred = atom_array_to_rdkit(
        original_no_bonds,
        infer_bonds=True,  # Infer bonds from coordinates
        hydrogen_policy="keep",
        sanitize=True,
    )
    result_inferred = atom_array_from_rdkit(
        mol_inferred,
        remove_hydrogens=False,
    )

    # (a) Total charge is the same
    charge_with_bonds = int(result_with_bonds.charge.sum())
    charge_inferred = int(result_inferred.charge.sum())
    assert charge_with_bonds == charge_inferred, (
        f"{ccd_code}: Total charge differs - " f"with bonds: {charge_with_bonds}, inferred: {charge_inferred}"
    )

    # (b) Bond connectivity is the same
    connectivity_with_bonds = _get_bond_connectivity(result_with_bonds.bonds)
    connectivity_inferred = _get_bond_connectivity(result_inferred.bonds)
    assert connectivity_with_bonds == connectivity_inferred, (
        f"{ccd_code}: Bond connectivity differs\n"
        f"  Bonds only in original: {connectivity_with_bonds - connectivity_inferred}\n"
        f"  Bonds only in inferred: {connectivity_inferred - connectivity_with_bonds}"
    )

    # (c) Hybridization is the same
    specified_mask = result_with_bonds.hyb != -1  # Only compare atoms with specified hybridization
    assert np.array_equal(
        result_with_bonds.hyb[specified_mask], result_inferred.hyb[specified_mask]
    ), f"{ccd_code}: Hybridization differs"

    # (d) Aromaticity is the same
    assert np.array_equal(
        result_with_bonds.is_aromatic, result_inferred.is_aromatic
    ), f"{ccd_code}: Aromaticity differs"


def _get_bond_connectivity(bonds: struc.BondList | None) -> set[tuple[int, int]]:
    """Extract bond connectivity (which atoms are bonded) ignoring bond types."""
    if bonds is None:
        return set()

    bond_array = bonds.as_array()  # Shape: (n_bonds, 3) - [atom1, atom2, type]
    return {(min(int(i), int(j)), max(int(i), int(j))) for i, j, _ in bond_array}


def test_nhyd_with_explicit_hydrogens():
    """nhyd must agree whether Hs are implicit or explicit graph neighbors.

    ATP (adenosine triphosphate) has a purine aromatic ring, sp3 sugar carbons, and
    phosphate oxygens — a good spread of heavy atoms with differing H counts.
    """

    def nhyd_by_name(hydrogen_policy: str) -> dict[str, int]:
        mol = ccd_code_to_rdkit("ATP", hydrogen_policy=hydrogen_policy)
        aa = atom_array_from_rdkit(mol, set_coord_if_available=False, remove_hydrogens=True)
        return dict(zip(aa.atom_name, aa.nhyd, strict=True))

    nhyd_implicit = nhyd_by_name("remove")
    nhyd_explicit = nhyd_by_name("keep")

    assert nhyd_implicit == nhyd_explicit


def test_an_sdf_can_keep_the_hydrogens_it_states():
    """RDKit drops the 32 hydrogens the CCD's HEM ideal SDF writes out, unless asked not to."""
    dropped = sdf_to_rdkit(TEST_DATA_IO / "HEM_ideal.sdf")
    kept = sdf_to_rdkit(TEST_DATA_IO / "HEM_ideal.sdf", remove_hydrogens=False)

    assert dropped.GetNumAtoms() == 43
    assert kept.GetNumAtoms() == 75
    assert sum(atom.GetAtomicNum() == 1 for atom in kept.GetAtoms()) == 32


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
