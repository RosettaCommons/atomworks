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
    get_morgan_fingerprint_from_rdkit_mol,
    smiles_to_rdkit,
)
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


def test_fixing_molecules():
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
    assert Chem.FindMolChiralCenters(ccd_code_to_rdkit("DAL")) == [(1, "R")], (
        "D-alanine should have a R configuration at the CA atom"
    )
    assert Chem.FindMolChiralCenters(ccd_code_to_rdkit("DCY")) == [(1, "S")], (
        "D-cystine should have a S configuration at the CA atom"
    )
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


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
