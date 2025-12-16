import numpy as np
import pytest
from rdkit import Chem

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from atomworks.ml.transforms.filters import HandleUndesiredResTokens, RemoveHydrogens
from atomworks.ml.transforms.rdkit_utils import (
    AddRDKitMoleculesForAtomizedMolecules,
    GenerateRDKitConformers,
    atom_array_from_rdkit,
)
from atomworks.ml.utils.testing import cached_parse

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


TEST_CASES = [
    # .. single small molecules
    {"pdb_id": "5ocm", "corrupt_charge": False},
    {"pdb_id": "5ocm", "corrupt_charge": True},
    # ... molecules with covalently attached ligands and glycans
    {"pdb_id": "1ivo", "corrupt_charge": False},
    {"pdb_id": "1ivo", "corrupt_charge": True},
    {"pdb_id": "3ne7", "corrupt_charge": False},
    {"pdb_id": "3ne7", "corrupt_charge": True},
    # ... atomizing standard residues mid-structure
    {"pdb_id": "6lyz", "corrupt_charge": False, "res_names_to_atomize": ["ALA"]},
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_rdkit_molecules_for_atomized_molecules(test_case):
    # Prepare input data
    data = cached_parse(test_case["pdb_id"])
    atom_array = data["atom_array"]

    if test_case["corrupt_charge"]:
        # ... obliterate the formal charge information
        atom_array.set_annotation("charge", np.zeros_like(atom_array.charge))

    res_names_to_atomize = test_case.get("res_names_to_atomize", [])
    res_names_to_ignore = [token for token in RF2AA_ATOM36_ENCODING.tokens if token not in res_names_to_atomize]

    # Apply the transform
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            FlagAndReassignCovalentModifications(),
            HandleUndesiredResTokens(["UNL"]),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=res_names_to_ignore,
                res_names_to_atomize=res_names_to_atomize,
            ),
            AddRDKitMoleculesForAtomizedMolecules(),
        ]
    )
    data = pipe(data)

    # Check if the rdkit key is added to the data dictionary
    assert "rdkit" in data

    # Check if RDKit molecules are created for each unique pn_unit_iid of atomized residues
    unique_pn_unit_iids = np.unique(data["atom_array"].pn_unit_iid[data["atom_array"].atomize])
    assert len(data["rdkit"]) == len(unique_pn_unit_iids)

    # Check if each RDKit molecule has the correct number of atoms
    for pn_unit_iid, rdmol in data["rdkit"].items():
        pn_unit_mask = (data["atom_array"].pn_unit_iid == pn_unit_iid) & data["atom_array"].atomize
        expected_num_atoms = np.sum(pn_unit_mask)
        assert rdmol.GetNumAtoms() > 0, f"RDKit molecule {pn_unit_iid} has no atoms"
        assert rdmol.GetNumAtoms() == expected_num_atoms, f"RDKit molecule {pn_unit_iid} has the wrong number of atoms"
        assert (
            Chem.SanitizeMol(rdmol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
        ), f"RDKit molecule {pn_unit_iid} failed sanitization"

        _mol_array = atom_array_from_rdkit(
            rdmol, set_coord_if_available=True, remove_hydrogens=True, remove_inferred_atoms=False
        )
        assert _mol_array.coord.shape == (
            expected_num_atoms,
            3,
        ), f"Atom array {pn_unit_iid} has the wrong number of coordinates"
        assert np.all(_mol_array.rdkit_atom_id >= 0), "RDKit atom ids are not correctly set"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generate_rdkit_conformers(test_case):
    # Prepare input data
    data = cached_parse(test_case["pdb_id"])
    atom_array = data["atom_array"]

    if test_case["corrupt_charge"]:
        # ... obliterate the formal charge information
        atom_array.set_annotation("charge", np.zeros_like(atom_array.charge))

    res_names_to_atomize = test_case.get("res_names_to_atomize", [])
    res_names_to_ignore = [token for token in RF2AA_ATOM36_ENCODING.tokens if token not in res_names_to_atomize]

    # Apply the transform
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            FlagAndReassignCovalentModifications(),
            HandleUndesiredResTokens(["UNL"]),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=res_names_to_ignore,
                res_names_to_atomize=res_names_to_atomize,
            ),
            AddRDKitMoleculesForAtomizedMolecules(),
            GenerateRDKitConformers(n_conformers=1, optimize_conformers=False),
        ]
    )
    data = pipe(data)

    assert "rdkit" in data

    # Check that each rdkit molecule has a conformer
    for rdmol in data["rdkit"].values():
        assert rdmol.GetNumConformers() > 0

    # Check that we can get the coordinates of the conformers as atom array coordinates
    for _pn_unit_iid, rdmol in data["rdkit"].items():
        mol_array = atom_array_from_rdkit(
            rdmol, set_coord_if_available=True, remove_hydrogens=True, remove_inferred_atoms=False
        )
        assert mol_array.coord.shape == (rdmol.GetNumAtoms(), 3)
        assert np.all(np.isfinite(mol_array.coord))


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
