"""PyTest function to check the assignment of PN unit IDs."""

import numpy as np
import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

PN_UNIT_IID_TEST_CASES = [
    {
        "pdb_id": "4ndz",
        "assembly_id": "1",
        "pn_unit_iids": ["A_1", "B_1", "C_1", "G_1,R_1", "H_1", "J_1", "Q_1", "S_1", "I_1,T_1", "U_1"],
        "q_pn_unit_iid": "C_1",
    },  # Covalently bonded ligands
    {
        "pdb_id": "4ndz",
        "assembly_id": "2",
        "q_pn_unit_iid": "D_1",
        "pn_unit_iids": [
            "D_1",
            "E_1",
            "F_1",
            "K_1",
            "L_1,W_1",
            "M_1",
            "N_1,Y_1",
            "O_1",
            "P_1",
            "V_1",
            "X_1",
            "Z_1",
        ],
    },  # Covalently bonded ligands, query PN unit of C_1, near the center
    {
        "pdb_id": "4ndz",
        "assembly_id": "1",
        "q_pn_unit_iid": "Q_1",
        "pn_unit_iids": ["A_1", "B_1", "C_1", "G_1,R_1", "H_1", "I_1,T_1", "J_1", "Q_1", "S_1", "U_1"],
    },  # Covalently bonded ligands, query PN unit of Q_1, off to the side
    {
        "pdb_id": "3ne7",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": ["A_1", "B_1", "C_1,D_1", "E_1", "H_1"],  # NOTE: E, H, and D are AF-3 excluded ligands
    },  # Covalently bonded ligands
    {
        "pdb_id": "1ivo",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": [
            "A_1",
            "B_1",
            "C_1",
            "D_1",
            "E_1",
            "F_1",
            "G_1",
            "H_1",
            "I_1",
            "J_1",
            "K_1",
            "L_1",
            "M_1",
        ],
    },  # Oligosaccharides covalently bonded to residues (glycoprotein)
    {
        "pdb_id": "1a8o",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": ["A_1", "A_2"],
    },  # Simple assembly with non-trivial symmetry
    {
        "pdb_id": "1rxz",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": [
            "A_1",
            "A_2",
            "A_3",
            "B_1",
            "B_2",
            "B_3",
        ],  # More complex assembly with non-trivial symmetry
    },
]


@pytest.mark.parametrize("test_case", PN_UNIT_IID_TEST_CASES)
def test_identifiers(test_case):
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(
        filename=path,
        build_assembly=(test_case["assembly_id"],),
    )
    atom_array = result["assemblies"][test_case["assembly_id"]][0]  # Choose first model

    generated_pn_unit_iids = sorted(np.unique(atom_array.pn_unit_iid.astype(str)))
    reference_pn_unit_iids = sorted(test_case["pn_unit_iids"])

    assert (
        generated_pn_unit_iids == reference_pn_unit_iids
    ), f"Generated PN unit instance IDs do not match reference PN unit IIDs for PDB ID {test_case['pdb_id']} and assembly_id {test_case['assembly_id']}."


MOLECULE_TEST_CASES = [
    {
        "pdb_id": "1ivo",
        "assembly_id": "1",
        "num_molecules": 4,
        "chain_iid_combinations": [
            ["A_1", "E_1", "F_1", "G_1", "H_1", "I_1", "J_1"],
            ["B_1", "K_1", "L_1", "M_1"],
            ["C_1"],
            ["D_1"],
        ],
    },
    {
        "pdb_id": "4js1",
        "assembly_id": "1",
        "num_molecules": 2,
        "chain_iid_combinations": [
            ["A_1", "B_1"],
            ["C_1"],
        ],
    },
    {
        "pdb_id": "1fyl",
        "assembly_id": "1",
        "num_molecules": 4,
        "chain_iid_combinations": [
            ["A_2"],
            ["A_4"],
            ["C_1"],
            ["C_3"],
        ],
    },
]


@pytest.mark.parametrize("test_case", MOLECULE_TEST_CASES)
def test_add_molecule_annotation(test_case: dict):
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(
        filename=path,
    )

    atom_array = result["assemblies"][test_case["assembly_id"]][0]  # Choose first model in the first assembly

    # Ensure that the number of molecules is correct
    assert len(np.unique(atom_array.molecule_iid)) == test_case["num_molecules"]

    # Ensure that the pn_unit_iid combinations are correct
    for chain_iid_combination in test_case["chain_iid_combinations"]:
        # Create the mask once and reuse it
        mask = np.isin(atom_array.chain_iid, chain_iid_combination)

        # Select atoms belonging to the molecule
        molecule_atoms = atom_array[mask]

        # Ensure that the molecule atoms have the same molecule id
        assert len(np.unique(molecule_atoms.molecule_iid)) == 1

        # Ensure no other atoms have the same molecule id
        unique_molecule_iid = np.unique(molecule_atoms.molecule_iid)[0]
        other_atoms = atom_array[~mask]
        assert not np.any(other_atoms.molecule_iid == unique_molecule_iid)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
