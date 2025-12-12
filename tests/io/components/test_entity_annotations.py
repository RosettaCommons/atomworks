import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.common import not_isin
from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import annotate_entities
from tests.io.conftest import get_pdb_path

# fmt: off
MOLECULE_ENTITY_TEST_CASES = [
    {
        # Protein-protein heteromer, with glycosylation
        "pdb_id": "1ivo",
        "chains_with_same_molecule_entity": [
            ["A", "E", "F", "G", "H", "I", "J"],
            ["B", "K", "L", "M"],
            ["C", "D"],
        ],
    },
    {
        # Protein-protein homomer, no transformations
        "pdb_id": "1mna",
        "chains_with_same_molecule_entity": [
            ["A", "B"],
        ],
    },
    {
        # Protein-protein homomer, with transformations
        "pdb_id": "1a8o",
        "chains_with_same_molecule_entity": [
            ["A"],
        ],
    },
    {
        # Protein-protein heteromer, with glycosylation, where two glycosylated chains have the same molecule entity (same bond connectivity, despite having multiple chains covalently bound)
        "pdb_id": "1hge",
        "chains_with_same_molecule_entity": [
            [
                "B", "N", "F", "X", "D", "S",
            ],  # Two equivalent glycosylated chains, each involving one protein and one glycan chain
            [
                "A", "J", "K", "G", "L", "C", "O", "P", "H", "Q", "E", "T", "U", "I", "V",
            ],  # Three equivalent glycosylated chains, each involving one protein and four glycan chains
            ["M", "R", "W"],  # Small molecules, all with the same entity ID
        ],
    },
]
# fmt: on


def validate_molecule_entity_annotations(atom_array: AtomArray, test_case: dict):
    # Check that the number of molecule entitys is correct
    assert len(np.unique(atom_array.molecule_entity)) == len(test_case["chains_with_same_molecule_entity"])

    for chain_ids in test_case["chains_with_same_molecule_entity"]:
        chains_mask = np.isin(atom_array.chain_id, chain_ids)

        # Check that the ground truth chains with the same molecule entity match the computed molecule entitys 1hge
        assert len(np.unique(atom_array.molecule_entity[chains_mask])) == 1

        # Check that no other chains have the same molecule entity
        molecule_entity = atom_array.molecule_entity[chains_mask][0]
        all_chain_ids_with_chain_entity = np.unique(atom_array.chain_id[atom_array.molecule_entity == molecule_entity])
        assert set(all_chain_ids_with_chain_entity) == set(chain_ids)


@pytest.mark.parametrize("test_case", MOLECULE_ENTITY_TEST_CASES)
def test_add_molecule_entity_annotation(test_case: dict):
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(
        filename=path,
        build_assembly="all",
    )
    assert result is not None
    assembly_atom_array = result["assemblies"]["1"][0]  # Check the first model of the first assembly
    validate_molecule_entity_annotations(assembly_atom_array, test_case)


def test_add_molecule_entity_annotation_on_modified_pdb():
    """
    Tests on a custom-modified example of a molecule.
    This test loads the "1hge" PDB example (see test_add_molecule_entity_annotation)
    and manually adjusts the bond list to break the symmetry between the multiple copies of the same molecule.
    """
    pdb_id = "1hge"
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        build_assembly="all",
    )
    atom_array = result["assemblies"]["1"][0]  # First model

    # Manually adjust the atom array so that the polymer chain C is covalently bound to the glycan chain H at the second, rather than the first, residue
    # We thus break the symmetry between the multiple copies of the same molecule
    # We expect that the molecule entity for chain C will be different from the other chains with the same chain entity ID
    atom_a_mask = (
        (atom_array.chain_id == "C")
        & (atom_array.res_name == "ASN")
        & (atom_array.res_id == 165)
        & (atom_array.atom_name == "ND2")
    )
    atom_a_index = np.where(atom_a_mask)[0][0]

    atom_b_mask = (
        (atom_array.chain_id == "H")
        & (atom_array.res_name == "NAG")
        & (atom_array.res_id == 1)
        & (atom_array.atom_name == "C1")
    )
    atom_b_index = np.where(atom_b_mask)[0][0]

    atom_c_mask = (
        (atom_array.chain_id == "H")
        & (atom_array.res_name == "NAG")
        & (atom_array.res_id == 2)
        & (atom_array.atom_name == "C2")
    )
    atom_c_index = np.where(atom_c_mask)[0][0]

    # Remove the existing bond between atom A and atom B...
    atom_array.bonds.remove_bond(atom_a_index, atom_b_index)

    # ...and create a new bond between atom A and atom C
    atom_array.bonds.add_bond(atom_a_index, atom_c_index)

    # manual test case
    # fmt: off
    test_case = {
        # Protein-protein heteromer, with glycosylation, where two glycosylated chains have the same molecule entity (same bond connectivity, despite having multiple chains covalently bound)
        "pdb_id": "1hge",
        "chains_with_same_molecule_entity": [
            [
                "B", "N", "F", "X", "D", "S",
            ],  # Three equivalent glycosylated chains, each involving one protein and one glycan chain
            [
                "A", "J", "K", "G", "L", "E", "T", "U", "I", "V",
            ],  # Two equivalent glycosylated chains, each involving one protein and four glycan chains
            [
                "C", "O", "P", "H", "Q",
            ],  # One glycosylated chain with a different bond connectivity (manual change)
            ["M", "R", "W"],  # Small molecules, all with the same entity ID
        ],
    }
    # fmt: on

    # Re-annotate the entities (since we manually changed the bonds)
    atom_array, _ = annotate_entities(
        atom_array, level="molecule", lower_level_id="pn_unit_id", lower_level_entity="pn_unit_entity"
    )

    validate_molecule_entity_annotations(atom_array, test_case)


ADD_CHAIN_ENTITY_TEST_CASES = [
    {"pdb_id": "1ivo", "equivalent_chains": [["A", "B"], ["C", "D"], ["E"], ["F", "G", "H", "I", "J", "K", "L", "M"]]},
    # TODO: Add more test cases, including ones that were previously failing
]


@pytest.mark.parametrize("test_case", ADD_CHAIN_ENTITY_TEST_CASES)
def test_regenerate_and_add_chain_entity_annotation(test_case):
    """
    Tests that we:
    - Regenerate the chain entities for equivalent chains (ensure all equivalent chains have the same chain_entity)
    - Add the chain entity annotation to the atom array
    """
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(filename=path, hydrogen_policy="remove")
    atom_array = result["assemblies"]["1"][0]  # First model, first assembly

    for equivalent_chains in test_case["equivalent_chains"]:
        chain_entity_atom_array = atom_array[np.isin(atom_array.chain_id, equivalent_chains)]
        chain_entity = np.unique(chain_entity_atom_array.chain_entity)

        # ... check that all equivalent chains have the same chain_entity
        assert len(chain_entity) == 1, f"Chains {equivalent_chains} do not have the same chain_entity"

        # ... that no other chains have the same chain_entity
        other_chain_atom_array = atom_array[not_isin(atom_array.chain_id, equivalent_chains)]
        assert not np.any(other_chain_atom_array.chain_entity == chain_entity), (
            f"Chains {equivalent_chains} share chain_entity with other chains"
        )

        # ... and that all chains with the same chain_entity have the same sequence
        sequences = [
            chain_entity_atom_array[chain_entity_atom_array.chain_id == chain_id].res_name
            for chain_id in equivalent_chains
        ]
        assert all(np.array_equal(sequences[0], arr) for arr in sequences[1:]), (
            "Sequences are not equal within an entity."
        )


if __name__ == "__main__":
    pytest.main([__file__])
