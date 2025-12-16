import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.ml.encoding_definitions import AF3_TOKENS
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.symmetry import (
    FindAutomorphismsWithNetworkX,
    find_automorphisms_with_networkx,
    generate_automorphisms_from_atom_array_with_networkx,
)
from atomworks.ml.utils.numpy import get_indices_of_non_constant_columns
from atomworks.ml.utils.testing import cached_parse

TEST_CASES = [
    {
        "ccd_code": "BEZ",
        "expected_automorphisms": 4,
    },
    {
        "ccd_code": "60C",
        "expected_automorphisms": 120,
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_find_automorphisms_from_atom_array_with_networkx(test_case: dict):
    """
    Test the find_automorphisms_with_networkx function on residues from the CCD,
    ensuring that the correct number of automorphisms are found.
    """
    residue_atom_array = struc.info.residue(test_case["ccd_code"])

    # ...remove hydrogens
    residue_atom_array = residue_atom_array[residue_atom_array.element != "H"]

    # ...and generate automorphisms
    automorphisms = generate_automorphisms_from_atom_array_with_networkx(residue_atom_array)

    assert len(automorphisms) == test_case["expected_automorphisms"], f"Failed for CCD code: {test_case['ccd_code']}"


def test_manual_generate_automorphs_with_networkx():
    """
    Test the generate_automorphisms_from_atom_array_with_networkx function on a few residues with known automorphisms.
    """
    # +---------------- ASP ----------------+
    # ASP has four automorphisms, all of which occur due to resonance (and thus would not be detected by RDKit or OpenBabel)
    residue_atom_array = struc.info.residue(
        "ASP"
    )  # Aspartate - 4 automorphisms (two sets of equivalent oxygens, given resonane)
    residue_atom_array = residue_atom_array[residue_atom_array.element != "H"]

    # ...generate automorphisms
    automorphisms = generate_automorphisms_from_atom_array_with_networkx(residue_atom_array).tolist()

    assert len(automorphisms) == 4, "Failed for CCD code: ASP"

    # Manual check
    expected_automorphisms = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Identity
        [0, 1, 2, 3, 4, 5, 7, 6, 8],  # Interchange OD1 and OD2
        [0, 1, 2, 8, 4, 5, 6, 7, 3],  # Interchange O and OXT
        [0, 1, 2, 8, 4, 5, 7, 6, 3],  # Interchange OD1 and OD2, O and OXT
    ]

    expected_automorphisms = sorted(expected_automorphisms)
    automorphisms = sorted(automorphisms)

    # Check that the lists are the same
    for expected, actual in zip(expected_automorphisms, automorphisms, strict=False):
        assert expected == actual


def array_in_list(array: np.ndarray, list_of_arrays: list[np.ndarray] | np.ndarray[np.ndarray]) -> bool:
    """Helper function to check if an array is in a list of arrays."""
    return any(np.array_equal(array, item) for item in list_of_arrays)


TEST_PDB_IDS = ["4js1", "6gej", "6wtf"]


@pytest.mark.parametrize("pdb_id", TEST_PDB_IDS)
def test_find_automorphisms_within_entire_structure(pdb_id: str):
    """
    Test the find_automorphisms_with_networkx function on an entire structure.
    Important to ensure that atoms are indexed correctly.
    """
    inputs = cached_parse(pdb_id, hydrogen_policy="remove")  # Example with covalent modifications, small molecules
    atomize_transform = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=AF3_TOKENS,
        move_atomized_part_to_end=False,
    )
    output = atomize_transform(inputs)

    find_automorphisms_transform = FindAutomorphismsWithNetworkX()
    automorphisms = find_automorphisms_transform(output)["automorphisms"]

    atom_array = output["atom_array"]
    for automorphism in automorphisms:
        # ...get the identity
        residue = atom_array[automorphism[0]]
        residue_name = residue.res_name[0]

        # skip of "OXT" is present in the residue (it's a terminal residue, and will have a different number of automorphisms)
        if "OXT" in residue.atom_name:
            continue

        # ...if it's a glycine, there should be no automorphisms
        if residue_name == "GLY":
            assert len(automorphism) == 1, "Glycine should have no automorphisms, unless it's terminal."
        # ...if it's an arginine, there should be 2 automorphisms, both involving nitrogens
        if residue_name == "ARG":
            assert len(automorphism) == 2, "Arginine should have 2 automorphisms."
            changing_column_indices = get_indices_of_non_constant_columns(automorphism)
            n_element = ELEMENT_NAME_TO_ATOMIC_NUMBER["N"]
            assert np.all(
                residue.atomic_number[changing_column_indices] == n_element
            ), "All automorphisms of Arginine should involve nitrogens."
        # ...if it's a tyrosine, there should be 2 automorphisms, all involving carbons
        if residue_name == "TYR":
            assert len(automorphism) == 2, "Tyrosine should have 2 automorphisms (carbons must swap together)"
            changing_column_indices = get_indices_of_non_constant_columns(automorphism)
            n_element = ELEMENT_NAME_TO_ATOMIC_NUMBER["C"]
            assert np.all(
                residue.atomic_number[changing_column_indices] == n_element
            ), "All automorphisms of Tyrosine should involve carbons."
        # ...isoleucine should have no automorphisms
        if residue_name == "ILE":
            assert len(automorphism) == 1, "Isoleucine should have no automorphisms."

        # ...calculate automorphisms
        local_automorphisms = generate_automorphisms_from_atom_array_with_networkx(residue)

        assert len(local_automorphisms) == len(
            automorphism
        ), "Unequal automorphism numbers; likely due to incorrect indexing."

        for local_automorphism in local_automorphisms:
            # ...map to the global frame, as given by the first row of the automorphism
            mapped_automorphism = automorphism[0][local_automorphism]

            # ...check that the automorphism is present
            assert array_in_list(
                mapped_automorphism, automorphism
            ), "Automorphism not found in global automorphism list."


@pytest.mark.benchmark(group="find_automorphisms")
def benchmark_find_automorphisms_with_networkx(benchmark):
    """
    Benchmark the find_automorphisms_with_networkx function.
    With current implementation, runs in negligible time for moderate-sized structures (<100ms).
    """
    inputs = cached_parse("6wtf")  # As of 11/5/2024, reports a mean time of 99.0154ms
    atomize_transform = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=AF3_TOKENS,
        move_atomized_part_to_end=False,
    )
    output = atomize_transform(inputs)

    benchmark(find_automorphisms_with_networkx, output["atom_array"])


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
