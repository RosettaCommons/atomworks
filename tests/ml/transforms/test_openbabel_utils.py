import pytest
from biotite.structure import AtomArray

from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.transforms.openbabel_utils import atom_array_from_openbabel, atom_array_to_openbabel
from atomworks.ml.utils.testing import cached_parse


def _get_test_case(pdb_id: str, selector: callable) -> AtomArray:
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]
    return selector(atom_array)


TEST_CASES = [
    _get_test_case("5ocm", lambda x: x[(x.res_name == "NAP") & (x.chain_id == "G")]),  # NAP
    _get_test_case("5ocm", lambda x: x[(x.res_name == "ALA") & (x.res_id == 17) & (x.chain_id == "A")]),
    cached_parse("5ocm")["atom_array"],  # Full structure with ligands & protein
    cached_parse("6lyz")["atom_array"],  # protein only structure
]

ANNOTATIONS_TO_COMPARE = ["chain_id", "res_name", "res_id", "atom_name", "element"]


@pytest.mark.parametrize("atom_array", TEST_CASES)
def test_with_explicit_hydrogens(atom_array):
    obmol = atom_array_to_openbabel(atom_array, infer_hydrogens=False, infer_aromaticity=False)
    array_reconstructed = atom_array_from_openbabel(obmol)
    assert_same_atom_array_or_stack(
        atom_array,
        array_reconstructed,
        compare_coords=True,
        compare_bonds=False,
        annotations_to_compare=ANNOTATIONS_TO_COMPARE,
    )


@pytest.mark.parametrize("atom_array", TEST_CASES)
def test_with_implicit_hydrogens_explicit_removed(atom_array):
    atom_array_no_hydrogens = atom_array[atom_array.atomic_number != 1]
    obmol = atom_array_to_openbabel(atom_array, infer_hydrogens=True, infer_aromaticity=False)
    array_reconstructed = atom_array_from_openbabel(obmol)
    assert_same_atom_array_or_stack(
        atom_array_no_hydrogens,
        array_reconstructed,
        compare_coords=True,
        compare_bonds=True,
        annotations_to_compare=ANNOTATIONS_TO_COMPARE,
    )


@pytest.mark.parametrize("atom_array", TEST_CASES)
def test_with_implicit_hydrogens_no_explicit(atom_array):
    atom_array_no_hydrogens = atom_array[atom_array.atomic_number != 1]
    obmol = atom_array_to_openbabel(atom_array, infer_hydrogens=True, infer_aromaticity=False)
    array_reconstructed = atom_array_from_openbabel(obmol)
    assert_same_atom_array_or_stack(
        atom_array_no_hydrogens,
        array_reconstructed,
        compare_coords=True,
        compare_bonds=True,
        annotations_to_compare=ANNOTATIONS_TO_COMPARE,
    )


@pytest.mark.parametrize("atom_array", TEST_CASES)
def test_ignoring_hydrogens(atom_array):
    atom_array_no_hydrogens = atom_array[atom_array.atomic_number != 1]
    obmol = atom_array_to_openbabel(atom_array_no_hydrogens, infer_hydrogens=False, infer_aromaticity=False)
    array_reconstructed = atom_array_from_openbabel(obmol)
    assert_same_atom_array_or_stack(
        atom_array_no_hydrogens,
        array_reconstructed,
        compare_coords=True,
        compare_bonds=True,
        annotations_to_compare=ANNOTATIONS_TO_COMPARE,
    )


if __name__ == "__main__":
    pytest.main([__file__])
