import biotite.structure as struc
import numpy as np
import pytest

from atomworks.io.utils.selection import (
    AtomSelection,
    AtomSelectionStack,
    ChainIdxSlice,
    ResIdxSlice,
    get_mask_from_selection_string,
    get_residue_starts,
    parse_pymol_string,
    parse_selection_string,
)


@pytest.fixture
def basic_atom_array() -> struc.AtomArray:
    """Creates a basic atom array with multiple residues across different chains."""
    return struc.array(
        [
            # Residue 1, Chain A
            struc.Atom(np.array([1, 1, 1]), chain_id="A", res_id=1, res_name="ALA", atom_name="N"),
            struc.Atom(np.array([1, 1, 2]), chain_id="A", res_id=1, res_name="ALA", atom_name="CA"),
            # Residue 2, Chain A
            struc.Atom(np.array([2, 1, 1]), chain_id="A", res_id=2, res_name="GLY", atom_name="N"),
            struc.Atom(np.array([2, 1, 2]), chain_id="A", res_id=2, res_name="GLY", atom_name="CA"),
            # Residue 3, Chain B
            struc.Atom(np.array([3, 1, 1]), chain_id="B", res_id=3, res_name="VAL", atom_name="N"),
            struc.Atom(np.array([3, 1, 2]), chain_id="B", res_id=3, res_name="VAL", atom_name="CA"),
        ]
    )


def test_get_residue_starts_basic(basic_atom_array: struc.AtomArray) -> None:
    """Test that get_residue_starts correctly identifies the start of each residue."""
    starts = get_residue_starts(basic_atom_array)
    assert len(starts) == 3
    assert list(starts) == [0, 2, 4]


def test_get_residue_starts_complex():
    # fmt: off
    atom_array = struc.array([
        struc.Atom(np.array([44.869,     8.188,    36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([45.024,     7.456,    34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([44.142,     6.714,    34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.669,     8.171,    36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.812,     8.982,    38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.152,     8.296,    39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.479,     9.3  ,    40.792 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="SD", element="16", charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.232,     8.184,    42.102 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CE", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.46 ,     8.724,    36.151 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="C",  element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.339,     9.907,    35.831 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O",  element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([58.656483, 34.763695, 36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.212917, 35.263927, 34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([60.296505, 34.87109 , 34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.271206, 33.732964, 36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([58.49736 , 33.451305, 38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.42145 , 33.22273 , 39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="2"),
    ])
    # fmt: on

    assert len(get_residue_starts(atom_array)) == 2


@pytest.mark.parametrize(
    "slice_args,expected_res_ids",
    [
        ((0, 2), [1, 1, 2, 2]),  # First two residues
        ((-2, None), [2, 2, 3, 3]),  # Last two residues
        ((1, 2), [2, 2]),  # Middle residue
        ((), [1, 1, 2, 2, 3, 3]),  # Full selection
    ],
)
def test_residx_slice(basic_atom_array: struc.AtomArray, slice_args: tuple, expected_res_ids: list[int]) -> None:
    """
    Test ResIdxSlice with various slicing parameters.

    Args:
        - basic_atom_array: Fixture providing test atom array
        - slice_args: Tuple of (start, stop) indices for slicing
        - expected_res_ids: Expected residue IDs after slicing
    """
    sliced = basic_atom_array[ResIdxSlice(*slice_args)]
    assert list(sliced.res_id) == expected_res_ids


@pytest.mark.parametrize(
    "slice_args,expected_chain_ids",
    [
        ((0, 1), ["A", "A", "A", "A"]),  # First chain
        ((0, 2), ["A", "A", "A", "A", "B", "B"]),  # First chain
        ((1, None), ["B", "B"]),  # Last chain
        ((0, 1), ["A", "A", "A", "A"]),  # Single chain
        ((), ["A", "A", "A", "A", "B", "B"]),  # Full selection
        ((None, -1), ["A", "A", "A", "A"]),  # All but last chain
    ],
)
def test_chainidx_slice(basic_atom_array: struc.AtomArray, slice_args: tuple, expected_chain_ids: list[str]) -> None:
    """
    Test ChainIdxSlice with various slicing parameters.

    Args:
        - basic_atom_array: Fixture providing test atom array
        - slice_args: Tuple of (start, stop) indices for slicing
        - expected_chain_ids: Expected chain IDs after slicing
    """
    sliced = basic_atom_array[ChainIdxSlice(*slice_args)]
    assert list(sliced.chain_id) == expected_chain_ids


def test_slice_behavior(basic_atom_array: struc.AtomArray) -> None:
    """Test slice behavior with out-of-bounds indices."""
    # Out of bounds slices should return empty arrays, not raise errors
    assert len(basic_atom_array[ResIdxSlice(10, 20)]) == 0
    assert len(basic_atom_array[ChainIdxSlice(10, 20)]) == 0

    # Negative indices should work as expected
    assert list(basic_atom_array[ResIdxSlice(-1, None)].res_id) == [3, 3]
    assert list(basic_atom_array[ChainIdxSlice(-1, None)].chain_id) == ["B", "B"]


def test_edge_cases() -> None:
    """Test edge cases with empty and single-atom arrays."""
    # Empty array
    empty_array = struc.AtomArray(0)
    assert len(empty_array[ResIdxSlice(0, 1)]) == 0
    assert len(empty_array[ChainIdxSlice(0, 1)]) == 0

    # Single atom array
    single_atom = struc.array([struc.Atom(np.array([1, 1, 1]), chain_id="A", res_id=1, res_name="ALA", atom_name="N")])
    assert len(single_atom[ResIdxSlice(0, 1)]) == 1
    assert len(single_atom[ChainIdxSlice(0, 1)]) == 1
    assert list(single_atom[ResIdxSlice(0, 1)].res_id) == [1]
    assert list(single_atom[ChainIdxSlice(0, 1)].chain_id) == ["A"]


def test_sequence_selection_init_and_repr():
    # Test valid initialization and repr
    selection = AtomSelection(chain_id="A", res_name="ARG", res_id="123", atom_name="CA")
    assert repr(selection) == "A/ARG/123/CA"

    # Test with some None values
    selection = AtomSelection(chain_id="A", res_name="ARG")
    assert repr(selection) == "A/ARG"

    # Test with * for wildcards
    selection = AtomSelection(chain_id="A", res_name="ARG", atom_name="CA")
    assert repr(selection) == "A/ARG/*/CA"


@pytest.mark.parametrize(
    "selection_string, pymol_string, expected_selection",
    [
        ("A/ARG/123/CA", "A/ARG`123/CA", AtomSelection(chain_id="A", res_name="ARG", res_id="123", atom_name="CA")),
        ("A/ARG", "A/ARG", AtomSelection(chain_id="A", res_name="ARG")),
        ("A/*/123/*", "A/*`123", AtomSelection(chain_id="A", res_id="123")),
    ],
)
def test_parse_selection_string(selection_string, pymol_string, expected_selection):
    # Test parsing using parse_selection_string and AtomSelection.from_str
    from_selection_string = parse_selection_string(selection_string)
    from_pymol_string = parse_pymol_string(pymol_string)
    assert from_selection_string == expected_selection
    assert from_pymol_string == expected_selection
    assert from_selection_string == AtomSelection.from_selection_str(selection_string)


def test_get_mask_from_selection_string(basic_atom_array: struc.AtomArray):
    # Test full match
    mask = get_mask_from_selection_string(basic_atom_array, "A/ALA/1/CA")
    expected_mask = np.array([False, True, False, False, False, False], dtype=bool)
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(mask, AtomSelection.from_selection_str("A/ALA/1/CA").get_mask(basic_atom_array))

    # Test partial match
    mask = get_mask_from_selection_string(basic_atom_array, "A/ALA")
    expected_mask = np.array([True, True, False, False, False, False], dtype=bool)
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(mask, AtomSelection.from_selection_str("A/ALA").get_mask(basic_atom_array))

    # Test no match raises ValueError
    with pytest.raises(ValueError, match="No atoms found for selection: A/VAL/1/CB"):
        get_mask_from_selection_string(basic_atom_array, "A/VAL/1/CB")


CONTIG_TEST_CASES = [
    ("A1-2", 2),
    ("A1-2, B3-3", 3),
]


@pytest.mark.parametrize("contig_test_case", CONTIG_TEST_CASES)
def test_get_mask_from_contig(contig_test_case: str):
    contig_string, expected_length = contig_test_case
    selection_stack = AtomSelectionStack.from_contig(contig_string)

    assert isinstance(selection_stack, AtomSelectionStack)
    assert len(selection_stack.selections) == expected_length


@pytest.mark.parametrize("contig_test_case", CONTIG_TEST_CASES)
def test_get_mask_from_contig_with_atom_array(basic_atom_array: struc.AtomArray, contig_test_case: str):
    contig_string, expected_length = contig_test_case
    selection_stack = AtomSelectionStack.from_contig(contig_string)
    residue_starts = get_residue_starts(basic_atom_array)
    mask = selection_stack.get_mask(basic_atom_array)

    assert isinstance(mask, np.ndarray)
    assert len(mask) == len(basic_atom_array)
    assert np.sum(mask[residue_starts]) == expected_length


def test_atom_selection_stack_get_center_of_mass(basic_atom_array: struc.AtomArray):
    """Test that get_center_of_mass returns the correct center for selected atoms."""
    selection_stack = AtomSelectionStack.from_contig("A1-2, B3-3")
    center_of_mass = selection_stack.get_center_of_mass(basic_atom_array)
    expected_center = np.mean(basic_atom_array[selection_stack.get_mask(basic_atom_array)].coord, axis=0)
    assert np.allclose(center_of_mass, expected_center)

    # test that it works with an atom array stack
    atom_array_stack = struc.stack([basic_atom_array, basic_atom_array])
    center_of_mass = selection_stack.get_center_of_mass(atom_array_stack)
    expected_center = np.mean(atom_array_stack.coord[:, selection_stack.get_mask(atom_array_stack)], axis=1)
    assert expected_center.shape == (2, 3)
    assert np.allclose(center_of_mass, expected_center)


def test_atom_selection_stack_get_principle_components(basic_atom_array: struc.AtomArray):
    """Test that get_principle_components returns correct principal axes for selected atoms."""
    selection_stack = AtomSelectionStack.from_contig("A1-2, B3-3")
    # AtomArray case
    pcs = selection_stack.get_principal_components(basic_atom_array)
    coords = basic_atom_array[selection_stack.get_mask(basic_atom_array)].coord
    coords_centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
    expected_pcs = vh.T
    assert pcs.shape == (3, 3)
    # Principal axes are unique up to sign, so compare absolute values
    assert np.allclose(np.abs(pcs), np.abs(expected_pcs))

    # AtomArrayStack case
    atom_array_stack = struc.stack([basic_atom_array, basic_atom_array])
    pcs_stack = selection_stack.get_principal_components(atom_array_stack)
    coords_stack = atom_array_stack.coord[:, selection_stack.get_mask(atom_array_stack), :]
    for i, model_coords in enumerate(coords_stack):
        model_centered = model_coords - model_coords.mean(axis=0)
        _, _, vh = np.linalg.svd(model_centered, full_matrices=False)
        expected_pcs = vh.T
        assert pcs_stack.shape == (2, 3, 3)
        assert np.allclose(np.abs(pcs_stack[i]), np.abs(expected_pcs))


def test_atom_selection_stack_from_query_ranges(basic_atom_array: struc.AtomArray) -> None:
    """Select a range of residue IDs within a chain using extended syntax."""
    selection_stack = AtomSelectionStack.from_query("A/*/1-2")
    mask = selection_stack.get_mask(basic_atom_array)
    # Expect residues 1 and 2 in chain A (first four atoms)
    expected_mask = np.array([True, True, True, True, False, False], dtype=bool)
    assert np.array_equal(mask, expected_mask)


def test_atom_selection_stack_from_query_multiple_tokens(basic_atom_array: struc.AtomArray) -> None:
    """Union of multiple selection tokens."""
    selection_stack = AtomSelectionStack.from_query(["A/ALA", "B/VAL"])  # include ALA in chain A and VAL in chain B
    mask = selection_stack.get_mask(basic_atom_array)
    # Expect ALA in chain A (first two atoms) and VAL in chain B (last two atoms)
    expected_mask = np.array([True, True, False, False, True, True], dtype=bool)
    assert np.array_equal(mask, expected_mask)


if __name__ == "__main__":
    pytest.main([__file__])
