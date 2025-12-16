import numpy as np
import pytest

from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.testing import assert_same_atom_array, is_same_in_group, is_same_in_segment
from tests.io.conftest import TEST_DATA_IO


@pytest.fixture
def atom_array():
    return load_any(
        TEST_DATA_IO / "6lyz.bcif", model=1, extra_fields=["charge", "b_factor", "occupancy"], include_bonds=True
    )


# ... test that we can detect changes in charges
@pytest.mark.parametrize(
    "annotation, change_value",
    [
        ("charge", 1),
        ("b_factor", 0.5),
        ("occupancy", 0.1),
        ("atom_name", "BLA"),
        ("element", "X"),
        ("chain_id", "?"),
        ("res_name", "###"),
        ("res_id", 999),
    ],
)
def test_annotations_change(atom_array, annotation, change_value):
    atom_array2 = atom_array.copy()
    assert_same_atom_array(atom_array, atom_array2)
    atom_array2.get_annotation(annotation)[0] = change_value
    with pytest.raises(AssertionError):
        assert_same_atom_array(atom_array, atom_array2)

    annotations = atom_array.get_annotation_categories()
    annotations_to_compare = [annot for annot in annotations if annot != annotation]
    assert_same_atom_array(atom_array, atom_array2, annotations_to_compare=annotations_to_compare)


def test_bonds_change(atom_array):
    atom_array2 = atom_array.copy()
    assert_same_atom_array(atom_array, atom_array2)
    atom_array2.bonds.add_bond(0, 1)
    with pytest.raises(AssertionError):
        assert_same_atom_array(atom_array, atom_array2)
    assert_same_atom_array(atom_array, atom_array2, compare_bonds=False)


def test_coords_change(atom_array):
    atom_array2 = atom_array.copy()
    assert_same_atom_array(atom_array, atom_array2)
    atom_array2.coord[0] = atom_array2.coord[0] + 1
    with pytest.raises(AssertionError):
        assert_same_atom_array(atom_array, atom_array2)
    assert_same_atom_array(atom_array, atom_array2, compare_coords=False)


def test_atom_array_length_change(atom_array):
    atom_array2 = atom_array.copy()
    assert_same_atom_array(atom_array, atom_array2)
    atom_array2 = atom_array2[:-1]
    with pytest.raises(AssertionError):
        assert_same_atom_array(atom_array, atom_array2)


def test_scrambled_order(atom_array):
    atom_array2 = atom_array.copy()
    assert_same_atom_array(atom_array, atom_array2)
    # Swap first two atoms
    swap_first_two = np.arange(len(atom_array))
    swap_first_two[0], swap_first_two[1] = swap_first_two[1], swap_first_two[0]
    atom_array2 = atom_array2[swap_first_two]
    with pytest.raises(AssertionError):
        assert_same_atom_array(atom_array, atom_array2, enforce_order=True, compare_coords=False)
    assert_same_atom_array(atom_array, atom_array2, enforce_order=False, compare_coords=False)


def test_is_same_in_segment():
    # Test with simple segments where all elements in each segment are the same
    groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1
    segment_start_stop = np.array([0, 3, 6, 8])  # 3 segments
    data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])  # Each segment has same value
    result = is_same_in_segment(segment_start_stop, data)
    assert np.array_equal(
        result, np.array([True, True, True])
    ), f"Failed simple segments test. Expected {np.array([True, True, True])}, got {result}"
    result_group = is_same_in_group(groups, data)
    assert np.array_equal(result_group, np.array([True, True, True]))

    data = np.array([True, True, True, False, False, False, True, True, True])
    result = is_same_in_segment(segment_start_stop, data)
    assert np.array_equal(result, np.array([True, True, True]))
    result_group = is_same_in_group(groups, data)
    assert np.array_equal(result_group, np.array([True, True, True]))

    data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    result = is_same_in_segment(segment_start_stop, data)
    assert np.array_equal(result, np.array([True, True, True]))
    result_group = is_same_in_group(groups, data)
    assert np.array_equal(result_group, np.array([True, True, True]))

    # Test with segments where not all elements are the same
    data = np.array([1, 1, 1, 2, 3, 2, 3, 3, 3])  # Middle segment has different values
    result = is_same_in_segment(segment_start_stop, data)
    assert np.array_equal(result, np.array([True, False, True]))
    result_group = is_same_in_group(groups, data)
    assert np.array_equal(result_group, np.array([True, False, True]))

    # Test with single-element segments
    segment_start_stop = np.array([0, 1, 2, 3])
    groups = np.array([0, 1, 2])
    data = np.array([1, 2, 3])
    result = is_same_in_segment(segment_start_stop, data)
    assert np.array_equal(result, np.array([True, True, True]))
    result_group = is_same_in_group(groups, data)
    assert np.array_equal(result_group, np.array([True, True, True]))


if __name__ == "__main__":
    pytest.main([__file__])
