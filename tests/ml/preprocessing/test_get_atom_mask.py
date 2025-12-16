import biotite.structure as struc
import numpy as np
import pytest

from atomworks.ml.preprocessing.utils.structure_utils import get_atom_mask_from_cell_list


@pytest.fixture
def mock_data():
    query_coords = np.random.rand(200, 3) * 3
    atom_array = np.random.rand(3000, 3) * 3
    cell_list_size = atom_array.shape[0]
    clash_distance = 1.5
    return query_coords, atom_array, cell_list_size, clash_distance


def test_get_atom_mask_from_cell_list(mock_data):
    """
    Test get_atom_mask_from_cell_list function with various edge cases.
    """

    query_coords, atom_array, cell_list_size, clash_distance = mock_data
    cell_list = struc.CellList(atom_array, cell_size=1.0)

    # No chunking needed
    chunk_size = query_coords.shape[0] * atom_array.shape[0] + 1  # Ensure no chunking is applied
    result = get_atom_mask_from_cell_list(query_coords, cell_list, cell_list_size, clash_distance, chunk_size)
    expected = cell_list.get_atoms(query_coords, clash_distance, as_mask=True)
    assert np.array_equal(result, expected), "Result with no chunking does not match expected output."

    # Chunking applied
    chunk_size = (query_coords.shape[0] * atom_array.shape[0]) // 10  # Ensure chunking is applied
    result = get_atom_mask_from_cell_list(query_coords, cell_list, cell_list_size, clash_distance, chunk_size)
    expected = cell_list.get_atoms(query_coords, clash_distance, as_mask=True)
    assert np.array_equal(result, expected), "Result with chunking does not match expected output."

    # max_rows_per_chunk = 1
    chunk_size = cell_list_size - 1  # Ensure max_rows_per_chunk is 1
    result = get_atom_mask_from_cell_list(query_coords, cell_list, cell_list_size, clash_distance, chunk_size=1)
    expected = cell_list.get_atoms(query_coords, clash_distance, as_mask=True)
    assert np.array_equal(result, expected), "Result with max_rows_per_chunk set to 1 does not match expected output."

    # Empty coord array
    empty_query_coords = np.empty((0, 3))
    result = get_atom_mask_from_cell_list(
        empty_query_coords, cell_list, cell_list_size, clash_distance, chunk_size=1000
    )
    expected = np.zeros((0, cell_list_size), dtype=bool)
    assert len(result) == len(expected), "Result with empty coord does not match expected output."

    # Very small chunk_size
    chunk_size = 1  # Set chunk_size to a very small value
    result = get_atom_mask_from_cell_list(query_coords, cell_list, cell_list_size, clash_distance, chunk_size)
    expected = cell_list.get_atoms(query_coords, clash_distance, as_mask=True)
    assert np.array_equal(result, expected), "Result with very small chunk_size does not match expected output."

    # Large coord array with small chunk_size
    large_query_coords = np.random.rand(10000, 3) * 3  # Large coord array
    chunk_size = 1000  # Small chunk_size
    result = get_atom_mask_from_cell_list(large_query_coords, cell_list, cell_list_size, clash_distance, chunk_size)
    expected = cell_list.get_atoms(large_query_coords, clash_distance, as_mask=True)
    assert np.array_equal(
        result, expected
    ), "Result with large coord array and small chunk_size does not match expected output."

    # Non-divisible chunk_size
    chunk_size = 1507  # Non-divisible chunk_size
    result = get_atom_mask_from_cell_list(query_coords, cell_list, cell_list_size, clash_distance, chunk_size)
    expected = cell_list.get_atoms(query_coords, clash_distance, as_mask=True)
    assert np.array_equal(result, expected), "Result with non-divisible chunk_size does not match expected output."
