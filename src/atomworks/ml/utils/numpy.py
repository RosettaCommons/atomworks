"""General utility functions for working with numpy arrays."""

import networkx as nx
import numpy as np


def select_data_by_id(
    select_ids: np.ndarray,
    data_ids: np.ndarray,
    data: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Select data from an array based on matching IDs.

    Args:
        select_ids (np.ndarray): Array of IDs to select.
        data_ids (np.ndarray): Array of IDs corresponding to the data.
        data (np.ndarray): Data array from which to select.
        axis (int, optional): Axis along which to select data. Defaults to 0.

    Returns:
        np.ndarray: Array of selected data.

    Raises:
        AssertionError: If the shape of `data` along `axis` does not match the length of `data_ids`.
        AssertionError: If `data_ids` contains duplicate values.

    Example:
        >>> to_ids = np.array([1, 5, 2, 20, 20, 2])
        >>> from_array = np.arange(10).repeat(6).reshape(10, 6)
        >>> from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])
        >>> select_data_by_id(to_ids, from_ids, from_array)
        array([[0., 0., 0., 0., 0., 0.],
               [2., 2., 2., 2., 2., 2.],
               [1., 1., 1., 1., 1., 1.],
               [8., 8., 8., 8., 8., 8.],
               [8., 8., 8., 8., 8., 8.],
               [1., 1., 1., 1., 1., 1.]])
    """
    assert data.shape[axis] == len(
        data_ids
    ), f"`data` must have `len(data_ids)` along axis `{axis}`, but got shape: {data.shape}"
    assert np.unique(data_ids).size == len(
        data_ids
    ), f"`data_ids` must be unique. Got duplicates ({np.unique(data_ids)}) in {data_ids}"

    id_to_idx = np.vectorize({id_: idx for idx, id_ in enumerate(data_ids)}.__getitem__, otypes=[data_ids.dtype])
    idxs_to_select = id_to_idx(select_ids)
    return np.take(data, idxs_to_select, axis=axis)


def insert_data_by_id_(
    to_fill: np.ndarray,
    to_fill_ids: np.ndarray,
    from_data: np.ndarray,
    from_data_ids: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Insert data into an array based on matching IDs.

    Args:
        to_fill (np.ndarray): Array to be filled.
        to_fill_ids (np.ndarray): Array of IDs corresponding to `to_fill`.
        from_data (np.ndarray): Data array from which to insert.
        from_data_ids (np.ndarray): Array of IDs corresponding to `from_data`.
        axis (int, optional): Axis along which to insert data. Defaults to 0.

    Returns:
        np.ndarray: Array with inserted data.

    Example:
        >>> to_array = np.zeros((7, 6))
        >>> to_ids = np.array([1, 5, 2, 17, 20, 20, 2])
        >>> from_array = np.arange(10).repeat(6).reshape(10, 6)
        >>> from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])
        >>> insert_data_by_id_(to_array, to_ids, from_array, from_ids)
        >>> print(to_array)
        array([[0., 0., 0., 0., 0., 0.],
               [2., 2., 2., 2., 2., 2.],
               [1., 1., 1., 1., 1., 1.],
               [0., 0., 0., 0., 0., 0.],
               [8., 8., 8., 8., 8., 8.],
               [8., 8., 8., 8., 8., 8.],
               [1., 1., 1., 1., 1., 1.]])
    """
    is_available = np.isin(to_fill_ids, from_data_ids)
    to_fill[is_available] = select_data_by_id(to_fill_ids[is_available], from_data_ids, from_data, axis=axis)


def unique_by_first_occurrence(arr: np.ndarray) -> np.ndarray:
    """
    Return unique elements of an array while preserving the order of their first occurrence.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Array of unique elements in the order of their first occurrence.

    Example:
        >>> unique_by_first_occurrence(np.array([4, 2, 2, 3, 1, 4]))
        array([4, 2, 3, 1])
    """
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx, kind="mergesort")]


def is_mask_contiguous(mask: np.ndarray) -> bool:
    """Check if a mask is contiguous."""
    if np.all(mask == mask[0]):
        return True
    mask_with_two_falses = np.concatenate([[False], mask, [False]]).astype(np.int8)
    diffs = np.diff(mask_with_two_falses)
    vals, counts = np.unique(diffs, return_counts=True)
    return np.array_equal(vals, [-1, 0, 1]) and counts[0] == 1 and counts[2] == 1


def get_connected_components_from_adjacency(adjacency: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of indices for each connected component according
    to the given adjacency matrix.
    """
    graph = nx.from_numpy_array(adjacency)
    return [np.array(list(component)) for component in nx.connected_components(graph)]


def not_isin(element: np.ndarray, test_element: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating where elements of `element` are not in `test_element`.

    Args:
        element (np.ndarray): Array to check.
        test_element (np.ndarray): Array to check against.

    Returns:
        np.ndarray: Boolean mask.

    Example:
        >>> not_isin(np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6]))
        array([ True, False,  True, False,  True])
    """
    return np.isin(element, test_element, invert=True)


def get_nearest_true_index_for_each_false(arr: np.ndarray) -> np.ndarray:
    """
    Get the index of the nearest True for each False in the array, breaking
    ties by choosing the nearest True to the left.

    Args:
        - arr (np.ndarray): A boolean numpy array.

    Returns:
        - np.ndarray: An array of length `np.sum(~arr)` where each entry is the index of the nearest True.

    Example:
        >>> arr = np.array([False, True, True, False, False, True, False])
        >>> get_nearest_true_index_for_each_false(arr)
        array([1, 2, 5, 5])
    """

    # ...find the indices where the values are True and False
    true_indices = np.where(arr)[0]
    false_indices = np.where(~arr)[0]

    # Short-circuit if there are no True entries or no False entries, as we can't proceed
    if len(true_indices) == 0 or len(false_indices) == 0:
        return np.array([])

    # ...for False entries, find the index of the nearest True

    # Calculate distances to the nearest True indices
    # Using broadcasting to calculate the distance matrix (e.g., outer difference)
    # i,j entry of the distance matrix is the distance between the i-th False and j-th True
    distances = np.abs(false_indices[:, np.newaxis] - true_indices)

    # Use argmin to find the index of the minimum distance
    # np.argmin will automatically break ties by choosing the first occurrence
    nearest_true_indices = true_indices[np.argmin(distances, axis=1)]

    return nearest_true_indices


def get_indices_of_non_constant_columns(arr: np.ndarray) -> np.ndarray:
    """Identify columns where values change between consecutive rows.

    Args:
        arr (np.ndarray): A 2D NumPy array where you want to find columns with changing values.

    Returns:
        np.ndarray: An array of column indices where values change between consecutive rows.

    Example:
        >>> arr = np.array(
        ...     [
        ...         [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161],
        ...         [151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 160],
        ...     ]
        ... )
        >>> find_changing_columns(arr)
        array([ 9, 10])
    """
    # Compute the differences between consecutive rows
    differences = np.diff(arr, axis=0)

    # Get the indices where the differences are non-zero
    changing_indices = np.nonzero(differences)

    # Return the column indices where changes occur
    return changing_indices[1]
