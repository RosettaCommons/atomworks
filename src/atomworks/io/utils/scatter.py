from collections.abc import Callable

import biotite.structure as struc
import numpy as np
from jaxtyping import Int, Shaped

Array = np.ndarray


# Group-wise operations (= non-contiguous or contiguous groups) ------------------------------
def get_groups(*arrays: np.ndarray) -> np.ndarray:
    """
    Get group indices for where the given arrays all agree.

    For a set of arrays of the same length, this function produces a 1D array of integer group indices. A
    group is defined by a unique combination of values at each position across the input arrays.

    For example, if `arrays` are `(a, b)`, then position `i` belongs to the same group as position `j` if
    and only if `a[i] == a[j]` and `b[i] == b[j]`.

    Args:
        *arrays: A variable number of 1D NumPy arrays of the same length.

    Returns:
        np.ndarray: A 1D NumPy array of integer group indices. The indices are assigned based on the
            lexicographical order of the unique value combinations.

    Raises:
        ValueError: If no arrays are provided, or if the arrays are not all 1D and of the same length.

    Example:
        >>> a = np.array([1, 1, 2, 2, 1])
        >>> b = np.array(["x", "y", "x", "y", "x"])
        >>> # Unique combinations are (1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')
        >>> # which are assigned indices 0, 1, 2, 3 respectively.
        >>> get_groups(a, b)
        array([0, 1, 2, 3, 0])

        >>> c = np.array([10, 20, 10, 20, 10])
        >>> # Unique combinations are (1, 'x', 10), (1, 'y', 20), (2, 'x', 10), (2, 'y', 20), (1, 'x', 10)
        >>> # which are assigned indices 0, 1, 2, 3 respectively
        >>> get_groups(a, b, c)
        array([0, 1, 2, 3, 0])

        >>> # For a single array, it's equivalent to finding unique values
        >>> d = np.array([10, 20, 10, 30])
        >>> get_groups(d)
        array([0, 1, 0, 2])
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")

    first_len = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.ndim != 1:
            raise ValueError(f"All arrays must be 1D, but array {i} has {arr.ndim} dimensions.")
        if len(arr) != first_len:
            raise ValueError(
                f"All arrays must have the same length, but array 0 has length {first_len} and array {i} has length {len(arr)}."
            )

    if len(arrays) == 1:
        _, group_indices = np.unique(arrays[0], return_inverse=True)
        return group_indices

    # For multiple arrays, we stack them to create an array where each row represents
    # the combination of values at that position. `np.unique` with `axis=0`
    # can then find the unique rows, and `return_inverse=True` gives us the
    # desired group indices.
    stacked_arrays = np.stack(arrays, axis=-1)
    _, group_indices = np.unique(stacked_arrays, axis=0, return_inverse=True)

    return group_indices


def apply_group_wise(group: np.ndarray, data: np.ndarray, func: Callable) -> np.ndarray:
    """
    Apply a reduction function to data grouped by group indices.

    This function performs a scatter-gather operation: it groups `data` by the values in `group` and applies
    the provided reduction function `func` to each group. The reduction function should return a scalar or
    a single array per group.

    Args:
        group: 1D array of group indices or labels. Must have the same length as the first dimension of `data`.
        data: Array of data to aggregate. If 1D, each value is assigned to a group; if 2D or higher, each row
            is assigned to a group.
        func: Function to apply to each group's data. Should accept a 1D or 2D array and return a scalar or
            array per group.

    Returns:
        np.ndarray: 1D array of aggregated values, one per unique group, ordered by sorted group label.

    Raises:
        ValueError: If `group` is not 1D or its length does not match `data.shape[0]`.

    Example:
        >>> groups = np.array([0, 1, 0, 2, 1])
        >>> data = np.array([10, 20, 30, 40, 50])
        >>> apply_group_wise(groups, data, np.sum)
        array([40, 70, 40])
    """
    # Sort the groups into adjacent segments
    sort_idx = group.argsort(kind="mergesort")

    # Apply the function to each segment
    segments = get_segments(group[sort_idx], add_exclusive_stop=True)
    return apply_segment_wise(segments, data[sort_idx], func)


def spread_group_wise(group: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Spread aggregated group data back to original positions.

    Given a 1D array of group indices and a 1D array of values (one per unique group), this function
    broadcasts the group value to all positions in the original array according to group membership.

    Args:
        group: 1D array of group indices or labels. Length N.
        data: 1D array of values, one per unique group. The order must match `np.unique(group)`.

    Returns:
        np.ndarray: 1D array of length N, where each position contains the value for its group.

    Raises:
        ValueError: If `group` or `data` are not 1D.

    Example:
        >>> group = np.array([0, 1, 0, 2, 1])
        >>> data = np.array([100, 200, 300])
        >>> spread_group_wise(group, data)
        array([100, 200, 100, 300, 200])
    """
    unique_groups, inverse_indices = np.unique(group, return_inverse=True)
    if len(unique_groups) != len(data):
        raise ValueError("Data length must match number of unique groups.")
    return data[inverse_indices]


def apply_and_spread_group_wise(group: np.ndarray, data: np.ndarray, func: Callable) -> np.ndarray:
    """
    Apply a group-wise reduction and broadcast the result back to all original positions.

    This function first aggregates `data` by `group` using `func` (see `apply_group_wise`), then
    spreads the aggregated values back to the original array shape so that each element receives
    the value for its group.

    Args:
        group: 1D array of group indices or labels.
        data: Data array to aggregate.
        func: Reduction function to apply to each group.

    Returns:
        np.ndarray: 1D array of aggregated values, broadcast to original positions.

    Example:
        >>> group = np.array([0, 1, 0, 2, 1])
        >>> data = np.array([10, 20, 30, 40, 50])
        >>> apply_and_spread_group_wise(group, data, np.mean)
        array([20., 35., 20., 40., 35.])
    """
    aggregated = apply_group_wise(group, data, func)
    return spread_group_wise(group, aggregated)


def safe_scatter(
    source_values: Shaped[Array, "s *rest"],  # noqa: F821
    mapping: Int[Array, "s"],  # noqa: F821
    target: Shaped[Array, "t *rest"],  # noqa: F821
) -> Shaped[Array, "t *rest"]:  # noqa: F821
    """Scatter ``source_values`` onto a copy of ``target``: source ``i`` writes to ``target[mapping[i]]``,
    ``mapping[i] == -1`` drops it. A string ``target`` dtype is widened so long labels are not truncated.

    >>> safe_scatter(np.array([10, 20, 30]), np.array([2, -1, 0]), np.zeros(4, dtype=int))
    array([30,  0, 10,  0])
    """
    if len(source_values) != len(mapping):
        raise ValueError(
            f"source_values ({len(source_values)}) and mapping ({len(mapping)}) must have the same length."
        )
    if target.dtype.kind in "US":
        out = target.astype(np.result_type(target.dtype, np.asarray(source_values).dtype))
    else:
        out = target.copy()
    matched = mapping >= 0
    out[mapping[matched]] = source_values[matched]
    return out


# Segment-wise operations (= contiguous groups) --------------------------------
def get_segments(*arrays: np.ndarray, add_exclusive_stop: bool = False) -> np.ndarray:
    """
    Compute segment boundaries where any of the input arrays change value.

    For a set of 1D arrays of the same length, this function returns the indices where the value of any array
    changes from one position to the next. The result is an array of start indices for each segment, and
    optionally includes an exclusive stop index at the end.

    Args:
        *arrays: One or more 1D NumPy arrays of the same length. Segments are defined by changes in any array.
        add_exclusive_stop: If True, append the exclusive stop index (length of the arrays) to the result.

    Returns:
        np.ndarray: 1D array of segment start indices. If `add_exclusive_stop` is True, the last element is the
            exclusive stop index (i.e., the length of the arrays).

    Raises:
        ValueError: If the input arrays are not all the same length.

    Example:
        >>> a = np.array([1, 1, 2, 2, 1])
        >>> b = np.array([0, 0, 0, 1, 1])
        >>> get_segments(a, b, add_exclusive_stop=True)
        array([0, 2, 3, 4, 5])
    """
    lengths = [array.shape[0] for array in arrays]
    length = lengths[0]
    if not all(length == length_ for length_ in lengths):
        raise ValueError("All arrays must have the same length")

    if length == 0:
        return np.array([], dtype=int)

    arrays_differ = np.zeros(length - 1, dtype=bool)
    for array in arrays:
        arrays_differ |= array[1:] != array[:-1]

    start_stop_idxs = np.where(arrays_differ)[0] + 1

    if add_exclusive_stop:
        return np.concatenate(([0], start_stop_idxs, [length]))
    return np.concatenate(([0], start_stop_idxs))


apply_segment_wise = struc.segments.apply_segment_wise
spread_segment_wise = struc.segments.spread_segment_wise

NUMPY_REDUCE_FUNCS = {
    np.any: np.logical_or.reduceat,
    np.all: np.logical_and.reduceat,
    np.maximum: np.maximum.reduceat,
    np.minimum: np.minimum.reduceat,
    np.max: np.maximum.reduceat,
    np.min: np.minimum.reduceat,
    np.add: np.add.reduceat,
    np.sum: np.add.reduceat,
}


def apply_and_spread_segment_wise(
    segment: np.ndarray,
    data: np.ndarray,
    func: Callable,
    axis: int | None = None,
) -> np.ndarray:
    """Apply a segment-wise reduction and broadcast the result back to all original positions.

    Uses :py:obj:`NUMPY_REDUCE_FUNCS` for common numpy ufuncs when ``axis`` is ``None``.

    Args:
        segment: 1D array of segment start/stop indices (with exclusive stop appended).
        data: The input data array.
        func: The function to apply to each segment.
        axis: The axis along which to apply the function. If ``None``, the function
            is applied to the entire segment.

    Returns:
        A new array with the same shape as ``data``, where the result of the
        function applied to each segment has been spread across the elements
        of that segment.

    Example:
        >>> import numpy as np
        >>> segment = np.array([0, 3, 6])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> apply_and_spread_segment_wise(segment, data, np.sum)
        array([ 6,  6,  6, 15, 15, 15])
    """
    if func in NUMPY_REDUCE_FUNCS and axis is None:
        aggregated = NUMPY_REDUCE_FUNCS[func](data, segment[:-1])
    else:
        aggregated = apply_segment_wise(segment, data, func, axis)
    return spread_segment_wise(segment, aggregated)
