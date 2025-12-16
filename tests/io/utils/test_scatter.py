import numpy as np
import pytest

from atomworks.io.utils.scatter import (
    apply_and_spread_group_wise,
    apply_and_spread_segment_wise,
    apply_group_wise,
    get_groups,
    get_segments,
    spread_group_wise,
)


def test_get_groups_single_array():
    d = np.array([10, 20, 10, 30])
    result = get_groups(d)
    expected = np.array([0, 1, 0, 2])
    assert np.array_equal(result, expected), f"Failed single array test. Expected {expected}, got {result}"


def test_get_groups_multiple_arrays():
    a = np.array([1, 1, 2, 2, 1])
    b = np.array(["x", "y", "x", "y", "x"])
    result = get_groups(a, b)
    expected = np.array([0, 1, 2, 3, 0])
    assert np.array_equal(result, expected), f"Failed multiple array test. Expected {expected}, got {result}"


def test_get_groups_more_multiple_arrays():
    a = np.array([1, 1, 2, 2, 1])
    b = np.array(["x", "y", "x", "y", "x"])
    c = np.array([10, 20, 10, 20, 10])
    result = get_groups(a, b, c)
    expected = np.array([0, 1, 2, 3, 0])
    assert np.array_equal(result, expected), f"Failed more multiple array test. Expected {expected}, got {result}"


def test_get_groups_empty_input():
    result = get_groups(np.array([]))
    expected = np.array([], dtype=int)
    assert np.array_equal(result, expected), f"Failed empty array test. Expected {expected}, got {result}"


def test_apply_group_wise_1d_sum():
    groups = np.array([0, 1, 0, 2, 1])
    data = np.array([10, 20, 30, 40, 50])
    result = apply_group_wise(groups, data, np.sum)
    expected = np.array([40, 70, 40])
    assert np.array_equal(result, expected), f"Failed 1D sum test. Expected {expected}, got {result}"


def test_apply_group_wise_1d_mean():
    groups = np.array([0, 1, 0, 2, 1])
    data = np.array([10, 20, 30, 40, 50])
    result = apply_group_wise(groups, data, np.mean)
    expected = np.array([20.0, 35.0, 40.0])
    assert np.allclose(result, expected), f"Failed 1D mean test. Expected {expected}, got {result}"


def test_apply_group_wise_2d_sum():
    groups = np.array([0, 1, 0, 1])
    data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = apply_group_wise(groups, data, lambda arr: np.sum(arr, axis=0))
    expected = np.array([[4, 4], [6, 6]])
    assert np.array_equal(
        np.stack(result), expected
    ), f"Failed 2D sum test. Expected {expected}, got {np.stack(result)}"


def test_spread_group_wise_basic():
    group = np.array([0, 1, 0, 2, 1])
    data = np.array([100, 200, 300])
    result = spread_group_wise(group, data)
    expected = np.array([100, 200, 100, 300, 200])
    assert np.array_equal(result, expected), f"Failed basic spread test. Expected {expected}, got {result}"


def test_spread_group_wise_unordered():
    group = np.array([10, 20, 10, 30, 20])
    data = np.array([1.0, 2.0, 3.0])
    result = spread_group_wise(group, data)
    expected = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    assert np.array_equal(result, expected), f"Failed unordered groups spread test. Expected {expected}, got {result}"


def test_apply_and_spread_group_wise():
    group = np.array([0, 1, 0, 2, 1])
    data = np.array([10, 20, 30, 40, 50])
    result = apply_and_spread_group_wise(group, data, np.mean)
    expected = np.array([20.0, 35.0, 20.0, 40.0, 35.0])
    assert np.allclose(result, expected), f"Failed apply and spread mean test. Expected {expected}, got {result}"


def test_get_segments_docstring():
    a = np.array([1, 1, 2, 2, 1])
    b = np.array([0, 0, 0, 1, 1])
    result = get_segments(a, b, add_exclusive_stop=True)
    result_groups = get_groups(a, b)
    result_from_groups = get_segments(result_groups, add_exclusive_stop=True)
    expected = np.array([0, 2, 3, 4, 5])
    assert np.array_equal(result, expected), f"Failed docstring test. Expected {expected}, got {result}"
    assert np.array_equal(
        result_from_groups, expected
    ), f"Failed docstring test. Expected {expected}, got {result_from_groups}"


def test_get_segments_single_array():
    a = np.array([1, 1, 2, 2, 1])
    result = get_segments(a, add_exclusive_stop=True)
    result_groups = get_groups(a)
    result_from_groups = get_segments(result_groups, add_exclusive_stop=True)
    expected = np.array([0, 2, 4, 5])
    assert np.array_equal(result, expected), f"Failed single array test. Expected {expected}, got {result}"
    assert np.array_equal(
        result_from_groups, expected
    ), f"Failed single array test. Expected {expected}, got {result_from_groups}"


def test_get_segments_no_changes():
    a = np.array([1, 1, 1, 1, 1])
    result = get_segments(a, add_exclusive_stop=True)
    result_groups = get_groups(a)
    result_from_groups = get_segments(result_groups, add_exclusive_stop=True)
    expected = np.array([0, 5])
    assert np.array_equal(result, expected), f"Failed no-change test. Expected {expected}, got {result}"
    assert np.array_equal(
        result_from_groups, expected
    ), f"Failed no-change test. Expected {expected}, got {result_from_groups}"


def test_get_segments_all_changes():
    a = np.array([1, 2, 3, 4, 5])
    result = get_segments(a, add_exclusive_stop=True)
    result_groups = get_groups(a)
    result_from_groups = get_segments(result_groups, add_exclusive_stop=True)
    expected = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(result, expected), f"Failed all-changes test. Expected {expected}, got {result}"
    assert np.array_equal(
        result_from_groups, expected
    ), f"Failed all-changes test. Expected {expected}, got {result_from_groups}"


def test_apply_and_spread_segment_wise_sum_and_mean():
    data = np.array([10, 20, 30, 40, 50, 60])
    groups = np.array([0, 0, 1, 1, 1, 2])
    segments = get_segments(groups, add_exclusive_stop=True)

    # Test with np.sum
    result_sum = apply_and_spread_segment_wise(segments, data, np.sum)
    result_sum_groups = apply_and_spread_group_wise(groups, data, np.sum)
    expected_sum = np.array([30, 30, 120, 120, 120, 60])
    assert np.array_equal(result_sum, expected_sum), f"Failed sum test. Expected {expected_sum}, got {result_sum}"
    assert np.array_equal(
        result_sum_groups, expected_sum
    ), f"Failed sum test. Expected {expected_sum}, got {result_sum_groups}"

    # Test with np.mean
    result_mean = apply_and_spread_segment_wise(segments, data, np.mean)
    result_mean_groups = apply_and_spread_group_wise(groups, data, np.mean)
    expected_mean = np.array([15.0, 15.0, 40.0, 40.0, 40.0, 60.0])
    assert np.allclose(result_mean, expected_mean), f"Failed mean test. Expected {expected_mean}, got {result_mean}"
    assert np.allclose(
        result_mean_groups, expected_mean
    ), f"Failed mean test. Expected {expected_mean}, got {result_mean_groups}"


if __name__ == "__main__":
    pytest.main([__file__])
