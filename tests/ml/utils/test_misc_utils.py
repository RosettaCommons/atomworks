import pytest
import torch

from atomworks.ml.utils.misc import grouped_sum, masked_mean

# Test cases for grouped_sum
GROUPED_SUM_TEST_CASES = [
    {
        "data": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "assignment": torch.tensor([0, 1, 0, 1]),
        "num_groups": 2,
        "expected_output": torch.tensor([[6, 8], [10, 12]]),
    },
    {
        "data": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "assignment": torch.tensor([0, 0, 0, 0]),
        "num_groups": 1,
        "expected_output": torch.tensor([[16, 20]]),
    },
    {
        "data": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
        "assignment": torch.tensor([0, 1, 0, 1]),
        "num_groups": 2,
        "expected_output": torch.tensor([[[10, 12], [14, 16]], [[18, 20], [22, 24]]]),
    },
]


@pytest.mark.parametrize("test_case", GROUPED_SUM_TEST_CASES)
def test_grouped_sum(test_case):
    output = grouped_sum(test_case["data"], test_case["assignment"], test_case["num_groups"])
    assert torch.equal(
        output, test_case["expected_output"]
    ), f"Expected {test_case['expected_output']}, but got {output}"


# Test cases for masked_mean
MASKED_MEAN_TEST_CASES = [
    {
        "mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.float32),
        "value": torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        "axis": 0,
        "drop_mask_channel": False,
        "expected_output": torch.tensor([3.0, 5.0]),
    },
    {
        "mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.float32),
        "value": torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        "axis": None,
        "drop_mask_channel": False,
        "expected_output": torch.tensor((2 / 3) * 4 + (1 / 3) * 3),
    },
    {
        "mask": torch.tensor([[[1], [0]], [[1], [1]]], dtype=torch.float32),
        "value": torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        "axis": 0,
        "drop_mask_channel": True,
        "expected_output": torch.tensor([3.0, 5.0]),
    },
]


@pytest.mark.parametrize("test_case", MASKED_MEAN_TEST_CASES)
def test_masked_mean(test_case):
    output = masked_mean(
        mask=test_case["mask"],
        value=test_case["value"],
        axis=test_case["axis"],
        drop_mask_channel=test_case["drop_mask_channel"],
    )
    assert torch.allclose(
        output, test_case["expected_output"], atol=1e-6
    ), f"Expected {test_case['expected_output']}, but got {output}"
