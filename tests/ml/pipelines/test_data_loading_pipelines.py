"""Includes tests to run through the data loading pipeline to ensure examples can process without error"""

import logging

import numpy as np
import pytest

from atomworks.ml.datasets import get_row_and_index_by_example_id
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from tests.conftest import skip_if_on_github_runner

logger = logging.getLogger(__name__)


@pytest.fixture
def datasets_to_test(
    af3_pdb_dataset,
    af3_validation_dataset,
    af2_distillation_dataset_with_metadata,
    af2_distillation_dataset_no_metadata,
    rf2aa_validation_dataset,
    rf2aa_pdb_dataset,
):
    """Create the list of datasets to test with actual dataset objects."""
    return [
        {
            "dataset": af3_pdb_dataset,
            "type": "train",
            "num_examples": 5,
        },
        {
            "dataset": af3_validation_dataset,
            "type": "validation",
            "num_examples": 1,
        },
        {
            "dataset": af2_distillation_dataset_with_metadata,
            "type": "train",
            "num_examples": 1,
        },
        {
            "dataset": af2_distillation_dataset_no_metadata,
            "type": "train",
            "num_examples": 1,
        },
        {
            "dataset": rf2aa_validation_dataset,
            "type": "validation",
            "num_examples": 1,
        },
        {
            "dataset": rf2aa_pdb_dataset,
            "type": "train",
            "num_examples": 1,
        },
    ]


def identity_collate_fn(batch):
    return batch


@pytest.mark.parametrize("dataset_to_test_index", range(6))
@pytest.mark.slow
@skip_if_on_github_runner
def test_data_loading_pipelines_with_random_examples(datasets_to_test, dataset_to_test_index):
    """Test random examples using a DataLoader with basic smoke tests."""
    dataset_to_test = datasets_to_test[dataset_to_test_index]
    dataset = dataset_to_test["dataset"]
    dataset_type = dataset_to_test["type"]

    # Select deterministic examples
    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        deterministic_indices = np.random.choice(len(dataset), dataset_to_test["num_examples"], replace=False)

    for index in deterministic_indices:
        sample = dataset[index]
        example_id = sample["example_id"]

        with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
            row = get_row_and_index_by_example_id(dataset, example_id)[
                "row"
            ]  # Check if we can reverse-engineer the row from the example_id
            assert row is not None, f"Failed to get row from example_id for example_id: {example_id}"
            assert sample is not None, f"Sample is None, with example_id: {example_id}"
            assert (
                row["example_id"] == example_id
            ), f"Row example_id does not match example_id for example_id: {example_id}"

            # For validation datasets, also check that the "ground_truth" key contains information on which chains/interfaces to score, and the map from token index to `chain_iid`
            if dataset_type == "validation":
                assert "ground_truth" in sample, f"Missing 'ground_truth' key in sample with example_id: {example_id}"
                assert (
                    "chain_iid_token_lvl" in sample["ground_truth"]
                ), f"Missing 'chain_iid_token_lvl' key in sample with example_id: {example_id}"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", "-m slow", __file__])
