import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import SequentialSampler, WeightedRandomSampler

from atomworks.ml.datasets.datasets import (
    ConcatDatasetWithID,
    PandasDataset,
    get_row_and_index_by_example_id,
)
from atomworks.ml.samplers import (
    MixedSampler,
    calculate_weights_for_pdb_dataset_df,
)


def create_dummy_dataset(length: int, name: str, dataset_class: PandasDataset = PandasDataset):
    data = pd.DataFrame(
        {
            "example_id": [f"example_{np.random.randint(0, 1_000_000_000)}" for _ in range(length)],
            "col_1": [np.random.randint(0, 100) for _ in range(length)],
            "col_2": [np.random.randint(0, 100) for _ in range(length)],
        }
    )
    data.attrs = {"base_path": "/example/base/path"}
    return dataset_class(data=data, name=name, id_column="example_id")


def test_nested_dummy_datasets():
    dataset1 = create_dummy_dataset(length=10, name="1")
    dataset2 = create_dummy_dataset(length=10, name="2")
    dataset3 = create_dummy_dataset(length=10, name="3")
    dataset4 = create_dummy_dataset(length=10, name="4")
    dataset5 = create_dummy_dataset(length=10, name="5")

    dataset_1_2 = ConcatDatasetWithID(datasets=[dataset1, dataset2])
    assert len(dataset_1_2) == 20

    dataset_3_4_5 = ConcatDatasetWithID(datasets=[dataset3, dataset4, dataset5])
    assert len(dataset_3_4_5) == 30

    dataset_1_2_3_4_5 = ConcatDatasetWithID(datasets=[dataset_1_2, dataset_3_4_5])
    assert len(dataset_1_2_3_4_5) == 50

    for idx in range(len(dataset_1_2_3_4_5)):
        _id = dataset_1_2_3_4_5.idx_to_id(idx)
        _idx = dataset_1_2_3_4_5.id_to_idx(_id)
        assert idx == _idx, f"idx: {idx}, _idx: {_idx}"

        row_and_idx = get_row_and_index_by_example_id(dataset_1_2_3_4_5, _id)
        row = row_and_idx["row"]
        _idx = row_and_idx["index"]
        assert all(row == dataset_1_2_3_4_5[idx])
        assert _idx == idx
        assert row.attrs["base_path"] is not None


def test_nested_datasets_with_weighted_samplers(rf2aa_interfaces_dataset, rf2aa_pn_units_dataset, rf2aa_pdb_dataset):
    # +------------------  Sampler ------------------+
    num_examples_per_epoch = 100

    # ... calculate the weights based on the AF-3 weighting methodology
    b_pn_unit = 0.5  # β_chain
    b_interface = 0.5  # β_interface
    alphas = {
        "a_prot": 3.0,
        "a_nuc": 3.0,
        "a_ligand": 1.0,
    }

    pn_units_dataset_weights = calculate_weights_for_pdb_dataset_df(
        dataset_df=rf2aa_pn_units_dataset.data, alphas=alphas, beta=b_pn_unit
    )
    interfaces_dataset_weights = calculate_weights_for_pdb_dataset_df(
        dataset_df=rf2aa_interfaces_dataset.data, alphas=alphas, beta=b_interface
    )
    pdb_dataset_weights = torch.cat([pn_units_dataset_weights, interfaces_dataset_weights])  # NOTE: Order matters!

    # ... and initialize one sampler for all PDB datasets, using the unified weights
    pdb_sampler = WeightedRandomSampler(
        weights=pdb_dataset_weights,
        num_samples=num_examples_per_epoch,  # We later override with proportional number of examples
        replacement=True,
    )

    # +---------------------------- Weighted mix of multiple datasets ----------------------------+

    # Define a MixedSampler with two samplers
    datasets_info = [
        {
            "name": "pdb",
            "dataset": rf2aa_pdb_dataset,
            "sampler": pdb_sampler,
            "probability": 0.2,
        },
        {
            # NOTE: Illustrative; dataset overlaps with the PDB_DATASET. In actuality, this would be a different dataset (e.g., distillation)
            "name": "pn_units",
            "dataset": rf2aa_pn_units_dataset,
            "sampler": SequentialSampler(rf2aa_pn_units_dataset),
            "probability": 0.8,
        },
        # etc.
    ]
    datasets = [dataset_info["dataset"] for dataset_info in datasets_info]

    # ...create a sampler that samples from both datasets, according to the probabilities
    mixed_sampler = MixedSampler(
        datasets_info=datasets_info,
        n_examples_per_epoch=100,
    )

    # ... create a dataset including both datasets
    concat_dataset = ConcatDatasetWithID(datasets=datasets)

    # +---------------------------- Tests and assertions ----------------------------+

    # Test getting examples using the example ID
    example_id_1 = "{['pdb', 'pn_units']}{7d9h}{2}{['B_1']}"
    example_id_2 = "{['pdb', 'interfaces']}{5s4p}{1}{['C_1', 'V_1']}"

    # Get the indices of the examples
    example_1_index = get_row_and_index_by_example_id(concat_dataset, example_id_1)["index"]
    example_2_index = get_row_and_index_by_example_id(concat_dataset, example_id_2)["index"]

    # Load the examples
    example_1 = concat_dataset[example_1_index]
    example_2 = concat_dataset[example_2_index]

    # Assert that the example IDs are correct
    assert example_id_1 == example_1["example_id"]
    assert example_id_2 == example_2["example_id"]

    # Sample from the sampler
    indices = list(mixed_sampler)
    assert len(indices) == 100

    # Check that 80% of the indices are from the (second copy of) the pn_units dataset
    # (all idxs >= len(pdb_dataset) should be from pn_units dataset)
    pn_unit_indices = [idx for idx in indices if idx >= len(rf2aa_pdb_dataset)]
    assert len(pn_unit_indices) == 80

    # Assert that the example_type is "pn_unit" for any example from the pn_units dataset
    for idx in pn_unit_indices[:2]:
        example = concat_dataset[idx]
        assert "pn_unit" in example["example_id"], "example_id does not contain 'pn_unit'"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
