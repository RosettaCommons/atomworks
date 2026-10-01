import pandas as pd
import pyarrow as pa
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, SequentialSampler

from atomworks.ml.samplers import (
    DistributedMixedSampler,
    LoadBalancedDistributedSampler,
    MixedSampler,
    calculate_weights_by_inverse_cluster_size,
    calculate_weights_for_pdb_dataset_df,
)


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def dummy_datasets():
    data1 = list(range(100))
    data2 = list(range(1000, 1100))  # Different range for testing
    data3 = list(range(2000, 2100))  # Different range for testing
    dataset1 = DummyDataset(data1)
    dataset2 = DummyDataset(data2)
    dataset3 = DummyDataset(data3)
    return dataset1, dataset2, dataset3


def test_distributed_mixed_sampler(dummy_datasets):
    """
    Test that the DistributedMixedSampler correctly samples from the datasets with the specified probabilities,
    ensuring that each node gets a different slice of the dataset and that the distribution of samples reflects
    the specified probabilities.
    """
    dataset1, dataset2, dataset3 = dummy_datasets

    # Samplers
    sampler_1 = SequentialSampler(dataset1)
    sampler_2 = SequentialSampler(dataset2)

    # Nested mixed sampler
    datasets_info_1 = [
        {"sampler": sampler_1, "dataset": dataset1, "probability": 0.9},
        {"sampler": sampler_2, "dataset": dataset2, "probability": 0.1},
    ]
    mixed_sampler = MixedSampler(datasets_info=datasets_info_1)

    # Outer mixed (distributed) sampler
    sampler_3 = SequentialSampler(dataset3)
    datasets_1_2_concat = ConcatDataset([dataset1, dataset2])
    datasets_info_2 = [
        {"sampler": mixed_sampler, "dataset": datasets_1_2_concat, "probability": 0.4},
        {"sampler": sampler_3, "dataset": dataset3, "probability": 0.6},
    ]
    datasets_1_2_3_concat = ConcatDataset([datasets_1_2_concat, dataset3])
    dist_mixed_sampler_rank_0 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=101,  # Odd number to test rounding
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=False,  # False is the more complex case
    )

    dist_mixed_sampler_rank_1 = DistributedMixedSampler(
        datasets_info=datasets_info_2,
        n_examples_per_epoch=101,
        num_replicas=2,
        rank=1,
        shuffle=True,
        drop_last=False,  # False is the more complex case
    )

    indices_node_0 = list(dist_mixed_sampler_rank_0)
    indices_node_1 = list(dist_mixed_sampler_rank_1)

    assert len(indices_node_0) == 51  # Rounding (based on 101 examples and 2 replicas, without drop_last)
    assert len(indices_node_1) == 51

    # Ensure the slices are different
    assert set(indices_node_0).isdisjoint(set(indices_node_1))

    # Combine indices from both nodes to check distribution
    combined_indices = indices_node_0 + indices_node_1

    # Check if the distribution is close to the expected 90-10 ratio
    dataset1_count = sum(1 for idx in combined_indices if idx < 100)
    dataset2_count = sum(1 for idx in combined_indices if 100 <= idx < 200)
    dataset3_count = sum(1 for idx in combined_indices if idx >= 200)

    assert dataset1_count == 37  # 40% * 90% = 37-ish
    assert dataset2_count == 4  # 40% * 10% = 4-ish
    assert dataset3_count == 61  # 60% of the samples should be from dataset3

    # Load indices from the concat_dataset and ensure they are in the expected range
    for idx in indices_node_0[:10] + indices_node_1[:10]:  # Check a few indices from each node
        item = datasets_1_2_3_concat[idx]
        if idx < 100:
            assert 0 <= item < 100  # Should be in the range of dataset1
        elif idx < 200:
            assert 1000 <= item < 1100
        else:
            assert 2000 <= item < 2100


@pytest.fixture
def dummy_dataset_with_n_tokens():
    # Create a dataset where each item has a different number of tokens
    data = pd.DataFrame({"size": [100, 400, 200, 800, 100, 800, 200, 900, 200]})
    return DummyDataset(data)


def test_load_balanced_distributed_sampler(dummy_dataset_with_n_tokens):
    """
    Test that the LoadBalancedDistributedSampler correctly balances "size" loads across replicas.
    """
    length_key = "size"

    # Create samplers for two replicas
    sampler_rank_0 = LoadBalancedDistributedSampler(
        dummy_dataset_with_n_tokens, key_to_balance=length_key, num_replicas=2, rank=0
    )
    sampler_rank_1 = LoadBalancedDistributedSampler(
        dummy_dataset_with_n_tokens, key_to_balance=length_key, num_replicas=2, rank=1
    )

    # Get indices for each rank
    indices_rank_0 = list(sampler_rank_0)
    indices_rank_1 = list(sampler_rank_1)

    # Calculate size counts for each rank
    sizes_rank_0 = [dummy_dataset_with_n_tokens.data[length_key][idx] for idx in indices_rank_0]
    sizes_rank_1 = [dummy_dataset_with_n_tokens.data[length_key][idx] for idx in indices_rank_1]

    # Assert that the size count is balanced
    assert abs(sum(sizes_rank_0) - sum(sizes_rank_1)) <= max(dummy_dataset_with_n_tokens.data[length_key])

    # Ensure that indices are disjoint (except for the last index, which may be the same due to padding)
    assert set(indices_rank_0[:-1]).isdisjoint(set(indices_rank_1[:-1]))


# ---------------------------------------------------------------------------
# Weight function tests
# ---------------------------------------------------------------------------

# Minimal table with 6 rows covering three clusters:
#   A (size 3): pdb 1abc (altloc seeds 0 and 1) + pdb 2abc (no altlocs)
#   B (size 1): pdb 3abc (no altlocs), pure nucleic acid
#   C (size 2): pdb 4abc (no altlocs, involves LOI) + pdb 5abc (no altlocs, peptide)
_WEIGHT_DATA = {
    "cluster": ["A", "A", "A", "B", "C", "C"],
    "pdb_id": ["1abc", "1abc", "2abc", "3abc", "4abc", "5abc"],
    "altloc_seed": [0, 1, None, None, None, None],
    "n_prot": [1, 1, 2, 0, 1, 0],
    "n_nuc": [0, 0, 0, 1, 0, 0],
    "n_ligand": [0, 0, 0, 0, 1, 0],
    "n_peptide": [0, 0, 0, 0, 0, 2],
    "involves_loi": [False, False, False, False, True, False],
}


@pytest.fixture(params=["pandas", "pyarrow"])
def weight_table(request):
    df = pd.DataFrame(_WEIGHT_DATA)
    if request.param == "pyarrow":
        return pa.Table.from_pandas(df)
    return df


def test_calculate_weights_by_inverse_cluster_size(weight_table):
    # Without altloc: weights = 1/cluster_size
    # cluster sizes: A=3, B=1, C=2
    expected_no_altloc = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0, 0.5, 0.5], dtype=torch.float64)
    weights = calculate_weights_by_inverse_cluster_size(weight_table, cluster_column="cluster")
    assert torch.allclose(weights.double(), expected_no_altloc)

    # With altloc: effective cluster size = sum of altloc weights per cluster
    # 1abc has 2 altloc seeds → contributes 1/2+1/2=1.0 to cluster A; 2abc contributes 1.0
    # effective sizes: A=2, B=1, C=2
    expected_altloc = torch.tensor([1 / 4, 1 / 4, 1 / 2, 1.0, 1 / 2, 1 / 2], dtype=torch.float64)
    weights = calculate_weights_by_inverse_cluster_size(
        weight_table, cluster_column="cluster", altloc_seed_column="altloc_seed", pdb_id_column="pdb_id"
    )
    assert torch.allclose(weights.double(), expected_altloc)


def test_calculate_weights_for_pdb_dataset_df(weight_table):
    alphas = {"a_prot": 1.0, "a_nuc": 0.5, "a_ligand": 2.0, "a_peptide": 0.25, "a_loi": 3.0}

    # Without altloc: w = (beta/cluster_size) * (n_prot + n_nuc + n_ligand + n_peptide + 2*is_loi)
    # row 0 (A): (1/3)*(1)         = 1/3
    # row 1 (A): (1/3)*(1)         = 1/3
    # row 2 (A): (1/3)*(2)         = 2/3
    # row 3 (B): (1/1)*(0.5)       = 0.5
    # row 4 (C): (1/2)*(1+2+3)     = 3
    # row 5 (C): (1/2)*(2*0.25)    = 0.25
    expected_no_altloc = torch.tensor([1 / 3, 1 / 3, 2 / 3, 0.5, 3.0, 0.25], dtype=torch.float64)
    weights = calculate_weights_for_pdb_dataset_df(weight_table, alphas=alphas, beta=1.0, cluster_column="cluster")
    assert torch.allclose(weights.double(), expected_no_altloc)

    # With altloc: effective cluster size = sum of altloc weights per cluster
    # altloc weights: 1abc→1/2, 1abc→1/2, 2abc→1, 3abc→1, 4abc→1, 5abc→1
    # effective sizes: A = 1/2+1/2+1 = 2, B = 1, C = 1+1 = 2
    # w = (1/eff_cluster_size) * altloc_weight * (n_prot + n_nuc + n_ligand + n_peptide + 2*is_loi)
    # row 0 (A, 1abc): (1/2) * (1/2) * (1)         = 1/4
    # row 1 (A, 1abc): (1/2) * (1/2) * (1)         = 1/4
    # row 2 (A, 2abc): (1/2) * 1     * (2)         = 1.0
    # row 3 (B, 3abc): (1/1) * 1     * (0.5)       = 0.5
    # row 4 (C, 4abc): (1/2) * 1     * (1+2+3)     = 3.0
    # row 5 (C, 5abc): (1/2) * 1     * (2*0.25)    = 0.25
    expected_altloc = torch.tensor([1 / 4, 1 / 4, 1.0, 0.5, 3.0, 0.25], dtype=torch.float64)
    weights = calculate_weights_for_pdb_dataset_df(
        weight_table,
        alphas=alphas,
        beta=1.0,
        cluster_column="cluster",
        altloc_seed_column="altloc_seed",
        pdb_id_column="pdb_id",
    )
    assert torch.allclose(weights.double(), expected_altloc)


if __name__ == "__main__":
    pytest.main([__file__])
