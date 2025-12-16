import gzip
import json
import pickle
from itertools import combinations

import numpy as np
import pytest
import torch

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA, UNKNOWN_AA, UNKNOWN_DNA, UNKNOWN_RNA
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, TokenEncoding
from atomworks.ml.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, ConvertToTorch
from atomworks.ml.transforms.encoding import EncodeAtomArray
from atomworks.ml.transforms.filters import RemoveHydrogens
from atomworks.ml.transforms.msa._msa_featurizing_utils import (
    assign_extra_rows_to_cluster_representatives,
    build_indices_should_be_counted_masks,
    build_msa_index_can_be_masked,
    mask_msa_like_bert,
    summarize_clusters,
    uniformly_select_rows,
)
from atomworks.ml.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeAF3,
    FeaturizeMSALikeRF2AA,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
    get_full_msa_profile_and_insertion_mean,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import token_iter
from tests.ml.conftest import (
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
    TEST_DATA_ML,
)


def generate_synthetic_msa(
    encoding: TokenEncoding, n_rows: int, n_tokens_across_chains: int, n_msa_cluster_representatives: int
) -> dict[str, torch.Tensor]:
    """
    Generate synthetic Multiple Sequence Alignment (MSA) data.

    Args:
        encoding (TokenEncoding): An object containing token encoding information.
        n_rows (int): Number of rows in the MSA.
        n_tokens_across_chains (int): Number of tokens across all chains.
        n_msa_cluster_representatives (int): Number of MSA cluster representatives to select.

    Returns:
        SyntheticMSAData: A dictionary containing various components of the synthetic MSA.
    """

    def get_token_range(token_list: list[int]) -> tuple[int, int]:
        return min(token_list), max(token_list) + 1

    def generate_msa_segment(token_range: tuple[int, int], shape: tuple[int, int]) -> torch.Tensor:
        return torch.randint(token_range[0], token_range[1], shape)

    amino_acid_tokens = [encoding.token_to_idx[res] for res in STANDARD_AA + (UNKNOWN_AA,)]
    rna_tokens = [encoding.token_to_idx[res] for res in STANDARD_RNA + (UNKNOWN_RNA,)]
    dna_tokens = [encoding.token_to_idx[res] for res in STANDARD_DNA + (UNKNOWN_DNA,)]
    atom_tokens = [encoding.token_to_idx[res] for res in [13, 33, 79, 5, 4, 35, 6, 20, 17, 27, 24, 29, 9, 26, 80, 53]]
    mask_token = encoding.token_to_idx["<M>"]

    # Generate full MSA profile
    full_msa_profile = torch.rand(n_tokens_across_chains, encoding.n_tokens)
    full_msa_profile[:, mask_token] = 0
    full_msa_profile /= full_msa_profile.sum(dim=1, keepdim=True)

    # Generate MSA segments
    example_protein_msa = generate_msa_segment(
        get_token_range(amino_acid_tokens), (n_rows, n_tokens_across_chains // 2)
    )
    example_rna_msa = generate_msa_segment(get_token_range(rna_tokens), (n_rows, n_tokens_across_chains // 10))
    example_dna_msa = generate_msa_segment(get_token_range(dna_tokens), (1, n_tokens_across_chains // 10)).repeat(
        n_rows, 1
    )
    example_atom_1_msa = generate_msa_segment(get_token_range(atom_tokens), (1, n_tokens_across_chains // 10)).repeat(
        n_rows, 1
    )
    example_atom_2_msa = generate_msa_segment(
        get_token_range(atom_tokens), (1, 2 * n_tokens_across_chains // 10)
    ).repeat(n_rows, 1)

    # Concatenate into a single MSA
    encoded_msa = torch.cat(
        [example_protein_msa, example_rna_msa, example_dna_msa, example_atom_1_msa, example_atom_2_msa], dim=1
    )

    # Generate masks
    msa_is_padded_mask = torch.randint(0, 2, (n_rows, n_tokens_across_chains)).bool()
    token_idx_has_msa = torch.zeros(n_tokens_across_chains, dtype=torch.bool)
    token_idx_has_msa[: (example_protein_msa.shape[1] + example_rna_msa.shape[1])] = True

    # Break apart the MSA into selected and not selected indices
    selected_indices, not_selected_indices = uniformly_select_rows(n_rows, n_msa_cluster_representatives)

    return {
        "encoded_msa": encoded_msa,
        "msa_is_padded_mask": msa_is_padded_mask,
        "token_idx_has_msa": token_idx_has_msa,
        "full_msa_profile": full_msa_profile,
        "selected_indices": selected_indices,
        "not_selected_indices": not_selected_indices,
        "example_protein_msa": example_protein_msa,
        "example_rna_msa": example_rna_msa,
        "example_dna_msa": example_dna_msa,
        "example_atom_1_msa": example_atom_1_msa,
        "example_atom_2_msa": example_atom_2_msa,
    }


def all_different(tensor_list: list[torch.Tensor]) -> bool:
    """
    Check if all tensors in the list are unique.

    Args:
        tensor_list (list): List of tensors to compare.

    Returns:
        bool: True if all tensors are different, False if any are equal.
    """
    return all(not torch.equal(t1, t2) for t1, t2 in combinations(tensor_list, 2))


def similar_stats(
    tensor_list: list[torch.Tensor],
    mean_lower: float = 0.3,
    mean_upper: float = 1.3,
    std_lower: float = 0.7,
    std_upper: float = 1.3,
) -> bool:
    """Check if tensor statistics are similar within specified ranges.

    Args:
        tensor_list (list): List of tensors to compare.
        mean_lower (float, optional): Lower bound for mean. Defaults to 0.3.
        mean_upper (float, optional): Upper bound for mean. Defaults to 1.3.
        std_lower (float, optional): Lower bound for std dev. Defaults to 0.7.
        std_upper (float, optional): Upper bound for std dev. Defaults to 1.3.

    Returns:
        bool: True if all tensors have similar stats, False otherwise.
    """
    means = [t.float().mean().item() for t in tensor_list]
    stds = [t.float().std().item() for t in tensor_list]
    mean_mean, mean_std = sum(means) / len(means), sum(stds) / len(stds)

    return all(
        mean_lower * mean_mean <= m <= mean_upper * mean_mean and std_lower * mean_std <= s <= std_upper * mean_std
        for m, s in zip(means, stds, strict=False)
    )


FILL_FULL_MSA_FROM_ENCODED_TEST_CASES = ["3ejj", "1mna", "1hge"]


@pytest.mark.parametrize("pdb_id", FILL_FULL_MSA_FROM_ENCODED_TEST_CASES)
def test_fill_full_msa_from_encoded(pdb_id):
    """
    Test if the full MSA is filled correctly from the encoded MSA through a series of logical assertions.

    In particular, we want to ensure:
    - The padding is carried over correctly (i.e., the padding in the encoded MSA is reflected in the full MSA)
    - The corresponding MSA columns match (i.e., after fancy indexing)
    - The insertions match (i.e., we didn't lose any)
    """
    data = cached_parse(pdb_id, hydrogen_policy="remove")

    encoding = RF2AA_ATOM36_ENCODING
    res_names_to_ignore = encoding.tokens[
        encoding.tokens != "ASP"
    ]  # Atomize aspartate so we can test atomization and MSA indexing
    pad_token = RF2AA_ATOM36_ENCODING.token_to_idx["UNK"]

    # Apply initial transforms
    # fmt: off
    pipeline = Compose([
        AddWithinPolyResIdxAnnotation(),
        LoadPolymerMSAs(protein_msa_dirs=PROTEIN_MSA_DIRS, rna_msa_dirs=RNA_MSA_DIRS, max_msa_sequences=100),
        PairAndMergePolymerMSAs(),
        AtomizeByCCDName(
            atomize_by_default=True, res_names_to_ignore=res_names_to_ignore, move_atomized_part_to_end=False
        ),
        EncodeAtomArray(encoding),
        # MSA featurize workflow
        EncodeMSA(encoding=encoding, token_to_use_for_gap=pad_token),
        FillFullMSAFromEncoded(pad_token=pad_token),
    ], track_rng_state=False)
    # fmt: on

    output = pipeline(data)
    atom_array = output["atom_array"]

    # Iterate through all tokens
    atomized_indices = []
    for index, token_atom_array in enumerate(token_iter(atom_array)):
        if token_atom_array.atomize[0]:
            # If this residue is atomized, ensure that the entire MSA column (other than the query sequence) is padding...
            assert np.all(
                output["encoded"]["msa"][1:, index] == pad_token
            ), f"MSA column for atomized residue {index} is not padding"

            # ...and the padding is represented in the full MSA details
            assert not output["full_msa_details"]["token_idx_has_msa"][index], "Token index has MSA when it should not"
            assert np.all(
                output["full_msa_details"]["msa_is_padded_mask"][1:, index]
            ), "MSA is not padded when it should be"
            atomized_indices.append(index)
        else:
            # If this residue is not atomized, ensure that the MSA matches with the pre-atomized MSA...
            within_poly_res_idx = token_atom_array.within_poly_res_idx[0]
            chain_id = token_atom_array.chain_id[0]
            encoded_old_msa = output["polymer_msas_by_chain_id"][chain_id]["encoded_msa"]
            msa_column_old = encoded_old_msa[:, within_poly_res_idx]
            msa_column_new = output["encoded"]["msa"][:, index]
            assert np.array_equal(
                msa_column_old, msa_column_new
            ), f"MSA column for non-atomized residue {index} does not match"

            # ...and that we are noting that this token has MSA
            assert output["full_msa_details"]["token_idx_has_msa"][
                index
            ], "Token index does not have MSA when it should"

    # Check that there are no insertions where there is MSA padding...
    msa_raw_ins = output["full_msa_details"]["msa_raw_ins"]
    msa_is_padded_mask = output["full_msa_details"]["msa_is_padded_mask"]
    assert np.sum(msa_raw_ins * msa_is_padded_mask) == 0, "There should be no insertions where there is MSA padding"

    # ...AND that there are no insertions where there are atomized tokens
    assert (
        np.sum(msa_raw_ins[:, atomized_indices]) == 0
    ), "There should be no insertions where there are atomized tokens"


ASSIGN_EXTRA_ROWS_TEST_CASES = [
    # Test case 1: No masking, but ignore specific tokens
    {
        "mask_position": torch.tensor([[False, False], [False, False], [False, False]], dtype=torch.bool),
        "encoded_msa": torch.tensor([[1, 2], [3, 4], [1, 4]], dtype=torch.int),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),  # Main MSA
        "not_selected_indices": torch.tensor([2], dtype=torch.int),  # Extra MSA -- we will match to a cluster
        "token_idx_has_msa": torch.tensor([True, True], dtype=torch.bool),  # Include all columns (tokens)
        "tokens_to_ignore": torch.tensor([1], dtype=torch.int),  # Ignore the "1" token
        "expected_assignment": torch.tensor(
            [1], dtype=torch.int
        ),  # Since we are ignoring the "1" token, we should assign to the "2" token (2 Hamming distance vs. 1 Hamming distance)
    },
    # Test case 2: Simplified example from the docstring, no masking, or ignoring tokens
    {
        "mask_position": torch.zeros((6, 5), dtype=bool),
        "encoded_msa": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 2, 2],
                [2, 2, 1, 0, 0],
                [3, 3, 3, 2, 2],
                [1, 1, 3, 3, 3],
            ],
            dtype=torch.int,
        ),
        "selected_indices": torch.tensor([0, 1, 2], dtype=torch.int),
        "not_selected_indices": torch.tensor([3, 4, 5], dtype=torch.int),
        "token_idx_has_msa": torch.tensor([True, True, True, True, True], dtype=torch.bool),
        "tokens_to_ignore": torch.tensor([], dtype=torch.int),
        "expected_assignment": torch.tensor([1, 2, 0], dtype=torch.int),
    },
    # Test case 3: Testing mask_position functionality
    {
        # Simulate masking a block of the MSA due to unpaired sequences
        "mask_position": torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        ),
        "encoded_msa": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 2, 2],
                [2, 2, 1, 0, 0],
                [3, 3, 3, 2, 2],
                [1, 1, 3, 3, 3],
            ],  # With the mask, most similar to row index 2 (previously was row index 0)
            dtype=torch.int,
        ),
        "selected_indices": torch.tensor([0, 1, 2], dtype=torch.int),
        "not_selected_indices": torch.tensor([3, 4, 5], dtype=torch.int),
        "token_idx_has_msa": torch.tensor([True, True, True, True, True], dtype=torch.bool),
        "tokens_to_ignore": torch.tensor([], dtype=torch.int),
        "expected_assignment": torch.tensor([1, 2, 2], dtype=torch.int),
    },
]


@pytest.mark.parametrize("test_case", ASSIGN_EXTRA_ROWS_TEST_CASES)
def test_assign_extra_rows_to_cluster_representatives(test_case):
    """
    Tests assignment of extra MSA rows to rows within the main MSA by Hamming distance, using hand-crafted test cases.
    This function is used in AF-2 and RF2-AA, but not in AF-3 (which eschews MSA clustering).

    Involves two functions:
    (1) `build_indices_should_be_counted_mask` to identify which indices should count towards the agreement sum
    (2) `assign_extra_rows_to_cluster_representatives` to assign the extra rows to the main MSA rows, based on the agreement sum
    """
    mask_position = test_case["mask_position"]
    encoded_msa = test_case["encoded_msa"]
    selected_indices = test_case["selected_indices"]
    not_selected_indices = test_case["not_selected_indices"]
    token_idx_has_msa = test_case["token_idx_has_msa"]
    tokens_to_ignore = test_case["tokens_to_ignore"]
    expected_assignment = test_case["expected_assignment"]

    index_should_be_counted_mask = build_indices_should_be_counted_masks(
        encoded_msa=encoded_msa,
        mask_position=mask_position,
        tokens_to_ignore=tokens_to_ignore,
        token_idx_has_msa=token_idx_has_msa,
    )  # [n_rows, n_tokens_across_chains] (bool)

    assignments = assign_extra_rows_to_cluster_representatives(
        cluster_representatives_msa=encoded_msa[selected_indices],
        clust_reps_should_be_counted_mask=index_should_be_counted_mask[selected_indices],
        extra_msa=encoded_msa[not_selected_indices],
        extra_msa_should_be_counted_mask=index_should_be_counted_mask[not_selected_indices],
    )

    assert torch.equal(assignments, expected_assignment), f"Expected {expected_assignment}, but got {assignments}"


SUMMARIZE_CLUSTERS_TEST_CASES = [
    # Test case 1: All mask_position is False
    {
        "encoded_msa": torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=torch.int64),
        "msa_raw_ins": torch.tensor([[0, 0, 2], [1, 0, 0], [0, 1, 0], [4, 0, 1]], dtype=torch.int),
        "mask_position": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "assignments": torch.tensor([0, 1], dtype=torch.int64),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),
        "not_selected_indices": torch.tensor([2, 3], dtype=torch.int),
        "msa_is_padded_mask": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "expected_profiles": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]], [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
        "expected_insertions": torch.tensor([[0.0, 0.5, 1.0], [5 / 2, 0, 0.5]]),
    },
    # Test case 2: Example from docstring
    {
        "encoded_msa": torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=torch.int64),
        "msa_raw_ins": torch.tensor([[0, 0, 2], [1, 0, 0], [0, 1, 0], [4, 0, 1]], dtype=torch.int),
        "mask_position": torch.tensor(
            [[False, False, False], [True, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "assignments": torch.tensor([0, 1], dtype=torch.int64),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),
        "not_selected_indices": torch.tensor([2, 3], dtype=torch.int),
        "msa_is_padded_mask": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, True], [False, False, True]], dtype=torch.bool
        ),
        "expected_profiles": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
        "expected_insertions": torch.tensor([[0.0, 0.5, 2.0], [4, 0.0, 0.0]]),
    },
]


@pytest.mark.parametrize("test_case", SUMMARIZE_CLUSTERS_TEST_CASES)
def test_summarize_clusters(test_case):
    """
    Tests the summarization of MSA clusters based on assignments, using hand-crafted test cases.
    This function is used in AF-2 and RF2-AA, but not in AF-3 (which eschews MSA clustering).
    """
    encoded_msa = test_case["encoded_msa"]
    msa_raw_ins = test_case["msa_raw_ins"]
    mask_position = test_case["mask_position"]
    assignments = test_case["assignments"]
    selected_indices = test_case["selected_indices"]
    not_selected_indices = test_case["not_selected_indices"]
    msa_is_padded_mask = test_case["msa_is_padded_mask"]
    expected_profiles = test_case["expected_profiles"]
    expected_insertions = test_case["expected_insertions"]

    msa_cluster_profiles, msa_cluster_ins = summarize_clusters(
        encoded_msa=encoded_msa,
        msa_raw_ins=msa_raw_ins,
        mask_position=mask_position,
        assignments=assignments,
        selected_indices=selected_indices,
        not_selected_indices=not_selected_indices,
        msa_is_padded_mask=msa_is_padded_mask,
        n_tokens=3,
    )

    assert torch.allclose(
        msa_cluster_profiles, expected_profiles, atol=1e-4
    ), f"Expected profiles {expected_profiles}, but got {msa_cluster_profiles}"
    assert torch.allclose(
        msa_cluster_ins, expected_insertions, atol=1e-4
    ), f"Expected insertions {expected_insertions}, but got {msa_cluster_ins}"


def test_mask_msa_like_bert():
    """
    Tests the generation and application of the BERT-style masking to the MSA through a series of logical assertions
    and statistical checks. Only the main MSA is masked; the extra MSA is left unchanged.
    This function is used in AF-2 and RF2-AA, but not in AF-3 (which eschews MSA masked token recovery).

    Includes:
    - Assertions to sanity-check outputs
    - Regression test to ensure that the output is consistent across runs
    """
    # ...initialize
    mask_behavior_probs = {"replace_with_random_aa": 0.1, "replace_with_msa_profile": 0.1, "do_not_replace": 0.1}
    mask_probability = 0.15
    encoding = RF2AA_ATOM36_ENCODING
    n_tokens_across_chains = 40
    mask_token = encoding.token_to_idx["<M>"]

    # ...generate synthetic data with a fixed seed
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        synthetic_msa = generate_synthetic_msa(
            encoding=encoding,
            n_msa_cluster_representatives=40,
            n_rows=50,
            n_tokens_across_chains=n_tokens_across_chains,
        )
    # ...unpack synthetic data
    msa_is_padded_mask = synthetic_msa["msa_is_padded_mask"]
    token_idx_has_msa = synthetic_msa["token_idx_has_msa"]
    encoded_msa = synthetic_msa["encoded_msa"]
    full_msa_profile = synthetic_msa["full_msa_profile"]
    selected_indices = synthetic_msa["selected_indices"]
    not_selected_indices = synthetic_msa["not_selected_indices"]

    # ...run the function using a fixed seed (since the seed may have changed while generating the synthetic data)
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        index_can_be_masked = build_msa_index_can_be_masked(
            msa_is_padded_mask=msa_is_padded_mask,
            token_idx_has_msa=token_idx_has_msa,
            encoded_msa=encoded_msa,
            encoding=encoding,
        )

        new_partial_msa, mask_position = mask_msa_like_bert(
            encoding=encoding,
            mask_behavior_probs=mask_behavior_probs,
            mask_probability=mask_probability,
            full_msa_profile=full_msa_profile,
            encoded_msa=encoded_msa[selected_indices],
            index_can_be_masked=index_can_be_masked[selected_indices],
        )
        new_encoded_msa = encoded_msa.clone()
        new_encoded_msa[selected_indices] = new_partial_msa

        # ...ensure things that weren't suppose to change, didn't change
        assert torch.equal(
            encoded_msa[msa_is_padded_mask], new_encoded_msa[msa_is_padded_mask]
        )  # Check that padding positions remain unchanged
        assert torch.equal(
            encoded_msa[not_selected_indices], new_encoded_msa[not_selected_indices]
        )  # Check that the extra MSA columns remained unchanged

        # ...check that mask_position holds the correct values
        assert torch.all(
            ~mask_position[~index_can_be_masked[selected_indices]]
        )  # Check that no masking occurs where we didn't want any
        assert torch.any(
            mask_position[index_can_be_masked[selected_indices]]
        )  # Check that there is masking where we did want some

        # ...ensure mask_position is False for all non-protein columns
        protein_columns = torch.zeros(n_tokens_across_chains, dtype=torch.int)
        protein_columns[: synthetic_msa["example_protein_msa"].shape[1]] = 1
        assert torch.all(~mask_position[:, ~protein_columns.bool()])

        # ...check that we have approximately the right number of mask tokens
        num_could_be_masked = index_can_be_masked[selected_indices].sum().item()
        expected_num_mask_applied = int(mask_probability * num_could_be_masked)
        actual_num_mask_applied = mask_position.sum().item()

        # ...calculate the standard deviation of the binomial distribution = sqrt(n * p * (1 - p))
        std_dev = (num_could_be_masked * mask_probability * (1 - mask_probability)) ** 0.5

        # ...check that the actual number of masks is within 2 standard deviations of the expected number
        assert abs(actual_num_mask_applied - expected_num_mask_applied) <= 2 * std_dev

        # ...check that the number of mask tokens is close to the expected number
        mask_token_probability = 1 - sum(list(mask_behavior_probs.values()))
        expected_num_mask_tokens = actual_num_mask_applied * mask_token_probability
        actual_num_mask_tokens = (new_partial_msa == mask_token).sum().item()

        # ...check that the number of mask tokens is within 2 standard deviations of the expected number
        std_dev = (actual_num_mask_applied * mask_token_probability * (1 - mask_token_probability)) ** 0.5
        assert abs(actual_num_mask_tokens - expected_num_mask_tokens) <= 2 * std_dev

        # ...execute regression tests, loading from a saved JSON
        saved_result_path = TEST_DATA_ML / "mask_msa_regression_test.json"

        # Uncomment to save new_encoded_msa for regression tests, as a JSON
        # with open(saved_result_path, "w") as f:
        #     json.dump(new_encoded_msa.tolist(), f)

        # ...check that the new_encoded_msa matches the saved results
        with open(saved_result_path) as f:
            old_results = json.load(f)
        assert torch.allclose(new_encoded_msa, torch.tensor(old_results), atol=1e-4, rtol=1e-4)


MSA_FEATURIZE_PIPELINE_TEST_CASES = ["3ejj"]


@pytest.mark.parametrize("pdb_id", MSA_FEATURIZE_PIPELINE_TEST_CASES)
def test_msa_featurize_like_rf2aa_full_pipeline(pdb_id):
    """
    Test the full MSA featurization pipeline for RF2-AA, including the encoding, MSA featurization, and masking.
    Conduct statistical checks and regression tests to ensure consistency across runs and avoid moving distributions across recycles.
    """
    # Hyperparameters (to be defined in Hydra)
    encoding = RF2AA_ATOM36_ENCODING
    n_msa_cluster_representatives = 20
    n_extra_rows = 20
    n_recycles = 5  # We choose 5 recycles to ensure we would find any drift across recycles
    probs = {
        "replace_with_random_aa": 0.1,
        "replace_with_msa_profile": 0.1,
        "do_not_replace": 0.1,
    }
    mask_probability = 0.15
    pad_token = encoding.token_to_idx["UNK"]

    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        data = cached_parse(pdb_id, hydrogen_policy="remove")
        pipeline = Compose(
            [
                RemoveHydrogens(),
                AddWithinPolyResIdxAnnotation(),
                LoadPolymerMSAs(protein_msa_dirs=PROTEIN_MSA_DIRS, rna_msa_dirs=RNA_MSA_DIRS, max_msa_sequences=1000),
                PairAndMergePolymerMSAs(),
                AtomizeByCCDName(
                    atomize_by_default=True, res_names_to_ignore=encoding.tokens, move_atomized_part_to_end=True
                ),
                EncodeAtomArray(encoding),
                # MSA featurize workflow
                EncodeMSA(encoding=encoding, token_to_use_for_gap=pad_token),
                FillFullMSAFromEncoded(pad_token=pad_token),
                ConvertToTorch(keys=["polymer_msas_by_chain_id", "encoded", "full_msa_details"]),
                FeaturizeMSALikeRF2AA(
                    n_recycles=n_recycles,
                    n_msa_cluster_representatives=n_msa_cluster_representatives,
                    n_extra_rows=n_extra_rows,
                    mask_behavior_probs=probs,
                    mask_probability=mask_probability,
                    encoding=encoding,
                    polymer_token_indices=torch.arange(
                        32
                    ),  # NOTE: This is hard-coded for the AA and NA tokens in the RF2AA Encoding (all non-atom tokens)
                ),
            ],
            track_rng_state=False,
        )
        output = pipeline(data)
        assert output is not None

        ############## Assertions ##############
        features_by_recycle = output["features_per_recycle_dict"]

        # List of keys to check for being different and having similar sums
        keys_to_check = [
            "first_row_of_msa",  # NOTE: This will fail if our test example has no polymers
            "cluster_representatives_msa_ground_truth",
            "cluster_representatives_msa_masked",
            "cluster_representatives_has_insertion",
            "cluster_representatives_insertion_value",
            "cluster_insertion_mean",
            "cluster_profile",
            "extra_msa",
            "extra_msa_has_insertion",
            "extra_msa_insertion_value",
            "bert_mask_position",
        ]

        for key in keys_to_check:
            tensor_list = features_by_recycle[key]
            assert all_different(tensor_list), f"{key} elements are not all different"
            assert similar_stats(tensor_list), f"{key} elements do not have similar means and standard deviations"

        ############## Regression test ##############

        # Save in the test directory
        saved_result_path = TEST_DATA_ML / f"{pdb_id}_featurize_msa_like_rf2aa_regression_test.pkl.gz"

        # Uncomment to save output['features_per_recycle_dict'] for regression tests, as a compressed pickle
        # with gzip.open(saved_result_path, "wb") as f:
        #     pickle.dump(output["features_per_recycle_dict"], f, protocol=pickle.HIGHEST_PROTOCOL)

        # Check that the new_encoded_msa matches the saved results
        with gzip.open(saved_result_path, "rb") as f:
            old_results = pickle.load(f)

        # For each key in the dictionary, check that the values match
        for key, old_values in old_results.items():
            new_values = output["features_per_recycle_dict"][key]
            assert torch.allclose(
                torch.stack(new_values), torch.stack(old_values), atol=1e-4, rtol=1e-4
            ), f"Failed at key: {key}. Difference: {set(new_values) - set(old_values)}"


@pytest.mark.parametrize("pdb_id", MSA_FEATURIZE_PIPELINE_TEST_CASES)
def test_msa_featurize_like_af3_full_pipeline(pdb_id):
    """
    Test the full MSA featurization pipeline for AF-3, including the encoding and MSA featurization (no masking).
    Conduct statistical checks and regression tests to ensure consistency across runs and avoid moving distributions across recycles.
    """
    # Hyperparameters (to be defined in Hydra)
    encoding = RF2AA_ATOM36_ENCODING
    n_recycles = 5  # We choose 5 recycles to ensure we would find any drift across recycles
    pad_token = encoding.token_to_idx["UNK"]

    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        data = cached_parse(pdb_id, hydrogen_policy="remove")
        pipeline = Compose(
            [
                AddWithinPolyResIdxAnnotation(),
                LoadPolymerMSAs(protein_msa_dirs=PROTEIN_MSA_DIRS, rna_msa_dirs=RNA_MSA_DIRS, max_msa_sequences=1000),
                PairAndMergePolymerMSAs(),
                AtomizeByCCDName(
                    atomize_by_default=True, res_names_to_ignore=encoding.tokens, move_atomized_part_to_end=True
                ),
                EncodeAtomArray(encoding),
                # MSA featurize workflow
                EncodeMSA(encoding=encoding, token_to_use_for_gap=pad_token),
                FillFullMSAFromEncoded(pad_token=pad_token),
                ConvertToTorch(keys=["polymer_msas_by_chain_id", "encoded", "full_msa_details"]),
                FeaturizeMSALikeAF3(
                    encoding=encoding,
                    n_recycles=n_recycles,
                    n_msa=100,
                ),
            ],
            track_rng_state=False,
        )
        output = pipeline(data)
        assert output is not None

        ############## Assertions ##############
        msa_features_per_recycle_dict = output["msa_features"]["msa_features_per_recycle_dict"]
        msa_static_features_dict = output["msa_features"]["msa_static_features_dict"]

        # List of keys to check for being different and having similar sums
        keys_to_check_across_recycles = [
            "msa",
            "has_insertion",
            "insertion_value",
        ]

        for key in keys_to_check_across_recycles:
            tensor_list = msa_features_per_recycle_dict[key]
            assert all_different(tensor_list), f"{key} elements are not all different"
            assert similar_stats(tensor_list), f"{key} elements do not have similar means and standard deviations"

        # Assert that insertion_values are between 0 and 1 (for the first recycle)
        assert torch.all(
            (msa_features_per_recycle_dict["insertion_value"][0] >= 0)
            & (msa_features_per_recycle_dict["insertion_value"][0] <= 1)
        )

        # Assert that has_insertion is a boolean tensor...
        assert (
            msa_features_per_recycle_dict["has_insertion"][0].dtype == torch.bool
        ), "has_insertion must be of boolean dtype"

        # ...and that there's at least one
        assert torch.any(
            msa_features_per_recycle_dict["has_insertion"][0]
        ), "There must be at least one insertion, if we're using examples with MSA's"

        ############## Regression test ##############

        # Save in the test directory
        saved_result_path = TEST_DATA_ML / f"{pdb_id}_featurize_msa_like_af3_regression_test.pkl.gz"

        # Uncomment to save output['msa_features'] for regression tests, as a compressed pickle
        # with gzip.open(saved_result_path, "wb") as f:
        #     pickle.dump(output["msa_features"], f, protocol=pickle.HIGHEST_PROTOCOL)

        # Check that the new_encoded_msa matches the saved results
        with gzip.open(saved_result_path, "rb") as f:
            old_results = pickle.load(f)

        # For each key in the features that change across recycles, check that the values match...
        for key, old_values in old_results["msa_features_per_recycle_dict"].items():
            new_values = msa_features_per_recycle_dict[key]
            assert torch.allclose(
                torch.stack(new_values), torch.stack(old_values), atol=1e-4, rtol=1e-4
            ), f"Failed at key: {key}. Difference: {set(new_values) - set(old_values)}"
        # ... and for the static features as well
        for key, old_value in old_results["msa_static_features_dict"].items():
            new_value = msa_static_features_dict[key]
            assert torch.allclose(
                new_value, old_value, atol=1e-4, rtol=1e-4
            ), f"Failed at key: {key}. Difference: {new_value - old_value}"


# Define a simple TokenEncoding class for testing
class DummyTokenEncoding:
    def __init__(self, n_tokens):
        self.n_tokens = n_tokens


TEST_FULL_MSA_PROFILE_AND_INSERTION_MEAN = [
    {
        # Test case for a simple MSA without padding
        "encoded_msa": torch.tensor([[0, 1, 2], [1, 2, 0]]),
        "msa_raw_ins": torch.tensor([[0, 1, 0], [2, 0, 1]]),
        "msa_is_padded_mask": torch.tensor([[False, False, False], [False, False, False]]),
        "encoding": DummyTokenEncoding(3),
        "expected_profile": torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]),
        "expected_ins_mean": torch.tensor([1.0, 0.5, 0.5]),
    },
    {
        # Test case for MSA with padding and masks
        "encoded_msa": torch.tensor([[0, 1, 2, 1], [1, 2, 0, 2], [2, 0, 1, 0], [0, 1, 2, 1]]),
        "msa_raw_ins": torch.tensor([[1, 0, 2, 1], [0, 1, 0, 0], [2, 1, 0, 3], [0, 0, 1, 0]]),
        "msa_is_padded_mask": torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, True, True, True],
                [True, True, True, True],
            ]
        ),
        "encoding": DummyTokenEncoding(3),
        "expected_profile": torch.tensor([[1 / 3, 1 / 3, 1 / 3], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]),
        "expected_ins_mean": torch.tensor([1.0, 0.5, 1.0, 1.0]),
    },
]


@pytest.mark.parametrize("test_case", TEST_FULL_MSA_PROFILE_AND_INSERTION_MEAN)
def test_get_full_msa_profile_and_insertion_mean(test_case):
    profile, ins_mean = get_full_msa_profile_and_insertion_mean(
        test_case["encoded_msa"], test_case["msa_raw_ins"], test_case["msa_is_padded_mask"], test_case["encoding"]
    )

    assert torch.allclose(profile, test_case["expected_profile"], atol=1e-6)
    assert torch.allclose(ins_mean, test_case["expected_ins_mean"], atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", "-m not very_slow", __file__])
