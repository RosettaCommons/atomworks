import numpy as np
import pytest
import torch

from atomworks.constants import STANDARD_AA
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms.atom_array import AddWithinChainInstanceResIdx, AddWithinPolyResIdxAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, ConvertToTorch
from atomworks.ml.transforms.bonds import (
    AddAF3TokenBondFeatures,
    AddTokenBondAdjacency,
    _atom_adjacency_to_token_adjacency,
    get_token_bond_adjacency,
)
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures
from atomworks.ml.transforms.filters import RemoveHydrogens
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_starts


# NOTE: This is a helper function to visualize the adjacency matrix and reduction
# useful for creating & debugging test cases
def _plot_adjacency_and_token_starts(adjacency: np.ndarray, token_start_end_idxs: np.ndarray = None):
    import matplotlib.pyplot as plt  # noqa: E402, lazy import

    plt.matshow(adjacency)
    # plot horizontal & vertical lines at `token_start_idxs`
    if token_start_end_idxs is not None:
        for i in token_start_end_idxs:
            plt.axhline(i - 0.5, color="red")
            plt.axvline(i - 0.5, color="red")
    plt.show()


def _adjacency_to_token_adjacency_slow(adjacency: np.ndarray, token_start_end_idxs: np.ndarray) -> np.ndarray:
    """Helper function to compute the token bond adjacency matrix from the atom bond adjacency matrix. Robust but slow."""
    token_start_idxs = token_start_end_idxs[:-1]
    token_end_idxs = token_start_end_idxs[1:]
    token_adjacency = np.zeros((len(token_start_idxs), len(token_start_idxs)), dtype=np.bool_)

    # Collapse all token blocks via `any`
    for i, (start_i, end_i) in enumerate(zip(token_start_idxs, token_end_idxs, strict=False)):
        for j, (start_j, end_j) in enumerate(zip(token_start_idxs, token_end_idxs, strict=False)):
            token_adjacency[i, j] = np.any(adjacency[start_i:end_i, start_j:end_j])

    # Remove diagonal
    np.fill_diagonal(token_adjacency, False)
    return token_adjacency


# Manual test-case, to inspect, run the below:
# _plot_adjacency_and_token_starts(TEST_CASE["adjacency"], TEST_CASE["token_start_end_idxs"])
# _plot_adjacency_and_token_starts(TEST_CASE["token_adjacency"])
TEST_CASE = {
    "adjacency": np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ]
    ),
    "token_start_end_idxs": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 27, 30]),
    "token_adjacency": np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ]
    ),
}


def test_adjacency_to_token_adjacency():
    token_adjacency = _atom_adjacency_to_token_adjacency(
        TEST_CASE["adjacency"], TEST_CASE["token_start_end_idxs"]
    ).astype(int)
    assert np.all(token_adjacency == TEST_CASE["token_adjacency"])


@pytest.mark.parametrize("pdb_id", ["6lyz", "4js1"])
def test_bond_adjaceny_transform(pdb_id):
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]
    assert np.all(
        get_token_bond_adjacency(atom_array)
        == _adjacency_to_token_adjacency_slow(
            atom_array.bonds.adjacency_matrix(), get_token_starts(atom_array, add_exclusive_stop=True)
        )
    ), "Token bond adjacency is not correct"


@pytest.mark.parametrize("pdb_id", ["6lyz", "4js1"])
def test_bond_adjaceny_transform_slow(pdb_id):
    data = cached_parse(pdb_id)
    pipe = Compose(
        [
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA,
                move_atomized_part_to_end=True,
            ),
            AddTokenBondAdjacency(),
        ]
    )
    data = pipe(data)
    atom_array = data["atom_array"]

    assert np.all(
        data["token_bond_adjacency"]
        == _adjacency_to_token_adjacency_slow(
            atom_array.bonds.adjacency_matrix(), get_token_starts(atom_array, add_exclusive_stop=True)
        )
    ), "Token bond adjacency is not correct"


AF3_TOKEN_BOND_FEATURES_TEST_CASES = [
    {
        "pdb_id": "5epq",
        "ligand-ligand-bonds": 352,  # include token bonds for atomized residues
        # NOTE: protein-ligand bonds should be 16, 4 ASNs, 2 bonds (one to the previous residue, one to the next)
        # and the bond matrix is symmetric, so we count each bond twice
        "protein-ligand-bonds": 16,
    }
]


@pytest.mark.parametrize("test_case", AF3_TOKEN_BOND_FEATURES_TEST_CASES)
def test_af3_token_bond_features(test_case: dict):
    data = cached_parse(test_case["pdb_id"])
    pipe = Compose(
        [
            RemoveHydrogens(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA,
                move_atomized_part_to_end=False,
            ),
            AddWithinChainInstanceResIdx(),
            AddWithinPolyResIdxAnnotation(),
            AddAF3TokenBondFeatures(),
            EncodeAF3TokenLevelFeatures(sequence_encoding=AF3SequenceEncoding()),
            ConvertToTorch(keys=["feats"]),
        ],
        track_rng_state=False,
    )
    data = pipe(data)
    feats = data["feats"]

    # check that the token bond adjacency matrix is symmetric
    assert torch.all(feats["token_bonds"] == feats["token_bonds"].T), "Token bond adjacency matrix is not symmetric"

    # Check that there are no protein-protein bonds
    token_starts = get_token_starts(data["atom_array"])
    not_atomized_token = torch.tensor(~data["atom_array"].atomize[token_starts])
    not_atomized_token_2d = torch.outer(not_atomized_token, not_atomized_token)
    assert torch.sum(feats["token_bonds"][not_atomized_token_2d]) == 0

    # Check that the number of ligand-ligand bonds is correct
    is_atomized_token = torch.tensor(data["atom_array"].atomize[token_starts])
    is_atomized_token_2d = torch.outer(is_atomized_token, is_atomized_token)

    assert torch.sum(feats["token_bonds"][is_atomized_token_2d]) == test_case["ligand-ligand-bonds"]

    # Check that the number of protein-ligand bonds is correct
    feats["token_bonds"][is_atomized_token_2d] = False
    assert torch.sum(feats["token_bonds"]) == test_case["protein-ligand-bonds"]


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
