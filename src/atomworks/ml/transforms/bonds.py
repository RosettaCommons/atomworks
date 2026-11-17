from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray

from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_atom_array_has_bonds,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import apply_segment_wise_2d, get_token_starts


def _atom_adjacency_to_token_adjacency(atom_adjacency: np.ndarray, token_start_end_idxs: np.ndarray) -> np.ndarray:
    """Helper function to compute the token bond adjacency matrix from the atom bond adjacency matrix."""
    # NOTE: This is separated out to allow for easy testing
    # reduce token segments
    token_adjacency = apply_segment_wise_2d(atom_adjacency, token_start_end_idxs, np.any)
    # remove diagonal (bonds to self)
    np.fill_diagonal(token_adjacency, False)
    return token_adjacency


def get_token_bond_adjacency(atom_array: AtomArray) -> np.ndarray:
    """Computes the token bond adjacency matrix from the atom bond adjacency matrix.

    This is done by performing a block-wise reduction of the atom adjacency matrix,
    where block (i, j) is the sub-matrix of the atom adjacency matrix for bonds
    between atoms of token i and j. The reduction is performed by `np.any`, which
    returns True if at least one bond exists between the two tokens.
    """
    atom_adjacency = atom_array.bonds.adjacency_matrix()
    token_start_end_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    return _atom_adjacency_to_token_adjacency(atom_adjacency, token_start_end_idxs)


class AddTokenBondAdjacency(Transform):
    """
    Adds the token bond adjacency matrix to the data.

    This transform computes the token bond adjacency matrix from the atom bond adjacency matrix
    and adds it to the data dictionary under the key `token_bond_adjacency`.

    The token bond adjacency matrix is a binary [`n_tokens`, `n_tokens`] matrix where element (i, j) is True if there is
    at least one bond between any atom in token i and any atom in token j, and False otherwise.

    Depends on the definition of `tokens` and therefore has to be applied after any transform that alters what is
    considered a token (e.g. `AtomizeByCCDName`) or that changes the order or number of tokens. By default, a token
    is defined as a residue in the input `AtomArray`.

    Raises:
        AssertionError: If the input data does not contain the required keys or types.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_has_bonds(data)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_bond_adjacency = get_token_bond_adjacency(atom_array)
        data["token_bond_adjacency"] = token_bond_adjacency
        return data


def get_af3_token_bond_features(atom_array: AtomArray) -> np.ndarray:
    """
    Generates AF3-style token bond features for an AtomArray.

    From AF3:
        Returns a 2D matrix indicating if there is a bond between any atom in
        token i and token j, restricted to just polymer-ligand and ligand-ligand
        bonds.

    Note:
        Distance-based bond filtering is handled during parsing via `filter_bonds_by_distance`.

    Args:
        - atom_array (AtomArray): The input AtomArray containing atomic coordinates and bond information.

    Returns:
        - np.ndarray: A boolean matrix where True indicates a bond between tokens that meets the specified criteria.
    """
    token_start_end_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    token_starts = token_start_end_idxs[:-1]
    token_bonds = _atom_adjacency_to_token_adjacency(atom_array.bonds.adjacency_matrix(), token_start_end_idxs)

    # remove poly-poly bonds
    is_poly_poly_bond = np.outer(~atom_array.atomize[token_starts], ~atom_array.atomize[token_starts])
    token_bonds[is_poly_poly_bond] = False
    return token_bonds


class AddAF3TokenBondFeatures(Transform):
    """
    Transform that generates AF3-style token bond features for an AtomArray.

    This transform creates a 2D matrix indicating if there is a bond between any atom in
    token i and token j, restricted to just polymer-ligand and ligand-ligand bonds.

    Note:
        Distance-based bond filtering is handled during parsing via `filter_bonds_by_distance`.
        The `distance_cutoff` argument has been removed - if you were using it, the filtering
        now happens automatically in parse.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def __init__(self, **kwargs):
        if "distance_cutoff" in kwargs:
            raise DeprecationWarning(
                "The `distance_cutoff` argument has been removed from AddAF3TokenBondFeatures. "
                "Distance-based bond filtering is now handled during parsing via `filter_bonds_by_distance`. "
                "Please remove this argument from your code."
            )

    def check_input(self, data: dict) -> None:
        check_atom_array_has_bonds(data)
        check_atom_array_annotation(data, ["is_polymer", "atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        af3_token_bond_features = get_af3_token_bond_features(atom_array)

        if "feats" not in data:
            data["feats"] = {}

        data["feats"]["token_bonds"] = af3_token_bond_features

        return data


class AddAtomLevelBondAdjacencyMatrix(Transform):
    """
    Adds the atom-level bond adjacency matrix to the data as a feature.

    This transform uses Biotite's adjacency_matrix() function to create a binary matrix
    where element (i, j) is 1 if atoms i and j are bonded, and 0 otherwise.

    The matrix is added to the data dictionary under data["feats"]["atom_level_bond_adjacency"].
    """

    def check_input(self, data: dict) -> None:
        check_atom_array_has_bonds(data)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Get the bond adjacency matrix from Biotite
        bond_adjacency_matrix = atom_array.bonds.adjacency_matrix().astype(np.int8)

        if "feats" not in data:
            data["feats"] = {}

        data["feats"]["atom_level_bond_adjacency"] = bond_adjacency_matrix

        return data
