from typing import ClassVar

import numpy as np
import scipy
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

# Constants copied from `chemdata` to decouple the RF2AA repository from the atomworks.ml pipeline
RF2AA_NO_BOND = 0
RF2AA_SINGLE_BOND = 1
RF2AA_DOUBLE_BOND = 2
RF2AA_TRIPLE_BOND = 3
RF2AA_AROMATIC_BOND = 4
RF2AA_RESIDUE_BB_BOND = 5
RF2AA_RESIDUE_ATOM_BOND = 6


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


def _biotite_bond_types_to_rf2aa_bond_types(biotite_bond_types: np.ndarray) -> np.ndarray:
    """Converts Biotite bond types to RF2AA bond types."""
    rf2aa_bond_types = np.full_like(
        biotite_bond_types, fill_value=7, dtype=np.int8
    )  # 7 maps to "other" bond, which is not represented in the ChemData enum
    rf2aa_bond_types[biotite_bond_types == 0] = RF2AA_NO_BOND
    rf2aa_bond_types[biotite_bond_types == 1] = RF2AA_SINGLE_BOND
    rf2aa_bond_types[biotite_bond_types == 2] = RF2AA_DOUBLE_BOND
    rf2aa_bond_types[biotite_bond_types == 3] = RF2AA_TRIPLE_BOND
    rf2aa_bond_types[biotite_bond_types > 4] = RF2AA_AROMATIC_BOND
    return rf2aa_bond_types


def _create_rf2aa_bond_features_matrix(
    token_bond_adjacency: np.ndarray, token_is_atom: np.ndarray, atom_biotite_bond_type_matrix: np.ndarray
) -> np.ndarray:
    """
    Create the RF2AA bond features matrix based on token adjacency, atomized masks, and Biotite bond types.

    Args:
        token_bond_adjacency (np.ndarray): Adjacency matrix indicating inter-token bonds.
        token_is_atom (np.ndarray): Boolean array indicating if tokens are atoms (vs. residues).
        atom_biotite_bond_type_matrix (np.ndarray): Matrix of Biotite bond types between atoms.

    Returns:
        np.ndarray: The bond features matrix, encoded uses the RF2AA BondType convention.
    """

    # ...initialize the bond features matrix, defaulting to no bond
    bond_features_matrix = np.full_like(token_bond_adjacency, fill_value=RF2AA_NO_BOND, dtype=np.int8)

    # ...fill in the residue-residue token bonds
    # If a token isn't atomized, then it must be a residue (either a protein, RNA, or DNA)
    token_is_residue = ~token_is_atom
    residue_matrix = np.outer(token_is_residue, token_is_residue)
    bond_features_matrix[residue_matrix] = RF2AA_RESIDUE_BB_BOND

    # ...fill in the residue-atom bonds
    atom_residue_matrix = np.outer(token_is_residue, token_is_atom)
    atom_residue_matrix |= np.transpose(atom_residue_matrix)
    atom_residue_matrix &= token_bond_adjacency
    bond_features_matrix[atom_residue_matrix] = RF2AA_RESIDUE_ATOM_BOND

    # ...fill in the small molecule bonds
    rf2aa_atom_bond_matrix = _biotite_bond_types_to_rf2aa_bond_types(atom_biotite_bond_type_matrix)
    bond_features_matrix[np.ix_(token_is_atom, token_is_atom)] = rf2aa_atom_bond_matrix

    # ...apply the token_bond_adjacency mask to zero-out non-bonded interactions
    bond_features_matrix[~token_bond_adjacency] = RF2AA_NO_BOND

    return bond_features_matrix


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


class AddRF2AABondFeaturesMatrix(Transform):
    """Adds a matrix indicating the RF2AA bond type between two nodes to the data.

    This transform builds from the Biotite bond type, modifying as needed for residue-residue and residue-atom mappings.
    We then add the matrix to the data dictionary under the key "rf2aa_bond_features_matrix".

    From the RF2AA supplement, Supplementary Methods Table 8: Inputs to RFAA:

    bond_feats | (L, L, 7) Pairwise bond adjacency matrix. Pairs of residues are either
               | single, double, triple, aromatic, residue-residue, residue-atom, or other.

    Specifically, we map to the following enum, as described in ChemData:
    - 0 = No bonds
    - 1 = Single bond
    - 2 = Double bond
    - 3 = Triple bond
    - 4 = Aromatic
    - 5 = Residue-residue
    - 6 = Residue-atom
    - 7 = Other

    We build the matrix from the Biotite bond types.
    The Biotite BondType enum contains the following mapping:

    - ANY = 0
    - SINGLE = 1
    - DOUBLE = 2
    - TRIPLE = 3
    - QUADRUPLE = 4
    - AROMATIC_SINGLE = 5
    - AROMATIC_DOUBLE = 6
    - AROMATIC_TRIPLE = 7

    The the index -1 is used for non-bonded interactions.

    Reference:
        `Biotite BondType Documentation <https://www.biotite-python.org/apidoc/biotite.structure.BondType.html#biotite.structure.BondType>`_
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, AddTokenBondAdjacency]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["token_bond_adjacency", "atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_has_bonds(data)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_bond_adjacency = data["token_bond_adjacency"]
        token_is_atom = atom_array.atomize[get_token_starts(atom_array)]

        # Get bond type matrix for atomized tokens
        atom_bond_type_matrix = atom_array.bonds.bond_type_matrix()  # [n_atoms, n_atoms]
        atom_biotite_bond_type_matrix = atom_bond_type_matrix[np.ix_(atom_array.atomize, atom_array.atomize)]

        # Create bond features matrix
        bond_features_matrix = _create_rf2aa_bond_features_matrix(
            token_bond_adjacency=token_bond_adjacency,
            token_is_atom=token_is_atom,
            atom_biotite_bond_type_matrix=atom_biotite_bond_type_matrix,
        )

        data["rf2aa_bond_features_matrix"] = bond_features_matrix

        return data


class AddRF2AATraversalDistanceMatrix(Transform):
    """Generates a matrix indicating the minimum amount of bonds to traverse between two nodes.

    We define the traversal distance between two protein nodes as zero.
    Sets the "traversal_distance_matrix" key in the data dictionary.

    From the RF2AA supplement, Supplementary Methods Table 8: Inputs to RFAA:

    dist_matrix | (L, L) Minimum amount of bonds to traverse between two nodes.
                | This is 0 between all protein nodes.
    """

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["rf2aa_bond_features_matrix"])

    def forward(self, data: dict) -> dict:
        rf2aa_bond_features_matrix = data["rf2aa_bond_features_matrix"]

        # RF2AA uses the following bond mapping, as described in ChemData:
        #     - 0 = No bonds
        #     - 1 = Single bond
        #     - 2 = Double bond
        #     - 3 = Triple bond
        #     - 4 = Aromatic
        #     - 5 = Residue-residue
        #     - 6 = Residue-atom
        #     - 7 = Other

        # Reduce the bond features matrix to only include atom-atom bonds
        atom_bonds = (rf2aa_bond_features_matrix > 0) * (rf2aa_bond_features_matrix < 5)

        # Compute the shortest path distance matrix using scipy
        traversal_distance_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds, directed=False)

        # Add to the data dictionary
        # NOTE: This matrix will have infinity values, which are handled downstream by the model
        data["rf2aa_traversal_distance_matrix"] = traversal_distance_matrix

        return data


def get_bond_distances(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond distance (adjacency) list as a 1D array."""
    coords = atom_array.coord
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    return np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)


def get_bond_distance_matrix(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond adjacency matrix with bond distances as values."""
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    bond_distances = get_bond_distances(atom_array)
    bond_distance_matrix = np.full((atom_array.array_length(), atom_array.array_length()), np.inf)
    bond_distance_matrix[atom1_idxs, atom2_idxs] = bond_distances
    bond_distance_matrix[atom2_idxs, atom1_idxs] = bond_distances
    return bond_distance_matrix


def get_af3_token_bond_features(atom_array: AtomArray, distance_cutoff: float = 2.4) -> np.ndarray:
    """
    Generates AF3-style token bond features for an AtomArray.
    For bonds between multi-atom tokens (i.e., residues), we define the "bond distance" as the minimum distance between an atom of one token and any atom of the other token.

    From AF3:
        Returns a 2D matrix indicating if there is a bond between any atom in
        token i and token j, restricted to just polymer-ligand and ligand-ligand
        bonds and bonds less than 2.4 Å during training.

    Args:
        - atom_array (AtomArray): The input AtomArray containing atomic coordinates and bond information.
        - distance_cutoff (float, optional): The maximum distance (in Angstroms) for considering a bond. Defaults to 2.4.

    Returns:
        - np.ndarray: A boolean matrix where True indicates a bond between tokens that meets the specified criteria.
    """
    token_start_end_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    token_starts = token_start_end_idxs[:-1]
    token_bonds = apply_segment_wise_2d(get_bond_distance_matrix(atom_array), token_start_end_idxs, np.min)

    # remove bonds above distance cutoff
    token_bonds = token_bonds < distance_cutoff

    # remove token self-bonds
    np.fill_diagonal(token_bonds, False)

    # remove poly-poly bonds
    is_poly_poly_bond = np.outer(~atom_array.atomize[token_starts], ~atom_array.atomize[token_starts])
    token_bonds[is_poly_poly_bond] = False
    return token_bonds


class AddAF3TokenBondFeatures(Transform):
    """
    Transform that generates AF3-style token bond features for an AtomArray.

    This transform creates a 2D matrix indicating if there is a bond between any atom in
    token i and token j, restricted to just polymer-ligand and ligand-ligand bonds and
    bonds less than a specified distance cutoff.

    Args:
        - distance_cutoff (float, optional): The maximum distance (in Angstroms) for considering a bond.
            Defaults to 2.4.

    Returns:
        - dict: A dictionary containing the input data and the new 'af3_token_bond_features' key with
            the computed boolean matrix.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def __init__(self, distance_cutoff: float = 2.4):
        self.distance_cutoff = distance_cutoff

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_has_bonds(data)
        check_atom_array_annotation(data, ["is_polymer", "atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        af3_token_bond_features = get_af3_token_bond_features(atom_array, self.distance_cutoff)

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
