"""Transforms that add masks for an AtomArray to the data"""

import logging
import warnings
from typing import Any

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger("atomworks.ml")


def compute_spatial_knn_mask(coords: np.ndarray, k: int) -> np.ndarray:
    """Compute the spatial KNN mask for an atom array"""
    assert coords.ndim == 2, "Coordinates must be a 2D array"
    num_atoms = coords.shape[0]

    # ... filter out 'nan' or 'inf' coordinates as calculating on
    #     non-nan coordinates will fail and no neighbors can be assigned to
    #     the atoms without coordinates
    is_finite = np.isfinite(coords).all(axis=1)
    idx_finite = np.where(is_finite)[0]
    if len(idx_finite) < num_atoms:
        warnings.warn(
            "Some atoms have no coordinates, they will not receive any neighbors in the spatial KNN mask", stacklevel=2
        )
    assert len(idx_finite) > k + 1, (
        f"Not enough atoms to calculate KNN mask with {k} neighbors, "
        f"but only {len(idx_finite)} atoms with coordinates found."
    )

    # ... get the k+1 nearest neighbors for each atom (self included)
    kdtree = KDTree(coords[idx_finite])
    knn_distances_, knn_indices = kdtree.query(coords[idx_finite], k=k + 1)

    # ... convert indices into boolean masks (atoms which had all 'nan' coords
    #     will have no neighbors and will be excluded from the mask)
    mask = np.zeros((num_atoms, num_atoms), dtype=bool)
    # ... map indices to full array space
    rows = idx_finite[:, None]  # Shape: (n_finite, 1)
    cols = idx_finite[knn_indices]  # Shape: (n_finite, k+1)
    mask[rows, cols] = True

    # ... set diagonal to 0 to exclude 'self' in mask
    np.fill_diagonal(mask, False)

    # ... check each atom that had coordinates was assigned k neighbors
    assert np.all(mask[idx_finite].sum(axis=1) == k), "Not all rows have k neighbors."
    # ... check diagonal is zero
    assert np.all(mask.diagonal() == 0), "Diagonal is not zero."

    return mask


class AddSpatialKNNMask(Transform):
    """
    Generate a spatial k-nearest neighbors mask for each atom in the input atom array
    and add it to the data with the key 'spatial_knn_masks' (shape: (n_atoms, n_atoms)).

    This mask is e.g. used as a local attention mask in diffusion for the input sequence
    based on given coordinates.

    Args:
        num_neighbors (int): The number of neighbors to keep.
        max_atoms_in_crop (int): The maximum allowed number of atoms
          in the crop. This transform builds an `(n_atoms, n_atoms)` mask, so this
          limit on the number of atoms avoids unexpected memory baloons.
          The default of 40'000 atoms should allow any crop of <1'538 tokens to pass
          (worst case: RNA guanine, which has 26 heavy atoms per token,
          resulting in 39,988 atoms for a structure made up of only guanine)
    """

    def __init__(self, num_neighbors: int, max_atoms_in_crop: int = 40_000):
        self.num_neighbors = num_neighbors
        self.max_atoms_in_crop = max_atoms_in_crop

    def check_input(self, data: dict[str, Any]) -> None:
        """
        Check if the input data contains the required keys and types.
        Args:
            data (Dict[str, Any]): The input data dictionary.
        Raises:
            KeyError: If a required key is missing from the input data.
            TypeError: If a value in the input data is not of the expected type.
        """
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_iid"])

        assert (
            len(data["atom_array"].coord) < self.max_atoms_in_crop
        ), "Number of atoms in the atom array exceeds the maximum number of atoms in the crop"

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a local attention mask for the input sequence based on given coordinates. only keep k nearest neighbors

        Args:
            data (dict[str, Any]): The input data dictionary.

        Returns:
            dict[str, Any]: The output data dictionary with the added 'spatial_knn_masks' key
                - 'spatial_knn_masks' (np.ndarray): Boolean mask of shape (n_atoms, n_atoms)
                  where True indicates that the atom is a k-nearest neighbor of the other atom.
                  NOTE: atoms with no coordinates will not receive any neighbors in the mask
                  (i.e. a row of all False values)
        """
        atom_array = data["atom_array"]

        # ... compute the masks
        k_nn_masks = compute_spatial_knn_mask(atom_array.coord, self.num_neighbors)

        data["spatial_knn_masks"] = k_nn_masks
        return data
