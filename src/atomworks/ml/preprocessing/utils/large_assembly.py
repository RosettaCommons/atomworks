"""Large assembly interface sampling utilities.

Implements the AF3 supplement sampling strategy for large assemblies:
"For bioassemblies with greater than 20 chains, we select a random interface token
(with a centre atom <15 Å to the centre atom of a token in another chain) and select
the closest 20 chains to this token based on minimum distance between any tokens centre atom."
"""

import logging

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from atomworks.ml.utils.token import get_af3_token_center_masks

logger = logging.getLogger(__name__)


def get_interface_atoms(
    atom_array: AtomArray,
    pn_unit_1_iid: str,
    pn_unit_2_iid: str,
    cutoff_distance: float = 15.0,
) -> np.ndarray:
    """Identify atoms at the interface between two PN units.

    Args:
        atom_array: The atom array containing the full structure.
        pn_unit_1_iid: First PN unit instance ID.
        pn_unit_2_iid: Second PN unit instance ID.
        cutoff_distance: Distance threshold in Angstroms for interface definition.

    Returns:
        Boolean mask of atoms within cutoff_distance of the other PN unit.
    """
    mask_1 = (atom_array.pn_unit_iid == pn_unit_1_iid) & (atom_array.occupancy > 0)
    mask_2 = (atom_array.pn_unit_iid == pn_unit_2_iid) & (atom_array.occupancy > 0)

    if not np.any(mask_1) or not np.any(mask_2):
        return np.zeros(len(atom_array), dtype=bool)

    tree_1 = KDTree(atom_array.coord[mask_1])
    tree_2 = KDTree(atom_array.coord[mask_2])

    dists = tree_1.sparse_distance_matrix(tree_2, max_distance=cutoff_distance, output_type="coo_matrix")

    is_at_interface = np.zeros(len(atom_array), dtype=bool)
    indices_1 = np.where(mask_1)[0]
    indices_2 = np.where(mask_2)[0]
    is_at_interface[indices_1[np.unique(dists.row)]] = True
    is_at_interface[indices_2[np.unique(dists.col)]] = True

    return is_at_interface


def find_closest_pn_units(
    atom_array: AtomArray,
    reference_coord: np.ndarray,
    n_closest: int = 20,
) -> list[str]:
    """Find the closest PN units based on minimum token center distance.

    Args:
        atom_array: The atom array containing the full structure.
        reference_coord: 3D coordinates of the reference point (shape: (3,)).
        n_closest: Number of closest PN units to return.

    Returns:
        List of PN unit IIDs sorted by distance (closest first).
    """
    token_center_mask = get_af3_token_center_masks(atom_array, enforce_one_per_token=False)
    token_coords = atom_array.coord[token_center_mask]
    token_iids = atom_array.pn_unit_iid[token_center_mask]

    valid_mask = np.isfinite(token_coords).all(axis=1)
    if not np.any(valid_mask):
        return []

    valid_coords = token_coords[valid_mask]
    valid_iids = token_iids[valid_mask]

    distances = np.linalg.norm(valid_coords - reference_coord, axis=1)

    unique_iids, inverse_indices = np.unique(valid_iids, return_inverse=True)

    sort_idx = np.argsort(inverse_indices)
    sorted_distances = distances[sort_idx]
    sorted_inverse = inverse_indices[sort_idx]

    group_starts = np.concatenate([[0], np.where(np.diff(sorted_inverse) != 0)[0] + 1])
    min_distances = np.minimum.reduceat(sorted_distances, group_starts)

    sorted_idx = np.argsort(min_distances)
    return [unique_iids[i] for i in sorted_idx[:n_closest]]


def sample_large_assembly_interface(
    atom_array: AtomArray,
    contacting_pairs: list[tuple[str, str]],
    interface_cutoff_distance: float = 15.0,
    n_closest: tuple[int, ...] = (20, 50),
    seed: int | None = None,
) -> dict[int, set[str]] | None:
    """Sample an interface and find closest PN units for large assemblies.

    Args:
        atom_array: The filtered atom array for the assembly.
        contacting_pairs: List of polymer-polymer contacting pairs (pn_unit_iids).
        interface_cutoff_distance: Cutoff for interface atom identification.
        n_closest: Tuple of n values for finding closest PN units.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping n -> set of closest pn_unit_iids, or None if sampling failed.
    """
    rng = np.random.RandomState(seed)

    if not contacting_pairs:
        logger.warning("No polymer-polymer interfaces found for large assembly")
        return None

    idx = rng.randint(len(contacting_pairs))
    pn_unit_1_iid, pn_unit_2_iid = contacting_pairs[idx]

    is_interface = get_interface_atoms(
        atom_array, pn_unit_1_iid, pn_unit_2_iid, cutoff_distance=interface_cutoff_distance
    )

    interface_indices = np.where(is_interface)[0]
    if len(interface_indices) == 0:
        logger.warning("No interface atoms found; falling back to None.")
        return None

    sampled_idx = interface_indices[rng.randint(len(interface_indices))]
    reference_coord = atom_array.coord[sampled_idx]

    return {n: set(find_closest_pn_units(atom_array, reference_coord, n_closest=n)) for n in n_closest}
