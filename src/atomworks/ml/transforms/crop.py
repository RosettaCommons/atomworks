import itertools
import logging
from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from atomworks.common import exists
from atomworks.io.transforms.atom_array import is_any_coord_nan
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import atom_id_to_atom_idx, atom_id_to_token_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_af3_token_center_coords,
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

logger = logging.getLogger("atomworks.ml")


class CropTransformBase(Transform):
    """
    Base class for crop-type transforms.
    """

    def __init__(self, annotate_crop_boundary: bool = False, crop_boundary_radius: float = 6.0, **kwargs):
        self.annotate_crop_boundary = annotate_crop_boundary
        self.crop_boundary_radius = crop_boundary_radius

    def _validate(self) -> None:
        assert self.crop_size > 0, "Crop size must be greater than 0"

    def __call__(self, data: dict) -> dict:
        if self.annotate_crop_boundary:
            atom_array = data["atom_array"]
            atom_array.set_annotation("precrop_hash", compute_local_hash(atom_array, self.crop_boundary_radius))

        data = super().__call__(data)

        if self.annotate_crop_boundary:
            atom_array = data["atom_array"]
            postcrop_hash = compute_local_hash(atom_array, self.crop_boundary_radius)
            atom_array.set_annotation("at_crop_boundary", atom_array.precrop_hash != postcrop_hash)
            # ... delete the precrop hash
            atom_array.del_annotation("precrop_hash")

        return data


def compute_local_hash(atom_array: AtomArray, radius: float = 6.0) -> np.ndarray:
    """
    Compute a local hash for each atom in the atom array.

    Currently, the hash is the number of neighbours within a given radius.

    Args:
        atom_array (AtomArray): The atom array to compute the local hash for.
        radius (float): The radius to use for the local hash.

    Returns:
        np.ndarray: A numpy array of shape (n_atoms,) containing the local hash for each atom.
    """
    # ... build kdtree
    n_atoms = atom_array.array_length()
    is_valid = ~is_any_coord_nan(atom_array)
    kdtree = KDTree(atom_array.coord[is_valid])

    # ... query local neighbourhoods
    neighbour_idxs: list[list[int]] = kdtree.query_ball_tree(kdtree, r=radius)

    # ... use number of neighbours as hash (# TODO: elaborate this if needed)
    num_neighbours = np.zeros(n_atoms, dtype=int)
    num_neighbours[is_valid] = list(map(len, neighbour_idxs))

    return num_neighbours


def crop_contiguous_af2_multimer(iids: list[int | str], instance_lens: list[int], crop_size: int) -> dict:
    """
    Crop contiguous tokens from the given instances to reach the given crop size probabilistically.

    Implements the `crop_contiguous` (algorithm 1 in section 7.2.1) of AF2 Multimer and section 2.7.2 of AF3.

    Args:
        iids (list[int | str]): List of instance identifiers.
        instance_lens (list[int]): List of lengths corresponding to each instance.
        crop_size (int): Desired number of tokens to crop. Must be greater than 0.

    Returns:
        keep_tokens (dict[int | str, np.ndarray[bool]]): Dictionary mapping instance identifiers
            (iids) to crop masks (i.e. boolean arrays) indicating which tokens to keep.

    References:
        `AF2 Multimer <https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf>`_
        `AF3 <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_

    Example:
        >>> iids = [1, 2, 3]
        >>> instance_lens = [3, 4, 2]
        >>> crop_size = 5
        >>> result = crop_contiguous_af2_multimer(iids, instance_lens, crop_size)
        >>> print(result)
        # Output might look like (probabilistic!):
        # {
        #     3: array([False, True]),
        #     2: array([False, True, True, False]),
        #     1: array([True, True, False])
        # }
    """
    iids = np.asarray(iids)
    instance_lens = np.asarray(instance_lens)

    assert crop_size > 0, "Crop size must be greater than 0"
    assert len(iids) == len(
        instance_lens
    ), f"Number of instance IDs ({len(iids)}) must match number of instance lengths ({len(instance_lens)})"
    assert (
        len(iids) == np.unique(iids).size
    ), f"Instance IDs must be unique, but got {len(iids)} IDs with only {np.unique(iids).size} unique values"

    # randomly permute the order of the instances to avoid cropping bias
    permutation = np.random.permutation(len(iids))
    iids = iids[permutation]
    instance_lens = instance_lens[permutation]

    # init variables to keep track of remaining budget
    n_budget = crop_size  # ... number of tokens that can still be added to the crop
    n_available = np.sum(instance_lens)  # ... number of tokens that are still available in the remaining instances

    keep_tokens = {}
    for iid, instance_len in zip(iids, instance_lens, strict=False):
        if n_budget == 0:
            # ... early stop if budget is already exhausted
            break

        # ... after cropping the current instance, n_remaining tokens are still available
        n_available -= instance_len

        # Determine the min/max crop sizes
        # ... maximally take at most
        #   (1) the remaining budget or
        #   (2) all tokens of the current instance,
        # whichever is smaller
        crop_size_max = min(n_budget, instance_len)

        # ... take at least
        #  (1) 0 if there is still more than enough tokens available to reach the budget
        #  (2) how much would be needed to reach the budget if we took all remaining available tokens
        #  (3) or take all tokens of the current instance if we cannot reach the budget even if
        #   we take all remaining available tokens
        crop_size_min = min(instance_len, max(0, n_budget - n_available))

        # ... sample a crop size for this instance and update the budget
        n_crop_in_instance = np.random.randint(crop_size_min, crop_size_max + 1)
        n_budget -= n_crop_in_instance
        assert n_budget >= 0, "The budget cannot be negative!"

        # ... sample a crop start position for this instance
        crop_start = np.random.randint(0, instance_len - n_crop_in_instance + 1)

        keep_token = np.zeros(instance_len, dtype=bool)
        keep_token[crop_start : crop_start + n_crop_in_instance] = True

        # ... add the crop to the keep dictionary
        keep_tokens[iid] = keep_token

    return keep_tokens  # dict[int | str, np.ndarray[bool]]


def get_spatial_crop_center(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    cutoff_distance: float = 15.0,
    raise_if_missing_query: bool = True,
) -> np.ndarray:
    """
    Sample a crop center from a spatial region of the atom array.

    Implements the selection of a crop center as described in AF3.
        In this procedure, polymer residues and ligand atoms are selected that
        are within close spatial distance of an interface atom. The interface
        atom is selected at random from the set of token centre atoms (defined
        in subsection 2.6) with a distance under 15 Å to another chain's token
        centre atom. For examples coming out of the Weighted PDB or Disordered
        protein PDB complex datasets, where a preferred chain or interface is
        provided (subsection 2.5), the reference atom is selected at random
        from interfacial token centre atoms that exist within this chain or
        interface.

    Args:
        atom_array (AtomArray): The array containing atom information.
        query_pn_unit_iids (list[str]): List of PN unit instance IDs to query.
        cutoff_distance (float, optional): The distance cutoff to consider for spatial proximity. Defaults to 15.0.
        raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
            query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
            pn_unit will be selected for the crop center.

    Returns:
        np.ndarray: A boolean mask indicating the crop center.
    """
    # ... get mask for query polymer/non-polymer unit
    is_query_pn_unit = np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)

    # ... get mask for occupied atoms
    is_occupied = atom_array.occupancy > 0

    # ... optionally provide a fallback when not all query pn_units are present
    if not raise_if_missing_query:
        available_query_pn_unit_iids = np.unique(atom_array.pn_unit_iid[is_query_pn_unit])

        # If only one of the query pn_units is present, we will just use that
        if len(available_query_pn_unit_iids) == 1 and len(query_pn_unit_iids) > 1:
            query_pn_unit_iids = available_query_pn_unit_iids
            logger.warning(
                f"Falling back to only available query pn_unit ({query_pn_unit_iids[0]}) for the crop center."
            )

        # If none of the query pn_units are present, we will randomly select one
        elif len(available_query_pn_unit_iids) == 0:
            all_available_pn_unit_iids = np.unique(atom_array.pn_unit_iid)
            query_pn_unit_iids = np.random.choice(all_available_pn_unit_iids, size=1)
            logger.warning(f"Falling back to randomly-selected pn_unit ({query_pn_unit_iids[0]}) for the crop center.")

        # Update the mask for query pn_unit
        is_query_pn_unit = np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)

    if len(query_pn_unit_iids) == 1:
        # If there's only one query unit, we don't need to check for spatial proximity,
        # so we can just return the mask for the query unit.
        can_be_crop_center = is_query_pn_unit & is_occupied
        assert np.any(
            can_be_crop_center
        ), f"No crop center found! It appears `query_pn_unit_iid` {query_pn_unit_iids} is not in the atom array or unresolved."

        return can_be_crop_center

    is_at_interface = np.zeros_like(is_query_pn_unit, dtype=bool)
    for pn_unit_1_iid, pn_unit_2_iid in itertools.combinations(query_pn_unit_iids, 2):
        # ... get mask, indices, and kdtree for pn_unit_1
        pn_unit_1_mask = (atom_array.pn_unit_iid == pn_unit_1_iid) & is_occupied
        pn_unit_1_indices = np.where(pn_unit_1_mask)[0]
        _tree1 = KDTree(atom_array.coord[pn_unit_1_mask])

        # ... get mask, indices, and kdtree for pn_unit_2
        pn_unit_2_mask = (atom_array.pn_unit_iid == pn_unit_2_iid) & is_occupied
        pn_unit_2_indices = np.where(pn_unit_2_mask)[0]
        _tree2 = KDTree(atom_array.coord[pn_unit_2_mask])

        dists = _tree1.sparse_distance_matrix(_tree2, max_distance=cutoff_distance, output_type="coo_matrix")

        # ... update the interface mask (by converting the local idxs to the global idxs)
        is_at_interface[pn_unit_1_indices[np.unique(dists.row)]] = True
        is_at_interface[pn_unit_2_indices[np.unique(dists.col)]] = True

    # ... assemble final crop mask
    can_be_crop_center = is_query_pn_unit & is_at_interface & is_occupied

    assert np.any(can_be_crop_center), "No crop center found!"
    return can_be_crop_center


def get_spatial_crop_mask(
    coord: np.ndarray, crop_center_idx: int, crop_size: int, jitter_scale: float = 1e-3
) -> np.ndarray:
    """
    Crop spatial tokens around a given `crop_center` by keeping the `crop_size` nearest neighbors (with jitter).

    Implements the `crop_spatial` (algorithm 2 in section 7.2.1) of AF2 Multimer and AF3

    Args:
        coord (np.ndarray): A 2D numpy array of shape (N, 3) representing the 3D token-level coordinates.
            Coordinates are expected to be in Angstroms.
        crop_center_idx (int): The index of the token to be used as the center of the crop.
        crop_size (int): The number of nearest neighbors to include in the crop.
        jitter_scale (float): The scale of the jitter to add to the coordinates.

    Returns:
        crop_mask (np.ndarray): A boolean mask of shape (N,) where True indicates that the token is within the crop.

    References:
        `AF2 Multimer <https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf>`_
        `AF3 <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_

    Example:
        >>> coord = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        >>> crop_center_idx = 1
        >>> crop_size = 2
        >>> crop_mask = get_spatial_crop_mask(coord, crop_center_idx, crop_size)
        >>> print(crop_mask)
        [ True  True False False]
    """
    assert coord.ndim == 2, f"Expected coord to be 2-dimensional, got {coord.ndim} dimensions"
    assert coord.shape[1] == 3, f"Expected coord to have 3 coordinates per point, got {coord.shape[1]}"
    assert (
        crop_center_idx < coord.shape[0]
    ), f"Crop center index {crop_center_idx} is out of bounds for coord array of length {coord.shape[0]}"
    assert crop_size > 0, f"Crop size must be positive, got {crop_size}"
    assert jitter_scale >= 0, f"Jitter scale must be non-negative, got {jitter_scale}"

    # Add small jitter to coordinates to break ties
    if jitter_scale > 0:
        coord = coord + np.random.normal(scale=jitter_scale, size=coord.shape)

    # ... get query center
    query_center = coord[crop_center_idx]

    # ... extract a mask for valid coordiantes (i.e. no `nan`'s, which indicate unknown token centers)
    #     including including unoccupied tokens in the crop
    is_valid = np.isfinite(coord).all(axis=1)

    # ... build a KDTree for efficient querying, excluding invalid coordinates
    tree = KDTree(coord[is_valid])

    # ... query the `crop_size` nearest neighbors of the crop center
    _, nearest_neighbor_idxs = tree.query(query_center, k=crop_size, p=2)
    # ... filter out missing neighbours (index equal to `tree.n`)
    nearest_neighbor_idxs = nearest_neighbor_idxs[nearest_neighbor_idxs < tree.n]

    # ... crop mask is True for the `crop_size` nearest neighbors of the crop center
    crop_mask = np.zeros(coord.shape[0], dtype=bool)
    is_valid_and_in_crop_idxs = np.where(is_valid)[0][nearest_neighbor_idxs]
    crop_mask[is_valid_and_in_crop_idxs] = True

    return crop_mask


class CropContiguousLikeAF3(CropTransformBase):
    """A transform that performs contiguous cropping similar to AF3.

    This class implements the contiguous cropping procedure as described in AF3. It selects a crop center
    from a contiguous region of the atom array and samples a crop around this center.

    WARNING: This transform is probabilistic if the atom array is larger than the crop size!

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf

    Attributes:
        crop_size (int): The maximum number of tokens to crop.
        keep_uncropped_atom_array (bool): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
        max_atoms_in_crop (int | None): Maximum number of atoms allowed in a crop. If None, no resizing is performed.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropSpatialLikeAF3",
        "CropContiguousLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def __init__(
        self, crop_size: int, keep_uncropped_atom_array: bool = False, max_atoms_in_crop: int | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.max_atoms_in_crop = max_atoms_in_crop
        self._validate()

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_iid", "atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        requires_crop = get_token_count(atom_array) > self.crop_size
        if requires_crop:
            # Extract chain data
            chain_iids = np.unique(atom_array.chain_iid)
            chain_n_tokens = [
                get_token_count(atom_array[atom_array.chain_iid == chain_iid]) for chain_iid in chain_iids
            ]

            # Sample crop as in AF2 multimer
            keep_token_dict = crop_contiguous_af2_multimer(chain_iids, chain_n_tokens, self.crop_size)

            # Turn crop information into atom-level mask
            is_token_start_idxs = get_token_starts(atom_array)
            is_token_in_crop = np.zeros_like(is_token_start_idxs, dtype=bool)
            for chain_iid, keep_token_idxs in keep_token_dict.items():
                chain_idxs = np.where(atom_array[is_token_start_idxs].chain_iid == chain_iid)[0]
                is_token_in_crop[chain_idxs[keep_token_idxs]] = True
            is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop)
        else:
            # ... no need to crop since the atom array is already small enough
            is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
            is_token_in_crop = np.ones(get_token_count(atom_array), dtype=bool)

        crop_info = {
            "type": self.__class__.__name__,
            "requires_crop": requires_crop,
            "crop_token_idxs": np.where(is_token_in_crop)[0],
            "crop_atom_idxs": np.where(is_atom_in_crop)[0],
        }

        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info,
            atom_array=atom_array,
            max_atoms=self.max_atoms_in_crop,
        )

        # Update data
        data["crop_info"] = crop_info
        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array
        data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]

        return data


def crop_spatial_like_af3(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    crop_size: int,
    jitter_scale: float = 1e-3,
    crop_center_cutoff_distance: float = 15.0,
    force_crop: bool = False,
    raise_if_missing_query: bool = True,
) -> dict:
    """Crop spatial tokens around a given `crop_center` by keeping the `crop_size` nearest neighbors (with jitter).

    Args:
        - atom_array (AtomArray): The atom array to crop.
        - query_pn_unit_iids (list[str]): List of query polymer/non-polymer unit instance IDs.
        - crop_size (int): The maximum number of tokens to crop.
        - jitter_scale (float, optional): Scale of jitter to apply when calculating distances.
            Defaults to 1e-3.
        - crop_center_cutoff_distance (float, optional): Maximum distance from query units to
            consider for crop center. Defaults to 15.0 Angstroms.
        - force_crop (bool, optional): Whether to force crop even if the atom array is already small enough.
            Defaults to False.
        - raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
            query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
            pn_unit will be selected for the crop center.

    Returns:
        dict: A dictionary containing crop information, including:
            - requires_crop (bool): Whether cropping was necessary.
            - crop_center_atom_id (int or np.nan): ID of the atom chosen as crop center.
            - crop_center_atom_idx (int or np.nan): Index of the atom chosen as crop center.
            - crop_center_token_idx (int or np.nan): Index of the token containing the crop center.
            - crop_token_idxs (np.ndarray): Indices of tokens included in the crop.
            - crop_atom_idxs (np.ndarray): Indices of atoms included in the crop.

    Note:
        This function implements the spatial cropping procedure as described in AlphaFold 3 and AlphaFold 2 Multimer.

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
    """
    token_segments = get_token_starts(atom_array, add_exclusive_stop=True)
    n_tokens = len(token_segments) - 1
    requires_crop = n_tokens > crop_size

    # ... get possible crop centers
    can_be_crop_center = get_spatial_crop_center(
        atom_array, query_pn_unit_iids, crop_center_cutoff_distance, raise_if_missing_query=raise_if_missing_query
    )

    # ... sample crop center atom
    crop_center_atom_id = np.random.choice(atom_array[can_be_crop_center].atom_id)
    crop_center_atom_idx = atom_id_to_atom_idx(atom_array, crop_center_atom_id)
    crop_center_token_idx = atom_id_to_token_idx(atom_array, crop_center_atom_id)

    # ... sample crop
    if force_crop or requires_crop:
        token_coords = get_af3_token_center_coords(atom_array)
        is_token_in_crop = get_spatial_crop_mask(
            token_coords, crop_center_token_idx, crop_size=crop_size, jitter_scale=jitter_scale
        )
        # ... spread token-level crop mask to atom-level
        is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop, token_starts=token_segments)
    else:
        # ... no need to crop since the atom array is already small enough
        is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
        is_token_in_crop = np.ones(n_tokens, dtype=bool)

    return {
        "requires_crop": requires_crop,  # whether cropping was necessary
        "crop_center_atom_id": crop_center_atom_id,  # atom_id of crop center
        "crop_center_atom_idx": crop_center_atom_idx,  # atom_idx of crop center
        "crop_center_token_idx": crop_center_token_idx,  # token_idx of crop center
        "crop_token_idxs": np.where(is_token_in_crop)[0],  # token_idxs in crop
        "crop_atom_idxs": np.where(is_atom_in_crop)[0],  # atom_idxs in crop
    }


def resize_crop_info_if_too_many_atoms(
    crop_info: dict,
    atom_array: AtomArray,
    max_atoms: int,
) -> dict:
    """
    Resizes crops that exceed the maximum allowed number of atoms by removing tokens based on the distance to the crop center.
    If no crop center is provided, the center of mass of the tokens in the crop is used as center.

    NOTE: This is mostly needed for AF3 when crops on nucleic acids have too many atoms to work with the current atom
    local attention when training on GPUs with less memory.

    Args:
        - crop_info (dict): Dictionary containing crop information. Must include:
            - crop_atom_idxs: Array of atom indices in the crop
            - crop_token_idxs: Array of token indices in the crop
        - atom_array (AtomArray): The atom array containing the full structure
        - max_atoms(int): Maximum number of atoms allowed in a crop. If None, no resizing is performed.

    Returns:
        dict: Updated crop_info dictionary with resized crop indices if necessary
    """
    assert "crop_atom_idxs" in crop_info, "crop_atom_idxs not found in crop"
    assert "crop_token_idxs" in crop_info, "crop_token_idxs not found in crop"
    crop_atom_idxs = crop_info["crop_atom_idxs"]
    crop_token_idxs = crop_info["crop_token_idxs"]
    crop_atom_array = atom_array[crop_atom_idxs]

    # Check if resizing is needed
    if not exists(max_atoms) or len(crop_atom_idxs) <= max_atoms:
        # ... no resizing needed
        return crop_info

    # Calculate distances to center token
    # ... get token center coordinates
    token_coords = get_af3_token_center_coords(crop_atom_array)  # [n_token, 3]
    if "crop_center_atom_idx" in crop_info:
        crop_center_coords = atom_array.coord[crop_info["crop_center_atom_idx"]]
    else:
        # ... use center of mass of tokens in crop as center coordinate
        crop_center_coords = np.mean(token_coords, axis=0)
    # ... calculate distances to center token
    dist_to_center = np.linalg.norm(token_coords - crop_center_coords, axis=1)
    sort_by_distance = np.argsort(dist_to_center)  # ascending

    # Calculate cumulative atoms find cut-off index at which we are within budget
    # ... get number of atoms per token
    n_atoms_per_token = apply_token_wise(
        array=crop_atom_array,
        data=np.ones(len(crop_atom_idxs), dtype=int),
        function=np.sum,
    )
    within_budget = np.cumsum(n_atoms_per_token[sort_by_distance]) <= max_atoms

    # Subset to tokens within budget
    # ...find the largest index that is within budget
    cutoff = np.max(np.where(within_budget)) + 1  # +1 because we want to include the cutoff index
    # ... get token indices within budget
    is_in_budget = sort_by_distance[:cutoff]
    token_idxs_in_budget = crop_token_idxs[is_in_budget]

    # Update atom idxs accordingly
    # ... create updated masks for chosen tokens
    is_chosen_token = np.zeros(get_token_count(atom_array), dtype=bool)
    is_chosen_token[token_idxs_in_budget] = True
    # ... get the atom indices that are within budget
    atom_idxs_in_budget = np.where(spread_token_wise(atom_array, is_chosen_token))[0]

    # Update crop info
    crop_info["crop_atom_idxs"] = atom_idxs_in_budget
    crop_info["crop_token_idxs"] = token_idxs_in_budget

    return crop_info


class CropSpatialLikeAF3(CropTransformBase):
    """
    A transform that performs spatial cropping similar to AF3 and AF2 Multimer.

    This class implements the spatial cropping procedure as described in AF3. It selects a crop center
    from a spatial region of the atom array and samples a crop around this center.

    WARNING: This transform is probabilistic if the atom array is larger than the crop size!

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf

    Attributes:
        crop_size (int): The maximum number of tokens to crop. Must be greater than 0.
        jitter_scale (float): The scale of the jitter to apply to the crop center. This is to break
            ties between atoms with the same spatial distance. Defaults to 1e-3.
        crop_center_cutoff_distance (float): The cutoff distance to consider for selecting crop
            centers. Measured in Angstroms. Defaults to 15.0.
        keep_uncropped_atom_array (bool): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AddGlobalAtomIdAnnotation", "AtomizeByCCDName"]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def __init__(
        self,
        crop_size: int,
        jitter_scale: float = 1e-3,
        crop_center_cutoff_distance: float = 15.0,
        keep_uncropped_atom_array: bool = False,
        force_crop: bool = False,
        max_atoms_in_crop: int | None = None,
        raise_if_missing_query: bool = True,
        **kwargs,
    ):
        """Initialize the CropSpatialLikeAF3 transform.

        Args:
            crop_size: The maximum number of tokens to crop. Must be greater than 0.
            jitter_scale: The scale of the jitter to apply to the crop center.
                This is to break ties between atoms with the same spatial distance. Defaults to 1e-3.
            crop_center_cutoff_distance: The cutoff distance to consider for
                selecting crop centers. Measured in Angstroms. Defaults to 15.0.
            keep_uncropped_atom_array: Whether to keep the uncropped atom array in the data.
                If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
                under the key `"atom_array"`. Defaults to `False`.
            force_crop: Whether to force crop even if the atom array is already small enough.
                Defaults to `False`.
            max_atoms_in_crop (int, optional): Maximum number of atoms allowed in a crop. If None, no resizing is performed.
                Defaults to None.
            raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
                query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
                pn_unit will be selected for the crop center.
        """
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.jitter_scale = jitter_scale
        self.crop_center_cutoff_distance = crop_center_cutoff_distance
        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.force_crop = force_crop
        self.max_atoms_in_crop = max_atoms_in_crop
        self.raise_if_missing_query = raise_if_missing_query
        self._validate()

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "atomize", "atom_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if data.get("query_pn_unit_iids"):
            query_pn_units = data["query_pn_unit_iids"]
        else:
            query_pn_units = np.unique(atom_array.pn_unit_iid)
            logger.info(f"No query PN unit(s) provided for spatial crop. Randomly selecting from {query_pn_units}.")

        crop_info = crop_spatial_like_af3(
            atom_array=atom_array,
            query_pn_unit_iids=query_pn_units,
            crop_size=self.crop_size,
            jitter_scale=self.jitter_scale,
            crop_center_cutoff_distance=self.crop_center_cutoff_distance,
            force_crop=self.force_crop,
            raise_if_missing_query=self.raise_if_missing_query,
        )
        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info,
            atom_array=atom_array,
            max_atoms=self.max_atoms_in_crop,
        )

        data["crop_info"] = {"type": self.__class__.__name__} | crop_info

        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array

        # Update data with cropped atom array
        data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]

        return data
