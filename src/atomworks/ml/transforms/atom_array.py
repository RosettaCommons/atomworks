"""Transforms on atom arrays."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure import AtomArray, get_residue_count, spread_residue_wise

from atomworks.enums import ChainType
from atomworks.io.utils.testing import has_annotation
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils import nested_dict
from atomworks.ml.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

logger = logging.getLogger(__name__)


# Convenience utils
# NOTE: We should move to the `utils` folder
def get_chain_instance_starts(array: AtomArray, add_exclusive_stop: bool = False) -> np.ndarray:
    """Get indices for an atom array, each indicating the beginning of a new chain instance (chain_iid).

    Inspired by `biotite.strucutre.get_chain_starts`.

    Args:
    - atom_array (AtomArray): The atom array to get the chain_iid starts from.
    - add_exclusive_stop (bool, optional): If True, add an exclusive stop to the chain_iid starts for the last chain instance. Defaults to False.

    Returns:
    - np.ndarray: An array of indices indicating the beginning of each chain instance.
    """
    # This mask is 'true' at indices where the chain_iid changes
    chain_iid_changes = array.chain_iid[1:] != array.chain_iid[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new chain instance
    chain_iid_starts = np.where(chain_iid_changes)[0] + 1

    # The first chain instance is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], chain_iid_starts, [array.array_length()]))
    else:
        return np.concatenate(([0], chain_iid_starts))


def chain_instance_iter(array: AtomArray) -> Iterator[AtomArray]:
    """Returns an iterator over the chain instances (chain_iid) in the atom array.

    This will match `biotite.structure.chain_iter` in the case where there are no transformations.
    """
    # The exclusive stop is appended to the residue starts
    starts = get_chain_instance_starts(array, add_exclusive_stop=True)
    return struc.segments.segment_iter(array, starts)


def atom_id_to_atom_idx(atom_array: AtomArray, atom_id: int) -> int:
    """Convert an atom ID to an atom index in the given array."""
    atom_idx = np.where(atom_array.atom_id == atom_id)[0]
    assert len(atom_idx) == 1, f"Expected 1 index for atom_id {atom_id}, got {atom_idx}"
    return atom_idx[0]


def atom_id_to_token_idx(atom_array: AtomArray, atom_id: int) -> int:
    """Convert an atom ID to a token index in the given array."""
    atom_idx = atom_id_to_atom_idx(atom_array, atom_id)

    # get the sorted token start idxs
    token_start_idxs = get_token_starts(atom_array)

    # the atom's token_idx is the matching or next lower token
    token_idx = np.searchsorted(token_start_idxs, atom_idx, side="right") - 1

    return token_idx


def apply_and_spread_residue_wise(
    atom_array: AtomArray, data: np.ndarray, function: Callable[[np.ndarray], np.generic], axis: int | None = None
) -> np.ndarray:
    """Apply a function residue wise and then spread the result to the atoms."""
    return struc.spread_residue_wise(atom_array, struc.apply_residue_wise(atom_array, data, function, axis))


def apply_and_spread_chain_wise(
    atom_array: AtomArray, data: np.ndarray, function: Callable[[np.ndarray], np.generic], axis: int | None = None
) -> np.ndarray:
    """Apply a function chain wise and then spread the result to the atoms."""
    return struc.spread_chain_wise(atom_array, struc.apply_chain_wise(atom_array, data, function, axis))


def _renumber_res_ids_around_reference(
    atom_array: AtomArray, ref: AtomArray, where: Literal["before", "after"]
) -> AtomArray:
    """Renumbers the residues in an AtomArray based on a reference.

    Residues in the new atom array will be continuous, with either the beginning or end
    lining up to the reference array's start or end.
    Assumes that the reference has correct residue ids and order.

    TODO: Shall we delete?
    """
    _res_start_stop_idxs = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    n_res = len(_res_start_stop_idxs) - 1
    if where == "before":
        ref_idx = ref.res_id[0]
        new_ids = np.arange(ref_idx - n_res, ref_idx)
    elif where == "after":
        ref_idx = ref.res_id[-1]
        new_ids = np.arange(ref_idx + 1, ref_idx + n_res + 1)
    else:
        raise ValueError(f"{where=} is not allowed. Must be one of 'before', 'after'")

    atom_array.res_id = struc.segments.spread_segment_wise(_res_start_stop_idxs, new_ids)
    return atom_array


class AddMoleculeSymmetricIdAnnotation(Transform):
    """Adds the `molecule_symmetric_id` annotation to the AtomArray.

    For a molecule, the symmetric_id is a unique integer within the set of molecules that share the same molecule_entity.

    Example:
    - If molecule_entity 0 has 3 molecules, they will have symmetric_ids 0, 1, 2.
    """

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["molecule_entity", "molecule_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Add the molecule_symmetric_id annotation to the AtomArray
        atom_array.add_annotation("molecule_symmetric_id", dtype=np.uint16)

        molecule_iids = np.unique(atom_array.molecule_iid)
        molecule_entity_counts = {}
        # Loop through every molecule
        for molecule_iid in molecule_iids:
            mask = atom_array.molecule_iid == molecule_iid

            # Get the molecule_entity (same for all atoms in the molecule)
            molecule_entity = atom_array.molecule_entity[mask][0]

            # Check whether the molecule_entity has been seen before
            if molecule_entity in molecule_entity_counts:
                molecule_entity_counts[molecule_entity] += 1
            else:
                molecule_entity_counts[molecule_entity] = 0

            # Assign a 0-indexed symmetric_id to the molecule
            symmetric_id = molecule_entity_counts[molecule_entity]
            atom_array.molecule_symmetric_id[mask] = symmetric_id

        data["atom_array"] = atom_array
        return data


class RenumberNonPolymerResidueIdx(Transform):
    """Re-numbers non-polymer residue indices to be one-indexed, similar to polymer residues.

    This transformation ensures that non-polymer residue indices start from 1, providing a consistent
    indexing scheme across both polymer and non-polymer residues. It addresses the issue where non-polymer
    residue indices may start at "101", which can lead to non-deterministic behavior.

    Note:
        The renumbering is applied to each non-polymer chain independently, ensuring that the indices
        are continuous and start from 1 for each chain.

    Returns:
        - data (dict): The updated data dictionary containing the modified atom_array with renumbered
            non-polymer residue indices.
    """

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["chain_iid", "res_id", "is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Get the non-polymer chain full IDs
        non_polymer_mask = ~atom_array.is_polymer
        non_polymer_chain_iids = np.unique(atom_array.chain_iid[non_polymer_mask])

        # Loop through every non-polymer chain, renumbering the residues
        for chain_iid in non_polymer_chain_iids:
            chain_mask = atom_array.chain_iid == chain_iid
            num_residues = struc.get_residue_count(atom_array[chain_mask])
            renumbered_res_ids = np.arange(1, num_residues + 1)  # 1-indexed
            atom_array.res_id[chain_mask] = struc.spread_residue_wise(atom_array[chain_mask], renumbered_res_ids)

        data["atom_array"] = atom_array
        return data


def get_within_poly_res_idx(atom_array: AtomArray) -> np.ndarray:
    """Get the within-polymer residue index for the atom array.

    For polymers, this is identical to within_chain_res_idx (since polymers have
    one chain per polymer). For non-polymers, the value is -1.

    Note:
        If `within_chain_res_idx` annotation exists, it will be reused for efficiency.
        Otherwise, it will be computed on-the-fly (same logic as AddWithinChainInstanceResIdx).

    Args:
        atom_array: The atom array to process. Must have `is_polymer` annotation.

    Returns:
        Array of within-polymer residue indices (0-indexed for polymers, -1 for non-polymers).
    """
    # Check if within_chain_res_idx already exists (performance optimization)
    if "within_chain_res_idx" in atom_array.get_annotation_categories():
        within_poly_res_idx = atom_array.within_chain_res_idx.copy()
    else:
        # Compute on-the-fly using same logic as AddWithinChainInstanceResIdx
        within_poly_res_idx = get_within_group_res_idx(atom_array, group_by="chain_iid")

    # Set non-polymers to -1
    within_poly_res_idx[~atom_array.is_polymer] = -1

    return within_poly_res_idx


def get_within_group_res_idx(atom_array: AtomArray, group_by: str) -> np.ndarray:
    """Get the within-group residue index for the atom array.

    Of note:
        - Groups do not need to be contiguous.
        - Groups are defined by the unique values of the `group_by` annotation.

    Args:
        atom_array (AtomArray): The atom array to process.
        group_by (str): The annotation name to group residues by (e.g., "chain_iid").

    Returns:
        np.ndarray: An array of within-group residue indices for each atom in the atom array.
    """
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)
    n_residues = len(residue_starts)

    # Get the group annotation for each residue (sample at residue starts)
    group_annotation = atom_array.get_annotation(group_by)
    residue_groups = group_annotation[residue_starts]

    # Compute within-group residue indices for each residue
    within_group_res_idx_per_residue = np.empty(n_residues, dtype=np.int32)

    for group in np.unique(residue_groups):
        mask = residue_groups == group
        within_group_res_idx_per_residue[mask] = np.arange(np.sum(mask))

    # Spread residue-wise indices to all atoms
    within_group_res_idx = struc.spread_residue_wise(atom_array, within_group_res_idx_per_residue)

    return within_group_res_idx


def get_within_group_atom_idx(atom_array: AtomArray, group_by: str) -> np.ndarray:
    """Get the within-group atom index for the atom array.

    Of note:
        - Groups do not need to be contiguous.
        - Groups are defined by the unique values of the `group_by` annotation.
    """
    within_group_atom_idx = np.empty(len(atom_array), dtype=np.int32)

    group_annotation = atom_array.get_annotation(group_by)

    for group_id in np.unique(group_annotation):
        group_mask = group_annotation == group_id
        in_group_atom_idx = np.arange(0, np.sum(group_mask))
        within_group_atom_idx[group_mask] = in_group_atom_idx

    return within_group_atom_idx


def get_within_entity_idx(
    atom_array: AtomArray, level: Literal["chain", "pn_unit", "molecule"]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Get the within-entity instance index for the atom array.
        - Allowed levels are "chain", "pn_unit", or "molecule".
        - Entities do not need to be contiguous.
        - Entities are defined by the unique values of the `{level}_entity` annotation.

    Args:
        - atom_array (AtomArray): The atom array to process.
        - level (Literal["chain", "pn_unit", "molecule"]): The level at which to calculate the within-entity index.

    Returns:
        - np.ndarray: An array of within-entity instance indices for each atom in the atom array.

    Example:
        >>> import biotite.structure as struc
        >>> atom_array = struc.AtomArray(7)
        >>> atom_array.set_annotation("chain_iid", ["A", "A", "B", "C", "D", "D", "E"])
        >>> atom_array.set_annotation("chain_entity", ["1", "1", "1", "1", "2", "2", "2"])
        >>> iids, within_entity_idx = get_within_entity_idx(atom_array, level="chain")
        >>> print(within_entity_idx)
        [0 0 1 2 0 0 1]
        >>> print(iids)
        ['A' 'B' 'C'] ['D' 'E']
    """
    within_entity_idx = np.empty(len(atom_array), dtype=np.int32)

    entity_annotation = atom_array.get_annotation(f"{level}_entity")
    instance_annotation = atom_array.get_annotation(f"{level}_iid")

    iids = []
    for entity_id in np.unique(entity_annotation):
        entity_mask = entity_annotation == entity_id

        in_entity_iids, in_entity_instance_idx = np.unique(instance_annotation[entity_mask], return_inverse=True)
        iids.append(in_entity_iids)
        within_entity_idx[entity_mask] = in_entity_instance_idx

    return iids, within_entity_idx


class AddWithinPolyResIdxAnnotation(Transform):
    """Adds the `within_poly_res_idx` (within polymer residue index) annotation.

    For polymers, the `within_poly_res_idx` is a zero-indexed, continuous residue
    index within the chain. For non-polymers, the value is set to -1.

    Note:
        The `within_poly_res_idx` is zero-indexed, since it is used as an index
        into the MSA. In contrast, the `res_id` annotation (derived from the mmCIF
        file) is one-indexed.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
    ]  # cropping changes the residue indices

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["chain_iid", "is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        within_poly_res_idx = get_within_poly_res_idx(atom_array)
        atom_array.set_annotation("within_poly_res_idx", within_poly_res_idx)

        data["atom_array"] = atom_array
        return data


def copy_annotation(atom_array: AtomArray, annotation_to_copy: str, new_annotation: str) -> AtomArray:
    """Copies an existing annotation from the AtomArray and assigns it a new name.

    Particularly useful for scenarios such as diffusive training, where the new annotation is altered (e.g., adding noise)
    without affecting the ground truth data.

    Args:
        atom_array (AtomArray): The AtomArray object containing the annotations.
        annotation_to_copy (str): The name of the annotation to be copied.
        new_annotation (str): The name for the new annotation.

    Returns:
        AtomArray: The AtomArray with the newly added annotation.

    Example:
        updated_atom_array = copy_annotation(atom_array, "coord", "coord_to_be_noised")
    """

    assert (
        new_annotation not in atom_array.get_annotation_categories() and new_annotation != "coord"
    ), f"Annotation {new_annotation} already exists in the AtomArray."

    if annotation_to_copy == "coord":
        # We must handle the special case of copying the coordinates (since "coord" is not technically an annotation)
        atom_array.set_annotation(new_annotation, atom_array.coord.copy())
    else:
        atom_array.set_annotation(new_annotation, copy.deepcopy(atom_array.get_annotation(annotation_to_copy)))

    return atom_array


class CopyAnnotation(Transform):
    """Copies an existing annotation from the AtomArray and assigns it a new name."""

    def __init__(self, annotation_to_copy: str, new_annotation: str):
        self.annotation_to_copy = annotation_to_copy
        self.new_annotation = new_annotation

    def check_input(self, data: dict) -> None:
        assert has_annotation(
            data["atom_array"], self.annotation_to_copy
        ), f"Annotation {self.annotation_to_copy} does not exist in the AtomArray."
        assert not has_annotation(
            data["atom_array"], self.new_annotation
        ), f"Annotation {self.new_annotation} already exists in the AtomArray."

    def forward(self, data: dict) -> dict:
        data["atom_array"] = copy_annotation(
            data["atom_array"], annotation_to_copy=self.annotation_to_copy, new_annotation=self.new_annotation
        )
        return data


class ApplyFunctionToAtomArray(Transform):
    """Apply a function to the atom array."""

    def __init__(self, func: Callable[[AtomArray], AtomArray]):
        self.func = func

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        data["atom_array"] = self.func(data["atom_array"])
        return data


def add_protein_termini_annotation(atom_array: AtomArray) -> AtomArray:
    """Adds the annotation is_N_terminus and is_C_terminus to the respective residues in the atom array.

    Args:
        atom_array (AtomArray): The AtomArray that the annotations will be added to

    Returns:
        AtomArray: The AtomArray with is_N_terminus and is_C_terminus annotations
    """

    is_linear_protein = np.isin(
        atom_array.chain_type, [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L]
    )  # We can't use PROTEINS from data_constants.py, since that includes CYCLIC_PSEUDO_PEPTIDE

    # Annotate N-termini
    is_first_in_chain = atom_array.res_id == 1
    atom_array.set_annotation("is_N_terminus", is_first_in_chain & is_linear_protein)

    # Annotate C-termini
    last_res_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)[1:] - 1
    is_last_in_chain = np.zeros(len(atom_array), dtype=bool)
    is_last_in_chain[last_res_idxs] = True
    is_last_in_chain = apply_and_spread_residue_wise(atom_array, is_last_in_chain, function=np.any)
    atom_array.set_annotation("is_C_terminus", is_last_in_chain & is_linear_protein)

    return atom_array


class AddProteinTerminiAnnotation(Transform):
    """Annotate protein termini (i.e. N- and C-terminus) for protein chains in the atom array."""

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["CropContiguousLikeAF3", "CropSpatialLikeAF3"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_id", "chain_id", "chain_type"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        data["atom_array"] = add_protein_termini_annotation(atom_array)
        return data


def add_global_atom_id_annotation(atom_array: AtomArray) -> AtomArray:
    """Adds a global atom ID annotation `atom_id` to the atom array.

    This annotation is useful for tracking atoms after operations such as cropping,
    slicing, or shuffling. The `atom_id` is generated as a sequence of integers
    corresponding to the number of atoms in the atom array.

    Args:
        atom_array (AtomArray): The AtomArray to which the atom ID annotation will be added.

    Returns:
        AtomArray: The AtomArray with the added `atom_id` annotation.
    """
    atom_array.set_annotation("atom_id", np.arange(len(atom_array), dtype=np.uint32))
    return atom_array


class AddGlobalAtomIdAnnotation(Transform):
    """Adds a global atom ID annotation to the atom array.

    Useful for keeping track of atoms after cropping, slicing or shuffling operations.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AddGlobalAtomIdAnnotation"]

    def __init__(self, allow_overwrite: bool = False):
        """
        Args:
            allow_overwrite (bool): Whether to allow overwriting an existing `atom_id` annotation.
        """
        self.allow_overwrite = allow_overwrite

    def check_input(self, data: dict) -> None:
        if "atom_id" in data["atom_array"].get_annotation_categories() and not self.allow_overwrite:
            raise ValueError("AtomArray already contains 'atom_id' annotation! It would be overwritten.")

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        data["atom_array"] = add_global_atom_id_annotation(atom_array)
        return data


def add_global_token_id_annotation(atom_array: AtomArray) -> AtomArray:
    """Adds a global token ID annotation `token_id` to the atom array.

    This annotation is useful for tracking tokens after operations such as cropping,
    slicing, or shuffling. The `token_id` is generated as a sequence of integers
    corresponding to the number of tokens in the atom array, and is spread across
    the atom array to maintain the association with each atom.

    Args:
        atom_array (AtomArray): The AtomArray to which the token ID annotation will be added.

    Returns:
        AtomArray: The AtomArray with the added `token_id` annotation.
    """
    token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)  # [n_tokens]
    atom_array.set_annotation("token_id", spread_token_wise(atom_array, token_id))
    return atom_array


class AddGlobalTokenIdAnnotation(Transform):
    """Adds a global token ID annotation `token_id` to the atom array.

    Useful for keeping track of tokens after cropping, slicing or shuffling operations.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AddGlobalTokenIdAnnotation"]

    def __init__(self, allow_overwrite: bool = False):
        self.allow_overwrite = allow_overwrite

    def check_input(self, data: dict) -> None:
        if "token_id" in data["atom_array"].get_annotation_categories() and not self.allow_overwrite:
            raise ValueError("AtomArray already contains 'token_id' annotation! It would be overwritten.")

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # add the global token id annotation
        data["atom_array"] = add_global_token_id_annotation(atom_array)
        return data


def add_global_res_id_annotation(atom_array: AtomArray) -> AtomArray:
    """Add a global residue ID annotation to the atom array."""
    res_id = np.arange(get_residue_count(atom_array), dtype=np.uint32)  # [n_residues]
    # Note that "res_id" already exists (as it is a standard field in CIF files), so we add the global version with a "_global" suffix
    # TODO: We should rename token_id, atom_id to token_id_global, atom_id_global so that we follow a consistent naming convention
    atom_array.set_annotation("res_id_global", spread_residue_wise(atom_array, res_id))
    return atom_array


class AddGlobalResIdAnnotation(Transform):
    """Adds a global residue ID annotation to the atom array."""

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AddGlobalResIdAnnotation"]

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["res_id"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = add_global_res_id_annotation(data["atom_array"])
        return data


class AddWithinChainInstanceResIdx(Transform):
    """Add the within-chain instance residue index to the atom array (0-indexed)."""

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["chain_iid", "res_id", "res_name"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # ... get within-chain residue index
        within_chain_res_idx = get_within_group_res_idx(atom_array, group_by="chain_iid")
        atom_array.set_annotation("within_chain_res_idx", within_chain_res_idx)

        data["atom_array"] = atom_array
        return data


def sort_poly_then_non_poly(atom_array: AtomArray, treat_atomized_as_non_poly: bool = True) -> AtomArray:
    """Sort the atom array such that polymer chains are first, followed by non-polymer chains.

    The order within the `poly` and `non_poly` chains is preserved.

    This function is useful for ensuring that models like `RF2AA`, which expect the input to be
    formatted as [polys, non-polys], receive the correctly ordered atom array.

    Args:
        - atom_array (AtomArray): The AtomArray to be sorted.
        - treat_atomized_as_non_poly (bool): If True, atomized structures are treated as non-polymer.
            Defaults to True.

    Returns:
        AtomArray: The sorted AtomArray with polymer chains first, followed by non-polymer chains.
    """
    is_atomized = np.zeros(len(atom_array), dtype=bool)
    if treat_atomized_as_non_poly and "atomize" in atom_array.get_annotation_categories():
        is_atomized = atom_array.atomize

    # Find indices of polymer and non-polymer atoms
    is_poly = atom_array.is_polymer & ~is_atomized
    is_non_poly = is_atomized | ~atom_array.is_polymer

    # Sort by indexing (instead of masking/slicing), since this leads to correctly
    # tracking and updating the inter-poly-non-poly bonds
    poly_idxs = np.where(is_poly)[0]
    non_poly_idxs = np.where(is_non_poly)[0]
    sort_poly_then_non_poly = np.concatenate([poly_idxs, non_poly_idxs])

    return atom_array[sort_poly_then_non_poly]


def sort_like_rf2aa(atom_array: AtomArray) -> AtomArray:
    """Sort the atom array such that non-polymer chains are sorted by their covalent bonds and PN unit IIDs."""
    is_atomized = np.zeros(len(atom_array), dtype=bool)
    if "atomize" in atom_array.get_annotation_categories():
        is_atomized = atom_array.atomize

    # Find indices of polymer and non-polymer atoms
    is_poly = atom_array.is_polymer & (~is_atomized)
    is_non_poly = is_atomized | (~atom_array.is_polymer)
    is_bonded_non_poly = np.zeros(len(atom_array), dtype=bool)
    for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
        pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
        is_bonded_non_poly[pn_unit_mask] = np.any(is_poly[pn_unit_mask]) & is_non_poly[pn_unit_mask]
    is_free_non_poly = is_non_poly & (~is_bonded_non_poly)
    assert np.sum(is_poly) + np.sum(is_bonded_non_poly) + np.sum(is_free_non_poly) == len(
        atom_array
    ), "overlapping groups"

    # Sort by indexing according to
    #  0: by poly / bonded non-poly / free non-poly
    #  1: within groups by moelcule_iid
    #  2: within molecules by pn_unit_iid
    #  3: within pn_units by chain_iid
    _sort_table = pd.DataFrame(
        {
            "atom_idx": np.arange(len(atom_array)),
            "group": is_poly.astype(np.int8)
            + 2 * is_bonded_non_poly.astype(np.int8)
            + 3 * is_free_non_poly.astype(np.int8),
            "molecule_entity": atom_array.molecule_entity,
            "molecule_iid": atom_array.molecule_iid,
            "pn_unit_iid": atom_array.pn_unit_iid,
            "chain_entity": atom_array.chain_entity,
            "chain_iid": atom_array.chain_iid,
        }
    )
    to_sorted = _sort_table.sort_values(
        by=["group", "molecule_entity", "molecule_iid", "pn_unit_iid", "chain_entity", "chain_iid", "atom_idx"]
    )["atom_idx"].values

    # ... ensure all indices occur exactly once
    assert np.all(np.sort(to_sorted) == np.arange(len(atom_array))), "indices must occur exactly once"

    return atom_array[to_sorted]


class SortLikeRF2AA(Transform):
    """Sort the atom array in 3 groups (in this order). Within each group the atoms are ordered by
    their pn_unit_iid (and within a pn_unit their order is preserved).

    - (1) polymer atoms
    - (2) non-poly atoms of a pn-unit bonded to a polymer (covalent modifications)
    - (3) non-poly atoms of a free-floating pn-unit (free-floating ligands)
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropSpatialLikeAF3",
        "CropContiguousLikeAF3",
    ]

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(
            data,
            ["is_polymer", "pn_unit_iid", "molecule_iid", "molecule_entity", "chain_entity", "chain_iid", "atomize"],
        )

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # perform the sorting
        data["atom_array"] = sort_like_rf2aa(atom_array)

        return data


class SortPolyThenNonPoly(Transform):
    """Sort the atom array such that polymer chains are first, followed by non-polymer chains.

    The order within the `poly` and `non_poly` chains is preserved.

    This transformation is useful for models like `RF2AA`, which expect the input to be formatted
    as [polys, non-polys].

    Args:
        - treat_atomized_as_non_poly (bool): If True, atomized structures are treated as non-polymer.
            Defaults to True.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropSpatialLikeAF3",
        "CropContiguousLikeAF3",
    ]

    def __init__(self, treat_atomized_as_non_poly: bool = True):
        self.treat_atomized_as_non_poly = treat_atomized_as_non_poly

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # perform the sorting
        data["atom_array"] = sort_poly_then_non_poly(atom_array, self.treat_atomized_as_non_poly)

        return data

        # TODO: Write tests; find an example to check
        # TODO: Trial-and-error a couple approaches to this challenge (e.g., best way to avoid liposomes)


class RaiseIfTooManyAtoms(Transform):
    def __init__(self, max_atoms: int):
        self.max_atoms = max_atoms

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "example_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        num_atoms = len(data["atom_array"])
        if num_atoms > self.max_atoms:
            example_id = data["example_id"]
            raise ValueError(f"{example_id} exceeds max allowed number of atoms! ({num_atoms:,} > {self.max_atoms:,}).")
        return data


def compute_atom_to_token_map(atom_array: AtomArray) -> dict:
    # ...assert that the token_id array is continuous (e.g., we applied AddGlobalTokenIDAnnotation post-crop)
    assert np.all((np.diff(atom_array.token_id) == 0) | (np.diff(atom_array.token_id) == 1))

    # ...assert that the token_id array is zero-indexed
    assert atom_array.token_id[0] == 0

    return atom_array.token_id.astype(np.int32)


class ComputeAtomToTokenMap(Transform):
    """Add length `[n_atom]` array to the `feats` dictionary that indicates the `token_id` for each atom."""

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AddGlobalTokenIdAnnotation"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["token_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_to_token_map = compute_atom_to_token_map(data["atom_array"])
        nested_dict.set(data, ("feats", "atom_to_token_map"), atom_to_token_map)
        return data
