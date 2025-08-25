"""
Transforms to handle featurization of edge cases with unresolved residues.

NOTE: Transforms that "filter" based on unresolved residues will be found in the "filters" file, not here.
"""

from typing import Any, ClassVar

import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import NUCLEIC_ACID_FRAME_ATOM_NAMES, PROTEIN_FRAME_ATOM_NAMES
from atomworks.enums import ChainTypeInfo
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.numpy import get_nearest_true_index_for_each_false
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_af3_token_center_masks,
    get_af3_token_representative_masks,
    spread_token_wise,
)


def mask_residues_with_specific_unresolved_atoms(
    atom_array: AtomArray,
    chain_type_to_atom_names: dict[tuple | list | Any, list[str]] | None = None,
    occupancy_threshold: float = 0.0,
) -> AtomArray:
    """If a residue has any unresolved atoms from the specified list, set the occupancy of the entire residue to zero.

    Args:
        atom_array (AtomArray): The atom array to modify.
        chain_type_to_atom_names (dict[tuple | list | ChainType, list[str]], optional): A dictionary mapping
            chain types to lists of atom names that should be checked for resolution. Keys can be:
            - Single chain type (e.g., ChainType.POLYPEPTIDE_L)
            - Tuple/list of chain types (e.g., ChainTypeInfo.PROTEINS)
            If None, uses the default AF-3 frame atoms. Defaults to None.
        occupancy_threshold (float): Atoms with occupancy <= this value are considered unresolved.
            Defaults to 0.0.

    Returns:
        AtomArray: The modified atom array.
    """

    # Use default AF-3 frame atoms if not specified
    if chain_type_to_atom_names is None:
        chain_type_to_atom_names = {
            ChainTypeInfo.PROTEINS: PROTEIN_FRAME_ATOM_NAMES,
            ChainTypeInfo.NUCLEIC_ACIDS: NUCLEIC_ACID_FRAME_ATOM_NAMES,
        }

    unresolved_backbone_mask = np.zeros(len(atom_array), dtype=bool)

    # ... subset to backbone atoms within polymers with unresolved coordinates
    # (We treat partially occupied atoms as occupied; e.g., those resolved from "altlocs")
    if "chain_type" in atom_array.get_annotation_categories():
        # Process each chain type group and its required atoms
        for chain_types, atom_names in chain_type_to_atom_names.items():
            # Handle both single chain types and tuples/lists of chain types
            if not isinstance(chain_types, tuple | list):
                chain_types = [chain_types]

            chain_type_mask = np.isin(atom_array.chain_type, chain_types)
            unresolved_backbone_mask |= (
                chain_type_mask
                & np.isin(atom_array.atom_name, atom_names)
                & (atom_array.occupancy <= occupancy_threshold)
            )
    else:
        # ... rely on atom names if chain type is not available
        all_atom_names = []
        for atom_names in chain_type_to_atom_names.values():
            all_atom_names.extend(atom_names)

        unresolved_backbone_mask = np.isin(atom_array.atom_name, all_atom_names) & (
            atom_array.occupancy <= occupancy_threshold
        )

    # Residue-wise mask for unresolved backbone atoms
    unresolved_backbone_res_mask = apply_and_spread_residue_wise(atom_array, unresolved_backbone_mask, function=np.any)

    # ... mask the occupancy of the entire residue
    atom_array.occupancy[unresolved_backbone_res_mask] = 0

    return atom_array


def mask_polymer_residues_with_unresolved_frame_atoms(
    atom_array: AtomArray, occupancy_threshold: float = 0.0
) -> AtomArray:
    """If a polymer residue has an unresolved backbone atom (occupancy <= occupancy_threshold), set the occupancy of the entire residue to zero.

    This is a backwards-compatible wrapper around mask_residues_with_specific_unresolved_atoms.
    """
    return mask_residues_with_specific_unresolved_atoms(atom_array, occupancy_threshold=occupancy_threshold)


class MaskResiduesWithSpecificUnresolvedAtoms(Transform):
    """For residues with at least one unresolved atom from the specified list, mask (set to occupancy zero) the entire residue.

    Helpful for e.g., when we are missing backbone frame atoms, since if we don't have frame atoms, then:
        - We cannot build residue frames
        - The local structure quality is likely poor

    We (and AF-3) consider the frame atoms to be:
        - Proteins: N, CA, C
        - Nucleic Acids: C1', C3', C4'

    As an example for proteins, see PDB ID `6Z3R`, which has unresolved C and CA atoms.
    As an example fo nucleic acids, see 7Z24, which has unresolved C1', C2', and C3' (but does have a resolved oxygen)

    NOTE: This transform must be applied before other transform that rely on the `occupancy` annotation.

    This transform allows specification of which atoms to check for each chain type; for MPNN, we consider the backbone oxygen (O) as well.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def __init__(
        self,
        chain_type_to_atom_names: dict[tuple | list | Any, list[str]] | None = None,
        occupancy_threshold: float = 0.0,
    ):
        self.chain_type_to_atom_names = chain_type_to_atom_names
        self.occupancy_threshold = occupancy_threshold

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy", "chain_type"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = mask_residues_with_specific_unresolved_atoms(
            data["atom_array"], self.chain_type_to_atom_names, self.occupancy_threshold
        )
        return data


class MaskPolymerResiduesWithUnresolvedFrameAtoms(Transform):
    """For residues with at least one unresolved frame atom, mask (set to occupancy zero) the entire residue.

    This is a backwards-compatible wrapper around MaskResiduesWithSpecificUnresolvedAtoms.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def __init__(self, occupancy_threshold: float = 0.0):
        self.occupancy_threshold = occupancy_threshold

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["occupancy", "chain_type"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = mask_polymer_residues_with_unresolved_frame_atoms(
            data["atom_array"], self.occupancy_threshold
        )
        return data


def place_unresolved_token_on_closest_resolved_token_in_sequence(
    atom_array: AtomArray, annotation_to_update: str = "coord_to_be_noised", annotation_to_copy: str = "coord"
) -> AtomArray:
    """Place all atoms within fully-unresolved residues on the closest resolved neighbor in sequence space.

    NOTE: For non-polymers, each atom is considered a token, so this transform will place unresolved
    atoms on the closest resolved token in sequence space (i.e., the previous or next atom).

    NOTE: We only perform the operation WITHIN chains, such that we don't resolve across chain boundaries.

    Args:
        atom_array (AtomArray): The atom array to modify.
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
        annotation_to_copy (str): The annotation to copy from the resolved atom to the unresolved atom. E.g., "coord" (if we want to copy the ground-truth),
            or "coord_to_be_noised" (if we want to copy the coordinates that will be noised, which may have been modified by previous transforms).
            In the AF-3 pipeline, we want to copy "coord_to_be_noised", to correctly resolve residues after applying PlaceUnresolvedTokenAtomsOnRepresentativeAtom.

    Returns:
        AtomArray: The modified atom array.
    """

    # ... loop through chains with unresolved atoms, such that we don't resolve across chain boundaries
    # (NOTE: We only iterate through chain instances containing any unresolved atoms for efficiency)
    for chain_iid in np.unique(atom_array.chain_iid[atom_array.occupancy == 0]):
        chain_mask = atom_array.chain_iid == chain_iid
        chain_atom_array = atom_array[chain_mask]

        # ... map each unresolved token to the nearest resolved token
        # (Here, we consider a token resolve if any atom within the token is resolved)
        is_token_resolved_token_level = apply_token_wise(
            chain_atom_array, chain_atom_array.occupancy, np.any
        )  # (n_tokens)
        is_token_resolved_atom_level = spread_token_wise(chain_atom_array, is_token_resolved_token_level)  # (n_atoms)

        # (Early exit if  all tokens are resolved)
        if np.all(is_token_resolved_atom_level):
            continue

        # (if no tokens are resolved, set nan's -> 0s)
        if np.all(~is_token_resolved_atom_level):
            if annotation_to_update == "coord":
                # (We must handle "coord" explicitly, as it is treated differently than other annotations)
                atom_array.coord[chain_mask] = np.nan_to_num(chain_atom_array.coord)
            else:
                atom_array.get_annotation(annotation_to_update)[chain_mask] = np.nan_to_num(
                    chain_atom_array.get_annotation(annotation_to_update)
                )
            continue

        # ... get the nearest resolved token indices for each unresolved token
        nearest_resolved_token_indices_token_wise = get_nearest_true_index_for_each_false(
            is_token_resolved_token_level
        )  # (n_tokens)
        nearest_resolved_token_indices_atom_wise = spread_token_wise(
            chain_atom_array[~is_token_resolved_atom_level], nearest_resolved_token_indices_token_wise
        )  # (n_atoms)

        # Where the entire token is unresolved, set the atom coordinates to the nearest resolved token representative atom coordinates

        # ... get coordinates of representative atoms from the specified annotation to copy
        if annotation_to_copy == "coord":
            source_coordinates = chain_atom_array.coord
        else:
            source_coordinates = chain_atom_array.get_annotation(annotation_to_copy)

        representative_atom_coordinates_atom_level = source_coordinates[
            get_af3_token_representative_masks(chain_atom_array)
        ]  # (n_atoms, 3)

        assert len(representative_atom_coordinates_atom_level) == len(is_token_resolved_token_level)

        # ... update the coordinates for the specified annotation (e.g., "coord" or "coord_to_be_noised")
        if annotation_to_update == "coord":
            # (We must handle "coord" explicitly, as it is treated differently than other annotations)
            chain_atom_array.coord[~is_token_resolved_atom_level] = representative_atom_coordinates_atom_level[
                nearest_resolved_token_indices_atom_wise
            ]
            atom_array.coord[chain_mask] = chain_atom_array.coord
        else:
            chain_atom_array.get_annotation(annotation_to_update)[~is_token_resolved_atom_level] = (
                representative_atom_coordinates_atom_level[nearest_resolved_token_indices_atom_wise]
            )
            atom_array.get_annotation(annotation_to_update)[chain_mask] = chain_atom_array.get_annotation(
                annotation_to_update
            )

    return atom_array


class PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(Transform):
    """Place fully unresolved tokens on their closest resolved neighbor in sequence space, breaking ties by choosing the "leftmost" neighbor.

    This heuristic is helpful to avoid noising unresolved residue coordinates from the origin during diffusion training.

    Args:
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
            NOTE: Must match the annotation used for `PlaceUnresolvedTokenAtomsOnRepresentativeAtom`.
        annotation_to_copy (str): The annotation to copy from the resolved atom to the unresolved atom.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AtomizeByCCDName",
        "PlaceUnresolvedTokenAtomsOnRepresentativeAtom",
    ]

    def __init__(
        self, annotation_to_update: str = "coord_to_be_noised", annotation_to_copy: str = "coord_to_be_noised"
    ) -> None:
        self.annotation_to_update = annotation_to_update
        self.annotation_to_copy = annotation_to_copy

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

        annotations_to_check = {"occupancy"}
        if self.annotation_to_update != "coord":
            # "coord" is a special annotation, and technically not in `atom_array.get_annotation_categories()`
            annotations_to_check.add(self.annotation_to_update)
        if self.annotation_to_copy != "coord":
            annotations_to_check.add(self.annotation_to_copy)

        check_atom_array_annotation(data, list(annotations_to_check))

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = place_unresolved_token_on_closest_resolved_token_in_sequence(
            data["atom_array"],
            annotation_to_update=self.annotation_to_update,
            annotation_to_copy=self.annotation_to_copy,
        )
        return data


def place_unresolved_token_atoms_on_token_representative_atom(
    atom_array: AtomArray, annotation_to_update: str = "coord_to_be_noised"
) -> AtomArray:
    """Place unresolved token atoms (e.g., side chain atoms) on the representative atom of the corresponding residue (token).

    In cases where the representative atom is unresolved, we also try the token center atom (e.g., if the CB is unresolved but the CA is resolved, like in 8E83 for chain A, residue 194).
    Helpful for diffusive models to avoid noising unresolved side-chain atoms from the origin.

    NOTE: For non-polymers, all atoms are considered tokens (and are atomized); in such cases this Transform will have no effect.

    Args:
        atom_array (AtomArray): The atom array to modify.
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).

    Returns:
        AtomArray: The modified atom array.
    """
    # ... get a mask of all unresolved atoms
    unresolved_atom_mask = atom_array.occupancy == 0

    # ... get the unique chain IIDs of polymers with unresolved atoms (as this transform only applies to polymers; e.g., chains without full atomization)
    chain_iids_with_unresolved_atoms = np.unique(atom_array.chain_iid[(unresolved_atom_mask) & (~atom_array.atomize)])

    # ... prepare a mask of representative atoms for each residue
    representative_atom_mask = get_af3_token_representative_masks(atom_array)

    # (For cases where the representative atom is unresolved, we also try the token center atom)
    center_atom_mask = get_af3_token_center_masks(atom_array)

    for chain_iid in chain_iids_with_unresolved_atoms:
        # NOTE: We cannot rely on the `is_polymer` annotation, as in some instances (like acyl groups) we may have non-polymer tokens within a polymer chain (see: 7RCU)
        residues_with_unresolved_atoms = np.unique(
            atom_array.res_id[(atom_array.chain_iid == chain_iid) & unresolved_atom_mask & (~atom_array.atomize)]
        )
        for res_id in residues_with_unresolved_atoms:
            # ... create a mask for the unresolved atoms in the residue
            residue_mask = (atom_array.chain_iid == chain_iid) & (atom_array.res_id == res_id)
            unresolved_atoms_in_residue_mask = residue_mask & unresolved_atom_mask

            # ... get a mask indicating where to place unresolved atoms
            placement_atom_mask = representative_atom_mask & residue_mask

            # ... if the chosen atom (by default, the representative atom) is unresolved, try the center atom
            if not np.any(placement_atom_mask & ~unresolved_atom_mask):
                placement_atom_mask = center_atom_mask & residue_mask

            # ... get the index of the representative atom (there should be exactly one instance of the chain)
            assert np.sum(placement_atom_mask) == 1
            placement_atom_idx = np.where(placement_atom_mask)[0]

            # ... set the unresolved atom coordinates to the placment atom coordinates
            if annotation_to_update == "coord":
                # (We must handle "coord" explicitly, as it is treated differently than other annotations)
                atom_array.coord[unresolved_atoms_in_residue_mask] = atom_array.coord[placement_atom_idx]
            else:
                atom_array.get_annotation(annotation_to_update)[unresolved_atoms_in_residue_mask] = (
                    atom_array.get_annotation(annotation_to_update)[placement_atom_idx]
                )

    return atom_array


class PlaceUnresolvedTokenAtomsOnRepresentativeAtom(Transform):
    """Place unresolved token atoms (e.g., side chain atoms) on the representative atom of the residue (token).

    Note that this Transform has no impact on non-polymers, as all atoms are considered tokens.

    Args:
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
        or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def __init__(self, annotation_to_update: str = "coord_to_be_noised") -> None:
        self.annotation_to_update = annotation_to_update

    def check_input(self, data: dict[str, Any]) -> None:
        annotations_to_check = ["occupancy"]
        if self.annotation_to_update != "coord":
            # "coord" is a special annotation, and technically not in `atom_array.get_annotation_categories()`
            annotations_to_check += [self.annotation_to_update]
        check_atom_array_annotation(data, annotations_to_check)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = place_unresolved_token_atoms_on_token_representative_atom(
            data["atom_array"], annotation_to_update=self.annotation_to_update
        )
        return data
