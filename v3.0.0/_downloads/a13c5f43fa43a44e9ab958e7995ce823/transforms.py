"""Pocket cropping and featurization for the model-building tutorial."""

from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys
from atomworks.ml.transforms.base import Transform


class CropToPocket(Transform):
    """Keep the ligand and nearby protein atoms."""

    requires_previous_transforms: ClassVar[list[str]] = ["RemoveHydrogens", "RemoveUnresolvedAtoms"]

    def __init__(self, radius: float = 10.0) -> None:
        super().__init__()
        self.radius = radius

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array", "query_pn_unit_iids", "query_is_polymer"])
        check_atom_array_annotation(data, ["pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = crop_to_pocket(
            data["atom_array"],
            query_pn_unit_iids=data["query_pn_unit_iids"],
            query_is_polymer=data["query_is_polymer"],
            radius=self.radius,
        )
        return data


class FeaturizeForDocking(Transform):
    """Add atom, bond, and coordinate arrays for the docking model."""

    requires_previous_transforms: ClassVar[list[str]] = ["CropToPocket"]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_atom_array_annotation(data, ["is_ligand"])

    def forward(self, data: dict) -> dict:
        data.update(featurize_for_docking(data["atom_array"]))
        return data


def crop_to_pocket(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    query_is_polymer: list[bool],
    radius: float = 10.0,
) -> AtomArray:
    """Keep the complete ligand and protein atoms within ``radius`` angstroms of it."""
    if radius <= 0:
        raise ValueError(f"Pocket radius must be positive, got {radius}")
    if len(query_pn_unit_iids) != 2 or len(query_is_polymer) != 2 or sum(query_is_polymer) != 1:
        raise ValueError("Expected two query PN units with exactly one polymer")

    protein_side = 0 if query_is_polymer[0] else 1
    ligand_side = 1 - protein_side
    ligand_iid = query_pn_unit_iids[ligand_side]
    protein_iid = query_pn_unit_iids[protein_side]

    ligand_mask = atom_array.pn_unit_iid == ligand_iid
    protein_mask = atom_array.pn_unit_iid == protein_iid
    ligand_coords = atom_array.coord[ligand_mask]
    protein_coords = atom_array.coord[protein_mask]

    if len(ligand_coords) == 0:
        raise ValueError(f"Ligand {ligand_iid} has no atoms")
    if len(protein_coords) == 0:
        raise ValueError(f"Protein {protein_iid} has no atoms")

    neighbors = cKDTree(protein_coords).query_ball_point(ligand_coords, r=radius)
    if not any(neighbors):
        raise ValueError(f"No protein atoms found within {radius} Å of ligand {ligand_iid}")

    pocket_local_indices = np.unique(np.concatenate(neighbors).astype(int))
    protein_global_indices = np.flatnonzero(protein_mask)
    ligand_global_indices = np.flatnonzero(ligand_mask)
    keep = np.sort(np.concatenate([protein_global_indices[pocket_local_indices], ligand_global_indices]))

    cropped = atom_array[keep]
    cropped.set_annotation("is_ligand", ligand_mask[keep])
    return cropped


def featurize_for_docking(atom_array: AtomArray) -> dict[str, np.ndarray]:
    """Build atom, bond, and coordinate arrays from a cropped pocket."""
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)
    input_coords = target_coords.copy()
    input_coords[is_ligand] = 0.0

    atomic_numbers = np.array(
        [ELEMENT_NAME_TO_ATOMIC_NUMBER.get(element.upper(), 0) for element in atom_array.element],
        dtype=np.int64,
    )
    if atom_array.bonds is None:
        raise ValueError("Cropped atom array has no bond list")
    edge_index = atom_array.bonds.as_array()[:, :2].T.astype(np.int64)

    return {
        "atomic_numbers": atomic_numbers,
        "is_ligand": is_ligand,
        "target_coords": target_coords,
        "edge_index": edge_index,
        "input_coords": input_coords,
    }
