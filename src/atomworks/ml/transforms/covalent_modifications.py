"""Transforms to handle covalent modifications"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from biotite.structure import AtomArray

from atomworks.ml.preprocessing.utils.structure_utils import get_inter_pn_unit_bond_mask
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform


def flag_and_reassign_covalent_modifications(atom_array: AtomArray) -> AtomArray:
    """
    Mark covalent modifications for atomization and reassign the corresponding
    PN unit annotations.

    Args:
        atom_array (AtomArray): Current `AtomArray` within the Transform pipeline

    Returns:
        AtomArray: The modified `AtomArray` with updated annotations for covalent
            modifications. The `pn_unit_id` and `pn_unit_iid` of polymer atoms are
            reassigned to those of the non-polymer unit they are bound to, and the
            `atomize` annotation is set to `True` for these atoms. Additionally, the
            entire pn_unit is marked with `is_covalent_modification = True`.

    NOTE: If `atomize` annotation is not present in the `AtomArray`, it will be added.
    NOTE: If `is_covalent_modification` annotation is not present in the `AtomArray`, it will be added.
    NOTE: We do not modify the `is_polymer` annotation, which will still refer to the protein chain
    for the atomized polymer atoms.
    """
    # Get all inter-PN unit bonds in the entry (i.e. between a polymer and a non-polymer PN unit)
    inter_pn_unit_bond_mask = get_inter_pn_unit_bond_mask(atom_array)
    bonds_to_check = atom_array.bonds.as_array()[inter_pn_unit_bond_mask]

    # Filter out bonds that are not between a polymer and a non-polymer PN unit
    bonds_to_check = bonds_to_check[
        # One atom is a polymer, the other is not => must be polymer/non-polymer bond
        atom_array.is_polymer[bonds_to_check[:, 0]] != atom_array.is_polymer[bonds_to_check[:, 1]]
    ]

    # Add the atomize annotation to the AtomArray, if not already present
    if "atomize" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("atomize", np.array([False] * len(atom_array)))

    # Add the is_covalent_modification annotation to the AtomArray, if not already present
    if "is_covalent_modification" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("is_covalent_modification", np.array([False] * len(atom_array)))

    # Loop through inter-molecular bonds
    # NOTE: There aren't likely to be many inter-molecular bonds in the entry, so vectorization is not necessary and would be less readable
    for bond in bonds_to_check:
        # Get the atoms involved in the inter-molecular bonds
        atom_a = atom_array[bond[0]]
        atom_b = atom_array[bond[1]]

        # Note which atom is in the polymer and which is in the non-polymer
        polymer_atom, non_polymer_atom = (atom_a, atom_b) if atom_a.is_polymer else (atom_b, atom_a)

        # Create a mask of the atoms in the residue that is covalently bound to the non-polymer PN unit
        # We can uniquely identify a residue by its res_id, pn_unit_iid, and chain_id (or chain_iid, either works)
        polymer_residue_mask = (
            (atom_array.res_id == polymer_atom.res_id)
            & (atom_array.chain_id == polymer_atom.chain_id)
            & (atom_array.pn_unit_iid == polymer_atom.pn_unit_iid)
        )

        # For all atoms in the target polymer residue, set the pn_unit_iid and the pn_unit_id to that of the non-polymer PN unit
        num_residues = np.sum(polymer_residue_mask)
        atom_array.pn_unit_id[polymer_residue_mask] = np.array([non_polymer_atom.pn_unit_id] * num_residues)
        atom_array.pn_unit_iid[polymer_residue_mask] = np.array([non_polymer_atom.pn_unit_iid] * num_residues)

        # Mark the non-polymer residue for atomization (now includes all atoms in the bonded polymer residue)
        atom_array.atomize[(atom_array.pn_unit_iid == non_polymer_atom.pn_unit_iid)] = True

        # Mark the entire pn_unit as a covalent modification
        atom_array.is_covalent_modification[(atom_array.pn_unit_iid == non_polymer_atom.pn_unit_iid)] = True

    return atom_array


class FlagAndReassignCovalentModifications(Transform):
    """Handles covalent modifications within the AtomArray.

    Covalent modifications, e.g., glycosylation, are handled by the following algorithm:

    for polymer residues with atoms covalently bound to a NON-POLYMER:
        for ALL atoms in the polymer residue:
            set the pn_unit_iid and pn_unit_id identifying annotations to that of the NON-POLYMER polymer/non-polymer unit
            set atomize = true (thus, this transform must be run before the Atomize transform)
            set is_covalent_modification = true (for the entire pn_unit)

    TODO: Break into two Transforms - one that flags, one that reassigns. Atomizing covalent modifications is a design choice
    that may not be desired in all pipelines. Annotating covalent modifications, however, is broadly useful.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, "AddGlobalTokenIdAnnotation"]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = flag_and_reassign_covalent_modifications(data["atom_array"])
        return data
