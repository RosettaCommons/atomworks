"""Utilities to identify and annotate non-token (e.g., small molecules, non-canonical amino acids) atoms within an AtomArray for downstream use."""

from __future__ import annotations

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.io.transforms.atom_array import add_pn_unit_iid_annotation
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.bonds import get_inter_pn_unit_bond_mask
from atomworks.io.utils.ccd import get_polymerization_atoms


def compute_standard_atomize_mask(atom_array: AtomArray) -> np.ndarray:
    """Return a boolean mask indicating which atoms should be atomized.

    Replicates the io-level equivalent of the ML pipeline sequence:
    ``FlagNonPolymersForAtomization`` → ``AnnotateCovalentModifications``
    → ``AtomizeByCCDName``.

    Works on a copy of *atom_array* so the input is never mutated.

    Args:
        atom_array: Input structure.  Must have bonds assigned.  ``is_polymer``
            is inferred via the annotator if absent.  ``pn_unit_iid`` is used
            to detect covalently-modified polymer residues when present.

    Returns:
        Boolean array of length ``atom_array.array_length()``.  ``True`` for atoms that
        should be treated atomistically (non-polymers and polymer residues
        covalently bound to a non-polymer unit).
    """
    arr = atom_array.copy()
    ensure_annotations(arr, "is_polymer")

    atomize = ~arr.is_polymer

    if "atomize" not in arr.get_annotation_categories():
        arr.set_annotation("atomize", atomize)
    else:
        arr.atomize |= atomize

    if "pn_unit_iid" in arr.get_annotation_categories():
        arr = flag_and_reassign_covalent_modifications(arr)
    elif "pn_unit_id" in arr.get_annotation_categories():
        if "transformation_id" not in arr.get_annotation_categories():
            arr.set_annotation("transformation_id", np.full(arr.array_length(), "1"))
        arr = add_pn_unit_iid_annotation(arr)
        arr = flag_and_reassign_covalent_modifications(arr)
    else:
        raise ValueError(
            "To compute the standard atomize mask, the input AtomArray must have either `pn_unit_iid` or `pn_unit_id` annotation. "
            "Please run `parse` or `prepare_atom_array` with `add_id_and_entity_annotations=True`."
        )

    standard_res_names = STANDARD_AA + STANDARD_RNA + STANDARD_DNA
    arr = atomize_by_ccd_name(arr, atomize_by_default=True, res_names_to_ignore=standard_res_names)
    return arr.atomize


def _validate_atomize(atom_array: AtomArray, atomize: np.ndarray) -> None:
    _all = struc.apply_residue_wise(atom_array, atomize, np.all)
    _any = struc.apply_residue_wise(atom_array, atomize, np.any)
    if np.any(_all != _any):
        raise ValueError("For each residue, all atoms must be atomized or none must be atomized.")


def atomize_by_ccd_name(
    atom_array: AtomArray,
    atomize_by_default: bool = True,
    res_names_to_atomize: list[str] = [],
    res_names_to_ignore: list[str] = [],
    move_atomized_part_to_end: bool = False,
    validate_atomize: bool = False,
) -> AtomArray:
    """
    Atomize residues by breaking down the res_name field into the actual element names.

    Args:
        atom_array (AtomArray): The atom array to atomize.
        atomize_by_default (bool): Whether to atomize residues by default.
        res_names_to_atomize (list[str]): List of residue names to atomize. Defaults to [].
        res_names_to_ignore (list[str]): List of residue names to ignore. These residues
            will only be atomized, if their `atomize` flag is already explicitly set to `True`, e.g. from a
            previous transform to sample random residues for atomization for data augmentation. Defaults to [].
        move_atomized_part_to_end (bool, optional): Whether to move atomized parts to the end of the array. Defaults to False.
            This is relevant for RF2AA, which follows the convention that atomized parts are grouped together at the end of the
            input.
        validate_atomize (bool, optional): Whether to validate that a residue is either atomized or not. Defaults to False.

    Returns:
        AtomArray: The atomized atom array. The `atomize` flag is set for each atom in the array.
            NOTE: The returned array may be reordered if `move_atomized_part_to_end` is True.
    """
    atomize = np.full(atom_array.array_length(), atomize_by_default, dtype=bool)

    # Exclude residues to ignore
    if len(res_names_to_ignore) > 0:
        atomize[np.isin(atom_array.res_name, res_names_to_ignore)] = False

    # Include residues to atomize
    if len(res_names_to_atomize) > 0:
        atomize[np.isin(atom_array.res_name, res_names_to_atomize)] = True

    # Include everything with the `atomize` flag from possible previous transforms
    #  this is used to manually define residues to atomize, e.g. as a data augmentation
    if "atomize" in atom_array.get_annotation_categories():
        atomize |= atom_array.atomize

    if validate_atomize:
        # ... validate that a residue is either atomized or not
        _validate_atomize(atom_array, atomize)

    # Perform atomization
    atom_array.set_annotation("atomize", atomize)

    if move_atomized_part_to_end:
        # as per RF2AA convention, the atomized parts are grouped together at the end
        #  of the input. This flag enables that.
        # NOTE: This needs to be done via `reshuffling` in order to preserve the correct
        #  bonding information.
        _idxs_pre_shuffling = np.arange(atom_array.array_length())
        reordered_idxs = np.concatenate(
            [_idxs_pre_shuffling[~atom_array.atomize], _idxs_pre_shuffling[atom_array.atomize]]
        )
        atom_array = atom_array[reordered_idxs]

    return atom_array


def flag_and_reassign_covalent_modifications(
    atom_array: AtomArray,
    p_reassign_modified_residue: float = 1.0,
    atomize_backbone_modifications: bool = True,
) -> AtomArray:
    """
    Mark covalent modifications for atomization and reassign the corresponding
    PN unit annotations.

    Args:
        atom_array (AtomArray): Current `AtomArray` within the Transform pipeline
        p_reassign_modified_residue: Probability of atomizing each side-chain-modified
            residue, sampled independently per residue and reused across its bonds.
            Must be between 0 and 1 inclusive. Defaults to `1.0`.
        atomize_backbone_modifications (bool, optional): Atomize the residue when the modifier
            attaches at its polymerization atom (a cyclization, crosslink, or terminal cap).
            Defaults to `True`; set `False` to treat such bonds as ordinary polymer linkages.

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
    if not 0.0 <= p_reassign_modified_residue <= 1.0:
        raise ValueError("p_reassign_modified_residue must be between 0 and 1 inclusive.")

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
        atom_array.set_annotation("atomize", np.array([False] * atom_array.array_length()))

    # Add the is_covalent_modification annotation to the AtomArray, if not already present
    if "is_covalent_modification" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("is_covalent_modification", np.array([False] * atom_array.array_length()))

    # Capture residue identities before PN unit annotations are reassigned.
    residue_starts = struc.get_residue_starts(atom_array)
    reassign_by_residue: dict[int, bool] = {}

    # Loop through inter-molecular bonds
    # NOTE: There aren't likely to be many inter-molecular bonds in the entry, so vectorization is not necessary and would be less readable
    for bond in bonds_to_check:
        # Get the atoms involved in the inter-molecular bonds
        atom_a = atom_array[bond[0]]
        atom_b = atom_array[bond[1]]

        # Note which atom is in the polymer and which is in the non-polymer
        polymer_atom, non_polymer_atom = (atom_a, atom_b) if atom_a.is_polymer else (atom_b, atom_a)

        # Backbone modifications use their own flag; side-chain decisions are shared across
        # all modifier bonds to the same original residue.
        is_backbone_modification = polymer_atom.atom_name in get_polymerization_atoms(polymer_atom.res_name)
        if is_backbone_modification:
            reassign = atomize_backbone_modifications
        else:
            polymer_atom_idx = bond[0] if atom_a.is_polymer else bond[1]
            residue_idx = int(np.searchsorted(residue_starts, polymer_atom_idx, side="right") - 1)
            if residue_idx not in reassign_by_residue:
                p = p_reassign_modified_residue
                reassign_by_residue[residue_idx] = p == 1.0 or (p > 0.0 and np.random.random() < p)
            reassign = reassign_by_residue[residue_idx]
        if reassign:
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
