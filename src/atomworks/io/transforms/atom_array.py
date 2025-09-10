"""Transforms operating predominantly on Biotite's AtomArray objects.

These operations should take as input, and return, AtomArray objects.
"""

import logging
from collections import Counter, defaultdict

import biotite.structure as struc
import networkx as nx
import numpy as np
import pandas as pd
from biotite.structure import AtomArray, AtomArrayStack, stack

from atomworks.common import listmap, not_isin, sum_string_arrays
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, HYDROGEN_LIKE_SYMBOLS, WATER_LIKE_CCDS
from atomworks.io.utils.bonds import (
    generate_inter_level_bond_hash,
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
    hash_graph,
)
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.io.utils.selection import annot_start_stop_idxs

logger = logging.getLogger("atomworks.io")

try:
    import hydride
except ImportError:
    logger.warning("Hydride library not found, hydrogens cannot be inferred. Pip install hydride to enable.")


def subset_atom_array(atom_array: AtomArray | AtomArrayStack, keep: np.ndarray) -> AtomArray | AtomArrayStack:
    """Subsets an AtomArray or AtomArrayStack by a boolean mask.

    Args:
        atom_array: The AtomArray or AtomArrayStack to subset.
        keep: Boolean mask indicating which atoms to keep.

    Returns:
        The subsetted AtomArray or AtomArrayStack.
    """
    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, keep]
    else:
        return atom_array[keep]


def is_any_coord_nan(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Returns a boolean mask indicating whether any coordinate is NaN for each atom.

    Args:
        atom_array: The AtomArray or AtomArrayStack to check.

    Returns:
        Boolean mask of shape [n_atoms] indicating NaN coordinates.
    """
    if isinstance(atom_array, AtomArrayStack):
        return np.isnan(atom_array.coord).any(axis=(0, -1))
    else:
        return np.isnan(atom_array.coord).any(axis=-1)


def remove_nan_coords(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Returns a copy of the AtomArray or AtomArrayStack with rows where any coordinate is NaN removed."""
    return subset_atom_array(atom_array, ~is_any_coord_nan(atom_array))


def remove_ccd_components(
    atom_array: AtomArray | AtomArrayStack, ccd_codes_to_remove: list[str]
) -> AtomArray | AtomArrayStack:
    """
    Remove atoms from the AtomArray or AtomArrayStack that have CCD codes in the ccd_codes_to_remove list.

    Parameters:
        atom_array (AtomArray): The array of atoms.
        ccd_codes_to_remove (list): A list of CCD codes to be removed from the atom array.

    Returns:
        AtomArray: The filtered atom array.
    """
    ccd_codes_to_remove = list(ccd_codes_to_remove)
    return subset_atom_array(atom_array, not_isin(atom_array.res_name, ccd_codes_to_remove))


def remove_hydrogens(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Removes hydrogens from the AtomArray or AtomArrayStack."""
    keep = not_isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)
    return subset_atom_array(atom_array, keep)


def remove_waters(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Removes waters from the AtomArray or AtomArrayStack."""
    return remove_ccd_components(atom_array, WATER_LIKE_CCDS)


def ensure_atom_array_stack(atom_array_or_stack: AtomArray | AtomArrayStack) -> AtomArrayStack:
    """Ensures that the input is an AtomArrayStack. If it is an AtomArray, it is converted to a stack."""
    if isinstance(atom_array_or_stack, AtomArray):
        return stack([atom_array_or_stack])
    elif isinstance(atom_array_or_stack, AtomArrayStack):
        return atom_array_or_stack
    else:
        raise TypeError(f"Expected AtomArray or AtomArrayStack, got {type(atom_array_or_stack)}")


def resolve_arginine_naming_ambiguity(atom_array: AtomArray, raise_on_error: bool = True) -> AtomArray:
    """
    Arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2)
    """
    # TODO: Generalize to AtomArrayStack
    arg_mask = atom_array.res_name == "ARG"
    arg_nh1_mask = (atom_array.atom_name == "NH1") & arg_mask
    arg_nh2_mask = (atom_array.atom_name == "NH2") & arg_mask
    arg_cd_mask = (atom_array.atom_name == "CD") & arg_mask

    try:
        cd_nh1_dist = np.linalg.norm(atom_array.coord[arg_cd_mask] - atom_array.coord[arg_nh1_mask], axis=-1)
        cd_nh2_dist = np.linalg.norm(atom_array.coord[arg_cd_mask] - atom_array.coord[arg_nh2_mask], axis=-1)
        both_finite = np.isfinite(cd_nh1_dist) & np.isfinite(cd_nh2_dist)

        # Check if there are any name swamps required
        local_to_swap = (cd_nh1_dist > cd_nh2_dist) & both_finite  # local mask
        # turn local mask into global mask
        to_swap = np.zeros(atom_array.array_length(), dtype=bool)
        to_swap[arg_nh1_mask] = local_to_swap
        to_swap[arg_nh2_mask] = local_to_swap

        # Swap NH1 and NH2 names if NH1 is further from CD than NH2
        if np.any(to_swap):
            logger.debug(f"Resolving {np.sum(local_to_swap)} arginine naming ambiguities.")
            prev_nh1_coord = atom_array.coord[arg_nh1_mask & to_swap]
            prev_nh2_coord = atom_array.coord[arg_nh2_mask & to_swap]

            atom_array.coord[arg_nh1_mask & to_swap] = prev_nh2_coord
            atom_array.coord[arg_nh2_mask & to_swap] = prev_nh1_coord

    except ValueError as e:
        if raise_on_error:
            raise e
        else:
            logger.warning(f"Error resolving arginine naming ambiguity: {e}. Returning original atom array.")

    return atom_array


def mse_to_met(atom_array: AtomArray) -> AtomArray:
    """Convert MSE residues (selenomethionine) to MET (methionine)."""
    mse_mask = atom_array.res_name == "MSE"
    if np.any(mse_mask):
        se_mask = (atom_array.atom_name == "SE") & mse_mask
        logger.debug(f"Converting {np.sum(se_mask)} MSE residues to MET.")

        # Update residue name, hetero flag, and element
        atom_array.res_name[mse_mask] = "MET"
        atom_array.hetero[mse_mask] = False
        atom_array.atom_name[se_mask] = "SD"

        # ... handle cases for integer or string representations of element
        _elt_prev = atom_array.element[se_mask][0]
        if _elt_prev == "SE":
            atom_array.element[se_mask] = "S"
        elif _elt_prev == ELEMENT_NAME_TO_ATOMIC_NUMBER["SE"]:
            atom_array.element[se_mask] = ELEMENT_NAME_TO_ATOMIC_NUMBER["S"]
        elif _elt_prev == str(ELEMENT_NAME_TO_ATOMIC_NUMBER["SE"]):
            atom_array.element[se_mask] = str(ELEMENT_NAME_TO_ATOMIC_NUMBER["S"])

        # Reorder atoms for canonical MET ordering
        atom_array_mse = atom_array[mse_mask]
        atom_array_mse = atom_array_mse[struc.info.standardize_order(atom_array_mse)]
        atom_array[mse_mask] = atom_array_mse

    return atom_array


def keep_last_residue(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """
    Removes duplicate residues in the atom array, keeping only the last occurrence.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.

    Returns:
        AtomArray: The atom array with duplicate residues removed.
    """
    atom_df = pd.DataFrame(
        {
            "chain_id": atom_array.chain_id,
            "res_id": atom_array.res_id,
            "res_name": atom_array.res_name,
        }
    )

    # Get the mask of duplicates based on the combination of chain_id, res_id, and res_name
    collapsed_df = atom_df.drop_duplicates(subset=["chain_id", "res_id", "res_name"])

    # Get duplicates based on res_id, keeping the last
    duplicate_mask = collapsed_df.duplicated(subset=["chain_id", "res_id"], keep="last")
    duplicates_df = collapsed_df[duplicate_mask]

    # Perform a left merge to find rows in atom_df that are also in duplicates_df
    merged_df = atom_df.merge(duplicates_df, on=["chain_id", "res_id", "res_name"], how="left", indicator=True)

    # Create a mask where True indicates the row is not in duplicates_df
    keep = merged_df["_merge"] == "left_only"

    # Remove rows from atom_array with the deletion mask
    return subset_atom_array(atom_array, keep)


def maybe_fix_non_polymer_at_symmetry_center(
    atom_array_stack: AtomArrayStack, clash_distance: float = 1.0, clash_ratio: float = 0.5
) -> AtomArrayStack:
    """
    In some PDB entries, non-polymer molecules are placed at the symmetry center and clash with themselves when
    transformed via symmetry operations. We should remove the duplicates in these cases, keeping the identity copy.

    We consider a non-polymer to be clashing with itself if at least `clash_ratio` of its atoms clash with the symmetric copy.

    Examples:
    — PDB ID `7mub` has a potassium ion at the symmetry center that when reflected with the symmetry operation clashes with itself.
    — PDB ID `1xan` has a ligand at a symmetry center that similarly when refelcted clashes with itself.

    Args:
        atom_array (AtomArray): The atom array to be patched.
        clash_distance (float): The distance threshold for two atoms to be considered clashing.
        clash_ratio (float): The percentage of atoms that must clash for the molecule to be considered clashing.

    Returns:
        AtomArray: The patched atom array.
    """
    # Select one model AtomArray to simplify computations
    atom_array = atom_array_stack[0]

    # Filter to only atoms with coordinates to avoid non-physical clashes at the origin
    if "occupancy" in atom_array.get_annotation_categories():
        resolved_mask = atom_array.occupancy > 0
    else:
        resolved_mask = np.ones(atom_array.array_length(), dtype=bool)
    resolved_atom_array = atom_array[(resolved_mask) & (~is_any_coord_nan(atom_array))]

    if not np.any(~resolved_atom_array.is_polymer):
        return atom_array_stack  # Early exit
    else:
        non_polymers = resolved_atom_array[~resolved_atom_array.is_polymer]  # [n]

        # Build cell list for rapid distance computations
        cell_list = struc.CellList(non_polymers, cell_size=3.0)

        # Quick check to see whether any non-polymer is closer than 0.05A to any other.
        clash_matrix = cell_list.get_atoms(non_polymers.coord, clash_distance, as_mask=True)  # [n, n]
        identity_matrix = np.identity(len(non_polymers), dtype=bool)
        if np.array_equal(clash_matrix, identity_matrix):
            return atom_array_stack  # Early exit
        else:
            # Remove identity matrix so we don't count self-clashes
            clash_matrix = clash_matrix & ~identity_matrix
        logger.debug("Found clashing non-polymer at a symmetry center, resolving.")

        # Get list of chain_ids with clashing atoms (for computational efficiency)
        clashing_atom_mask = np.sum(clash_matrix, axis=1) > 0
        clashing_chain_ids = np.unique(non_polymers.chain_id[clashing_atom_mask])

        # For each clashing chain, we check whether any non-polymer is clashing with a symmetric copy of itself
        # We count the clashes with each symmetric copy of itself and remove those that have a clash ratio above the threshold
        # We keep the identity transformation, or the lowest transformation ID in the case of multiple symmetric copies
        chain_iids_to_remove = []
        for chain_id in clashing_chain_ids:
            chain_mask = non_polymers.chain_id == chain_id
            mask = chain_mask & clashing_atom_mask  # Mask for clashing atoms in the current chain
            chain_clash_matrix = clash_matrix[mask][:, mask]

            # Loop through possible transformation ID's
            transformation_ids_to_check = sorted(np.unique(non_polymers.transformation_id[mask].astype(str)).tolist())
            while transformation_ids_to_check:
                transformation_id = str(transformation_ids_to_check.pop(0))
                transformation_mask = non_polymers.transformation_id == str(transformation_id)
                # Create matrix where the rows correspond to the atoms of the current transformation and the columns corresponded to the other transformations
                chain_clash_matrix = clash_matrix[mask & transformation_mask][
                    :, mask & ~transformation_mask
                ]  # [current transformation clashing atoms, other transformations clashing atoms]
                # We can then count clashes by transformation ID
                transformation_id_matrix = np.tile(
                    non_polymers.transformation_id[mask & ~transformation_mask], (chain_clash_matrix.shape[0], 1)
                )

                # Apply chain_clash_matrix to transformation_id_matrix so we can count clashes by transformation ID
                clashing_transformation_ids = np.where(chain_clash_matrix, transformation_id_matrix, None).flatten()
                clash_count_by_transformation_id = Counter(
                    clashing_transformation_ids[clashing_transformation_ids != np.array(None)]
                )
                threshold = clash_ratio * np.sum(chain_mask & transformation_mask)

                # For each transformation ID with a clash ratio above the threshold, note the chain_iid to remove, and remove from the list to check
                transformation_ids_to_remove = [
                    trans_id for trans_id, count in clash_count_by_transformation_id.items() if count > threshold
                ]
                chain_iids_to_remove.extend([f"{chain_id}_{trans_id}" for trans_id in transformation_ids_to_remove])
                transformation_ids_to_check = [
                    id_ for id_ in transformation_ids_to_check if str(id_) not in transformation_ids_to_remove
                ]

        # Filter and return
        keep_mask = not_isin(atom_array.chain_iid, np.array(chain_iids_to_remove, dtype=atom_array.chain_iid.dtype))
        atom_array_stack = atom_array_stack[:, keep_mask]
        return atom_array_stack


def add_polymer_annotation(atom_array: AtomArray | AtomArrayStack, chain_info_dict: dict) -> AtomArray | AtomArrayStack:
    """Adds an annotation to the atom array to indicate whether a chain is a polymer.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.
        chain_info_dict (dict): Dictionary containing the sequence details of each chain.

    Returns:
        AtomArray: The updated atom array with the polymer annotation added.
    """
    chain_ids = atom_array.get_annotation("chain_id")
    is_polymer = np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])
    atom_array.set_annotation("is_polymer", is_polymer)
    return atom_array


def update_nonpoly_seq_ids(atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
    """
    Updates the sequence IDs of non-polymeric chains in the atom array to the author sequence IDs.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.
        chain_info_dict (dict): Dictionary containing the sequence details of each chain.

    Returns:
        AtomArray: The updated atom array with the sequence IDs updated for non-polymeric chains.
    """
    # For non-polymeric chains, we use the author sequence ids
    author_seq_ids = atom_array.get_annotation("auth_seq_id")
    chain_ids = atom_array.get_annotation("chain_id")

    # Create mask based on the is_polymer column
    non_polymer_mask = ~np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])

    # Update the atom_array_label with the (1-indexed) author sequence ids
    atom_array.res_id[non_polymer_mask] = author_seq_ids[non_polymer_mask]

    return atom_array


def _safe_to_int(x: str | int | None) -> int:
    """Robustly convert values to integers: map '.', empty strings, and None to -1; parse numerics otherwise"""
    if x is None:
        return -1
    s = str(x).strip()
    if s in (".", ""):
        return -1
    try:
        return int(s)
    except Exception:
        return -1


def replace_negative_res_ids_with_auth_seq_id(atom_array: AtomArray) -> AtomArray:
    """
    Replaces res_id values of -1 with the corresponding auth_seq_id values.

    When loading from the PDB, this step is generally not needed; however, some AF-3 predictions
    have negative res_ids without labeling chains as non-polymeric via the entity_id field.

    Args:
        atom_array (AtomArray): The atom array to fix.

    Returns:
        AtomArray: The updated atom array with negative res_ids replaced by auth_seq_ids.
    """
    author_seq_ids = atom_array.get_annotation("auth_seq_id")
    negative_res_id_mask = atom_array.res_id == -1

    # Convert auth_seq_ids to int if they are strings (as they are sometimes from AF-3 predictions)
    if author_seq_ids.dtype.kind in "UO":  # Unicode or Object (string-like)
        author_seq_ids = np.frompyfunc(_safe_to_int, 1, 1)(author_seq_ids).astype(int)

    atom_array.res_id[negative_res_id_mask] = author_seq_ids[negative_res_id_mask]

    return atom_array


def add_charge_from_ccd_codes(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """
    Adds charge annotations to an atom array based on the Chemical Component Dictionary (CCD) codes.

    Retrieves charge information from the CCD for each residue and assigns it to matching atoms.
    If a residue or atom is not found in the CCD, a charge of 0 is assigned. If a charge annotation
    is already present, it is overwritten.

    Args:
        atom_array: The atom array to which charge annotations will be added.
            Can be either an AtomArray or AtomArrayStack.

    Returns:
        The input atom array with added charge annotations.


    WARNING: This function will assume that each residue in the atom array is exactly as in the CCD.
       Therefore the charges will be incorrect if it is in a different protonation state, or has been
       ionized or else in the original structure. Use this function with caution!

    NOTE: If you want to add charges to canonical amino acids based on the pH, take a look at the
        'hydride.estimate_amino_acid_charges' function instead.

    Example:
        >>> atom_array = load_any("6lyz.cif", model=1)
        >>> atom_array_with_charges = add_charge_from_ccd_codes(atom_array)
    """
    # Warn if a charge annotation is already present
    if "charge" in atom_array.get_annotation_categories():
        logger.info("Charge annotation already present in atom array. It will be overwritten.")

    # Build up a lookup table (res_name, atom_name) -> charge for each res_name that appears in the atom_array
    unique_res_names = np.unique(atom_array.res_name)
    charge_lookup_table: dict[tuple[str, str], float] = {}

    for res_name in unique_res_names:
        try:
            ccd_array = atom_array_from_ccd_code(res_name)
            # Use dictionary comprehension to build lookup entries for this residue
            charge_lookup_table.update(
                {
                    (res_name, atom_name): charge
                    for atom_name, charge in zip(ccd_array.atom_name, ccd_array.charge, strict=False)
                }
            )
        except ValueError:
            logger.info(f"CCD charge look-up failed for {res_name}. Assuming charge is 0 for all atoms.")
            continue

    # Create the charge annotations for the atom array
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    charges = np.array(
        [charge_lookup_table.get((res, atom), 0) for res, atom in zip(res_names, atom_names, strict=False)]
    )

    # Set the charges annotation
    atom_array.set_annotation("charge", charges)

    return atom_array


def add_hydrogen_atom_positions(
    atom_array: AtomArray | AtomArrayStack,
    residue_level_annots_to_copy_to_hydrogens: list[str] = [],
) -> AtomArray | AtomArrayStack:
    """Add hydrogens using biotite supported hydride library.

    Removes any existing hydrogens first, then adds new hydrogens using the hydride library.

    Args:
        atom_array: The atom array to which hydrogens will be added.
        residue_level_annots_to_copy_to_hydrogens (list[str]): A list of residue-level annotations that will be copied
            over to the newly-added hydrogens for each residue.

    Returns:
        The updated atom array with hydrogens added, preserving the input type.
    """
    # Remove existing hydrogens
    atom_array = remove_hydrogens(atom_array)

    # Determine which fields to copy from the original array to the new hydrogens
    fields_to_copy_from_residue_if_present = ["auth_seq_id", "label_entity_id"]
    fields_to_copy_from_residue_if_present.extend(residue_level_annots_to_copy_to_hydrogens)
    fields_to_copy_from_residue_if_present = list(
        set(fields_to_copy_from_residue_if_present).intersection(set(atom_array.get_annotation_categories()))
    )

    # Ensure charge annotation exists
    if "charge" not in atom_array.get_annotation_categories():
        atom_array = add_charge_from_ccd_codes(atom_array)

    # Helper function to copy annotations from one array to another
    def _copy_missing_annotations_residue_wise(
        from_array: AtomArray, to_array: AtomArray, fields_to_copy: list[str]
    ) -> AtomArray:
        """Copy specified annotations residue-wise from one AtomArray to another. Updates annotations in-place."""
        residue_starts = struc.get_residue_starts(from_array)
        residue_starts_atom_array = from_array[residue_starts]
        annot = {item: getattr(residue_starts_atom_array, item) for item in fields_to_copy_from_residue_if_present}
        for field in fields_to_copy:
            updated_field = struc.spread_residue_wise(to_array, annot[field])
            to_array.set_annotation(field, updated_field)
        return to_array

    def _add_hydrogens_nan_tolerant(atom_array: AtomArray) -> AtomArray:
        """Adds hydrogens to the input AtomArray, safely handling the case in which some atoms have NaN coordinates"""
        original_nan_coords_mask = np.any(np.isnan(atom_array.coord), axis=1)
        if np.any(original_nan_coords_mask):
            # Temporarily set NaN coordinates to zero so that hydride doesn't error
            atom_array.coord[original_nan_coords_mask] = np.zeros((np.sum(original_nan_coords_mask), 3))

            # Add hydrogens using hydride
            result_atom_array, original_atoms_mask = hydride.add_hydrogen(atom_array)

            # Reset the coordinates of atoms that originally had at least one NaN coordinate to be fully NaN
            originally_nan_inds = np.arange(result_atom_array.array_length())[original_atoms_mask][
                original_nan_coords_mask
            ]
            result_atom_array.coord[originally_nan_inds, :] = np.nan

            # For any newly-added hydrogens bonded to heavy atoms with NaN coordinates, set their coordinates to NaN as well
            result_nan_coords_mask = np.any(np.isnan(result_atom_array.coord), axis=1)
            heavy_atom_nan_idces = np.where(result_nan_coords_mask & ~(result_atom_array.element == "H"))[0]
            for idx in heavy_atom_nan_idces:
                bonded_atoms = result_atom_array.bonds.get_bonds(idx)[0]
                bonded_h_atoms = bonded_atoms[result_atom_array[bonded_atoms].element == "H"]
                new_bonded_h_atoms = bonded_h_atoms[~original_atoms_mask[bonded_h_atoms]]
                result_atom_array.coord[new_bonded_h_atoms, :] = np.nan
        else:
            result_atom_array, original_atoms_mask = hydride.add_hydrogen(atom_array)

        return result_atom_array

    if isinstance(atom_array, AtomArrayStack):
        updated_arrays = []
        for old_arr in atom_array:
            arr = _add_hydrogens_nan_tolerant(old_arr)
            arr = _copy_missing_annotations_residue_wise(old_arr, arr, fields_to_copy_from_residue_if_present)
            updated_arrays.append(arr)

        ret_array = struc.stack(updated_arrays)

    elif isinstance(atom_array, AtomArray):
        arr = _add_hydrogens_nan_tolerant(atom_array)
        ret_array = _copy_missing_annotations_residue_wise(atom_array, arr, fields_to_copy_from_residue_if_present)

    return ret_array


def add_pn_unit_id_annotation(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """
    Adds the polymer/non-polymer unit ID (pn_unit_id) annotation to the AtomArray.
    Two covalently bonded ligands are considered one PN unit, but a ligand bonded to a protein is considered two PN units.
    See the README glossary for more details on how we define `chains`, `pn_units`, and `molecules` within this codebase.

    Args:
        atom_array (AtomArray): The AtomArray to process.

    Returns:
        atom_array (AtomArray): The AtomArray including the `pn_unit_id` annotation.
    """
    # ...initialize the pn_unit_id to chain_id (we will later update for multi-chain non-polymer PN units)
    pn_unit_id_annotation = atom_array.chain_id.astype(object)

    # ...make the NetworkX graph for non-polymer chains
    non_polymer_atom_array = atom_array[~atom_array.is_polymer]
    connected_chains = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(non_polymer_atom_array, "chain_id"))

    for connected_chain in connected_chains:
        # ...set the same the pn_unit_id for each chain in the connected chain
        pn_unit_id = ",".join(sorted(connected_chain))
        for chain_id in connected_chain:
            pn_unit_id_annotation[atom_array.chain_id == chain_id] = pn_unit_id

    atom_array.set_annotation("pn_unit_id", pn_unit_id_annotation.astype(str))

    return atom_array


def add_pn_unit_iid_annotation(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Adds the polymer/non-polymer unit instance ID (pn_unit_iid) annotation to the AtomArray or AtomArrayStack."""
    # ...create an array that concatenates the pn_unit_id and transformation_id
    _temp_pn_unit_iid = sum_string_arrays(atom_array.pn_unit_id, "_", atom_array.transformation_id)
    _final_pn_unit_iid = np.full(atom_array.array_length(), fill_value="", dtype=object)

    # ...iterate through unique pn_unit_iids
    # (We implicitly assume that a given pn_unit_id will have the same transformation_id across all atoms in the unit)
    for pn_unit_iid in np.unique(_temp_pn_unit_iid):
        # ...get the pn_unit_atom_array
        pn_unit_atom_array = subset_atom_array(atom_array, _temp_pn_unit_iid == pn_unit_iid)
        # ...get the transformation_id and pn_unit_id (which is the same for all atoms in the unit)
        transformation_id = pn_unit_atom_array.transformation_id[0]
        pn_unit_id = pn_unit_atom_array.pn_unit_id[0].astype(str)

        # ...split apart the pn_unit_id by commas
        pn_unit_ids = pn_unit_id.split(",")

        # ...add the transformation_id to each pn_unit_id
        pn_unit_iids = [f"{unit_id}_{transformation_id}" for unit_id in pn_unit_ids]

        # ...join the instance-level identifiers back into a single string
        pn_unit_iid_formatted = ",".join(pn_unit_iids)

        # ...update the AtomArray with the instance-level identifier
        _final_pn_unit_iid[_temp_pn_unit_iid == pn_unit_iid] = pn_unit_iid_formatted

    atom_array.set_annotation("pn_unit_iid", _final_pn_unit_iid.astype(str))

    return atom_array


def add_molecule_id_annotation(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Adds the molecule ID (molecule_id) annotation to the AtomArray."""
    # ...initialize the pn_unit_id to chain_id (we will later update for multi-chain non-polymer PN units)
    atom_array.add_annotation("molecule_id", dtype=np.int16)

    # ...make the NetworkX graph for all pn_units
    connected_pn_units = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(atom_array, "pn_unit_id"))

    # ...iterate through connected pn_units
    for idx, connected_pn_unit in enumerate(connected_pn_units):
        # ...set the same the molecule_id for each pn_unit in the connected pn_unit
        molecule_id = idx
        for pn_unit_id in connected_pn_unit:
            atom_array.molecule_id[atom_array.pn_unit_id == pn_unit_id] = molecule_id

    return atom_array


def add_molecule_iid_annotation(atom_array_stack: AtomArrayStack) -> AtomArrayStack:
    """Adds the molecule instance ID (molecule_iid) annotation to the AtomArrayStack"""
    # ...concatenate molecule_id and transformation_id to create a unique molecule instance ID
    molecule_iids_str = np.char.add(
        atom_array_stack.molecule_id.astype(str), atom_array_stack.transformation_id.astype(str)
    )

    # ...map each unique molecule_iid to an integer (0-indexed)
    _, inverse_indices = np.unique(molecule_iids_str, return_inverse=True)

    # ...set the annotation
    atom_array_stack.set_annotation("molecule_iid", inverse_indices.astype(np.int16))

    return atom_array_stack


def annotate_entities(
    atom_array: AtomArray,
    level: str,
    lower_level_id: str | list[str],
    lower_level_entity: str,
    add_inter_level_bond_hash: bool = True,
) -> tuple[AtomArray, dict]:
    """
    Annotates entities in an AtomArray at a given `id` level, based on the connectivity and annotations at the lower level.

    The intended use is, for example:
        - For the `molecule` level, `molecule_entities` are generated for each `molecule_id` based on the connectivty
            at the `pn_unit` level.
        - For the `pn_unit` level, `pn_unit_entities` are generated for each `pn_unit_id` based on the connectivty
            at the `chain` level.
        - For the `chain` level, `chain_entities` are generated for each `chain_id` based on the connectivty at the `residue`
            level.

    Args:
        - atom_array (AtomArray): The AtomArray to process.
        - level (str): The level at which to annotate entities (e.g., "chain", "pn_unit", "entity")
        - lower_level_id (str | list[str]): A list of annotations to consider for determining segment boundaries at a lower level.
            E.g. "pn_unit_id", "chain_id" or "res_id".
        - lower_level_entity (str): The annotation to use for identifying entities at the lower level.
            E.g. "pn_unit_entity", "chain_entity" or "res_name".
        - add_inter_level_bond_hash (bool): Whether to add a hash of the inter-level bonds to the entity hash.
            For some cases, this may be necessary to distinguish entities (e.g., when determining molecule-level
            entities). In others (e.g., for polymers), this may be overkill.

    Returns:
        - Tuple[AtomArray, dict]: A tuple containing:
            - atom_array (AtomArray): The updated AtomArray with the entity annotation.
            - entities_info (dict): A dictionary mapping entity IDs to lists of instance IDs.

    Example:
        >>> atom_array = AtomArray(...)
        >>> entities_at_level, entities_info = annotate_entities(
        ...     atom_array, level="chain", lower_level_id="res_id", lower_level_entity="res_name"
        ... )
        >>> print(entities_at_level)
        [0, 0, 1, 1, 2, 2]
        >>> print(entities_info)
        {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    """
    _next_available_entity_id = 0
    _hash_to_entity_id = {}

    ids_at_level = np.unique(atom_array.get_annotation(level + "_id"))

    # ... initialize annotations to fill
    entities_annotation = np.zeros(len(atom_array), dtype=int)
    entities_info = defaultdict(list)

    for instance_id in np.unique(ids_at_level):
        is_instance = atom_array.get_annotation(level + "_id") == instance_id
        instance = atom_array[is_instance]

        # ... get connectivity and node annotations for the coarse graph at the lower level
        _, edges = get_coarse_graph_as_nodes_and_edges(instance, lower_level_id)
        instance_graph = nx.Graph()
        instance_graph.add_edges_from(edges)

        # ... set node attributes to lower level entities
        lower_level_iter = struc.segments.segment_iter(
            instance, annot_start_stop_idxs(instance, lower_level_id, add_exclusive_stop=True)
        )
        node_attrs = {
            idx: lower_level_instance.get_annotation(lower_level_entity)[0]
            for idx, lower_level_instance in enumerate(lower_level_iter)
        }
        nx.set_node_attributes(instance_graph, node_attrs, "node")

        # ... create the graph hash
        hash = hash_graph(instance_graph, node_attr="node")

        # ... add the inter-level bond hash (only consider the first lower level id; since we hash at the atom-level, this simplication is valid)
        if add_inter_level_bond_hash:
            hash += generate_inter_level_bond_hash(
                atom_array=instance,
                lower_level_id=lower_level_id[0] if isinstance(lower_level_id, list) else lower_level_id,
                lower_level_entity=lower_level_entity,
            )

        # ... check if the graph has been seen before
        if hash in _hash_to_entity_id:
            entity_id = _hash_to_entity_id[hash]
        else:
            entity_id = _next_available_entity_id
            _hash_to_entity_id[hash] = entity_id
            _next_available_entity_id += 1

        # ... assign the entity id to the instance
        entities_annotation[is_instance] = entity_id
        entities_info[entity_id].append(instance_id)

    atom_array.set_annotation(level + "_entity", entities_annotation)

    return atom_array, dict(entities_info)


def add_chain_iid_annotation(atom_array_stack: AtomArrayStack) -> AtomArrayStack:
    """Adds the chain instance ID (chain_iid) annotation to the AtomArrayStack"""
    # ...concatenate chain_id and transformation_id to create a unique chain instance ID
    chain_iid = sum_string_arrays(
        atom_array_stack.chain_id,
        "_",
        atom_array_stack.transformation_id,
    )
    atom_array_stack.set_annotation("chain_iid", chain_iid)
    return atom_array_stack


def add_iid_annotations_to_assemblies(
    assemblies_dict: dict[str | int, AtomArray | AtomArrayStack],
) -> dict[str | int, AtomArray | AtomArrayStack]:
    """Adds chain, PN unit, and molecule IIDs to assembly AtomArrayStacks."""
    for assembly_id, assembly in assemblies_dict.items():
        # ...add chain IIDs
        assembly = add_chain_iid_annotation(assembly)

        # ...check if we added bonds to the atom array

        # ...add PN unit IIDs, if we have the `pn_unit_id` annotation
        if "pn_unit_id" in assembly.get_annotation_categories():
            assembly = add_pn_unit_iid_annotation(assembly)

        # ...add molecule IIDs, if we have the `molecule_id` annotation
        if "molecule_id" in assembly.get_annotation_categories():
            assembly = add_molecule_iid_annotation(assembly)

        # ...update the dictionary
        assemblies_dict[assembly_id] = assembly

    return assemblies_dict


def add_id_and_entity_annotations(atom_array: AtomArray) -> AtomArray:
    """Adds all 6 ('chain', 'pn_unit', 'molecule') x ('id', 'entity') annotations to the AtomArray."""
    # ...annotate PN units (requires bonds)
    atom_array = add_pn_unit_id_annotation(atom_array)

    # ...annotate molecules (requires bonds)
    atom_array = add_molecule_id_annotation(atom_array)

    levels = ["chain", "pn_unit", "molecule"]
    lower_level_ids = ["res_id", "chain_id", "pn_unit_id"]
    lower_level_entities = ["res_name", "chain_entity", "pn_unit_entity"]

    for level, lower_level_id, lower_level_entity in zip(levels, lower_level_ids, lower_level_entities, strict=False):
        # ...annotate entities at appropriate level
        atom_array, _ = annotate_entities(
            atom_array=atom_array,
            level=level,
            lower_level_id=lower_level_id,
            lower_level_entity=lower_level_entity,
        )

    return atom_array


def add_chain_type_annotation(
    atom_array: AtomArray | AtomArrayStack, chain_info_dict: dict
) -> AtomArray | AtomArrayStack:
    """
    Adds a chain_type annotation to the AtomArray.

    Args:
        - atom_array (AtomArray | AtomArrayStack): The full atom array.
        - chain_info_dict (dict): A dictionary mapping chain IDs to chain information.

    Returns:
        - AtomArray | AtomArrayStack: The AtomArray with the chain_type annotation added as an integer.
    """
    # Add annotation for chain_type as an integer
    atom_array.add_annotation("chain_type", dtype=np.int8)
    for chain_id in np.unique(atom_array.chain_id):
        chain_type = chain_info_dict[chain_id]["chain_type"]
        # We use the integer representation of the ChainType enum for efficiency
        atom_array.chain_type[atom_array.chain_id == chain_id] = chain_type.value

    # Return the modified atom array
    return atom_array


def add_atomic_number_annotation(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Adds the atomic number (atomic_number) annotation to the AtomArray."""
    atom_array.set_annotation(
        "atomic_number",
        np.array(listmap(ELEMENT_NAME_TO_ATOMIC_NUMBER.get, np.char.upper(atom_array.element)), dtype=np.int8),
    )
    return atom_array
