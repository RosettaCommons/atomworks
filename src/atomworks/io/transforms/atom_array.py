"""Transforms operating predominantly on Biotite's AtomArray objects.

These operations should take as input, and return, AtomArray objects.
"""

import logging
from collections import Counter, defaultdict

import biotite.structure as struc
import networkx as nx
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.common import exists, not_isin, sum_string_arrays
from atomworks.constants import (
    WATER_LIKE_CCDS,
)
from atomworks.io.utils.atom_array import remove_hydrogens  # noqa: F401
from atomworks.io.utils.atom_array_plus import stack_any
from atomworks.io.utils.bonds import (
    generate_inter_level_bond_hash,
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
    hash_graph,
)
from atomworks.io.utils.chain_info import update_sequences_from_res_names
from atomworks.io.utils.selection import annot_start_stop_idxs

logger = logging.getLogger("atomworks.io")


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


def remove_waters(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Removes waters from the AtomArray or AtomArrayStack."""
    return remove_ccd_components(atom_array, WATER_LIKE_CCDS)


def ensure_atom_array_stack(atom_array_or_stack: AtomArray | AtomArrayStack | list[AtomArray]) -> AtomArrayStack:
    """Ensures that the input is an AtomArrayStack. If it is an AtomArray or list, it is converted to a stack."""
    if isinstance(atom_array_or_stack, list):
        return stack_any(atom_array_or_stack)
    elif isinstance(atom_array_or_stack, AtomArray):
        return stack_any([atom_array_or_stack])
    elif isinstance(atom_array_or_stack, AtomArrayStack):
        return atom_array_or_stack
    else:
        raise TypeError(f"Expected AtomArray, AtomArrayStack, or list[AtomArray], got {type(atom_array_or_stack)}")


def resolve_arginine_naming_ambiguity(atom_array: AtomArray, raise_on_error: bool = True) -> AtomArray:
    """Arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2)"""
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


def mse_to_met(
    atom_array: AtomArray | AtomArrayStack,
    *,
    chain_info: dict[str, dict] | None = None,
) -> AtomArray | AtomArrayStack:
    """Convert MSE residues (selenomethionine) to MET (methionine).

    Within crystal structures, selenomethionine (MSE) are often used to solve the phase problem.

    Args:
      atom_array: Atom array or stack to modify.
      chain_info: Optional chain information dictionary to update. If provided,
        will update ``res_name``, ``processed_entity_canonical_sequence``, and
        ``processed_entity_non_canonical_sequence`` fields for chains containing MSE.

    Returns:
      The modified atom array (same object, modified in-place).
      Modifies the chain_info dictionary in-place if provided.
    """
    # Unify handling by converting to stack
    is_single_model = isinstance(atom_array, AtomArray)
    atom_array = ensure_atom_array_stack(atom_array)

    mse_mask = atom_array.res_name == "MSE"
    if np.any(mse_mask):
        # Create a mask for the selenium atom within MSE, which we will convert to sulfur (SD)
        se_mask = (atom_array.atom_name == "SE") & mse_mask
        logger.debug(f"Converting {np.sum(se_mask)} MSE residues to MET.")

        # Update residue name, hetero flag, and element (broadcasts across all models)
        atom_array.res_name[mse_mask] = "MET"
        atom_array.hetero[mse_mask] = False
        atom_array.atom_name[se_mask] = "SD"
        atom_array.element[se_mask] = "S"

        # Reorder atoms to canonical MET ordering (all models at once)
        mse_indices = np.where(mse_mask)[0]
        mse_atoms = atom_array[0][mse_mask]  # Get MSE atoms from first model
        mse_order = struc.info.standardize_order(mse_atoms)

        # Build full reordering index: identity for non-MSE, reordered for MSE
        full_index = np.arange(atom_array.array_length())
        full_index[mse_indices] = mse_indices[mse_order]

        # Reorder entire stack at once
        atom_array = atom_array[:, full_index]

    # Update chain_info if provided
    if exists(chain_info):
        for chain_data in chain_info.values():
            if "res_name" not in chain_data:
                continue

            # Ensure res_name is a list and check for MSE as an element (not substring)
            res_names = chain_data.get("res_name", [])
            has_mse = isinstance(res_names, list) and "MSE" in res_names
            if not has_mse:
                continue

            # Convert MSE to MET in three-letter sequence
            chain_data["res_name"] = ["MET" if res_name == "MSE" else res_name for res_name in chain_data["res_name"]]

            # Recompute one-letter sequences after conversion
            update_sequences_from_res_names(chain_data)

    # Return same type as input
    return atom_array[0] if is_single_model else atom_array


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

        # Fast path when only diagonal elements present
        n_clashes = np.count_nonzero(clash_matrix)
        n_atoms = len(non_polymers)
        if n_clashes == n_atoms:
            return atom_array_stack

        # Remove identity matrix so we don't count self-clashes
        identity_matrix = np.identity(n_atoms, dtype=bool)
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
    is_polymer = np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids], dtype=bool)
    atom_array.set_annotation("is_polymer", is_polymer)
    return atom_array


def update_nonpoly_seq_ids(atom_array: AtomArray | AtomArrayStack, chain_info_dict: dict) -> AtomArray | AtomArrayStack:
    """Updates the sequence IDs of non-polymeric chains in the atom array to the author sequence IDs.

    Args:
        atom_array (AtomArray | AtomArrayStack): The atom array containing the chain information.

    Returns:
        AtomArray | AtomArrayStack: The updated atom array with the sequence IDs updated for non-polymeric chains.

    TODO: Delete this function when we update regression tests so we remove reliance on `auth_seq_id` altogether
    """
    chain_ids = atom_array.get_annotation("chain_id")
    author_seq_ids = atom_array.get_annotation("auth_seq_id")
    non_polymer_mask = ~np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids], dtype=bool)

    # ... update the chain_info dictionary
    for chain_id in np.unique(atom_array.chain_id[non_polymer_mask]):
        original_res_ids = chain_info_dict[chain_id]["res_id"]
        chain_mask = atom_array.chain_id == chain_id
        new_res_ids = []
        for res_id in original_res_ids:
            res_id_mask = atom_array.res_id == res_id
            new_res_ids.append(
                author_seq_ids[res_id_mask & chain_mask][0]
            )  # Assuming all atoms in the residue have the same author seq id
        chain_info_dict[chain_id]["res_id"] = new_res_ids

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


def add_charge_from_ccd_codes(*args, **kwargs) -> AtomArray:
    """Removed. Use :py:func:`atomworks.io.utils.ccd.add_annotations_from_ccd` instead."""
    raise DeprecationWarning(
        "add_charge_from_ccd_codes() has been deprecated!"
        "Use `add_annotations_from_ccd(atom_array, annotations=['charge'])` from "
        "`atomworks.io.utils.ccd` for an equivalent CCD-based charge lookup. "
        "If you want bond-aware charge inference (not just a CCD lookup), consider "
        "`prepare_atom_array(..., add_missing_atoms=True)` instead."
    )


def add_pn_unit_id_annotation(
    atom_array: AtomArray | AtomArrayStack,
    overwrite: bool = True,
    exclude_bond_types: set[struc.BondType] | None = None,
) -> AtomArray | AtomArrayStack:
    """Adds the polymer/non-polymer unit ID (pn_unit_id) annotation to the AtomArray.

    Two covalently bonded ligands are considered one PN unit, but a ligand bonded to a protein is considered two PN units.
    See the README glossary for more details on how we define `chains`, `pn_units`, and `molecules` within this codebase.

    Args:
        atom_array: The AtomArray to process.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.
        exclude_bond_types: Bond types to exclude when determining connectivity.

    Returns:
        The AtomArray including the ``pn_unit_id`` annotation.
    """
    # Check if annotation exists and skip if not overwriting
    if not overwrite and "pn_unit_id" in atom_array.get_annotation_categories():
        return atom_array

    # ...initialize the pn_unit_id to chain_id (we will later update for multi-chain non-polymer PN units)
    pn_unit_id_annotation = atom_array.chain_id.astype(object)

    # ...make the NetworkX graph for non-polymer chains
    non_polymer_atom_array = atom_array[~atom_array.is_polymer]
    connected_chains = get_connected_nodes(
        *get_coarse_graph_as_nodes_and_edges(non_polymer_atom_array, "chain_id", exclude_bond_types=exclude_bond_types)
    )

    for connected_chain in connected_chains:
        # ...set the same the pn_unit_id for each chain in the connected chain
        pn_unit_id = ",".join(sorted(connected_chain))
        for chain_id in connected_chain:
            pn_unit_id_annotation[atom_array.chain_id == chain_id] = pn_unit_id

    atom_array.set_annotation("pn_unit_id", pn_unit_id_annotation.astype(str))

    return atom_array


def add_pn_unit_iid_annotation(
    atom_array: AtomArray | AtomArrayStack, overwrite: bool = True
) -> AtomArray | AtomArrayStack:
    """Adds the polymer/non-polymer unit instance ID (pn_unit_iid) annotation to the AtomArray or AtomArrayStack.

    Optimized to avoid expensive subarray operations by using vectorized operations and boolean masks.
    For symmetric assemblies with many identical chains, this provides significant speedup.

    Args:
        atom_array: The AtomArray or AtomArrayStack to annotate.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.

    Returns:
        The AtomArray or AtomArrayStack with pn_unit_iid annotation added.
    """
    # Check if annotation exists and skip if not overwriting
    if not overwrite and "pn_unit_iid" in atom_array.get_annotation_categories():
        return atom_array

    # ...create an array that concatenates the pn_unit_id and transformation_id
    _temp_pn_unit_iid = sum_string_arrays(atom_array.pn_unit_id, "_", atom_array.transformation_id)
    _final_pn_unit_iid = np.full(atom_array.array_length(), fill_value="", dtype=object)

    # Use boolean masks to access first atom of each unit, then broadcast results
    unique_pn_unit_iids = np.unique(_temp_pn_unit_iid)

    # Iterate through unique pn_unit_iids
    # (We implicitly assume that a given pn_unit_id will have the same transformation_id across all atoms in the unit)
    for pn_unit_iid in unique_pn_unit_iids:
        mask = _temp_pn_unit_iid == pn_unit_iid

        # Find first atom index in this unit (all atoms in unit have same pn_unit_id and transformation_id)
        first_atom_idx = np.where(mask)[0][0]

        # ...get the transformation_id and pn_unit_id (which is the same for all atoms in the unit)
        transformation_id = atom_array.transformation_id[first_atom_idx]
        pn_unit_id = str(atom_array.pn_unit_id[first_atom_idx])

        # ...split apart the pn_unit_id by commas
        pn_unit_ids = pn_unit_id.split(",")

        # ...add the transformation_id to each pn_unit_id
        pn_unit_iids = [f"{unit_id}_{transformation_id}" for unit_id in pn_unit_ids]

        # ...join the instance-level identifiers back into a single string
        pn_unit_iid_formatted = ",".join(pn_unit_iids)

        # ...update the AtomArray with the instance-level identifier
        _final_pn_unit_iid[mask] = pn_unit_iid_formatted

    atom_array.set_annotation("pn_unit_iid", _final_pn_unit_iid.astype(str))

    return atom_array


def add_molecule_id_annotation(
    atom_array: AtomArray | AtomArrayStack,
    overwrite: bool = True,
    exclude_bond_types: set[struc.BondType] | None = None,
) -> AtomArray | AtomArrayStack:
    """Adds the molecule ID (molecule_id) annotation to the AtomArray.

    Args:
        atom_array: The AtomArray to process.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.
        exclude_bond_types: Bond types to exclude when determining connectivity.

    Returns:
        The AtomArray including the ``molecule_id`` annotation.
    """
    # Check if annotation exists and skip if not overwriting
    if not overwrite and "molecule_id" in atom_array.get_annotation_categories():
        return atom_array

    # Initialize annotation (will overwrite if exists)
    if "molecule_id" not in atom_array.get_annotation_categories():
        atom_array.add_annotation("molecule_id", dtype=np.int16)

    # ...make the NetworkX graph for all pn_units
    connected_pn_units = get_connected_nodes(
        *get_coarse_graph_as_nodes_and_edges(atom_array, "pn_unit_id", exclude_bond_types=exclude_bond_types)
    )

    # ...iterate through connected pn_units
    for idx, connected_pn_unit in enumerate(connected_pn_units):
        # ...set the same the molecule_id for each pn_unit in the connected pn_unit
        molecule_id = idx
        for pn_unit_id in connected_pn_unit:
            atom_array.molecule_id[atom_array.pn_unit_id == pn_unit_id] = molecule_id

    return atom_array


def add_molecule_iid_annotation(atom_array_stack: AtomArrayStack, overwrite: bool = True) -> AtomArrayStack:
    """Adds the molecule instance ID (molecule_iid) annotation to the AtomArrayStack.

    Args:
        atom_array_stack: The AtomArrayStack to annotate.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.

    Returns:
        The AtomArrayStack with molecule_iid annotation added.
    """
    # Check if annotation exists and skip if not overwriting
    if not overwrite and "molecule_iid" in atom_array_stack.get_annotation_categories():
        return atom_array_stack

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
    overwrite: bool = True,
    exclude_bond_types: set[struc.BondType] | None = None,
) -> tuple[AtomArray, dict[int, list[str]]]:
    """Annotates entities in an AtomArray at a given `id` level, based on the connectivity and annotations at the lower level.

    The intended use is, for example:
        - For the `molecule` level, `molecule_entities` are generated for each `molecule_id` based on the connectivty
            at the `pn_unit` level.
        - For the `pn_unit` level, `pn_unit_entities` are generated for each `pn_unit_id` based on the connectivty
            at the `chain` level.
        - For the `chain` level, `chain_entities` are generated for each `chain_id` based on the connectivty at the `residue`
            level.

    Args:
        atom_array: The AtomArray to process.
        level: The level at which to annotate entities (e.g., "chain", "pn_unit", "entity").
        lower_level_id: A list of annotations to consider for determining segment boundaries at a lower level.
            E.g. "pn_unit_id", "chain_id" or "res_id".
        lower_level_entity: The annotation to use for identifying entities at the lower level.
            E.g. "pn_unit_entity", "chain_entity" or "res_name".
        add_inter_level_bond_hash: Whether to add a hash of the inter-level bonds to the entity hash.
            For some cases, this may be necessary to distinguish entities (e.g., when determining molecule-level
            entities). In others (e.g., for polymers), this may be overkill.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.

    Returns:
        Tuple of (annotated atom_array, entity_dict).

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
    entity_annotation_name = f"{level}_entity"

    # Check if annotation exists and skip if not overwriting
    if not overwrite and entity_annotation_name in atom_array.get_annotation_categories():
        # Return empty dict since we didn't compute anything
        return atom_array, {}

    _next_available_entity_id = 0
    _hash_to_entity_id = {}

    ids_at_level = np.unique(atom_array.get_annotation(level + "_id"))

    # ... initialize annotations to fill
    entities_annotation = np.zeros(len(atom_array), dtype=np.int16)
    entities_info = defaultdict(list)

    for instance_id in np.unique(ids_at_level):
        is_instance = atom_array.get_annotation(level + "_id") == instance_id
        instance = atom_array[is_instance]

        # ... get connectivity and node annotations for the coarse graph at the lower level
        _, edges = get_coarse_graph_as_nodes_and_edges(instance, lower_level_id, exclude_bond_types=exclude_bond_types)
        instance_graph = nx.Graph()
        instance_graph.add_edges_from(edges)

        # ... set node attributes to lower level entities (vectorized)
        # Direct annotation indexing instead of creating AtomArray objects via segment_iter()
        segment_boundaries = annot_start_stop_idxs(instance, lower_level_id, add_exclusive_stop=True)
        entity_annotation = instance.get_annotation(lower_level_entity)
        node_attrs = {idx: entity_annotation[start] for idx, start in enumerate(segment_boundaries[:-1])}
        nx.set_node_attributes(instance_graph, node_attrs, "node")

        # ... create the graph hash
        hash = hash_graph(instance_graph, node_attr="node")

        # ... add the inter-level bond hash (only consider the first lower level id; since we hash at the atom-level, this simplication is valid)
        if add_inter_level_bond_hash:
            hash += generate_inter_level_bond_hash(
                atom_array=instance,
                lower_level_id=lower_level_id[0] if isinstance(lower_level_id, list) else lower_level_id,
                lower_level_entity=lower_level_entity,
                exclude_bond_types=exclude_bond_types,
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


def add_chain_iid_annotation(atom_array_stack: AtomArrayStack, overwrite: bool = True) -> AtomArrayStack:
    """Adds the chain instance ID (chain_iid) annotation to the AtomArrayStack.

    Args:
        atom_array_stack: The AtomArrayStack to annotate.
        overwrite: If ``True``, recompute and replace annotation if it exists.
            If ``False``, skip if annotation already exists. Defaults to ``True``.

    Returns:
        The AtomArrayStack with chain_iid annotation added.
    """
    # Check if annotation exists and skip if not overwriting
    if not overwrite and "chain_iid" in atom_array_stack.get_annotation_categories():
        return atom_array_stack

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
    """Adds chain, PN unit, and molecule IIDs to assembly AtomArrayStacks.

    This is a convenience wrapper around :py:func:`add_iid_annotations` that operates on
    a dictionary of assemblies.
    """
    for assembly_id, assembly in assemblies_dict.items():
        assemblies_dict[assembly_id] = add_iid_annotations(assembly, overwrite=True)
    return assemblies_dict


def add_iid_annotations(
    atom_array_or_stack: AtomArray | AtomArrayStack,
    overwrite: bool = True,
) -> AtomArray | AtomArrayStack:
    """Add instance ID annotations (chain_iid, pn_unit_iid, molecule_iid).

    Requires transformation_id annotation to exist in the atom array/stack.
    Analogous to :py:func:`add_id_and_entity_annotations` but for instance IDs.

    Instance IDs are created by concatenating base IDs with transformation_id:

    - chain_iid = chain_id + "_" + transformation_id
    - pn_unit_iid = pn_unit_id + "_" + transformation_id
    - molecule_iid = molecule_id + "_" + transformation_id

    Args:
        atom_array_or_stack: AtomArray or AtomArrayStack with transformation_id annotation.
        overwrite: If ``True``, recompute and replace annotations if they exist.
            If ``False``, skip individual annotations that already exist. Defaults to ``True``.

    Returns:
        Same type as input with iid annotations added.

    Raises:
        ValueError: If transformation_id annotation is not present.
    """
    # Check for transformation_id
    if "transformation_id" not in atom_array_or_stack.get_annotation_categories():
        raise ValueError(
            "transformation_id annotation required for instance IDs. "
            "Instance IDs are created by concatenating base IDs with transformation_id."
        )

    # Pass overwrite to each helper function (matches add_id_and_entity_annotations pattern)
    atom_array_or_stack = add_chain_iid_annotation(atom_array_or_stack, overwrite=overwrite)

    if "pn_unit_id" in atom_array_or_stack.get_annotation_categories():
        atom_array_or_stack = add_pn_unit_iid_annotation(atom_array_or_stack, overwrite=overwrite)

    if "molecule_id" in atom_array_or_stack.get_annotation_categories():
        atom_array_or_stack = add_molecule_iid_annotation(atom_array_or_stack, overwrite=overwrite)

    return atom_array_or_stack


def add_id_and_entity_annotations(
    atom_array: AtomArray | AtomArrayStack,
    overwrite: bool = True,
    exclude_bond_types: set[struc.BondType] | None = frozenset({struc.BondType.COORDINATION}),
) -> AtomArray | AtomArrayStack:
    """Adds all 6 ('chain', 'pn_unit', 'molecule') x ('id', 'entity') annotations to the AtomArray.

    Args:
        atom_array: The AtomArray or AtomArrayStack to annotate.
        overwrite: If ``True``, recompute and replace annotations if they exist.
            If ``False``, skip individual annotations that already exist. Defaults to ``True``.
        exclude_bond_types: Bond types to exclude when determining pn_unit / molecule
            connectivity and entity hashing. Defaults to ``{COORDINATION}`` so that
            coordination (dative) bonds do not merge metals into their ligand's
            molecule/entity — a metal coordinated to a ligand is its own molecule.
            Pass an empty set to count all bonds.

    Returns:
        The AtomArray or AtomArrayStack with added annotations.
    """
    # For stacks, compute on model[0] and copy annotations back
    is_stack = isinstance(atom_array, AtomArrayStack)
    if is_stack:
        model = atom_array[0]
    else:
        model = atom_array

    # Pass overwrite to each sub-function
    model = add_pn_unit_id_annotation(model, overwrite=overwrite, exclude_bond_types=exclude_bond_types)
    model = add_molecule_id_annotation(model, overwrite=overwrite, exclude_bond_types=exclude_bond_types)

    levels = ["chain", "pn_unit", "molecule"]
    lower_level_ids = ["res_id", "chain_id", "pn_unit_id"]
    lower_level_entities = ["res_name", "chain_entity", "pn_unit_entity"]
    inter_level_bond_hashes = [False, True, True]

    for level, lower_level_id, lower_level_entity, inter_level_bond_hash in zip(
        levels, lower_level_ids, lower_level_entities, inter_level_bond_hashes, strict=False
    ):
        model, _ = annotate_entities(
            atom_array=model,
            level=level,
            lower_level_id=lower_level_id,
            lower_level_entity=lower_level_entity,
            add_inter_level_bond_hash=inter_level_bond_hash,
            overwrite=overwrite,
            exclude_bond_types=exclude_bond_types,
        )

    # Copy computed annotations back to the stack
    if is_stack:
        _ID_ENTITY_ANNOTATIONS = [  # noqa: N806
            "pn_unit_id",
            "molecule_id",
            "chain_entity",
            "pn_unit_entity",
            "molecule_entity",
        ]
        for annot_name in _ID_ENTITY_ANNOTATIONS:
            if annot_name in model.get_annotation_categories():
                atom_array.set_annotation(annot_name, model.get_annotation(annot_name))
        return atom_array

    return model


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
