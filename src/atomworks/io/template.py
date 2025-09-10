import logging
import os
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, BondList

import atomworks.io.transforms.atom_array as ta
from atomworks.common import exists, immutable_lru_cache
from atomworks.constants import CCD_MIRROR_PATH, DO_NOT_MATCH_CCD
from atomworks.io.utils.bonds import (
    correct_bond_types_for_nucleophilic_additions,
    correct_formal_charges_for_specified_atoms,
    get_inferred_polymer_bonds,
    get_struct_conn_bonds,
)
from atomworks.io.utils.ccd import atom_array_from_ccd_code, check_ccd_codes_are_available
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array
from atomworks.io.utils.selection import get_annotation
from atomworks.io.utils.testing import has_ambiguous_annotation_set

logger = logging.getLogger(__file__)


@immutable_lru_cache(maxsize=200)
def get_empty_ccd_template(
    ccd_code: str,
    *,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    remove_hydrogens: bool = True,
    **res_wise_annotations: int | float | str,
) -> AtomArray:
    """
    Creates an empty template AtomArray from a Chemical Component Dictionary (CCD) entry with optional residue-wise
    annotations.

    Args:
        - ccd_code (str): The three-letter code of the chemical component to create a template for.
        - ccd_mirror_path (os.PathLike, optional): Path to the local CCD mirror directory. Defaults to CCD_MIRROR_PATH.
        - remove_hydrogens (bool, optional): Whether to remove hydrogen atoms from the template. Defaults to True.
        - **res_wise_annotations: Additional residue-wise annotations to add to the template. Values can be int, float,
            or str and will be broadcast to all atoms in the template.

    Returns:
        - AtomArray: An empty template structure with nan coordinates but with bonds and annotations from the CCD entry,
            plus any additional specified annotations.

    Example:
        >>> template = get_empty_ccd_template("ALA", chain_id="A", res_id=1, occupancy=1.0)
    """
    template_cc = atom_array_from_ccd_code(ccd_code, ccd_mirror_path, coords=None)

    if remove_hydrogens:
        template_cc = ta.remove_hydrogens(template_cc)

    # set default residue-wise annotations
    for annot, value in res_wise_annotations.items():
        if value is not None:
            template_cc.set_annotation(annot, np.full(len(template_cc), value))

    return template_cc


def match_residue_to_template(
    template: AtomArray,
    real: AtomArray,
    res_mask: np.ndarray | None = None,
    use_ccd_charges: bool = False,
) -> AtomArray:
    """
    Matches atoms from a real structure to a template structure, copying over coordinates and annotations while preserving
    the template's topology.

    The function attempts to match atoms first by standard atom names, then by alternative atom IDs if available and if
    they provide better matching. Coordinates and annotations from matched atoms in the real structure are copied to the
    template. Unmatched atoms in the real structure are dropped with a warning.

    Args:
        - template (AtomArray): Template structure containing the reference topology and complete set of atoms.
        - real (AtomArray): Real structure containing the atoms to be matched to the template.
        - res_mask (np.ndarray, optional): A mask of atoms in the real structure to match to the template. Defaults to
            None, which matches all atoms in the real structure.
        - use_ccd_charges (bool): Whether to keep template charges (True) or copy charges from real structure (False).

    Returns:
        - AtomArray: Template structure with coordinates and annotations copied from matched atoms in the real structure.

    Raises:
        - ValueError: If multiple atoms in the real structure have the same name.

    Notes:
        - Atoms in real structure not found in template are dropped (with warning)
        - If multiple template atoms match a real atom, only first match is used (with warning)
        - Records whether alternative atom IDs were used for matching in 'uses_alt_atom_id' annotation
    """
    if res_mask is None:
        res_mask = np.ones(real.array_length(), dtype=bool)

    # Get global indices of relevant entries from real for faster indexing
    gidx = np.where(res_mask)[0]

    # ... get information about the residue
    ccd_code = real.res_name[gidx[0]]
    annotations = set(real.get_annotation_categories())

    # ... fail if there are multiple atoms with the same name in the `real` array
    atom_names = real.atom_name[gidx]
    if len(np.unique(atom_names)) != len(gidx):
        raise ValueError(f"CCD {ccd_code}: Multiple atoms with the same name in \n{real[gidx]}")

    # ... determine whether to use the standard or alternative atom naming
    match_by = "atom_name"
    n_matches_std = np.sum(np.isin(atom_names, template.atom_name, assume_unique=True))
    if ("alt_atom_id" in template.get_annotation_categories()) and n_matches_std < len(gidx):
        n_matches_alt = np.sum(np.isin(atom_names, template.alt_atom_id, assume_unique=True))
        match_alt = n_matches_alt > n_matches_std
        match_by = "alt_atom_id" if match_alt else "atom_name"
        if match_alt:
            logger.warning(f"CCD {ccd_code}: Having to use alternative atom IDs for matching.")
    # ... and record what we used to match
    template.set_annotation("uses_alt_atom_id", [(match_by == "alt_atom_id")] * len(template))

    # ... compute matching indices
    _, template_idxs, local_match_idxs = np.intersect1d(
        template.get_annotation(match_by), atom_names, assume_unique=True, return_indices=True
    )

    # ... fill the annotations
    match_idxs = gidx[local_match_idxs]  # global indices of real atoms that matched
    template.coord[template_idxs] = real.coord[match_idxs]
    template.occupancy[template_idxs] = real.occupancy[match_idxs] if "occupancy" in annotations else 1.0
    if "ins_code" in annotations:
        template.ins_code[template_idxs] = real.ins_code[match_idxs]
    if "b_factor" in annotations:
        template.b_factor[template_idxs] = real.b_factor[match_idxs]
    if not use_ccd_charges:
        template.charge[template_idxs] = real.charge[match_idxs]

    # ... copy over general residue annotations
    template.chain_id = [real.chain_id[gidx[0]]] * len(template)
    template.res_id = [real.res_id[gidx[0]]] * len(template)
    template.ins_code = [real.ins_code[gidx[0]]] * len(template)

    # ... copy over chain_iid annotation, if present
    if "chain_iid" in real.get_annotation_categories():
        template.set_annotation("chain_iid", [real.chain_iid[gidx[0]]] * len(template))

    # ... return matched array
    return template


def _find_residue_mask_fast(
    residue_keys: np.ndarray,
    sorted_keys: np.ndarray,
    sort_idx: np.ndarray,
    chain_id: str,
    res_name: str,
    res_id: int,
) -> np.ndarray:
    """
    Efficient method of getting a residue mask from a sorted list of residue keys.

    Args:
        - residue_keys: Structured np array of residue keys to search through
        - sorted_keys: Sorted list of residue keys
        - sort_idx: Index of the sorted list (get from doing np.argsort(residue_keys))
        - chain_id: Chain ID of the residue
        - res_name: Residue name of the residue
        - res_id: Residue ID of the residue

    Returns:
        - mask: Boolean mask of the residue keys
    """
    key = np.array([(chain_id, res_name, res_id)], dtype=residue_keys.dtype)

    # Find start and end indices using binary search
    start_idx = np.searchsorted(sorted_keys, key)[0]
    end_idx = np.searchsorted(sorted_keys, key, side="right")[0]

    # Create mask
    mask = np.zeros(len(residue_keys), dtype=bool)
    if start_idx < end_idx:
        mask[sort_idx[start_idx:end_idx]] = True

    return mask


def build_template_atom_array(
    chain_info_dict: dict[str, dict[str, Any]],
    atom_array: AtomArray | None = None,
    remove_hydrogens: bool = True,
    use_ccd_charges: bool = True,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    custom_residues: dict[str, AtomArray] | None = None,
) -> AtomArray:
    """
    Builds a template AtomArray by matching residues to CCD templates and copying coordinates/annotations from an existing
    structure.

    For each residue in chain_info_dict, creates a template from the Chemical Component Dictionary (CCD) and either:
    1. Copies coordinates and annotations from matching atoms in atom_array, or
    2. Leaves coordinates as NaN if no matching atoms exist, or
    3. Copies atoms verbatim from atom_array / custom_residues dictionary if no CCD template exists (e.g., for UNL) or for CCD codes that
        we want to ignore for matching (e.g., for water molecules)

    In the case where an atom_array is provided, the function is robust to ambiguous (chain_id, res_id, res_name)
    combinations, provided that the chain_iid annotation is present and resolves the ambiguity.

    Args:
        chain_info_dict (dict): Dictionary mapping chain IDs to dicts containing residue information with keys:
            - 'res_id': List of residue IDs
            - 'res_name': List of residue names (CCD codes)
            - 'is_polymer': Boolean indicating if chain is polymeric
            - 'chain_type': Chain type enum value
        atom_array (AtomArray, optional): Structure containing coordinates and annotations to copy. Defaults to None.
        remove_hydrogens (bool, optional): Whether to remove hydrogens from CCD templates. Defaults to True.
        use_ccd_charges (bool, optional): Whether to use charges from CCD (True) or atom_array (False). Defaults to True.
        ccd_mirror_path (os.PathLike, optional): Path to local CCD mirror. Defaults to CCD_MIRROR_PATH.
        custom_residues (dict, optional): Dictionary mapping CCD codes to custom AtomArrays. Can be thought as "expanding"
            the CCD to include additional residues (e.g., during inference with custom NCAA). Defaults to None.

    Returns:
        AtomArray: Template structure with coordinates and annotations copied from atom_array where available.

    Raises:
        ValueError: If chains in atom_array don't match chains in chain_info_dict.
    """
    # ... check if the chain_to_sequence_map is consistent with the atom_array
    if exists(atom_array) and (not set(struc.get_chains(atom_array)) == set(chain_info_dict)):
        raise ValueError(
            "Mismatch between `atom_array` and `chain_to_sequence`! "
            f"Atom array contains chains {struc.get_chains(atom_array)} but chain_to_sequence "
            f"contains chains {chain_info_dict.keys()}."
        )

    # ... extract the relevant entries from the atom_array if it exists
    all_false = lambda n: np.zeros(n, dtype=bool)  # noqa: E731, convenience function
    chain_ids = atom_array.chain_id if exists(atom_array) else all_false(0)
    res_ids = atom_array.res_id if exists(atom_array) else all_false(0)
    res_names = atom_array.res_name if exists(atom_array) else all_false(0)

    # Determine if we have ambiguous residue annotations, e.g. if parsing an AtomArray with multiple transformations
    use_chain_iids = False
    if exists(atom_array):
        residue_start_indices = struc.get_residue_starts(atom_array)
        residue_start_atoms = atom_array[residue_start_indices]
        if has_ambiguous_annotation_set(residue_start_atoms, annotation_set=["chain_id", "res_id", "res_name"]):
            if "chain_iid" not in atom_array.get_annotation_categories():
                raise ValueError(
                    "Ambiguous residue annotations detected. This happens when there are residues that "
                    "have the same `(chain_id, res_id, res_name)` identifier. "
                    "This happens for example when you have a bio-assembly with multiple copies "
                    "of a chain that only differ by `transformation_id`.\n"
                    "You can fix this for example by re-naming the chains to be named uniquely. "
                    "For the purposes of this function, you can also add a unambiguous chain_iid annotation instead. "
                )
            elif has_ambiguous_annotation_set(residue_start_atoms, annotation_set=["chain_iid", "res_id", "res_name"]):
                raise ValueError(
                    "Ambiguous bond annotations detected. This happens when there are atoms that "
                    "have the same `(chain_id, res_id, res_name)` identifier. "
                    "This happens for example when you have a bio-assembly with multiple copies "
                    "of a chain that only differ by `transformation_id`.\n"
                    "In this case, falling back to the `chain_iid` annotation was insufficient to resolve the ambiguity."
                    "You can fix this for example by re-naming the chains to be named uniquely. "
                    "For the purposes of this function, you can also add a unambiguous chain_iid annotation instead. "
                )
            else:
                use_chain_iids = True
                chain_iids = atom_array.chain_iid
                chain_info_dict = initialize_chain_info_from_atom_array(atom_array, use_chain_iids=True)

    # ... extract the relevant entries from the chain_info_dict for readability
    chain_identifier_to_res_ids = {
        chain_identifier: chain_info_dict[chain_identifier]["res_id"] for chain_identifier in chain_info_dict
    }
    chain_identifier_to_res_names = {
        chain_identifier: chain_info_dict[chain_identifier]["res_name"] for chain_identifier in chain_info_dict
    }
    chain_identifier_to_is_polymer = {
        chain_identifier: chain_info_dict[chain_identifier]["is_polymer"] for chain_identifier in chain_info_dict
    }
    chain_identifier_to_types = {
        chain_identifier: chain_info_dict[chain_identifier]["chain_type"] for chain_identifier in chain_info_dict
    }
    annotations = set(atom_array.get_annotation_categories()) if exists(atom_array) else set()

    # ... create a list of atoms based on the reference CCD entries
    template_residues = []
    chain_identifiers = chain_iids if use_chain_iids else chain_ids

    # ... get the sorted list of residue keys. This will make the residue mask lookup much faster.
    residue_keys = np.array(
        list(zip(chain_identifiers, res_names, res_ids, strict=True)),
        dtype=np.dtype([("chain_id", "object"), ("res_name", "object"), ("res_id", "<i4")]),
    )
    sort_idx = np.argsort(residue_keys)
    sorted_keys = residue_keys[sort_idx]

    for chain_identifier in list(dict.fromkeys(chain_info_dict)):
        chain_res_ids = chain_identifier_to_res_ids[chain_identifier]
        chain_res_names = chain_identifier_to_res_names[chain_identifier]
        chain_is_polymer = chain_identifier_to_is_polymer[chain_identifier]
        chain_type = chain_identifier_to_types[chain_identifier].value  # chain_type is an IntEnum; we want the value

        assert len(chain_res_ids) == len(chain_res_names), "Length mismatch between chain_res_ids, chain_res_names!"

        for res_id, (res_id_original, ccd_code) in enumerate(zip(chain_res_ids, chain_res_names, strict=True), start=1):
            res_id_original = int(res_id_original)
            # ... and corresponding mask
            res_mask = _find_residue_mask_fast(
                residue_keys, sorted_keys, sort_idx, chain_identifier, ccd_code, res_id_original
            )

            # res_mask might all False, in which case we fall back to the chain_id in the chain_info_dict (if present)
            if res_mask.any():
                chain_id = atom_array.chain_id[res_mask][0] if exists(atom_array) else chain_identifier
            elif not use_chain_iids:
                chain_id = chain_identifier
            else:
                raise ValueError(
                    f"Could not infer chain_id, since chain_iids are being used but no atoms were found for residue "
                    f"{ccd_code} in chain {chain_identifier} with ID {res_id_original}."
                )

            # ... if we cannot get a template from the CCD (e.g., UNL), we check if the code was provided via the `custom_residues` argument,
            # or copy over the atoms from the atom_array verbatim
            if (ccd_code in DO_NOT_MATCH_CCD) or not check_ccd_codes_are_available(
                [ccd_code], ccd_mirror_path, mode="warn"
            ):
                if custom_residues and ccd_code in custom_residues:
                    # (Use the provided custom residue if it exists)
                    real = custom_residues[ccd_code]
                else:
                    if not res_mask.any():
                        # ... skip if we cannot find the residue in the reference atom_array
                        logger.warning(
                            f"No atoms found for residue {ccd_code} in chain {chain_identifier} with ID {res_id_original}!"
                        )
                        continue

                    # (Copy from the given AtomArray)
                    real = atom_array[res_mask]

                n_atoms = real.array_length()
                real.set_annotation("stereo", np.full(n_atoms, fill_value="N", dtype="<U1"))
                real.set_annotation("is_leaving_atom", all_false(n_atoms))
                real.set_annotation("is_backbone_atom", all_false(n_atoms))
                real.set_annotation("is_n_terminal_atom", all_false(n_atoms))
                real.set_annotation("is_c_terminal_atom", all_false(n_atoms))
                real.set_annotation("uses_alt_atom_id", all_false(n_atoms))
                real.set_annotation("chain_type", [chain_type] * n_atoms)
                real.set_annotation("chain_id", [chain_id] * n_atoms)
                real.set_annotation("charge", get_annotation(real, "charge", default=np.zeros(n_atoms)))
                real.set_annotation("is_polymer", [chain_is_polymer] * n_atoms)

                if "res_id" not in real.get_annotation_categories():
                    # ... if the res_id annotation does not exist, we create it
                    real.set_annotation("res_id", [res_id_original] * n_atoms)

                template_residues.append(real)
                continue

            # ... get empty template (no occupation, nan coordinates)
            tmpl = get_empty_ccd_template(
                ccd_code,
                ccd_mirror_path=ccd_mirror_path,
                remove_hydrogens=remove_hydrogens,
                # ... add required residue-wise annotations
                chain_id=chain_id,
                occupancy=0.0,
                # ... add custom residue-wise annotations if they exist
                is_polymer=chain_is_polymer,
                b_factor=np.nan if "b_factor" in annotations else None,
                chain_type=chain_type,
            )
            # ... to make caching efficient, add the res_id annotation separately,
            #     since this will differ between residues of the same chain
            tmpl.res_id = np.full(len(tmpl), res_id_original if chain_is_polymer else res_id)

            # ... copy over the annotations & coordinates from the atom_array if the residue exists
            if res_mask.any():
                tmpl = match_residue_to_template(
                    template=tmpl, real=atom_array, res_mask=res_mask, use_ccd_charges=use_ccd_charges
                )

            template_residues.append(tmpl)

    # ... concatenate all template residues into a single AtomArray
    template_array = struc.concatenate(template_residues)

    return template_array


def add_inter_residue_bonds(
    atom_array: AtomArray,
    struct_conn_dict: dict,
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    fix_bond_types: bool = True,
    fix_formal_charges: bool = True,
) -> AtomArray:
    """
    Adds inter-residue bonds to an AtomArray and correctly handles leaving groups.

    This function performs several steps:
    1. Infers and adds polymer bonds between residues
    2. Adds additional inter-residue bonds from struct_conn records
    3. Removes leaving atoms from bond formation
    4. Fixes formal charges on atoms involved in inter-residue bonds

    Args:
        atom_array (AtomArray): Input structure to which bonds will be added
        struct_conn_dict (dict, optional): Dictionary containing structural connectivity information. Defaults to {}.
        add_bond_types_from_struct_conn (list[str], optional): Types of bonds to add from struct_conn. Defaults to
            ["covale"].
        fix_formal_charges (bool, optional): Whether to fix formal charges on atoms involved in inter-residue bonds.
        fix_bond_types (bool, optional): Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.

    Returns:
        AtomArray: Output structure with inter-residue bonds.
    """

    # ... infer inter-residue polymer bonds
    polymer_bonds, polymer_bond_leaving_atom_idxs = get_inferred_polymer_bonds(atom_array)

    # ... create any remaining inter-residue bonds that
    #     are specified in struct_conn
    struct_conn_bonds, struct_conn_leaving_atom_idxs = get_struct_conn_bonds(
        atom_array,
        struct_conn_dict=struct_conn_dict,
        add_bond_types=add_bond_types_from_struct_conn,
    )

    # ... merge all inter-residue bonds
    inter_bonds = BondList(
        atom_count=atom_array.array_length(),
        bonds=np.concatenate((polymer_bonds, struct_conn_bonds)),
    )

    # ... and add them to the atom array
    atom_array.bonds = atom_array.bonds.merge(inter_bonds)

    # ... and record which atoms make inter-residue bonds
    atoms_with_inter_bonds = np.unique(inter_bonds.as_array()[:, :2])
    makes_inter_bond = np.zeros(atom_array.array_length(), dtype=bool)
    makes_inter_bond[atoms_with_inter_bonds] = True

    # ... merge all leaving group indices
    is_leaving = np.zeros(len(atom_array), dtype=bool)
    is_leaving[np.concatenate((polymer_bond_leaving_atom_idxs, struct_conn_leaving_atom_idxs), axis=0)] = True

    # ... and remove them from the atom array
    atom_array = atom_array[~is_leaving]
    makes_inter_bond = makes_inter_bond[~is_leaving]

    # ... fix bond types of newly bonded atoms, where needed
    # (We must fix bonds before fixing formal charges, since the bond degree is used to infer the formal charge)
    if fix_bond_types and np.any(makes_inter_bond):
        atom_array = correct_bond_types_for_nucleophilic_additions(atom_array, to_update=makes_inter_bond)

    # ... fix charges of newly bonded atoms, where needed
    if fix_formal_charges:
        if hasattr(atom_array, "charge"):
            atom_array = correct_formal_charges_for_specified_atoms(atom_array, to_update=makes_inter_bond)
        else:
            logger.warning("Cannot fix formal charges: AtomArray has no 'charge' annotation.")

    return atom_array


def add_missing_atoms(
    atom_array: AtomArray,
    chain_info_dict: dict[str, dict[str, Any]],
    struct_conn_dict: dict = {},
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    remove_hydrogens: bool = True,
    use_ccd_charges: bool = True,
    fix_formal_charges: bool = True,
    fix_bond_types: bool = True,
) -> AtomArray:
    """
    Adds missing atoms to an AtomArray by matching residues to CCD templates and handling inter-residue bonds.

    This function performs several steps:
    1. Matches residues to CCD templates to add missing atoms and intra-residue bonds
    2. Infers and adds polymer bonds between residues
    3. Adds additional inter-residue bonds from struct_conn records
    4. Removes leaving atoms from bond formation
    5. Fixes formal charges on atoms involved in inter-residue bonds

    Args:
        atom_array (AtomArray): Input structure containing atoms to be completed.
        chain_info_dict (dict): Dictionary mapping chain IDs to dicts containing 'res_id', 'res_name', 'is_polymer', and
            'chain_type' info.
        struct_conn_dict (dict, optional): Dictionary containing structural connectivity information. Defaults to {}.
        add_bond_types_from_struct_conn (list[str], optional): Types of bonds to add from struct_conn. Defaults to
            ["covale"].
        remove_hydrogens (bool, optional): Whether to remove hydrogen atoms from templates. Defaults to True.
        use_ccd_charges (bool, optional): Whether to use charges from CCD or input structure. Defaults to True.
        fix_formal_charges (bool, optional): Whether to fix formal charges on atoms involved in inter-residue bonds.
        fix_bond_types (bool, optional): Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.

    Returns:
        AtomArray: Completed structure with missing atoms added and proper bonding.

    Raises:
        ValueError: If chain_info_dict is inconsistent with atom_array chains.
    """
    # ... match all residues to a CCD template
    #     (unless no CCD template esits, in which case we copy over)
    #     this also creates the intra-residue bonds from the CCD
    atoms = build_template_atom_array(
        chain_info_dict=chain_info_dict,
        atom_array=atom_array,
        use_ccd_charges=use_ccd_charges,
        remove_hydrogens=False
        if fix_formal_charges
        else remove_hydrogens,  # we keep hydrogens here, to allow fixing formal charges
    )

    # Add inter-residue bonds and remove leaving groups
    atoms = add_inter_residue_bonds(
        atom_array=atoms,
        struct_conn_dict=struct_conn_dict,
        add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
        fix_formal_charges=fix_formal_charges,
        fix_bond_types=fix_bond_types,
    )

    # ... remove hydrogens
    if remove_hydrogens:
        atoms = ta.remove_hydrogens(atoms)

    return atoms
