"""
Utility functions for the detection, and creation, of bonds in a structure.
"""

__all__ = [
    "generate_inter_level_bond_hash",
    "get_coarse_graph_as_nodes_and_edges",
    "get_connected_nodes",
    "get_inferred_polymer_bonds",
    "get_struct_conn_bonds",
    "hash_atom_array",
    "hash_graph",
]

import hashlib
import logging
from typing import Any

import biotite.structure as struc
import networkx as nx
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.io.pdbx.convert import (
    PDBX_BOND_TYPE_TO_TYPE_ID,
    _filter_bonds,
    _filter_canonical_links,
    _get_struct_conn_col_name,
)

from atomworks.common import sum_string_arrays, to_hashable
from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    CHEM_TYPE_POLYMERIZATION_ATOMS,
    DEFAULT_VALENCE,
    HYDROGEN_LIKE_SYMBOLS,
    NA_LIKE_CHEM_TYPES,
    STRUCT_CONN_BOND_ORDER_TO_INT,
    STRUCT_CONN_BOND_TYPES,
)
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.utils.ccd import get_chem_comp_leaving_atom_names, get_chem_comp_type
from atomworks.io.utils.selection import get_annotation, get_residue_starts
from atomworks.io.utils.testing import has_ambiguous_annotation_set

logger = logging.getLogger("atomworks.io")


def _get_leaving_atom_idxs_for(atom_name: str, res_name: str, atom_names: np.ndarray, offset: int = 0) -> np.ndarray:
    """
    Get the indices of the leaving atoms for a given residue and atom.
    """
    leaving_atoms = get_chem_comp_leaving_atom_names(res_name).get(atom_name, ())
    return offset + np.where(np.isin(atom_names, leaving_atoms))[0]


def get_inferred_polymer_bonds(atom_array: AtomArray) -> tuple[list[tuple[int, int, struc.BondType]], np.ndarray]:
    """
    Infers and returns polymer bonds between consecutive residues in an atom array based on chemical component types
    and chain types.

    The function identifies bonds by looking at consecutive residues within the same chain and determining the
    appropriate bonding atoms based on either the chain type (as a fallback) or more detailed chemical component
    types. It also tracks leaving atoms that are displaced during bond formation. Leaving groups are inferred from
    the CCD entries for the chemical components. If a CCD code is missing from your local CCD mirror,
    leaving groups will not be inferred.

    Args:
        - atom_array (AtomArray): The atom array containing the structure information. Must include annotations for
            chain_id, res_id, res_name, and atom_name. Optionally includes chain_type annotation.

    Returns:
        - polymer_bonds (np.array[[int, int, struc.BondType]]): List of tuples containing (atom1_idx, atom2_idx,
            bond_type) for each inferred polymer bond.
        - leaving_atom_idxs (np.ndarray): Array of atom indices that represent leaving groups displaced during bond
            formation.

    Example:
        >>> # Create an atom array with two consecutive peptide residues
        >>> atom_array = AtomArray(length=10)
        >>> atom_array.chain_id = ["A"] * 10
        >>> atom_array.res_id = [1] * 5 + [2] * 5
        >>> atom_array.res_name = ["ALA"] * 5 + ["GLY"] * 5
        >>> atom_array.atom_name = ["N", "CA", "C", "OXT", "CB"] + ["N", "CA", "C", "O", "H2"]
        >>> # Get the polymer bonds
        >>> bonds, leaving = get_inferred_polymer_bonds(atom_array)
        >>> print(bonds)  # Shows C-N peptide bond between residues
        [(2, 5, <BondType.SINGLE>)]  # C of ALA to N of GLY
        >>> print(leaving)  # Shows leaving OXT from C and H2 from N (other hydrogen atom names not shown for simplicity)
        [array([3]), array([9])]
    """
    # ... initialize return values
    bonds: list[tuple[int, int, struc.BondType]] = []
    leaving: list[np.ndarray] = []

    # ... get annotations we need to work with
    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    chain_types = get_annotation(atom_array, "chain_type", default=None)
    is_polymer = get_annotation(atom_array, "is_polymer", default=np.zeros(atom_array.array_length(), dtype=bool))

    # ... get iterators over the residues
    residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    this_res_starts = residue_starts[:-2]
    next_res_starts = residue_starts[1:-1]
    next_res_stops = residue_starts[2:]

    # ... loop over the residues and add the bonds
    for this_res_start, next_res_start, next_res_stop in zip(
        this_res_starts, next_res_starts, next_res_stops, strict=False
    ):
        # ... skip if residues are not on the same chain
        if chain_ids[this_res_start] != chain_ids[next_res_start]:
            continue

        # ... and skip if residues don't have consecutive res_id's
        #     (NOTE: same res_id is allowed, if ins_code is different)
        if res_ids[next_res_start] - res_ids[this_res_start] > 1:
            continue

        # ... get fallback default bonding atoms based on chain type
        bonding_atoms = None
        if chain_types is not None:
            chain_type = ChainType.as_enum(chain_types[this_res_start])
            bonding_atoms = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(chain_type, None)

        # ... get (more detailed) bonding atoms based on chem-comp types
        this_link = get_chem_comp_type(res_names[this_res_start], mode="warn")
        next_link = get_chem_comp_type(res_names[next_res_start], mode="warn")

        # ... decide which bonds to form:
        both_aa = (this_link in AA_LIKE_CHEM_TYPES) and (next_link in AA_LIKE_CHEM_TYPES)
        both_na = (this_link in NA_LIKE_CHEM_TYPES) and (next_link in NA_LIKE_CHEM_TYPES)
        if (this_link in CHEM_TYPE_POLYMERIZATION_ATOMS) and (both_aa or both_na):
            bonding_atoms = CHEM_TYPE_POLYMERIZATION_ATOMS[this_link]

        # ... add the bonds if we have bonding atoms
        if bonding_atoms is not None:
            # bonding_atoms: tuple[str, str] = (atom1_name, atom2_name)
            atom1_name, atom2_name = bonding_atoms

            # ... get the atoms names within the current residues
            this_res_atom_names = atom_names[this_res_start:next_res_start]
            next_res_atom_names = atom_names[next_res_start:next_res_stop]

            # ... find the indices of the bonding atoms based on the atoms names
            atom1_idx = np.where(this_res_atom_names == atom1_name)[0]
            atom2_idx = np.where(next_res_atom_names == atom2_name)[0]

            if len(atom1_idx) == 0 or len(atom2_idx) == 0:
                # ... bonding atoms are not found in the adjacent residues
                # ... -> skip this bond
                logger.info(
                    f"Bonding atoms {atom1_name} and {atom2_name} not found "
                    f"in the adjacent residues {this_res_start} and {next_res_start}!"
                )
                continue

            # ... add the bond
            bonds.append(
                (
                    this_res_start + atom1_idx[0],  # ... add global atom idx offset
                    next_res_start + atom2_idx[0],  # ... add global atom idx offset
                    struc.BondType.SINGLE,
                )
            )

            # ... compute the leaving atoms
            leaving_this_res = _get_leaving_atom_idxs_for(
                atom_name=atom1_name,
                res_name=res_names[this_res_start],
                atom_names=this_res_atom_names,
                offset=this_res_start,
            )
            leaving_next_res = _get_leaving_atom_idxs_for(
                atom_name=atom2_name,
                res_name=res_names[next_res_start],
                atom_names=next_res_atom_names,
                offset=next_res_start,
            )
            leaving.append(leaving_this_res) if len(leaving_this_res) > 0 else None
            leaving.append(leaving_next_res) if len(leaving_next_res) > 0 else None

            # ... optionally add `is_polymer` annotation to the atom array
            is_polymer[this_res_start:next_res_stop] = True

    if "is_polymer" not in atom_array.get_annotation_categories():
        # ... if polymer annotation was not present before, we set it here based on the inferred bonds
        atom_array.set_annotation("is_polymer", is_polymer)

    return np.array(bonds).reshape(-1, 3), np.concatenate(leaving) if len(leaving) > 0 else np.array([], dtype=int)


def get_struct_conn_dict_from_atom_array(
    atom_array: AtomArray,
) -> dict[str, np.ndarray]:
    """Returns a struct_conn dictionary corresponding to a given AtomArray.

    This contains the keys used in `get_struct_conn_bonds`.
    NOTE: These AtomArray-derived struct_conn_dicts will never contain disulfide or hydrogen bonds,
    as Biotite does not distinguish these in the BondList. Possible types are "covale" and "metalc".

    Args:
        atom_array (AtomArray): The atom array to get the struct_conn dictionary from.

    Returns:
        dict[str, np.ndarray]: The struct_conn dictionary.
    """

    struct_conn_dict = {}

    for res_array in struc.residue_iter(atom_array):
        if len(np.unique(res_array.atom_name)) != len(res_array.atom_name):
            raise ValueError(
                "Duplicate atom names detected in the same residue -- cannot infer struct_conn. "
                "This may happen when a non-polymer is loaded from a CIF file without using `atomworks.io.parser.parse`. "
            )

    # Keep only inter-residue bonds
    bond_array = _filter_bonds(atom_array, "inter")
    if len(bond_array) == 0:
        return struct_conn_dict

    # Filter out 'standard' links, i.e. backbone bonds between adjacent canonical
    # nucleotide/amino acid residues
    bond_array = bond_array[~_filter_canonical_links(atom_array, bond_array)]
    if len(bond_array) == 0:
        return struct_conn_dict

    use_iids = False  # By default, we use chain_ids to determine bonds
    has_chain_iids = "chain_iid" in atom_array.get_annotation_categories()

    # Determine whether we need to fall back to using chain_iids
    if has_ambiguous_annotation_set(atom_array):
        if not has_chain_iids:
            raise ValueError(
                "Ambiguous bond annotations detected. This happens when there are atoms that "
                "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
                "This happens for example when you have a bio-assembly with multiple copies "
                "of a chain that only differ by `transformation_id`.\n"
                "You can fix this for example by re-naming the chains to be named uniquely. "
                "For the purposes of this function, you can also add a unambiguous chain_iid annotation instead. "
            )
        elif has_ambiguous_annotation_set(
            atom_array, annotation_set=["chain_iid", "res_id", "res_name", "atom_name", "ins_code"]
        ):
            raise ValueError(
                "Ambiguous bond annotations detected. This happens when there are atoms that "
                "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
                "This happens for example when you have a bio-assembly with multiple copies "
                "of a chain that only differ by `transformation_id`.\n"
                "In this case, falling back to the `chain_iid` annotation was insufficient to resolve the ambiguity."
                "You can fix this for example by re-naming the chains to be named uniquely. "
                "For the purposes of this function, you can also add a unambiguous chain_iid annotation instead. "
            )
        else:
            use_iids = True

    # Add the bond type information
    struct_conn_dict["conn_type_id"] = np.array([PDBX_BOND_TYPE_TO_TYPE_ID[btype] for btype in bond_array[:, 2]])

    label_asym_id_field = "chain_iid" if use_iids else "chain_id"
    cif_field_to_annot = {
        "label_asym_id": label_asym_id_field,
        "label_comp_id": "res_name",
        "label_seq_id": "res_id",
        "label_atom_id": "atom_name",
        "pdbx_PDB_ins_code": "ins_code",
    }

    for col_name, annot_name in cif_field_to_annot.items():
        annot = atom_array.get_annotation(annot_name)
        # ...for each bond partner
        for i in range(2):
            atom_indices = bond_array[:, i]
            struct_conn_dict[_get_struct_conn_col_name(col_name, i + 1)] = annot[atom_indices].astype(str)

    return struct_conn_dict


def get_struct_conn_bonds(
    atom_array: AtomArray,
    struct_conn_dict: dict[str, np.ndarray],
    add_bond_types: list[str] = ["covale"],
    raise_on_failure: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds bonds from the 'struct_conn' category of a CIF block to an atom array. Only covalent bonds are considered.

    Args:
        atom_array (AtomArray): The atom array used to get atom indices.
        struct_conn_dict (dict[str, np.ndarray]): The struct_conn category of a CIF block as a dictionary.
            E.g. (Only mandatory fields, as defined by the RCSB, are shown)
            ```
                {
                'conn_type_id': array(['disulf', ...]),
                'ptnr1_label_asym_id': array(['A', ...]),
                'ptnr1_label_comp_id': array(['CYS', ...]),
                'ptnr1_label_seq_id': array(['6', ...]),
                'ptnr1_label_atom_id': array(['SG', ...]),
                'ptnr1_symmetry': array(['1_555', ...]),
                'ptnr2_label_asym_id': array(['A', ...]),
                'ptnr2_label_comp_id': array(['CYS', ...]),
                'ptnr2_label_seq_id': array(['127', ...]),
                'ptnr2_label_atom_id': array(['SG', ...]),
                'ptnr2_symmetry': array(['1_555', ...]),
                }
            ```
            However, in this function, we only require the following fields:
                - conn_type_id (e.g., "covale")
                - ptnr1_label_asym_id (chain_id or chain_iid, e.g., "A" or "A_1")
                - ptnr1_label_comp_id (residue name in the CCD, e.g., "CYS")
                - ptnr1_label_seq_id (residue ID, e.g., "6")
                - ptnr1_label_atom_id (atom name, e.g., "SG")
                - ptnr2_label_asym_id
                - ptnr2_label_comp_id
                - ptnr2_label_seq_id
                - ptnr2_label_atom_id

        add_bond_types (list[str]): A list of bond types that should be added. Valid bond types
            are: ["covale", "disulf", "metalc", "hydrog"]. Defaults to ["covale"], which is
            the use-case in structure-prediction, where we would a-priori know covalent bonds
            (except for disulfides).
        raise_on_failure(bool): If True, raise an error if specified bonds cannot be made (e.g.,
            if the atoms are missing). Defaults to False.

        NOTE: While chain_iid annotations are allowed, a given bond is expected to contain only one annotation type,
            i.e. both chain_id or both chain_iid

    Returns:
        bonds (np.array[[int, int, struc.BondType]]): A List of bonds to be added to the atom array.
        leaving (np.ndarray): An array of indices of atoms that are leaving groups for bookkeeping.

    Reference:
        `struct_conn.conn_type_id <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.conn_type_id.html>`_
    """

    def match_or_wildcard(array: np.ndarray, value: str) -> np.ndarray:
        if value == "*":
            return np.ones_like(array, dtype=bool)
        return array == value

    # ... validate input
    invalid_bond_types = set(add_bond_types) - STRUCT_CONN_BOND_TYPES
    if len(invalid_bond_types) > 0:
        raise ValueError(
            f"Invalid bond type(s) provided: {invalid_bond_types}! Valid bond types are: {STRUCT_CONN_BOND_TYPES}"
        )
    if len(struct_conn_dict) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=int)

    # ... convert struct_conn_dict to a DataFrame
    struct_conn_df = pd.DataFrame(struct_conn_dict)
    struct_conn_df = struct_conn_df[struct_conn_df["conn_type_id"].isin(add_bond_types)]
    if struct_conn_df.empty:
        # ... skip if no bonds to add
        return np.empty((0, 3), dtype=int), np.empty((0,), dtype=int)
    logger.debug(f"Attempting to add {len(struct_conn_df)} bonds from `struct_conn`")

    # ... extract relevant annotations
    chain_ids = atom_array.chain_id
    res_names = atom_array.res_name
    res_ids = atom_array.res_id
    ins_codes = atom_array.ins_code
    atom_names = atom_array.atom_name
    is_polymer = atom_array.is_polymer
    global_atom_idx = np.arange(atom_array.array_length())
    alt_atom_ids = get_annotation(atom_array, "alt_atom_id", default=atom_names)
    uses_alt_atom_id = get_annotation(atom_array, "uses_alt_atom_id", default=np.zeros(len(atom_array), dtype=bool))

    all_res_names = np.append(np.unique(res_names), "*")
    all_chain_ids = np.unique(chain_ids)
    polymer_chain_ids = np.unique(chain_ids[is_polymer])

    # Get iid-level annotations if present
    if "chain_iid" in atom_array.get_annotation_categories():
        chain_iids = atom_array.chain_iid
        all_chain_iids = np.unique(chain_iids)
        polymer_chain_iids = np.unique(chain_iids[is_polymer])
    else:
        chain_iids = None
        all_chain_iids = None
        polymer_chain_iids = None

    # ... initialize return values
    bonds: list[tuple[int, int, struc.BondType]] = []
    leaving: list[np.ndarray] = []

    for _, row in struct_conn_df.iterrows():
        res_name1 = str(row["ptnr1_label_comp_id"])
        res_name2 = str(row["ptnr2_label_comp_id"])
        if (res_name1 not in all_res_names) or (res_name2 not in all_res_names):
            # ... skip if the residues were removed from the structure
            if raise_on_failure:
                raise ValueError(f"Residue {res_name1} or {res_name2} not found in the atom array!")
            continue

        chain_id1 = row["ptnr1_label_asym_id"]
        chain_id2 = row["ptnr2_label_asym_id"]

        # Default to using id-level identifiers
        relevant_chain_identifiers = chain_ids
        relevant_polymer_chain_identifiers = polymer_chain_ids

        # If iid-level identifiers are present, use these as a fallback
        if (chain_id1 not in all_chain_ids) or (chain_id2 not in all_chain_ids):
            if (chain_iids is not None) and (chain_id1 in all_chain_iids) and (chain_id2 in all_chain_iids):
                relevant_chain_identifiers = chain_iids
                relevant_polymer_chain_identifiers = polymer_chain_iids
            else:
                # ... skip, but warn if the chains are not present in the structure
                logger.info(
                    f"Found covalent bond involving chains {chain_id1} and {chain_id2}, but at least one "
                    "chain was removed during cleaning. This is likely because the chain is made up of a "
                    "residue that is not in the local CCD. This should automatically be resolved once you "
                    "update your CCD, unless you are working with an outdated structure file."
                )
                if raise_on_failure:
                    raise ValueError(f"Chain {chain_id1} or {chain_id2} not found in the atom array!")
                continue

        # For non-polymers, we use the auth_seq_id if available and valid (i.e., not "." or "?"); otherwise we use the label_seq_id
        # (Required to avoid ambiguity, since if using `label` only we may have multiple residue within a
        # chain with the same label_seq_id and the same res_name; see: 6MUB)

        res_id1 = int(
            row["ptnr1_label_seq_id"]
            if ((chain_id1 in relevant_polymer_chain_identifiers) or ("ptnr1_auth_seq_id" not in row))
            and row["ptnr1_label_seq_id"] != "."
            else row["ptnr1_auth_seq_id"]
        )
        res_id2 = int(
            row["ptnr2_label_seq_id"]
            if ((chain_id2 in relevant_polymer_chain_identifiers) or ("ptnr2_auth_seq_id" not in row))
            and row["ptnr2_label_seq_id"] != "."
            else row["ptnr2_auth_seq_id"]
        )

        ins_code1 = row.get("pdbx_ptnr1_PDB_ins_code", "")
        ins_code2 = row.get("pdbx_ptnr2_PDB_ins_code", "")
        ins_code1 = "" if ins_code1 in (".", "?") else ins_code1
        ins_code2 = "" if ins_code2 in (".", "?") else ins_code2

        # ... get masks for the residues to which atoms 1 & 2 belong
        in_res1 = (
            (relevant_chain_identifiers == chain_id1)
            & (res_ids == res_id1)
            & match_or_wildcard(res_names, res_name1)
            & (ins_codes == ins_code1)
        )
        in_res2 = (
            (relevant_chain_identifiers == chain_id2)
            & (res_ids == res_id2)
            & match_or_wildcard(res_names, res_name2)
            & (ins_codes == ins_code2)
        )

        if (not in_res1.any()) or (not in_res2.any()):
            logger.info(
                f"Residue {chain_id1}/{res_id1}/{res_name1} or {chain_id2}/{res_id2}/{res_name2} "
                "not found in the atom array!"
            )
            if raise_on_failure:
                raise ValueError(
                    f"Residue {chain_id1}/{res_id1}/{res_name1} or {chain_id2}/{res_id2}/{res_name2} "
                    "not found in the atom array!"
                )
            continue

        in_res1_start = global_atom_idx[in_res1][0]
        in_res2_start = global_atom_idx[in_res2][0]

        # Ensure that the we picked the correct residue (to handle sequence heterogeneity; see PDB ID `3nez` for an example)
        #  (short circuit eval to avoid indexing errors in cases where we don't have one of the residues due to seq. heterogeneity
        #   - e.g. 3nez)
        if (
            (in_res1.sum() == 0)
            or (in_res2.sum() == 0)
            or (res_name1 != res_names[in_res1_start] if res_name1 != "*" else False)
            or (res_name2 != res_names[in_res2_start] if res_name2 != "*" else False)
        ):
            logger.info(
                f"Covalent bond involving residues {chain_id1}/{res_id1}/{res_name1} and "
                f"{chain_id2}/{res_id2}/{res_name2} was found in `struct_conn`, but the "
                f"residues are not present in the atom array. This is likely due to "
                f"resolved sequence heterogeneity which removed one of the residues."
            )
            if raise_on_failure:
                raise ValueError(
                    f"Residue {chain_id1}/{res_id1}/{res_name1} or {chain_id2}/{res_id2}/{res_name2} "
                    "not found in the atom array!"
                )
            continue

        # If all residues are present, we can proceed with identifying the global indices of the
        # atoms in the bond and add the bond
        # ... get the indices of the atoms and append to the list
        atom_name1 = row["ptnr1_label_atom_id"]
        atom_name2 = row["ptnr2_label_atom_id"]

        # ... skip, but warn if the atoms (either the standard are not present in the atom array
        all_names = np.concatenate((atom_names, alt_atom_ids))
        if (atom_name1 not in all_names) or (atom_name2 not in all_names):
            logger.info(
                f"Covalent bond involving atoms {atom_name1} and {atom_name2} was found in `struct_conn`, but the "
                "atoms are not present in the residue's AtomArray!"
            )
            continue

        if uses_alt_atom_id[in_res1_start]:
            atom1_local_idx = np.where(alt_atom_ids[in_res1] == atom_name1)[0][0]
        else:
            atom1_local_idx = np.where(atom_names[in_res1] == atom_name1)[0][0]

        if uses_alt_atom_id[in_res2_start]:
            atom2_local_idx = np.where(alt_atom_ids[in_res2] == atom_name2)[0][0]
        else:
            atom2_local_idx = np.where(atom_names[in_res2] == atom_name2)[0][0]

        # ... convert local atom indices to global indices
        atom1_global_idx = in_res1_start + atom1_local_idx
        atom2_global_idx = in_res2_start + atom2_local_idx

        # ... add the bond
        # Metal coordination bonds don't have a `pdbx_value_order`, so these are handled separately
        if row["conn_type_id"] == "metalc":
            bonds.append([atom1_global_idx, atom2_global_idx, struc.BondType.COORDINATION])
        else:
            bond_order = STRUCT_CONN_BOND_ORDER_TO_INT.get(row.get("pdbx_value_order"), 1)
            bonds.append([atom1_global_idx, atom2_global_idx, struc.BondType(bond_order)])

        # ... and identify the leaving atoms
        leaving_res1 = _get_leaving_atom_idxs_for(
            atom_name=atom_names[atom1_global_idx],
            res_name=res_name1,
            atom_names=atom_names[in_res1],
            offset=in_res1_start,
        )
        leaving_res2 = _get_leaving_atom_idxs_for(
            atom_name=atom_names[atom2_global_idx],
            res_name=res_name2,
            atom_names=atom_names[in_res2],
            offset=in_res2_start,
        )
        leaving.append(leaving_res1) if len(leaving_res1) > 0 else None
        leaving.append(leaving_res2) if len(leaving_res2) > 0 else None

    return np.array(bonds).reshape(-1, 3), np.concatenate(leaving) if len(leaving) > 0 else np.array([], dtype=int)


def get_coarse_graph_as_nodes_and_edges(
    atom_array: AtomArray, annotations: str | tuple[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the coarse-grained nodes and edges at the given annotation level based on the atom array's bond connectivity.

    Args:
        - atom_array (AtomArray): The atom array containing atomic information and bonds.
        - annotations (str | tuple[str]): A single annotation or a tuple of annotations to be used for node
            identification.

    Returns:
        - nodes (np.ndarray): An array of unique nodes, each represented by a combination of annotations.
        - edges (np.ndarray): An array of edges, where each edge is a tuple of node indices representing a bond
            between two nodes.

    Example:
        >>> atom_array = cached_parse("5ocm")["atom_array"]
        >>> nodes, edges = get_coarse_graph(atom_array, ["chain_id", "transformation_id"])
        >>> print(nodes)
        array([('A', '1'), ('F', '1'), ('G', '1'), ('H', '1'), ('I', '1'),
               ('W', '1'), ('X', '1'), ('Y', '1')],
              dtype=[('chain_id', '<U4'), ('transformation_id', '<U1')])
        >>> print(edges)
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [5, 5],
               [6, 6]])
    """
    annotations = [annotations] if isinstance(annotations, str) else annotations

    atom1, atom2, _ = atom_array.bonds.as_array().T

    if len(annotations) > 1:
        _annots = np.zeros(
            len(atom_array), dtype=[(annot, atom_array.get_annotation(annot).dtype) for annot in annotations]
        )
        for annot in annotations:
            _annots[annot] = atom_array.get_annotation(annot)  # [n_atoms, n_annotations]
    else:
        _annots = atom_array.get_annotation(annotations[0])  # [n_atoms]

    annot1 = _annots[atom1]  # [n_bonds, n_annotations]
    annot2 = _annots[atom2]  # [n_bonds, n_annotations]

    nodes = np.unique(_annots, axis=0)  # [n_nodes, n_annotations]
    self_edges = np.vstack([nodes, nodes]).T  # [n_nodes, 2]
    edges = np.unique(np.vstack([self_edges, np.vstack([annot1, annot2]).T]), axis=0)  # [n_edges, 2]

    # Map nodes to integers
    node_to_idx = {to_hashable(node): i for i, node in enumerate(nodes)}
    if len(edges) > 0:
        edges = np.apply_along_axis(
            lambda x: (node_to_idx[to_hashable(x[0])], node_to_idx[to_hashable(x[1])]), 1, edges
        )

    return nodes, edges


def get_connected_nodes(nodes: np.ndarray, edges: np.ndarray) -> list[list[Any]]:
    """Returns connected nodes as a mapped list given corresponding arrays of nodes and edges.

    Example:
        >>> nodes = np.array([("A", "1"), ("B", "1"), ("C", "1"), ("D", "1")])
        >>> edges = np.array([[0, 1], [0, 2], [1, 2]])
        >>> connected_nodes = get_connected_nodes(nodes, edges)
        >>> print(connected_nodes)
        [[("A", "1"), ("B", "1"), ("C", "1")], [("D", "1")]]
    """
    # ...make the graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # ...return lists of connected chains
    return [[nodes[x] for x in component] for component in nx.connected_components(graph)]


def hash_graph(
    graph: nx.Graph,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    iterations: int = 3,
    digest_size: int = 16,
) -> str:
    """
    Computes a hash for a given graph using the Weisfeiler-Lehman (WL) graph hashing algorithm and additionally
    adds a node and edge attribute hash, if specified, to deal with common edge cases where WL fails (e.g.
    disconnected graphs).

    Args:
        - graph (networkx.Graph): The input graph to be hashed.
        - node_attr (str | None): The node attribute to be used for hashing. If None, node attributes are ignored.
        - edge_attr (str | None): The edge attribute to be used for hashing. If None, edge attributes are ignored.
        - iterations (int): The number of iterations for the WL algorithm. Default is 3.
        - digest_size (int): The size of the hash digest for WL. Default is 16.

    Returns:
        - str: The computed hash of the graph.

    Example:
        >>> import networkx as nx
        >>> G = nx.gnm_random_graph(10, 15)
        >>> hash_graph(G)
        '504894f49dd84b17c391b163af69624b'
    """
    # ... compute WL-hash
    hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        graph, node_attr=node_attr, edge_attr=edge_attr, iterations=iterations, digest_size=digest_size
    )

    if node_attr is not None:
        # ... add number of unique nodes to hash
        hash += f"_{len(graph.nodes)}"
        # ... add number of unique node attributes with counts to hash
        node_attr_dict = nx.get_node_attributes(graph, node_attr)
        hash += "_" + ",".join(
            [
                f"{elt}:{count}"
                for elt, count in zip(*np.unique(list(node_attr_dict.values()), return_counts=True), strict=False)
            ]
        )
    if edge_attr is not None:
        # ... add number of unique edges to hash
        hash += f"_{len(graph.edges)}"
    return hash


def _atom_array_to_networkx_graph(
    atom_array: AtomArray,
    annotations: tuple[str] = ("element", "atom_name"),
    bond_order: bool = True,
    cast_aromatic_bonds_to_same_type: bool = True,
) -> nx.Graph:
    """Convert an AtomArray to a NetworkX graph."""
    # ... create the bond graph
    bonds = atom_array.bonds.as_array()

    # ... create the bond graph for the atom array, adding all nodes first to ensure correct indexing
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(range(len(atom_array)))
    bond_list = []

    # ... add edges from bond list
    if len(bonds) > 0:
        bond_list = [tuple(bond) for bond in bonds[:, :2]]
        bond_graph.add_edges_from(bond_list)

    # ... annotate the bond graph with bond order
    if bond_order:
        bond_type = bonds[:, -1]
        if cast_aromatic_bonds_to_same_type:
            bond_type[bond_type == struc.BondType.AROMATIC_SINGLE] = 0
            bond_type[bond_type == struc.BondType.AROMATIC_DOUBLE] = 0
            bond_type[bond_type == struc.BondType.AROMATIC_TRIPLE] = 0
        nx.set_edge_attributes(
            bond_graph, {tuple(bond): type for bond, type in zip(bond_list, bond_type, strict=False)}, "bond_type"
        )

    # ... annotate the bond graph with the desired node annotations
    if annotations:
        node_data = sum_string_arrays(*[atom_array.get_annotation(annot).astype(str) for annot in annotations])
        # ... map the node annotations to the bond graph
        nx.set_node_attributes(bond_graph, {n: node_data[n] for n in bond_graph.nodes()}, "node_data")

    return bond_graph


def hash_atom_array(
    atom_array: AtomArray,
    annotations: tuple[str] = ("element", "atom_name"),
    bond_order: bool = True,
    cast_aromatic_bonds_to_same_type: bool = False,
    use_md5: bool = False,
    md5_length: int | None = None,
) -> str:
    """
    Computes a hash for an AtomArray based on the bond connectivity and the selected node annotations.

    Args:
        atom_array (AtomArray): The array of atoms to hash
        annotations (tuple[str]): The node annotations to include in the hash
        bond_order (bool): Whether to include bond order in the hash
        cast_aromatic_bonds_to_same_type (bool): Whether to treat all aromatic bonds as the same type
        use_md5 (bool): Whether to use MD5 hashing on the output
        md5_length (int | None): If using MD5, the number of characters to keep from the hash. If None, returns full hash.

    Returns:
        str: The computed hash
    """
    # ... create the bond graph
    bond_graph = _atom_array_to_networkx_graph(
        atom_array,
        annotations=annotations,
        bond_order=bond_order,
        cast_aromatic_bonds_to_same_type=cast_aromatic_bonds_to_same_type,
    )

    hash_str = hash_graph(
        bond_graph, node_attr="node_data" if annotations else None, edge_attr="bond_type" if bond_order else None
    )

    if use_md5:
        hash_str = hashlib.md5(hash_str.encode()).hexdigest()
        if md5_length is not None:
            hash_str = hash_str[:md5_length]

    return hash_str


def generate_inter_level_bond_hash(
    atom_array: AtomArray, lower_level_id: str, lower_level_entity: str | None = None
) -> str:
    """
    Generates a hash string representing the inter-level bonds within an AtomArray.
    When computing entities IDs, we must consider inter-level bonds at the atom- and residue-level to avoid ambiguity.

    Args:
        atom_array (AtomArray): The array of atoms containing bond and annotation information.
        lower_level_id (str): The level which to find, and hash, the inter-level bonds. For example, when computing molecule entities, we'd consider the inter-PN Unit bonds.
        lower_level_entity (str } None): An additional entity annotation to use when computing the hash. Optional; if None, then only residue ID, residue name, and atom name are used.

    Returns:
        str: A hash string representing the inter-level bonds.
    """
    # ...find the inter-level bonds
    bond_a = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 0]]
    bond_b = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 1]]
    inter_level_bonds = atom_array.bonds.as_array()[bond_a != bond_b]

    if inter_level_bonds.size:
        # ...loop over the bonds and create a (sorted) list of tuples with the relevant information
        bond_tuples = []
        for atom_idx in range(inter_level_bonds.shape[0]):
            atom_a = atom_array[inter_level_bonds[atom_idx, 0]]
            atom_b = atom_array[inter_level_bonds[atom_idx, 1]]
            bond_tuples.append(
                tuple(
                    sorted(
                        [
                            (
                                getattr(atom_a, lower_level_entity) if lower_level_entity else None,
                                atom_a.res_id,
                                atom_a.res_name,
                                atom_a.atom_name,
                            ),
                            (
                                getattr(atom_b, lower_level_entity) if lower_level_entity else None,
                                atom_b.res_id,
                                atom_b.res_name,
                                atom_b.atom_name,
                            ),
                        ]
                    )
                )
            )

        # ...sort the list of tuples, and hash
        return str(hash(tuple(sorted(bond_tuples))))
    else:
        return ""


def spoof_struct_conn_dict_from_string(bonds: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Spoof a struct_conn_dict from a list of bond strings.

    NOTE: For SMILES, atoms are named with their element type and the order that they
    appear in the SMILES string. For example, the first carbon atom in a SMILES string
    would be named "C1", the second "C2", and so on.

    NOTE: We only support covalent bonds.

    TODO: Use AtomSelection to parse the bond strings

    Args:
        bonds (list[tuple[str, str]]): A list of bond strings.
            Each bond string should be in the format:
            "CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME, CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME"

            In PyMol, you can hover over an atom to display the relevant information.
            For example, clicking a CA atom in PyMol prints in the console:
            ```
            >>>  You clicked /6wtf/A/A/ALA`294/CA
            ```
            We could then copy this information to specify an atom, "A/ALA/294/CA"

    Returns:
        dict[str, list[str]]: A dictionary in struct_conn format.

    Example:
        ```
        >>> bonds = [
        ...     ("A/THR/4/CG", "D/UNL/1/C5"),
        ...     ("A/CYS/5/SG", "A/CYS/137/SG")
        ... ]
        >>> struct_conn_dict = spoof_struct_conn_dict_from_string(bonds)
        >>> print(struct_conn_dict)
        {
            'conn_type_id': ['covale', 'covale'],
            'ptnr1_label_asym_id': ['A', 'A'],
            'ptnr1_label_comp_id': ['THR', 'CYS'],
            'ptnr1_label_seq_id': ['4', '5'],
            'ptnr1_label_atom_id': ['CG', 'SG'],
            'ptnr2_label_asym_id': ['D', 'A'],
            'ptnr2_label_comp_id': ['UNL', 'CYS'],
            'ptnr2_label_seq_id': ['1', '137'],
            'ptnr2_label_atom_id': ['C5', 'SG'],
        }
        ```

    """
    struct_conn_dict = {
        "conn_type_id": [],
        "ptnr1_label_asym_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr2_label_asym_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr2_label_atom_id": [],
    }

    for bond in bonds:
        try:
            # Split the bond string into two parts
            ptnr1, ptnr2 = bond

            # Parse the first partner
            ptnr1_chain_id, ptnr1_res_name, ptnr1_res_id, ptnr1_atom_name = ptnr1.split("/")
            struct_conn_dict["ptnr1_label_asym_id"].append(ptnr1_chain_id)
            struct_conn_dict["ptnr1_label_comp_id"].append(ptnr1_res_name)
            struct_conn_dict["ptnr1_label_seq_id"].append(ptnr1_res_id)
            struct_conn_dict["ptnr1_label_atom_id"].append(ptnr1_atom_name)

            # Parse the second partner
            ptnr2_chain_id, ptnr2_res_name, ptnr2_res_id, ptnr2_atom_name = ptnr2.split("/")
            struct_conn_dict["ptnr2_label_asym_id"].append(ptnr2_chain_id)
            struct_conn_dict["ptnr2_label_comp_id"].append(ptnr2_res_name)
            struct_conn_dict["ptnr2_label_seq_id"].append(ptnr2_res_id)
            struct_conn_dict["ptnr2_label_atom_id"].append(ptnr2_atom_name)

            # Assuming all bonds are covalent for simplicity; adjust as needed
            struct_conn_dict["conn_type_id"].append("covale")

        except ValueError as e:
            raise ValueError(f"Error parsing bond string '{bond}': {e}") from None

    return struct_conn_dict


def _get_bond_degree_per_atom(atom_array: struc.AtomArray) -> np.ndarray:
    """
    Returns the total degree (= sum of bond orders) for each atom.
    """
    # Count both ends of each edge
    edge_list = atom_array.bonds._bonds[:, :2]
    weights = atom_array.bonds._bonds[:, -1].copy()

    # ... remove aromaticity from the weights:
    weights[weights == struc.BondType.AROMATIC_SINGLE] = 1
    weights[weights == struc.BondType.AROMATIC_DOUBLE] = 2
    weights[weights == struc.BondType.AROMATIC_TRIPLE] = 3

    degree = np.bincount(edge_list.ravel(), weights=np.repeat(weights, 2))

    # ... pad in case of unbonded atoms
    if len(degree) <= atom_array.array_length():
        degree = np.pad(degree, (0, atom_array.array_length() - len(degree)))

    return degree


def correct_formal_charges_for_specified_atoms(atom_array: struc.AtomArray, to_update: np.ndarray) -> struc.AtomArray:
    """
    Fix formal charges for atoms in an AtomArray after forming bonds between CCD components.

    Args:
        atom_array (AtomArray): The AtomArray to fix.
        to_update (np.ndarray): A boolean mask of atoms whose formal charges should be fixed.
            These are normally the atoms for which bonds were modified.

    Returns:
        AtomArray: The AtomArray with fixed formal charges.
    """
    # ... check that the AtomArray has hydrogens
    if not np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS).any():
        logger.warning("Hydrogens not given. Cannot fix formal charges.")
        return atom_array

    # ... get valences (masked for elements with no default valence)
    _invalid = -10
    default_valence = np.array([DEFAULT_VALENCE.get(elt, _invalid) for elt in atom_array.element[to_update]])

    # ... compute total number of bonds per atom
    degree = _get_bond_degree_per_atom(atom_array)[to_update]

    # ... compute formal charge
    formal_charge = degree - default_valence

    # ... update the relevant entries
    valid = default_valence != _invalid

    # ... convert local indices to global indices
    global_idxs = np.arange(atom_array.array_length())[to_update]
    atom_array.charge[global_idxs[valid]] = formal_charge[valid]
    return atom_array


def correct_bond_types_for_nucleophilic_additions(
    atom_array: struc.AtomArray, to_update: np.ndarray
) -> struc.AtomArray:
    """
    Account for nucleophilic additions that result in carbons that violate the octet rule.

    In some cases (see: 1TQH), there is no leaving group specified, since the bond is formed by a nucleophilic addition to a carbonyl carbon.
    In this case, we should convert the C=O double bond to a C-O single bond.

    Args:
        atom_array (AtomArray): The AtomArray to fix.
        to_update (np.ndarray): A boolean mask of atoms that are candidates for correction.

    Returns:
        AtomArray: The AtomArray with fixed bond types.
    """
    updated_carbon_mask = (atom_array.element == "C") & to_update

    if not updated_carbon_mask.any():
        # (Early exit)
        return atom_array

    invalid_carbon_mask = (_get_bond_degree_per_atom(atom_array) > 4) & updated_carbon_mask

    bonds_arr = atom_array.bonds.as_array()
    for c_idx in np.where(invalid_carbon_mask)[0]:
        mask = (bonds_arr[:, 0] == c_idx) | (bonds_arr[:, 1] == c_idx)

        # If any of the bonds are to a hyrogen, we skip
        # (Handling hydrogens requires inferring leaving atoms, which is out-of-scope for this function)
        if np.any(np.isin(atom_array.element[bonds_arr[mask, 0]], HYDROGEN_LIKE_SYMBOLS)) or np.any(
            np.isin(atom_array.element[bonds_arr[mask, 1]], HYDROGEN_LIKE_SYMBOLS)
        ):
            continue

        # Check if any of the bonds are double bonds to an oxygen
        for bond_idx in np.where(mask)[0]:
            atom1, atom2, bond_type = bonds_arr[bond_idx]
            other_idx = atom2 if atom1 == c_idx else atom1

            if atom_array.element[other_idx] == "O" and bond_type == struc.BondType.DOUBLE:
                # Set the bond order to single and log a warning
                atom_array.bonds.remove_bond(atom1, atom2)
                atom_array.bonds.add_bond(atom1, atom2, struc.BondType.SINGLE)
                logger.warning(
                    f"Corrected C=O double bond to single bond between atom {c_idx} (C) and {other_idx} (O) due to nucleophilic addition (degree > 4). "
                    f"chain_id: {atom_array.chain_id[c_idx]}, res_name: {atom_array.res_name[c_idx]}, res_id: {atom_array.res_id[c_idx]}, atom_name of invalid carbon: {atom_array.atom_name[c_idx]}, atom_name of oxygen: {atom_array.atom_name[other_idx]}"
                )
                break

    return atom_array
