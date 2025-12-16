"""Transforms operating on Biotite's CIFBlock and CIFCategory objects.

These transforms are used to extract information from the CIFBlock and return a dictionary containing processed information.
"""

import logging
import os
import re
from contextlib import suppress
from datetime import datetime

import biotite.structure as struc
import numpy as np
import pandas as pd
import toolz
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock

from atomworks.common import exists
from atomworks.constants import CCD_MIRROR_PATH
from atomworks.enums import ChainType
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code

logger = logging.getLogger("atomworks.io")


def category_to_df(cif_block: CIFBlock, category: str) -> pd.DataFrame | None:
    """Convert a CIF block to a pandas DataFrame.

    Args:
        cif_block: The CIF block to convert.
        category: The category name to extract.

    Returns:
        DataFrame containing the category data, or None if category doesn't exist.
    """
    return pd.DataFrame(category_to_dict(cif_block, category)) if category in cif_block else None


def category_to_dict(cif_block: CIFBlock, category: str) -> dict[str, np.ndarray]:
    """Convert a CIF block to a dictionary.

    Args:
        cif_block: The CIF block to convert.
        category: The category name to extract.

    Returns:
        Dictionary containing the category data as numpy arrays.
    """
    if exists(cif_block.get(category)):
        return toolz.valmap(lambda x: x.as_array(), dict(cif_block[category]))
    else:
        return {}


def initialize_chain_info_from_category(cif_block: CIFBlock, atom_array: AtomArray) -> dict:
    """Extracts chain entity-level information from the CIF block.

    Requires the categories 'entity' and 'entity_poly' to be present in the CIF block.

    In particular, this function adds the following information to the chain_info_dict:
        - The RCSB entity ID for each chain (e.g., 1, 2, 3, etc.)
        - The chain type as an IntEnum (e.g., polypeptide(L), non-polymer, etc.)
        - The unprocessed one-letter entity canonical and non-canonical sequences.
        - A boolean flag indicating whether the chain is a polymer.
        - The EC numbers for the chain.

    Note that three-letter sequence information is added to the chain_info_dict in a later step.

    Args:
        cif_block (CIFBlock): Parsed CIF block.
        atom_array (AtomArray): Atom array containing the chain information.

    Returns:
        dict: Dictionary containing the sequence details of each chain.
    """
    assert "entity" in cif_block, "entity category not found in CIF block."
    assert "entity_poly" in cif_block, "entity_poly category not found in CIF block."

    # ... initialize
    chain_info_dict = {}

    # Step 1: Build a mapping of chain id to entity id from the `atom_site`
    chain_ids = atom_array.get_annotation("chain_id")
    rcsb_entities = atom_array.get_annotation("label_entity_id").astype(str)
    unique_chain_entity_map = dict(zip(chain_ids, rcsb_entities, strict=True))

    # Step 2: Load additional chain information
    rcsb_entity_df = category_to_df(cif_block, "entity")
    rcsb_entity_df["id"] = rcsb_entity_df["id"].astype(str)
    rcsb_entity_df.rename(columns={"type": "entity_type", "pdbx_ec": "ec_numbers"}, inplace=True)
    rcsb_entity_dict = rcsb_entity_df.set_index("id").to_dict(orient="index")

    # From `entity_poly`
    polymer_df = category_to_df(cif_block, "entity_poly")

    required_columns = ["entity_id", "type", "pdbx_strand_id"]
    optional_columns = ["pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can"]
    polymer_df = polymer_df[required_columns + [col for col in optional_columns if col in polymer_df.columns]]

    # Rename columns if they exist
    rename_map = {
        "type": "polymer_type",
        "pdbx_seq_one_letter_code": "non_canonical_sequence",
        "pdbx_seq_one_letter_code_can": "canonical_sequence",
    }
    polymer_df.rename(columns=rename_map, inplace=True)

    polymer_df["entity_id"] = polymer_df["entity_id"].astype(str)
    polymer_dict = polymer_df.set_index("entity_id").to_dict(orient="index")

    # Step 3: Merge additional information into the dictionary
    for chain_id, rscb_entity in unique_chain_entity_map.items():
        chain_info = rcsb_entity_dict.get(rscb_entity, {})
        polymer_info = polymer_dict.get(rscb_entity, {})
        if chain_info.get("ec_numbers", "?") != "?":
            ec_numbers = [ec.strip() for ec in chain_info.get("ec_numbers", "").split(",")]
        else:
            ec_numbers = []

        # First check if the chain is a polymer; if so, use the polymer type (which is more specific). Otherwise, use the entity type
        chain_type = ChainType.as_enum(polymer_info.get("polymer_type", chain_info.get("entity_type", "non-polymer")))

        chain_info_dict[chain_id] = {
            "rcsb_entity": rscb_entity,
            "chain_type": chain_type,
            "unprocessed_entity_canonical_sequence": polymer_info.get("canonical_sequence", "").replace("\n", ""),
            "unprocessed_entity_non_canonical_sequence": polymer_info.get("non_canonical_sequence", "").replace(
                "\n", ""
            ),
            "is_polymer": chain_type.is_polymer(),
            "ec_numbers": ec_numbers,
        }

    return chain_info_dict


def get_metadata_from_category(cif_block: CIFBlock, fallback_id: str | None = None) -> dict:
    """
    Extract metadata from the CIF block.
    If the `entry.id` field is not present in the CIF block, the `fallback_id` is used instead (e.g., the filename of the CIF).

    From RCSB CIF files, this function extracts:
        - ID (e.g., PDB ID)
        - Method (e.g., X-ray, NMR, etc.)
        - Deposition date (initial)
        - Release date (smallest revision date)
        - Resolution (e.g., 5.0, 3.0, etc.)

    For custom CIF files (e.g., distillation), this function extracts:
        - Extra metadata (all other categories)

    Arguments:
        cif_block (CIFBlock): The CIF block to extract metadata from.
        fallback_id (str): A fallback ID to use if the `entry.id` field is not present in the CIF block.
    """
    metadata = {}

    # Assert that if the "entry.id" field is NOT present, a fallback ID is provided
    assert (
        "entry" in cif_block and "id" in cif_block["entry"]
    ) or fallback_id is not None, "No ID found in CIF block or provided as fallback."

    # Set the ID field, using the fallback if necessary
    metadata["id"] = (
        cif_block["entry"]["id"].as_item().lower()
        if "entry" in cif_block and "id" in cif_block["entry"]
        else fallback_id.lower()
    )

    # +---------------- Look for standard RCSB metadata categories, default to None if not found ----------------+
    exptl = cif_block.get("exptl", None)
    status = cif_block.get("pdbx_database_status", None)
    refine = cif_block.get("refine", None)
    em_reconstruction = cif_block.get("em_3d_reconstruction", None)

    # Method
    metadata["method"] = ",".join(exptl["method"].as_array()).replace(" ", "_") if exptl and "method" in exptl else None

    # Initial deposition date and release date to the PDB
    metadata["deposition_date"] = (
        status["recvd_initial_deposition_date"].as_item()
        if status and "recvd_initial_deposition_date" in status
        else None
    )

    # The relevant release date is the smallest `pdbx_audit_revision_history.revision_date` entry
    if "pdbx_audit_revision_history" in cif_block and "revision_date" in cif_block["pdbx_audit_revision_history"]:
        revision_dates = cif_block["pdbx_audit_revision_history"]["revision_date"].as_array()
    else:
        revision_dates = None

    if revision_dates is not None:
        # Convert string dates to datetime objects
        date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in revision_dates]
        # Find the smallest date, convert back to string
        smallest_date = min(date_objects)
        metadata["release_date"] = smallest_date.strftime("%Y-%m-%d")
    else:
        metadata["release_date"] = None

    # Resolution
    metadata["resolution"] = None
    if refine:
        with suppress(KeyError, ValueError):
            metadata["resolution"] = float(refine["ls_d_res_high"].as_item())

    if metadata["resolution"] is None and em_reconstruction:
        with suppress(KeyError, ValueError):
            metadata["resolution"] = float(em_reconstruction["resolution"].as_item())

    # Serialize the catch-all metadata cateogry, if it exists (we can later load with CIFCategory.deserialize() at will)
    metadata["extra_metadata"] = cif_block["extra_metadata"].serialize() if "extra_metadata" in cif_block else None

    return metadata


def load_monomer_sequence_information_from_category(
    cif_block: CIFBlock, chain_info_dict: dict, atom_array: AtomArray, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH
) -> dict:
    """Load monomer sequence information into a chain_info_dict

    Uses:
        (a) The CIFCategory 'entity_poly_seq' as the sequence ground-truth for polymers.
        (b) The AtomArray as the ground-truth for non-polymers.

    We must rely on the CIFCategory 'entity_poly_seq' for polymers, as the AtomArray may not contain the full sequence information (e.g., unresolved residues)
    For non-polymers, there's no standard equivalent to 'entity_poly_seq', so we must use the AtomArray to get the sequence information.

    When loading both polymer and non-polymer sequences, we also filter out unknown or otherwise ignored residues.

    Args:
        cif_block (CIFBlock): The CIF block containing the monomer sequence information.
        chain_info_dict (dict): The dictionary where the monomer sequence information will be stored.
        atom_array (AtomArray): The atom array used to get the sequence for non-polymers.

    Returns:
        The updated chain_info_dict with monomer sequence information. Adds the following keys:
            - 'res_name': The CCD residue names for each chain.
            - 'res_id': The residue IDs for each chain (does not perform re-indexing)
            - 'processed_entity_non_canonical_sequence': The processed non-canonical sequence for each chain.
            - 'processed_entity_canonical_sequence': The processed canonical sequence for each chain.
            - 'has_sequence_heterogeneity': A boolean flag indicating whether the chain has
    """
    # Assert that entity_poly_seq category is present
    assert "entity_poly_seq" in cif_block, "entity_poly_seq category not found in CIF block."

    # Handle polymers by using `entity_poly_seq`
    polymer_seq_df = category_to_df(cif_block, "entity_poly_seq")
    polymer_seq_df = polymer_seq_df.loc[:, ["entity_id", "num", "mon_id"]].rename(
        columns={"entity_id": "rcsb_entity", "num": "res_id", "mon_id": "res_name"}
    )

    # Keep only the last occurrence of each residue
    duplicates = polymer_seq_df.duplicated(subset=["rcsb_entity", "res_id"], keep="last")
    entities_with_sequence_heterogeneity = polymer_seq_df[duplicates]["rcsb_entity"].unique()
    if duplicates.any():
        logger.info("Sequence heterogeneity detected, keeping only the last occurrence of each residue.")
        polymer_seq_df = polymer_seq_df[~duplicates]

    # Map rcsb_entity to lists of residue names and residue IDs
    polymer_seq_df["rcsb_entity"] = polymer_seq_df["rcsb_entity"].astype(int)
    polymer_entity_id_to_res_names_and_ids = {
        rcsb_entity: {"res_name": group["res_name"].tolist(), "res_id": group["res_id"].tolist()}
        for rcsb_entity, group in polymer_seq_df.groupby("rcsb_entity")
    }

    # Build up the chain_info_dict with the sequence information
    res_starts = get_residue_starts(atom_array)
    # ... get the unique chain IDs by order of first appearance in the AtomArray
    chain_ids = dict.fromkeys(struc.get_chains(atom_array))
    for chain_id in chain_ids:
        rcsb_entity = int(chain_info_dict[chain_id]["rcsb_entity"])

        if rcsb_entity in polymer_entity_id_to_res_names_and_ids:
            # For polymers, we use the stored entity residue list
            residue_names = polymer_entity_id_to_res_names_and_ids[rcsb_entity]["res_name"]
            chain_type = chain_info_dict[chain_id]["chain_type"]
            if residue_names:
                chain_info_dict[chain_id]["res_name"] = residue_names
                chain_info_dict[chain_id]["res_id"] = polymer_entity_id_to_res_names_and_ids[rcsb_entity]["res_id"]

                # Create the processed single-letter sequence representations
                processed_entity_non_canonical_sequence = [
                    get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=False)
                    for ccd_code in residue_names
                ]
                processed_entity_canonical_sequence = [
                    get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=True)
                    for ccd_code in residue_names
                ]
                chain_info_dict[chain_id]["processed_entity_non_canonical_sequence"] = "".join(
                    processed_entity_non_canonical_sequence
                )
                chain_info_dict[chain_id]["processed_entity_canonical_sequence"] = "".join(
                    processed_entity_canonical_sequence
                )
        else:
            # For non-polymers, we must re-compute every time, since entities are not guaranteed to have the same monomer sequence (e.g., for H2O chains)
            chain_res_starts = res_starts[atom_array.chain_id[res_starts] == chain_id]
            chain_info_dict[chain_id]["res_name"] = list(atom_array.res_name[chain_res_starts])
            chain_info_dict[chain_id]["res_id"] = list(atom_array.res_id[chain_res_starts])

        chain_info_dict[chain_id]["has_sequence_heterogeneity"] = (
            str(rcsb_entity) in entities_with_sequence_heterogeneity
        )

    # Remove entries from chain_info_dict that have no residues
    chain_info_dict = {
        chain_id: chain_info for chain_id, chain_info in chain_info_dict.items() if "res_name" in chain_info
    }

    return chain_info_dict


def get_ligand_of_interest_info(cif_block: CIFBlock) -> dict:
    """Extract ligand of interest information from a CIF block.

    Reference:
        `PDB101 Small Molecule Ligands Guide <https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/small-molecule-ligands>`_
    """
    # Extract binary flag for whether the ligand of interest is specified
    # NOTE: This is being used in addition to the below as it has slightly higher coverage across the PDB
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entry_details.has_ligand_of_interest.html
    has_loi = category_to_dict(cif_block, "pdbx_entry_details").get("has_ligand_of_interest", np.array(["N"]))[0] == "Y"

    # Extract which ligand is of interest if specified
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entity_instance_feature.feature_type.html
    entity_instance_feature = category_to_dict(cif_block, "pdbx_entity_instance_feature")
    comp_id_names = entity_instance_feature.get("comp_id", np.array([], dtype="<U3"))
    comp_id_mask = entity_instance_feature.get("feature_type", np.array([])) == "SUBJECT OF INVESTIGATION"

    return {
        "ligand_of_interest": list(comp_id_names[comp_id_mask]),
        "has_ligand_of_interest": has_loi | (len(comp_id_names) > 0),
    }


_PH_SINGLE_VALUE_PATTERN = r"p[hH]\s*([0-9]+(?:\.[0-9]+)?)"
_PH_RANGE_PATTERN = r"\s*(?:to|/|-)\s*"


def _parse_ph_range(ph_str: str) -> list[float] | None:
    """
    Extracts numeric pH values from a string range.

    Args:
        ph_str: The string containing pH information from the exptl_crystal_grow section of the CIF file.
                Examples of valid formats:
                    - "7.0-8.0"                    (5hs6)
                    - "pH8.0"                      (4oji)
    Returns:
        A list of floats [min_pH, max_pH], or None if parsing fails.
    """

    def _is_valid_ph_range(ph_vals: list[float]) -> bool:
        """Validates that all pH values are within the reasonable range (0-14)."""
        return all(0 <= ph_val <= 14 for ph_val in ph_vals)

    ph_str = str(ph_str).strip().lower()
    # CASE 1: Handle string with embedded "pH" and single number (e.g., "pH 7.5", "ph8.0")
    match = re.search(_PH_SINGLE_VALUE_PATTERN, ph_str)
    if match:
        ph_vals = [float(match.group(1))] * 2
        return ph_vals if _is_valid_ph_range(ph_vals) else None
    # CASE 2: Handle explicit numeric range (e.g., "6.5 to 7.5", "6.5/7.5", "6.5 - 7.5")
    parts = re.split(_PH_RANGE_PATTERN, ph_str)
    try:
        ph_vals = [float(p) for p in parts if p]
        return ph_vals if _is_valid_ph_range(ph_vals) else None
    except ValueError:
        return None


def extract_crystallization_details(crystal_dict: dict) -> dict[str, list[float] | None]:
    """
    Extracts crystallization details from the crystallization dictionary.

    Args:
        crystal_dict: Dictionary for the exptl_crystal_grow CIF category.

    Returns:
        A dictionary with crystallization details. Currently includes:
        - "pH": A list of two floats [min_pH, max_pH], or None if unavailable.
    """
    ph_col = crystal_dict.get("pH", [])
    ph_range_field = crystal_dict.get("pdbx_pH_range", [""])[0]
    details_field = crystal_dict.get("pdbx_details", [""])[0]

    try:
        if isinstance(ph_col, list | np.ndarray) and len(ph_col) > 1:
            # pH values are provided as a list of numbers (e.g., [5.5, 6.0, 6.5])
            ph_vals = [float(min(ph_col)), float(max(ph_col))]
        elif ph_col in [["?"], ["."]]:
            # pH field is missing or ambiguous
            if ph_range_field in ["?", "."] or "+" in str(ph_range_field):
                # pH range is also missing or invalid (e.g., contains "+", or is "?")
                # Try to extract from pdbx_details as fallback
                ph_vals = _parse_ph_range(details_field)
            else:
                # Try to parse pH range string (e.g., "pH8.0" or "6.5 - 7.5")
                ph_vals = _parse_ph_range(ph_range_field)
        else:
            # Assume a single pH value in either list or scalar form (e.g., ["7.5"] or 7.5)
            if isinstance(ph_col, list):
                ph_val = float(ph_col[0])
            elif isinstance(ph_col, np.ndarray):
                # Handle numpy arrays properly by extracting the first element
                ph_val = float(ph_col.flat[0])
            else:
                ph_val = float(ph_col)
            ph_vals = [ph_val, ph_val]

        # Consistent float formatting (or None)
        if ph_vals:
            ph_floats = [float(v) for v in ph_vals]
            return {"pH": [min(ph_floats), max(ph_floats)]}
        else:
            return {"pH": None}

    except Exception as e:
        logger.warning(f"Error parsing pH values: {e}")
        return {"pH": None}
