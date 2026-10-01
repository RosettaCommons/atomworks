"""Transforms operating on Biotite's CIFBlock and CIFCategory objects.

These transforms are used to extract information from the CIFBlock and return a dictionary containing processed information.
"""

import logging
import re
from contextlib import suppress
from datetime import datetime

import numpy as np
import pandas as pd
import toolz
from biotite.structure.io.pdbx import BinaryCIFBlock, BinaryCIFCategory, CIFBlock, CIFCategory

from atomworks.common import exists

logger = logging.getLogger("atomworks.io")


def category_to_df(
    cif_block_or_category: CIFBlock | BinaryCIFBlock | CIFCategory | BinaryCIFCategory, category: str | None = None
) -> pd.DataFrame | None:
    """Convert CIF component to pandas DataFrame.

    Accepts either ``(CIFBlock, category_name)`` or ``(CIFCategory)`` directly.
    Supports both text CIF and binary CIF (BinaryCIF) formats.

    Args:
        cif_block_or_category: :py:class:`~biotite.structure.io.pdbx.CIFBlock`,
            :py:class:`~biotite.structure.io.pdbx.BinaryCIFBlock`, or
            :py:class:`~biotite.structure.io.pdbx.CIFCategory` | :py:class:`~biotite.structure.io.pdbx.BinaryCIFCategory`
        category: Category name when passing CIFBlock/BinaryCIFBlock, omit when passing CIFCategory

    Returns:
        DataFrame containing the category data, or None if category doesn't exist (CIFBlock mode only)
    """
    # Check for both CIFBlock and BinaryCIFBlock
    if isinstance(cif_block_or_category, CIFBlock | BinaryCIFBlock):
        # Extract category from CIFBlock or BinaryCIFBlock
        cif_block = cif_block_or_category
        if category in cif_block:
            return pd.DataFrame(category_to_dict(cif_block, category))
        return None
    else:
        # Convert CIFCategory directly
        category_obj = cif_block_or_category
        return pd.DataFrame(category_to_dict(category_obj))


def category_to_dict(
    cif_block_or_category: CIFBlock | BinaryCIFBlock | CIFCategory | BinaryCIFCategory, category: str | None = None
) -> dict[str, np.ndarray]:
    """Convert CIF component to dict mapping column names to numpy arrays.

    Accepts either ``(CIFBlock, category_name)`` or ``(CIFCategory)`` directly.
    Supports both text CIF and binary CIF (BinaryCIF) formats.

    Args:
        cif_block_or_category: :py:class:`~biotite.structure.io.pdbx.CIFBlock`,
            :py:class:`~biotite.structure.io.pdbx.BinaryCIFBlock`, or
            :py:class:`~biotite.structure.io.pdbx.CIFCategory` | :py:class:`~biotite.structure.io.pdbx.BinaryCIFCategory`
        category: Category name when passing CIFBlock/BinaryCIFBlock, omit when passing CIFCategory

    Returns:
        Dict mapping column names to numpy arrays
    """
    # Check for both CIFBlock and BinaryCIFBlock (they don't share a common base class)
    if isinstance(cif_block_or_category, CIFBlock | BinaryCIFBlock):
        # Extract category from CIFBlock or BinaryCIFBlock
        cif_block = cif_block_or_category
        if exists(cif_block.get(category)):
            return toolz.valmap(lambda x: x.as_array(), dict(cif_block[category]))
        else:
            return {}
    else:
        # Convert CIFCategory directly
        category_obj = cif_block_or_category
        return {key: value.as_array() for key, value in category_obj.items()}


# Entity ID Concepts:
# - label_entity (GIVEN): From data source, stored in chain_info["label_entity"] (CIF files only)
# - chain_entity (DERIVED): Computed via graph hashing, stored in atom_array annotations


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
        - Chem comp type (elements of atomworks.constants.CHEM_COMP_TYPES)

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
