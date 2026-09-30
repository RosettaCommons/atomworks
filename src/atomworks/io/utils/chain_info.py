"""Helpers to extract chain-level information from AtomArray and enrich with CIF data."""

__all__ = ["build_chain_info"]

import logging

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.common import exists
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array import subset_to_first_transformation
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import convert_to_one_letter_sequences, infer_chain_type_from_three_letter

logger = logging.getLogger("atomworks.io")


def update_sequences_from_res_names(chain_data: dict) -> None:
    """Update one-letter sequences in chain_data based on res_name list.

    Modifies ``chain_data`` in-place to add/update:
      - ``processed_entity_non_canonical_sequence``
      - ``processed_entity_canonical_sequence``

    For polymer chains, sequences are computed from 3-letter ``res_name`` list.
    For non-polymer chains, sequences are 'X' repeated for each residue.

    Args:
      chain_data: Chain info dict with ``res_name``, ``chain_type``, and ``is_polymer`` fields.
    """
    chain_type = chain_data["chain_type"]
    res_names = chain_data["res_name"]

    if chain_type.is_polymer():
        non_canon, canon = convert_to_one_letter_sequences(res_names, chain_type)
        chain_data["processed_entity_non_canonical_sequence"] = non_canon
        chain_data["processed_entity_canonical_sequence"] = canon
    else:
        # Non-polymers get one 'X' per residue as sequence representation
        chain_data["processed_entity_non_canonical_sequence"] = "X" * len(res_names)
        chain_data["processed_entity_canonical_sequence"] = "X" * len(res_names)


def build_chain_info(
    atom_array: AtomArray | AtomArrayStack,
    *,
    entity: dict[str, np.ndarray] | None = None,
    entity_poly: dict[str, np.ndarray] | None = None,
    entity_poly_seq: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Build chain information dictionary from available inputs.

    Flow:
    (1) Extract baseline chain info from atom_array (inferring chain_type if not annotated)
    (2) If available, enrich with entity/entity_poly metadata (e.g., overwriting chain_type, adding label_entity, EC numbers, unprocessed sequences)
    (3) If available, enrich polymer sequences with poly_seq (e.g., adding unresolved N-terminal residues, detecting sequence heterogeneity)

    Args:
        atom_array: Atom array containing chain information.
        entity: Entity metadata from :py:func:`~atomworks.io.transforms.categories.category_to_dict`.
        entity_poly: Polymer entity metadata from :py:func:`~atomworks.io.transforms.categories.category_to_dict`.
        entity_poly_seq: Polymer sequence data from :py:func:`~atomworks.io.transforms.categories.category_to_dict`.

    Returns:
        Dict keyed by chain ID with fields: ``res_name``, ``res_id``, ``chain_type``,
        ``is_polymer``, ``processed_entity_{canonical,non_canonical}_sequence``,
        optionally ``label_entity``, ``ec_numbers``, ``unprocessed_entity_*``, ``has_sequence_heterogeneity``.
    """
    # --- Setup ---
    # Handle stack vs single model
    is_stack = isinstance(atom_array, AtomArrayStack)
    atom_array = atom_array[0] if is_stack else atom_array

    # Subset to first transformation, if applicable
    atom_array = subset_to_first_transformation(atom_array)

    # --- (1) Baseline from AtomArray ---
    chain_info = _extract_baseline_from_atom_array(atom_array)

    # --- (2) Enrich with Entity Info ---
    if exists(entity) or exists(entity_poly):
        chain_info = _enrich_with_entity_info(chain_info, entity, entity_poly)

    # --- (3) Enrich with Poly Seq ---
    if entity_poly_seq is not None:
        chain_info = _enrich_with_poly_seq(chain_info, entity_poly_seq)

    # --- (4) Compute sequences from final res_name lists ---
    for info in chain_info.values():
        update_sequences_from_res_names(info)

    # --- Cleanup ---
    current_chains = set(np.unique(atom_array.chain_id))
    chain_info = {k: v for k, v in chain_info.items() if k in current_chains and "res_name" in v}

    return chain_info


def _extract_baseline_from_atom_array(atom_array: AtomArray) -> dict[str, dict]:
    """Extract baseline chain info from atom_array."""
    assert "chain_id" in atom_array.get_annotation_categories(), "chain_id annotation not found in atom array"

    # Initialize chain_info dict
    chain_info = {}

    # Get residue-level annotations by subsetting to residue starts
    _res_starts = get_residue_starts(atom_array)
    chain_identifiers = atom_array.chain_id[_res_starts]
    res_ids = atom_array.res_id[_res_starts]
    res_names = atom_array.res_name[_res_starts]
    hetero = atom_array.hetero[_res_starts]

    # Get unique chain identifiers (in order of appearance)
    unique_chain_identifiers, indices = np.unique(chain_identifiers, return_index=True)
    unique_chain_identifiers = unique_chain_identifiers[np.argsort(indices)]

    # ... build dictionary entries for each chain
    for chain_identifier in unique_chain_identifiers:
        is_in_chain = chain_identifiers == chain_identifier
        seq = res_names[is_in_chain]
        chain_mask_atom_level = atom_array.chain_id == chain_identifier

        # --- Determine chain_type ---
        if "chain_type" not in atom_array.get_annotation_categories():
            # INFER from sequence
            if np.all(hetero[is_in_chain]):
                # EDGE CASE: If all atoms are "HETATM", override to non-polymer
                chain_type = ChainType.NON_POLYMER
            else:
                chain_type = infer_chain_type_from_three_letter(seq)
        else:
            # USE existing annotation from first atom in chain (should be consistent across chain)
            chain_type = atom_array.chain_type[chain_mask_atom_level][0]
            chain_type = ChainType.as_enum(chain_type)

        # --- Initialize chain info entry ---
        chain_info[chain_identifier] = {
            "chain_type": chain_type,
            "is_polymer": chain_type.is_polymer(),
            "res_id": res_ids[is_in_chain].tolist(),
            "res_name": res_names[is_in_chain].tolist(),
        }

        # Add label_entity if annotation exists
        if "label_entity_id" in atom_array.get_annotation_categories():
            entity_id = str(atom_array.label_entity_id[chain_mask_atom_level][0])
            chain_info[chain_identifier]["label_entity"] = entity_id

    return chain_info


def _enrich_with_entity_info(
    chain_info: dict[str, dict],
    entity: dict[str, dict] | None,
    entity_poly: dict[str, dict] | None,
) -> dict[str, dict]:
    """Enrich chain info with entity/entity_poly metadata."""
    # Sanity check: if entity data provided, we need label_entity keys in chain_info to know which chains to enrich
    has_entity_data = exists(entity) or exists(entity_poly)
    any_chain_has_label = any("label_entity" in info for info in chain_info.values())

    if has_entity_data and not any_chain_has_label:
        raise ValueError(
            "Entity data provided (entity/entity_poly) but no chains have label_entity field. "
            "The atom_array must have label_entity_id annotation to use entity enrichment."
        )

    if not any_chain_has_label:
        # No entity data available, skip enrichment
        return chain_info

    # Build entity lookup dict: {entity_id: {key: value}}
    entity_lookup = {}
    if entity and "id" in entity:
        for i, entity_id in enumerate(entity["id"]):
            entity_lookup[str(entity_id)] = {key: str(val[i]) for key, val in entity.items()}

    # Build entity_poly lookup dict: {entity_id: {key: value}}
    poly_lookup = {}
    if entity_poly and "entity_id" in entity_poly:
        for i, entity_id in enumerate(entity_poly["entity_id"]):
            poly_lookup[str(entity_id)] = {key: str(val[i]) for key, val in entity_poly.items()}

    # Enrich each chain
    for chain_id in chain_info:
        entity_id = chain_info[chain_id].get("label_entity")
        if not entity_id:
            continue

        # Get entity and entity_poly data from lookup dicts
        entity_data = entity_lookup.get(entity_id, {})
        poly_data = poly_lookup.get(entity_id, {})

        # Add EC numbers (if applicable)
        ec_raw = entity_data.get("pdbx_ec", "?")
        if ec_raw and ec_raw != "?":
            chain_info[chain_id]["ec_numbers"] = [ec.strip() for ec in ec_raw.split(",")]
        else:
            chain_info[chain_id]["ec_numbers"] = []

        # Add unprocessed sequences from entity_poly
        if "pdbx_seq_one_letter_code_can" in poly_data:
            chain_info[chain_id]["unprocessed_entity_canonical_sequence"] = poly_data[
                "pdbx_seq_one_letter_code_can"
            ].replace("\n", "")

        if "pdbx_seq_one_letter_code" in poly_data:
            chain_info[chain_id]["unprocessed_entity_non_canonical_sequence"] = poly_data[
                "pdbx_seq_one_letter_code"
            ].replace("\n", "")

        # Override chain_type and is_polymer with more specific information from the `_entity` or `_entity_poly` categories, if available
        if "type" in poly_data:
            # First preference: use chain type from entity_poly if available
            poly_chain_type = ChainType.as_enum(poly_data["type"])
            chain_info[chain_id]["chain_type"] = poly_chain_type
            chain_info[chain_id]["is_polymer"] = poly_chain_type.is_polymer()
        elif "type" in entity_data:
            # Fallback: use type from entity category if polymer_type not available for this entity
            entity_type = entity_data["type"].strip().lower()
            if entity_type == "polymer":
                # _entity.type states membership, not the polymer chemistry.
                # Re-infer the chemistry from the observed residue names.
                inferred = chain_info[chain_id]["chain_type"]
                if not inferred.is_polymer():
                    inferred = infer_chain_type_from_three_letter(chain_info[chain_id]["res_name"])
                entity_chain_type = inferred if inferred.is_polymer() else ChainType.OTHER_POLYMER
            else:
                entity_chain_type = ChainType.as_enum(entity_type)
            chain_info[chain_id]["chain_type"] = entity_chain_type
            chain_info[chain_id]["is_polymer"] = entity_chain_type.is_polymer()

    return chain_info


def _enrich_with_poly_seq(
    chain_info: dict[str, dict],
    entity_poly_seq: dict[str, np.ndarray] | None,
) -> dict[str, dict]:
    """Enrich chain info with poly_seq, merging observed + missing residues."""
    for chain_id in chain_info:
        entity_id = chain_info[chain_id].get("label_entity")
        if not entity_id:
            continue

        # Check if this entity has poly_seq data
        entity_mask = entity_poly_seq["entity_id"].astype(str) == str(entity_id)
        if not np.any(entity_mask):
            continue

        # Detect sequence heterogeneity
        has_heterogeneity = False
        if "hetero" in entity_poly_seq:
            has_heterogeneity = bool(np.any(entity_poly_seq["hetero"][entity_mask].astype(str) == "y"))
        chain_info[chain_id]["has_sequence_heterogeneity"] = has_heterogeneity

        # If entity has poly_seq data, MERGE sequences (polymers)
        # ... sequence from entity_poly_seq (what's in the CIF file, including missing residues)
        poly_seq_res_ids = entity_poly_seq["num"][entity_mask].astype(int)
        poly_seq_res_names = entity_poly_seq["mon_id"][entity_mask]

        # ... existing sequence (from what's observed in the structure; e.g., after alt_loc resolution)
        existing_res_ids = np.array(chain_info[chain_id]["res_id"]).astype(int)
        existing_res_names = np.array(chain_info[chain_id]["res_name"])

        # Subset the sequence from entity_poly_seq to only residues that are NOT existing
        missing_mask = ~np.isin(poly_seq_res_ids, existing_res_ids)
        missing_res_ids = poly_seq_res_ids[missing_mask]
        missing_res_names = poly_seq_res_names[missing_mask]

        if missing_res_ids.size > 0:
            # Ensure no duplicates in missing residues
            assert len(missing_res_ids) == len(set(missing_res_ids)), (
                f"Chain {chain_id}: Duplicate res_ids found in missing residues from entity_poly_seq. "
                f"Missing res_ids: {missing_res_ids}"
            )

            # Concatenate existing and missing residues (e.g., add unresolved N-terminal residues)
            combined_res_ids = np.concatenate([existing_res_ids, missing_res_ids])
            combined_res_names = np.concatenate([existing_res_names, missing_res_names])

            # Sort by res_id to maintain proper sequence order
            sort_indices = np.argsort(combined_res_ids)
            sorted_res_ids = combined_res_ids[sort_indices]
            sorted_res_names = combined_res_names[sort_indices]

            # Update chain_info with merged sequences
            chain_info[chain_id]["res_id"] = sorted_res_ids.tolist()
            chain_info[chain_id]["res_name"] = sorted_res_names.tolist()

    return chain_info
