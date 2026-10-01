"""Standardization utilities for structural files."""

import logging
import os
from typing import Literal

import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import (
    CCD_MIRROR_PATH,
    DO_NOT_MATCH_CCD,
    HYDROGEN_LIKE_SYMBOLS,
    STANDARD_POLYMER_RESIDUES,
)
from atomworks.io.utils.ccd import get_atom_names_for_residue, get_available_ccd_codes
from atomworks.io.utils.selection import get_residue_starts

logger = logging.getLogger(__name__)

_CANONICAL_RESIDUES: frozenset[str] = frozenset(STANDARD_POLYMER_RESIDUES)


def _handle_unmatched_atom(
    atom_name: str,
    res_name: str,
    atom_array: AtomArray,
    atom_idx: int,
    start_idx: int,
    on_mismatch: Literal["filter", "warn", "raise"],
    on_mismatch_non_heavy: Literal["filter", "warn", "raise"] | None,
    atoms_to_keep: np.ndarray,
) -> None:
    """Handle an atom that matches neither standard nor alternative names."""
    # Determine atom type and select appropriate policy
    is_non_heavy = atom_array.element[atom_idx] in HYDROGEN_LIKE_SYMBOLS
    policy = on_mismatch_non_heavy if (is_non_heavy and on_mismatch_non_heavy is not None) else on_mismatch
    atom_type_str = "non-heavy atom" if is_non_heavy else "heavy atom"

    # Get residue context for error message
    chain_id = atom_array.chain_id[start_idx]
    res_id = atom_array.res_id[start_idx]
    msg = (
        f"Atom '{atom_name}' in residue {res_name} "
        f"(chain {chain_id}, res_id {res_id}) matches neither "
        f"standard nor alternative CCD names"
    )

    # Apply policy
    if policy == "raise":
        raise ValueError(msg)
    elif policy == "filter":
        logger.warning(f"{msg} - removing {atom_type_str} " f"(will be re-added by add_missing_atoms if valid)")
        atoms_to_keep[atom_idx] = False
    elif policy == "warn":
        logger.warning(msg)


def standardize_atom_names(
    atom_array: AtomArray,
    on_mismatch: Literal["filter", "warn", "raise"] = "raise",
    on_mismatch_non_heavy: Literal["filter", "warn", "raise"] | None = None,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    non_canonical_only: bool = True,
) -> AtomArray:
    """Standardize atom names to CCD canonical naming.

    Processes each residue, attempting to match atoms using:
    1. Standard CCD atom names
    2. Alternative CCD atom names (if available), converting to standard
    3. Handling unmatched atoms according to ``on_mismatch`` policy

    NOTE: If we use are using a custom CCD registry, that entry will
    be considered "standard" (even if its names don't match the canonical
    CCD).

    Args:
        atom_array: Structure with potentially non-standard atom names.
        on_mismatch: How to handle atoms matching neither standard nor alternative.
            Applied to heavy atoms, and to non-heavy atoms if ``on_mismatch_non_heavy``
            is not specified:
            - ``"raise"``: Raise ``ValueError`` on first unmatched atom (default)
            - ``"filter"``: Remove unmatched atoms and log warning.
            - ``"warn"``: Keep unmatched atoms and log warning
        on_mismatch_non_heavy: Optional separate policy for non-heavy atoms
            (H, D, T isotopes). If ``None`` (default), uses ``on_mismatch`` value.
        ccd_mirror_path: Path to local CCD mirror.
        non_canonical_only: If ``True`` (default), only standardize non-canonical residues.
            Canonical residues (standard 20 amino acids, standard RNA/DNA nucleotides)
            will be skipped. Defaults to ``True`` for efficiency.

    Returns:
        AtomArray with standardized names. May be shorter than input if
        ``on_mismatch="filter"`` removes atoms.
    """
    # Normalize
    ccd_mirror_path_str = str(ccd_mirror_path or "")

    available_ccds = get_available_ccd_codes(ccd_mirror_path_str)

    # Pre-fetch atom name data for all unique residues
    unique_res_names = np.unique(atom_array.res_name)
    res_name_to_atom_names = {}
    for res_name in unique_res_names:
        if non_canonical_only and res_name in _CANONICAL_RESIDUES:
            continue
        if res_name not in DO_NOT_MATCH_CCD and res_name in available_ccds:
            template_data = get_atom_names_for_residue(res_name, ccd_mirror_path_str)
            if template_data[0]:  # Check if std_names_set is not empty
                res_name_to_atom_names[res_name] = template_data

    atoms_to_keep = np.ones(len(atom_array), dtype=bool)

    # Loop over residue
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    for start_idx, end_idx in zip(_res_starts, _res_ends, strict=False):
        res_name = atom_array.res_name[start_idx]

        # Skip if we don't have template data for this residue
        if res_name not in res_name_to_atom_names:
            continue

        # Get pre-fetched template data: standard names, alt names, and mapping alt -> standard
        std_names_set, alt_names_set, alt_to_std = res_name_to_atom_names[res_name]

        # Check if all atoms already match standard names - if so, skip processing this residue
        residue_atom_names = atom_array.atom_name[start_idx:end_idx]
        if std_names_set.issuperset(residue_atom_names):
            continue

        # Decide if we should match by standard atom names or alternative atom IDs (if available)
        match_by = "atom_name"
        if alt_names_set:  # We have alternative names available from the CCD
            n_matches_std = sum(1 for name in residue_atom_names if name in std_names_set)
            n_matches_alt = sum(1 for name in residue_atom_names if name in alt_names_set)

            if n_matches_alt > n_matches_std:
                match_by = "alt_atom_id"
                # Log message about using alternative atom IDs for this residue (since this isn't typical)
                chain_id = atom_array.chain_id[start_idx]
                res_id = atom_array.res_id[start_idx]
                logger.info(f"Residue {res_name} (chain {chain_id}, res_id {res_id}): using alternative atom IDs")

        # Process each atom in the residue
        for j, atom_name in enumerate(residue_atom_names):
            atom_idx = start_idx + j
            matched = False

            # Match and standardize atom name
            if match_by == "atom_name":
                if atom_name in std_names_set:
                    matched = True
                elif atom_name in alt_to_std:
                    atom_array.atom_name[atom_idx] = alt_to_std[atom_name]
                    matched = True
            else:  # match_by == "alt_atom_id"
                if atom_name in alt_names_set and atom_name in alt_to_std:
                    atom_array.atom_name[atom_idx] = alt_to_std[atom_name]
                    matched = True
                elif atom_name in std_names_set:
                    matched = True

            # Handle unmatched atoms
            if not matched:
                _handle_unmatched_atom(
                    atom_name,
                    res_name,
                    atom_array,
                    atom_idx,
                    start_idx,
                    on_mismatch,
                    on_mismatch_non_heavy,
                    atoms_to_keep,
                )

    # Apply filter
    if not np.all(atoms_to_keep):
        n_removed = np.sum(~atoms_to_keep)
        logger.info(f"Filtered {n_removed} atoms with non-standard names")
        atom_array = atom_array[atoms_to_keep]

    return atom_array
