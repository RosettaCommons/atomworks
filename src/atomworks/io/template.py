import functools
import logging
import os
from itertools import pairwise
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.bonds import connect_via_residue_names

from atomworks.constants import (
    CCD_MIRROR_PATH,
    CHAIN_LEVEL_ANNOTATIONS,
    HYDROGEN_LIKE_SYMBOLS,
    MOLECULE_LEVEL_ANNOTATIONS,
    PN_UNIT_LEVEL_ANNOTATIONS,
    RESIDUE_LEVEL_ANNOTATIONS,
    TRANSFORMATION_LEVEL_ANNOTATIONS,
)
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import insert_atoms
from atomworks.io.utils.bonds import (
    build_bond_dict_for_atom_array,
)
from atomworks.io.utils.ccd import (
    _get_base_ccd_template,
    _standard_ccd_only_cache,
    get_available_ccd_codes,
)
from atomworks.io.utils.extra_fields import (
    get_default_array,
)
from atomworks.io.utils.link_chemistry import (
    add_polymer_bonds,
    get_chem_comp_leaving_atom_groups,
    resolve_link_chemistry,
)
from atomworks.io.utils.scatter import get_segments

logger = logging.getLogger(__file__)


@_standard_ccd_only_cache(functools.lru_cache(maxsize=2048))
def _get_missing_atoms_template(
    ccd_code: str,
    actual_atom_names: frozenset[str],  # Must be hashable for lru_cache
    ccd_mirror_path: str,
    hydrogen_policy: Literal["keep", "remove"],
) -> AtomArray | None:
    """Get template AtomArray for missing atoms in a residue (cached implementation).

    Args:
        ccd_code: 3-letter CCD code (e.g., "ALA")
        actual_atom_names: Set of atom names currently present
        ccd_mirror_path: Path to CCD mirror (as string), or empty string to use Biotite's built-in CCD
        hydrogen_policy: Whether to keep or remove hydrogens.

    Returns:
        Template with only missing atoms, or None if no atoms are missing

    Raises:
        ValueError: If actual_atom_names contains atoms not in CCD template
    """
    full_template = _get_base_ccd_template(ccd_code, ccd_mirror_path, "keep")
    actual_list = list(actual_atom_names)

    # Validate: no unexpected atoms
    unexpected = ~np.isin(actual_list, full_template.atom_name, assume_unique=True)
    if np.any(unexpected):
        unexpected_names = [actual_list[i] for i in np.where(unexpected)[0]]
        raise ValueError(f"CCD {ccd_code}: Unexpected atoms not in CCD template: {unexpected_names}")

    # Find missing atoms
    template = _get_base_ccd_template(ccd_code, ccd_mirror_path, hydrogen_policy)
    template_atom_names = template.atom_name
    missing_mask = ~np.isin(template_atom_names, actual_list)

    if not np.any(missing_mask):
        return None

    # Return subset with only missing atoms
    return template[missing_mask]


@_standard_ccd_only_cache(functools.lru_cache(maxsize=2048))
def _non_terminal_expected_atom_count(
    ccd_code: str,
    ccd_mirror_path: str,
    hydrogen_policy: str,
    chain_type: ChainType,
) -> int | None:
    """Expected non-terminal atom count, or None for unknown or ambiguous polymer leaving groups."""
    bond_atoms = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(chain_type)
    if bond_atoms is None:
        return None

    leaving_groups = get_chem_comp_leaving_atom_groups(ccd_code, ccd_mirror_path)
    polymer_leaving: set[str] = set()
    for bond_atom in bond_atoms:
        groups = leaving_groups.get(bond_atom, ())
        if len(groups) > 1:
            return None
        polymer_leaving.update(*groups)

    template = _get_base_ccd_template(ccd_code, ccd_mirror_path, hydrogen_policy)
    # Intersect with template to respect hydrogen_policy (H atoms may already be absent)
    return len(template) - len(polymer_leaving & set(template.atom_name))


def get_empty_ccd_template(
    ccd_code: str,
    *,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    hydrogen_policy: Literal["keep", "remove"] = "remove",
    extra_field_specs: dict[str, dict[str, Any]] | None = None,
    **res_wise_annotations: int | float | str,
) -> AtomArray:
    """Get empty CCD template with safe independent copy.

    Creates an empty template AtomArray from a Chemical Component Dictionary (CCD)
    entry with optional residue-wise annotations. Returns an independent copy that
    can be safely modified without affecting the cached template.

    Args:
        ccd_code: The three-letter code of the chemical component to create a template for.
        ccd_mirror_path: Path to the local CCD mirror directory. Defaults to CCD_MIRROR_PATH.
        hydrogen_policy: Whether to keep or remove hydrogen atoms from the template. Defaults to ``"remove"``.
        extra_field_specs: Dict mapping field names to spec dicts with "default" and "dtype" keys. If provided,
            extra fields will be initialized with their default values for new template atoms.
        **res_wise_annotations: Additional residue-wise annotations to add to the template.
            Values can be int, float, or str and will be broadcast to all atoms in the template.

    Returns:
        AtomArray: An empty template structure with nan coordinates but with bonds and
            annotations from the CCD entry, plus any additional specified annotations.
            This is an independent copy that can be safely modified.

    Example:
        >>> template = get_empty_ccd_template("ALA", chain_id="A", res_id=1, occupancy=1.0)
    """
    # Get cached base template and make an independent copy for annotation
    template = _get_base_ccd_template(ccd_code, str(ccd_mirror_path or ""), hydrogen_policy).copy()

    n_atoms = len(template)
    for annot, value in res_wise_annotations.items():
        if value is not None:
            template.set_annotation(annot, np.full(n_atoms, value))

    # Initialize extra fields with default values
    if extra_field_specs:
        for field_name, spec in extra_field_specs.items():
            if spec["default"] is not None or spec["dtype"] is not None:
                template.set_annotation(field_name, get_default_array(spec, n_atoms))

    return template


def add_missing_atoms_for_chain(
    sequence: list[str],
    res_ids: list[int],
    atom_array: AtomArray | None = None,
    chain_mask: np.ndarray | None = None,
    chain_type: ChainType | None = None,
    chain_id: str | None = None,
    *,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    hydrogen_policy: Literal["keep", "remove"] = "keep",
    extra_field_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[AtomArray], list[int]]:
    """Build templates for missing/incomplete residues and determine insertion positions.

    Handles two use cases:

    1. **From scratch** (``atom_array=None``) — ``chain_id`` and ``chain_type`` are
       required; every residue in ``sequence`` is treated as completely missing.
    2. **Fill gaps** — ``atom_array`` and ``chain_mask`` select an existing chain;
       residues absent or partially resolved relative to the CCD template are filled in.

    Args:
        sequence: List of 3-letter CCD codes for this chain.
        res_ids: List of residue IDs corresponding to ``sequence``.
        atom_array: Full structure (all chains).  Pass ``None`` to build from scratch.
        chain_mask: Boolean mask over ``atom_array`` selecting atoms in this chain.
            Required when ``atom_array`` is provided.
        chain_type: Chain type for this chain.  Required when ``atom_array`` is ``None``.
        chain_id: Chain ID.  Required when ``atom_array`` is ``None``; otherwise inferred
            from ``atom_array``.
        ccd_mirror_path: Path to CCD mirror.
        extra_field_specs: Extra field specifications used to initialise additional
            annotations in the inserted templates.

    Returns:
        Tuple of ``(templates_to_insert, insertion_positions)`` where
        ``templates_to_insert`` is a list of AtomArrays for missing/incomplete residues
        and ``insertion_positions`` is a list of atom indices (insert before index *i*).
    """
    # ... get available CCD codes for proactive checking
    available_ccds = get_available_ccd_codes(str(ccd_mirror_path or ""))
    # A custom CCD registry stays fixed while this chain is filled.
    expected_atom_count = functools.cache(_non_terminal_expected_atom_count)
    missing_atoms_template = functools.cache(_get_missing_atoms_template)

    if atom_array is None:
        # From-scratch: no existing atoms, every residue will be CASE 1
        if chain_id is None or chain_type is None:
            raise ValueError(
                "_add_missing_atoms_for_chain requires chain_id and chain_type " "when atom_array is None."
            )

        # ... defaults
        chain_indices = np.array([], dtype=int)
        res_id_to_bounds: dict[int, tuple[int, int]] = {}
        sorted_existing_res_ids: list[int] = []
        res_id_to_sorted_idx: dict[int, int] = {}
        chain_level_annots: dict[str, Any] = {
            "chain_id": chain_id,
            "chain_type": np.int8(chain_type),
            "is_polymer": chain_type.is_polymer(),
        }
    else:
        # Existing AtomArray
        # ... pre-compute chain-level data structures for O(1) lookups
        chain_indices = np.where(chain_mask)[0]  # Maps local → global positions
        if len(chain_indices) == 0:
            return [], []  # Handle empty chain edge case

        # Extract chain_id from first atom in chain
        inferred_chain_id = atom_array.chain_id[chain_indices[0]]
        if chain_id is not None:
            assert (
                chain_id == inferred_chain_id
            ), f"chain_id mismatch: provided {chain_id!r} but atom_array has {inferred_chain_id!r}"
        chain_id = inferred_chain_id

        # +--- Compute within-chain residue boundaries ----+
        annots_for_residues = [
            annot
            for annot in ["chain_id", "res_name", "res_id", "ins_code", "transformation_id"]
            if annot in atom_array.get_annotation_categories()
        ]
        chain_annot_arrays = [atom_array.get_annotation(annot)[chain_indices] for annot in annots_for_residues]

        # Compute residue boundaries (relative to the chain start)
        _res_start_ends = get_segments(*chain_annot_arrays, add_exclusive_stop=True)
        chain_res_starts, chain_res_ends = _res_start_ends[:-1], _res_start_ends[1:]

        # O(1) lookup: res_id → (local_start, local_end)
        res_id_to_bounds = {}
        for i, start_idx in enumerate(chain_res_starts):
            res_id_at_start = atom_array.res_id[chain_indices[start_idx]]
            res_id_to_bounds[res_id_at_start] = (start_idx, chain_res_ends[i])

        # Pre-compute sorted res_ids once (so that later we can do O(log n) lookups instead of O(n) .index() calls)
        sorted_existing_res_ids = sorted(res_id_to_bounds.keys())
        res_id_to_sorted_idx = {res_id: idx for idx, res_id in enumerate(sorted_existing_res_ids)}

        # Copy chain-level and higher annotations from the FIRST atom in this chain
        chain_level_annots = {}
        for annot in (
            *CHAIN_LEVEL_ANNOTATIONS,
            *PN_UNIT_LEVEL_ANNOTATIONS,
            *MOLECULE_LEVEL_ANNOTATIONS,
            *TRANSFORMATION_LEVEL_ANNOTATIONS,
        ):
            if annot in atom_array.get_annotation_categories():
                chain_level_annots[annot] = atom_array.get_annotation(annot)[chain_indices[0]]

        if chain_type is not None:
            if "chain_type" in chain_level_annots:
                assert (
                    ChainType.as_enum(chain_level_annots["chain_type"]) == chain_type
                ), f"chain_type mismatch for chain {chain_id}: provided {chain_type!r} but atom_array has {ChainType.as_enum(chain_level_annots['chain_type'])!r}"
            chain_level_annots["chain_type"] = np.int8(chain_type)
            chain_level_annots["is_polymer"] = chain_type.is_polymer()
        elif "chain_type" not in chain_level_annots:
            raise ValueError(f"chain_type is required for chain {chain_id}: not provided and not in atom_array.")

    templates_to_insert = []
    insertion_positions = []

    for res_idx, (res_id, ccd_code) in enumerate(zip(res_ids, sequence, strict=True)):
        # +----- Build template for missing/partial residue -----+
        if res_id not in res_id_to_bounds:
            # CASE 1: Residue completely missing - need full template with annotations
            template_kwargs = {**chain_level_annots, "res_id": res_id, "occupancy": 0.0, "b_factor": np.nan}

            if ccd_code not in available_ccds:
                raise ValueError(
                    f"Cannot add completely missing residue {ccd_code} (res_id={res_id}, chain={chain_id}): "
                    f"No CCD template available in mirror, registry, or Biotite. "
                    f"This residue is completely absent from the structure and requires a valid CCD template. "
                    f"Consider: (1) checking CCD mirror path, (2) registering custom template via "
                    f"register_custom_ccd_entry(), or (3) providing partial atoms for this residue."
                )

            # CCD is available - use it (no try/except)
            atoms_to_insert = get_empty_ccd_template(
                ccd_code,
                ccd_mirror_path=ccd_mirror_path,
                hydrogen_policy=hydrogen_policy,
                extra_field_specs=extra_field_specs,
                **template_kwargs,
            )
        else:
            # CASE 2: Residue exists - check what's missing
            local_start, local_end = res_id_to_bounds[res_id]

            # Validate residue name
            actual_res_name = atom_array.res_name[chain_indices[local_start]]
            assert actual_res_name == ccd_code, (
                f"Mismatch: residue ID {res_id} in chain {chain_id} "
                f"has name {actual_res_name} instead of {ccd_code}"
            )

            # Extract atom names
            actual_atom_names = atom_array.atom_name[chain_indices[local_start:local_end]]

            # Check CCD availability
            if ccd_code not in available_ccds:
                logger.debug(
                    f"CCD {ccd_code} not available, cannot determine missing atoms for "
                    f"partial residue (res_id={res_id}, chain={chain_id}). "
                    f"Assuming all expected atoms are present."
                )
                continue

            # Fast path: non-terminal residues usually have complete atom sets
            # NOTE: Assumes that we have first standardized atom names
            is_terminal = res_idx == 0 or res_idx == len(sequence) - 1
            if not is_terminal:
                expected = expected_atom_count(
                    ccd_code,
                    str(ccd_mirror_path or ""),
                    hydrogen_policy,
                    ChainType.as_enum(chain_level_annots["chain_type"]),
                )
                if expected is not None:
                    if hydrogen_policy == "remove":
                        actual_elements = atom_array.element[chain_indices[local_start:local_end]]
                        actual_count = sum(1 for e in actual_elements if e not in HYDROGEN_LIKE_SYMBOLS)
                    else:
                        actual_count = len(actual_atom_names)
                    if actual_count == expected:
                        continue  # All expected atoms present

            # Slow path: terminal residues, partial residues, or non-standard chain types
            actual_atom_names_set = frozenset(actual_atom_names)

            # CCD is available - get missing atoms template
            # For partial residues, we catch errors (e.g., unexpected atoms) and assume complete
            try:
                missing_template = missing_atoms_template(
                    ccd_code, actual_atom_names_set, str(ccd_mirror_path or ""), hydrogen_policy
                )
            except (ValueError, AttributeError) as e:
                logger.debug(
                    f"Cannot determine missing atoms for partial residue {ccd_code} "
                    f"(res_id={res_id}, chain={chain_id}): {e}. "
                    f"Assuming all expected atoms are present."
                )
                continue

            if missing_template is None:
                continue  # No missing atoms

            # Make a copy and add residue-specific annotations
            atoms_to_insert = missing_template.copy()
            n_atoms = len(atoms_to_insert)

            # Add chain-level and residue-level annotations
            for annot, value in chain_level_annots.items():
                atoms_to_insert.set_annotation(annot, np.full(n_atoms, value))

            atoms_to_insert.set_annotation("occupancy", np.zeros(n_atoms))
            atoms_to_insert.set_annotation("b_factor", np.full(n_atoms, np.nan))

            # Copy residue-level annotations from first atom in the residue
            # (E.g., res_id, res_name, hetero, ins_code)
            first_atom_idx = chain_indices[local_start]
            for annot in RESIDUE_LEVEL_ANNOTATIONS:
                if annot in atom_array.get_annotation_categories():
                    annot_value = atom_array.get_annotation(annot)[first_atom_idx]
                    atoms_to_insert.set_annotation(annot, np.full(n_atoms, annot_value))

            # Initialize extra fields
            if extra_field_specs:
                for field_name, spec in extra_field_specs.items():
                    if spec["default"] is not None or spec["dtype"] is not None:
                        atoms_to_insert.set_annotation(field_name, get_default_array(spec, n_atoms))

        # +----- Calculate insertion position for these atom(s) -----+
        # Determine index of next residue in sorted list
        if res_id not in res_id_to_bounds:
            # Missing residue: find where it should be in sorted order
            next_idx = np.searchsorted(sorted_existing_res_ids, res_id)
        else:
            # Partial residue: insert after current residue
            next_idx = res_id_to_sorted_idx[res_id] + 1

        # Calculate global insertion position based on next_idx
        if next_idx >= len(sorted_existing_res_ids):
            # Insert at end of chain (or position 0 when building from scratch)
            global_insert_pos = int(chain_indices[-1]) + 1 if len(chain_indices) > 0 else 0
        else:
            # Insert before next residue
            next_res_id = sorted_existing_res_ids[next_idx]
            next_local_start = res_id_to_bounds[next_res_id][0]
            global_insert_pos = chain_indices[next_local_start]

        templates_to_insert.append(atoms_to_insert)
        insertion_positions.append(global_insert_pos)

    return templates_to_insert, insertion_positions


def add_missing_atoms(
    atom_array: AtomArray,
    chain_info_dict: dict[str, dict[str, Any]],
    *,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    hydrogen_policy: Literal["keep", "remove"] = "keep",
    extra_field_specs: dict[str, dict[str, Any]] | None = None,
) -> AtomArray:
    """Add atoms that are implied by the sequence but missing from the structure.

    To identify missing atoms, we match based on the CCD template for each residue.

    Fills gaps in a partially-resolved structure by inserting completely or partially
    missing residues alongside existing atoms.

    Args:
        atom_array: Existing structure to fill gaps in.
        chain_info_dict: Dict mapping chain IDs to residue info with keys:

            - ``"res_name"``: List of 3-letter CCD codes (e.g., ``["ALA", "GLY"]``).
            - ``"res_id"``: List of residue IDs (e.g., ``[1, 2, 3]``).
            - ``"chain_type"``: Chain type as a :py:class:`ChainType` enum or string.
              Inferred from the existing structure if omitted.
        ccd_mirror_path: Path to local CCD mirror.
        extra_field_specs: Dict mapping field names to spec dicts with
            ``"default"``/``"dtype"`` keys for extra annotations.

    Returns:
        AtomArray with missing residues inserted (``occupancy=0.0``, NaN coords,
        including hydrogens).
    """
    if "charge" not in atom_array.get_annotation_categories():
        raise ValueError(
            "Input atom_array is missing 'charge' annotation. "
            "Ensure the structure was loaded via parse() or get_structure(), which always sets charge."
        )

    # Collect ALL insertions across ALL chains
    all_templates = []
    all_positions = []

    # Use chain_iid if available (for bio-assemblies), otherwise chain_id
    chain_identifier_key = "chain_id"
    if "chain_iid" in atom_array.get_annotation_categories():
        chain_identifier_key = "chain_iid"
    elif "transformation_id" in atom_array.get_annotation_categories():
        raise ValueError(
            "Structure has transformation_id but no chain_iid annotation. "
            "Call add_iid_annotations() first to disambiguate chain instances."
        )

    # Loop over chain identifiers (preserve first-occurrence order)
    chain_identifiers = atom_array.get_annotation(chain_identifier_key)
    _, _first_occ = np.unique(chain_identifiers, return_index=True)
    unique_chain_identifiers = chain_identifiers[np.sort(_first_occ)]

    # Process each unique chain (or chain instance for bio-assemblies)
    for chain_identifier in unique_chain_identifiers:
        # Create mask for this chain instance
        chain_mask = chain_identifiers == chain_identifier

        # Get the actual chain_id for looking up sequence in chain_info_dict
        chain_id = atom_array.chain_id[np.where(chain_mask)[0][0]]
        assert (
            chain_id in chain_info_dict
        ), f"Chain ID {chain_id} (from chain_identifier {chain_identifier}) not found in chain_info_dict!"

        # Extract sequence info
        chain_info = chain_info_dict[chain_id]
        sequence = chain_info["res_name"]
        res_ids = chain_info["res_id"]

        # Get the template atom arrays and indices of where to insert these atom arrays for this chain
        chain_type = ChainType.as_enum(chain_info["chain_type"])
        templates, positions = add_missing_atoms_for_chain(
            atom_array=atom_array,
            chain_mask=chain_mask,
            sequence=sequence,
            res_ids=res_ids,
            chain_type=chain_type,
            ccd_mirror_path=ccd_mirror_path,
            hydrogen_policy=hydrogen_policy,
            extra_field_specs=extra_field_specs,
        )

        all_templates.extend(templates)
        all_positions.extend(positions)

    if all_templates:
        atom_array = insert_atoms(atom_array, all_templates, all_positions)

    ensure_annotations(atom_array, "atomic_number", "chem_comp_type")

    return atom_array


def infer_bonds_from_residue_names(
    atom_array: AtomArray,
    custom_bond_dict: dict[str, dict[tuple[str, str], int]] | None = None,
    sanitize: bool = True,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    """Add bonds to an AtomArray, using the CCD as ground-truth.

    Adds two types of bonds:
    1. Intra-residue bonds based on the CCD template for each residue.
    2. Inter-residue bonds inferred from the sequence (e.g., peptide bonds between consecutive amino acids).

    Does NOT add inter-chain and/or `struct_conn` bonds (which are assumed to have been added separately).

    When ``sanitize=True``, cleans the resultant structure by:
    - Removing leaving atoms and fixing bond orders for nucleophilic additions
    - Fixing formal charges on atoms involved in inter-residue bonds
    - Correcting charged amide nitrogens

    Args:
        atom_array: Structure to add bonds to. Requires a ``charge`` annotation, even when ``sanitize=False``.
        custom_bond_dict: Optional custom bonds. Maps residue names to
            ``{(atom1_name, atom2_name): bond_type_int}``. Overrides CCD for
            those residues.
        sanitize: If ``True``, resolves leaving atoms, fixes bond orders,
            formal charges, and amide nitrogens. Requires hydrogens or nhyd
            annotation to be present. Defaults to ``True``.
        ccd_mirror_path: Path to local CCD mirror.

    Returns:
        AtomArray with bonds added and processed (modified in-place).
    """

    assert "charge" in atom_array.get_annotation_categories(), "Bond inference requires a 'charge' annotation."

    # Edge case: empty array
    if len(atom_array) == 0:
        return atom_array

    # Build complete bond dictionary: custom bonds (CIF) + CCD bonds
    custom_bond_dict = build_bond_dict_for_atom_array(
        atom_array,
        custom_bond_dict=custom_bond_dict,
        ccd_mirror_path=ccd_mirror_path,
    )

    # Reuse layouts only within this call, including repeated names from alternate locations.
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    atom_names = atom_array.atom_name.tolist()
    res_names = atom_array.res_name
    bond_templates, bond_chunks = {}, []
    for start, stop in pairwise(residue_starts):
        key = res_names[start], tuple(atom_names[start:stop])
        if key not in bond_templates:
            bond_templates[key] = connect_via_residue_names(
                atom_array[start:stop], inter_residue=False, custom_bond_dict=custom_bond_dict
            ).as_array()
        residue_bonds = bond_templates[key].copy()
        residue_bonds[:, :2] += int(start)
        bond_chunks.append(residue_bonds)
    bonds = struc.BondList(len(atom_array), np.concatenate(bond_chunks))

    # Merge with existing bonds if present; otherwise, set new bonds
    if atom_array.bonds is not None:
        atom_array.bonds = atom_array.bonds.merge(bonds)
    else:
        atom_array.bonds = bonds

    # Add polymer inter-residue bonds, skipping pairs already bonded (e.g. via struct_conn)
    atom_array = add_polymer_bonds(atom_array)

    return resolve_link_chemistry(atom_array) if sanitize else atom_array
