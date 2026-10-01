"""Record definitions for preprocessing pipeline.

This module defines the core dataclasses for the normalized three-table design:
- Assembly: Entry-level metadata and statistics
- PNUnit: Polymer/non-polymer unit information
- Interface: Contact information between PN units
- ContactInfo: Intermediate contact data during processing
"""

import json
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Serialization Utilities
# =============================================================================


def to_python_native(obj: Any) -> Any:
    """Convert numpy/Path/datetime/Enum to Python native types recursively."""
    if isinstance(obj, dict):
        return {str(k): to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_python_native(item) for item in obj]
    if isinstance(obj, set):
        return [to_python_native(item) for item in sorted(obj)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return obj


def serialize_value(value: Any) -> Any:
    """Serialize a value for parquet storage (JSON for complex types)."""
    native = to_python_native(value)
    if isinstance(native, dict | list):
        return json.dumps(native)
    return native


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Serialize a record for parquet storage.

    Handles conversion of:
    - dicts/lists/sets -> JSON strings
    - Path -> str
    - Enum -> value
    - datetime -> ISO format string
    - numpy types -> native Python types
    """
    return {key: serialize_value(value) for key, value in record.items()}


def deserialize_field(value: Any) -> Any:
    """Deserialize a field from parquet storage.

    Attempts to parse JSON strings back to dicts/lists.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def to_dict_with_exclude(obj: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Serialize dataclass, converting tuples to lists, with optional exclusion."""
    exclude = exclude or set()
    result = {}
    for f in fields(obj):
        if f.name in exclude:
            continue
        val = getattr(obj, f.name)
        if isinstance(val, tuple):
            val = list(val)
        result[f.name] = val
    return result


# =============================================================================
# Core Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class Assembly:
    """Assembly-level record with metadata and statistics.

    This represents the entry-level data for a biological assembly,
    including experimental metadata, quality metrics, and token counts.
    """

    # Identifiers
    pdb_id: str
    assembly_id: str
    path: str

    # Experimental metadata
    resolution: float | None
    deposition_date: str | None
    release_date: str | None
    method: str | None
    ph: float | None

    # Assembly statistics
    clash_severity: str = "no-clash"
    num_polymer_pn_units: int = 0
    num_resolved_atoms_in_processed_assembly: int = 0
    total_num_atoms_in_unprocessed_assembly: int = 0
    fraction_zero_occupancy_atoms: float = 0.0
    all_pn_unit_iids_after_processing: tuple[str, ...] = ()

    # Bond flags
    has_unphysical_bonds: bool = False
    has_non_biological_bonds: bool = False

    # Token counts
    n_atomized_tokens: int = 0
    n_non_atomized_tokens: int = 0
    n_tokens_total: int = 0

    # Closest chains (for large assembly sampling)
    closest_20_pn_unit_iids: tuple[str, ...] = ()
    closest_50_pn_unit_iids: tuple[str, ...] = ()

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dictionary with optional field exclusion."""
        return to_dict_with_exclude(self, exclude)


@dataclass(frozen=True, slots=True)
class PNUnit:
    """Polymer or non-polymer unit record.

    This represents a single PN unit within an assembly, containing
    type information, structural properties, and sequence data for polymers.
    """

    # Foreign keys
    pdb_id: str
    assembly_id: str

    # Identifiers
    pn_unit_id: str
    pn_unit_iid: str
    molecule_id: str
    molecule_iid: str
    transformation_id: str

    # Type information
    pn_unit_type: str
    is_polymer: bool
    is_metal: bool = False
    is_loi: bool = False

    # Structural properties
    num_resolved_atoms: int = 0
    num_resolved_residues: int | None = None
    is_multichain: bool = False
    is_multiresidue: bool = False

    # Polymer-specific fields
    sequence_length: int | None = None
    has_non_canonical_residue: bool | None = None
    processed_entity_canonical_sequence: str | None = None
    processed_entity_non_canonical_sequence: str | None = None
    processed_entity_canonical_sequence_hash: str | None = None
    processed_entity_non_canonical_sequence_hash: str | None = None
    ec_numbers: tuple[str, ...] | None = None

    # Non-polymer-specific fields
    non_polymer_res_names: str | None = None
    bonded_polymer_pn_units: tuple[str, ...] | None = None
    ranking_model_fit: float | None = None

    # Contact information
    primary_polymer_partner_iid: str | None = None
    # Each contact dict has: pn_unit_iid, num_atoms, num_contacts, min_distance
    contacting_pn_unit_iids: tuple[dict[str, Any], ...] = ()

    # Sampling weights (for training data balancing)
    n_prot: int = 0
    n_peptide: int = 0
    n_nuc: int = 0
    n_ligand: int = 0

    # Closest chain flags (computed in _add_closest_chain_info)
    within_20_closest_chains: bool = True
    within_50_closest_chains: bool = True

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dictionary with optional field exclusion."""
        return to_dict_with_exclude(self, exclude)


@dataclass(frozen=True, slots=True)
class Interface:
    """Interface record between two PN units.

    This represents contact information between a pair of PN units,
    including distance metrics and interaction properties.
    """

    # Foreign keys
    pdb_id: str
    assembly_id: str
    pn_unit_1_iid: str
    pn_unit_2_iid: str

    # Contact properties
    num_contacts: int
    min_distance: float
    involves_covalent_modification: bool = False
    is_inter_molecule: bool = False
    involves_loi: bool = False
    involves_metal: bool = False

    # Closest chain flags (computed in _add_closest_chain_info)
    within_20_closest_chains: bool = True
    within_50_closest_chains: bool = True

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dictionary with optional field exclusion."""
        return to_dict_with_exclude(self, exclude)


# =============================================================================
# Helper Dataclasses
# =============================================================================


@dataclass
class ContactInfo:
    """Intermediate contact data during processing.

    This is a mutable container used during interface computation,
    before conversion to immutable Interface records.
    """

    pn_unit_1_iid: str
    pn_unit_2_iid: str
    num_contacts: int
    min_distance: float
    involves_covalent_modification: bool = False
