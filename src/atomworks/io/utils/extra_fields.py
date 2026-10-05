"""Utilities for extra field defaults and dtype inference."""

import logging
from typing import Any, Literal

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from atomworks.common import as_list

logger = logging.getLogger(__name__)

# Translation map for builtin field names to CIF field names
_TRANSLATE_BUILTIN_FIELDS = {
    "atom_id": "id",
    "charge": "pdbx_formal_charge",
    "b_factor": "B_iso_or_equiv",
    "occupancy": "occupancy",
}

ExtraFieldsType = list[str] | set[str] | frozenset[str] | dict[str, dict[str, Any] | None] | Literal["all"] | None


def infer_dtype_from_values(values: np.ndarray) -> tuple[np.dtype, np.ndarray]:
    """Infer dtype from values and convert to numeric type if possible.

    Attempts conversion in priority order: int → float → string.

    Args:
        values: Input array to infer dtype from.

    Returns:
        Tuple of (inferred_dtype, converted_array).
    """
    if np.issubdtype(values.dtype, np.number):
        return values.dtype, values
    for dtype in (np.int64, np.float64):
        try:
            return np.dtype(dtype), np.array(values, dtype=dtype)
        except (ValueError, TypeError):
            pass
    if np.isin(values, ["True", "False"]).all():
        values = np.array([v == "True" for v in values], dtype=bool)
        return values.dtype, values
    return values.dtype, values


def normalize_extra_fields(
    extra_fields: ExtraFieldsType,
    require_defaults: bool = False,
) -> dict[str, dict[str, Any]]:
    """Normalize extra_fields to dict of {name: {"default": ..., "dtype": ...}}.

    Args:
        extra_fields: Iterable of field names (list, set, frozenset) or dict with specs.
        require_defaults: If True, raise error for fields without defaults.

    Returns:
        Dict mapping field name to spec dict with "default" and "dtype" keys.
    """
    if extra_fields is None or extra_fields == "all":
        return {}

    result: dict[str, dict[str, Any]] = {}

    if isinstance(extra_fields, dict):
        for name, spec in extra_fields.items():
            if spec is None:
                result[name] = {"default": None, "dtype": None}
            else:
                default = spec.get("default")
                dtype = spec.get("dtype")
                # Infer dtype from default if not provided
                if dtype is None and default is not None:
                    dtype = np.array([default]).dtype
                result[name] = {"default": default, "dtype": dtype}
    else:
        # Handle list-like types (list, set, frozenset, OmegaConf ListConfig)
        # as_list handles these via duck typing
        for name in as_list(extra_fields):
            result[name] = {"default": None, "dtype": None}

    if require_defaults:
        missing = [n for n, s in result.items() if s["default"] is None and s["dtype"] is None]
        if missing:
            logger.warning(
                f"Skipping extra_fields without defaults (required when add_missing_atoms=True): {missing}. "
                f"To load these fields, provide defaults: extra_fields={{'{missing[0]}': {{'default': 0}}}}"
            )
            # Remove fields without defaults
            result = {n: s for n, s in result.items() if n not in missing}

    return result


def merge_extra_fields(
    required_fields: list[str],
    extra_fields: ExtraFieldsType,
) -> ExtraFieldsType:
    """Merge required fields into extra_fields.

    Args:
        required_fields: Fields that must be included.
        extra_fields: User-provided extra_fields (None, list, dict, or "all").

    Returns:
        Merged extra_fields maintaining input type. Returns list if extra_fields is None,
        otherwise preserves the input type (list→list, dict→dict, "all"→"all").
    """
    if extra_fields is None:
        return list(required_fields)

    if extra_fields == "all":
        return "all"

    if isinstance(extra_fields, dict):
        # Add required fields with None spec if not present
        result = dict(extra_fields)
        for field in required_fields:
            if field not in result:
                result[field] = None
        return result

    # List-like types (list, set, frozenset)
    field_list = as_list(extra_fields)
    return list(set(required_fields) | set(field_list))


def ensure_field_in_extra_fields(
    field_name: str,
    extra_fields: ExtraFieldsType,
    spec: dict[str, Any] | None = None,
) -> ExtraFieldsType:
    """Add field_name to extra_fields if not already present.

    Args:
        field_name: Name of field to ensure is present.
        extra_fields: Current extra_fields (None, list, dict, or "all").
        spec: Optional spec dict for the field (used when creating dict output from None).

    Returns:
        Updated extra_fields maintaining input type. If extra_fields is None and spec is
        provided, returns a dict; otherwise returns a list. For other types, preserves
        the input type.
    """
    if extra_fields == "all":
        return "all"

    if extra_fields is None:
        # Return dict if spec provided, else list
        if spec is not None:
            return {field_name: spec}
        return [field_name]

    if isinstance(extra_fields, dict):
        if field_name not in extra_fields:
            return {**extra_fields, field_name: spec}
        return extra_fields

    # List-like types
    field_list = as_list(extra_fields)
    if field_name not in field_list:
        return [*field_list, field_name]
    return list(field_list)


def get_default_array(spec: dict[str, Any], n_atoms: int) -> np.ndarray:
    """Create array of default values from a field spec.

    If only dtype is provided, uses sensible defaults: nan for floats, -1 for ints.

    Args:
        spec: Dict with ``"default"`` and ``"dtype"`` keys.
        n_atoms: Length of the array to create.

    Returns:
        Array filled with the default value.
    """
    default = spec["default"]
    dtype = spec["dtype"]

    if default is None and dtype is not None:
        # Generate sensible default from dtype
        if np.issubdtype(dtype, np.floating):
            default = np.nan
        elif np.issubdtype(dtype, np.integer):
            default = -1
        elif np.issubdtype(dtype, np.bool_):
            default = False
        else:
            default = ""

    if dtype is None and default is not None:
        dtype = np.array([default]).dtype

    return np.full(n_atoms, default, dtype=dtype)


def transfer_annotation_if_exists(
    source: AtomArray,
    target: AtomArray,
    annot: str,
    source_idx: int,
) -> None:
    """Transfer annotation value from source to target if it exists.

    Broadcasts source[source_idx] value to all atoms in target.

    Args:
        source: AtomArray to copy from.
        target: AtomArray to copy to (modified in-place).
        annot: Annotation name.
        source_idx: Index in source to take value from.
    """
    if annot in source.get_annotation_categories():
        value = source.get_annotation(annot)[source_idx]
        target.set_annotation(annot, np.full(len(target), value))


def filter_extra_fields(extra_fields: list[str], atom_site: pdbx.CIFCategory) -> list[str]:
    """Filter the extra fields to only include fields that are actually present in the file."""
    # Get all available fields
    available_fields = set(atom_site.keys())

    filtered = [
        field_name
        for field_name in extra_fields
        if _TRANSLATE_BUILTIN_FIELDS.get(field_name, field_name) in available_fields
    ]

    # Log missing fields only if debug is enabled
    if logger.isEnabledFor(logging.DEBUG):
        missing = set(extra_fields) - set(filtered)
        for field_name in missing:
            logger.debug("Field %s not found in file, ignoring.", field_name)

    return filtered


def apply_extra_field_dtypes(
    atom_array_stack: AtomArrayStack | AtomArray,
    extra_field_specs: dict,
    field_names: list[str] | None = None,
) -> AtomArrayStack | AtomArray:
    """Apply dtype inference/conversion for extra field annotations.

    Args:
        atom_array_stack: Structure containing the annotations to convert.
        extra_field_specs: Dict mapping field names to spec dicts with dtype info.
        field_names: Fields to process. Defaults to keys from ``extra_field_specs``.

    Returns:
        The input structure with converted annotation dtypes.
    """
    if field_names is None:
        field_names = list(extra_field_specs.keys())

    annotations = atom_array_stack.get_annotation_categories()

    for field_name in field_names:
        if field_name not in annotations:
            continue

        values = atom_array_stack.get_annotation(field_name)
        spec = extra_field_specs.get(field_name)
        dtype = spec.get("dtype") if spec is not None else None

        if dtype is not None:
            try:
                # Fill CIF unknown/inapplicable markers and empty strings with the declared default.
                # Object dtype permits mixing string input with numeric defaults before conversion.
                if spec.get("default") is not None:
                    values = np.where(np.isin(values, ["?", ".", ""]), spec["default"], values.astype(object))
                converted = np.array(values, dtype=dtype)
                atom_array_stack.del_annotation(field_name)
                atom_array_stack.set_annotation(field_name, converted)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert field '{field_name}' to dtype {dtype}")
        elif field_name in (
            "chain_entity",
            "pn_unit_entity",
            "molecule_entity",
            "label_entity_id",
            "chain_type",
            "charge",
        ):
            # entity fields use int16 _except_ ... chain_type/charge use int8
            target_dtype = np.int8 if field_name in ("chain_type", "charge") else np.int16
            try:
                # Delete annotation first to avoid dtype promotion
                atom_array_stack.del_annotation(field_name)
                atom_array_stack.set_annotation(field_name, np.array(values, dtype=target_dtype))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {field_name} values to {target_dtype.__name__}")
        else:
            inferred_dtype, converted = infer_dtype_from_values(values)
            if inferred_dtype != values.dtype:
                # set_annotation would promote to a common dtype if an existing annotation is present
                atom_array_stack.del_annotation(field_name)
                atom_array_stack.set_annotation(field_name, converted)

    return atom_array_stack
