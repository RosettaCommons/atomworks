"""Serialization and deserialization helpers for StandardAnnotation objects.

These helpers are used by :py:func:`~atomworks.io.utils.io_utils._to_cif_or_bcif`
(writing) and by :py:func:`~atomworks.io.parser.parse` (reading) to persist all
registered :py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase`
subclasses into CIF extra-category blocks and restore them on load.

**Serialization tiers**:

- **0-body** annotations (a single system-wide value): written once in their own CIF
  category, as a single row with no ``idx`` columns.
- **Scalar 1-body** annotations (shape ``(n_atoms,)``): written to ``atom_site``
  via biotite's ``extra_fields`` mechanism, like any regular ``AtomArray``
  annotation.
- **Nonscalar 1-body** annotations (shape ``(n_atoms, k)``) and
  **2-body** annotations (``AnnotationList2D``): written as CIF extra-categories
  using a shared ``idx0, [idx1,] val0, val1, …`` format.  For nonscalar 1-body
  ``idx0 = [0, 1, …, n_atoms-1]`` (one index column); for 2-body ``idx0`` and
  ``idx1`` hold the pair atom indices (two index columns). Raises
  :py:exc:`ValueError` for 2-body annotations if the row count exceeds
  ``_MAX_MULTI_BODY_ROWS``
"""

import warnings
from typing import Literal

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
    AtomArrayPlusStack,
    as_atom_array_plus,
)
from atomworks.io.utils.selection import get_annotation, get_annotation_categories
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase

# Maximum number of rows to save for multi-body AtomArrayPlus annotations (e.g. pairwise distances)
_MAX_MULTI_BODY_ROWS = 10_000

# This is a closed compatibility list for reading legacy mask fields; new
# annotations must persist values.
_LEGACY_MASK_FIELDS = {
    "sequence": "mask_sequence_1_residue",
    "coordinate": "mask_coordinate_1_atom",
    "index": "mask_index_1_residue",
}

# ---------------------------------------------------------------------------
# Low-level CIF dict helpers
# ---------------------------------------------------------------------------


def _atleast_2d_last(arr: np.ndarray) -> np.ndarray:
    """Same as np.atleast_2d but adds a dimension at the end instead of the beginning."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return np.expand_dims(arr, axis=-1)
    else:
        return arr


def _idxs_values_to_cif_dict(idxs: np.ndarray, values: np.ndarray) -> dict:
    """Convert index and value arrays to a CIF category dict."""
    idxs = _atleast_2d_last(idxs)  # (N, n_body)
    values = _atleast_2d_last(values)  # (N, n_values)
    return {f"idx{i}": idxs[:, i] for i in range(idxs.shape[1])} | {
        f"val{i}": values[:, i] for i in range(values.shape[1])
    }


def _annotation_to_cif_idxs_values(
    annotation: np.ndarray | AnnotationList2D,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(idxs, values)`` from an annotation for CIF serialization.

    For :py:class:`~atomworks.io.utils.atom_array_plus.AnnotationList2D` (2-body):
    returns ``(pairs, values)``. Scalars use one row with no index columns.

    For ``np.ndarray`` (nonscalar 1-body): returns ``(arange(n_atoms), annotation)``.
    """
    if isinstance(annotation, AnnotationList2D):
        return annotation.pairs, annotation.values
    elif annotation.ndim == 0:
        return np.empty((1, 0), dtype=int), annotation
    else:
        return np.arange(n_atoms), annotation


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_standard_annotation_categories(
    cif_block: pdbx.CIFBlock,
) -> dict[str, type[StandardAnnotationBase]]:
    """Scan a CIF block and return found StandardAnnotation categories.

    Returns:
        Dict mapping category names to their registered StandardAnnotation.
    """
    classes_by_field = {
        field: annotation_cls
        for annotation_cls in STANDARD_ANNOTATIONS
        for field in (annotation_cls.full_name, *annotation_cls.aliases)
    }
    return {key: classes_by_field[key] for key in cif_block if key in classes_by_field}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _get_available_standard_annotations(
    atom_array: AtomArray,
    n_body: int | Literal["all"] = "all",
) -> list[type[StandardAnnotationBase]]:
    """Return registered StandardAnnotations present on ``atom_array``."""
    return [
        sa_cls
        for sa_cls in STANDARD_ANNOTATIONS
        if (n_body == "all" or sa_cls.n_body == n_body) and sa_cls.has_annotation(atom_array)
    ]


def _canonicalize_standard_annotation_aliases(atom_array: AtomArray) -> None:
    """Materialize canonical fields on a serialization copy without deleting aliases."""
    for annotation_cls in _get_available_standard_annotations(atom_array):
        if annotation_cls.full_name not in get_annotation_categories(atom_array, n_body=annotation_cls.n_body):
            annotation_cls.set_annotation(atom_array, annotation_cls.annotation(atom_array, default="raise"))


def _get_scalar_1body_sa_names(atom_array: AtomArray) -> list[str]:
    """Return names of 1-body StandardAnnotations on ``atom_array`` that should go into ``atom_site``.

    Nonscalar 1-body annotation values are excluded (they go into a CIF
    extra-category instead).

    Args:
        atom_array: The structure whose annotations are inspected.

    Returns:
        Annotation names suitable for passing as ``extra_fields`` to
        :py:func:`~biotite.structure.io.pdbx.set_structure`.
    """
    names: list[str] = []
    for sa_cls in _get_available_standard_annotations(atom_array, n_body=1):
        annotation = sa_cls.annotation(atom_array)
        if annotation.ndim == 1:
            names.append(sa_cls.full_name)
    return names


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_standard_annotation_categories(atom_array: AtomArray) -> dict[str, dict]:
    """Serialize 0-body, nonscalar 1-body and 2-body annotations to CIF extra-category dicts.

    Scalar 1-body annotations are excluded here; they are written to ``atom_site``
    via :py:func:`_get_scalar_1body_sa_names` instead.

    Args:
        atom_array: The structure to read annotations from.

    Returns:
        Dict suitable for passing as ``extra_categories`` to
        :py:func:`~atomworks.io.utils.io_utils._write_categories_to_block`.

    Raises:
        ValueError: If a 2-body annotation has more than ``MAX_MULTI_BODY_ROWS`` pairs.
    """
    extra_categories: dict[str, dict] = {}
    n_atoms = atom_array.array_length()

    for annotation_cls in _get_available_standard_annotations(atom_array):
        annotation = annotation_cls.annotation(atom_array)

        if annotation_cls.n_body == 1 and annotation.ndim == 1:
            continue  # Scalar 1-body annotations are saved in atom_site extra_fields
        if annotation_cls.n_body == 2 and len(annotation.pairs) > _MAX_MULTI_BODY_ROWS:
            raise ValueError(
                f"StandardAnnotation '{annotation_cls.full_name}' would write {len(annotation.pairs)} rows "
                f"(pairs), exceeding the limit of {_MAX_MULTI_BODY_ROWS}. "
            )

        idxs, values = _annotation_to_cif_idxs_values(annotation, n_atoms)
        extra_categories[annotation_cls.full_name] = _idxs_values_to_cif_dict(idxs, values)

    return extra_categories


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def _deserialize_standard_annotation_cif_category(
    cif_category: pdbx.CIFCategory,
    dtype: np.dtype,
    annotation_cls: type[StandardAnnotationBase],
    atom_array: AtomArray,
    model_num: int | None = None,
) -> np.ndarray | AnnotationList2D | None:
    """Deserialize one CIF extra-category into its annotation value.

    System annotations use their registered body order; otherwise, ``idx*`` columns determine the layout:

    - **0-body annotation** → one scalar value with no index columns.
    - **1 index column** → 1-body: ``val*`` columns are scattered into a
      default-filled array of shape ``(n_atoms[, k])`` using ``idx0`` as the
      atom-index selector.
    - **2 index columns** → 2-body: ``idx0`` and ``idx1`` form pair indices;
      an :py:class:`~atomworks.io.utils.atom_array_plus.AnnotationList2D` is
      constructed and returned.

    When ``model_num`` is provided and a ``model_num`` column exists in the
    category, only rows matching that model are used. System values must agree across a stack.
    """
    # Filter rows by model_num if present in the category
    row_mask = None
    if model_num is not None and "model_num" in cif_category:
        model_col = cif_category["model_num"].as_array(int)
        row_mask = model_col == model_num
        if not row_mask.any():
            return

    idxs = [cif_category[k].as_array(int) for k in sorted(k for k in cif_category if k.startswith("idx"))]
    vals = []
    for key in sorted(k for k in cif_category if k.startswith("val")):
        column = cif_category[key]
        vals.append(column.as_array(str) == "True" if np.issubdtype(dtype, np.bool_) else column.as_array(dtype))
    if not vals:
        raise ValueError("CIF category has no 'val' columns.")

    if row_mask is not None:
        idxs = [idx[row_mask] for idx in idxs]
        vals = [val[row_mask] for val in vals]

    val_array = np.stack(vals, axis=1) if len(vals) > 1 else vals[0]
    n_atoms = atom_array.array_length()

    if annotation_cls.n_body == 0:
        if idxs:
            raise ValueError("0-body CIF categories must not contain index columns.")
        if val_array.ndim != 1 or not len(val_array):
            raise ValueError("0-body CIF categories must contain scalar values.")
        if len(val_array) != 1 and "model_num" not in cif_category:
            raise ValueError("0-body CIF categories must contain exactly one row without model_num.")
        if not np.array_equal(val_array, np.full_like(val_array, val_array[0]), equal_nan=val_array.dtype.kind in "fc"):
            raise ValueError("Conflicting system values across models; select one model with ParseConfig(model=...).")
        return np.asarray(val_array[0], dtype=dtype)
    elif len(idxs) <= 1:
        # 1-body: scatter val columns into a full-length array.
        idx_array = idxs[0] if idxs else np.arange(n_atoms)
        full = np.asarray(annotation_cls.default_annotation(atom_array), dtype=dtype).copy()
        expected_shape = (n_atoms,) if val_array.ndim == 1 else (n_atoms,) + val_array.shape[1:]
        if full.shape != expected_shape:
            raise ValueError(
                f"StandardAnnotation '{annotation_cls.full_name}' default has shape {full.shape}, "
                f"but its CIF values require {expected_shape}."
            )
        full[idx_array] = val_array.astype(dtype)
        return full
    else:
        # 2-body: reconstruct AnnotationList2D from pair index columns.
        pairs = np.stack(idxs, axis=1)
        value = val_array.astype(dtype)
        return AnnotationList2D(n_atoms=n_atoms, pairs=pairs, values=value)


def _handle_legacy_standard_annotations(atom_array: AtomArrayPlus | AtomArrayPlusStack) -> None:
    """Restore legacy masks and normalize integer Index values to booleans.

    Sequence and Coordinate retain saved targets where the legacy mask is active,
    falling back to the structure when targets are absent or sequence targets are
    unspecified. Inactive positions receive defaults. Legacy fields are removed after conversion.
    """
    for annotation_cls in STANDARD_ANNOTATIONS:
        legacy_name = _LEGACY_MASK_FIELDS.get(annotation_cls.name)
        if legacy_name is None:
            continue
        legacy_mask = get_annotation(atom_array, legacy_name, n_body=annotation_cls.n_body)

        if annotation_cls.name == "index":
            canonical = get_annotation(atom_array, annotation_cls.full_name, n_body=1)
            if canonical is not None and np.asarray(canonical).dtype.kind != "b":
                annotation = np.asarray(legacy_mask, dtype=bool) if legacy_mask is not None else canonical != -1
                atom_array.del_annotation(annotation_cls.full_name)
                annotation_cls.set_annotation(atom_array, annotation)
            elif canonical is None and legacy_mask is not None:
                annotation_cls.set_annotation(atom_array, np.asarray(legacy_mask, dtype=bool))
            if legacy_mask is not None:
                atom_array.del_annotation(legacy_name)
            continue

        if legacy_mask is None:
            continue
        legacy_mask = np.asarray(legacy_mask, dtype=bool)
        source = annotation_cls.annotation(atom_array, default=None)
        if source is None:
            ground_truth_value = getattr(annotation_cls, "ground_truth_value", None)
            if ground_truth_value is None:
                raise ValueError(f"Legacy mask restoration is not defined for '{annotation_cls.name}'.")
            source = ground_truth_value(atom_array)
            if source is None:
                raise ValueError(f"Legacy mask restoration is not defined for '{annotation_cls.name}'.")
        elif annotation_cls.name == "sequence":
            missing_target = legacy_mask & (source == annotation_cls.default_value)
            source = np.where(missing_target, annotation_cls.ground_truth_value(atom_array), source)
        annotation = annotation_cls.default_annotation(atom_array)
        annotation = annotation.astype(np.result_type(annotation.dtype, source.dtype))
        if isinstance(atom_array, AtomArrayPlusStack) and source.shape == (
            atom_array.stack_depth(),
            *annotation.shape,
        ):
            if atom_array.stack_depth() == 1:
                annotation[legacy_mask] = source[0, legacy_mask]
                annotation_cls.set_annotation(atom_array, annotation)
            else:
                per_model = np.broadcast_to(annotation, source.shape).copy()
                per_model[:, legacy_mask] = source[:, legacy_mask]
                atom_array.set_per_stack_annotation(annotation_cls.full_name, per_model)
        elif source.shape == annotation.shape:
            annotation[legacy_mask] = source[legacy_mask]
            annotation_cls.set_annotation(atom_array, annotation)
        else:
            raise ValueError(
                f"Cannot restore StandardAnnotation '{annotation_cls.name}' from legacy mask field "
                f"'{legacy_name}': target values have shape {source.shape}, but the annotation "
                f"requires shape {annotation.shape}."
            )
        atom_array.del_annotation(legacy_name)


def _deserialize_standard_annotations(
    atom_array: AtomArray | AtomArrayStack,
    cif_block: pdbx.CIFBlock,
    model_num: int | None = None,
    *,
    restore_legacy: bool = True,
) -> AtomArrayPlus | AtomArrayPlusStack:
    """Reconstruct all StandardAnnotations from a CIF block onto an AtomArray.

    Args:
        atom_array: The AtomArray (or stack) to annotate.
        cif_block: The CIF block containing the serialized annotations.
        model_num: When provided, filters SA category rows to only those
            matching this CIF model ID. Without it, system values must agree across models.
        restore_legacy: Reconstruct annotation values from legacy mask fields and the structure.
            Parsers wait until assembly transforms have produced the final coordinate frame.

    Returns:
        The atom array promoted to :py:class:`AtomArrayPlus` with all found
        StandardAnnotations restored.

    Raises:
        ValueError: If a system annotation is malformed or conflicts across models.
    """
    if isinstance(atom_array, AtomArray):
        atom_array = as_atom_array_plus(atom_array)
    elif isinstance(atom_array, AtomArrayStack):
        atom_array = AtomArrayPlusStack.from_atom_array_stack(atom_array)
    else:
        raise ValueError(f"AtomArray or AtomArrayStack expected, got {type(atom_array)}")

    category_values: dict[type[StandardAnnotationBase], tuple[str, np.ndarray | AnnotationList2D]] = {}
    for key, annotation_cls in _detect_standard_annotation_categories(cif_block).items():
        try:
            value = _deserialize_standard_annotation_cif_category(
                cif_block[key], annotation_cls.storage_dtype(), annotation_cls, atom_array, model_num=model_num
            )
        except (ValueError, KeyError) as e:
            if annotation_cls.n_body == 0:
                raise ValueError(f"Cannot deserialize system annotation '{key}': {e}") from e
            warnings.warn(f"Failed to deserialize StandardAnnotation '{key}': {e}", stacklevel=2)
            continue
        if value is None:
            continue
        existing = annotation_cls.annotation(atom_array, default=None)
        if existing is not None and not annotation_cls._annotations_equal(existing, value):
            raise ValueError(
                f"Conflicting atom_site fields and CIF category '{key}' for StandardAnnotation '{annotation_cls.name}'."
            )
        if annotation_cls in category_values:
            previous_key, previous_value = category_values[annotation_cls]
            if not annotation_cls._annotations_equal(previous_value, value):
                raise ValueError(
                    f"Conflicting CIF categories '{previous_key}' and '{key}' both resolve to "
                    f"StandardAnnotation '{annotation_cls.name}'."
                )
        else:
            category_values[annotation_cls] = (key, value)

    for annotation_cls, (_, value) in category_values.items():
        annotation_cls.set_annotation(atom_array, value)

    if restore_legacy:
        _handle_legacy_standard_annotations(atom_array)

    return atom_array
