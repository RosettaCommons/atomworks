"""Canonical per-atom entity identifiers."""

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack


def normalize_entity_id(array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Normalize entity IDs to strings, accepting ``label_entity_id`` as an input alias.

    Return the input unchanged when already canonical or when neither annotation exists.
    Otherwise return a copy with only ``entity_id``. Conflicting aliases raise ValueError.
    """
    categories = array.get_annotation_categories()
    has_entity_id = "entity_id" in categories
    has_legacy_alias = "label_entity_id" in categories

    # Avoid copying arrays that need no normalization.
    if not has_entity_id and not has_legacy_alias:
        return array
    if not has_legacy_alias and array.entity_id.dtype.kind == "U":
        return array

    # Resolve the canonical values and validate any duplicate alias.
    source_annotation = "entity_id" if has_entity_id else "label_entity_id"
    entity_ids = array.get_annotation(source_annotation).astype(str)
    if has_entity_id and has_legacy_alias:
        legacy_ids = array.label_entity_id.astype(str)
        if not np.array_equal(entity_ids, legacy_ids):
            raise ValueError("Conflicting 'entity_id' and legacy 'label_entity_id' annotations.")

    # Replace annotations on a copy so the input and its dtypes stay untouched.
    result = array.copy()
    if has_legacy_alias:
        result.del_annotation("label_entity_id")
    if has_entity_id:
        result.del_annotation("entity_id")
    result.set_annotation("entity_id", entity_ids)
    return result
