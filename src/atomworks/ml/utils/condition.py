"""Utilities for condition annotations on AtomArray objects.

The canonical save/load path is now :py:func:`atomworks.io.utils.io_utils.to_cif_file`
(writing) and :py:func:`atomworks.io.parse` / :py:func:`atomworks.io.utils.io_utils.load_any`
(reading), both of which handle all registered
:py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase` annotations automatically
when ``save_standard_annotations=True`` / ``load_standard_annotations=True``.
"""

import warnings

from biotite.structure import AtomArray

from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
)
from atomworks.ml.conditions import CONDITIONS


def fill_missing_conditions_with_defaults(atom_array: AtomArray) -> AtomArray:
    """Fill missing condition annotations with their defaults.

    .. deprecated::
        Use :py:meth:`~atomworks.io.utils.standard_annotations.base.ConditionAccessor.fill_missing_conditions_with_defaults`
        instead.

    Args:
        atom_array: The AtomArray to fill. Modified in-place and returned.

    Returns:
        The atom array with all missing condition defaults applied.
    """
    warnings.warn(
        "fill_missing_conditions_with_defaults() is deprecated. "
        "Use CONDITIONS.fill_missing_conditions_with_defaults() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CONDITIONS.fill_missing_conditions_with_defaults(atom_array)


def default_annotations_and_conditions_from_registry(
    atom_array: AtomArray,
    annotations: list[str],
) -> AtomArrayPlus:
    """Add annotations with default values using the annotator and standard annotation registries.

    Generates defaults from ``ANNOTATOR_REGISTRY`` and
    :py:data:`~atomworks.io.utils.standard_annotations.base.STANDARD_ANNOTATIONS` registry.
    Ignores annotations without known default generation methods.
    Promotes the input ``atom_array`` to an :py:class:`~atomworks.io.utils.atom_array_plus.AtomArrayPlus`
    in order to handle n-body annotations.

    Args:
        atom_array: AtomArray to annotate (modified in-place).
        annotations: Specific annotation names to generate.
    """
    atom_array = as_atom_array_plus(atom_array)

    for name in annotations:
        try:
            ensure_annotations(atom_array, name)
        except ValueError as e:
            warnings.warn(f"Failed to generate annotation '{name}': {e}; skipping.", stacklevel=2)

    return atom_array
