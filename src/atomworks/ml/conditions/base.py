"""
Condition system which subclasses StandardAnnotation, providing a separate registry for
StandardAnnotations typically provided as conditions to ML models.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils.scatter import safe_scatter
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standard_annotations.base import (
    AnnotationRegistryAccessor,
    StandardAnnotationBase,
    StandardAnnotationMeta,
)

logger = logging.getLogger("atomworks.ml")

__all__ = [
    "CONDITIONS",
    "ConditionBase",
    "ConditionConflictError",
]


class ConditionConflictError(Exception):
    """A ``transfer`` target position is already conditioned to a value conflicting with the incoming one."""


class ConditionMeta(StandardAnnotationMeta):
    """Metaclass for model-input conditions.

    Inherits all validation and registration from :py:class:`StandardAnnotationMeta`.
    Additionally registers concrete subclasses into ``_condition_registry``,
    a subset of ``StandardAnnotationMeta._registry``.
    """

    _condition_registry: ClassVar[dict[str, type[ConditionBase]]] = {}

    def __new__(meta, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):  # noqa: N804
        cls = super().__new__(meta, name, bases, namespace, **kwargs)

        annotation_name = namespace.get("name")
        if annotation_name:
            cls.has_ground_truth = cls.ground_truth_value.__func__ is not ConditionBase.ground_truth_value.__func__
            # Also register in condition-specific registry
            ConditionMeta._condition_registry[annotation_name] = cls

        return cls


class ConditionBase(StandardAnnotationBase, metaclass=ConditionMeta):
    """Abstract base class for all model-input conditions.

    A condition is a :py:class:`StandardAnnotationBase` that is semantically
    a model input (e.g., sequence, coordinates, chain pairing).

    Attributes:
        (required)
        name: The name of the condition.
        n_body: The number of bodies involved in the condition.
        level: The level at which the condition applies.

        (optional)
        is_symmetric: Whether the condition is symmetric. Only
            applies if ``n_body > 1``. Otherwise always ``True``.
        (derived)
        full_name: The full, systematic name of the condition of the form
            ``condition_{name}_{n_body}_{level}``.
        has_ground_truth: Whether the condition overrides :py:meth:`ground_truth_value`.
    """

    _prefix: ClassVar[str] = "condition"
    has_ground_truth: ClassVar[bool] = False

    @classmethod
    def ground_truth_value(cls, atom_array: AtomArray) -> np.ndarray | None:
        """The condition's target read off the structure, or ``None`` if it is not structure-derived."""

    @classmethod
    def set_annotation_from_ground_truth(cls, atom_array: AtomArray, mask: np.ndarray) -> None:
        """Set ground truth at selected atoms and defaults everywhere else, in place.

        ``mask`` must be a boolean array of shape ``(n_atoms,)``, including for
        residue-level conditions. Only one-body conditions with ground truth are
        supported. Values come from the current structure, replacing any saved
        targets; existing aliases are synchronized by :py:meth:`set_annotation`.
        """
        if cls.n_body != 1 or not cls.has_ground_truth:
            raise ValueError(f"{cls.name} must be a one-body condition with a ground-truth value.")
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (atom_array.array_length(),):
            raise ValueError(
                f"Expected a boolean mask of shape ({atom_array.array_length()},), "
                f"got {mask.shape} with dtype {mask.dtype}."
            )

        annotation = cls.default_annotation(atom_array).copy()
        if mask.any():
            ground_truth = cls.ground_truth_value(atom_array)
            if ground_truth is None:
                raise ValueError(f"{cls.name} returned no ground-truth value.")
            annotation[mask] = ground_truth[mask]
        cls.set_annotation(atom_array, annotation)

    @classmethod
    def transfer(
        cls,
        source: AtomArray,
        target: AtomArray,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        *,
        on_mismatch: Literal["overwrite", "warn", "raise"] = "raise",
    ) -> None:
        """Copy this condition from ``source[source_indices]`` onto
        ``target[target_indices]``, pair by pair -- the two index arrays are equal length and aligned.

        Args:
            source: AtomArray to read the condition annotation from.
            target: AtomArray written in place.
            source_indices: Source atoms to read, aligned with ``target_indices``.
            target_indices: Target atoms to write, aligned with ``source_indices``.
            on_mismatch: When a target position is already set to a *different* value -- ``"raise"``
                (default, :py:class:`ConditionConflictError`), ``"warn"`` (log + overwrite), ``"overwrite"``.

        Empty indices are a no-op; the write is atomic (``target`` is untouched if the call raises).
        ``target_indices`` are assumed unique -- duplicates resolve last-write-wins and are not
        conflict-checked.
        """
        source_indices = np.asarray(source_indices)
        target_indices = np.asarray(target_indices)
        if len(source_indices) != len(target_indices):
            raise ValueError(
                f"{cls.full_name}.transfer needs equal-length, aligned source/target indices; "
                f"got {len(source_indices)} and {len(target_indices)}."
            )
        if len(source_indices) == 0:
            return

        incoming = cls.annotation(source, default="generate")[source_indices]
        target_annotation = cls.annotation(target, default="generate")
        if on_mismatch != "overwrite":
            already = cls.mask_from_annotation(target_annotation)[target_indices]
            if already.any() and not np.array_equal(target_annotation[target_indices][already], incoming[already]):
                res_ids = np.unique(target.res_id[target_indices][already]).tolist()
                existing = target_annotation[target_indices][already]
                msg = f"{cls.full_name} conflict at target res_id {res_ids}: {existing!r} vs incoming {incoming[already]!r}"
                if on_mismatch == "raise":
                    raise ConditionConflictError(msg)
                logger.warning(msg)

        # Commit only after everything that can raise has succeeded.
        cls.set_annotation(target, safe_scatter(incoming, target_indices, target_annotation))


class ConditionAccessor(AnnotationRegistryAccessor):
    """Provides dynamic, attribute-based access to all registered conditions.

    Iterates ``ConditionMeta._condition_registry``, which contains only
    :py:class:`ConditionBase` subclasses (not other standard annotations).
    """

    _entry_kind: ClassVar[str] = "condition"

    @property
    def _registry(self) -> dict[str, type[ConditionBase]]:
        return ConditionMeta._condition_registry

    def fill_missing_conditions_with_defaults(self, atom_array: AtomArray) -> AtomArray:
        """Fill all missing condition annotations with their defaults.

        Iterates all registered conditions and generates default annotation values
        for any that are not already present on the AtomArray.

        Args:
            atom_array: The AtomArray to fill. Modified in-place and returned.

        Returns:
            The atom array with all missing condition defaults applied.
        """
        all_annotation_categories = set(get_annotation_categories(atom_array, n_body="all"))
        for condition_cls in self:
            if not any(name in all_annotation_categories for name in (condition_cls.full_name, *condition_cls.aliases)):
                default_annotation = condition_cls.annotation(atom_array, default="generate")
                condition_cls.set_annotation(atom_array, default_annotation)

        return atom_array

    def __repr__(self) -> str:
        return f"Conditions({self.list()})"


# Singleton instances for easy, clean access
CONDITIONS = ConditionAccessor()
