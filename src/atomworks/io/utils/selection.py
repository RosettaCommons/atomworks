"""Segment slicing and annotation helpers for ``AtomArray`` and ``AtomArrayStack``.

For the atom-selection string languages (the ``sel('...')`` / path-selection DSL and the
pandas-like query language), see :py:mod:`atomworks.io.utils.query`.

Key public objects:
- :py:class:`~atomworks.io.utils.selection.SegmentSlice`
- :py:func:`~atomworks.io.utils.selection.get_residue_starts`

See individual docstrings for usage and examples.
"""

__all__ = ["annot_start_stop_idxs", "get_annotation", "get_residue_starts"]

from abc import ABC, abstractmethod
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

import atomworks.io.utils.atom_array_plus as aap
from atomworks.io.utils.scatter import get_segments


def annot_start_stop_idxs(
    atom_array: AtomArray | AtomArrayStack, annots: str | list[str], add_exclusive_stop: bool = False
) -> np.ndarray:
    """Computes the start and stop indices for segments in an AtomArray where any of the specified annotation(s) change.

    Args:
      atom_array: The AtomArray to process.
      annots: Annotation name or names to define segments.
      add_exclusive_stop: Append an exclusive stop index at the end. Defaults to ``False``.

    Returns:
      1D array of start/stop indices that bound segments.

    Example:
        >>> atom_array = AtomArray(...)
        >>> start_stop_idxs = annot_start_stop_idxs(atom_array, annots="chain_id", add_exclusive_stop=True)
        >>> print(start_stop_idxs)
        [0, 5, 10, 15]
    """
    if atom_array.array_length() == 0:
        return np.array([], dtype=int)

    if isinstance(annots, str):
        annots = [annots]

    annots: list[str]
    annot_data: list[np.ndarray] = [atom_array.get_annotation(annot) for annot in annots]
    start_stop_idxs = get_segments(*annot_data, add_exclusive_stop=add_exclusive_stop)
    return start_stop_idxs


def get_residue_starts(atom_array: AtomArray | AtomArrayStack, add_exclusive_stop: bool = False) -> np.ndarray:
    """Get the start (and optionally stop) indices of residues in an AtomArray.

    This is a more robust version of :py:func:`biotite.structure.residues.get_residue_starts`
    that additionally differentiates residues across different ``transformation_id`` values
    when present. It is backwards compatible if the annotation is absent.

    Args:
      atom_array: Structure to analyze.
      add_exclusive_stop: Append an exclusive stop index at the end. Defaults to ``False``.

    Returns:
      1D array of residue boundary indices.

    References:
      * `Biotite get_residue_starts`_

      .. _Biotite get_residue_starts: https://github.com/biotite-dev/biotite/blob/231eefed334e1d3509c1b7cb3f2bfd71d4b0eeb0/src/biotite/structure/residues.py#L35
    """
    _annots_to_check = ["chain_id", "res_id", "ins_code", "transformation_id"]
    existing_annots = atom_array.get_annotation_categories()
    annots_to_check = [annot for annot in _annots_to_check if annot in existing_annots]
    return annot_start_stop_idxs(atom_array, annots=annots_to_check, add_exclusive_stop=add_exclusive_stop)


def _validate_n_body_and_type(atom_array: AtomArray | AtomArrayStack, n_body: int, operation: str) -> None:
    """Validate ``n_body`` value and structure type.

    Args:
      atom_array: Structure to validate.
      n_body: Annotation dimensionality (0, 1 or 2).
      operation: Description used in error messages.

    Raises:
      ValueError: If ``n_body`` is not 1 but ``atom_array`` cannot hold such annotations.
      NotImplementedError: If ``n_body`` is not 0, 1 or 2.
    """
    if n_body != 1 and not isinstance(atom_array, aap._AtomArrayPlusBase):
        raise ValueError(f"Cannot {operation} with n_body={n_body} on non-AtomArrayPlus!")

    if n_body not in (0, 1, 2):
        raise NotImplementedError(f"Cannot {operation} with n_body={n_body}!")


def get_annotation(
    atom_array: AtomArray | AtomArrayStack, annot: str, n_body: int | None = None, default: Any = None
) -> np.ndarray:
    """Return an annotation array if present, otherwise ``default``.

    If ``n_body`` is ``None``, the dimensionality is auto-detected by probing 1D, then 2D, then 0D
    annotation categories.

    Args:
      atom_array: Structure to query.
      annot: Annotation category name.
      n_body: 0 for a system-wide value, 1 for 1D annotations, 2 for 2D annotations;
        auto-detected if ``None``.
      default: Value to return if the annotation is missing. Defaults to ``None``.

    Returns:
      The requested annotation array or ``default`` if missing.
    """
    if n_body is not None:
        _validate_n_body_and_type(atom_array, n_body, f"get annotation for {annot}")
    else:
        # Auto-detect annotation dimensionality if n_body not specified
        for body in (1, 2, 0):
            if annot in get_annotation_categories(atom_array, n_body=body):
                return get_annotation(atom_array, annot, n_body=body)

    if n_body is not None and annot in get_annotation_categories(atom_array, n_body=n_body):
        if isinstance(atom_array, aap._AtomArrayPlusBase):
            return atom_array.get_annotation(annot, n_body=n_body)
        return atom_array.get_annotation(annot)

    return default


def get_annotation_categories(atom_array: AtomArray | AtomArrayStack, n_body: int | Literal["all"] = 1) -> list[str]:
    """Get annotation categories for the specified n_body.

    Args:
      atom_array: Structure to query.
      n_body: ``0`` for a system-wide value, ``1`` for 1D, ``2`` for 2D, or ``"all"`` for all three.

    Returns:
      Names of available annotation categories for the requested dimensionality.
    """
    if n_body == "all":
        return [name for body in (0, 1, 2) for name in get_annotation_categories(atom_array, n_body=body)]
    if n_body == 1:
        return atom_array.get_annotation_categories()
    if n_body in (0, 2) and isinstance(atom_array, aap._AtomArrayPlusBase):
        return atom_array.get_annotation_categories(n_body=n_body)
    return []


class SegmentSlice(ABC):
    """Abstract base class for slicing segments of an AtomArray or AtomArrayStack.

    Provides functionality analogous to Python's built-in slice object but operates on structural segments
    (e.g., residues or chains indices) rather than individual atom indices. To subclass, implement the
    `_get_segment_bounds` method to return the start and stop indices of the segments.

    For example:
        - to slice residues 0-2: `atom_array[ResIdxSlice(0, 2)]`
        - to slice chains 0-1: `atom_array[ChainIdxSlice(0, 2)]`
        - to slice to the last two residues: `atom_array[ResIdxSlice(-2, None)]`

    Args:
      start: Starting segment index. Defaults to ``None``.
      stop: Exclusive ending segment index. Defaults to ``None``.
    """

    def __init__(self, start: int | None = None, stop: int | None = None):
        self.start = start
        self.stop = stop

    @abstractmethod
    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        pass

    def __call__(self, atom_array: AtomArray | AtomArrayStack) -> slice:
        """Creates a slice object for the specified segment range in the atom array.

        Args:
          atom_array: Structure to slice.

        Returns:
          A Python ``slice`` that can be used to index ``atom_array``.
        """
        seg_bounds = self._get_segment_bounds(atom_array)
        n_segments = len(seg_bounds) - 1
        if n_segments < 0:
            # edge case: empty array
            return slice(0, 0)

        seg_slice = slice(self.start, self.stop)
        start, stop, _ = seg_slice.indices(n_segments)

        return slice(seg_bounds[start], seg_bounds[stop])


class ResIdxSlice(SegmentSlice):
    """Slice atoms by residue indices.

    Residues are segmented by changes in ``chain_id``, ``res_name``, ``res_id``,
    ``ins_code``, or ``transformation_id``.

    Example:
        >>> atom_array = AtomArray(...)
        >>> res_slice = ResIdxSlice(0, 2)
        >>> sliced_atom_array = atom_array[res_slice]  # <-- returns a new AtomArray with the first two residues
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return get_residue_starts(atom_array, add_exclusive_stop=True)


class ChainIdxSlice(SegmentSlice):
    """Slice atoms by chain indices.

    Allows for selecting ranges of chains using Python slice-like syntax. Each chain is considered
    as a segment, defined by changes in the chain_id annotation.

    Example:
        >>> atom_array = AtomArray(...)
        >>> chain_slice = ChainIdxSlice(0, 1)
        >>> sliced_atom_array = atom_array[chain_slice]  # <-- returns a new AtomArray with the first chain
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return struc.get_chain_starts(atom_array, add_exclusive_stop=True)
