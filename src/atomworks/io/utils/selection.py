"""Utility functions for selecting segments of an AtomArray"""

__all__ = ["annot_start_stop_idxs", "get_annotation", "get_residue_starts"]

import re
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.scatter import get_segments


def annot_start_stop_idxs(
    atom_array: AtomArray | AtomArrayStack, annots: str | list[str], add_exclusive_stop: bool = False
) -> np.ndarray:
    """
    Computes the start and stop indices for segments in an AtomArray where any of the specified annotation(s) change.

    Args:
        - atom_array (AtomArray): The AtomArray to process.
        - annots (str | list[str]): The annotation(s) to consider for determining segment boundaries.
        - add_exclusive_stop (bool): If True, an exclusive stop index (the length of the AtomArray) is added to the result.

    Returns:
        - np.ndarray: An array of start and stop indices for segments where the annotations change.

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

    More robust version of `biotite.structure.residues.get_residue_starts` that also
    differentiates between residues resulting from different transformation ids.

    Backwards compatible with `biotite.structure.residues.get_residue_starts` if the
    `transformation_id` annotation is not present.

    Reference:
        `Biotite residues.py <https://github.com/biotite-dev/biotite/blob/231eefed334e1d3509c1b7cb3f2bfd71d4b0eeb0/src/biotite/structure/residues.py#L35>`_
    """
    _annots_to_check = ["chain_id", "res_name", "res_id", "ins_code", "transformation_id"]
    existing_annots = atom_array.get_annotation_categories()
    annots_to_check = [annot for annot in _annots_to_check if annot in existing_annots]
    return annot_start_stop_idxs(atom_array, annots=annots_to_check, add_exclusive_stop=add_exclusive_stop)


def _validate_n_body_and_type(atom_array: AtomArray | AtomArrayStack, n_body: int, operation: str) -> None:
    """Validate n_body parameter and atom_array type compatibility."""
    if n_body > 1 and not isinstance(atom_array, (AtomArrayPlus | AtomArrayStack)):
        raise ValueError(f"Cannot {operation} with n_body={n_body} on non-AtomArrayPlus!")

    if n_body not in (1, 2):
        raise NotImplementedError(f"Cannot {operation} with n_body={n_body}!")


def get_annotation(
    atom_array: AtomArray | AtomArrayStack, annot: str, n_body: int | None = None, default: Any = None
) -> np.ndarray:
    """Get the annotation for an AtomArray or AtomArrayStack if it exists, otherwise return the default value."""
    if n_body is not None:
        _validate_n_body_and_type(atom_array, n_body, f"get annotation for {annot}")
    else:
        # Auto-detect annotation dimensionality if n_body not specified
        for body in (1, 2):
            if annot in get_annotation_categories(atom_array, n_body=body):
                return get_annotation(atom_array, annot, n_body=body)

    if n_body == 1 and annot in atom_array.get_annotation_categories():
        return atom_array.get_annotation(annot)
    elif n_body == 2 and annot in atom_array.get_annotation_2d_categories():
        return atom_array.get_annotation_2d(annot)

    return default


def get_annotation_categories(atom_array: AtomArray | AtomArrayStack, n_body: int | Literal["all"] = 1) -> list[str]:
    """Get annotation categories for the specified n_body.

    Args:
        atom_array: The AtomArray or AtomArrayStack to query.
        n_body: 1 for 1D annotations, 2 for 2D annotations, or "all" for all available n_body.

    Returns:
        categories: list[str] List of annotation category names.
    """
    # Map n_body to the corresponding method name
    n_body_to_method = {
        1: "get_annotation_categories",
        2: "get_annotation_2d_categories",
    }

    if n_body == "all":
        categories = []
        for method_name in n_body_to_method.values():
            if hasattr(atom_array, method_name):
                categories.extend(getattr(atom_array, method_name)())
        return categories
    elif n_body in n_body_to_method:
        method_name = n_body_to_method[n_body]
        if hasattr(atom_array, method_name):
            return getattr(atom_array, method_name)()
        else:
            return []
    else:
        return []


class SegmentSlice(ABC):
    """
    Abstract base class for slicing segments of an AtomArray or AtomArrayStack.

    Provides functionality analogous to Python's built-in slice object but operates on structural segments
    (e.g., residues or chains indices) rather than individual atom indices. To subclass, implement the
    `_get_segment_bounds` method to return the start and stop indices of the segments.

    For example:
        - to slice residues 0-2: `atom_array[ResIdxSlice(0, 2)]`
        - to slice chains 0-1: `atom_array[ChainIdxSlice(0, 2)]`
        - to slice to the last two residues: `atom_array[ResIdxSlice(-2, None)]`

    Args:
        - start (int | None): The starting segment index. If None, starts from the beginning.
        - stop (int | None): The ending segment index (exclusive). If None, continues to the end.
    """

    def __init__(self, start: int | None = None, stop: int | None = None):
        self.start = start
        self.stop = stop

    @abstractmethod
    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        pass

    def __call__(self, atom_array: AtomArray | AtomArrayStack) -> slice:
        """
        Creates a slice object for the specified segment range in the atom array.

        Args:
            - atom_array (AtomArray | AtomArrayStack): The structure to slice.

        Returns:
            - slice: A slice object that can be used to index the atom array.
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
    """
    Slice atoms by residue indices.

    Allows for selecting ranges of residues using Python slice-like syntax. Each residue is considered
    as a segment, defined by changes in chain_id, res_name, res_id, ins_code, or transformation_id.

    Example:
        >>> atom_array = AtomArray(...)
        >>> res_slice = ResIdxSlice(0, 2)
        >>> sliced_atom_array = atom_array[res_slice]  # <-- returns a new AtomArray with the first two residues
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return get_residue_starts(atom_array, add_exclusive_stop=True)


class ChainIdxSlice(SegmentSlice):
    """
    Slice atoms by chain indices.

    Allows for selecting ranges of chains using Python slice-like syntax. Each chain is considered
    as a segment, defined by changes in the chain_id annotation.

    Example:
        >>> atom_array = AtomArray(...)
        >>> chain_slice = ChainIdxSlice(0, 1)
        >>> sliced_atom_array = atom_array[chain_slice]  # <-- returns a new AtomArray with the first chain
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return struc.get_chain_starts(atom_array, add_exclusive_stop=True)


class AtomSelection:
    """Class that represents a selection of atoms in a molecular structure.

    We can specify a selection by chain_id, res_name, res_id, atom_name, and (optionally) transformation_id.

    For example:
        - If we specify only chain_id, we will select all atoms in that chain (across all transformations)
        - If we specify chain_id and res_name, we will select all atoms in that chain and residue
        - If we specify only atom_name, we will select all atoms with that name, regardless of chain or residue
    """

    def __init__(
        self,
        chain_id: str = "*",
        res_name: str = "*",
        res_id: int | str = "*",
        atom_name: str = "*",
        transformation_id: int | str = "*",
    ):
        self.chain_id = chain_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.res_id = int(res_id) if res_id != "*" else res_id
        self.transformation_id = str(transformation_id)

    def __str__(self) -> str:
        parts = [self.chain_id, self.res_name, str(self.res_id), self.atom_name, str(self.transformation_id)]

        # Remove trailing '*' values
        while parts and parts[-1] == "*":
            parts.pop()

        return "/".join(parts)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            # Convert the string to an AtomSelection for comparison
            other = self.from_str(other)

        if not isinstance(other, AtomSelection):
            return False

        return (
            self.chain_id == other.chain_id
            and self.res_name == other.res_name
            and self.res_id == other.res_id
            and self.atom_name == other.atom_name
            and self.transformation_id == other.transformation_id
        )

    @classmethod
    def from_str(cls, selection_string: str) -> "AtomSelection":
        """Create a new AtomSelection from a selection string.

        Selection strings are of the form: `CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME/TRANSFORMATION_ID`
        We use "*" as a wildcard to select all atoms in a given granularity.

        """
        selection = parse_selection_string(selection_string)
        return cls(
            chain_id=selection.chain_id,
            res_name=selection.res_name,
            res_id=selection.res_id,
            atom_name=selection.atom_name,
            transformation_id=selection.transformation_id,
        )

    @classmethod
    def from_pymol_str(cls, pymol_string: str) -> "AtomSelection":
        """Create a new AtomSelection from a PyMOL string.

        PyMOL strings, found by clicking on an atom or residue, are of the form: CHAIN_ID/RES_NAME`RES_ID/ATOM_NAME
        For example: "A/ASP`37/OD2"

        We introduce "*" as a wildcard to select all atoms in a given granularity.
        """
        selection = parse_pymol_string(pymol_string)
        return cls(
            chain_id=selection.chain_id,
            res_name=selection.res_name,
            res_id=selection.res_id,
            atom_name=selection.atom_name,
        )

    def get_mask(self, atom_array: AtomArray) -> np.ndarray:
        """Create a boolean mask using this AtomSelection on an AtomArray."""
        return get_mask_from_atom_selection(atom_array, self)

    def get_idxs(self, atom_array: AtomArray) -> np.ndarray:
        """Get the indices of atoms selected by this AtomSelection."""
        return np.where(self.get_mask(atom_array))[0]


def parse_selection_string(selection_string: str) -> AtomSelection:
    """Convert a selection string into a AtomSelection dataclass.

    Selection strings are of the form: `CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME/TRANSFORMATION_ID`

    We use "*" as a wildcard to select all atoms in a given granularity.

    Example:
        >>> parse_selection_string("A/ALA/1/CA")
        AtomSelection(chain_id='A', res_name='ALA', res_id=1, atom_name='CA')
        >>> parse_selection_string("*/ALA/*/CB")  # (select all CB atoms in ALA residues)
        AtomSelection(chain_id='*', res_name='ALA', res_id='*', atom_name='CB')
        >>> parse_selection_string("A/ALA/")
        AtomSelection(chain_id='A', res_name='ALA')
        >>> parse_selection_string("A/*/*/*/1")
        AtomSelection(chain_id='A', res_name='*', res_id='*', atom_name='*', transformation_id=1)
    """
    granularity_tiers = ["chain_id", "res_name", "res_id", "atom_name", "transformation_id"]
    values = selection_string.split("/")

    # Create a dictionary with available tiers and values
    selection_dict = {tier: value for tier, value in zip(granularity_tiers, values, strict=False) if value != "*"}

    return AtomSelection(**selection_dict)


def parse_pymol_string(pymol_string: str) -> AtomSelection:
    """Convert a PyMOL selection string into an AtomSelection instance.

    PyMOL selection strings are of the form: CHAIN_ID/RES_NAME`RES_ID/ATOM_NAME
    Wildcards can be used with "*".

    PyMOL selection strings do not support transformation_id.

    Examples:
        >>> parse_pymol_string("A/ASP`37/OD2")
        AtomSelection(chain_id='A', res_name='ASP', res_id=37, atom_name='OD2')
        >>> parse_pymol_string("A/ASP")
        AtomSelection(chain_id='A', res_name='ASP', res_id='*', atom_name='*')
        >>> parse_pymol_string("*/ASP`*/OD2")
        AtomSelection(chain_id='*', res_name='ASP', res_id='*', atom_name='OD2')
    """
    # Replace backtick with slash to standardize the format
    standardized_string = pymol_string.replace("`", "/")
    return parse_selection_string(standardized_string)


def get_mask_from_selection_string(atom_array: AtomArray, selection_string: str) -> np.ndarray:
    """Create a boolean mask from an AtomArray sequence selection string.

    Selection strings are of the form: `CHAIN_ID/RES_NAME/RES_ID/ATOM_NAME/TRANSFORMATION_ID`

    We use "*" as a wildcard to select all atoms in a given granularity.

    Example:
        >>> atom_array = AtomArray(...)
        >>> mask = get_mask_from_selection_string(atom_array, "A/ALA/1/CA")
        [False, True, False, False, ...]
    """
    return get_mask_from_atom_selection(atom_array, parse_selection_string(selection_string))


def get_mask_from_atom_selection(atom_array: AtomArray, atom_selection: AtomSelection) -> np.ndarray:
    """Create a boolean mask from a AtomSelection dataclass."""
    mask = np.ones(atom_array.array_length(), dtype=bool)

    # ... add the masks
    if atom_selection.chain_id and atom_selection.chain_id != "*":
        mask &= atom_array.chain_id == atom_selection.chain_id

    if atom_selection.res_name and atom_selection.res_name != "*":
        mask &= atom_array.res_name == atom_selection.res_name

    if atom_selection.res_id and atom_selection.res_id != "*":
        mask &= atom_array.res_id == atom_selection.res_id

    if atom_selection.atom_name and atom_selection.atom_name != "*":
        mask &= atom_array.atom_name == atom_selection.atom_name

    if atom_selection.transformation_id and atom_selection.transformation_id != "*":
        mask &= atom_array.transformation_id == atom_selection.transformation_id

    if not np.any(mask):
        raise ValueError(f"No atoms found for selection: {atom_selection}")

    return mask


class AtomSelectionStack:
    """Class that represents a stack of AtomSelections.

    Useful for managing multiple selections and applying them to an AtomArrayStack.
    Notably, enables the use of a single selection string to select multiple segments.
    """

    def __init__(self, selections: list[AtomSelection]):
        self.selections = selections

    @classmethod
    def from_contig_string(cls, contig_string: str) -> "AtomSelectionStack":
        # First define a regex that matches the elements of the contig string
        CONTIG_REGEX = re.compile(r"([A-Za-z]+)(\d+)-(\d+)")  # noqa
        selections = []
        for selection in contig_string.replace(" ", "").split(","):
            match = CONTIG_REGEX.match(selection)
            if not match:
                raise ValueError(f"Invalid contig string: {selection}")
            chain_id, start, stop = match.groups()
            # Create a new AtomSelection for each match
            for i in range(int(start), int(stop) + 1):
                # Create a new AtomSelection for each residue in the range
                atom_selection = AtomSelection(chain_id=chain_id, res_id=i)
                selections.append(atom_selection)
        return cls(selections)

    def get_mask(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Create a boolean mask using this AtomSelection on an AtomArray."""
        return reduce(np.logical_or, [selection.get_mask(atom_array) for selection in self.selections])

    def get_center_of_mass(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Get the center of mass of the selected atoms in the AtomArray."""
        mask = self.get_mask(atom_array)
        if not np.any(mask):
            raise ValueError("No atoms selected by the AtomSelectionStack.")

        if isinstance(atom_array, AtomArray):
            return atom_array.coord[mask].mean(axis=0)
        elif isinstance(atom_array, AtomArrayStack):
            return atom_array.coord[:, mask].mean(axis=1)
        else:
            raise ValueError(f"Cannot get center of mass for {type(atom_array)}!")

    def get_principal_components(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Get the principal components of the selected atoms in the AtomArray.

        Returns:
            - np.ndarray: Principal axes (eigenvectors). For AtomArray: (3, 3). For AtomArrayStack: (n_models, 3, 3).
        """
        mask = self.get_mask(atom_array)
        if not np.any(mask):
            raise ValueError("No atoms selected by the AtomSelectionStack.")

        if isinstance(atom_array, AtomArray):
            coords = atom_array.coord[mask]  # (N_atoms, 3)
            coords_centered = coords - coords.mean(axis=0)
            # SVD for principal axes
            _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
            return vh.T  # (3, 3), columns are principal axes
        elif isinstance(atom_array, AtomArrayStack):
            coords = atom_array.coord[:, mask, :]  # (n_models, N_atoms, 3)
            pcs = []
            for model_coords in coords:
                model_centered = model_coords - model_coords.mean(axis=0)
                _, _, vh = np.linalg.svd(model_centered, full_matrices=False)
                pcs.append(vh.T)  # (3, 3)
            return np.stack(pcs, axis=0)  # (n_models, 3, 3)
        else:
            raise ValueError(f"Cannot get principal components for {type(atom_array)}!")
