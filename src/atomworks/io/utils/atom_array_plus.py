import copy
import logging
import numbers
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, Union

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

logger = logging.getLogger("atomworks.io")

T = TypeVar("T")


def _get_sensible_default(dtype: Any) -> Any:
    """Return a sensible default value for the given NumPy dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    elif np.issubdtype(dtype, np.integer):
        return -1
    elif np.issubdtype(dtype, np.bool_):
        return False
    elif np.issubdtype(dtype, np.str_):
        return ""
    else:
        return None


def _to_index_array(index: Any, length: int) -> np.ndarray:
    """
    Convert an index of arbitrary type into an index array.

    Modified from: https://github.com/biotite-dev/biotite/blob/6dc52bc7d5a2f4fc287bffb2fbb615622dd027b4/src/biotite/structure/bonds.pyx

    Args:
        index: The index to convert (can be slice, tuple, array, etc.).
        length: The length of the array being indexed.

    Returns:
        np.ndarray: Array of indices.
    """
    if isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer):
        return index
    else:
        all_indices = np.arange(length, dtype=int)
        return all_indices[index]


def _to_positive_index_array(index_array: np.ndarray, length: int) -> np.ndarray:
    """
    Convert potentially negative values in an array into positive values and check for out-of-bounds values.

    Modified from: https://github.com/biotite-dev/biotite/blob/6dc52bc7d5a2f4fc287bffb2fbb615622dd027b4/src/biotite/structure/bonds.pyx

    Args:
        index_array (np.ndarray): Array of indices (may contain negatives).
        length (int): Length of the array being indexed.

    Returns:
        np.ndarray: Array of positive indices.

    Raises:
        IndexError: If any index is out of bounds.
    """
    index_array = index_array.copy()
    orig_shape = index_array.shape
    index_array = index_array.flatten()

    negatives = index_array < 0

    index_array[negatives] = length + index_array[negatives]

    if (index_array < 0).any():
        raise IndexError(f"Index {np.min(index_array)} is out of range for an array of length {length}")

    if (index_array >= length).any():
        raise IndexError(f"Index {np.max(index_array)} is out of range for an array of length {length}")

    return index_array.reshape(orig_shape)


class AnnotationList2D(Generic[T]):
    """
    Sparse list of pairwise (i, j, value) annotations for an AtomArrayPlus.

    Args:
        n_atoms (int): Number of atoms in the parent AtomArrayPlus.
            Required for dense array conversion.
        pairs (np.ndarray): (N, 2) array of atom index pairs.
        values (np.ndarray): (N,) array of values for each pair.

    Example:
        >>> ann = AnnotationList2D(3, np.array([[0, 1], [1, 2]]), np.array([1.0, 2.0]))
        >>> ann.as_array()
        array([[0, 1, 1.0], [1, 2, 2.0]], dtype=object)
    """

    def __init__(self, n_atoms: int, pairs: np.ndarray, values: np.ndarray):
        # ... ensure we get numpy arrays
        pairs, values = np.asarray(pairs), np.asarray(values)

        # ... ensure pairs is an integer array
        if not np.issubdtype(pairs.dtype, np.integer):
            pairs = pairs.astype(np.int32)

        # ... reshape pairs to (N, 2) array if it's not already
        if pairs.size > 0:
            if len(pairs.shape) == 1:
                pairs = pairs.reshape(-1, 2)
            elif pairs.shape[1] != 2:
                raise ValueError(f"pairs must have shape (N, 2), got {pairs.shape}")
        else:
            pairs = np.empty((0, 2), dtype=np.int32)

        # (Sanity checks)
        assert pairs.shape[1] == 2 and len(pairs) == len(values), "pairs must be (N, 2) array and match values length"

        if pairs.size > 0:
            assert (pairs.min() >= 0) and (pairs.max() < n_atoms), "pairs must contain only valid atom indices"

        self.n_atoms, self.pairs, self.values = n_atoms, pairs, values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_atoms={self.n_atoms}, pairs={self.pairs}, values={self.values})"

    @classmethod
    def from_dense_array(
        cls, dense: np.ndarray, default: Any | None = None, max_pairs: int = 1_000_000
    ) -> "AnnotationList2D":
        """
        Create an AnnotationList2D from a dense (n_atoms, n_atoms) matrix (e.g., distance matrix).

        Only non-default values (as specified by `default`) are stored.
        If default is None, a sensible default is chosen based on the dtype of dense.
        """
        n_atoms = dense.shape[0]

        if default is None:
            default = _get_sensible_default(dense.dtype)

        # Create mask for non-default values, handling NaN values appropriately
        if isinstance(default, float) and np.isnan(default):
            mask = ~np.isnan(dense)
        else:
            mask = dense != default

        # Check if we have too many non-default values (which may become a performance bottleneck)
        n_non_default = np.sum(mask)
        if n_non_default > max_pairs:
            raise ValueError(
                f"Too many non-default values ({n_non_default:,}) exceeds maximum allowed "
                f"({max_pairs:,}). Consider either:\n"
                f"(a) increasing `max_pairs`\n"
                f"(b) using a different `default` value that leads to a sparser matrix\n"
                f"(c) not relying on a 2D annotation list at all."
            )

        # Get indices and values of non-default entries
        i, j = np.where(mask)

        # Create pairs and values arrays
        values = dense[i, j]
        pairs = np.stack([i, j], axis=1)

        # Return sparse annotation list
        return cls(n_atoms, pairs, values)

    @classmethod
    def concatenate(cls, lists: list["AnnotationList2D"], n_atoms_list: list[int]) -> "AnnotationList2D":
        """
        Concatenate multiple AnnotationList2D objects, offsetting indices as needed.

        Args:
            lists (list[AnnotationList2D]): The lists to concatenate.
            n_atoms_list (list[int]): Number of atoms in each corresponding list.

        Returns:
            AnnotationList2D: Concatenated annotation list.
        """
        all_pairs = []
        all_values = []
        offset = 0

        for alist, n_atoms in zip(lists, n_atoms_list, strict=False):
            # ... loop over lists, offsetting indices as needed
            if len(alist.pairs) > 0:
                pairs_offset = alist.pairs + offset
                all_pairs.append(pairs_offset)
                all_values.append(alist.values)
            offset += n_atoms

        # Concatenate pairs and values, or return empty arrays if no pairs
        if all_pairs:
            pairs_cat = np.vstack(all_pairs)
            values_cat = np.concatenate(all_values)
        else:
            pairs_cat = np.empty((0, 2), dtype=int)
            values_cat = np.empty((0,), dtype=lists[0].values.dtype if lists else float)

        return cls(sum(n_atoms_list), pairs_cat, values_cat)

    def as_array(self) -> np.ndarray:
        """
        Returns a copy of the internal sparse representation.

        Returns:
            np.ndarray: Array of shape (N, 3) where each row contains:
                - Column 0: Index of first atom
                - Column 1: Index of second atom
                - Column 2: Annotation value
        """
        arr = np.empty((len(self.pairs), 3), dtype=object)
        arr[:, 0:2] = self.pairs
        arr[:, 2] = self.values
        return arr.copy()

    def as_dense_array(self, default: Any | None = None) -> np.ndarray:
        """
        Converts sparse annotation data to a dense matrix representation.

        If default is None, a sensible default is chosen based on the dtype of self.values.
        """
        if default is None:
            default = _get_sensible_default(self.values.dtype)

        # Initialize dense array with default value
        arr = np.full(
            (self.n_atoms, self.n_atoms),
            default,
            dtype=self.values.dtype,
        )

        # Fill the matrix with the annotation values
        arr[self.pairs[:, 0], self.pairs[:, 1]] = self.values

        return arr

    def add_row(self, i: int, j: int, value: Any) -> None:
        """
        Add a new row (i, j, value) to the annotation list, similar to BondList.add_bond.

        Args:
            i (int): Index of the first atom.
            j (int): Index of the second atom.
            value (Any): The annotation value for this pair.
        """
        self.pairs = np.vstack([self.pairs, [i, j]])
        self.values = np.append(self.values, value)

    def symmetrized(self) -> "AnnotationList2D":
        """
        Return a symmetrized version of this AnnotationList2D (does not modify in place).

        If (i,j) exists but (j,i) doesn't, adds (j,i) with the same value.
        If both exist, raises error if values differ.

        This algorithm uses vectorized operations on the pairs and values arrays to avoid looping over pairs
        and realizing the large arrays in memory.

        Returns:
            Symmetric AnnotationList2D.

        Raises:
            ValueError: If (i,j) and (j,i) both exist with different values.
        """
        if len(self.pairs) == 0:
            return AnnotationList2D(self.n_atoms, self.pairs.copy(), self.values.copy())

        upper_triangle_pairs_mask = self.pairs[:, 0] <= self.pairs[:, 1]  # includes diagonal
        lower_triangle_pairs_mask = self.pairs[:, 0] > self.pairs[:, 1]  # does NOT include diagonal

        upper_triangle_pairs = self.pairs[upper_triangle_pairs_mask]
        lower_triangle_pairs = self.pairs[lower_triangle_pairs_mask]

        upper_triangle_sort_idx = np.argsort(upper_triangle_pairs[:, 0] * self.n_atoms + upper_triangle_pairs[:, 1])
        upper_triangle_pairs = upper_triangle_pairs[upper_triangle_sort_idx]

        lower_triangle_pairs_reverse = lower_triangle_pairs[
            :, ::-1
        ]  # needs to be sorted again because we reversed the order (want to sort by first index now)
        lower_reverse_sort_idx = np.argsort(
            lower_triangle_pairs_reverse[:, 0] * self.n_atoms + lower_triangle_pairs_reverse[:, 1]
        )
        lower_reverse_pairs = lower_triangle_pairs_reverse[lower_reverse_sort_idx]

        upper_values = self.values[upper_triangle_pairs_mask][upper_triangle_sort_idx]
        lower_reverse_values = self.values[lower_triangle_pairs_mask][lower_reverse_sort_idx]

        if len(upper_values) == len(lower_reverse_values) and np.array_equal(upper_triangle_pairs, lower_reverse_pairs):
            # all values should be the same since the pairs are the same
            if np.issubdtype(upper_values.dtype, np.number):
                if not np.allclose(upper_values, lower_reverse_values, equal_nan=True):
                    raise ValueError(f"Asymmetric input values: {upper_values} != {lower_reverse_values}")
            else:
                if not np.array_equal(upper_values, lower_reverse_values):
                    raise ValueError(f"Asymmetric input values: {upper_values} != {lower_reverse_values}")

            # if they are, great! Just return the original list here
            return AnnotationList2D(self.n_atoms, self.pairs.copy(), self.values.copy())

        # ... inputs are not symmetric, but maybe we can rescue

        upper_triangle_pairs_keys = upper_triangle_pairs[:, 0] * self.n_atoms + upper_triangle_pairs[:, 1]
        lower_triangle_pairs_reverse_keys = lower_reverse_pairs[:, 0] * self.n_atoms + lower_reverse_pairs[:, 1]

        # find overlaps using searchsorted
        search_idx = np.searchsorted(
            upper_triangle_pairs_keys, lower_triangle_pairs_reverse_keys, side="left"
        )  # returns where each lower_triangle_pairs_reverse_key would be inserted in upper_triangle_pairs_keys
        valid = search_idx < len(
            upper_triangle_pairs_keys
        )  # possible that there are some search_idx keys that are outside of len_upper_triangle_pairs_keys since
        matches = upper_triangle_pairs_keys[search_idx[valid]] == lower_triangle_pairs_reverse_keys[valid]

        if np.any(matches):  # there are overlaps, make sure they are equal
            overlap_lower_values = lower_reverse_values[valid][matches]
            overlap_upper_values = upper_values[search_idx[valid]][matches]

            if np.issubdtype(overlap_lower_values.dtype, np.number):
                if not np.allclose(overlap_lower_values, overlap_upper_values, equal_nan=True):
                    raise ValueError(f"Asymmetric values: {overlap_lower_values} != {overlap_upper_values}")
            else:
                if not np.array_equal(overlap_lower_values, overlap_upper_values):
                    raise ValueError(f"Asymmetric values: {overlap_lower_values} != {overlap_upper_values}")

            # if there's matches, lets remove the duplicates from one of the lists (in this case, lower triangle)
            # can't just use matches because it is not necessarily the right length (not same length as valid)
            overlap_idx = np.where(valid)[0][matches]
            lower_reverse_pairs = np.delete(lower_reverse_pairs, overlap_idx, axis=0)
            lower_reverse_values = np.delete(lower_reverse_values, overlap_idx, axis=0)

        # add in unique values to one combined upper-triangle list
        combined_pairs = np.vstack([upper_triangle_pairs, lower_reverse_pairs])
        combined_values = np.concatenate([upper_values, lower_reverse_values])

        # symmetrize! (but don't duplicate diagonal elements where i == j)
        diagonal_mask = combined_pairs[:, 0] == combined_pairs[:, 1]
        off_diagonal_pairs = combined_pairs[~diagonal_mask]
        off_diagonal_values = combined_values[~diagonal_mask]

        # Combine diagonal, off-diagonal, and reversed off-diagonal
        symmetrized_pairs = np.vstack([combined_pairs, off_diagonal_pairs[:, ::-1]])
        symmetrized_values = np.concatenate([combined_values, off_diagonal_values])

        return AnnotationList2D(self.n_atoms, symmetrized_pairs, symmetrized_values)

    def __getitem__(self, index: Any) -> "AnnotationList2D | tuple[int, int, T]":
        """
        Subset or index the annotation list.

        Supports:
            - Integer indexing
            - Boolean mask indexing
            - Integer array/fancy indexing (including slices, tuples, etc.)
        """
        # CASE 1: Integer indexing
        if isinstance(index, int | np.integer):
            return (self.pairs[index, 0], self.pairs[index, 1], self.values[index])

        # CASE 2: Boolean mask indexing
        if isinstance(index, np.ndarray) and index.dtype == bool:
            mask = index
            offsets = np.cumsum(~mask.astype(bool), dtype=np.int32)

            keep = mask[self.pairs[:, 0]] & mask[self.pairs[:, 1]]

            pairs_kept = self.pairs[keep]
            values_kept = self.values[keep]

            if len(pairs_kept) == 0:
                return AnnotationList2D(
                    np.sum(mask), np.empty((0, 2), dtype=int), np.empty((0,), dtype=self.values.dtype)
                )

            new_pairs = np.stack(
                [pairs_kept[:, 0] - offsets[pairs_kept[:, 0]], pairs_kept[:, 1] - offsets[pairs_kept[:, 1]]], axis=1
            )

            return AnnotationList2D(np.sum(mask), new_pairs, values_kept)

        # CASE 3: Anything else (slice, tuple, fancy index, etc.)
        idx = _to_index_array(index, self.n_atoms)
        idx = _to_positive_index_array(idx, self.n_atoms)

        inverse_index = np.full(self.n_atoms, -1, dtype=int)
        inverse_index[idx] = np.arange(len(idx))

        keep = (inverse_index[self.pairs[:, 0]] != -1) & (inverse_index[self.pairs[:, 1]] != -1)

        pairs_kept = self.pairs[keep]
        values_kept = self.values[keep]

        if len(pairs_kept) == 0:
            return AnnotationList2D(len(idx), np.empty((0, 2), dtype=int), np.empty((0,), dtype=self.values.dtype))

        new_pairs = np.stack([inverse_index[pairs_kept[:, 0]], inverse_index[pairs_kept[:, 1]]], axis=1)

        return AnnotationList2D(len(idx), new_pairs, values_kept)

    def __len__(self) -> int:
        return len(self.values)


class _AtomArrayPlusBase:
    """
    Mixin for AtomArrayPlus and AtomArrayPlusStack to support 2D (pairwise) annotations.

    Provides methods for setting, getting, copying, and comparing 2D annotations.
    """

    _annot_2d: dict[str, "AnnotationList2D"]

    def __init__(self, *args, **kwargs) -> None:
        self._annot_2d: dict[str, AnnotationList2D] = {}

    def set_annotation_2d(self, name: str, pairs: Sequence[Sequence[int]], values: Sequence[Any]) -> None:
        """
        Set (create or replace) a 2D annotation.

        Args:
            name (str): Name of the annotation.
            pairs (Sequence[Sequence[int]]): List of (i, j) pairs.
            values (Sequence[Any]): List of values for each pair.
        """
        if not isinstance(pairs, np.ndarray) or not np.issubdtype(pairs.dtype, np.integer):
            pairs = np.array(pairs, dtype=np.int32)
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._annot_2d[name] = AnnotationList2D(self.array_length(), pairs, values)

    def get_annotation_2d(self, name: str) -> "AnnotationList2D":
        """Return a 2D annotation (AnnotationList2D)."""
        if name not in self._annot_2d:
            raise ValueError(f"2D annotation category '{name}' does not exist")
        return self._annot_2d[name]

    def get_annotation_2d_categories(self) -> list[str]:
        """Return a list of all 2D annotation names (categories)."""
        return list(self._annot_2d.keys())

    def del_annotation_2d(self, name: str) -> None:
        """Remove a 2D annotation category.

        Args:
            name: The 2D annotation category to remove.
        """
        if name in self._annot_2d:
            del self._annot_2d[name]

    def _copy_2d_annotations(self, clone: Any) -> None:
        """Deep copy 2D annotations to the clone."""
        clone._annot_2d = copy.deepcopy(self._annot_2d)

    def __eq_2d_annotations__(self, other: Any, equal_nan: bool = True) -> bool:
        """
        Check if the 2D annotations of this object are equal to another.

        Args:
            other: The object to compare with.
            equal_nan (bool): Whether to treat NaNs as equal.

        Returns:
            bool: True if 2D annotations are equal, False otherwise.
        """
        if set(self._annot_2d.keys()) != set(other._annot_2d.keys()):
            return False
        for key in self._annot_2d:
            if not np.array_equal(self._annot_2d[key].pairs, other._annot_2d[key].pairs):
                return False
            # Always pass equal_nan argument for .values comparison
            if not np.array_equal(self._annot_2d[key].values, other._annot_2d[key].values, equal_nan=equal_nan):
                return False
        return True

    def __getitem_2d_annotations__(self, item: Any) -> dict[str, "AnnotationList2D"]:
        """Subset all 2D annotations using the given item/index."""
        return {k: v[item] for k, v in self._annot_2d.items()}


class AtomArrayPlus(_AtomArrayPlusBase, AtomArray):
    """
    Extension of AtomArray supporting arbitrary 2D (pairwise) annotations.

    2D annotations are stored as sparse lists of (i, j, value), accessible via set/get_annotation_2d methods.
    Automatically filters 2D annotations on slicing, following BondList semantics.
    """

    def __init__(self, *args, **kwargs) -> None:
        AtomArray.__init__(self, *args, **kwargs)
        _AtomArrayPlusBase.__init__(self)

    @classmethod
    def from_atom_array(cls, atom_array: AtomArray) -> "AtomArrayPlus":
        """
        Create an AtomArrayPlus from an existing AtomArray, copying all data.

        Args:
            atom_array (AtomArray): The AtomArray to convert.

        Returns:
            AtomArrayPlus: The new AtomArrayPlus instance.
        """
        obj = cls(len(atom_array))
        vars(obj).update(vars(atom_array))
        obj._annot_2d = {}
        return obj

    def as_atom_array(self) -> AtomArray:
        """
        Convert the AtomArrayPlus object back to an AtomArray object (removes 2D annotations).

        Returns:
            AtomArray: A copy of this object as a plain AtomArray.
        """
        atom_array = AtomArray.__copy_create__(self)
        AtomArray.__copy_fill__(self, atom_array)
        return atom_array

    def __getitem__(self, item: Any) -> Union["AtomArrayPlus", Any]:
        """
        Slice the AtomArrayPlus object, filtering 2D annotations as well (similar in spirit to how slicing an AtomArray also slices the BondList).

        Args:
            item: The index or slice to apply.

        Returns:
            AtomArrayPlus or Atom: The sliced object.
        """
        # Slice the AtomArray as usual
        result = super().__getitem__(item)

        # If the result is a single Atom, just return it (by definition, we will have no 2D annotations if we have a single atom)
        if not isinstance(result, AtomArray):
            return result

        # Otherwise, create a new AtomArrayPlus and filter 2D annotations
        new_obj = self.__class__.__new__(self.__class__)

        # Copy AtomArray internals
        for attr in self.__dict__:
            if attr != "_annot_2d":
                setattr(new_obj, attr, getattr(result, attr))
        new_obj._annot_2d = self.__getitem_2d_annotations__(item)
        return new_obj

    def __copy_create__(self) -> "AtomArrayPlus":
        """
        Create a new, empty AtomArrayPlus of the same length as this one.

        Returns:
            AtomArrayPlus: A new instance with the same length.
        """
        return AtomArrayPlus(self.array_length())

    def __copy_fill__(self, clone: "AtomArrayPlus") -> None:
        """
        Fill the clone with all data from this instance, including a deep copy of 2D annotations.

        Args:
            clone (AtomArrayPlus): The freshly instantiated copy to fill.
        """
        super().__copy_fill__(clone)
        self._copy_2d_annotations(clone)

    def equal_annotations(self, item: Any, equal_nan: bool = True) -> bool:
        """
        Check if the annotations of this AtomArrayPlus are equal to another AtomArrayPlus.

        Args:
            item: The AtomArrayPlus to compare with.
            equal_nan (bool): Whether to count NaN values as equal.

        Returns:
            bool: True if the annotations are equal, False otherwise.
        """
        return super().equal_annotations(item, equal_nan=equal_nan) and self.__eq_2d_annotations__(
            item, equal_nan=equal_nan
        )


class AtomArrayPlusStack(_AtomArrayPlusBase, AtomArrayStack):
    """
    A collection of multiple AtomArrayPlus instances with support for 2D annotations and per-stack 1D annotations.

    The per-stack annotations system is designed for model-specific information like B-factors,
    occupancy, or any custom data that varies between models in an ensemble or trajectory.

    Args:
        depth (int): Number of models in the stack.
        length (int): Number of atoms in each model.
    """

    def __init__(self, depth: int, length: int) -> None:
        AtomArrayStack.__init__(self, depth, length)
        _AtomArrayPlusBase.__init__(self)
        self._annot_per_stack = {}

    @classmethod
    def from_atom_array_stack(cls, atom_array_stack: AtomArrayStack) -> "AtomArrayPlusStack":
        """
        Create an AtomArrayPlusStack from an AtomArrayStack.

        Args:
            atom_array_stack: The AtomArrayStack to convert.

        Returns:
            AtomArrayPlusStack: The new stack with copied data from the original stack.
        """
        new_stack = cls(atom_array_stack.stack_depth(), atom_array_stack.array_length())
        # ... copy all attributes from the original atom array stack
        vars(new_stack).update(vars(atom_array_stack))
        # ... initialize empty 2D annotation dictionary
        new_stack._annot_2d = {}
        # ... initialize empty per-stack annotation dictionary
        new_stack._annot_per_stack = {}

        return new_stack

    def as_atom_array_stack(self) -> AtomArrayStack:
        """Convert the AtomArrayPlusStack back to an AtomArrayStack object (removes 2D annotations and per-stack annotations)."""
        atom_array_stack = AtomArrayStack.__copy_create__(self)
        AtomArrayStack.__copy_fill__(self, atom_array_stack)
        return atom_array_stack

    def get_array(self, index: int) -> AtomArrayPlus:
        """Obtain the AtomArrayPlus instance of the stack at the specified index."""
        array = AtomArrayPlus.from_atom_array(super().get_array(index))
        # Copy 2D annotations
        for name, ann in self._annot_2d.items():
            array._annot_2d[name] = ann

        # Apply per-stack annotations for this index
        for name, annot in self._annot_per_stack.items():
            # Get the annotation for this specific stack index and set it on the array
            array.set_annotation(name, annot[index])

        return array

    def get_per_stack_annotation(self, name: str) -> np.ndarray:
        """Get a per-stack 1D annotation array with shape (stack_depth, array_length)."""
        if name not in self._annot_per_stack:
            raise ValueError(f"Per-stack annotation category '{name}' does not exist")
        return self._annot_per_stack[name]

    def set_per_stack_annotation(self, name: str, array: np.ndarray) -> None:
        """
        Set a per-stack 1D annotation with shape (stack_depth, array_length).

        Args:
            name: The name of the annotation.
            array: Array with shape (stack_depth, array_length).

        Raises:
            ValueError: If the array shape is incorrect.
        """
        array = np.asarray(array)
        if array.shape != (self.stack_depth(), self.array_length()):
            raise ValueError(
                f"Expected array shape ({self.stack_depth()}, {self.array_length()}), but got {array.shape}"
            )
        self._annot_per_stack[name] = array

    def get_per_stack_annotation_categories(self) -> list[str]:
        """Get all per-stack 1D annotation category names."""
        return list(self._annot_per_stack.keys())

    def to_per_stack_annotation(self, name: str) -> None:
        """Convert a normal 1D annotation to a per-stack annotation.

        Creates a per-stack annotation where each model initially has identical
        values from the original annotation. After conversion, the values can be
        modified independently for each model.

        Args:
            name: The annotation category to convert.

        Raises:
            ValueError: If the annotation doesn't exist.
        """
        if name not in self._annot:
            raise ValueError(f"Annotation category '{name}' does not exist")

        # Create a per-stack version by repeating the annotation for each model
        shared_annot = self._annot[name]
        per_stack_annot = np.tile(shared_annot, (self.stack_depth(), 1))

        # Set as per-stack annotation and remove original
        self._annot_per_stack[name] = per_stack_annot
        del self._annot[name]

    def from_per_stack_annotation(self, name: str) -> None:
        """Convert a per-stack 1D annotation back to a normal shared annotation.

        Args:
            name: The per-stack annotation category to convert.

        Raises:
            ValueError: If the per-stack annotation doesn't exist or if the values differ across models.
        """
        if name not in self._annot_per_stack:
            raise ValueError(f"Per-stack annotation category '{name}' does not exist")

        per_stack_annot = self._annot_per_stack[name]
        first_model = per_stack_annot[0]

        # Check if values differ across models
        for i in range(1, self.stack_depth()):
            if not np.array_equal(per_stack_annot[i], first_model, equal_nan=True):
                raise ValueError(f"Cannot convert '{name}' to a shared annotation: values differ across models.")

        # Set as shared annotation and remove per-stack version
        self.set_annotation(name, first_model)
        del self._annot_per_stack[name]

    def __getitem__(self, index: Any) -> Union["AtomArrayPlusStack", "AtomArrayPlus", Any]:
        """Slice the AtomArrayPlusStack, filtering 2D annotations and per-stack annotations."""
        result = super().__getitem__(index)

        # Case 1: Integer index (e.g., stack[5]) -> returns a single AtomArrayPlus model
        # Case 2: Tuple with integer first index (e.g., stack[5, 0:10]) -> returns AtomArrayPlus or Atom
        # Both cases are handled correctly by our overridden get_array method, which adds per-stack annotations
        if isinstance(index, numbers.Integral) or (isinstance(index, tuple) and isinstance(index[0], numbers.Integral)):
            return result

        # Case 3: Any other indexing (e.g., stack[1:3], stack[:, 0:10], stack[mask])
        # The parent method returns an AtomArrayStack, but we need to convert it to AtomArrayPlusStack
        # and copy over our special annotations
        new_stack = AtomArrayPlusStack.from_atom_array_stack(result)

        # Case 3a: Two-dimensional indexing (stack[:, 0:10]) - slice both stack and atom dimensions
        if isinstance(index, tuple) and len(index) == 2:
            stack_index, atom_index = index

            # Filter 2D annotations according to atom index
            new_stack._annot_2d = self.__getitem_2d_annotations__(atom_index)

            # Filter per-stack annotations using both indices
            for name, annot in self._annot_per_stack.items():
                new_stack._annot_per_stack[name] = annot[stack_index, atom_index]

        # Case 3b: One-dimensional indexing (stack[1:3], stack[mask]) - only stack dimension
        else:
            # Keep all 2D annotations since they apply to all atoms
            new_stack._annot_2d = copy.deepcopy(self._annot_2d)

            # Filter per-stack annotations using only the stack index
            for name, annot in self._annot_per_stack.items():
                new_stack._annot_per_stack[name] = annot[index]

        return new_stack

    def __copy_create__(self) -> "AtomArrayPlusStack":
        """Create a new, empty AtomArrayPlusStack of the same shape as this one."""
        return AtomArrayPlusStack(self.stack_depth(), self.array_length())

    def __copy_fill__(self, clone: "AtomArrayPlusStack") -> None:
        """Fill the clone with all data from this instance, including a deep copy of 2D annotations and per-stack annotations."""
        super().__copy_fill__(clone)
        self._copy_2d_annotations(clone)
        clone._annot_per_stack = {name: np.copy(annot) for name, annot in self._annot_per_stack.items()}

    def equal_annotations(self, item: Any, equal_nan: bool = True) -> bool:
        """Check if the annotations of this AtomArrayPlusStack are equal to another AtomArrayPlusStack."""
        if not super().equal_annotations(item, equal_nan=equal_nan):
            return False

        if not self.__eq_2d_annotations__(item, equal_nan=equal_nan):
            return False

        # Check if per-stack annotations are equal
        if not hasattr(item, "_annot_per_stack"):
            return False

        if set(self._annot_per_stack.keys()) != set(item._annot_per_stack.keys()):
            return False

        # Compare per-stack annotations
        for name in self._annot_per_stack:
            self_annot = self._annot_per_stack[name]
            item_annot = item._annot_per_stack[name]

            # Only use equal_nan for floating point data
            if equal_nan and np.issubdtype(self_annot.dtype, np.floating):
                if not np.array_equal(self_annot, item_annot, equal_nan=True):
                    return False
            else:
                if not np.array_equal(self_annot, item_annot):
                    return False

        return True

    def add_per_stack_annotation(self, name: str, values: np.ndarray) -> None:
        """
        Add a per-stack 1D annotation.

        This is a convenience method that's equivalent to set_per_stack_annotation
        but creates a new per-stack annotation from the given array.

        Args:
            name: The name of the annotation.
            values: Array with shape (stack_depth, array_length).

        Raises:
            ValueError: If the values shape is incorrect.
        """
        self.set_per_stack_annotation(name, values)


def as_atom_array(array: AtomArrayPlus | AtomArray | struc.Atom | list[struc.Atom]) -> AtomArray:
    """
    Ensures that an AtomArrayPlus, AtomArray, list of Atoms, or Atom object is converted to an AtomArray.

    If the input is already an AtomArray, it is returned unchanged.

    Args:
        array: The input to convert, which can be an AtomArrayPlus, AtomArray,
               a single Atom, or a list of Atoms.

    Returns:
        AtomArray: The converted atom array.
    """
    if isinstance(array, AtomArrayPlus):
        # ... AtomArrayPlus (must check before AtomArray, as AtomArray is a subclass of AtomArrayPlus)
        return array.as_atom_array()
    if isinstance(array, AtomArray):
        # ... unchanged
        return array
    elif isinstance(array, struc.Atom):
        # ... single atom
        return struc.array([array])
    elif isinstance(array, list) and all(isinstance(a, struc.Atom) for a in array):
        # ... list of Atoms
        return struc.array(array)


def as_atom_array_plus(array: AtomArrayPlus | AtomArray | struc.Atom | list[struc.Atom]) -> AtomArrayPlus:
    """
    Ensures that an AtomArrayPlus, AtomArray, list of Atoms, or Atom object is converted to an AtomArrayPlus.

    If the input is already an AtomArrayPlus, it is returned unchanged.

    Args:
        array: The input to convert, which can be an AtomArrayPlus, AtomArray,
               a single Atom, or a list of Atoms.

    Returns:
        AtomArrayPlus: The converted atom array.
    """
    if isinstance(array, AtomArrayPlus):
        return array
    return AtomArrayPlus.from_atom_array(as_atom_array(array))


def concatenate_atom_array_plus(arrays: list[AtomArrayPlus | AtomArray | struc.Atom]) -> AtomArrayPlus:
    """
    Concatenate multiple AtomArrayPlus objects, including their 2D annotations.

    Only annotations present in all arrays are concatenated.

    Args:
        arrays (list[AtomArrayPlus]): List of AtomArrayPlus objects to concatenate.

    Returns:
        AtomArrayPlus: Concatenated AtomArrayPlus object.
    """
    # Use standard AtomArray concatenation for base AtomArrays
    #  (Biotite's concatenate is optimized for lists of AtomArrays)
    base_arrays = [as_atom_array(arr) for arr in arrays]
    arr_cat = struc.concatenate(base_arrays)

    # Find common annotation names for 2D annotations
    all_names = [set(arr._annot_2d.keys()) for arr in arrays if hasattr(arr, "_annot_2d")]
    common_names = set.intersection(*all_names) if all_names else set()

    n_atoms_list = [len(arr) for arr in base_arrays]
    annotations_2d = {}

    for name in common_names:
        empty_annotation = lambda array: AnnotationList2D(array.array_length(), [], [])  # noqa: E731
        lists: list[AnnotationList2D] = [
            arr._annot_2d[name] if hasattr(arr, "_annot_2d") else empty_annotation(as_atom_array(arr)) for arr in arrays
        ]
        annotations_2d[name] = AnnotationList2D.concatenate(lists, n_atoms_list)

    # Create new AtomArrayPlus and assign annotations
    result = AtomArrayPlus.from_atom_array(arr_cat)
    for attr in arr_cat.__dict__:
        setattr(result, attr, getattr(arr_cat, attr))
    result._annot_2d = annotations_2d

    return result


def concatenate_any(
    arrays: list[AtomArrayPlus | AtomArray | struc.Atom | list[struc.Atom]],
) -> AtomArrayPlus | AtomArray:
    """
    Concatenate a list of AtomArrayPlus or AtomArray objects.

    If any of the objects are AtomArrayPlus, the result will be an AtomArrayPlus.

    Args:
        arrays (list[Any]): List of arrays to concatenate.

    Returns:
        Any: Concatenated array of the appropriate type.
    """
    if not arrays:
        raise ValueError("Input list is empty.")

    if any(isinstance(arr, AtomArrayPlus) for arr in arrays):
        return concatenate_atom_array_plus(arrays)

    return struc.concatenate(arrays)


def insert_atoms(
    arr: AtomArray | AtomArrayPlus,
    new_atoms: list[AtomArray | AtomArrayPlus | struc.Atom],
    insert_positions: list[int],
) -> AtomArray | AtomArrayPlus:
    """
    Insert atoms into an AtomArray or AtomArrayPlus BEFORE the specified positions.

    Atoms are first concatenated to the end, then the array is sorted so that the new atoms appear at the specified positions.
    The function is robust to both AtomArray and AtomArrayPlus, and preserves 2D annotations and bonds.
    """
    n_atoms_orig = arr.array_length()
    assert isinstance(new_atoms, list) and isinstance(
        new_atoms[0], AtomArray | AtomArrayPlus | struc.Atom
    ), "new_atoms must be a list of AtomArray, AtomArrayPlus, or Atom objects."
    assert len(new_atoms) == len(insert_positions), "Each new atom must have a corresponding insert position."
    arr_all = concatenate_any([arr, *new_atoms])

    # Build a mapping from position to list of new atom indices to insert there
    insert_map = defaultdict(list)
    offset = n_atoms_orig
    for pos, new_arr in zip(insert_positions, new_atoms, strict=True):
        if isinstance(new_arr, struc.Atom):
            insert_map[pos].append(offset)
            offset += 1
        else:
            insert_map[pos].extend(list(range(offset, offset + len(new_arr))))
            offset += len(new_arr)

    result_indices = []
    orig_idx = 0
    for pos in range(n_atoms_orig + 1):
        # Insert all new atoms scheduled for this position
        result_indices.extend(insert_map.get(pos, []))
        # Insert the original atom, unless we're past the end
        if pos < n_atoms_orig:
            result_indices.append(orig_idx)
            orig_idx += 1
    assert (
        len(result_indices) == arr_all.array_length()
    ), "Result indices must match the length of the concatenated array"

    return arr_all[result_indices]


def stack_atom_array_plus(arrays: list[AtomArrayPlus]) -> AtomArrayPlusStack:
    """
    Create an AtomArrayPlusStack from a list of AtomArrayPlus.

    All atom arrays must have an equal number of atoms and equal annotation arrays (including 2D annotations).
    Arrays must have identical 1D annotations.

    TODO: Optionally, allow differing annotations (which will be converted to per-stack annotations)

    Args:
        arrays: List of AtomArrayPlus objects to stack.

    Returns:
        AtomArrayPlusStack: A stack of the input arrays.

    Raises:
        ValueError: If annotations differ between arrays.
    """
    if not arrays:
        raise ValueError("Cannot stack empty list of arrays")

    ref_array = arrays[0]

    # Check for annotation equality
    for i, array in enumerate(arrays[1:], start=1):
        if not array.equal_annotations(ref_array):
            raise ValueError(
                f"The annotations of the atom array at index {i} are not equal to the annotations "
                f"of the atom array at index 0."
            )

    # Create a stack of AtomArrays...
    atom_arrays = [array.as_atom_array() for array in arrays]
    atom_array_stack = struc.stack(atom_arrays)

    # ... and convert to AtomArrayPlusStack
    array_stack = AtomArrayPlusStack.from_atom_array_stack(atom_array_stack)

    # Add 2D annotations from the first array to the stack (only for those 2D annotations that are present in all arrays)
    for name in ref_array.get_annotation_2d_categories():
        if all(name in array.get_annotation_2d_categories() for array in arrays):
            array_stack._annot_2d[name] = ref_array.get_annotation_2d(name)

    return array_stack


def stack_any(arrays: list[AtomArray | AtomArrayPlus]) -> AtomArrayStack | AtomArrayPlusStack:
    """
    Stack a list of AtomArray or AtomArrayPlus objects.

    If any of the objects are AtomArrayPlus, the result will be an AtomArrayPlusStack.
    All atom arrays must have identical 1D annotations.

    Args:
        arrays: List of AtomArray or AtomArrayPlus objects to stack.

    Returns:
        AtomArrayStack or AtomArrayPlusStack: A stack of the input arrays.
    """
    if any(isinstance(arr, AtomArrayPlus) for arr in arrays):
        return stack_atom_array_plus([as_atom_array_plus(arr) for arr in arrays])
    return struc.stack(arrays)


def as_atom_array_plus_stack(
    arrays: AtomArray
    | AtomArrayPlus
    | AtomArrayStack
    | AtomArrayPlusStack
    | list[AtomArray | AtomArrayPlus | struc.Atom]
    | struc.Atom,
) -> AtomArrayPlusStack:
    """Convert various input types to an AtomArrayPlusStack."""
    # Already the target type
    if isinstance(arrays, AtomArrayPlusStack):
        return arrays

    # Convert from regular stack
    if isinstance(arrays, AtomArrayStack):
        return AtomArrayPlusStack.from_atom_array_stack(arrays)

    # Convert from list
    if isinstance(arrays, list):
        arrays_plus = [as_atom_array_plus(arr) for arr in arrays]
        return stack_atom_array_plus(arrays_plus)

    # Single array (AtomArray, AtomArrayPlus, or Atom) - wrap in list and stack
    return stack_atom_array_plus([as_atom_array_plus(arrays)])
