"""Collection of monkey patches for biotite.

This module provides patches and extensions to the Biotite library to enhance
functionality and fix version-specific issues.

References:
    `Biotite Documentation <https://www.biotite-python.org/>`_
    `Biotite Structure Module <https://www.biotite-python.org/apidoc/biotite.structure.html>`_
"""

from collections.abc import Callable

import biotite.structure as struc
import numpy as np
from biotite.structure import Atom, AtomArray, AtomArrayStack

__all__ = [
    "monkey_patch_biotite",
]

_HAS_BEEN_PATCHED = False


def apply_if_version_lt(version: str, min_version: str) -> Callable:
    """Decorator to apply a function only if the given version is less than the given minimal version.

    Args:
        version: Version to check.
        min_version: Minimal semantic version (e.g. "0.38.0"). If the given version is lower, the
            decorated function is called; otherwise, it is a no-op.

    Example:
        @apply_if_version_lt(biotite.__version__, "0.38.0")
        def patch_bug():
            # Patch code here
            ...

    Returns:
        Decorator that conditionally applies the function.
    """
    from functools import wraps

    def version_tuple(version: str) -> tuple[int, ...]:
        # Only consider numeric parts, ignore pre/post-release tags
        return tuple(int(part) for part in version.split(".") if part.isdigit())

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current = version_tuple(version)
            minimum = version_tuple(min_version)
            if current < minimum:
                return func(*args, **kwargs)
            return None

        return wrapper

    return decorator


def _add_transformation_id_to_struct_conn(
    struct_conn: dict[str, np.ndarray],
    atom_array: AtomArray,
    bond_array: np.ndarray,
) -> None:
    """Add transformation_id fields to struct_conn for bond disambiguation.

    Adds ptnr{1,2}_transformation_id custom fields to struct_conn,
    enabling correct bond reconstruction in assemblies.

    Args:
        struct_conn: Dictionary to add transformation_id fields to (modified in-place).
        atom_array: Atom array containing transformation_id annotation.
        bond_array: Bond array with shape (n_bonds, 2+) where columns 0,1 are atom indices.
    """
    if "transformation_id" not in atom_array.get_annotation_categories():
        raise ValueError("transformation_id annotation required to add transformation IDs to struct_conn")

    transformation_id = atom_array.transformation_id
    for i in range(2):
        atom_indices = bond_array[:, i]
        struct_conn[f"ptnr{i+1}_transformation_id"] = transformation_id[atom_indices].astype(str)


def _add_query_mask_idxs_methods() -> None:
    """Add `query`, `mask`, and `idxs` methods to `AtomArray` and `AtomArrayStack`."""
    from atomworks.io.utils.query import idxs, mask, query

    def query_method(self: AtomArray | AtomArrayStack, expr: str) -> AtomArray | AtomArrayStack:
        """
        Query the AtomArray using pandas-like syntax.

        Examples
        --------
        >>> # Using function calls
        >>> array.query("~has_nan_coord() & has_bonds()")

        >>> # Combining with regular attributes
        >>> array.query("has_bonds() & (chain_id == 'A') & (atom_name == 'CA')")
        """
        return query(self, expr)  # type: ignore

    def mask_method(self: AtomArray | AtomArrayStack, expr: str) -> np.ndarray:
        """
        Query the AtomArray using pandas-like syntax and return a boolean mask.
        """
        return mask(self, expr)  # type: ignore

    def idxs_method(self: AtomArray | AtomArrayStack, expr: str) -> np.ndarray:
        """
        Query the AtomArray using pandas-like syntax and return the indices of the matching atoms.
        """
        return idxs(self, expr)  # type: ignore

    struc.AtomArray.query = query_method
    struc.AtomArrayStack.query = query_method
    struc.AtomArray.mask = mask_method
    struc.AtomArrayStack.mask = mask_method
    struc.AtomArray.idxs = idxs_method
    struc.AtomArrayStack.idxs = idxs_method


def _enable_lean_atom_array_repr() -> None:
    """Improve the AtomArray representation to be leaner (only shows at most 20 atoms), for debugging."""
    if not getattr(struc.AtomArray, "_repr_lean", False):
        original_repr = struc.AtomArray.__repr__

        def lean_atom_array_repr(self: struc.AtomArray) -> str:
            """Lean AtomArray representation that only shows at most 20 atoms (first 10 and last 10)."""
            atoms = ""
            n_atoms = self.array_length()
            for i in range(0, n_atoms):
                if len(atoms) == 0:
                    atoms = "\n\t" + self.get_atom(i).__repr__()
                elif i >= 10 and i < (n_atoms - 10):
                    if i == 10:
                        atoms += "\n\t... (" + str(n_atoms - 21) + " not shown) ..."
                    continue
                else:
                    atoms = atoms + ",\n\t" + self.get_atom(i).__repr__()
            return f"AtomArray([{atoms}\n])"

        struc.AtomArray.__repr__ = lean_atom_array_repr
        struc.AtomArray._repr_original = original_repr
        struc.AtomArray._repr_lean = True


def _enable_segment_slices_in_atom_arrays() -> None:
    """Enable `SegmentSlice` in `AtomArray` slicing."""
    from atomworks.io.utils.selection import SegmentSlice

    if not getattr(struc.AtomArray, "_getitem_new", False):
        original_getitem = struc.AtomArray.__getitem__

        def getitem_with_segment_slices(self, item):
            if isinstance(item, SegmentSlice):
                item = item(self)
            return original_getitem(self, item)

        struc.AtomArray.__getitem__ = getitem_with_segment_slices
        struc.AtomArray._getitem_original = original_getitem
        struc.AtomArray._getitem_new = True


def _update_get_residue_starts() -> None:
    """Improve the `get_residue_starts` function to disambiguate symmetry copies."""
    from atomworks.io.utils.selection import get_residue_starts

    struc.get_residue_starts = get_residue_starts

    # Needed to patch other functions from struc.residues
    struc.residues.get_residue_starts = get_residue_starts


def _update_array() -> None:
    """Improve the `array` function to not truncate the datatype of annotations."""

    def array(atoms: list[Atom]) -> AtomArray:
        """Patch of Biotite's `array` function to not truncate the datatype of annotations.

        Args:
            atoms: The atoms to be combined in an array. All atoms must share the same
                annotation categories.

        Returns:
            The listed atoms as array.

        Raises:
            ValueError: If atoms do not share the same annotation categories.

        Examples:
            Creating an atom array from atoms:

            >>> atom1 = Atom([1, 2, 3], chain_id="A")
            >>> atom2 = Atom([2, 3, 4], chain_id="A")
            >>> atom3 = Atom([3, 4, 5], chain_id="B")
            >>> atom_array = array([atom1, atom2, atom3])
            >>> print(atom_array)
                A       0                       1.000    2.000    3.000
                A       0                       2.000    3.000    4.000
                B       0                       3.000    4.000    5.000
        """
        # Check if all atoms have the same annotation names
        # Equality check requires sorting
        names = sorted(atoms[0]._annot.keys())
        for i, atom in enumerate(atoms):
            if sorted(atom._annot.keys()) != names:
                raise ValueError(
                    f"The atom at index {i} does not share the same " f"annotation categories as the atom at index 0"
                )
        array = AtomArray(len(atoms))

        for name in names:
            if hasattr(atoms[0]._annot[name], "dtype"):
                # (Preserve dtype if possible)
                dtype = atoms[0]._annot[name].dtype
            else:
                dtype = type(atoms[0]._annot[name])
            annotation_values = [atom._annot[name] for atom in atoms]
            annotation_values = np.array(annotation_values, dtype=dtype)  # maintain dtype
            array.set_annotation(name, annotation_values)
        array._coord = np.stack([atom.coord for atom in atoms])
        return array

    struc.array = array


def _concatenate() -> None:
    """
    Almost identical to biotite.structure.concatenate, but with a temporary fix
    for the UnboundLocalError issue in concat_atoms assignment.

    Once the PR for the fix is merged and released, we can remove this patch.
    issue: https://github.com/biotite-dev/biotite/issues/855
    biotite PR for the fix: https://github.com/biotite-dev/biotite/pull/857
    """
    from biotite.sequence import Sequence
    from biotite.structure import AtomArray, AtomArrayStack
    from biotite.structure.bonds import BondList

    # Ensure that the atoms can be iterated over multiple times
    def concatenate(atoms: list[AtomArray | AtomArrayStack]) -> AtomArray | AtomArrayStack:
        """
        Concatenate multiple :class:`AtomArray` or :class:`AtomArrayStack` objects into
        a single :class:`AtomArray` or :class:`AtomArrayStack`, respectively.

        Parameters
        ----------
        atoms : iterable object of AtomArray or AtomArrayStack
            The atoms to be concatenated.
            :class:`AtomArray` cannot be mixed with :class:`AtomArrayStack`.

        Returns
        -------
        concatenated_atoms : AtomArray or AtomArrayStack
            The concatenated atoms, i.e. its ``array_length()`` is the sum of the
            ``array_length()`` of the input ``atoms``.

        Notes
        -----
        The following rules apply:

        - Only the annotation categories that exist in all elements are transferred.
        - The box of the first element that has a box is transferred, if any.
        - The bonds of all elements are concatenated, if any element has associated bonds.
        For elements without a :class:`BondList` an empty :class:`BondList` is assumed.

        Examples
        --------

        >>> atoms1 = array(
        ...     [
        ...         Atom([1, 2, 3], res_id=1, atom_name="N"),
        ...         Atom([4, 5, 6], res_id=1, atom_name="CA"),
        ...         Atom([7, 8, 9], res_id=1, atom_name="C"),
        ...     ]
        ... )
        >>> atoms2 = array(
        ...     [
        ...         Atom([1, 2, 3], res_id=2, atom_name="N"),
        ...         Atom([4, 5, 6], res_id=2, atom_name="CA"),
        ...         Atom([7, 8, 9], res_id=2, atom_name="C"),
        ...     ]
        ... )
        >>> print(concatenate([atoms1, atoms2]))
                    1      N                1.000    2.000    3.000
                    1      CA               4.000    5.000    6.000
                    1      C                7.000    8.000    9.000
                    2      N                1.000    2.000    3.000
                    2      CA               4.000    5.000    6.000
                    2      C                7.000    8.000    9.000
        """
        # Ensure that the atoms can be iterated over multiple times
        if not isinstance(atoms, Sequence):
            atoms = list(atoms)

        length = 0
        depth = None
        element_type = None
        common_categories = set(atoms[0].get_annotation_categories())
        box = None
        has_bonds = False
        for element in atoms:
            if element_type is None:
                if not isinstance(element, (AtomArray, AtomArrayStack)):
                    raise TypeError("Expected 'AtomArray' or 'AtomArrayStack', " f"but got '{type(element).__name__}'")
                element_type = type(element)
            else:
                if not isinstance(element, element_type):
                    raise TypeError(f"Cannot concatenate '{type(element).__name__}' " f"with '{element_type.__name__}'")
            length += element.array_length()
            if isinstance(element, AtomArrayStack):
                if depth is None:
                    depth = element.stack_depth()
                else:
                    if element.stack_depth() != depth:
                        raise IndexError("The stack depths are not equal")
            common_categories &= set(element.get_annotation_categories())
            if element.box is not None and box is None:
                box = element.box
            if element.bonds is not None:
                has_bonds = True

        # Use depth to decide: AtomArrayStack sets depth, AtomArray (and subclasses) leave it None.
        # Prefer depth over element_type so subclasses / re-exports of AtomArray still get the correct type.
        if depth is None:
            concat_atoms = AtomArray(length)
        else:
            concat_atoms = AtomArrayStack(depth, length)
        concat_atoms.coord = np.concatenate([element.coord for element in atoms], axis=-2)
        for category in common_categories:
            concat_atoms.set_annotation(
                category,
                np.concatenate([element.get_annotation(category) for element in atoms], axis=0),
            )
        concat_atoms.box = box
        if has_bonds:
            # Concatenate bonds of all elements
            concat_atoms.bonds = BondList.concatenate(
                [element.bonds if element.bonds is not None else BondList(element.array_length()) for element in atoms]
            )

        return concat_atoms

    struc.atoms.concatenate = concatenate
    struc.concatenate = concatenate


def _update_set_inter_residue_bonds() -> None:
    """Patch ``_set_inter_residue_bonds`` to add custom ``transformation_id`` fields to struct_conn."""
    import biotite.structure.io.pdbx.convert as pdbx_convert
    from biotite.structure.io.pdbx.convert import (
        _filter_bonds,
        _filter_canonical_links,
    )
    from biotite.structure.io.pdbx.convert import (
        _set_inter_residue_bonds as _set_inter_residue_bonds_original,
    )

    # ruff: noqa
    def _set_inter_residue_bonds(array, atom_site):
        struct_conn = _set_inter_residue_bonds_original(array, atom_site)
        if struct_conn is None:
            return None
        if "transformation_id" in array.get_annotation_categories():
            bond_array = _filter_bonds(array, "inter")
            bond_array = bond_array[~_filter_canonical_links(array, bond_array)]
            _add_transformation_id_to_struct_conn(struct_conn, array, bond_array)
        return struct_conn

    pdbx_convert._set_inter_residue_bonds = _set_inter_residue_bonds


def monkey_patch_biotite() -> None:
    """Monkey-patch biotite to add query, mask, and idxs methods to AtomArray and AtomArrayStack."""
    global _HAS_BEEN_PATCHED

    if _HAS_BEEN_PATCHED:
        # ... ensure that the monkey patching is only applied once
        return

    _add_query_mask_idxs_methods()
    _enable_lean_atom_array_repr()
    _enable_segment_slices_in_atom_arrays()
    _update_get_residue_starts()
    _update_array()
    _concatenate()
    _update_set_inter_residue_bonds()

    _HAS_BEEN_PATCHED = True
