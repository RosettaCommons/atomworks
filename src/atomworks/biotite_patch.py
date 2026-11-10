"""Collection of monkey patches for biotite.

This module provides patches and extensions to the Biotite library to enhance
functionality and fix version-specific issues.

References:
    `Biotite Documentation <https://www.biotite-python.org/>`_
    `Biotite Structure Module <https://www.biotite-python.org/apidoc/biotite.structure.html>`_
"""

import functools
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
    """Preserve NumPy scalar dtypes and array-valued annotation shapes."""
    original = struc.array

    @functools.wraps(original)
    def array(atoms: list[Atom]) -> AtomArray:
        names = atoms[0]._annot.keys()
        for i, atom in enumerate(atoms):
            if atom._annot.keys() != names:
                raise ValueError(f"The atom at index {i} does not share the annotation categories of atom 0")
        result = AtomArray(len(atoms))
        for name in names:
            values = [atom._annot[name] for atom in atoms]
            if isinstance(values[0], (np.ndarray, np.generic, str)):
                annotation = np.array(values)
            else:
                annotation = np.empty(len(atoms), dtype=type(values[0]))
                annotation[:] = values
            result.set_annotation(name, annotation)
        result.coord = np.array([atom.coord for atom in atoms])
        return result

    struc.array = array


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


def _update_cif_loop_parsing() -> None:
    """Distribute CIF loop tokens by column without a Python operation per value.

    Biotite still normalizes multiline fields and tokenizes quoted values. The flat
    token list temporarily adds one reference per value while columns are sliced out.
    """
    from biotite.structure.io.pdbx import cif

    original = cif.CIFCategory._deserialize_looped

    @functools.wraps(original)
    def deserialize_looped(lines: list[str]) -> dict[str, list[str]]:
        column_names = []
        for line in lines:
            line = line.strip()
            if line[0] != "_":
                break
            column_names.append(line.split(".")[1])

        n_columns = len(column_names)
        if not n_columns or len(set(column_names)) != n_columns:
            return original(lines)

        values = []
        for line in lines[n_columns:]:
            if line[0] == ";" or "'" in line or '"' in line:
                values.extend(cif._split_one_line(line))
            else:
                values.extend(line.split())
        if len(values) % n_columns:
            raise cif.DeserializationError("Category contains columns with different lengths")
        return {name: values[i::n_columns] for i, name in enumerate(column_names)}

    cif.CIFCategory._deserialize_looped = staticmethod(deserialize_looped)


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
    _update_set_inter_residue_bonds()
    _update_cif_loop_parsing()

    _HAS_BEEN_PATCHED = True
