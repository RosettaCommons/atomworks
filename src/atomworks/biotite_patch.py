"""Collection of monkey patches for biotite.

This module provides patches and extensions to the Biotite library to enhance
functionality and fix version-specific issues.

References:
    `Biotite Documentation <https://www.biotite-python.org/>`_
    `Biotite Structure Module <https://www.biotite-python.org/apidoc/biotite.structure.html>`_
"""

from typing import Callable


import biotite
from biotite.structure import AtomArray, AtomArrayStack, Atom
import numpy as np
import biotite.structure as struc

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


def _add_query_mask_idxs_methods() -> None:
    """Add `query`, `mask`, and `idxs` methods to `AtomArray` and `AtomArrayStack`."""
    from atomworks.io.utils.query import query, mask, idxs

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

        setattr(struc.AtomArray, "__repr__", lean_atom_array_repr)
        setattr(struc.AtomArray, "_repr_original", original_repr)
        setattr(struc.AtomArray, "_repr_lean", True)


def _enable_segment_slices_in_atom_arrays() -> None:
    """Enable `SegmentSlice` in `AtomArray` slicing."""
    from atomworks.io.utils.selection import SegmentSlice

    if not getattr(struc.AtomArray, "_getitem_new", False):
        original_getitem = struc.AtomArray.__getitem__

        def getitem_with_segment_slices(self, item):
            if isinstance(item, SegmentSlice):
                item = item(self)
            return original_getitem(self, item)

        setattr(struc.AtomArray, "__getitem__", getitem_with_segment_slices)
        setattr(struc.AtomArray, "_getitem_original", original_getitem)
        setattr(struc.AtomArray, "_getitem_new", True)


def _update_get_residue_starts() -> None:
    """Improve the `get_residue_starts` function to disambiguate symmetry copies."""
    from atomworks.io.utils.selection import get_residue_starts  # noqa: E402

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
                    f"The atom at index {i} does not share the same annotation categories as the atom at index 0"
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


def _update_pdbx_set_structure() -> None:
    """Improve the `set_structure` function to handle altloc atoms."""

    # fmt: off
    # ruff: noqa
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx.convert import (
        MaskValue,
        _check_non_empty,
        _determine_entity_id,
        _get_or_create_block,
        _repeat,
        _set_inter_residue_bonds,
        _set_intra_residue_bonds,
        unitcell_from_vectors,
    )

    def set_structure(
        pdbx_file,
        array,
        data_block=None,
        include_bonds=False,
        extra_fields=[],
    ):
        """
        Set the ``atom_site`` category with atom information from an
        :class:`AtomArray` or :class:`AtomArrayStack`.

        This will save the coordinates, the mandatory annotation categories
        and the optional annotation categories
        ``atom_id``, ``b_factor``, ``occupancy`` and ``charge``.
        If the atom array (stack) contains the annotation ``'atom_id'``,
        these values will be used for atom numbering instead of continuous
        numbering.
        Furthermore, inter-residue bonds will be written into the
        ``struct_conn`` category.

        Parameters
        ----------
        pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
            The file object.
        array : AtomArray or AtomArrayStack
            The structure to be written. If a stack is given, each array in
            the stack will be in a separate model.
        data_block : str, optional
            The name of the data block.
            Default is the first (and most times only) data block of the
            file.
            If the data block object is passed directly to `pdbx_file`,
            this parameter is ignored.
            If the file is empty, a new data block will be created.
        include_bonds : bool, optional
            If set to true and `array` has associated ``bonds`` , the
            intra-residue bonds will be written into the ``chem_comp_bond``
            category.
            Inter-residue bonds will be written into the ``struct_conn``
            independent of this parameter.
        extra_fields : list of str, optional
            List of additional fields from the ``atom_site`` category
            that should be written into the file.
            Default is an empty list.

        Notes
        -----
        In some cases, the written inter-residue bonds cannot be read again
        due to ambiguity to which atoms the bond refers.
        This is the case, when two equal residues in the same chain have
        the same (or a masked) `res_id`.

        Examples
        --------

        >>> import os.path
        >>> file = CIFFile()
        >>> set_structure(file, atom_array)
        >>> file.write(os.path.join(path_to_directory, "structure.cif"))

        """
        _check_non_empty(array)

        block = _get_or_create_block(pdbx_file, data_block)
        Category = block.subcomponent_class()
        Column = Category.subcomponent_class()

        # Fill PDBx columns from information
        # in structures' attribute arrays as good as possible
        atom_site = Category()
        atom_site["group_PDB"] = np.where(array.hetero, "HETATM", "ATOM")
        atom_site["type_symbol"] = np.copy(array.element)
        atom_site["label_atom_id"] = np.copy(array.atom_name)
        if "altloc_id" in array.get_annotation_categories():
            atom_site["label_alt_id"] = np.copy(array.altloc_id)
        else:
            atom_site["label_alt_id"] = Column(
                # AtomArrays do not store altloc atoms
                np.full(array.array_length(), "."),
                np.full(array.array_length(), MaskValue.INAPPLICABLE),
            )
        atom_site["label_comp_id"] = np.copy(array.res_name)
        atom_site["label_asym_id"] = np.copy(array.chain_id)
        if "chain_entity" in array.get_annotation_categories():
            atom_site["label_entity_id"] = np.copy(array.chain_entity)
        else:
            atom_site["label_entity_id"] = _determine_entity_id(array.chain_id)
        atom_site["label_seq_id"] = np.copy(array.res_id)
        atom_site["pdbx_PDB_ins_code"] = Column(
            np.copy(array.ins_code),
            np.where(array.ins_code == "", MaskValue.INAPPLICABLE, MaskValue.PRESENT),
        )
        atom_site["auth_seq_id"] = atom_site["label_seq_id"]
        atom_site["auth_comp_id"] = atom_site["label_comp_id"]
        atom_site["auth_asym_id"] = atom_site["label_asym_id"]
        atom_site["auth_atom_id"] = atom_site["label_atom_id"]

        annot_categories = array.get_annotation_categories()
        if "atom_id" in annot_categories:
            atom_site["id"] = np.copy(array.atom_id)
        if "b_factor" in annot_categories:
            atom_site["B_iso_or_equiv"] = np.copy(array.b_factor)
        if "occupancy" in annot_categories:
            atom_site["occupancy"] = np.copy(array.occupancy)
        if "charge" in annot_categories:
            atom_site["pdbx_formal_charge"] = Column(
                np.array([f"{int(c):+d}" if c != 0 else "?" for c in array.charge]),
                np.where(array.charge == 0, MaskValue.MISSING, MaskValue.PRESENT),
            )

        # Handle all remaining custom fields
        if len(extra_fields) > 0:
            # ... check to avoid clashes with standard annotations
            _standard_annotations = [
                "hetero",
                "element",
                "atom_name",
                "res_name",
                "chain_id",
                "res_id",
                "ins_code",
                "atom_id",
                "b_factor",
                "occupancy",
                "charge",
            ]
            _reserved_annotation_names = list(atom_site.keys()) + _standard_annotations

            for annot in extra_fields:
                if annot in _reserved_annotation_names:
                    raise ValueError(
                        f"Annotation name '{annot}' is reserved and cannot be written to as extra field. "
                        "Please choose another name."
                    )
                atom_site[annot] = np.copy(array.get_annotation(annot))

        if array.bonds is not None:
            struct_conn = _set_inter_residue_bonds(array, atom_site)
            if struct_conn is not None:
                block["struct_conn"] = struct_conn
            if include_bonds:
                chem_comp_bond = _set_intra_residue_bonds(array, atom_site)
                if chem_comp_bond is not None:
                    block["chem_comp_bond"] = chem_comp_bond

        # In case of a single model handle each coordinate
        # simply like a flattened array
        if isinstance(array, AtomArray) or (isinstance(array, AtomArrayStack) and array.stack_depth() == 1):
            # 'ravel' flattens coord without copy
            # in case of stack with stack_depth = 1
            atom_site["Cartn_x"] = np.copy(np.ravel(array.coord[..., 0]))
            atom_site["Cartn_y"] = np.copy(np.ravel(array.coord[..., 1]))
            atom_site["Cartn_z"] = np.copy(np.ravel(array.coord[..., 2]))
            atom_site["pdbx_PDB_model_num"] = np.ones(array.array_length(), dtype=np.int32)
        # In case of multiple models repeat annotations
        # and use model-specific coordinates
        else:
            atom_site = _repeat(atom_site, array.stack_depth())
            coord = np.reshape(array.coord, (array.stack_depth() * array.array_length(), 3))
            atom_site["Cartn_x"] = np.copy(coord[:, 0])
            atom_site["Cartn_y"] = np.copy(coord[:, 1])
            atom_site["Cartn_z"] = np.copy(coord[:, 2])
            atom_site["pdbx_PDB_model_num"] = np.repeat(
                np.arange(1, array.stack_depth() + 1, dtype=np.int32),
                repeats=array.array_length(),
            )
        if "atom_id" not in annot_categories:
            # Count from 1
            atom_site["id"] = np.arange(1, len(atom_site["group_PDB"]) + 1)
        block["atom_site"] = atom_site

        # Write box into file
        if array.box is not None:
            # PDBx files can only store one box for all models
            # -> Use first box
            if array.box.ndim == 3:
                box = array.box[0]
            else:
                box = array.box
            len_a, len_b, len_c, alpha, beta, gamma = unitcell_from_vectors(box)
            cell = Category()
            cell["length_a"] = len_a
            cell["length_b"] = len_b
            cell["length_c"] = len_c
            cell["angle_alpha"] = np.rad2deg(alpha)
            cell["angle_beta"] = np.rad2deg(beta)
            cell["angle_gamma"] = np.rad2deg(gamma)
            block["cell"] = cell

    pdbx.set_structure = set_structure
    # fmt: on


def _update_coord() -> None:
    """Patch the `coord` function to use isinstance (necessary for AtomArrayPlus)"""

    def coord(item):
        """
        Get the atom coordinates of the given array.

        This may be directly and :class:`Atom`, :class:`AtomArray` or
        :class:`AtomArrayStack` or
        alternatively an (n x 3) or (m x n x 3)  :class:`ndarray`
        containing the coordinates.

        Parameters
        ----------
        item : Atom or AtomArray or AtomArrayStack or ndarray
            Returns the :attr:`coord` attribute, if `item` is an
            :class:`Atom`, :class:`AtomArray` or :class:`AtomArrayStack`.
            Directly returns the input, if `item` is a :class:`ndarray`.

        Returns
        -------
        coord : ndarray
            Atom coordinates.
        """

        if isinstance(item, (Atom, struc.atoms._AtomArrayBase)):
            return item.coord
        elif isinstance(item, np.ndarray):
            return item.astype(np.float32, copy=False)
        else:
            return np.array(item, dtype=np.float32)

    struc.atoms.coord = coord

    # These pyx files also need to be updated with the new version
    struc.celllist.to_coord = coord
    struc.sasa.CellList = struc.celllist.CellList


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
    _update_pdbx_set_structure()
    _update_coord()  # TODO: Remove once biotite 1.5.0 is released (will contain this fix)

    _HAS_BEEN_PATCHED = True
