"""Utility functions for writings tests for AtomArray objects."""

__all__ = ["assert_same_atom_array"]

import io
import os
from collections.abc import Iterable

import biotite.structure as struc
import numpy as np
from biotite.database import rcsb
from biotite.structure.atoms import AtomArray, AtomArrayStack

import atomworks.io.utils.bonds as cb
from atomworks.constants import PDB_MIRROR_PATH
from atomworks.io.utils.scatter import apply_group_wise, apply_segment_wise


def get_pdb_path(pdbid: str, mirror_path: str | os.PathLike = PDB_MIRROR_PATH) -> str:
    """Get the local path to a PDB file based on the provided mirror path.

    Args:
        pdbid (str): The PDB ID.
        mirror_path (str | os.PathLike, optional): Path to the PDB mirror directory.
            Defaults to PDB_MIRROR_PATH constant.

    Returns:
        str: The local path to the PDB file.

    Raises:
        FileNotFoundError: If the file does not exist at the expected location or
            if no mirror path is provided.
    """
    if mirror_path is None:
        raise FileNotFoundError("No mirror path provided.")
    pdbid = pdbid.lower()
    filename = os.path.join(mirror_path, pdbid[1:3], f"{pdbid}.cif.gz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
    return filename


def get_pdb_path_or_buffer(pdb_id: str) -> str | io.StringIO:
    """Returns a local file path or an in-memory buffer for a given PDB ID.

    Args:
        pdb_id (str): The PDB identifier of the structure.

    Returns:
        str | io.StringIO: The local file path to the structure file if available,
        otherwise an in-memory buffer containing the fetched file.
    """
    try:
        # ... if file is locally available
        return get_pdb_path(pdb_id)
    except FileNotFoundError:
        # ... otherwise, fetch the file from RCSB
        return rcsb.fetch(pdb_id, format="cif")


def is_same_in_segment(segment_start_stop: np.ndarray, data: np.ndarray, raise_if_false: bool = False) -> np.ndarray:
    """Check if all elements in a segment are the same.

    Args:
        segment_start_stop (np.ndarray): Array of segment start and stop indices (end of segment is inclusive),
            as obtained from `struc.get_residue_starts(... add_exclusive_stop=True)` for example.
        data (np.ndarray): Data array to check for sameness within segments.

    Returns:
        np.ndarray: Boolean array indicating whether all elements in each segment are the same.
    """
    all_same = lambda x: np.all(x == x[0]) if len(x) > 0 else True  # noqa: E731
    is_segment_valid = apply_segment_wise(segment_start_stop, data, all_same)
    return is_segment_valid


def is_same_in_group(groups: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Check if all elements in `data` are the same within each group defined by `groups`.

    Args:
        groups: 1D array of group identifiers, same length as `data`.
        data: 1D array of data values to check for sameness within each group.

    Returns:
        np.ndarray: Boolean array of shape (n_groups,) indicating whether all elements in each group are the same.

    Example:
        >>> groups = np.array([1, 1, 2, 2, 2, 3])
        >>> data = np.array([5, 5, 7, 7, 7, 9])
        >>> is_same_in_group(groups, data)
        array([ True,  True,  True])
        >>> data = np.array([5, 5, 7, 8, 7, 9])
        >>> is_same_in_group(groups, data)
        array([ True, False,  True])
    """
    is_same = lambda x: np.all(x == x[0]) if len(x) > 0 else True  # noqa: E731
    is_group_data_same = apply_group_wise(groups, data, is_same)
    return is_group_data_same


def is_monotonic_increasing(arr: np.ndarray, strict: bool = True) -> bool:
    """Check if an array is monotonically increasing.

    Args:
        arr (np.ndarray): Array to check.
    """
    if strict:
        return np.all(np.diff(arr) > 0)
    else:
        return np.all(np.diff(arr) >= 0)


def is_monotonic_decreasing(arr: np.ndarray, strict: bool = True) -> bool:
    """Check if an array is monotonically decreasing.

    Args:
        arr (np.ndarray): Array to check.
    """
    if strict:
        return np.all(np.diff(arr) < 0)
    else:
        return np.all(np.diff(arr) <= 0)


def is_increasing_without_gaps(arr: np.ndarray, strict: bool = True) -> bool:
    """Check if an array is monotonically increasing without gaps.

    Args:
        arr (np.ndarray): Array to check.
    """
    if strict:
        return np.all(np.diff(arr) == 0)
    else:
        return np.all((np.diff(arr) == 0) | (np.diff(arr) == 1))


def has_annotation(arr: AtomArray, annotation: str | list[str]) -> bool:
    """Check if an AtomArray has an annotation.

    Args:
        arr: AtomArray to check.
        annotation: Annotation(s) to check for.
    """
    existing_annotations = frozenset(["coord", *arr.get_annotation_categories()])
    if isinstance(annotation, str):
        return annotation in existing_annotations
    else:
        return set(annotation).issubset(existing_annotations)


def _get_atom_array_stats(arr: AtomArray) -> str:
    msg = f"AtomArray: {len(arr)} atoms, {struc.get_residue_count(arr)} residues, {struc.get_chain_count(arr)} chains\n"
    msg += f"\t... unique chain ids: {np.unique(arr.chain_id)}\n"
    msg += f"\t... unique residue ids: {np.unique(arr.res_id)}\n"
    msg += f"\t... unique atom types: {np.unique(arr.atom_name)}\n"
    msg += f"\t... unique elements: {np.unique(arr.element)}\n"
    return msg


def assert_same_atom_array(
    arr1: AtomArray | AtomArrayStack,
    arr2: AtomArray | AtomArrayStack,
    compare_coords: bool = True,
    compare_bonds: bool = True,
    compare_box: bool = False,
    annotations_to_compare: list[str] | None = None,
    enforce_order: bool = True,
    compare_bond_order: bool = True,
    _n_mismatches_to_show: int = 5,
) -> None:
    """Asserts that two AtomArray objects are equal.

    Args:
        arr1 (AtomArray): The first AtomArray to compare.
        arr2 (AtomArray): The second AtomArray to compare.
        compare_coords (bool, optional): Whether to compare coordinates. Defaults to True.
        compare_bonds (bool, optional): Whether to compare bonds. Defaults to True.
        compare_box (bool, optional): Whether to compare the box attribute. Defaults to False.
        annotations_to_compare (list[str] | None, optional): List of annotation categories to compare.
            Defaults to None, in which case all annotations are compared.
        enforce_order (bool, optional): Whether to enforce the order of the atoms. Defaults to True.
            NOTE: Enforcing order is much faster; use False only when strictly necessary.
        compare_bond_order (bool, optional): Whether to compare bond order. Defaults to True.
        _n_mismatches_to_show (int, optional): Number of mismatches to show. Defaults to 20.

    WARNING: If AtomArrayStack objects are passed, only the first array is compared.

    Raises:
        AssertionError: If the AtomArray objects are not equal.
    """
    assert isinstance(
        arr1, AtomArray | AtomArrayStack
    ), f"arr1 is not an AtomArray or AtomArrayStack but has type {type(arr1)}"
    assert isinstance(
        arr2, AtomArray | AtomArrayStack
    ), f"arr2 is not an AtomArray or AtomArrayStack but has type {type(arr2)}"

    # Copy both arrays to avoid modifying the original arrays
    arr1 = arr1.copy()
    arr2 = arr2.copy()

    # If the input is a stack, only compare the first array
    if isinstance(arr1, AtomArrayStack):
        arr1 = arr1[0]
    if isinstance(arr2, AtomArrayStack):
        arr2 = arr2[0]

    # Compare lengths, down to the residue-level if necessary
    if len(arr1) != len(arr2):
        msg = "AtomArrays are not the same length!\n"

        # Find the chains that are different lengths
        for chain_id in np.unique(arr1.chain_id):
            arr1_chain_aa = arr1[arr1.chain_id == chain_id]
            arr2_chain_aa = arr2[arr2.chain_id == chain_id]

            if len(arr1_chain_aa) != len(arr2_chain_aa):
                msg += f"+--------- Mismatches for chain: {chain_id} -----------+\n"
                # Find the residues that are different lengths
                for res_id in np.unique(arr1_chain_aa.res_id):
                    arr1_res_aa = arr1_chain_aa[arr1_chain_aa.res_id == res_id]
                    arr2_res_aa = arr2_chain_aa[arr2_chain_aa.res_id == res_id]

                    # Give an informative error message
                    if len(arr1_res_aa) != len(arr2_res_aa):
                        msg += f"Mismatch at residue {res_id}:\n"
                        msg += f"\tarr1: {_get_atom_array_stats(arr1_res_aa)}\n"
                        msg += f"\tarr2: {_get_atom_array_stats(arr2_res_aa)}\n"

        raise AssertionError(msg)

    if compare_coords:
        assert (
            arr1.coord.shape == arr2.coord.shape
        ), f"Coord shapes do not match: {arr1.coord.shape} != {arr2.coord.shape}"
        if not np.allclose(arr1.coord, arr2.coord, equal_nan=True, atol=1e-3, rtol=1e-3):
            mismatch_idxs = np.where(arr1.coord != arr2.coord)[0]
            msg = f"Coords do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.coord[idx]} != {arr2.coord[idx]}\n"
            raise AssertionError(msg)

    # Not returned by `get_annotation_categories`
    if compare_box:
        if arr1._box is None:
            assert arr2._box is None
        else:
            assert np.array_equal(arr1._box, arr2._box, equal_nan=True)

    if annotations_to_compare is None:
        arr1_annotation_keys = arr1.get_annotation_categories()
        arr2_annotation_keys = arr2.get_annotation_categories()
        missing_in_arr1 = set(arr2_annotation_keys) - set(arr1_annotation_keys)
        missing_in_arr2 = set(arr1_annotation_keys) - set(arr2_annotation_keys)
        assert len(missing_in_arr1) == 0, f"Annotations missing in arr1: {missing_in_arr1}"
        assert len(missing_in_arr2) == 0, f"Annotations missing in arr2: {missing_in_arr2}"
        annotations_to_compare = arr1_annotation_keys

    if enforce_order:
        # Compare annotations directly
        for annotation in annotations_to_compare:
            if annotation not in arr1.get_annotation_categories():
                raise AssertionError(f"Annotation {annotation} not in arr1.")
            if annotation not in arr2.get_annotation_categories():
                raise AssertionError(f"Annotation {annotation} not in arr2.")

            # Check if the arrays contain floating-point numbers (in which case, we allow NaN == NaN)
            if np.issubdtype(arr1.get_annotation(annotation).dtype, np.floating) and np.issubdtype(
                arr2.get_annotation(annotation).dtype, np.floating
            ):
                arrays_equal = np.array_equal(
                    arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=True
                )
            else:
                arrays_equal = np.array_equal(
                    arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=False
                )

            if not arrays_equal:
                mismatch_idxs = np.where(arr1.get_annotation(annotation) != arr2.get_annotation(annotation))[0]
                msg = (
                    f"Annotation {annotation} does not match at {len(mismatch_idxs)} indices. First few mismatches:"
                    + "\n"
                )
                for idx in mismatch_idxs[:_n_mismatches_to_show]:
                    msg += (
                        f"\t{idx}: {arr1.get_annotation(annotation)[idx]} != {arr2.get_annotation(annotation)[idx]}\n"
                    )
                    if idx >= _n_mismatches_to_show:
                        break
                raise AssertionError(msg)
    else:
        # Convert annotations to a sorted list of tuples and compare (order-invariant)
        def convert_atom_array_to_sorted_tuples(arr: AtomArray, annotations: list[str]) -> list[tuple]:
            atoms = []
            for atom in arr:
                atom_info = [(annotation, atom.__getattr__(annotation)) for annotation in annotations]
                atoms.append(tuple(sorted(atom_info)))
            return sorted(atoms)

        arr1_atoms_sorted = convert_atom_array_to_sorted_tuples(arr1, annotations_to_compare)
        arr2_atoms_sorted = convert_atom_array_to_sorted_tuples(arr2, annotations_to_compare)

        if arr1_atoms_sorted != arr2_atoms_sorted:
            msg = "Annotations do not match. First few mismatches:\n"
            for idx, atom in enumerate(set(arr1_atoms_sorted).symmetric_difference(set(arr2_atoms_sorted))):
                msg += f"\t{idx}: {atom}\n"
                if idx >= _n_mismatches_to_show:
                    break
            raise AssertionError(msg)

    if compare_bonds:
        assert arr1.bonds is not None, "arr1.bonds is None"
        assert arr2.bonds is not None, "arr2.bonds is None"

        # TODO: Switch to using the `convert_bond_type` method once we upgrade to Biotite v1.4.0
        # structure.bonds.convert_bond_type(struc.bonds.BondType.COORDINATION, struc.bonds.BondType.SINGLE)
        mask_1 = arr1.bonds._bonds[:, 2] == struc.bonds.BondType.COORDINATION
        arr1.bonds._bonds[mask_1, 2] = struc.bonds.BondType.SINGLE

        mask_2 = arr2.bonds._bonds[:, 2] == struc.bonds.BondType.COORDINATION
        arr2.bonds._bonds[mask_2, 2] = struc.bonds.BondType.SINGLE

        if enforce_order:
            # Compare bond arrays directly
            bonds1 = arr1.bonds.as_array()
            bonds2 = arr2.bonds.as_array()
            if not compare_bond_order:
                bonds1 = bonds1[:, :2]
                bonds2 = bonds2[:, :2]
            if not np.array_equal(bonds1, bonds2):
                mismatch_idxs = np.where(bonds1 != bonds2)[0]
                msg = f"Bonds do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
                for idx in mismatch_idxs[:_n_mismatches_to_show]:
                    msg += f"\t{idx}: {bonds1[idx]} != {bonds2[idx]}\n"
                raise AssertionError(msg)
        else:
            # Check graph isomorphisms, labeling nodes with element
            arr1_hash = cb.hash_atom_array(
                arr1, annotations=["element"], bond_order=compare_bond_order, cast_aromatic_bonds_to_same_type=True
            )
            arr2_hash = cb.hash_atom_array(
                arr2, annotations=["element"], bond_order=compare_bond_order, cast_aromatic_bonds_to_same_type=True
            )
            assert arr1_hash == arr2_hash, f"Graph hashes do not match: {arr1_hash} != {arr2_hash}"


def has_ambiguous_annotation_set(
    atom_array: AtomArray,
    annotation_set: Iterable = ["chain_id", "res_id", "res_name", "atom_name", "ins_code"],
) -> bool:
    """Detect whether a given set of annotations is insufficient to distinguish
        all atoms in the input AtomArray.

    For example, this is used to detect ambiguous annotation of the structure
    that would lead to loss of information when writing out the structure.

    This happens because the `struct_conn` category distinguishes bonds
    between different atoms based on the 5-tuple:
        (chain_id, res_id, res_name, atom_name, ins_code)

    To properly save bonds with a structure, make sure that all atoms
    have unique 5-tuples.

    Args:
        atom_array (AtomArray): The atom array to check for ambiguous annotations.
        annotation_set (Iterable, optional): The set of annotations to check for ambiguity.
        Defaults to ["chain_id", "res_id", "res_name", "atom_name", "ins_code"], which is relevant for determining possible bond ambiguity.


    Returns:
        bool: True if ambiguous annotations are detected, False otherwise.
    """
    # Create a structured array with the 5-tuple elements
    identifier_dtypes = [
        (
            annotation,
            atom_array.get_annotation(annotation).dtype
            if annotation in atom_array.get_annotation_categories()
            else "U1",
        )
        for annotation in annotation_set
    ]

    structured_array = np.empty(atom_array.array_length(), dtype=identifier_dtypes)
    for category in identifier_dtypes:
        name, dtype = category
        structured_array[name] = (
            atom_array.get_annotation(name)
            if name in atom_array.get_annotation_categories()
            else ["."] * atom_array.array_length()
        )

    # Use numpy's unique function with return_counts=True to find duplicates
    _, counts = np.unique(structured_array, return_counts=True)

    # If any count is greater than 1, we have ambiguous annotations
    return np.any(counts > 1)
