"""Utility functions for writings tests for AtomArray objects."""

import functools
import io
import os
from collections.abc import Iterable
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.database import rcsb
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.info import standardize_order
from rdkit import Chem

from atomworks.constants import PDB_MIRROR_PATH
from atomworks.io.utils.atom_array import subset_to_first_transformation
from atomworks.io.utils.atom_array_plus import AnnotationList2D, as_atom_array_plus
from atomworks.io.utils.bonds import remap_intra_residue_coordination_bonds
from atomworks.io.utils.scatter import apply_group_wise, apply_segment_wise
from atomworks.io.utils.selection import get_annotation, get_annotation_categories, get_residue_starts
from atomworks.io.utils.sequence import convert_to_one_letter_sequences


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


def _get_atom_array_stats(arr: AtomArray) -> str:
    msg = f"AtomArray: {len(arr)} atoms, {struc.get_residue_count(arr)} residues, {struc.get_chain_count(arr)} chains\n"
    msg += f"\t... unique chain ids: {np.unique(arr.chain_id)}\n"
    msg += f"\t... unique residue ids: {np.unique(arr.res_id)}\n"
    msg += f"\t... unique atom types: {np.unique(arr.atom_name)}\n"
    msg += f"\t... unique elements: {np.unique(arr.element)}\n"
    return msg


def _cast_annotation_to_common_dtype(annot1: AtomArray, annot2: AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Cast annotations to common dtype for comparison.

    Rules:
    - Integer types → int64
    - Float types → float64
    - String types → str_ (no change)
    - Bool types → bool (no change)

    Args:
        annot1: First annotation.
        annot2: Second annotation
        annotation: Annotation name to cast.

    Returns:
        Tuple of casted annotations from annot1 and annot2.
    """

    # Check if both are numeric
    if np.issubdtype(annot1.dtype, np.integer) and np.issubdtype(annot2.dtype, np.integer):
        return annot1.astype(np.int64), annot2.astype(np.int64)
    elif np.issubdtype(annot1.dtype, np.floating) and np.issubdtype(annot2.dtype, np.floating):
        return annot1.astype(np.float64), annot2.astype(np.float64)
    else:
        # Return as-is for strings, bools, etc.
        return annot1, annot2


def _sort_bond_array(bonds: np.ndarray) -> np.ndarray:
    """Sort bond array lexicographically for deterministic comparison."""
    if len(bonds) == 0:
        return bonds

    bonds = bonds.copy()

    # Normalize: smaller index first
    mask = bonds[:, 0] > bonds[:, 1]
    bonds[mask, :2] = bonds[mask, 1::-1]

    # Sort lexicographically
    sort_idx = np.lexsort((bonds[:, 2], bonds[:, 1], bonds[:, 0]))
    return bonds[sort_idx]


def _reorder_by_canonical_graph(atom_array: AtomArray) -> AtomArray:
    """Reorder atoms within residues using canonical graph ranking."""
    # Reorder atoms within all residues by canonical ranking
    reorder_indices = np.arange(len(atom_array))

    # Get residue boundaries
    res_start_stops = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    res_starts, res_stops = res_start_stops[:-1], res_start_stops[1:]

    for start, stop in zip(res_starts, res_stops, strict=False):
        residue = atom_array[start:stop]

        # Skip if no bonds - we cannot define a canonical ordering
        if residue.bonds is None or len(residue.bonds.as_array()) == 0:
            continue

        # Get canonical ordering for this residue
        try:
            # ... create a simplified molecule with only connectivity (no stereochemistry or charge)
            mol_simplified = Chem.RWMol()

            # Add atoms (without charges or stereochemistry, which can be ambiguous)
            for element in residue.element:
                atom = Chem.Atom(element)
                atom.SetNoImplicit(True)  # Don't add implicit hydrogens
                mol_simplified.AddAtom(atom)

            # Add bonds (all as single bonds to ignore bond type differences)
            bond_array = residue.bonds.as_array()
            for bond_idx in range(len(bond_array)):
                atom1, atom2 = bond_array[bond_idx, :2]
                mol_simplified.AddBond(int(atom1), int(atom2), Chem.BondType.SINGLE)

            # Get canonical ordering
            mol_simplified = mol_simplified.GetMol()
            canonical_ranks = Chem.CanonicalRankAtoms(mol_simplified, breakTies=True)
            local_order = np.argsort(canonical_ranks)

            # Update global reorder indices
            reorder_indices[start:stop] = start + local_order
        except Exception:
            # If conversion fails, keep original order for this residue
            continue

    return atom_array[reorder_indices]


def _compare_annotation_arrays(
    annot1: np.ndarray,
    annot2: np.ndarray,
    annotation_name: str,
    cast_to_common_dtype: bool,
    _n_mismatches_to_show: int,
) -> None:
    if cast_to_common_dtype:
        annot1, annot2 = _cast_annotation_to_common_dtype(annot1, annot2)

    # Check if the arrays contain floating-point numbers (in which case, we allow NaN == NaN)
    if np.issubdtype(annot1.dtype, np.floating) and np.issubdtype(annot2.dtype, np.floating):
        arrays_equal = np.array_equal(annot1, annot2, equal_nan=True)
    else:
        arrays_equal = np.array_equal(annot1, annot2, equal_nan=False)

    if not arrays_equal:
        mismatch_idxs = np.where(annot1 != annot2)[0]
        msg = (
            f"Annotation {annotation_name} does not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
        )
        for idx in mismatch_idxs[:_n_mismatches_to_show]:
            msg += f"\t{idx}: {annot1[idx]} != {annot2[idx]}\n"
            if idx >= _n_mismatches_to_show:
                break
        raise AssertionError(msg)


def _assert_same_atom_array(
    arr1: AtomArray,
    arr2: AtomArray,
    compare_coords: bool = True,
    compare_bonds: bool = True,
    compare_box: bool = False,
    annotations_to_compare: list[str] | Literal["arr1"] | None = None,
    enforce_order: bool = True,
    compare_bond_order: bool = True,
    cast_to_common_dtype: bool = False,
    _n_mismatches_to_show: int = 5,
) -> None:
    """Asserts that two AtomArray objects are equal. Does not accept AtomArrayPlus

    Args:
        arr1 (AtomArray): The first AtomArray to compare.
        arr2 (AtomArray): The second AtomArray to compare.
        **kwargs: See documentation in ``assert_same_atom_array``

    Raises:
        AssertionError: If the AtomArray objects are not equal.
    """
    assert isinstance(arr1, AtomArray), f"arr1 is not an AtomArray but has type {type(arr1)}"
    assert isinstance(arr2, AtomArray), f"arr2 is not an AtomArray but has type {type(arr2)}"

    # Copy both arrays to avoid modifying the original arrays
    arr1 = arr1.copy()
    arr2 = arr2.copy()

    # Promote to AtomArrayPlus for unified handling
    arr1 = as_atom_array_plus(arr1)
    arr2 = as_atom_array_plus(arr2)

    # Compare lengths, down to the residue-level if necessary
    if arr1.array_length() != arr2.array_length():
        msg = "AtomArrays are not the same shape!\n"

        # Find the chains that are different lengths
        for chain_id in np.unique(arr1.chain_id):
            arr1_chain_aa = arr1[arr1.chain_id == chain_id]
            arr2_chain_aa = arr2[arr2.chain_id == chain_id]

            if arr1_chain_aa.array_length() != arr2_chain_aa.array_length():
                msg += f"+--------- Mismatches for chain: {chain_id} -----------+\n"
                # Find the residues that are different lengths
                for res_id in np.unique(arr1_chain_aa.res_id):
                    arr1_res_aa = arr1_chain_aa[arr1_chain_aa.res_id == res_id]
                    arr2_res_aa = arr2_chain_aa[arr2_chain_aa.res_id == res_id]

                    # Give an informative error message
                    if arr1_res_aa.array_length() != arr2_res_aa.array_length():
                        msg += f"Mismatch at residue {res_id}:\n"
                        msg += f"\tarr1: {_get_atom_array_stats(arr1_res_aa)}\n"
                        msg += f"\tarr2: {_get_atom_array_stats(arr2_res_aa)}\n"

        raise AssertionError(msg)

    if compare_coords:
        assert (
            arr1.coord.shape == arr2.coord.shape
        ), f"Coord shapes do not match: {arr1.coord.shape} != {arr2.coord.shape}"
        if not np.allclose(arr1.coord, arr2.coord, equal_nan=True, atol=1e-3, rtol=1e-3):
            # Use np.isclose with equal_nan=True to correctly identify mismatches
            # This handles NaN coordinates properly (NaN == NaN with equal_nan=True)
            mismatch_mask = ~np.isclose(arr1.coord, arr2.coord, equal_nan=True, atol=1e-3, rtol=1e-3)
            # Find atoms (rows) with any coordinate component mismatch
            mismatch_atom_idxs = np.where(np.any(mismatch_mask, axis=1))[0]
            msg = f"Coords do not match for {len(mismatch_atom_idxs)} atoms. First few mismatches:" + "\n"
            for idx in mismatch_atom_idxs[:_n_mismatches_to_show]:
                msg += f"\tAtom {idx}: {arr1.coord[idx]} != {arr2.coord[idx]}\n"
            raise AssertionError(msg)

    # Not returned by `get_annotation_categories`
    if compare_box:
        if arr1._box is None:
            assert arr2._box is None
        else:
            assert np.array_equal(arr1._box, arr2._box, equal_nan=True)

    if annotations_to_compare is None or annotations_to_compare == "arr1":
        arr1_annotation_keys = get_annotation_categories(arr1, n_body="all")
        arr2_annotation_keys = get_annotation_categories(arr2, n_body="all")

        if annotations_to_compare is None:
            missing_in_arr1 = set(arr2_annotation_keys) - set(arr1_annotation_keys)
            assert len(missing_in_arr1) == 0, f"Annotations missing in arr1: {missing_in_arr1}"

        missing_in_arr2 = set(arr1_annotation_keys) - set(arr2_annotation_keys)
        assert len(missing_in_arr2) == 0, f"Annotations missing in arr2: {missing_in_arr2}"

        annotations_to_compare = arr1_annotation_keys

    if not enforce_order:
        # Check for unsupported mode: comparing bond order without atom names is ambiguous
        if "atom_name" not in annotations_to_compare and compare_bond_order:
            raise ValueError(
                "Cannot compare bond order with enforce_order=False when atom_name is not being compared. "
                "Either set enforce_order=True, compare_bond_order=False, or include 'atom_name' in annotations_to_compare."
            )

        # If atom_name is being compared, we can trust the ordering - use standard CCD ordering
        if "atom_name" in annotations_to_compare:
            arr1 = arr1[standardize_order(arr1)]
            arr2 = arr2[standardize_order(arr2)]

        # If atom_name is NOT being compared, use canonical graph ordering if bonds available
        elif compare_bonds and arr1.bonds is not None and arr2.bonds is not None:
            # Reorder all residues using canonical graph ranking
            arr1 = _reorder_by_canonical_graph(arr1)
            arr2 = _reorder_by_canonical_graph(arr2)
        else:
            raise ValueError(
                "Cannot compare AtomArrays without enforcing order when atom_name is not being compared and bonds are not available. "
            )

    # Compare annotations directly
    for annotation in annotations_to_compare:
        if annotation not in get_annotation_categories(arr1, n_body="all"):
            raise AssertionError(f"Annotation {annotation} not in arr1.")
        if annotation not in get_annotation_categories(arr2, n_body="all"):
            raise AssertionError(f"Annotation {annotation} not in arr2.")

        comparison_fn = functools.partial(
            _compare_annotation_arrays,
            annotation_name=annotation,
            cast_to_common_dtype=cast_to_common_dtype,
            _n_mismatches_to_show=_n_mismatches_to_show,
        )

        annot1 = get_annotation(arr1, annotation)
        annot2 = get_annotation(arr2, annotation)
        is_pairwise = isinstance(annot1, AnnotationList2D)
        assert is_pairwise == isinstance(
            annot2, AnnotationList2D
        ), f"Annotation {annotation} has different body orders."
        if is_pairwise:
            comparison_fn(annot1.pairs, annot2.pairs)
            comparison_fn(annot1.values, annot2.values)
        else:
            assert annot1.shape == annot2.shape, f"Annotation {annotation} has different shapes."
            comparison_fn(np.atleast_1d(annot1), np.atleast_1d(annot2))

    if compare_bonds:
        arr1_has_bonds = arr1.bonds is not None and len(arr1.bonds.as_array()) > 0
        arr2_has_bonds = arr2.bonds is not None and len(arr2.bonds.as_array()) > 0

        if not arr1_has_bonds and not arr2_has_bonds:
            # Both have no bonds (None or empty) — equivalent
            pass
        elif arr1_has_bonds != arr2_has_bonds:
            raise AssertionError(
                f"Bond mismatch: arr1 has {'bonds' if arr1_has_bonds else 'no bonds'}, "
                f"arr2 has {'bonds' if arr2_has_bonds else 'no bonds'}"
            )
        else:
            # Intra-residue COORDINATION cannot survive CIF round-trip
            arr1 = remap_intra_residue_coordination_bonds(arr1)
            arr2 = remap_intra_residue_coordination_bonds(arr2)

        if arr1_has_bonds and arr2_has_bonds:
            bonds1 = arr1.bonds.as_array()
            bonds2 = arr2.bonds.as_array()
            if not enforce_order:
                # Compare bonds with sorting
                bonds1 = _sort_bond_array(bonds1)
                bonds2 = _sort_bond_array(bonds2)

            if not compare_bond_order:
                bonds1 = bonds1[:, :2]
                bonds2 = bonds2[:, :2]

            if not np.array_equal(bonds1, bonds2):
                mismatch_idxs = np.where(np.any(bonds1 != bonds2, axis=1))[0]
                msg = f"Bonds do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
                for idx in mismatch_idxs[:_n_mismatches_to_show]:
                    msg += f"\t{idx}: {bonds1[idx]} != {bonds2[idx]}\n"
                raise AssertionError(msg)


def assert_same_atom_array_or_stack(
    arr1: AtomArray | AtomArrayStack,
    arr2: AtomArray | AtomArrayStack,
    compare_coords: bool = True,
    compare_bonds: bool = True,
    compare_box: bool = False,
    annotations_to_compare: list[str] | Literal["arr1"] | None = None,
    enforce_order: bool = True,
    compare_bond_order: bool = True,
    cast_to_common_dtype: bool = False,
    _n_mismatches_to_show: int = 5,
) -> None:
    """Asserts that two AtomArray or AtomArrayStack objects are equal.

    Args:
        arr1 (AtomArray): The first AtomArray or AtomArrayStack to compare.
        arr2 (AtomArray): The second AtomArray or AtomArrayStack to compare.
        compare_coords (bool, optional): Whether to compare coordinates. Defaults to True.
        compare_bonds (bool, optional): Whether to compare bonds. Defaults to True.
        compare_box (bool, optional): Whether to compare the box attribute. Defaults to False.
        annotations_to_compare (list[str] | Literal["arr1"] | None, optional):
            List of annotation categories to compare, or "arr1" to compare on all annotations in arr1.
            Defaults to None, in which case all annotations are compared.
        enforce_order (bool, optional): Whether to enforce the order of the atoms. Defaults to True.
            NOTE: Enforcing order is much faster; use False only when strictly necessary.
        compare_bond_order (bool, optional): Whether to compare bond order. Defaults to True.
        cast_to_common_dtype (bool, optional): Whether to cast numeric annotations to a common
            dtype before comparison. Useful for tests where dtype preservation is not guaranteed
            (e.g., CIF roundtrips). Defaults to False.
        _n_mismatches_to_show (int, optional): Number of mismatches to show. Defaults to 5.

    Raises:
        AssertionError: If the AtomArray or AtomArrayStack objects are not equal.
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

    # Ensure input types match
    if arr1.shape != arr2.shape:
        raise AssertionError(f"arr1 shape {arr1.shape} does not match arr2 shape {arr2.shape}.")

    # Call the appropriate number of single-AtomArray comparison based on the input type
    comparison_kwargs = {
        "compare_coords": compare_coords,
        "compare_bonds": compare_bonds,
        "compare_box": compare_box,
        "annotations_to_compare": annotations_to_compare,
        "enforce_order": enforce_order,
        "compare_bond_order": compare_bond_order,
        "cast_to_common_dtype": cast_to_common_dtype,
        "_n_mismatches_to_show": _n_mismatches_to_show,
    }

    if isinstance(arr1, AtomArray):
        _assert_same_atom_array(arr1, arr2, **comparison_kwargs)
    else:
        for i in range(arr1.stack_depth()):
            _assert_same_atom_array(arr1[i], arr2[i], **comparison_kwargs)


def has_ambiguous_annotation_set(
    atom_array: AtomArray,
    annotation_set: Iterable[str] = ("chain_id", "res_id", "res_name", "atom_name", "ins_code"),
) -> bool:
    """Detect whether a given set of annotations is insufficient to distinguish all atoms.

    Used to detect ambiguous annotations that would lose information on CIF write,
    since ``struct_conn`` distinguishes bonds by the 5-tuple
    ``(chain_id, res_id, res_name, atom_name, ins_code)``.
    """
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
    for name, _dtype in identifier_dtypes:
        structured_array[name] = (
            atom_array.get_annotation(name)
            if name in atom_array.get_annotation_categories()
            else ["."] * atom_array.array_length()
        )

    _, counts = np.unique(structured_array, return_counts=True)
    return np.any(counts > 1)


def assert_same_annotation_cardinality(
    arr1: AtomArray,
    arr2: AtomArray,
    annotations: list[str],
) -> None:
    """Assert annotations have the same number of unique values in both arrays.

    Args:
        arr1: First atom array.
        arr2: Second atom array.
        annotations: List of annotation names to check cardinality for.
    """
    for annot in annotations:
        # Check annotation exists in both arrays
        if annot not in arr1.get_annotation_categories():
            raise AssertionError(f"Annotation '{annot}' not in arr1")
        if annot not in arr2.get_annotation_categories():
            raise AssertionError(f"Annotation '{annot}' not in arr2")

        # Get unique value counts
        n_unique_1 = len(np.unique(arr1.get_annotation(annot)))
        n_unique_2 = len(np.unique(arr2.get_annotation(annot)))

        # Compare cardinality
        if n_unique_1 != n_unique_2:
            raise AssertionError(
                f"Annotation '{annot}' has different cardinality: "
                f"{n_unique_1} unique values in arr1 vs {n_unique_2} in arr2"
            )


def verify_atom_array_chain_info_consistency(
    chain_info: dict[str, Any],
    atom_array: AtomArray,
    verify_sequences: bool = True,
) -> None:
    """Verify that atom array and chain_info are consistent.

    Args:
        chain_info: The chain_info dictionary.
        atom_array: The processed atom array.
        verify_sequences: Whether to verify 1-letter sequences match. Defaults to ``True``.

    Raises:
        AssertionError: If inconsistencies are detected between atom array and chain_info.
    """
    atom_array = subset_to_first_transformation(atom_array)
    chain_info = chain_info.copy()

    # Make residue start mask
    res_start_mask = np.zeros(atom_array.array_length(), dtype=bool)
    res_starts = get_residue_starts(atom_array)
    res_start_mask[res_starts] = True

    chain_ids = np.unique(atom_array.chain_id)
    for chain_id in chain_ids:
        # Check that chain is present
        if chain_id not in chain_info:
            raise AssertionError(f"Chain ID {chain_id} found in atom array but not in chain_info")

        # Get stored res_names from chain_info
        stored_res_names = chain_info[chain_id].get("res_name", [])
        if not stored_res_names:
            continue

        # Get observed res_names from atom array (at residue starts only)
        chain_mask = atom_array.chain_id == chain_id
        chain_res_start_mask = chain_mask & res_start_mask
        observed_res_names = [str(res_name) for res_name in atom_array.res_name[chain_res_start_mask]]

        if sorted(observed_res_names) != sorted(stored_res_names):
            raise AssertionError(
                f"Chain {chain_id}: residue name multiset mismatch — residues may be non-contiguous\n"
                f"  Observed: {observed_res_names}\n"
                f"  Stored:   {stored_res_names}"
            )

        # Compare: must match exactly (this function only runs when add_missing_atoms=True)
        if observed_res_names != stored_res_names:
            raise AssertionError(
                f"Chain {chain_id}: res_name order mismatch\n"
                f"  Observed: {observed_res_names}\n"
                f"  Stored:   {stored_res_names}"
            )

        # Verify 1-letter sequences match (optional)
        if verify_sequences:
            chain_type = chain_info[chain_id].get("chain_type")
            if chain_type:
                # Use utility that returns both sequences in one call
                computed_non_canonical, computed_canonical = convert_to_one_letter_sequences(
                    stored_res_names, chain_type
                )

                stored_canonical = chain_info[chain_id].get("processed_entity_canonical_sequence", "")
                stored_non_canonical = chain_info[chain_id].get("processed_entity_non_canonical_sequence", "")

                if computed_canonical != stored_canonical:
                    raise AssertionError(
                        f"Chain {chain_id}: Canonical sequence mismatch\n"
                        f"  Computed: {computed_canonical}\n"
                        f"  Stored:   {stored_canonical}"
                    )

                if computed_non_canonical != stored_non_canonical:
                    raise AssertionError(
                        f"Chain {chain_id}: Non-canonical sequence mismatch\n"
                        f"  Computed: {computed_non_canonical}\n"
                        f"  Stored:   {stored_non_canonical}"
                    )
