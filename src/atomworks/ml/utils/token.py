from collections.abc import Callable, Iterator

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils.sequence import (
    is_glycine,
    is_protein_unknown,
    is_purine,
    is_pyrimidine,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
)


def get_token_starts(array: AtomArray, add_exclusive_stop: bool = False) -> np.ndarray:
    """
    Get indices for an atom array, each indicating the beginning of
    a token.

    Inspired by `biotite.structure.get_residue_starts`.

    A new token starts:
      - If `atomize` is True
      - If either the chain ID, residue ID, insertion code
        or residue name changes from one to the next atom.

    Args:
        array (AtomArray): The atom array to get the token starts from.
        add_exclusive_stop (bool, optional): If True, add an exclusive stop to the token starts for
            the last residue. Defaults to False.

    Returns:
        np.ndarray: An array of indices indicating the beginning of each token.
    """
    if array.array_length() == 0:
        # ... early exit if the array is empty
        return np.array([])

    # These mask are 'true' at indices where the value changes
    if "atomize" in array.get_annotation_categories():
        atomize_positions = array.atomize
    else:
        atomize_positions = np.zeros(array.array_length(), dtype=bool)

    if "chain_iid" in array.get_annotation_categories():
        chain_id_changes = array.chain_iid[1:] != array.chain_iid[:-1]
    else:
        chain_id_changes = array.chain_id[1:] != array.chain_id[:-1]

    res_id_changes = array.res_id[1:] != array.res_id[:-1]
    ins_code_changes = array.ins_code[1:] != array.ins_code[:-1]
    res_name_changes = array.res_name[1:] != array.res_name[:-1]

    # If any of these annotation arrays change, a new residue starts
    residue_change_mask = (
        chain_id_changes | res_id_changes | ins_code_changes | res_name_changes | atomize_positions[1:]
    )

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new residue
    residue_starts = np.where(residue_change_mask)[0] + 1

    # The first residue is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], residue_starts, [array.array_length()]))
    else:
        return np.concatenate(([0], residue_starts))


def get_token_count(array: AtomArray) -> int:
    """
    Returns the number of distinct tokens in the atom array.

    This function counts the number of tokens based on the changes in
    the atom array's annotations. It will match the behavior of
    `biotite.structure.get_residue_count` when the atom array does not
    have the `atomize` annotation or if `atomize` is False for all atoms.

    Returns:
        int: The number of distinct tokens in the atom array.
    """
    return len(get_token_starts(array))


def get_token_masks(array: AtomArray, indices: np.ndarray) -> np.ndarray:
    """
    Get boolean masks indicating the tokens to which the given atom
    indices belong.

    Args:
        - array (AtomArray or AtomArrayStack): The atom array (stack) to determine the residues from.
        - indices (ndarray, dtype=int): An array of indices indicating the atoms to get the corresponding
          residues for. Negative indices are not allowed.

    Returns:
        - residues_masks (ndarray, dtype=bool): A 2D boolean array where each row corresponds to a given index
          in `indices`. Each row masks the atoms that belong to the same residue as the atom at the given index.

    See also:
        - get_residue_masks_for
        - get_token_starts
        - get_token_starts_for
        - get_token_positions
    """

    starts = get_token_starts(array, add_exclusive_stop=True)
    return struc.segments.get_segment_masks(starts, indices)


def get_token_starts_for(array: AtomArray, indices: np.ndarray) -> np.ndarray:
    """
    Retrieves the indices that point to the start of the token for each specified atom index.

    This function is useful for identifying the beginning of the token associated with each atom in the
    provided indices. It is particularly relevant in contexts where atoms are grouped into tokens based
    on their annotations.

    Args:
        - array (AtomArray or AtomArrayStack): The atom array (or stack) from which to determine the
          residue starts.
        - indices (ndarray, dtype=int, shape=(k,)): An array of atom indices for which the corresponding
          residue starts are to be retrieved. Negative indices are not permitted.

    Returns:
        - start_indices (ndarray, dtype=int, shape=(k,)): An array of indices pointing to the start of
          the tokens corresponding to the input `indices`.

    See also:
        - get_residue_starts_for
        - get_token_starts
        - get_token_masks
        - get_token_positions
    """
    starts = get_token_starts(array, add_exclusive_stop=True)
    return struc.segments.get_segment_starts_for(starts, indices)


def token_iter(array: AtomArray) -> Iterator[AtomArray]:
    """Returns an iterator over the tokens in the atom array.

    This will match `biotite.structure.residue_iter` in the case
    where the atom array does not have the `atomize` annotation,
    or if `atomize` is False everywhere.
    """
    # The exclusive stop is appended to the residue starts
    starts = get_token_starts(array, add_exclusive_stop=True)
    return struc.segments.segment_iter(array, starts)


def spread_token_wise(array: AtomArray, input_data: np.ndarray, token_starts: np.ndarray | None = None) -> np.ndarray:
    """Analogous to biotite's `spread_residue_wise`."""
    if token_starts is None:
        token_starts = get_token_starts(array, add_exclusive_stop=True)
    return struc.segments.spread_segment_wise(token_starts, input_data)


def apply_token_wise(
    array: AtomArray,
    data: np.ndarray,
    function: Callable,
    axis: int | None = None,
    token_starts: np.ndarray | None = None,
) -> np.ndarray:
    """Analogous to biotite's `apply_residue_wise`."""
    if token_starts is None:
        token_starts = get_token_starts(array, add_exclusive_stop=True)
    return struc.segments.apply_segment_wise(token_starts, data, function, axis)


def apply_and_spread_token_wise(
    atom_array: AtomArray,
    data: np.ndarray,
    function: Callable,
    axis: int | None = None,
    token_starts: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a function token wise and then spread the result to the atoms."""
    if token_starts is None:
        token_starts = get_token_starts(atom_array, add_exclusive_stop=True)
    return spread_token_wise(atom_array, apply_token_wise(atom_array, data, function, axis, token_starts), token_starts)


def apply_segment_wise_2d(array: np.ndarray, segment_start_end_idxs: np.ndarray, reduce_func: Callable) -> np.ndarray:
    """Reduces a 2D array by applying a reduction function to specified segments along both rows and columns.

    NOTE: Segments must be contiguous, rectangular blocks (sub-matrices) of the 2D array.

    Args:
        array (np.ndarray): A 2D numpy array to be reduced.
        group_start_end_idxs (np.ndarray): A 1D numpy array of indices indicating the start and end of each block.
            The first element must be 0 and the last element must be the number of rows in the array.
        reduce_func (Callable): A function to apply to each segment. This function should take an array and return
            a reduced value.

    Returns:
        np.ndarray: A 2D numpy array that has been reduced along both rows and columns.

    Example:
        >>> array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        >>> segment_start_end_idxs = np.array([0, 2, 3])
        >>> apply_segment_wise_2d(array, segment_start_end_idxs, reduce_func=np.sum)
        array([
            [12, 9],
            [15, 9]
        ])
    """
    assert array.ndim == 2, "Array must be 2D"
    assert segment_start_end_idxs.ndim == 1, "Group start end idxs must be 1D"
    assert segment_start_end_idxs[0] == 0, "Group start end idxs must start with 0"
    assert (
        segment_start_end_idxs[-1] == array.shape[0]
    ), "Group start end idxs must end with the number of rows in the array"
    assert np.all(np.diff(segment_start_end_idxs) > 0), "Group start end idxs must be strictly increasing"

    # reduce along rows
    array = struc.segments.apply_segment_wise(segment_start_end_idxs, array, reduce_func, axis=0)

    # reduce along columns (transpose and then apply segment wise along axis 0 again)
    # ... NOTE: For some reason, `apply_segment_wise` fails when applied along axis 1, which is why
    #       we perform the axis-flip via transpose and then flip back.
    array = struc.segments.apply_segment_wise(segment_start_end_idxs, array.T, reduce_func, axis=0).T

    return array


def get_af3_token_representative_masks(atom_array: AtomArray, enforce_one_per_token: bool = True) -> np.ndarray:
    """Returns a boolean mask indicating the representative atoms of the tokens in the atom array.

    From the AF-3 supplement, section 4.4. (Distogram prediction):
        > ...where the pairwise token distances use the representative atom for each token: CB
        for protein residues (CA for glycine), C4 for purines and C2 for pyrimidines.
        All ligands already have a single atom per token.

    NOTE: "Representative" atoms are distinct from "center" atoms, which are used during cropping.

    Args:
        atom_array (AtomArray): The atom array to get the representative atoms of.
        enforce_one_per_token (bool, optional): If True, raises an error if the number of representative atoms
            does not match the number of tokens. Defaults to True.

    Returns:
        np.ndarray: A boolean mask indicating the representative atoms of the tokens in the atom array.
    """
    assert (
        "atomize" in atom_array.get_annotation_categories()
    ), "Atomize annotation is missing. Run AtomizeByCCDName Transform for magical atomization of ligands"
    pyrimidine_representative_atom = is_pyrimidine(atom_array.res_name) & (atom_array.atom_name == "C2")
    purine_representative_atom = is_purine(atom_array.res_name) & (atom_array.atom_name == "C4")
    unknown_na_representative_atom = is_unknown_nucleotide(atom_array.res_name) & (atom_array.atom_name == "C4")

    glycine_representative_atom = is_glycine(atom_array.res_name) & (atom_array.atom_name == "CA")
    protein_residue_not_glycine_representative_atom = is_standard_aa_not_glycine(atom_array.res_name) & (
        atom_array.atom_name == "CB"
    )
    unknown_protein_residue_representative_atom = is_protein_unknown(atom_array.res_name) & (
        atom_array.atom_name == "CA"
    )

    atoms = atom_array.atomize

    is_representative_atom = (
        pyrimidine_representative_atom
        | purine_representative_atom
        | unknown_na_representative_atom
        | glycine_representative_atom
        | protein_residue_not_glycine_representative_atom
        | unknown_protein_residue_representative_atom
        | atoms
    )
    if enforce_one_per_token and (is_representative_atom.sum() != get_token_count(atom_array)):
        raise ValueError(
            f"Number of representative atoms ({is_representative_atom.sum()}) does not match number"
            f"of tokens ({get_token_count(atom_array)}). This is likely due to you filtering out"
            "some atoms from the atom array that are then missing as represenatives."
        )

    return is_representative_atom


def get_af3_token_representative_idxs(atom_array: AtomArray) -> np.ndarray:
    """
    Returns the indices of the representative atoms of the tokens in the atom array.

    See "get_af3_token_representative_masks" for more details on what constitutes a representative atom.

    Args:
        atom_array (AtomArray): The atom array to get the representative atom indices from.

    Returns:
        np.ndarray: An array of indices corresponding to the representative atoms of the tokens.
    """
    return np.where(get_af3_token_representative_masks(atom_array))[0]


def get_af3_token_representative_coords(atom_array: AtomArray) -> np.ndarray:
    """
    Returns the representative coordinates of the tokens in the atom array.

    See "get_af3_token_representative_masks" for more details on what constitutes a representative atom.

    Args:
        atom_array (AtomArray): The atom array to get the representative coordinates of.

    Returns:
        np.ndarray: The representative coordinates of the tokens in the atom array.
    """
    return atom_array.coord[get_af3_token_representative_masks(atom_array)]


def get_af3_token_center_masks(atom_array: AtomArray, enforce_one_per_token: bool = True) -> np.ndarray:
    """Returns a boolean mask indicating the center atoms of the tokens in the atom array as per the AF3 definition.

    NOTE: "Center" atoms are distinct from "representative" atoms, which are used during distogram prediction (and more closely represent the center of mass).

    For each token we also designate a token center atom, used in various places below:
        - CA for standard amino acids
        - C1' for standard nucleotides
        - For other cases take the first and only atom as they are tokenized per-atom.

    Args:
        atom_array (AtomArray): The atom array to get the center atoms of.
        enforce_one_per_token (bool, optional): If True, raises an error if the number of center atoms
            does not match the number of tokens. Defaults to True.

    Returns:
        np.ndarray: A boolean mask indicating the center atoms of the tokens in the atom array.

    Reference:
        `AF3 Supplementary Information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_

    """
    assert (
        "atomize" in atom_array.get_annotation_categories()
    ), "Atomize annotation is missing. Run AtomizeByCCDName Transform first!"

    is_center_atom = (
        atom_array.atomize  # the atom itself for un-atomized tokens
        | (atom_array.atom_name == "CA")  # CA for amino acids
        | (atom_array.atom_name == "C1'")  # C1' for nucleotides
    )
    if enforce_one_per_token and (is_center_atom.sum() != get_token_count(atom_array)):
        raise ValueError(
            f"Number of center atoms ({is_center_atom.sum()}) does not match"
            f"number of tokens ({get_token_count(atom_array)}). This is likely"
            "due to you filtering out some atoms from the atom array that are"
            "then missing as centers."
        )

    return is_center_atom


def get_af3_token_center_idxs(atom_array: AtomArray) -> np.ndarray:
    """
    Returns the indices of the center atoms of the tokens in the atom array as per the AF3 definition.
    """
    return np.where(get_af3_token_center_masks(atom_array))[0]


def get_af3_token_center_coords(atom_array: AtomArray) -> np.ndarray:
    """
    Returns the center coordinates of the tokens in the atom array as per the AF3 definition.

    For each token we also designate a token center atom, used in various places below:
        - CA for standard amino acids
        - C1' for standard nucleotides
        - For other cases take the first and only atom as they are tokenized per-atom.

    If a token center cannot be assigned (e.g. because the token center atom is unoccupied),
    the center coordinate is set to `np.nan`.

    Args:
        atom_array (AtomArray): The atom array to get the center coordinates of.

    Returns:
        np.ndarray: The center coordinates of the tokens in the atom array.

    Reference:
        `AF3 Supplementary Information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_

    Example:
        >>> # Contrived example showing only a few tokens and annotations per residue for illustration
        >>> array = AtomArray(
            res_name="ALA", atom_name="CA", coord=np.array([0, 0, 0]),
            res_name="ALA", atom_name="C", coord=np.array([1, 0, 0]),
            res_name="ALA", atom_name="O", coord=np.array([2, 0, 0]),
            res_name="NAP", atom_name="P1", coord=np.array([3, 0, 0]),
            res_name="U",   atom_name="C1'", coord=np.array([4, 0, 0]),
        )
        >>> get_af3_token_center_coords(array)
        array([[0, 0, 0], [3, 0, 0], [4, 0, 0]])
    """
    return atom_array.coord[get_af3_token_center_masks(atom_array)]
