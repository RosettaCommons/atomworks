"""Utility functions for MSA pairing."""

from __future__ import annotations

import logging
from functools import reduce

import numpy as np

from atomworks.ml.transforms.msa._msa_constants import AMINO_ACID_ONE_LETTER_TO_INT
from atomworks.ml.utils.misc import cumcount

logger = logging.getLogger(__name__)


def _get_matched_indices(msa: dict, shared_tax_ids: np.array) -> np.array:
    """
    Given an MSA with associated tax_ids
        (1) drop all entries that do not appear in `shared_tax_ids` and
        (2) sort the remaining entries first by `tax_id` (high to low) and then by sequence similarity (high to low).
            The `query` sequence is guaranteed to be sorted in the first position.

    NOTE: If the `seq_similarity_with_query_sequence` is provided in the MSA dictionary, it will be used for sorting.

    Args:
        msa (dict): Dictionary containing 'tax_ids' and 'msa' keys.
        shared_tax_ids (np.array): Array of tax IDs to keep.

    Returns:
        np.array: The sorted indices to index into the values of the `msa` dictionary.
        Entries with `tax_ids` that are not in the `shared_tax_ids` are dropped.

    Example:
    >>> msa = {
    ...     "tax_ids": np.array([101, 102, 103, 104, 102]),
    ...     "msa": np.array(
    ...         [
    ...             ["A", "T", "G", "C"],  # Query
    ...             ["A", "T", "G", "A"],  # Matches 3/4
    ...             ["G", "T", "G", "C"],  # Tax ID not shared, this entry will be dropped
    ...             ["A", "A", "A", "C"],  # Matches 2/4
    ...             ["A", "T", "A", "A"],  # Matches 2/4
    ...         ]
    ...     ),
    ... }
    >>> shared_tax_ids = [101, 102, 104]
    >>> _get_matched_indices(msa, shared_tax_ids)
    array([0, 1, 3, 4])
    """
    query_tax_id = msa["tax_ids"][0]
    mask = np.isin(msa["tax_ids"], shared_tax_ids)
    msa_indices = np.where(mask)[0]

    tax_ids = msa["tax_ids"][mask]

    if "seq_similarity_with_query_sequence" in msa:
        # Use provided sequence similarity
        seq_similarity = msa["seq_similarity_with_query_sequence"][mask]
    else:
        # Compute sequence similarity if not provided
        seq_similarity = (msa["msa"][mask] == msa["msa"][0:1]).mean(axis=1)

    # Create a priority array that gives higher priority to sequences with the same tax_id as the query
    # This way, we ensure the query sequence is always the first index returned (since its sequence identity is 100%)
    priority = (tax_ids == query_tax_id).astype(int)

    # Perform lexicographic sort with priority, tax_ids, and seq_similarity
    sort_indices = np.lexsort((seq_similarity, tax_ids, priority))[::-1]

    matched_indices = msa_indices[sort_indices]
    return matched_indices


def _remove_extraneous_taxid_copies(
    msa_a: dict, msa_b: dict, i_paired_a: np.ndarray, i_paired_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes indices in i_paired_a and i_paired_b until the tax_ids in both match not only in tax_id
    but also in repeat number.

    Args:
        msa_a (dict): A dictionary containing:
            - "tax_ids" (np.ndarray): An array of taxonomic IDs for msa_a.
        msa_b (dict): A dictionary containing:
            - "tax_ids" (np.ndarray): An array of taxonomic IDs for msa_b.
        i_paired_a (np.ndarray): Array of indices for msa_a.
        i_paired_b (np.ndarray): Array of indices for msa_b.

    Returns:
        tuple: Two numpy arrays of indices (i_paired_a, i_paired_b) with extraneous copies removed.

    Example:
        >>> msa_a = {
        ...     "tax_ids": np.array(["a", "a", "a", "b", "b"]),
        ... }
        >>> msa_b = {
        ...     "tax_ids": np.array(["a", "a", "b", "b", "b"]),
        ... }
        >>> i_paired_a = np.array([0, 1, 2, 3, 4])
        >>> i_paired_b = np.array([0, 1, 2, 3, 4])
        >>> _remove_extraneous_taxid_copies(msa_a, msa_b, i_paired_a, i_paired_b)
        (array([0, 1, 3, 4]), array([0, 1, 2, 3]))
    """
    tax_ids_a = msa_a["tax_ids"][i_paired_a]
    tax_ids_b = msa_b["tax_ids"][i_paired_b]

    counts_a = np.char.add("_", cumcount(tax_ids_a).astype(str))
    counts_b = np.char.add("_", cumcount(tax_ids_b).astype(str))

    tax_ids_a_with_counts = np.char.add(tax_ids_a, counts_a)
    tax_ids_b_with_counts = np.char.add(tax_ids_b, counts_b)

    discard_mask_a = np.isin(tax_ids_a_with_counts, tax_ids_b_with_counts)
    discard_mask_b = np.isin(tax_ids_b_with_counts, tax_ids_a_with_counts)

    i_paired_a = i_paired_a[discard_mask_a]
    i_paired_b = i_paired_b[discard_mask_b]
    return i_paired_a, i_paired_b


def _get_paired(msa_a: dict, msa_b: dict, shared_tax_ids: np.array) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized implementation of the following.

    Given a set of tax_ids that are shared between two MSAs, this function:
        1. Finds the indices of the sequences in the MSAs that
            have those tax_ids, and returns them in sorted lexicographic order
            first by tax_id and then by how well the sequence matches the query.
        2. Removes extraneous copies of tax_ids in the indices found in step 1, such that
            both index sets have the same taxids with the same multiplicity.
    """
    i_paired_a = _get_matched_indices(msa_a, shared_tax_ids)
    i_paired_b = _get_matched_indices(msa_b, shared_tax_ids)

    i_paired_a, i_paired_b = _remove_extraneous_taxid_copies(msa_a, msa_b, i_paired_a, i_paired_b)

    return i_paired_a, i_paired_b


def join_two_msas_by_tax_id(
    msa_a: dict, msa_b: dict, unpaired_padding: np.ndarray, add_residue_is_paired_feature: bool = False
) -> dict:
    """
    Joins (or "pairs") 2 MSAs by matching sequences with the same taxonomic ID.
    Sequences that aren't paired will be added to the bottom of the joined MSA in a block-diagonal fashion, with padding.

    Conceptually (--- is a purely padded sequence which matches the length of query_a / query_b):
    msa/ins                   tax_ids          is_paired
    --------------------------------------------------------
     [query_a, query_b],       "query",           True        # always `True` for queries, tax_id is "query" since a/b may or may not be from the same taxon
     [seq1_a , seq1_b ],       104,               True        # there may be more than one pair per tax id
     [seq2_a , seq2_b ],       104,               True
     [seq3_a , seq3_b ],       103,               True
     [seq4_a , ---    ],       105,               False
     [seq5_a , ---    ],       102,               False
     [---    , seq4_b ],       100,               False

    Args:
        msa_a (dict):
            First MSA to be joined, with keys:
            - `msa` (N_seq, L_seq) (Can by any data type, e.g., string, integers, etc.)
            - `ins` (N_seq, L_seq)
            - `tax_id` (N_seq,)
            - `msa_is_padded_mask` (N_seq, L_seq)
            - `all_paired` (N_seq,) (optional, will be added if not present)
            - `any_paired` (N_seq,) (optional, will be added if not present)
        msa_b (dict):
            Second MSA to be joined, with keys `msa`, `ins`, and `tax_ids`.
        unpaired_padding (np.ndarray):
            Scalar array for unpaired sequences. Must match the dtype of the MSA data.
        add_residue_is_paired_feature (bool):
            Whether to add a binary feature indicating whether a residue is part of a paired MSA.

    Returns:
        dict: Paired MSA, with keys `msa`, `ins`, `tax_ids`, `any_paired`, `all_paired`, and `msa_is_padded_mask`.
    """
    msa_a_num_residues, msa_b_num_residues = msa_a["msa"].shape[1], msa_b["msa"].shape[1]
    msa_a_num_sequences, msa_b_num_sequences = msa_a["msa"].shape[0], msa_b["msa"].shape[0]

    # ...get the tax IDs that are shared between the two MSAs, with duplicates allowed
    shared_tax_ids = msa_a["tax_ids"][np.isin(msa_a["tax_ids"], msa_b["tax_ids"])]

    # ...remove shared empty strings
    shared_tax_ids = shared_tax_ids[shared_tax_ids != ""]

    # ...ensure query sequence is shared
    query_tax_id = msa_a["tax_ids"][0]
    assert query_tax_id in shared_tax_ids, "Query sequence tax ID must be in the shared tax IDs"

    # ...pair sequences, sorting first by tax_id and then by how well the sequence matches the query
    # Here, we drop out any unpaired sequences (keeping the best match)
    # We also force the query sequence to be the first index returned for both MSAs
    i_paired_a, i_paired_b = _get_paired(msa_a, msa_b, shared_tax_ids)

    # ...get indices of sequences that are not paired
    # NOTE: Since np.setdiff1d returns a sorted array, the unpaired rows will have their relative order preserved
    # This ensures that the fully unpaired sequences are block-diagonal and at the bottom of the MSA, even after many iterations.
    i_unpaired_a = np.setdiff1d(np.arange(msa_a_num_sequences), i_paired_a)
    i_unpaired_b = np.setdiff1d(np.arange(msa_b_num_sequences), i_paired_b)

    n_paired, n_unpaired_a, n_unpaired_b = (
        len(i_paired_a),  # Same as len(i_paired_b)
        len(i_unpaired_a),
        len(i_unpaired_b),
    )

    # ...concatenate paired sequences
    msa_paired = np.concatenate([msa_a["msa"][i_paired_a], msa_b["msa"][i_paired_b]], axis=1)
    ins_paired = np.concatenate([msa_a["ins"][i_paired_a], msa_b["ins"][i_paired_b]], axis=1)
    msa_paired_is_padded_mask = np.concatenate(
        [msa_a["msa_is_padded_mask"][i_paired_a], msa_b["msa_is_padded_mask"][i_paired_b]], axis=1
    )

    # ...pad unpaired sequences with gaps
    msa_dtype = msa_a["msa"].dtype
    assert msa_dtype == msa_b["msa"].dtype, "MSA data types must match"

    # Sparse packing, with padding on the off-diagonal, AF-Multimer style
    msa_a_unpaired = np.concatenate(
        [
            msa_a["msa"][i_unpaired_a],
            np.full((n_unpaired_a, msa_b_num_residues), unpaired_padding, dtype=msa_dtype),
        ],
        axis=1,
    )

    ins_a_unpaired = np.concatenate(
        [
            msa_a["ins"][i_unpaired_a],
            np.full((n_unpaired_a, msa_b_num_residues), 0, dtype=msa_a["ins"].dtype),
        ],
        axis=1,
    )

    msa_b_unpaired = np.concatenate(
        [
            np.full((n_unpaired_b, msa_a_num_residues), unpaired_padding, dtype=msa_dtype),
            msa_b["msa"][i_unpaired_b],
        ],
        axis=1,
    )

    ins_b_unpaired = np.concatenate(
        [
            np.full((n_unpaired_b, msa_a_num_residues), 0, dtype=msa_b["ins"].dtype),
            msa_b["ins"][i_unpaired_b],
        ],
        axis=1,
    )

    # ...create padding masks to keep track of which MSA indices are meaningful

    msa_a_unpaired_is_padded_mask = np.concatenate(
        [
            msa_a["msa_is_padded_mask"][i_unpaired_a],
            np.ones((n_unpaired_a, msa_b_num_residues), dtype=bool),
        ],
        axis=1,
    )

    msa_b_unpaired_is_padded_mask = np.concatenate(
        [
            np.ones((n_unpaired_b, msa_a_num_residues), dtype=bool),
            msa_b["msa_is_padded_mask"][i_unpaired_b],
        ],
        axis=1,
    )

    # ...stack paired & unpaired
    msa = np.concatenate([msa_paired, msa_a_unpaired, msa_b_unpaired], axis=0)
    ins = np.concatenate([ins_paired, ins_a_unpaired, ins_b_unpaired], axis=0)
    tax_ids = np.concatenate(
        [
            msa_a["tax_ids"][i_paired_a],  # Same as msa_b["tax_ids"][i_paired_b]
            msa_a["tax_ids"][i_unpaired_a],
            msa_b["tax_ids"][i_unpaired_b],
        ],
        axis=0,
    )
    msa_is_padded_mask = np.concatenate(
        [msa_paired_is_padded_mask, msa_a_unpaired_is_padded_mask, msa_b_unpaired_is_padded_mask], axis=0
    )

    # ...label sequences that were paired
    any_paired = np.concatenate(
        [
            np.ones(n_paired, dtype=bool),
            msa_a["any_paired"][i_unpaired_a] if "any_paired" in msa_a else np.zeros(n_unpaired_a, dtype=bool),
            np.zeros(n_unpaired_b, dtype=bool),
        ],
    )
    all_paired = np.concatenate(
        [
            msa_a["all_paired"][i_paired_a] if "all_paired" in msa_a else np.ones(n_paired, dtype=bool),
            np.zeros(n_unpaired_a, dtype=bool),
            np.zeros(n_unpaired_b, dtype=bool),
        ],
    )

    # ... label residues that were paired, if requested
    if add_residue_is_paired_feature:
        # Within the any_paired region, any unpaired regions will be masked (and the converse always holds).
        # This correspondence does not hold in general, as fully unpaired sequences are not considered masked.
        residue_is_paired = ~msa_is_padded_mask
        residue_is_paired[~any_paired] = False

        # (Sanity check)
        assert np.all(residue_is_paired[all_paired, :]), "Residues in all_paired rows should all be paired"

    # ...and assert that the first row is still the query sequence as a sanity check
    assert np.all(
        msa[0] == np.concatenate([msa_a["msa"][0], msa_b["msa"][0]])
    ), "Query sequence must be the first row of the MSA"

    result = {
        "msa": msa,
        "ins": ins,
        "tax_ids": tax_ids,
        "any_paired": any_paired,
        "all_paired": all_paired,
        "msa_is_padded_mask": msa_is_padded_mask,
    }

    if add_residue_is_paired_feature:
        result["residue_is_paired"] = residue_is_paired

    return result


def join_multiple_msas_by_tax_id(
    msas: list,
    unpaired_padding: np.ndarray = np.array([AMINO_ACID_ONE_LETTER_TO_INT["-"]], dtype=np.int8),  # noqa: B008
    dense: bool = False,
    shuffle_unpaired_sequences: bool = False,
    add_residue_is_paired_feature: bool = False,
) -> dict:
    """
    Join multiple MSAs by tax_id, merging them sequentially and updating pairing information.
    Returns the merged MSA with updated pairing information.

    Conceptually, this will look like:
    ----------------------------------
     [query_a, query_b, query_c]
     [seq1_a,  seq_1b,  seq1_c ] # (all 3 pairs)
     [  ...                    ]
     [seqX_a,  seqX_b,  ---    ] # (all 2 pairs, a,b first)
     [ ...                     ]
     [seqX_a,  ---  ,   seqX_c ] # (all 2 pairs, a,c)
     [ ...                     ]
     [---,     seqX_b,  seqX_c ] # (all 2 paris, b,c)
     [...                      ]
     [seqX_a,  ---,     ---    ] # in dense (AF3 style), only completely unpaired
     [ ...                     ] # parts would be `collapsed`
     [---   ,  seqX_b,  ---    ]
     [ ...                     ]
     [---,     ---,     seqX_c ]

    Args:
        msas (list): List of MSAs to be merged. MSAs will be merged sequentially (chain ordering matters). MSAs must be numpy arrays, but can be of any data type.
        unpaired_padding (Any): Padding for unpaired sequences. Datatype must match the datatype of the MSAs. Defaults to the integer encoding of "-" (gap).
        dense (bool, optional): Whether to densely pack unpadded sequences (AF-3 style), or use sparse block matrices (AF-Multimer style). Defaults to False.
        shuffle_unpaired (bool, optional): Whether to shuffle unpaired sequences before collapsing during dense merging. Defaults to False.
        add_residue_is_paired_feature (bool): Whether to add a binary feature indicating whether a residue is part of a paired MSA.

    Returns:
        dict: Merged MSA with updated pairing information.
    """
    if shuffle_unpaired_sequences and not dense:
        raise ValueError(
            "shuffle_unpaired_sequences can only be used in dense mode; set dense=True to use shuffle_unpaired_sequences"
        )

    # Assert that the unpaired padding has the same dtype as the MSA data
    assert msas[0]["msa"].dtype == unpaired_padding.dtype, "unpaired_padding must have the same dtype as the MSA data"

    result = reduce(
        lambda a, b: join_two_msas_by_tax_id(
            a, b, unpaired_padding=unpaired_padding, add_residue_is_paired_feature=add_residue_is_paired_feature
        ),
        msas,
    )

    # If we are in dense mode, we need to handle unpaired sequences differently
    if dense:
        # Break the MSA into paired and unpaired sequences
        unpaired_msa = result["msa"][~result["any_paired"]]
        unpaired_ins = result["ins"][~result["any_paired"]]

        # Re-separate by chain
        separated_unpaired_msas = []
        separated_unpaired_ins = []
        separated_unpaired_is_padded_mask = []
        start_idx = 0
        for msa in msas:
            num_residues = msa["msa"].shape[1]
            end_idx = start_idx + num_residues

            msa = unpaired_msa[:, start_idx:end_idx]
            ins = unpaired_ins[:, start_idx:end_idx]

            # Remove rows with all padding tokens
            mask = ~np.all(msa == unpaired_padding, axis=1)
            msa = msa[mask]
            ins = ins[mask]

            # Optionally shuffle the unpaired rows
            # This may be helpful for training, otherwise the pairing of unpaired sequences is deterministic
            if shuffle_unpaired_sequences:
                # Generate shuffled indices
                indices = np.arange(msa.shape[0])
                np.random.shuffle(indices)

                # Apply the same shuffle to both msa and ins using the shuffled indices
                msa = msa[indices]
                ins = ins[indices]

            separated_unpaired_msas.append(msa)
            separated_unpaired_ins.append(ins)
            start_idx = end_idx

        # Find the largest unpaired MSA and pad the others to match
        largest_unpaired_msa = max(msa.shape[0] for msa in separated_unpaired_msas)
        for i, msa in enumerate(separated_unpaired_msas):
            if msa.shape[0] < largest_unpaired_msa:
                # Pad MSA
                msa_padding = np.full(
                    (largest_unpaired_msa - msa.shape[0], msa.shape[1]), unpaired_padding, dtype=msa.dtype
                )
                separated_unpaired_msas[i] = np.concatenate([msa, msa_padding], axis=0)

                # Assign padding mask
                padding_mask = np.concatenate(
                    [
                        np.zeros((msa.shape[0], msa.shape[1]), dtype=bool),
                        np.ones((largest_unpaired_msa - msa.shape[0], msa.shape[1]), dtype=bool),
                    ],
                    axis=0,
                )
                separated_unpaired_is_padded_mask.append(padding_mask)

                # Pad insertion array
                ins = separated_unpaired_ins[i]
                ins_padding = np.zeros((largest_unpaired_msa - msa.shape[0], ins.shape[1]), dtype=ins.dtype)
                separated_unpaired_ins[i] = np.concatenate([ins, ins_padding], axis=0)
            else:
                # Assign padding mask (all zeros)
                separated_unpaired_is_padded_mask.append(np.zeros((msa.shape[0], msa.shape[1]), dtype=bool))

        # Concatenate the unpaired MSAs along the columns
        unpaired_msa = np.concatenate(separated_unpaired_msas, axis=1)
        unpaired_ins = np.concatenate(separated_unpaired_ins, axis=1)
        unpaired_is_padded_mask = np.concatenate(separated_unpaired_is_padded_mask, axis=1)

        # Add back in the paired MSAs along the row dimension
        result["msa"] = np.concatenate([result["msa"][result["any_paired"]], unpaired_msa], axis=0)
        result["ins"] = np.concatenate([result["ins"][result["any_paired"]], unpaired_ins], axis=0)
        result["msa_is_padded_mask"] = np.concatenate(
            [result["msa_is_padded_mask"][result["any_paired"]], unpaired_is_padded_mask], axis=0
        )

        # We set all dense, unpaired tax IDs to be empty strings
        tax_id_dtype = result["tax_ids"][0].dtype
        result["tax_ids"] = np.concatenate(
            [result["tax_ids"][result["any_paired"]], np.full(unpaired_msa.shape[0], "", dtype=tax_id_dtype)], axis=0
        )

        # Update pairing information
        result["any_paired"] = result["any_paired"][
            : result["msa"].shape[0]
        ]  # Trim to match the new MSA shape; we only re-ordered the unpaired sequences
        result["all_paired"] = result["all_paired"][: result["msa"].shape[0]]
        if add_residue_is_paired_feature:
            result["residue_is_paired"] = result["residue_is_paired"][: result["msa"].shape[0], :]

    return result
