"""Transforms on MSAs"""

from __future__ import annotations

import logging
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from biotite.structure import AtomArray

from atomworks.common import exists
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, AF3SequenceEncoding, TokenEncoding
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import ConvertToTorch, Transform
from atomworks.ml.transforms.msa._msa_constants import (
    AMINO_ACID_ONE_LETTER_TO_INT,
    GAP_THREE_LETTER,
    MSA_INTEGER_TO_THREE_LETTER,
)
from atomworks.ml.transforms.msa._msa_featurizing_utils import (
    assign_extra_rows_to_cluster_representatives,
    build_indices_should_be_counted_masks,
    build_msa_index_can_be_masked,
    mask_msa_like_bert,
    summarize_clusters,
    transform_ins_counts,
    uniformly_select_rows,
)
from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path, load_msa_data_from_path
from atomworks.ml.transforms.msa._msa_pairing_utils import join_multiple_msas_by_tax_id
from atomworks.ml.utils.io import cache_to_disk_as_pickle
from atomworks.ml.utils.misc import grouped_count
from atomworks.ml.utils.token import apply_token_wise, get_token_count, get_token_starts

logger = logging.getLogger(__name__)


class PairAndMergePolymerMSAs(Transform):
    """Pairs and merges multiple polymer MSAs by tax_id.

    Ensures that the query sequence is always the first sequence in the MSA.

    Stores results in "polymer_msas_by_chain_id" in the data dictionary, with keys:
        - msa: The merged MSA
        - ins: The merged insertion array
        - msa_is_padded_mask: A mask indicating whether a given position in the MSA is padded due to unpaired sequences (1) or not (0)
        - tax_ids: The merged taxonomic IDs
        - any_paired: A boolean array indicating whether a sequence is paired with any other sequence
        - all_paired: A boolean array indicating whether a sequence is paired with all other sequences

    Unpaired sequences can be handled in two ways:
        - Dense pairing: Unpaired sequences are densely packed at the bottom of the MSA (AF-3 style).
        - Sparse pairing: Unpaired sequences are block-diagonally added to the bottom of the MSA (AF-Multimer style).

    Args:
        unpaired_padding (Any): The MSA token to use for padding unpaired sequences. Defaults to the integer representation of the gap token.
        dense (bool): Whether to densely pack unpaired sequences at the bottom of the MSA. If False, unpaired sequences are block-diagonally added to the bottom of the MSA.
        add_residue_is_paired_feature (bool): Whether to add a binary feature indicating whether a residue is part of a paired sequence.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["LoadPolymerMSAs"]

    def __init__(
        self,
        unpaired_padding: int = AMINO_ACID_ONE_LETTER_TO_INT["-"],  # Integer representation of gap token
        dense: bool = False,
        add_residue_is_paired_feature: bool = False,
    ):
        if not (
            isinstance(unpaired_padding, int) and np.iinfo(np.int8).min <= unpaired_padding <= np.iinfo(np.int8).max
        ):
            raise ValueError(
                f"unpaired_padding={unpaired_padding} is not representable as np.int8. "
                f"Must be an integer in [{np.iinfo(np.int8).min}, {np.iinfo(np.int8).max}]."
            )
        self.unpaired_padding = np.array(unpaired_padding, dtype=np.int8)
        self.dense = dense
        self.add_residue_is_paired_feature = add_residue_is_paired_feature

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["polymer_msas_by_chain_id"])

    def forward(self, data: dict) -> dict:
        # If we have no polymer MSAs, we can skip this step
        if len(data["polymer_msas_by_chain_id"]) == 0:
            return data

        atom_array = data["atom_array"]

        # Create a map from unique entity IDs to constituent chain IDs
        # We directly generate the mapping from the AtomArray since the `rcsb_entity` may be inaccurate post-processing
        # We need entity-level information to pair chains that belong to separate entities; otherwise, we simply concatenate the MSAs
        chain_with_msa_entity_to_ids = {}
        chain_with_msa_id_to_entity = {}

        for chain_entity in np.unique(atom_array.chain_entity):
            # ...get unique chain IDs corresponding to the current chain entity
            chain_ids = np.unique(atom_array.chain_id[atom_array.chain_entity == chain_entity])

            # If we have an MSA for any of the chains in the entity:
            if any(chain_id in data["polymer_msas_by_chain_id"] for chain_id in chain_ids):
                # ...store the chain IDs for the entity
                chain_with_msa_entity_to_ids[chain_entity] = chain_ids

                # ...and map each chain ID back to its chain entity
                for chain_id in chain_ids:
                    chain_with_msa_id_to_entity[chain_id] = chain_entity

        # Loop through entities to:
        # (1) Create a list of MSAs to pair, choosing one chain per entity ID (as they are all the same, by definition)
        # (2) Keep track of the number of residues for each entity ID
        msa_list = []
        num_residues_by_chain_with_msa_entity = {}
        for chain_entity in chain_with_msa_entity_to_ids:
            first_chain_id = chain_with_msa_entity_to_ids[chain_entity][0]  # All chains in the entity are the same
            msa = data["polymer_msas_by_chain_id"][first_chain_id]
            msa_list.append(msa)
            num_residues_by_chain_with_msa_entity[chain_entity] = msa["msa"].shape[1]

        # Create masks for each entity to index into the merged MSA
        entity_masks = {}
        chain_entity_array = np.concatenate(
            [
                [chain_entity] * num_residues_by_chain_with_msa_entity[chain_entity]
                for chain_entity in chain_with_msa_entity_to_ids
                if chain_entity in num_residues_by_chain_with_msa_entity
            ]
        )
        for chain_entity in chain_with_msa_entity_to_ids:
            entity_masks[chain_entity] = chain_entity_array == chain_entity

        if len(msa_list) > 1:
            # Heteromeric complex - pair and merge the MSAs
            merged_polymer_msas = join_multiple_msas_by_tax_id(
                msa_list,
                unpaired_padding=self.unpaired_padding,
                dense=self.dense,
                add_residue_is_paired_feature=self.add_residue_is_paired_feature,
            )
        else:
            # Homomeric complex - no need to pair, we will concatenate the MSAs later
            merged_polymer_msas = msa_list[0]

            # We consider homomers to be unpaired
            merged_polymer_msas["all_paired"] = np.zeros(merged_polymer_msas["msa"].shape[0], dtype=bool)
            merged_polymer_msas["any_paired"] = np.zeros(merged_polymer_msas["msa"].shape[0], dtype=bool)
            if self.add_residue_is_paired_feature:
                merged_polymer_msas["residue_is_paired"] = np.zeros(merged_polymer_msas["msa"].shape, dtype=bool)

        # Distribute entity-level MSAs to chain-level MSAs by pointing each chain_id to the MSA for its corresponding chain_entity
        polymer_msas_by_chain_entity = {}
        for chain_entity in chain_with_msa_entity_to_ids:
            entity_mask = entity_masks[chain_entity]
            msa = merged_polymer_msas["msa"][:, entity_mask]
            ins = merged_polymer_msas["ins"][:, entity_mask]
            msa_is_padded_mask = merged_polymer_msas["msa_is_padded_mask"][:, entity_mask]

            polymer_msas_by_chain_entity[chain_entity] = {
                "msa": msa,
                "ins": ins,
                "msa_is_padded_mask": msa_is_padded_mask,
                "tax_ids": merged_polymer_msas["tax_ids"],  # Common across entities (sequence dimension)
                "any_paired": merged_polymer_msas["any_paired"],  # Common across entities (sequence dimension)
                "all_paired": merged_polymer_msas["all_paired"],  # Common across entities (sequence dimension)
            }

            if self.add_residue_is_paired_feature:
                polymer_msas_by_chain_entity[chain_entity]["residue_is_paired"] = merged_polymer_msas[
                    "residue_is_paired"
                ][:, entity_mask]

        for chain_id in data["polymer_msas_by_chain_id"]:
            chain_entity = chain_with_msa_id_to_entity[chain_id]
            # NOTE: We deep copy as a precaution, since if multiple chains point to the same dictionary object, we may modify the dictionary in place
            data["polymer_msas_by_chain_id"][chain_id] = deepcopy(polymer_msas_by_chain_entity[chain_entity])

        return data


def load_polymer_msas(
    atom_array: AtomArray,
    chain_info: dict,
    protein_msa_dirs: list[dict[str, str]],
    rna_msa_dirs: list[dict[str, str]],
    max_msa_sequences: int = 10_000,
    msa_cache_dir: PathLike | None = None,
    use_paths_in_chain_info: bool = True,
    raise_if_missing_msa_for_protein_of_length_n: int | None = None,
    unk_symbol: str = "X",
) -> dict[str, np.array]:
    """
    Load MSAs for all polymer chains in the AtomArray and store them in a dictionary. See the LoadPolymerMSAs transform for more information
    Args:
        atom_array (AtomArray): The AtomArray for the full structure
        chain_info (dict): A dictionary containing chain information, including:
            - processed_entity_non_canonical_sequence: The non-canonical sequence for the chain
            - processed_entity_canonical_sequence: The canonical sequence for the chain
            - chain_type: The type of the chain (e.g., protein, RNA)
            - msa_path (optional): The path to the MSA file for the chain, if available
        protein_msa_dirs (list[dict[str, str]]): The directories containing the protein MSAs and their associated file types.
        rna_msa_dirs (list[dict[str, str]]): The directories containing the RNA MSAs and their associated file types.
        max_msa_sequences (int): The maximum number of sequences to load from the MSA files. Defaults to 10_000.
        msa_cache_dir (PathLike | None): The directory to cache the parsed MSA data (since loading from text files is slow). If None, caching is turned off.
        use_paths_in_chain_info (bool): Whether to use the MSA paths provided in the chain_info dictionary. If True, we will first check the chain_info dictionary for MSA paths.
        raise_if_missing_msa_for_protein_of_length_n (int | None): If provided, raises an error if a protein of length >= n is missing an MSA file.
    Returns:
        dict[str, np.array]: A dictionary mapping chain IDs to their corresponding MSA data
    """
    msas_by_chain_id = {}

    # NOTE: If `msa_cache_dir` is `None`, the cache decorator will be a no-op
    cached_load_msa_data_from_path = cache_to_disk_as_pickle(msa_cache_dir)(load_msa_data_from_path)

    for chain_id in np.unique(atom_array.chain_id[np.isin(atom_array.chain_type, ChainType.get_polymers())]):
        non_canonical_sequence = chain_info[chain_id]["processed_entity_non_canonical_sequence"]
        canonical_sequence = chain_info[chain_id]["processed_entity_canonical_sequence"]
        chain_type = chain_info[chain_id]["chain_type"]

        # Set the query chain tax_id to "query" to avoid pairing issues downstream (we force all query sequences to be paired with themselves)
        # Subsequent occurrences of the query sequence will not have the "query" tax ID, and will be paired appropriately
        query_chain_msa_tax_id = "query"

        # ... find the path
        msa_file_path = None
        if (
            use_paths_in_chain_info
            and "msa_path" in chain_info[chain_id]
            and chain_info[chain_id]["msa_path"] is not None
        ):
            # Use provided path
            msa_file_path = Path(chain_info[chain_id]["msa_path"])
        else:
            # Check both canonical and non-canonical sequences
            for sequence in [non_canonical_sequence, canonical_sequence]:
                if chain_type.is_protein() and protein_msa_dirs:
                    msa_file_path = get_msa_path(sequence, protein_msa_dirs)
                    if msa_file_path is None and unk_symbol != "X":
                        sequence = sequence.replace("X", unk_symbol)
                        msa_file_path = get_msa_path(sequence, protein_msa_dirs)
                elif chain_type == ChainType.RNA and rna_msa_dirs:
                    msa_file_path = get_msa_path(sequence, rna_msa_dirs)
                    if not msa_file_path:
                        # Older MSAs replace U->T. If no matches try replacing
                        msa_file_path = get_msa_path(sequence.replace("U", "T"), rna_msa_dirs)
                if msa_file_path:
                    break

        if msa_file_path is None:
            # If no MSA file path is found, we skip this chain
            if raise_if_missing_msa_for_protein_of_length_n is not None:  # noqa: SIM102
                if chain_type.is_protein() and len(canonical_sequence) >= raise_if_missing_msa_for_protein_of_length_n:
                    raise ValueError(f"MSA file not found for protein of length {len(canonical_sequence)}")
            continue

        assert msa_file_path.exists(), f"MSA file not found at given path: {msa_file_path}"

        # ... load the MSA data from the specified path
        msa_data = cached_load_msa_data_from_path(
            msa_file_path=msa_file_path,
            chain_type=chain_type,
            max_msa_sequences=max_msa_sequences,
            query_tax_id=query_chain_msa_tax_id,
        )

        if msa_data["msa"] is not None:
            msas_by_chain_id[chain_id] = {
                **msa_data,
                "msa_is_padded_mask": np.zeros(msa_data["msa"].shape, dtype=bool),  # 1 = padded, 0 = not padded
            }

    return msas_by_chain_id


class LoadPolymerMSAs(Transform):
    """Load MSAs for all polymer chains in the AtomArray.

    For the MSAs that are found, store the MSA (as a np.array of integers), insertions,
    tax IDs, and pre-computed sequence similarities in `polymer_msas_by_chain_id`
    indexed by chain_id (e.g., "A").

    Note that MSAs may be found in two ways:
        (1) By loading from the MSA files on disk based on the sequence hash(e.g., for training data).
        (2) By using specific MSA paths provided in the chain_info dictionary (e.g., for inference).

    We check both the canonical and non-canonical sequences for MSAs, preferring the canonical sequence if both are present.

    Args:
        protein_msa_dirs (list[dict]): The directories containing the protein MSAs and
            their associated file types, as a list of dictionaries. If multiple
            directories are provided, all of them will be searched. Keys in the dictionary
            are:
                - dir (str): The directory where the MSA files are stored.
                - extension (str): The file extension of the MSA files (e.g., ".a3m.gz" or ".fasta").
                - directory_depth (int, optional): The directory nesting depth, i.e., the MSA file
                  might be stored at `dir/d8/07/d8074f77ba.a3m.gz`. Must be sharded
                  by the first two characters of the sequence hash. Defaults to 0 (flat directory).
            Note:
                (a) The files must be named using the SHA-256 hash of the sequence (see `hash_sequence` in
                    `utils/misc`).
                (b) Order matters - directories will be searched in the order provided, and the first match will be returned.
        rna_msa_dirs (list[dict]): The directories containing the RNA MSAs and their
            associated file types, as a list of dictionaries. See `protein_msa_dirs`
            for directory structure details.
        use_paths_in_chain_info (bool): Whether to use the MSA paths provided in the chain_info dictionary.
            E.g., for inference mode. If True, we will first check the chain_info dictionary for MSA paths.
        max_msa_sequences (int, optional): The maximum number of sequences to load from
            the MSA files. Defaults to 10000. Only applies when loading; further
            sub-sampling of the MSA occurs downstream (e.g., for the standard or extra MSA stack).
            AF-3 used a large value (~16K), but our MSAs on disk are already pre-filtered to 10K.
        msa_cache_dir (PathLike, optional): The directory to cache the parsed MSA data
            (since loading from text files is slow). If None, caching is turned off.
        raise_if_missing_msa_for_protein_of_length_n (int | None): If provided, raises an error if a protein of length >= n is missing an MSA file.
        unk_symbol (string): The character to use for unknown residues.  Defaults to 'X'.

    The `polymer_msas_by_chain_id` dictionary which is added contains the following keys:
        - msa: The MSA as a 2D np.array of integers, using the encoding specified in
          `_msa_constants.py`. Note that this encoding is transitory and will be
          converted to model-specific token indices later.
        - ins: The insertion array for the MSA, indicating the number of insertions to
          the LEFT of a given index, stored as a 2D np.array of integers.
        - tax_ids: The taxonomic IDs for each sequence in the MSA, stored as a 1D
          np.array of strings.
        - sequence_similarity: The sequence similarity to the query sequence for each
          row in the MSA.
        - msa_is_padded_mask: A mask indicating whether a given position in the MSA is
          padded (0) or not (1); defaults to 1 for all positions. Used downstream when
          filling the full MSA from the encoded MSA.
    """

    max_msa_sequences: int
    protein_msa_dirs: list[dict]
    rna_msa_dirs: list[dict]

    def __init__(
        self,
        protein_msa_dirs: list[
            dict
        ] = [],  # Example: [{"dir": "/path/to/protein/msas", "extension": ".a3m.gz", "directory_depth": 2}]
        rna_msa_dirs: list[dict] = [],
        max_msa_sequences: int = 10000,
        msa_cache_dir: PathLike | None = None,
        use_paths_in_chain_info: bool = True,
        raise_if_missing_msa_for_protein_of_length_n: int | None = None,
        unk_symbol: str = "X",
    ):
        self.max_msa_sequences = max_msa_sequences
        self.protein_msa_dirs = protein_msa_dirs
        self.rna_msa_dirs = rna_msa_dirs
        self.msa_cache_dir = msa_cache_dir
        self.use_paths_in_chain_info = use_paths_in_chain_info
        self.raise_if_missing_msa_for_protein_of_length_n = raise_if_missing_msa_for_protein_of_length_n
        self.unk_symbol = unk_symbol

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "chain_id"])

    def forward(self, data: dict) -> dict:
        polymer_msas_by_chain_id = load_polymer_msas(
            atom_array=data["atom_array"],
            chain_info=data["chain_info"],
            protein_msa_dirs=self.protein_msa_dirs,
            rna_msa_dirs=self.rna_msa_dirs,
            max_msa_sequences=self.max_msa_sequences,
            msa_cache_dir=self.msa_cache_dir,
            use_paths_in_chain_info=self.use_paths_in_chain_info,
            raise_if_missing_msa_for_protein_of_length_n=self.raise_if_missing_msa_for_protein_of_length_n,
            unk_symbol=self.unk_symbol,
        )
        data["polymer_msas_by_chain_id"] = polymer_msas_by_chain_id

        return data


class EncodeMSA(Transform):
    """
    Encode a MSA from MSA-general integer representations to model-specific token indices using the Enoding.

    Args:
        token_encoding (TokenEncoding): The TokenEncoding object to use for encoding the MSA.
        token_to_use_for_gap (int): The (integer) token to use for gaps in the MSA.
            If None, we leave the gap token (<G>) and will raise an error if it is not present in the TokenEncoding.
            NOTE: RF2AA converts gap tokens to padding tokens (UNK), whereas AF-3 uses a separate gap token (e.g., keep as None).

    NOTE:
        - The input MSA is expected to be in integer format, where each integer corresponds
          to a specific amino acid or nucleotide as defined in AMINO_ACID_ONE_LETTER_TO_INT
          and RNA_NUCLEOTIDE_ONE_LETTER_TO_INT.
        - The output MSA will have integers corresponding to the token indices defined
          in the TokenEncoding.
        - The lookup_for_encoding table handles the mapping from MSA integers to
          TokenEncoding indices.
    """

    requires_previous_transforms: ClassVar[list[str]] = ["LoadPolymerMSAs"]

    def __init__(self, encoding: TokenEncoding | AF3SequenceEncoding, token_to_use_for_gap: int | None = None):
        # ... create a lookup table to map from MSA integers to token indices
        lookup_for_encoding = np.zeros(len(MSA_INTEGER_TO_THREE_LETTER), dtype=int)
        for tmp_int, three_letter in MSA_INTEGER_TO_THREE_LETTER.items():
            if three_letter == GAP_THREE_LETTER and token_to_use_for_gap is not None:
                # ... if we defined a substitute token for gaps, use it
                lookup_for_encoding[tmp_int] = token_to_use_for_gap
            else:
                # ... otherwise, we assume that the gap token is present in the encoding
                lookup_for_encoding[tmp_int] = encoding.token_to_idx[three_letter]

        self.lookup_for_encoding = lookup_for_encoding
        self.token_to_use_for_gap = token_to_use_for_gap

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["polymer_msas_by_chain_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # ...loop through all of the polymer chain IDs still present in the atom array (may be a subset of `polymer_msas_by_chain_id`)
        polymer_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer])
        for chain_id in polymer_chain_ids:
            # ...check if we have an MSA for this chain
            if chain_id in data["polymer_msas_by_chain_id"]:
                msa = data["polymer_msas_by_chain_id"][chain_id]["msa"]
                # ...encode the MSA (to tokens integers), based on the lookup table
                encoded_msa = self.lookup_for_encoding[msa]  # [n_rows, n_res_in_chain] (int)
                # ...set the encoded MSA in the output in-place
                data["polymer_msas_by_chain_id"][chain_id]["encoded_msa"] = encoded_msa

        return data


class FillFullMSAFromEncoded(Transform):
    """
    Fills in the full MSA from the encoded MSA, using the atom array to determine the order of the tokens.
    Starts by creating full np.arrays with default values (padding tokens), and then fills in the encoded MSA by looping over chain instances.

    This function requires that all MSAs have the same number of rows, but does not require them to necessarily be paired.

    Specifically:
        - If we cropped or otherwise removed residues, we drop them from the MSA (via indexing) to ensure the MSA is consistent with the atom array.
        - If we atomized residues, we drop the atomized pieces from the MSA, and include only the encoded atomized tokens.

    Attributes:
        pad_token (str): The token used for padding in the MSA. The pad token should match the padding token used when padding unpaired MSA sequences.
        add_residue_is_paired_feature (bool): Whether to add a binary feature indicating whether a residue is part of a paired MSA.
            Must match the value used in PairAndMergePolymerMSAs.

    Returns:
        The full MSA, with padding, as a 2D np.array of integers, stored in `data["encoded"]["msa"]`.
        Additionally, we store the following details in `data["full_msa_details"]`:
            - token_idx_has_msa (np.array): A mask indicating whether a given token has an MSA (1) or not (0).
            - msa_is_padded_mask (np.array): A mask indicating whether a given position in the MSA is padded (1) or not (0).
            - msa_raw_ins (np.array): The raw insertion counts for the MSA, before encoding.

    Example:
        If the atom array token order is:
        ```
        [
            Chain A, Residue 1 (A),
            Chain A, Residue 2 (R) [atomized, covalent modification],
            Chain A, Residue 3 (C),
            Chain B, Residue 1 (glycan) [atomized, non-polymer]
        ]
        ```
        And the MSA for Chain A is:
        ```
        [["A", "R", "C"], ["A", "R", "D"]]
        ```
        Then the expected `data["encoded"]["msa"]` would be:
        ```
        [
            [ "A", "R_1", "R_2", "C", "B_1", "B_2" ],
            [ "A", <PAD>, <PAD>, "D", <PAD>, <PAD> ]
        ]
        ```
        Where "R_1", "R_2" are the atomized tokens for the residue, and "B_1", "B_2" are the atomized tokens for the glycan.
        NOTE: Amino acids are represented as letters for clarity; in reality, they would be tokens (integers).

        The expected `data["full_msa_details"]` would be:
        ```
        {
            "token_idx_has_msa": [1, 0, 0, 1, 0, 0],
            "msa_is_padded_mask": [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1]],
            "msa_raw_ins": [...],  # Not shown
        }
        ```
    """

    requires_previous_transforms: ClassVar[list[str]] = ["EncodeMSA", AtomizeByCCDName, AddWithinPolyResIdxAnnotation]

    def __init__(self, pad_token: str, add_residue_is_paired_feature: bool = False):
        self.PAD_TOKEN = pad_token
        self.add_residue_is_paired_feature = add_residue_is_paired_feature

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["polymer_msas_by_chain_id", "encoded"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # If we have no polymer MSAs (either all polymer sequences have no MSAs, or we have no polymer sequences)...
        if len(data["polymer_msas_by_chain_id"]) == 0:
            # ... we set `full_encoded_msa` to be the query sequence (expanded to 2D)
            full_encoded_msa = np.expand_dims(data["encoded"]["seq"], axis=0)  # [1, n_tokens_across_chains] (int)
            num_tokens_in_example = full_encoded_msa.shape[1]
            data["full_msa_details"] = {
                # ... we set `token_idx_has_msa` to all zeros
                "token_idx_has_msa": np.zeros(num_tokens_in_example, dtype=bool),
                # ... we set `msa_is_padded_mask` to all zeros
                "msa_is_padded_mask": np.zeros((1, num_tokens_in_example), dtype=bool),
                # ... we set `msa_raw_ins` to all zeros
                "msa_raw_ins": np.zeros((1, num_tokens_in_example), dtype=int),
            }
            if self.add_residue_is_paired_feature:
                # ... we set `msa_residue_is_paired` to all zeros
                data["full_msa_details"]["msa_residue_is_paired"] = np.zeros((1, num_tokens_in_example), dtype=bool)
            data["encoded"]["msa"] = full_encoded_msa
            # ...and we early return!
            return data

        # If we have polymer MSAs, figure out how many rows we need in the full MSA (n_rows)...
        first_chain_id = next(
            iter(data["polymer_msas_by_chain_id"])
        )  # All chains have the same number, as defined in the function definition...
        assert all(
            data["polymer_msas_by_chain_id"][chain_id]["encoded_msa"].shape[0]
            == data["polymer_msas_by_chain_id"][first_chain_id]["encoded_msa"].shape[0]
            for chain_id in data["polymer_msas_by_chain_id"]
        )  # ... but we check anyways
        first_encoded_msa = data["polymer_msas_by_chain_id"][first_chain_id]["encoded_msa"]
        n_rows = first_encoded_msa.shape[0]

        # Check that the given padding token matches the padding token used when padding unpaired MSA sequences, if applicable
        existing_pad_tokens = data["polymer_msas_by_chain_id"][first_chain_id]["msa_is_padded_mask"] * first_encoded_msa
        if np.any(existing_pad_tokens):
            token = existing_pad_tokens.flat[np.flatnonzero(existing_pad_tokens)[0]]
            assert (
                token == self.PAD_TOKEN
            ), f"Given padding token {self.PAD_TOKEN} does not match existing padding token {token}"

        # Set up empty encoded msa (purely padded) for encoded msa
        token_count = get_token_count(atom_array)
        full_encoded_msa = np.full(
            (n_rows, token_count), self.PAD_TOKEN, dtype=int
        )  # [n_rows, n_tokens_across_chains] (int)
        full_msa_is_padded_mask = np.ones(
            (n_rows, token_count), dtype=bool
        )  # [n_rows, n_tokens_across_chains] (bool) 1 = padded, 0 = not padded
        full_msa_ins = np.zeros((n_rows, token_count), dtype=int)  # [n_rows, n_tokens_across_chains] (int)

        # Sanity check on residue pairing feature
        contains_residue_pairing_feature = "residue_is_paired" in data["polymer_msas_by_chain_id"][first_chain_id]
        if self.add_residue_is_paired_feature != contains_residue_pairing_feature:
            raise ValueError(
                f"The residue pairing feature is {'not' if not contains_residue_pairing_feature else ''} present in the input MSA, "
                f"but `add_residue_is_paired_feature` is set to {self.add_residue_is_paired_feature}."
            )

        # Instantiate residue pairing feature if requested
        if self.add_residue_is_paired_feature:
            full_msa_residue_is_paired = np.zeros(
                (n_rows, token_count), dtype=bool
            )  # [n_rows, n_tokens_across_chains] (bool)

        # Create a mask indicating whether any atom in each token is atomized
        is_token_atomized = apply_token_wise(atom_array, atom_array.atomize, np.any)  # [n_tokens_across_chains] (bool)

        # Loop through all `chain_iids` (polymer/non-polymer) and populate relevant columns in encoding
        token_idx_has_msa = np.zeros(get_token_count(atom_array), dtype=bool)  # [n_tokens_across_chains] (bool)
        for chain_iid in np.unique(atom_array.chain_iid):
            is_atom_in_chain = atom_array.chain_iid == chain_iid  # [n_atoms_total] (bool)

            chain_instance_atom_array = atom_array[is_atom_in_chain]
            chain_id = chain_instance_atom_array.chain_id[0]

            # Check if we have an MSA for this chain
            if chain_id in data["polymer_msas_by_chain_id"]:
                # ... if so, get the encoded MSA and the mask
                chain_encoded_msa = data["polymer_msas_by_chain_id"][chain_id][
                    "encoded_msa"
                ]  # [n_rows, n_res_in_chain] (int)
                msa_is_padded_mask = data["polymer_msas_by_chain_id"][chain_id][
                    "msa_is_padded_mask"
                ]  # [n_rows, n_res_in_chain] (bool)
                msa_ins = data["polymer_msas_by_chain_id"][chain_id]["ins"]  # [n_rows, n_res_in_chain] (int)
                if self.add_residue_is_paired_feature:
                    residue_pairing = data["polymer_msas_by_chain_id"][chain_id][
                        "residue_is_paired"
                    ]  # [n_rows, n_res_in_chain] (bool)

                # ... create a global mask to indicate whether any atom in each token is in this chain
                global_is_token_in_chain = apply_token_wise(
                    atom_array, is_atom_in_chain, np.any
                )  # [n_tokens_across_chains] (bool)

                # ... index into the MSA, dropping the atomized pieces, and any residues that may have been cropped or otherwise removed
                non_atomized_atoms = chain_instance_atom_array[~chain_instance_atom_array.atomize]
                if len(non_atomized_atoms) == 0:
                    # (Skip if there are no non-atomized atoms in this chain)
                    continue

                within_poly_res_idx = non_atomized_atoms[
                    get_token_starts(non_atomized_atoms)
                ].within_poly_res_idx  # [n_non_atomized_res_in_chain] (int)
                subselected_encoded_msa = chain_encoded_msa[
                    :, within_poly_res_idx
                ]  # [n_rows, n_non_atomized_res_in_chain] (int)
                subselected_msa_is_padded_mask = msa_is_padded_mask[
                    :, within_poly_res_idx
                ]  # [n_rows, n_non_atomized_res_in_chain] (bool)
                subselected_msa_ins = msa_ins[:, within_poly_res_idx]  # [n_rows, n_non_atomized_res_in_chain] (int)
                if self.add_residue_is_paired_feature:
                    subselected_residue_pairing = residue_pairing[
                        :, within_poly_res_idx
                    ]  # [n_rows, n_non_atomized_res_in_chain] (bool)

                # ... set all non-atomized tokens in this chain (e.g., full residues) to the subselected MSA
                mask = global_is_token_in_chain & (~is_token_atomized)  # [n_tokens_across_chains] (bool)
                full_encoded_msa[:, mask] = subselected_encoded_msa  # [n_rows, n_tokens_across_chains] (int)
                full_msa_is_padded_mask[:, mask] = (
                    subselected_msa_is_padded_mask  # [n_rows, n_tokens_across_chains] (bool)
                )
                full_msa_ins[:, mask] = subselected_msa_ins  # [n_rows, n_tokens_across_chains] (int)
                if self.add_residue_is_paired_feature:
                    full_msa_residue_is_paired[:, mask] = subselected_residue_pairing
                token_idx_has_msa[mask] = True  # [n_tokens_across_chains] (bool)

        # ... for the first row, set the tokens directly from the output of the `Atomize` transform (i.e., the atomized tokens, and anything without an MSA)
        # (Note that this also handles setting the MSA for polymers without MSAs, and non-polymers)
        full_encoded_msa[0] = data["encoded"]["seq"]  # [n_tokens_across_chains] (int)
        full_msa_is_padded_mask[0] = False  # [n_tokens_across_chains] (bool)
        if self.add_residue_is_paired_feature:
            full_msa_residue_is_paired[0] = True  # [n_tokens_across_chains] (bool)

        data["encoded"]["msa"] = full_encoded_msa  # [n_rows, n_tokens_across_chains] (int)
        data["full_msa_details"] = {
            "token_idx_has_msa": token_idx_has_msa,  # [n_tokens_across_chains] (bool)
            "msa_is_padded_mask": full_msa_is_padded_mask,  # [n_rows, n_tokens_across_chains] (bool)
            # The insertions are not yet encoded (still raw counts), so we store them separately
            "msa_raw_ins": full_msa_ins,  # [n_rows, n_tokens_across_chains] (int)
        }

        if self.add_residue_is_paired_feature:
            data["full_msa_details"]["msa_residue_is_paired"] = full_msa_residue_is_paired

        return data


class FeaturizeMSALikeRF2AA(Transform):
    """Featurizes the MSA in the style of RF2AA, returning one featurized set of outputs for each recycle.

    Args:
        encoding (TokenEncoding): The encoding object to use for the MSA.
        n_recycles (int): The number of recycles to perform. We will generate a unique featurized MSA for each recycle.
        n_msa_cluster_representatives (int): The number of MSA cluster representatives to select. The remaining sequences (up to `n_extra_rows`) will be used as extra MSA.
        n_extra_rows (int): The number of extra MSA rows to use. If there are fewer than `n_extra_rows` remaining sequences, we will use all of them.
        mask_behavior_probs (dict): A dictionary containing the probabilities for each BERT-style mask behavior. The keys are:
            - "replace_with_random_aa": The probability of replacing a masked index with a uniformly random amino acid.
            - "replace_with_msa_profile": The probability of replacing a masked index with an amino acid sampled from the MSA profile.
            - "do_not_replace": The probability of keeping the original amino acid at a masked index.
            - The final probability, "replace_with_mask_token", is implicitly 1 - sum(probs.values()).
        mask_probability (float): The probability of masking a given token in the MSA.
        polymer_token_indices (torch.Tensor, optional): Tensor of token indices that correspond to polymer residues. If not provided, we assume all tokens are polymer residues. Used for optimization.

    For each recycle, performs four primary steps:
        (1) Select cluster representatives from the full MSA
        (2) Mask the cluster representatives using a BERT-style mask
        (3) Assign the remaining sequences to the cluster representatives
        (4) Summarize the clusters into profiles and mean insertions at each position

    Outputs:
        For each recycle, we store the following in `data["msa_features_per_recycle_dict"]` (inspired by AF-2 Supplement, Table 1):
        - "first_row_of_msa": The first row of the MSA, which is the query sequence.
        - "cluster_representatives_msa_masked": The (masked) MSA for the cluster representatives.
        - "cluster_representatives_has_insertion": A mask indicating whether a given position in the cluster representatives MSA has an insertion.
        - "cluster_representatives_insertion_value": The raw insertion value at each position in the cluster representatives MSA, transformed to [0,1]
        - "cluster_insertion_mean": The mean insertion value at each position in the cluster representatives MSA, transformed to [0,1]
        - "cluster_profile": The MSA profile for the cluster representatives.
        - "extra_msa": The MSA for the extra sequences.
        - "extra_msa_has_insertion": A mask indicating whether a given position in the extra MSA has an insertion.
        - "extra_msa_insertion_value": The raw insertion value at each position in the extra MSA, transformed to [0,1]
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "FillFullMSAFromEncoded",
        "EncodeMSA",
        ConvertToTorch,
    ]

    def __init__(
        self,
        *,
        encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
        n_recycles: int,
        n_msa_cluster_representatives: int,
        n_extra_rows: int,
        mask_behavior_probs: dict,
        mask_probability: float,
        polymer_token_indices: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        self.encoding = encoding
        self.n_recycles = n_recycles
        self.n_msa_cluster_representatives = n_msa_cluster_representatives
        self.n_extra_rows = n_extra_rows
        self.mask_behavior_probs = mask_behavior_probs
        self.mask_probability = mask_probability
        self.polymer_token_indices = polymer_token_indices
        self.eps = eps

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["encoded", "full_msa_details"])

    def forward(self, data: dict) -> dict:
        # (Unpack)
        encoded_msa = data["encoded"]["msa"]  # [n_rows, n_tokens_across_chains] (int)
        token_idx_has_msa = data["full_msa_details"]["token_idx_has_msa"]  # [n_tokens_across_chains] (bool)
        msa_is_padded_mask = data["full_msa_details"]["msa_is_padded_mask"]  # [n_rows, n_tokens_across_chains] (bool)
        msa_raw_ins = data["full_msa_details"]["msa_raw_ins"]  # [n_rows, n_tokens_across_chains] (int)

        # ...select either the first `n_msa_cluster_representatives` rows or all rows, whichever is smaller
        n_rows, n_seq = encoded_msa.shape
        n_msa_cluster_representatives = min(self.n_msa_cluster_representatives, n_rows)

        # ...compute the raw MSA profile, which is required for the BERT-style masking of the MSA, where with a 10% probability, we replace
        # an amino acid with an amino sampled from the MSA profile at a given position
        full_msa_profile = grouped_count(
            encoded_msa,
            mask=~msa_is_padded_mask,  # ... ignore padding when computing the profile
            groups=[
                torch.zeros(n_rows, dtype=torch.long),  # ... assign all sequences to the same group
                torch.arange(n_seq),  # ... assign each seq position to a different group
            ],
            n_tokens=self.encoding.n_tokens,  # ... return a float tensor
            dtype=torch.float,  # ... return a float tensor
        ).squeeze()  # [n_tokens_across_chains, n_tokens] (float)
        full_msa_profile /= (
            full_msa_profile.sum(dim=-1, keepdim=True) + self.eps
        )  # [n_tokens_across_chains, n_tokens] (float)

        # ...generate a unique MSA (both cluster representative MSA and extra MSA) for each recycle (up to `n_recycles`)
        msa_features_per_recycle_dict = {
            "first_row_of_msa": [],  # [n_tokens_across_chains] (int)
            "cluster_representatives_msa_ground_truth": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)
            "cluster_representatives_msa_masked": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)
            "cluster_representatives_has_insertion": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
            "cluster_representatives_insertion_value": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            "cluster_insertion_mean": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            "cluster_profile": [],  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)
            "extra_msa": [],  # [n_not_selected_rows, n_tokens_across_chains] (int)
            "extra_msa_has_insertion": [],  # [n_not_selected_rows, n_tokens_across_chains] (bool)
            "extra_msa_insertion_value": [],  # [n_not_selected_rows, n_tokens_across_chains] (float)
            "bert_mask_position": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
        }
        for _ in range(self.n_recycles):
            # ============================================================
            # (1) SELECT CLUSTER REPRESENTATIVES FROM THE FULL MSA
            # ============================================================

            # Select the MSA cluster representatives using the preferred sampling strategy
            selected_indices, not_selected_indices = uniformly_select_rows(
                n_rows, n_msa_cluster_representatives, preserve_first_index=True
            )

            # ...correct `token_idx_has_msa` for edge cases
            # (If we only have one selected sequence, then we don't actually have any MSA's)
            if selected_indices.numel() == 1 and not_selected_indices.numel() == 0:
                token_idx_has_msa = torch.zeros_like(token_idx_has_msa, dtype=torch.bool)

            # ============================================================
            # (2) MASK THE CLUSTER REPRESENTATIVES WITH BERT-STYLE MASK
            # ============================================================

            # ...mask the MSA, using the BERT-style approach from AF2
            # (We only apply the mask to the cluster representatives, not the extra MSA)
            index_can_be_masked = build_msa_index_can_be_masked(
                msa_is_padded_mask=msa_is_padded_mask,
                token_idx_has_msa=token_idx_has_msa,
                encoded_msa=encoded_msa,
                encoding=self.encoding,
            )  # [n_rows, n_tokens_across_chains] (bool)
            mask_position = torch.zeros_like(encoded_msa, dtype=torch.bool)  # [n_rows, n_tokens_across_chains] (int)
            partial_masked_msa, partial_mask_position = mask_msa_like_bert(
                encoding=self.encoding,
                mask_behavior_probs=self.mask_behavior_probs,
                mask_probability=self.mask_probability,
                full_msa_profile=full_msa_profile,
                encoded_msa=encoded_msa[selected_indices],
                index_can_be_masked=index_can_be_masked[selected_indices],
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)

            # Clone the encoded MSA to avoid modifying the original...
            encoded_and_masked_msa = encoded_msa.detach().clone()

            # ...and update the masked positions
            encoded_and_masked_msa[selected_indices] = partial_masked_msa
            mask_position[selected_indices] = partial_mask_position

            # ============================================================
            # (3) ASSIGN THE EXTRA SEQUENCES TO THE CLUSTER REPRESENTATIVES
            # ============================================================

            # Define the tokens to ignore when clustering
            # NOTE: We would also need to ignore the gap token, if present in the encoding; we could consider having an encoding function to return "special" tokens to ignore.
            tokens_to_ignore = torch.tensor([self.encoding.token_to_idx[token_name] for token_name in ["<M>", "UNK"]])

            index_should_be_counted_mask = build_indices_should_be_counted_masks(
                encoded_msa=encoded_and_masked_msa,
                mask_position=mask_position,
                tokens_to_ignore=tokens_to_ignore,
                token_idx_has_msa=token_idx_has_msa,
            )  # [n_rows, n_tokens_across_chains] (bool)

            if not_selected_indices.numel() > 0:
                # ...if we have extra sequences, assign them to the cluster representatives
                assignments = assign_extra_rows_to_cluster_representatives(
                    cluster_representatives_msa=encoded_and_masked_msa[selected_indices],
                    clust_reps_should_be_counted_mask=index_should_be_counted_mask[selected_indices],
                    extra_msa=encoded_and_masked_msa[not_selected_indices],
                    extra_msa_should_be_counted_mask=index_should_be_counted_mask[not_selected_indices],
                )  # [n_not_selected_rows] (int)
            else:
                # ...if we have no extra sequences, we set the assignments to an empty tensor
                assignments = torch.tensor([], dtype=torch.int)

            # ============================================================
            # (4) SUMMARIZE THE CLUSTERS INTO PROFILES AND MEAN INSERTIONS AND SUBSELECT THE EXTA MSA
            # ============================================================
            msa_cluster_profiles = torch.zeros(
                (*encoded_and_masked_msa[selected_indices].shape, self.encoding.n_tokens), dtype=torch.float
            )  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)
            msa_cluster_mean_ins = torch.zeros_like(
                encoded_and_masked_msa[selected_indices], dtype=torch.float
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)

            # Summarize the clusters into profiles and mean insertions at each position. We can perform two optimizations here:
            # (1) We only need to summarize where we have MSA's; we will handle the atomized residues (and positions without MSAs) later
            # (2) We only need to worry about polymer tokens; all atom tokens for indices with MSAs will be zero by definition

            # ...determine the token indices we should be considering
            polymer_token_indices = (
                torch.arange(self.encoding.n_tokens, dtype=torch.long)
                if self.polymer_token_indices is None
                else self.polymer_token_indices
            )

            if torch.any(token_idx_has_msa):
                # ...compute the profiles and mean insertions for the cluster representatives
                msa_cluster_profiles_with_msas_poly_tokens, msa_cluster_mean_ins_with_msas = summarize_clusters(
                    encoded_msa=encoded_and_masked_msa[
                        :, token_idx_has_msa
                    ],  # Optimization 1: Only consider tokens with MSAs
                    msa_raw_ins=msa_raw_ins[:, token_idx_has_msa],
                    mask_position=mask_position[:, token_idx_has_msa],
                    assignments=assignments,
                    selected_indices=selected_indices,
                    not_selected_indices=not_selected_indices,
                    msa_is_padded_mask=msa_is_padded_mask[:, token_idx_has_msa],
                    n_tokens=polymer_token_indices.shape[
                        0
                    ],  # Optimization 2: Only consider non-atom tokens. NOTE: Hyper-specific to RF2AA; should be generalized in the future
                    eps=self.eps,
                )  # [n_msa_cluster_representatives, n_tokens_with_msas, n_polymer_tokens] (float), [n_msa_cluster_representatives, n_tokens_with_msas] (float)
            else:
                # ...if we have no tokens with MSAs, we set the profiles and mean insertions to zeros
                msa_cluster_profiles_with_msas_poly_tokens = torch.zeros(
                    (n_msa_cluster_representatives, 0, polymer_token_indices.shape[0]), dtype=torch.float
                )
                msa_cluster_mean_ins_with_msas = torch.zeros((n_msa_cluster_representatives, 0), dtype=torch.float)

            # ...if we used a subset of the tokens, we need to map the profiles (but not insertions, since those don't have a token dimension) back to the full token set, padding with zeros
            if polymer_token_indices.shape[0] < self.encoding.n_tokens:
                msa_cluster_profiles_with_msas = torch.zeros(
                    ((*tuple(msa_cluster_profiles_with_msas_poly_tokens.shape[:-1]), self.encoding.n_tokens)),
                    dtype=torch.float,
                )  # [n_msa_cluster_representatives, n_tokens_with_msas, n_tokens] (float)
                msa_cluster_profiles_with_msas[:, :, polymer_token_indices] = msa_cluster_profiles_with_msas_poly_tokens
            else:
                msa_cluster_profiles_with_msas = msa_cluster_profiles_with_msas_poly_tokens

            # ...fill in the profiles and mean insertions for the cluster representatives
            msa_cluster_profiles[:, token_idx_has_msa] = msa_cluster_profiles_with_msas
            msa_cluster_mean_ins[:, token_idx_has_msa] = msa_cluster_mean_ins_with_msas
            del msa_cluster_profiles_with_msas, msa_cluster_mean_ins_with_msas

            # Now, handle the atomized residues and positions without MSAs:
            # (a) For insertions, they should be zeros everywhere by definition, since we have no MSA (which is handled by the initialization)
            # (b) For profiles, they should be 1 for the index of the amino acid in the query sequence, and 0 elsewhere (e.g., one-hot encoding of the query sequence)
            query_sequence_no_msa_profile = (
                torch.nn.functional.one_hot(encoded_and_masked_msa[0, ~token_idx_has_msa], self.encoding.n_tokens)
                .unsqueeze(0)
                .float()
            )  # [1, n_tokens_without_msas, n_tokens] (float)
            non_query_no_msa_profile = torch.zeros(
                ((n_msa_cluster_representatives - 1, *tuple(query_sequence_no_msa_profile.shape[1:]))),
                dtype=torch.float,
            )  # [n_msa_cluster_representatives - 1, n_tokens_with_msas, n_tokens] (float)
            msa_cluster_profiles_without_msas = torch.cat(
                [query_sequence_no_msa_profile, non_query_no_msa_profile], dim=0
            )  # [n_msa_cluster_representatives, n_tokens_without_msas, n_tokens] (float)

            msa_cluster_profiles[:, ~token_idx_has_msa] = msa_cluster_profiles_without_msas
            del msa_cluster_profiles_without_msas, query_sequence_no_msa_profile, non_query_no_msa_profile

            # ...subselect the extra MSA rows

            # From AF2 Supplement, section 1.2.7:
            #   (...)
            #   4. The MSA sequences that have not been selected as cluster centres
            #   at step 1 are used to randomly sample N_{extra_seq} sequences
            #   without replacement. If there are less than N_{extra_seq} remaining
            #   sequences available, all of them are used.
            #   (...)
            if not_selected_indices.shape[0] >= self.n_extra_rows:
                # ...if we have enough extra sequences, we randomly sample `n_extra_rows` of them
                shuffled_indices = torch.randperm(not_selected_indices.shape[0])
                not_selected_indices = not_selected_indices[
                    shuffled_indices[: self.n_extra_rows]
                ]  # [n_extra_rows] (int)

            # ============================================================
            # (5) BUILD THE RETURN DICTIONARY
            # ============================================================

            # Sequence
            msa_features_per_recycle_dict["first_row_of_msa"].append(
                encoded_and_masked_msa[0]
            )  # [n_tokens_across_chains] (int)

            # ...without masks (ground truth for masked token recovery)
            msa_features_per_recycle_dict["cluster_representatives_msa_ground_truth"].append(
                encoded_msa[selected_indices]
            )

            # +------- Information about the msa cluster representatives (NOT the clusters themselves) -------+
            # ...with masks
            msa_features_per_recycle_dict["cluster_representatives_msa_masked"].append(
                encoded_and_masked_msa[selected_indices]
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)

            # ...insertions
            msa_features_per_recycle_dict["cluster_representatives_has_insertion"].append(
                msa_raw_ins[selected_indices] > 0
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
            msa_features_per_recycle_dict["cluster_representatives_insertion_value"].append(
                transform_ins_counts(msa_raw_ins[selected_indices])
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)

            # +------- Aggregated information about the msa clusters (e.g, profiles, insertions) -------+
            msa_features_per_recycle_dict["cluster_insertion_mean"].append(
                transform_ins_counts(msa_cluster_mean_ins)
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            msa_features_per_recycle_dict["cluster_profile"].append(
                msa_cluster_profiles
            )  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)

            # +------- Information about the extra MSA -------+
            # NOTE: At a minimum, the extra MSA will contain the query sequence (a RF2AA novelty)
            extra_msa = encoded_and_masked_msa[not_selected_indices]
            if extra_msa.shape[0] > 0:
                # ...replace the first row of the extra MSA with the (masked) query sequence (a RF2AA novelty)
                extra_msa[0] = encoded_and_masked_msa[0]
            else:
                # ...if there's no extra MSA, we need to create a dummy row with the query sequence (a RF2AA novelty)
                extra_msa = encoded_and_masked_msa[0].unsqueeze(0)

            msa_features_per_recycle_dict["extra_msa"].append(extra_msa)  # [n_extra_rows, n_tokens_across_chains] (int)

            # ...insertions
            msa_features_per_recycle_dict["extra_msa_has_insertion"].append(
                msa_raw_ins[not_selected_indices] > 0
                if not_selected_indices.shape[0] > 0
                else torch.zeros_like(extra_msa, dtype=torch.bool)
            )  # [n_extra_rows, n_tokens_across_chains] (bool)
            msa_features_per_recycle_dict["extra_msa_insertion_value"].append(
                transform_ins_counts(msa_raw_ins[not_selected_indices])
                if not_selected_indices.shape[0] > 0
                else torch.zeros_like(extra_msa, dtype=torch.float)
            )  # [n_extra_rows, n_tokens_across_chains] (float)

            # +------- Mask information -------+
            msa_features_per_recycle_dict["bert_mask_position"].append(
                mask_position[selected_indices]
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)

        data["features_per_recycle_dict"] = msa_features_per_recycle_dict
        return data


class FeaturizeMSALikeAF3(Transform):
    """
    Featurizes the MSA in the style of AF3, returning one featurized set of outputs for each recycle.

    From the AF3 supplement, the MSA features are:

        | Feature         | Shape                  | Description                                                           |
        |-----------------|------------------------|-----------------------------------------------------------------------|
        | msa             | [N_msa, N_token, 32]   | One-hot encoding of the processed MSA, using the same classes as      |
        |                 |                        | restype.                                                              |
        | has_deletion    | [N_msa, N_token]       | Binary feature indicating if there is a deletion to the left of each  |
        |                 |                        | position in the MSA.                                                  |
        | deletion_value  | [N_msa, N_token]       | Raw deletion counts (the number of deletions to the left of each MSA  |
        |                 |                        | position) are transformed to [0, 1] using 2/π * arctan(d/3).          |
        | profile         | [N_token, 32]          | Distribution across restypes in the main MSA. Computed before MSA     |
        |                 |                        | processing (subsection 2.3).                                          |
        | deletion_mean   | [N_token]              | Mean number of deletions at each position in the main MSA. Computed   |
        |                 |                        | before MSA processing (subsection 2.3). Like `deletion_value`,        |
        |                 |                        | the mean deletions are transformed to [0, 1].                         |

    NOTE: The statement "Computed before MSA processing" is somewhat ambiguous; we interpret it as meaning that the DeepMind team computed the full profile
    and deletion mean before truncating the MSA to their `N_msa = 16,384` limit. We will adhere as closely to this interpretation as possible;
    however, our MSA's stored on disk only contain a maximum of 10,000 sequences, so we will compute the profile and deletion mean based on this limit.

    NOTE: We use "N_token_across_chains" to refer to the number of tokens across all chains in the MSA, including atomized tokens; AF-3 refers to this as "N_token".
    Meanwhile, we use "N_tokens" to refer to the number of residue tokens in the encoding; AF-3 directly refers to this number as "32".

    Initialization arguments:
        encoding (AF3SequenceEncoding): The encoding object to use for the MSA. For AF-3, this should include the 32 classes used in the restype encoding.
        n_recycles (int): The number of recycles to perform. We will generate a unique featurized MSA for each recycle.
        n_msa (int): The number of MSA sequences to flow into the model. If there are fewer than `n_msa` sequences, we will use all of them.

    Outputs:
        For each recycle, we store the following in `data["msa_features_per_recycle_dict"]`:
            - "msa": Shape [n_msa, n_tokens_across_chains, n_tokens]. One-hot encoding of the (possibly truncated) MSA, using the same classes as restype.
            - "has_insertion": Shape [n_msa, n_tokens_across_chains]. Binary feature indicating if there is an insertion to the left of each position in the MSA.
            - "insertion_value": Shape [n_msa, n_tokens_across_chains]. Raw insertion counts (the number of insertions to the left of each MSA position) are transformed to [0, 1] using 2/π * arctan(i/3).
        We also store the following in `data["msa_static_features_dict"]`, which do not change across recycles:
            - "profile": Shape [n_tokens_across_chains, n_tokens]. Distribution across restypes in the main MSA. Computed before MSA truncation.
            - "insertion_mean": Shape [n_tokens_across_chains]. Mean number of insertions to the left of each position in the main MSA. Computed before MSA truncation.

    Reference:
        `AF3 Supplement, Table 5 <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "FillFullMSAFromEncoded",
        "EncodeMSA",
        ConvertToTorch,
    ]

    def __init__(
        self,
        *,
        encoding: AF3SequenceEncoding,
        n_recycles: int,
        n_msa: int,
        eps: float = 1e-6,
    ):
        self.encoding = encoding
        self.n_recycles = n_recycles
        self.n_msa = n_msa
        self.eps = eps

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["encoded", "full_msa_details"])

    def forward(self, data: dict) -> dict:
        # ...unpack the MSA data
        encoded_msa = data["encoded"]["msa"]  # [n_rows, n_tokens_across_chains] (int)
        msa_is_padded_mask = data["full_msa_details"]["msa_is_padded_mask"]  # [n_rows, n_tokens_across_chains] (bool)

        # ...get the MSA features
        msa_features = featurize_msa_like_af3(
            encoded_msa=encoded_msa,
            n_recycles=self.n_recycles,
            msa_is_padded_mask=msa_is_padded_mask,
            msa_raw_ins=data["full_msa_details"]["msa_raw_ins"],
            n_msa=self.n_msa,
            encoding=self.encoding,
            eps=self.eps,
            residue_is_paired=data["full_msa_details"].get("msa_residue_is_paired", None),
        )

        # ...add them to the data dictionary
        data["msa_features"] = msa_features
        return data


def featurize_msa_like_af3(
    encoded_msa: torch.Tensor,
    msa_is_padded_mask: torch.Tensor,
    msa_raw_ins: torch.Tensor,
    n_msa: int,
    encoding: AF3SequenceEncoding,
    n_recycles: int,
    eps: float = 1e-6,
    residue_is_paired: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Functional version of FeaturizeMSALikeAF3. See FeaturizeMSALikeAF3 for more details."""
    # ...select either the first `n_msa` rows or all rows, whichever is smaller
    n_rows, _ = encoded_msa.shape
    n_msa = min(n_msa, n_rows)

    full_msa_profile, ins_mean = get_full_msa_profile_and_insertion_mean(
        encoded_msa=encoded_msa,
        msa_is_padded_mask=msa_is_padded_mask,
        msa_raw_ins=msa_raw_ins,
        encoding=encoding,
        eps=eps,
    )

    # ...generate features for each recycle
    msa_features_per_recycle_dict = {
        "msa": [],  # [n_msa, n_tokens_across_chains, n_tokens] (float)
        "has_insertion": [],  # [n_msa, n_tokens_across_chains] (bool)
        "insertion_value": [],  # [n_msa, n_tokens_across_chains] (float)
    }

    if exists(residue_is_paired):
        msa_features_per_recycle_dict["residue_is_paired"] = []

    for _ in range(n_recycles):
        # ...uniformly select n_msa sequences from the n_rows sequences in the (paired) MSA
        selected_indices, _ = uniformly_select_rows(n_rows, n_msa, preserve_first_index=True)

        # ...fill in the MSA features
        msa_features_per_recycle_dict["msa"].append(
            F.one_hot(encoded_msa[selected_indices], num_classes=encoding.n_tokens)
        )
        msa_features_per_recycle_dict["has_insertion"].append(msa_raw_ins[selected_indices] > 0)
        msa_features_per_recycle_dict["insertion_value"].append(transform_ins_counts(msa_raw_ins[selected_indices]))
        if exists(residue_is_paired):
            msa_features_per_recycle_dict["residue_is_paired"].append(residue_is_paired[selected_indices])

    # ...and the features that do not differ across recycles
    msa_static_features_dict = {
        "profile": full_msa_profile,  # [n_tokens_across_chains, n_tokens] (float)
        "insertion_mean": ins_mean,  # [n_tokens_across_chains] (float)
    }

    return {
        "msa_features_per_recycle_dict": msa_features_per_recycle_dict,
        "msa_static_features_dict": msa_static_features_dict,
    }


def get_full_msa_profile_and_insertion_mean(
    encoded_msa: torch.Tensor,
    msa_raw_ins: torch.Tensor,
    msa_is_padded_mask: torch.Tensor,
    encoding: TokenEncoding | AF3SequenceEncoding,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the full MSA profile and insertion mean on the untruncated MSA"""
    # ...select either the first `n_msa` rows or all rows, whichever is smaller
    n_rows, n_seq = encoded_msa.shape

    # ...compute the FULL MSA token profile (e.g., before truncation to `n_msa` sequences)
    full_msa_profile = grouped_count(
        encoded_msa,
        mask=~msa_is_padded_mask,  # ... ignore padding when computing the profile
        groups=[
            torch.zeros(n_rows, dtype=torch.long),  # ... assign all sequences to the same group
            torch.arange(n_seq),  # ... assign each seq position to a different group
        ],
        n_tokens=encoding.n_tokens,  # ... return a float tensor
        dtype=torch.float,  # ... return a float tensor
    ).squeeze()  # [n_tokens_across_chains, n_tokens] (float)
    # ...normalize
    full_msa_profile /= full_msa_profile.sum(dim=-1, keepdim=True) + eps  # [n_tokens_across_chains, n_tokens] (float)

    # ...compute the FULL MSA deletion (insertion) profile (e.g., before truncation to `n_msa` sequences)
    ins_mean = (msa_raw_ins * ~msa_is_padded_mask).sum(dim=0).float()  # [n_tokens] (float)
    # ... normalize
    ins_mean /= (~msa_is_padded_mask).sum(dim=0) + eps  # [n_tokens] (float)

    return full_msa_profile, ins_mean
