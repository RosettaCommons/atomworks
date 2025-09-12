import re
import string
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import numpy as np

from atomworks.enums import ChainType
from atomworks.ml.transforms.msa._msa_constants import (
    AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
    RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
)
from atomworks.ml.utils.io import get_sharded_file_path, open_file
from atomworks.ml.utils.misc import hash_sequence


def extract_tax_id(line: str, unknown_tax_id: str = "") -> str:
    """Extract taxonomy ID from the header line"""
    # ...extract the TaxID from the header line
    # (Example line: ">UniRef100_A0A183IZU9 Kinesin-like protein n=1 Tax=Soboliphyme baturini TaxID=241478 RepID=A0A183IZU9_9BILA")
    match = re.search(r"TaxID=(\d+)", line)
    if match:
        return match.group(1)
    return unknown_tax_id  # (unknown tax ID, which must be handled correctly when pairing downstream)


def parse_msa(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Routes to the appropriate MSA parser based on the file extension."""
    if filename.name.endswith((".a3m", ".a3m.gz")):
        return parse_a3m(filename, maxseq, query_tax_id)
    elif filename.name.endswith((".afa", ".afa.gz", ".fasta", ".fasta.gz")):
        return parse_fasta(filename, maxseq, query_tax_id)
    else:
        raise ValueError(f"Unsupported MSA file extension: {filename.name}")


def remove_header_from_msa_file(fstream: Iterable[str]) -> Iterable[str]:
    """Skips lines in the file stream until the first line starting with '>'. Returns the rest of the lines."""
    for line in fstream:
        if line.startswith(">"):
            # Return the rest of the lines starting from the first header line
            yield line
            break

    # Continue yielding the rest of the lines
    for line in fstream:
        yield line


def parse_fasta(filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query") -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a FASTA (.afa or .fasta) file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.

    NOTE: As written, we do not handle insertions; we set the insertion array to all zeros. Currently, RNA MSAs do not have insertions.
    If that changes in the future, we will need to update this function.

    TODO: Update this function to handle insertions. Note that in FASTA files, insertions must be handled differently than in A3M files.
    For FASTA, we would need to remove all gaps from the query sequence, and consider non-gap characters in those columns as insertions.

    Args:
        filename (PathLike): The path to the FASTA file (can be gzipped).
        maxseq (int): The maximum number of sequences to read from the file (for processing speed).
        query_tax_id (str): The taxonomy ID for the query sequence.

    Returns:
        msa (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
        ins (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
        tax_ids (np.ndarray): Array of shape (N,) containing the taxonomy IDs for each sequence in the MSA.

    Reference:
        `UniProt FASTA Header Documentation <https://www.uniprot.org/help/fasta-headers>`_
    """
    msa = []
    ins = []
    tax_ids = []

    fstream = remove_header_from_msa_file(open_file(filename))

    for index, line in enumerate(fstream):
        # Extract taxonomy ID from the header line, but don't process like the rest of the MSA
        if line[0] == ">":
            if index == 0:
                # ...force the query sequence to have the query tax ID
                tax_ids.append(query_tax_id)  # query sequence
            else:
                # ...extract the TaxID from the header line
                tax_ids.append(extract_tax_id(line))

            # ...don't process the header line any further
            continue

        # ...remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # ...append to MSA (no lowercase letters in FASTA files that we need to remove)
        msa.append(line)

        # ...get the sequence length
        L = len(msa[-1])

        # HACK: There are never insertions in RNA MSAs, so we set the insertion array to all zeros
        i = np.zeros(L)
        ins.append(i)

        # ...break if we've reached the maximum number of sequences
        if len(msa) >= maxseq:
            break

    # ...convert lists to numpy arrays for return
    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    ins_array = np.array(ins)
    tax_ids_array = np.array(tax_ids)

    return msa_array, ins_array, tax_ids_array


def parse_a3m(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an A3M file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.
    A3M files differ from A2M files in that dots (".") are discarded for compactness; thus, lines may be different lengths (but we still have the same number of aligned columns).

    While parsing:
        - Keep track of number of insertions to the LEFT of each position.
        - Keep track of taxonomy IDs to support MSA pairing downstream.

    NOTE: Files must contain only ASCII characters; we do not handle Unicode characters.

    Parameters:
        filename (str or Path): The path to the A3M file, can be gzipped.
        maxseq (int): The maximum number of sequences to read from the file (for processing speed). Passed from the associated Transform.
        query_tax_id (str): The taxonomy ID for the query sequence.

    Returns:
        msa (np.ndarray):
            Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
            Each element is a byte string representing the amino acid or nucleotide at that position.
        ins (np.ndarray):
            Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
            Tracks the number of insertions (relative to the query sequence) before (to the LEFT of) an aligned column. If there's an
            insertion before a position, the value will be > 0; otherwise it will be 0. For the query sequence, this will be all zeros.
        tax_ids (np.ndarray):
            Array of shape (N,) containing the taxonomy IDs for each sequence in the MSA.

    Reference:
        `A3M Format Documentation <https://yanglab.qd.sdu.edu.cn/trRosetta/msa_format.html#a3m>`_
    """
    msa = []
    ins = []
    tax_ids = []

    # ...create a translation table to remove lowercase letters
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # ...open the file
    fstream = remove_header_from_msa_file(open_file(filename))

    for index, line in enumerate(fstream):
        # Extract taxonomy ID from the header line, but don't process like the rest of the MSA
        if line[0] == ">":
            if index == 0:
                # ...force the query sequence to have the query tax ID
                tax_ids.append(query_tax_id)  # query sequence
            else:
                # ...extract the TaxID from the header line
                tax_ids.append(extract_tax_id(line))

            # ...don't process the header line any further
            continue

        # ...remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # ...remove lowercase letters and append to MSA
        # (lowercase letters represent insertion positions between alignment columns)
        msa.append(line.translate(table))

        # ...get the sequence length
        # (since we removed lowercase letters, and we're using a3m without dot representations, all sequences should be the same length)
        L = len(msa[-1])

        # (0 - match or gap; 1 - insertion)
        a = np.array([0 if c.isupper() or c == "-" else 1 for c in line])
        i = np.zeros(L)

        if np.sum(a) > 0:
            # ...get the positions of insertions
            pos = np.where(a == 1)[0]

            # ...shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # ...get position of insertions in cleaned sequence and their length
            pos, num = np.unique(a, return_counts=True)

            # ...append to the matrix of insertions
            # (num represents the number of insertions to the LEFT of the index specified by pos)
            i[pos] = num

        ins.append(i)

        # ...break if we've reached the maximum number of sequences
        if len(msa) >= maxseq:
            break

    fstream.close()

    # ...convert lists to numpy arrays for return
    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    ins_array = np.array(ins)
    tax_ids_array = np.array(tax_ids)

    return msa_array, ins_array, tax_ids_array


def get_msa_path(seq: str, msa_dirs: list[dict[str, str]]) -> Path | None:
    """Retrieve the path to the MSA file for a given sequence.

    Args:
        seq (str): The one-letter sequence for which to find the MSA. May be a protein or RNA sequence.
        msa_dirs (list[dict[str, str]]): Dictionaries to search for the hashed sequence. Keys in the dictionary are:
            - dir (str): The directory where the MSA files are stored.
            - extension (str): The file extension of the MSA files (e.g., ".a3m.gz" or ".fasta").
            - directory_depth (int, optional): The directory nesting depth, i.e., the MSA file
                might be stored at `dir/d8/07/d8074f77ba.a3m.gz`. Defaults to 0 (flat directory).
            Note that the dictionaries are searched in order, and the first match is returned.
    """
    sequence_hash = hash_sequence(seq)
    for msa_dir in msa_dirs:
        msa_file = get_sharded_file_path(
            Path(msa_dir["dir"]), sequence_hash, msa_dir["extension"], msa_dir.get("directory_depth", 0)
        )
        if msa_file.exists():
            return msa_file
    return None


def load_msa_data_from_path(
    msa_file_path: PathLike, chain_type: ChainType, max_msa_sequences: int = 10_000, query_tax_id: str = "query"
) -> dict[str, np.array]:
    """Given an MSA file path and the corresponding chain type, load the MSA data.

    We must consider the ChainType to determine how to convert the MSA to our intermediate integer representation
    (since the single-letter representations of amino acids and nucleotides overlap)

    Args:
        msa_file_path (PathLike): The path to the MSA file (can be gzipped).
        chain_type (ChainType): The type of the chain (e.g., Protein or RNA).
        max_msa_sequences (int): The maximum number of sequences to read from the file (for processing speed).
        query_tax_id (str): The taxonomy ID for the query sequence. Defaults to "query"; ensures the query sequence is paired with itself.
    """
    # ... parse the MSA file (handles both A3M or FASTA formats)
    msa, ins, tax_ids = parse_msa(msa_file_path, maxseq=max_msa_sequences, query_tax_id=query_tax_id)

    # ... convert to integers, using the protein one-letter ASCII lookup table
    if chain_type.is_protein():
        msa = AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE[msa.view(np.uint8)]
    elif chain_type == ChainType.RNA:
        msa = RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE[msa.view(np.uint8)]
    else:
        raise ValueError(f"Unsupported chain type for MSAs: {chain_type}")

    # Sequence similarity to the query sequence for each row in the MSA
    sequence_similarity = (msa == msa[0:1]).mean(axis=1) if msa is not None else None

    return {"msa": msa, "ins": ins, "tax_ids": tax_ids, "sequence_similarity": sequence_similarity}
