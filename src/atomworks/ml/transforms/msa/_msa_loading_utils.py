import io
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import numba
import numpy as np

from atomworks.enums import ChainType, MSAFileExtension
from atomworks.io.utils.io_utils import apply_sharding_pattern, build_sharding_pattern
from atomworks.ml.transforms.msa._msa_constants import (
    AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
    RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
)
from atomworks.ml.utils.io import opened_file
from atomworks.ml.utils.misc import hash_sequence

# Marker for TaxID extraction
_TAXID_MARKER = "TaxID="
_TAXID_MARKER_LEN = len(_TAXID_MARKER)


def extract_tax_id(line: str, unknown_tax_id: str = "") -> str:
    """Extract taxonomy ID from the header line using string operations (faster than regex)."""
    # Example: ">UniRef100_A0A183IZU9 Kinesin-like protein n=1 Tax=Soboliphyme baturini TaxID=241478 RepID=A0A183IZU9_9BILA"
    idx = line.find(_TAXID_MARKER)
    if idx == -1:
        return unknown_tax_id

    # Extract substring after "TaxID=" and find where digits end
    start = idx + _TAXID_MARKER_LEN
    # Find next space or end of line - TaxID is always followed by space or EOL
    end = line.find(" ", start)
    if end == -1:
        end = len(line)

    tax_id = line[start:end].rstrip()
    return tax_id if tax_id else unknown_tax_id


def get_msa_format_from_extension(filename: PathLike) -> str:
    """Determine MSA format (a3m or fasta) from filename, ignoring compression.

    Args:
        filename: Path to the MSA file.

    Returns:
        Format string: either "a3m" or "fasta".
    """
    name = str(filename).lower()

    # Check a3m formats (including all compression variants)
    for ext in [MSAFileExtension.A3M_ZST, MSAFileExtension.A3M_GZ, MSAFileExtension.A3M]:
        if name.endswith(ext.value):
            return "a3m"

    # Check fasta formats (including all compression variants)
    for ext in [MSAFileExtension.AFA_ZST, MSAFileExtension.AFA_GZ, MSAFileExtension.AFA]:
        if name.endswith(ext.value):
            return "fasta"

    # Also support .fasta extension (common alternative to .afa)
    for ext in [".fasta.zst", ".fasta.gz", ".fasta"]:
        if name.endswith(ext):
            return "fasta"

    raise ValueError(f"Unsupported MSA file extension: {filename}")


def parse_msa(
    source: PathLike | str | None = None,
    maxseq: int = 10000,
    query_tax_id: str = "query",
    msa_format: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an MSA ``source`` into raw ``(msa, ins, tax_ids)`` arrays, routing by format.

    ``source`` may be a path or an open text stream; pass ``msa_format`` (``"a3m"``/``"fasta"``) for a stream,
    else it is inferred from the extension. Supports .a3m/.afa with optional .gz/.zst.
    """
    if msa_format is None:
        msa_format = get_msa_format_from_extension(source)

    if msa_format == "a3m":
        return parse_a3m(source, maxseq, query_tax_id)
    elif msa_format == "fasta":
        return parse_fasta(source, maxseq, query_tax_id)
    else:
        raise ValueError(f"Unsupported MSA format: {msa_format}")


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


def parse_fasta(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a FASTA (.afa or .fasta) file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.

    NOTE: As written, we do not handle insertions; we set the insertion array to all zeros. Currently, RNA MSAs do not have insertions.
    If that changes in the future, we will need to update this function.

    TODO: Update this function to handle insertions. Note that in FASTA files, insertions must be handled differently than in A3M files.
    For FASTA, we would need to remove all gaps from the query sequence, and consider non-gap characters in those columns as insertions.

    TODO: Deprecate and use Biotite to load FASTA files.

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

    with opened_file(filename) as raw_stream:
        fstream = remove_header_from_msa_file(raw_stream)

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


@numba.njit(cache=True)
def _count_insertions_and_extract_aligned(line_bytes: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Count insertions per aligned position and extract aligned sequence bytes.

    Insertions are lowercase letters (ASCII 97-122). Aligned characters are uppercase (65-90) or dash (45).
    Returns insertion counts per aligned position and the aligned sequence bytes.
    """
    ins = np.zeros(L, dtype=np.float64)
    aligned = np.empty(L, dtype=np.uint8)
    aligned_idx = 0

    for b in line_bytes:
        if 97 <= b <= 122:  # lowercase a-z = insertion
            ins[aligned_idx] += 1
        else:
            aligned[aligned_idx] = b
            aligned_idx += 1

    return ins, aligned


def parse_a3m(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads an A3M file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.

    A3M files differ from A2M files in that dots (".") are discarded for compactness; thus, lines may be different lengths (but we still have the same number of aligned columns).

    While parsing:
        - Keep track of number of insertions to the LEFT of each position.
        - Keep track of taxonomy IDs to support MSA pairing downstream.

    NOTE: Files must contain only ASCII characters; we do not handle Unicode characters.

    Args:
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
    tax_ids = []
    sequences_data = []  # List of (ins_array, aligned_bytes) tuples
    L = None  # Sequence length, determined from first sequence

    with opened_file(filename) as fstream:
        # Skip any lines before the first header (e.g., comments), then process the first header.
        # The first sequence is always the query, so we assign it `query_tax_id` directly instead
        # of extracting a TaxID from the header. This ensures proper MSA pairing downstream.
        for line in fstream:
            if line.startswith(">"):
                tax_ids.append(query_tax_id)
                break

        # Process remaining lines (headers get TaxID extracted, sequence lines get parsed)
        for line in fstream:
            # Skip empty lines and handle null bytes only if present
            if not line or line == "\n":
                continue
            if "\x00" in line:
                line = line.replace("\x00", "")
                if not line:
                    continue

            # Extract taxonomy ID from header lines
            if line[0] == ">":
                tax_ids.append(extract_tax_id(line))
                continue

            # Strip whitespace for sequence lines
            line = line.rstrip()
            if not line:
                continue

            # Convert to bytes for numba processing
            line_bytes = np.frombuffer(line.encode("ascii"), dtype=np.uint8)

            # Determine L from first sequence (count non-lowercase characters)
            if L is None:
                L = np.sum((line_bytes < 97) | (line_bytes > 122))

            # Process with numba-jitted function
            ins_row, aligned_row = _count_insertions_and_extract_aligned(line_bytes, L)
            sequences_data.append((ins_row, aligned_row))

            if len(sequences_data) >= maxseq:
                break

    # Build final arrays
    n_seqs = len(sequences_data)
    msa_array = np.empty((n_seqs, L), dtype="S1")
    ins_array = np.empty((n_seqs, L), dtype=np.float64)

    for i, (ins_row, aligned_row) in enumerate(sequences_data):
        msa_array[i] = aligned_row.view("S1")
        ins_array[i] = ins_row

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
        depth = msa_dir.get("directory_depth", 0)
        sharding_pattern = build_sharding_pattern(depth=depth, chars_per_dir=2)
        sharded_path = apply_sharding_pattern(sequence_hash, sharding_pattern)
        msa_file = Path(msa_dir["dir"]) / sharded_path.with_suffix(msa_dir["extension"])
        if msa_file.exists():
            return msa_file
    return None


def load_msa(
    source: PathLike | str | bytes,  # also accepts an open text stream
    chain_type: ChainType,
    max_msa_sequences: int = 10_000,
    query_tax_id: str = "query",
    *,
    suffix: str = MSAFileExtension.A3M,
) -> dict[str, np.array]:
    """Load model-ready MSA integer features from ``source`` — a file path, raw ``bytes``, or a text stream.

    Wraps :func:`parse_msa` (the low-level path/stream -> raw ``(msa, ins, tax_ids)`` parser) and converts to our
    integer representation via a ChainType-specific ASCII lookup (protein vs RNA one-letter codes overlap), plus a
    per-row query-similarity.

    Args:
        source: An MSA file path (may be gzipped/zstd), raw ``bytes``, or an open text stream.
        chain_type: The type of the chain (e.g., Protein or RNA).
        max_msa_sequences: The maximum number of sequences to read (for processing speed).
        query_tax_id: The taxonomy ID for the query sequence. Defaults to "query"; ensures the query
            sequence is paired with itself.
        suffix: Selects the a3m/fasta format for in-memory sources (bytes/stream). Paths infer it from
            the extension.
    """
    if isinstance(source, bytes | bytearray):
        source = io.StringIO(bytes(source).decode())
    # in-memory sources (bytes/stream) take the format from `suffix`; paths infer it from the extension
    msa_format = get_msa_format_from_extension(suffix) if hasattr(source, "read") else None
    msa, ins, tax_ids = parse_msa(source, maxseq=max_msa_sequences, query_tax_id=query_tax_id, msa_format=msa_format)

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
