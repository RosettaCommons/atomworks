from os import PathLike
from pathlib import Path

import pandas as pd

from atomworks.ml.utils.misc import hash_sequence


def wrap_sequence(sequence: str, line_length: int = 80) -> str:
    """Wrap a sequence string to a specified line length.

    Args:
        sequence (str): The sequence string to wrap.
        line_length (int): The maximum line length. Default is 80.

    Returns:
        str: The wrapped sequence string.
    """
    return "\n".join(sequence[i : i + line_length] for i in range(0, len(sequence), line_length))


def create_fasta_file_from_df(
    pn_units_df: PathLike | str | pd.DataFrame, sequence_col_name: str, output_path: PathLike | str
) -> None:
    """Create a FASTA file from sequences stored as a dataframe in a Parquet file.

    Wraps sequences to 80 characters per line, filters null/all-X sequences, and deduplicates.
    For simple unwrapped FASTA from a list of sequences, see
    :py:func:`~atomworks.ml.preprocessing.msa.generating.create_fasta_with_hashed_headers`.

    Args:
        pn_units_df (pd.DataFrame | PathLike | str): Dataframe, as a path Parquet or object directly, containing a column with the sequences to be clustered.
        sequence_col_name (str): The name of the column containing the canonical sequences to be clustered.
        output_path (PathLike | str): Path to where the fasta file will be saved. Must end in .fasta extension.
    """
    # Load the pn_unit_df, if it is not already a DataFrame
    if not isinstance(pn_units_df, pd.DataFrame):
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    # Remove rows where the sequence is not given
    df = df[df[sequence_col_name].notnull()]

    # Remove rows where the sequence is all unknown ("X")
    df = df[df[sequence_col_name].apply(lambda x: not all(char == "X" for char in x))]

    # Create output directory if it does not exist
    output_path = Path(output_path)
    if output_path.suffix != ".fasta":
        raise ValueError("Output path must end in .fasta")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write all sequences to FASTA file, de-duplicating as we go
    seen_protein_hashes = set()
    with open(output_path, "w") as output_fasta_file:
        for sequence in df[sequence_col_name]:
            sequence_hash = hash_sequence(sequence)

            # Skip if we have already seen this sequence
            if sequence_hash in seen_protein_hashes:
                continue

            wrapped_sequence = wrap_sequence(sequence)
            output_fasta_file.write(f">{sequence_hash}\n{wrapped_sequence}\n")
            seen_protein_hashes.add(sequence_hash)
