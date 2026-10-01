"""Generic sequence clustering using MMseqs2."""

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from atomworks.ml.preprocessing.clustering.config import MMseqs2Config
from atomworks.ml.preprocessing.clustering.mmseqs2 import run_mmseqs2_clustering
from atomworks.ml.preprocessing.utils.fasta import create_fasta_file_from_df
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger(__name__)


def _hash_sequence_or_none(s: Any) -> str | None:
    """Hash sequence or return None if value is NA."""
    return hash_sequence(s) if pd.notna(s) else None


def add_cluster_column(
    df: pd.DataFrame,
    sequence_column: str,
    output_column: str = "cluster",
    config: MMseqs2Config | None = None,
) -> pd.DataFrame:
    """Add cluster column to DataFrame based on sequence similarity.

    Generic clustering - works with any DataFrame containing sequences.

    Args:
        df: DataFrame with sequences to cluster.
        sequence_column: Name of column containing sequences.
        output_column: Name of output cluster column.
        config: MMseqs2 clustering configuration. If None, uses default MMseqs2Config() (40% identity, 80% coverage).

    Returns:
        DataFrame with cluster column added.
    """
    if config is None:
        config = MMseqs2Config()

    df = df.copy()

    # Filter to rows with valid sequences
    has_seq = df[sequence_column].notna()
    if not has_seq.any():
        df[output_column] = None
        return df

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        fasta_path = work_dir / "sequences.fasta"

        # Create FASTA from sequences
        create_fasta_file_from_df(df[has_seq], sequence_column, fasta_path)

        # Run MMseqs2
        cluster_df = run_mmseqs2_clustering(fasta_path, config, work_dir)

        if cluster_df.empty:
            df[output_column] = None
            return df

        # Build hash -> cluster_rep lookup
        df["_seq_hash"] = df[sequence_column].apply(_hash_sequence_or_none)

        # Merge cluster assignments
        cluster_lookup = cluster_df.set_index("seq_hash")["cluster_rep_hash"].to_dict()
        df[output_column] = df["_seq_hash"].map(cluster_lookup)
        df = df.drop(columns=["_seq_hash"])

    logger.info(f"Added {output_column} column ({config.cluster_identity*100:.0f}% identity)")
    return df


def add_cluster_column_to_file(
    input_path: Path,
    sequence_column: str,
    output_column: str = "cluster",
    config: MMseqs2Config | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Add cluster column to parquet file.

    Convenience wrapper around add_cluster_column() for file I/O.
    """
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    df = add_cluster_column(df, sequence_column, output_column, config)

    out = output_path or input_path
    df.to_parquet(out, index=False)
    logger.info(f"Saved to {out}")
    return df
