import logging
import shutil
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from os import PathLike, devnull
from pathlib import Path

import pandas as pd

from atomworks.common import exists
from atomworks.ml.preprocessing.constants import NA_VALUES
from atomworks.ml.preprocessing.utils.fasta import create_fasta_file_from_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CoverageMode(IntEnum):
    """MMSeqs2 coverage modes for clustering. For details, see
    github.com/soedinglab/mmseqs2/wiki#how-to-set-the-right-alignment-coverage-to-cluster"""

    BIDIRECTIONAL = 0  # Alignment covers at least specified fraction of query and at least specified fraction of target
    TARGET = 1  # Alignment covers at least specified fraction of target
    QUERY = 2  # Alignment covers at least specified fraction of query
    TARGET_IN_QUERY = 3  # Target sequence length is at least specified fraction of query length
    QUERY_IN_TARGET = 4  # Query sequence length is at least specified fraction of target length
    SHORTER_IN_OTHER = 5  # Short sequence is at least specified fraction of the other sequence length


class ClusterMode(IntEnum):
    """MMSeqs2 cluster modes. For details, see github.com/soedinglab/mmseqs2/wiki#clustering-modes"""

    GREEDY_SET_COVER = 0  # Greedy set cover algorithm -- standard
    CONNECTED_COMPONENT = 1  # Connected component algorithm -- slower, but can cover more remote homologs
    GREEDY_INCREMENTAL = 2  # Greedy incremental algorithm


@dataclass
class MMSeqs2Config:
    """Input configuration for MMseqs2 clustering."""

    cluster_identity: float = 0.4  # Sequence identity threshold for clustering
    coverage: float = 0.8  # Coverage threshold: usage depends on coverage mode
    sensitivity: float = 8.0  # Sensitivity: ~1.0 faster; ~4.0 fast; ~7.5 sensitive
    cluster_mode: ClusterMode = ClusterMode.GREEDY_SET_COVER  # Clustering mode: see `ClusterMode` for details
    coverage_mode: CoverageMode = CoverageMode.BIDIRECTIONAL  # Coverage mode: see `CoverageMode` for details

    # Validating inputs
    def __post_init__(self):
        if self.cluster_identity < 0 or self.cluster_identity > 1:
            raise ValueError(f"Invalid cluster_identity {self.cluster_identity}: Must be in the range [0,1]")
        if self.coverage < 0 or self.coverage > 1:
            raise ValueError(f"Invalid coverage threshold {self.coverage}: Must be in the range [0,1]")


def run_mmseqs2_clustering(
    input_fasta: str | PathLike,
    clustering_config: MMSeqs2Config = MMSeqs2Config(),  # noqa: B008
    temp_dir: PathLike | str | None = None,
) -> pd.DataFrame:
    """Runs MMseqs2 clustering on the input FASTA file.

    Args:
        input_fasta (str): Path to the input FASTA file. The headers for the sequences should be unique, as they are used to identify the sequences in the output.
        clustering_config (MMSeqs2Config): Configuration for MMseqs2 clustering. See `MMSeqs2Config` for details.
        tmp_dir (PathLike | str): Path to the temporary directory where MMseqs2 will write intermediate files. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the clustering results with columns ["cluster_rep_seq_hash", "seq_hash"].

    Example:
        cluster_rep_seq_hash, seq_hash
        afe56282ba3, afe56282ba3
        afe56282ba3, ee1f80a23f3
        afe56282ba3, 4a2caa18797
        afe56282ba3, 19f7ce1eed1

    References:
        `PDB clustering approach <https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering>`_
        `MMseqs2 documentation <https://github.com/soedinglab/mmseqs2/wiki>`_
        CLI documentation for the `easy-cluster` command: `mmseqs easy-cluster -h`
    """
    # If input is a Path object, convert it to a string
    if isinstance(input_fasta, Path):
        input_fasta = str(input_fasta)

    # Ensure the temp directory exists and is a string
    temp_dir = str(Path.cwd() / "temp" if not exists(temp_dir) else Path(temp_dir))

    try:
        # Run MMseqs2 easy-cluster command
        logger.info(f"Running MMseqs2 easy-cluster with {clustering_config}...")
        with open(devnull, "w") as fnull:
            subprocess.run(
                [
                    "mmseqs",
                    "easy-cluster",
                    input_fasta,
                    "result",
                    temp_dir,
                    "--min-seq-id",
                    str(
                        clustering_config.cluster_identity
                    ),  # Sequence identity threshold for clustering, typically 0.4 for proteins, and 1.0 for nucleic acids and peptides
                    "-c",
                    str(clustering_config.coverage),  # Coverage threshold for clustering, typically 0.8
                    "-s",
                    str(
                        clustering_config.sensitivity
                    ),  # MMSeqs sensitivity for clustering: ~1.0 = faster, ~4.0 = fast, ~7.5 = sensitive
                    "--cluster-mode",
                    str(
                        int(clustering_config.cluster_mode)
                    ),  # 0 = standard, 1 = Connected component algorithm, slower but capable of covering more remote homologs. See https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering
                    "--cov-mode",
                    str(
                        int(clustering_config.coverage_mode)
                    ),  # Bi-directional coverage requirements (0) likely best for full-length proteins (but possible failure mode for fragment vs. full protein)
                ],
                check=True,
                stdout=fnull,
                stderr=fnull,
            )

        logger.info("Clustering completed! Parsing TSV output file into a pandas DataFrame...")

        current_dir = Path.cwd()
        cluster_file = current_dir / "result_cluster.tsv"

        # Load the TSV output file into a DataFrame
        df = pd.read_csv(
            cluster_file,
            sep="\t",
            header=None,
            names=["cluster_rep_seq_hash", "seq_hash"],
            keep_default_na=False,
            na_values=NA_VALUES,
        )

        logger.info(f"DataFrame created with {len(df)} rows!")

        return df

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while running MMseqs2: {e}")


def cluster_all_sequences(
    pn_units_df: PathLike | str | pd.DataFrame,
    sequence_col_name: str,
    sequence_hash_col_name: str,
    output_col_prefix: str = "q_pn_unit_cluster",
    set_to_cluster_col: bool = False,
    output_path: str | None = None,
    clustering_configs: list[MMSeqs2Config] = [MMSeqs2Config(), MMSeqs2Config(cluster_identity=1.0)],  # noqa: B008
) -> pd.DataFrame:
    """Clusters input sequences from a DataFrame and merges the cluster information back into the DataFrame.

    This function performs the following steps:
    1. Creates (deduplicated) FASTA files for sequences from the input DataFrame.
    2. Runs MMseqs2 clustering on the generated FASTA files for each specified parameter configuration.
    3. Merges the clustering results back into the original DataFrame.
    4. Saves the updated DataFrame with clustering information to a specified output path.

    Args:
        pn_units_df_path (PathLike | str | pd.DataFrame): Path to the input DataFrame stored as a Parquet file, or the DataFrame directly.
        sequence_col_name (str): The name of the column containing the canonical sequences to be clustered.
        sequence_hash_col_name (str): The name of the column containing the sequence hash for the sequences to be clustered.
        output_col_prefix (str): Prefix for the name of the output column containing the representative sequence hash for each cluster.
        set_to_cluster_col (bool): If True, sets the 'cluster' column of the output dataframe equal to last of the computed clusters.
        output_path (str | None): Path to save the output DataFrame with clustering information. If None, returns the dataframe without saving.
        clustering_configs (list[MMSeqs2Config]): List of MMSeqs2Config objects specifying the clustering configurations to run.


    Columns added to DataFrame:
        - {output_col_prefix}_{cluster_mode_str}_rep_seq_hash: Representative sequence hash for protein clusters with the given configuration.
        - <optional> cluster: If `set_to_cluster_col` is True, this column is set equal to the above column.

    Returns:
        None
    """
    current_dir = Path.cwd()
    temp_dir = current_dir / "tmp"
    mmseqs_fasta_path = temp_dir / "sequences_to_cluster.fasta"

    if not isinstance(pn_units_df, pd.DataFrame):
        pn_units_df = Path(pn_units_df)
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    # Create FASTA file for the dataframe, and save it in the temp directory
    logger.info("Creating FASTA file...")
    create_fasta_file_from_df(df, sequence_col_name, mmseqs_fasta_path)
    logger.info(f"FASTA file saved to {temp_dir}; will be cleaned up after clustering.")

    for config in clustering_configs:
        # Run MMSeqs2 clustering
        cluster_df = run_mmseqs2_clustering(
            str(mmseqs_fasta_path),
            config,
            temp_dir=temp_dir,
        )

        # Create a short string of the cluster mode for the column name
        cluster_mode_str = (
            f"(id:{float(config.cluster_identity)!s})"
            + f"(cov:{float(config.coverage)!s})"
            + f"(cov_mode:{int(config.coverage_mode)!s})"
        ).replace(".", ",")

        logger.info("Merging clustering information into the master DataFrame...")

        # Merge clusters into the master DataFrame

        # ...drop the `cluster_mode_str` col, if it already exists
        cluster_col = f"{output_col_prefix}_{cluster_mode_str}_rep_seq_hash"
        if cluster_col in df.columns:
            df = df.drop(columns=[cluster_col])

        # ...merge and rename
        df = (
            df.merge(
                cluster_df[["seq_hash", "cluster_rep_seq_hash"]],
                left_on=sequence_hash_col_name,
                right_on="seq_hash",
                how="left",
            )
            .rename(columns={"cluster_rep_seq_hash": cluster_col})
            .set_index(df.index)
        )
        logger.info(f"Merged clusters for {cluster_mode_str} configuration.")

        # If desired, set the 'cluster' column equal to the computed cluster
        if set_to_cluster_col:
            df["cluster"] = df[cluster_col]

        # Drop the redundant 'seq_hash' column from the merge
        if "seq_hash" in df.columns:
            df.drop(columns=["seq_hash"], inplace=True)

        logger.info(f"Clustering completed for {cluster_mode_str} configuration!")

    logger.info("Clustering complete!")
    if output_path is not None:
        # Save before cleaning up, in case of errors
        logger.info(f"Saving to {output_path}...")
        df.to_parquet(output_path, index=False)
        logger.info(f"DataFrame with clustering information saved to {output_path}")

    # Remove everything in the temp directory
    logger.info("Cleaning up...")
    try:
        shutil.rmtree(temp_dir)

        # Remove files created by MMseqs2 in the current directory
        current_dir = Path.cwd()
        filenames = ["result_cluster.tsv", "result_all_seqs.fasta", "result_rep_seq.fasta"]
        for filename in filenames:
            file_path = current_dir / filename
            if file_path.exists():
                file_path.unlink()

    except Exception as e:
        logger.error(f"Error removing temp directory {temp_dir}: {e}")

    if not exists(output_path):
        return df
