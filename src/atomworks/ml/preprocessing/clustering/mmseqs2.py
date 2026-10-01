"""MMseqs2 wrapper for sequence clustering."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from atomworks.ml.executables.mmseqs2 import MMseqs2
from atomworks.ml.preprocessing.clustering.config import MMseqs2Config

logger = logging.getLogger(__name__)


def run_mmseqs2_clustering(
    input_fasta: Path,
    config: MMseqs2Config | None = None,
    working_dir: Path | None = None,
) -> pd.DataFrame:
    """Run MMseqs2 easy-cluster on input FASTA file.

    Args:
        input_fasta: Path to FASTA file (headers = sequence hashes).
        config: MMseqs2 clustering configuration.
        working_dir: Directory for MMseqs2 files. Uses temp dir if None.

    Returns:
        DataFrame with columns [cluster_rep_hash, seq_hash].
    """
    if config is None:
        config = MMseqs2Config()

    use_temp = working_dir is None
    work_path = Path(tempfile.mkdtemp()) if use_temp else working_dir
    work_path.mkdir(parents=True, exist_ok=True)

    try:
        result_prefix = work_path / "result"
        tmp_dir = work_path / "tmp"

        logger.info(f"Running MMseqs2 with identity={config.cluster_identity}")
        mmseqs_bin = str(MMseqs2.get_or_initialize().get_bin_path())
        subprocess.run(
            [
                mmseqs_bin,
                "easy-cluster",
                str(input_fasta),
                str(result_prefix),
                str(tmp_dir),
                "--min-seq-id",
                str(config.cluster_identity),
                "-c",
                str(config.coverage),
                "-s",
                str(config.sensitivity),
                "--cluster-mode",
                str(int(config.cluster_mode)),
                "--cov-mode",
                str(int(config.coverage_mode)),
            ],
            check=True,
            capture_output=True,
        )

        cluster_tsv = work_path / "result_cluster.tsv"
        df = pd.read_csv(
            cluster_tsv,
            sep="\t",
            header=None,
            names=["cluster_rep_hash", "seq_hash"],
        )
        logger.info(f"Clustered {len(df)} sequences")
        return df

    finally:
        if use_temp:
            shutil.rmtree(work_path, ignore_errors=True)


def cleanup_mmseqs2_files(directory: Path) -> None:
    """Remove MMseqs2 output files from directory."""
    for pattern in ["result_cluster.tsv", "result_all_seqs.fasta", "result_rep_seq.fasta"]:
        for f in directory.glob(pattern):
            f.unlink(missing_ok=True)
