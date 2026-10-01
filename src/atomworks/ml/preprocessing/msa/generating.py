"""Generate MSAs from protein sequences using MMseqs2 or HHblits.

Supports both GPU-accelerated MMseqs2 and CPU-based HHblits backends. MSA files are
automatically organized with proper hashing, sharding, and compression.

Example:
    .. code-block:: python

       from atomworks.ml.preprocessing.msa.generating import make_msas_from_csv, MSAGenerationConfig

       config = MSAGenerationConfig(backend="hhblits", threads=8, max_final_sequences=5000)
       make_msas_from_csv("sequences.csv", "output_msas/", sequence_column="seq", config=config)

References:
    * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
    * `ColabFold MMseqs2 Search Script`_ - Original implementation and documentation.

    .. _ColabFold MMseqs2 Search Script: https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
"""

import dataclasses
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from os import PathLike
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from atomworks.constants import _load_env_var
from atomworks.enums import MSAFileExtension
from atomworks.ml.executables.hhblits import HHblits
from atomworks.ml.executables.mmseqs2 import MODULE_OUTPUT_POS, MMseqs2
from atomworks.ml.preprocessing.msa.filtering import (
    HHFilterConfig,
    MSAFilterConfig,
    count_sequences_in_msa,
    filter_msas,
    run_hhfilter,
)
from atomworks.ml.preprocessing.msa.finding import find_msas
from atomworks.ml.preprocessing.msa.organizing import MSAOrganizationConfig, organize_msas
from atomworks.ml.preprocessing.msa.server import MSAServerConfig, make_msas_mmseqs_server
from atomworks.ml.utils.misc import hash_sequence

LOCAL_DB_PATH_GPU = _load_env_var("COLABFOLD_LOCAL_DB_PATH_GPU")
LOCAL_DB_PATH_CPU = _load_env_var("COLABFOLD_LOCAL_DB_PATH_CPU")
NET_DB_PATH_GPU = _load_env_var("COLABFOLD_NET_DB_PATH_GPU")
NET_DB_PATH_CPU = _load_env_var("COLABFOLD_NET_DB_PATH_CPU")

COLABFOLD_DB_NAME = "colabfold_envdb_202108_db"
UNIREF30_DB_NAME = "uniref30_2302_db"

logger = logging.getLogger(__name__)

logger.info(
    "Initialized ColabFold MSA generation module\n"
    f"Local CPU DB Path: {LOCAL_DB_PATH_CPU}\n"
    f"Local GPU DB Path: {LOCAL_DB_PATH_GPU}\n"
    f"Net CPU DB Path: {NET_DB_PATH_CPU}\n"
    f"Net GPU DB Path: {NET_DB_PATH_GPU}"
)


def create_fasta_with_hashed_headers(sequences: list[str], output_file: PathLike) -> None:
    """Create a FASTA file from sequence strings with SHA-256 hashed headers.

    Note: For wrapped sequences with deduplication, see
    :py:func:`~atomworks.ml.preprocessing.utils.fasta.create_fasta_file_from_df`.

    Args:
        sequences: List of protein sequence strings.
        output_file: Path to the output FASTA file.
    """
    with open(output_file, "w") as f:
        for sequence_string in sequences:
            header = hash_sequence(sequence_string)
            f.write(f">{header}\n{sequence_string}\n")


@dataclasses.dataclass
class MMseqs2SearchConfig:
    """Configuration for MMseqs2 search parameters.

    Default values match those used in the working ColabFold script (reference below).

    Args:
        filter: Whether to filter the MSA.
        num_iterations: Number of MMseqs2 search iterations.
        max_seqs: Maximum number of cluster centers in the MSA.
        search_eval: Search e-value threshold.
        expand_eval: E-value threshold for expandaln.
        expand_max_seq_id: Maximum sequence identity for expandaln.
        align_eval: E-value threshold for align.
        diff: Keep at least this many sequences in each MSA block.
        qsc: Reduce diversity using minimum score threshold.
        filter_qsc: Filter diversity using minimum score threshold.
        filter_max_seq_id: Maximum sequence identity for filtering.
        filter_min_enable: Minimum number of sequences to keep in each MSA block.
        filter_qid: Sequence identity thresholds for filtering.
        max_accept: Maximum accepted alignments before stopping.
        prefilter_mode: Prefiltering algorithm (0: k-mer, 1: ungapped, 2: exhaustive).
        s: MMseqs2 sensitivity. Lower = faster but sparser MSAs.
        db_load_mode: Database preload mode (0: auto, 1: fread, 2: mmap, 3: mmap+touch).

    References:
        * `ColabFold MMseqs2 Search Script`_ - Original implementation and documentation.

        .. _ColabFold MMseqs2 Search Script: https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
    """

    filter: bool = True
    num_iterations: int = 3
    max_seqs: int = 10_000
    search_eval: float = 0.1
    expand_eval: float = math.inf
    expand_max_seq_id: float = 0.95
    align_eval: float = 10.0
    diff: int = 3000
    qsc: float = -20.0
    filter_qsc: float = 0.0
    filter_max_seq_id: float = 0.95
    filter_min_enable: int = 1000
    filter_qid: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    max_accept: int = 1_000_000
    prefilter_mode: int = 0
    s: float = 8.0  # Set to None to use k-score instead
    db_load_mode: int = 2


@dataclasses.dataclass
class HHblitsSearchConfig:
    """Configuration for HHblits iterative search parameters.

    Default values match the original HHblits MSA generation pipeline. Database paths
    are loaded from ``HHBLITS_UNIREF30_DB_PATH`` and ``HHBLITS_BFD_DB_PATH`` env vars if not provided.

    Args:
        mact: MAC realignment threshold for HHblits.
        maxfilt: Maximum hits allowed through prefilter.
        neffmax: Maximum Neff (effective sequence count) value.
        use_bfd: Whether to fall back to BFD when UniRef30 yields insufficient sequences.
    """

    uniref30_db_path: str | None = None
    bfd_db_path: str | None = None
    e_values: list[float] = dataclasses.field(default_factory=lambda: [1e-10, 1e-3])
    bfd_e_value: float = 1e-3
    min_seqs_high_cov: int = 2000
    min_seqs_low_cov: int = 4000
    high_cov: float = 75.0
    low_cov: float = 50.0
    identity: float = 90.0
    max_filter_seqs: int = 100_000
    n_iterations: int = 4
    mem: int = 64
    use_bfd: bool = True
    mact: float = 0.35
    maxfilt: int = 10_000_000
    neffmax: float = 20.0
    cov: int = 25
    maxseq: int = 1_000_000
    realign_max: int = 100_000_000

    def __post_init__(self):
        if self.uniref30_db_path is None:
            self.uniref30_db_path = _load_env_var("HHBLITS_UNIREF30_DB_PATH")
        if self.bfd_db_path is None:
            self.bfd_db_path = _load_env_var("HHBLITS_BFD_DB_PATH")

        if self.uniref30_db_path is None:
            raise ValueError(
                "uniref30_db_path is required for HHblits. "
                "Set HHBLITS_UNIREF30_DB_PATH environment variable or provide uniref30_db_path parameter."
            )
        if self.use_bfd and self.bfd_db_path is None:
            raise ValueError(
                "bfd_db_path is required when use_bfd=True. "
                "Set HHBLITS_BFD_DB_PATH environment variable, provide bfd_db_path parameter, or set use_bfd=False."
            )


@dataclasses.dataclass
class MSAGenerationConfig:
    """Configuration for MSA generation.

    This dataclass encapsulates all user-facing configuration options for MSA generation,
    supporting both MMseqs2 and HHblits backends.

    Args:
        sharding_pattern: Directory sharding pattern for file organization.
        output_extension: File extension and compression for output files.
        use_env: Whether to include environmental (metagenomic) database (MMseqs2 only).
        gpu: Whether to use GPU acceleration (MMseqs2 only).
        gpu_server: Whether to use GPU server (requires gpu=True, MMseqs2 only).
        threads: Number of CPU threads for search operations (used by both MMseqs2 and HHblits).
        use_local_temp_dir: Whether to use local temporary directory for intermediate files (MMseqs2 only).
        max_final_sequences: Maximum number of sequences in the final MSA after HHFilter.
        check_existing: Whether to check for existing MSAs before generation.
        existing_msa_dirs: Directories to check for existing MSAs. If None, uses PROTEIN_MSA_DIRS env var.
        search_config: Advanced MMseqs2 search configuration (MMseqs2 only).
        backend: ``"mmseqs2"``, ``"hhblits"``, or remote ``"mmseqs2_server"``.
        server_config: Remote server configuration; only constructed for the remote backend.
        server_max_final_sequences: Optional local HHfilter limit for remote results.
            None avoids requiring a local HHfilter binary.
        hhblits_search_config: HHblits search configuration (HHblits only). If None when using HHblits
            backend, a default config is constructed at generation time.

    References:
        * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
    """

    sharding_pattern: str = "/0:2/"
    output_extension: str = MSAFileExtension.A3M_GZ.value
    use_env: bool = True
    gpu: bool = False
    gpu_server: bool = False
    threads: int = 4
    use_local_temp_dir: bool = True
    max_final_sequences: int = 10000
    check_existing: bool = False
    existing_msa_dirs: list[PathLike] | None = None
    search_config: MMseqs2SearchConfig = dataclasses.field(default_factory=lambda: MMseqs2SearchConfig())
    backend: str = "mmseqs2"
    hhblits_search_config: HHblitsSearchConfig | None = None
    server_config: MSAServerConfig | None = None
    server_max_final_sequences: int | None = None

    def __post_init__(self):
        if self.backend not in ("mmseqs2", "hhblits", "mmseqs2_server"):
            raise ValueError(f"Invalid backend: {self.backend!r}. Must be 'mmseqs2', 'hhblits', or 'mmseqs2_server'.")

        # If we're using GPU, also use the GPU server by default
        if self.gpu and not self.gpu_server:
            logger.info("GPU is enabled, setting gpu_server to True")
            self.gpu_server = True


def _get_database_path(gpu: bool = False) -> Path:
    """Determine which database path to use, falling back from local to network paths.

    Args:
        gpu: Whether to use GPU databases.

    Returns:
        Path to the database directory.

    Raises:
        ValueError: If no database paths are configured.
    """
    if gpu:
        # For GPU, try local GPU path first, then network path
        if LOCAL_DB_PATH_GPU and Path(LOCAL_DB_PATH_GPU).exists():
            logger.info(f"Using local GPU database path: {LOCAL_DB_PATH_GPU}")
            return Path(LOCAL_DB_PATH_GPU)
        elif NET_DB_PATH_GPU:
            logger.info(f"Local GPU database path not found, using network path: {NET_DB_PATH_GPU}")
            return Path(NET_DB_PATH_GPU)
        else:
            raise ValueError(
                "No GPU database paths configured. Please set COLABFOLD_LOCAL_DB_PATH_GPU "
                "or COLABFOLD_NET_DB_PATH_GPU environment variables."
            )
    else:
        # For CPU, try local CPU path first, then network path
        if LOCAL_DB_PATH_CPU and Path(LOCAL_DB_PATH_CPU).exists():
            logger.info(f"Using local CPU database path: {LOCAL_DB_PATH_CPU}")
            return Path(LOCAL_DB_PATH_CPU)
        elif NET_DB_PATH_CPU:
            logger.info(f"Local CPU database path not found, using network path: {NET_DB_PATH_CPU}")
            return Path(NET_DB_PATH_CPU)
        else:
            raise ValueError(
                "No CPU database paths configured. Please set COLABFOLD_LOCAL_DB_PATH_CPU "
                "or COLABFOLD_NET_DB_PATH_CPU environment variables."
            )


def _create_isolated_db_path(original_db_path: Path) -> tuple[Path, Path]:
    """Create an isolated database path with symlinks to avoid GPU server socket conflicts.

    When multiple jobs run concurrently on the same node using gpu_server mode, they share
    the same /dev/shm socket file because MMseqs2 generates the socket ID from the database
    path. By creating symlinks in a unique temp directory, each job gets a unique socket ID.

    Args:
        original_db_path: Path to the original database directory.

    Returns:
        Tuple of (isolated_db_path, temp_dir) where:
        - isolated_db_path: Path to the symlinked database directory
        - temp_dir: Path to the temp directory (caller should clean up)
    """
    # Create a unique temp directory for this job
    unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    temp_dir = Path(tempfile.gettempdir()) / f"mmseqs_isolated_{unique_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks to all files in the original database directory
    for item in original_db_path.iterdir():
        link_path = temp_dir / item.name
        if not link_path.exists():
            link_path.symlink_to(item.resolve())

    logger.info(f"Created isolated database path: {temp_dir} -> {original_db_path}")
    return temp_dir, temp_dir


def _make_mmseqs_db_from_fasta(fasta_file: PathLike, output_dir: PathLike) -> Path:
    """Create a MMseqs2 database from a FASTA file.

    Args:
        fasta_file: Path to the FASTA file.
        output_dir: Path to the output directory.

    Returns:
        Path to the output database.
    """
    mmseqs2 = MMseqs2.get_or_initialize()
    output_db = Path(output_dir) / "qdb"
    subprocess.check_call(
        [
            str(mmseqs2.get_bin_path()),
            "createdb",
            str(fasta_file),
            str(output_db),
            "--shuffle",
            "0",
            "--dbtype",
            "1",
        ]
    )
    return output_db


def _start_gpu_server(
    db_name: Path, max_seqs: int, db_load_mode: int, prefilter_mode: int, wait_time: int = 20
) -> subprocess.Popen:
    """Start the GPU server using the initialized MMseqs2 executable.

    Args:
        db_name: Path to the database name.
        max_seqs: Maximum number of sequences.
        db_load_mode: Database loading mode.
        prefilter_mode: Prefilter mode to use.
        wait_time: Time to wait for server startup in seconds.

    Returns:
        Process object for the GPU server.
    """
    mmseqs2 = MMseqs2.get_or_initialize()
    mmseqs_bin = str(mmseqs2.get_bin_path())

    cmd = [
        mmseqs_bin,
        "gpuserver",
        str(db_name),
        "--max-seqs",
        str(max_seqs),
        "--db-load-mode",
        str(db_load_mode),
        "--prefilter-mode",
        str(prefilter_mode),
    ]
    gpu_server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

    time.sleep(
        wait_time
    )  # TODO They recently updated MMseqs2 to automatically wait for the server to start. Once they release a new version and we update our local installation, this can be removed

    return gpu_server_process


def _run_mmseqs(params: list[str | Path]) -> None:
    """Run an MMseqs2 command using the initialized executable.

    Args:
        params: List of parameters to pass to the MMseqs2 command.
    """
    mmseqs2 = MMseqs2.get_or_initialize()
    mmseqs_bin = str(mmseqs2.get_bin_path())

    module = params[0]
    if module in MODULE_OUTPUT_POS:
        output_pos = MODULE_OUTPUT_POS[module]
        output_path = Path(params[output_pos]).with_suffix(".dbtype")
        if output_path.exists():
            logger.info(f"Skipping {module} because {output_path} already exists")
            return

    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs_bin} {params_log}")
    subprocess.check_call([mmseqs_bin] + [str(p) for p in params])


def _run_mmseqs_search_and_filter(
    base: str,
    dbbase: str,
    db_name: str,
    db_suffix1: str,
    db_suffix2: str,
    output_name: str,
    db_load_mode: int,
    threads: int,
    search_param: list[str],
    expand_param: list[str],
    filter_param: list[str],
    align_eval: float,
    max_accept: int,
    qsc: float,
    align_alt_ali: int = 10,
    qid: bool = False,
    filter_diff: int = 0,
    inner_filter_max_seq_id: float = 1.0,
    inner_filter_min_enable: int = 100,
    profile_input: str = "qdb",
    tmp_dir: str = "tmp",
    start_gpu_server: bool = False,
    gpu_server_max_seqs: int = 10_000,
    gpu_server_db_load_mode: int = 2,
) -> None:
    """Execute core ColabFold MSA generation pipeline.

    Helper function to run MMseqs2 search, expand alignment and filter results. This is the basic pipeline used in ColabFold to generate
    high quality MSAs using MMseqs2.

    First we search against the target database and create alignments of the results. Then we expand these alignments using an alignment
    of the target database. We can then realign the expanded alignments which ultimately results in a higher quality alignments. The
    resulting alignments are then filtered and converted to an MSA format.

    Args:
        base: Directory for the results (and intermediate files).
        dbbase: Path to the database and indices you downloaded and created with setup_databases.sh.
        db_name: Name of the database to search against.
        db_suffix1: Suffix for the database to search against.
        db_suffix2: Suffix for the database to search against.
        output_name: Name of the output file.
        db_load_mode: Database preload mode 0: auto, 1: fread, 2: mmap, 3: mmap+touch.
        threads: Number of threads to use.
        search_param: Extra parameters for the search.
        expand_param: Extra parameters for the expandaln.
        filter_param: Extra parameters for the filterresult.
        align_eval: E-val threshold for align.
        max_accept: Maximum accepted alignments before alignment calculation for a query is stopped.
        qsc: filterresult - reduce diversity of output MSAs using min score thresh.
        align_alt_ali: Number of alternative alignments to keep.
        qid: filterresult - Reduce diversity of output MSAs using min.seq. idendity with query sequences.
        filter_diff: filterresult - Keep at least this many seqs in each MSA block.
        inner_filter_max_seq_id: Inner filterresult - Maximum sequence identity for filtering.
        inner_filter_min_enable: Inner filterresult - Minimum number of sequences to keep in each MSA block.
        profile_input: Profile input (usually qdb).
        tmp_dir: Temporary directory.
        start_gpu_server: Whether to start (and stop) an MMseqs2 GPU server around the search.
        gpu_server_max_seqs: ``--max-seqs`` for the GPU server (must match the search's max_seqs).
        gpu_server_db_load_mode: ``--db-load-mode`` for the GPU server (must match the search's db_load_mode).

    References:
        * `ColabFold Paper`_ - MSA generation methodology

        .. _ColabFold Paper: https://www.nature.com/articles/s41592-022-01488-1
    """

    if start_gpu_server:
        logger.info("Setting up GPU server...")
        gpu_server_process = _start_gpu_server(
            dbbase.joinpath(db_name),
            max_seqs=gpu_server_max_seqs,
            db_load_mode=gpu_server_db_load_mode,
            prefilter_mode=1,  # GPU only supports ungapped prefilter
        )
        logger.info("GPU server setup complete")

    _run_mmseqs(
        [
            "search",
            base.joinpath(profile_input),
            dbbase.joinpath(db_name),
            base.joinpath("res"),
            base.joinpath(tmp_dir),
            "--threads",
            str(threads),
            *search_param,
        ],
    )

    if start_gpu_server:
        logger.info("Stopping GPU server...")
        gpu_server_process.terminate()  # Send SIGTERM
        gpu_server_process.wait()
        logger.info("GPU server stopped")

    if profile_input == "qdb":
        # Move and symlink databases (only needed for first uniref search)
        _run_mmseqs(["mvdb", base.joinpath(f"{tmp_dir}/latest/profile_1"), base.joinpath("prof_res")])
        _run_mmseqs(["lndb", base.joinpath("qdb_h"), base.joinpath("prof_res_h")])
        align_profile = "prof_res"
    else:
        align_profile = f"{tmp_dir}/latest/profile_1"

    # Expand the alignment from search against an alignment of the target database to improve alignment quality
    _run_mmseqs(
        [
            "expandaln",
            base.joinpath(profile_input),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res"),
            dbbase.joinpath(f"{db_name}{db_suffix2}"),
            base.joinpath("res_exp"),
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
            *expand_param,
        ],
    )

    # Realign using the expanded alignment to improve alignment quality
    _run_mmseqs(
        [
            "align",
            base.joinpath(align_profile),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp"),
            base.joinpath("res_exp_realign"),
            "--db-load-mode",
            str(db_load_mode),
            "-e",
            str(align_eval),
            "--max-accept",
            str(max_accept),
            "--threads",
            str(threads),
            "--alt-ali",
            str(align_alt_ali),
            "-a",
        ],
    )

    # Filter the alignment to remove low quality alignments
    _run_mmseqs(
        [
            "filterresult",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp_realign"),
            base.joinpath("res_exp_realign_filter"),
            "--db-load-mode",
            str(db_load_mode),
            "--qid",
            str(int(qid)),
            "--qsc",
            str(qsc),
            "--diff",
            str(filter_diff),
            "--threads",
            str(threads),
            "--max-seq-id",
            str(inner_filter_max_seq_id),
            "--filter-min-enable",
            str(inner_filter_min_enable),
        ],
    )

    # Convert the filtered alignment to a multiple sequence alignment
    _run_mmseqs(
        [
            "result2msa",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp_realign_filter"),
            base.joinpath(output_name),
            "--msa-format-mode",
            "6",
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
            *filter_param,
        ],
    )

    # Cleanup intermediate files
    _run_mmseqs(["rmdb", base.joinpath("res_exp_realign_filter")])
    _run_mmseqs(["rmdb", base.joinpath("res_exp_realign")])
    _run_mmseqs(["rmdb", base.joinpath("res_exp")])
    _run_mmseqs(["rmdb", base.joinpath("res")])


def _mmseqs_search_monomer(
    dbbase: Path,
    base: Path,
    uniref_db: Path,
    metagenomic_db: Path,
    use_env: bool = True,
    filter: bool = True,
    search_eval: float = 0.1,
    expand_eval: float = math.inf,
    expand_max_seq_id: float = 0.95,
    align_eval: float = 10.0,
    diff: int = 3000,
    qsc: float = -20.0,
    filter_qsc: float = 0.0,
    filter_max_seq_id: float = 0.95,
    filter_min_enable: int = 1000,
    filter_qid: str = "0.0,0.2,0.4,0.6,0.8,1.0",
    max_accept: int = 1_000_000,
    num_iterations: int = 3,
    max_seqs: int = 10_000,
    prefilter_mode: int = 0,
    s: float = 8,  # Set to None to use k-score instead
    db_load_mode: int = 2,
    threads: int = 32,
    gpu: int = 0,
    gpu_server: int = 0,
) -> None:
    """Run MMseqs2 search with ColabFold database set.

    Searches each database (UniRef, metagenomic) sequentially, merges alignments, and converts to MSA format.
    Results are always unpacked into individual .a3m files named by input sequence index.

    Runs search (see _run_mmseqs_search_and_filter) on each database (uniref, metagenomic) in turn.
    Alignments from these searches are then merged and converted to an MSA format. The results
    are formatted in individual .a3m files with names corresponding to the input sequence index (0.a3m, 1.a3m, etc.)

    NOTE: Unless specified otherwise, all parameters' default values are from the ColabFold script.

    Args:
        dbbase: Path to the database and indices you downloaded and created with setup_databases.sh.
        base: Directory for the results (and intermediate files).
        uniref_db: UniRef database.
        metagenomic_db: Environmental database (usually ColabFold metagenomics database).
        use_env: Whether to use the environmental database.
        filter: Whether to filter the MSA.
        search_eval: Search e-value threshold.
        expand_eval: E-val threshold for 'expandaln'.
        expand_max_seq_id: Maximum sequence identity for 'expandaln'.
        align_eval: E-val threshold for 'align'.
        diff: filterresult - Keep at least this many seqs in each MSA block.
        qsc: filterresult - reduce diversity of output MSAs using min score thresh.
        filter_qsc: filterresult - reduce diversity of output MSAs using min score thresh.
        filter_max_seq_id: filterresult - Maximum sequence identity for filtering.
        filter_min_enable: filterresult - Minimum number of sequences to keep in each MSA block.
        filter_qid: filterresult - Reduce diversity of output MSAs using min.seq. idendity with query sequences.
        max_accept: align - Maximum accepted alignments before alignment calculation for a query is stopped.
        num_iterations: Number of iterations for the search.
        max_seqs: Maximum number of sequences to search.
        prefilter_mode: Prefiltering algorithm to use: 0: k-mer (high-mem), 1: ungapped (high-cpu), 2: exhaustive (no prefilter, very slow).
        s: MMseqs2 sensitivity. Lowering this will result in a much faster search but possibly sparser MSAs. By default, the k-mer threshold is directly set to the same one of the server, which corresponds to a sensitivity of ~8.
        db_load_mode: Database preload mode 0: auto, 1: fread, 2: mmap, 3: mmap+touch.
        threads: Number of threads to use.
        gpu: Whether to use GPU (1) or not (0).
        gpu_server: Whether to use GPU server (1) or not (0).

    References:
        * `ColabFold MMseqs2 Search Script`_

        .. _ColabFold MMseqs2 Search Script: https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
    """
    if filter:
        # ColabFold filter-mode overrides: these three values are hardcoded by the upstream
        # pipeline when filter=True and intentionally discard user-supplied values.
        # See https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
        align_eval = 10
        qsc = 0.8
        max_accept = 100_000

    # check db types and make sure they exist
    used_dbs = [uniref_db]
    if use_env:
        used_dbs.append(metagenomic_db)
    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {dbbase.joinpath(db)} does not exist")
        if (
            not dbbase.joinpath(f"{db}.idx").is_file() and not dbbase.joinpath(f"{db}.idx.index").is_file()
        ) or os.environ.get("MMSEQS_IGNORE_INDEX", False):
            logger.info("Search does not use index")
            db_load_mode = 0
            db_suffix1 = "_seq"
            db_suffix2 = "_aln"
        else:
            db_suffix1 = ".idx"
            db_suffix2 = ".idx"

    # prep additional params for search, filter, and expand
    search_param = [
        "--num-iterations",
        str(num_iterations),
        "--db-load-mode",
        str(db_load_mode),
        "-a",
        "-e",
        str(search_eval),
        "--max-seqs",
        str(max_seqs),
    ]
    if gpu:
        search_param += [
            "--gpu",
            str(gpu),
            "--prefilter-mode",
            "1",
        ]  # gpu version only supports ungapped prefilter currently
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        if s is not None:  # sensitivy can only be set for non-gpu version, gpu version runs at max sensitivity
            search_param += ["-s", f"{s:.1f}"]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    filter_param = [
        "--filter-msa",
        str(int(filter)),
        "--filter-min-enable",
        str(filter_min_enable),
        "--diff",
        str(diff),
        "--qid",
        str(filter_qid),
        "--qsc",
        str(filter_qsc),
        "--max-seq-id",
        str(filter_max_seq_id),
    ]
    # UniRef expandaln: full params (matches ColabFold)
    uniref_expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        str(expand_eval),
        "--expand-filter-clusters",
        str(int(filter)),
        "--max-seq-id",
        str(expand_max_seq_id),
    ]
    # Metagenomic expandaln: reduced params (matches ColabFold, preserves metagenomic diversity)
    metagenomic_expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        str(expand_eval),
    ]

    # search and filter uniref
    if not base.joinpath("uniref.a3m").with_suffix(".a3m.dbtype").exists():
        _run_mmseqs_search_and_filter(
            base,
            dbbase,
            uniref_db,
            db_suffix1,
            db_suffix2,
            "uniref.a3m",
            db_load_mode,
            threads,
            search_param,
            uniref_expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
            start_gpu_server=bool(gpu_server),
            gpu_server_max_seqs=max_seqs,
            gpu_server_db_load_mode=db_load_mode,
        )
    else:
        logger.info(f"Skipping {uniref_db} search because uniref.a3m already exists")

    # search and filter metagenomic
    if use_env and not base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m").with_suffix(".a3m.dbtype").exists():
        _run_mmseqs_search_and_filter(
            base,
            dbbase,
            metagenomic_db,
            db_suffix1,
            db_suffix2,
            "bfd.mgnify30.metaeuk30.smag30.a3m",
            db_load_mode,
            threads,
            search_param,
            metagenomic_expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
            profile_input="prof_res",
            tmp_dir="tmp3",
            start_gpu_server=bool(gpu_server),
            gpu_server_max_seqs=max_seqs,
            gpu_server_db_load_mode=db_load_mode,
        )
    elif use_env:
        logger.info(f"Skipping {metagenomic_db} search because bfd.mgnify30.metaeuk30.smag30.a3m already exists")

    # merge alignments
    if use_env:
        _run_mmseqs(
            [
                "mergedbs",
                base.joinpath("qdb"),
                base.joinpath("final.a3m"),
                base.joinpath("uniref.a3m"),
                base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"),
            ],
        )
        _run_mmseqs(["rmdb", base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        _run_mmseqs(["rmdb", base.joinpath("uniref.a3m")])
    else:
        _run_mmseqs(["mvdb", base.joinpath("uniref.a3m"), base.joinpath("final.a3m")])
        _run_mmseqs(["rmdb", base.joinpath("uniref.a3m")])

    # unpack alignments into individual .a3m files
    _run_mmseqs(
        [
            "unpackdb",
            base.joinpath("final.a3m"),
            base.joinpath("."),
            "--unpack-name-mode",
            "0",
            "--unpack-suffix",
            ".a3m",
        ],
    )
    _run_mmseqs(["rmdb", base.joinpath("final.a3m")])

    # cleanup
    _run_mmseqs(["rmdb", base.joinpath("prof_res")])
    _run_mmseqs(["rmdb", base.joinpath("prof_res_h")])
    shutil.rmtree(base.joinpath("tmp"))
    if use_env:
        shutil.rmtree(base.joinpath("tmp3"))


def _run_hhblits_search(
    input_file: PathLike,
    output_file: PathLike,
    db_path: str,
    config: HHblitsSearchConfig,
    e_value: float,
    cpu: int = 4,
) -> None:
    """Run a single HHblits search against a database.

    Args:
        input_file: Path to the input FASTA or A3M file.
        output_file: Path to the output A3M file.
        db_path: Path to the HH-suite formatted database.
        config: HHblits search configuration.
        e_value: E-value threshold for inclusion.
        cpu: Number of CPU threads for HHblits (``-cpu``).
    """
    hhblits = HHblits.get_or_initialize()
    hhblits.run_command(
        "-i",
        str(input_file),
        "-oa3m",
        str(output_file),
        "-o",
        "/dev/null",
        "-mact",
        str(config.mact),
        "-maxfilt",
        str(config.maxfilt),
        "-neffmax",
        str(config.neffmax),
        "-cov",
        str(config.cov),
        "-cpu",
        str(cpu),
        "-nodiff",
        "-realign_max",
        str(config.realign_max),
        "-maxseq",
        str(config.maxseq),
        "-maxmem",
        str(config.mem),
        "-n",
        str(config.n_iterations),
        "-d",
        str(db_path),
        "-e",
        str(e_value),
        "-v",
        "0",
    )


def _hhblits_iterative_search_single(
    sequence: str,
    seq_hash: str,
    output_dir: Path,
    config: HHblitsSearchConfig,
    cpu: int = 4,
) -> Path | None:
    """Run iterative HHblits search for a single sequence.

    Searches UniRef30 with increasing e-value thresholds, filtering at each step.
    Falls back to BFD if insufficient sequences are found.

    Args:
        sequence: Protein sequence string.
        seq_hash: SHA-256 hash of the sequence.
        output_dir: Directory for the final output A3M file.
        config: HHblits search configuration.
        cpu: Number of CPU threads for HHblits (``-cpu``).

    Returns:
        Path to the final A3M file, or None if no results were produced.
    """
    out_a3m = output_dir / f"{seq_hash}.a3m"

    if out_a3m.exists():
        logger.debug(f"Skipping {seq_hash}: output already exists at {out_a3m}")
        return out_a3m

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write single-sequence FASTA
        fasta_file = tmp_path / f"{seq_hash}.fasta"
        fasta_file.write_text(f">{seq_hash}\n{sequence}\n")

        prev_a3m: Path | str = fasta_file
        result_a3m: Path | None = None

        # Iterative UniRef30 searches with increasing e-values
        for e_value in config.e_values:
            uniref_out = tmp_path / f"uniref30.{e_value}.a3m"
            _run_hhblits_search(prev_a3m, uniref_out, config.uniref30_db_path, config, e_value, cpu=cpu)

            # Filter at high coverage
            high_cov_file = tmp_path / f"uniref30.{e_value}.id{config.identity:.0f}cov{config.high_cov:.0f}.a3m"
            run_hhfilter(
                uniref_out, high_cov_file, maxseq=config.max_filter_seqs, id=config.identity, cov=config.high_cov
            )

            # Filter at low coverage
            low_cov_file = tmp_path / f"uniref30.{e_value}.id{config.identity:.0f}cov{config.low_cov:.0f}.a3m"
            run_hhfilter(
                uniref_out, low_cov_file, maxseq=config.max_filter_seqs, id=config.identity, cov=config.low_cov
            )

            n_high = count_sequences_in_msa(high_cov_file)
            n_low = count_sequences_in_msa(low_cov_file)
            logger.debug(
                f"[{seq_hash}] UniRef30 e={e_value}: {n_high} seqs at {config.high_cov}% cov, "
                f"{n_low} seqs at {config.low_cov}% cov"
            )

            # Use low-coverage file as input for next iteration
            prev_a3m = low_cov_file

            if n_high > config.min_seqs_high_cov:
                result_a3m = high_cov_file
                break
            elif n_low > config.min_seqs_low_cov:
                result_a3m = low_cov_file
                break

        # Fall back to BFD if needed
        if result_a3m is None and config.use_bfd and config.bfd_db_path:
            e_value = config.bfd_e_value
            bfd_out = tmp_path / f"bfd.{e_value}.a3m"
            _run_hhblits_search(prev_a3m, bfd_out, config.bfd_db_path, config, e_value, cpu=cpu)

            bfd_high_cov = tmp_path / f"bfd.{e_value}.id{config.identity:.0f}cov{config.high_cov:.0f}.a3m"
            run_hhfilter(bfd_out, bfd_high_cov, maxseq=config.max_filter_seqs, id=config.identity, cov=config.high_cov)

            bfd_low_cov = tmp_path / f"bfd.{e_value}.id{config.identity:.0f}cov{config.low_cov:.0f}.a3m"
            run_hhfilter(bfd_out, bfd_low_cov, maxseq=config.max_filter_seqs, id=config.identity, cov=config.low_cov)

            n_high = count_sequences_in_msa(bfd_high_cov)
            n_low = count_sequences_in_msa(bfd_low_cov)
            logger.debug(
                f"[{seq_hash}] BFD e={e_value}: {n_high} seqs at {config.high_cov}% cov, "
                f"{n_low} seqs at {config.low_cov}% cov"
            )

            prev_a3m = bfd_low_cov

            if n_high > config.min_seqs_high_cov:
                result_a3m = bfd_high_cov
            elif n_low > config.min_seqs_low_cov:
                result_a3m = bfd_low_cov

        # If still no result, use the last processed file (if it's not just the input FASTA)
        if result_a3m is None and Path(prev_a3m) != fasta_file and Path(prev_a3m).exists():
            logger.info(f"[{seq_hash}] Insufficient sequences after all searches; using last processed file")
            result_a3m = Path(prev_a3m)

        # Copy result to output directory
        if result_a3m is not None and result_a3m.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_a3m, out_a3m)
            return out_a3m

    return None


def make_msas_hhblits(
    sequences: str | list[str],
    output_dir: PathLike,
    max_final_sequences: int = 10_000,
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
    search_config: HHblitsSearchConfig | None = None,
    cpu: int = 4,
) -> None:
    """Generate MSAs from protein sequences using HHblits (CPU-only, HH-suite).

    Runs iterative HHblits searches per sequence against UniRef30 (and optionally BFD),
    then organizes and filters the results following the same post-processing as MMseqs2.

    Args:
        sequences: A single protein sequence string or list of protein sequences.
        output_dir: Path to the output directory where MSA files will be saved.
        max_final_sequences: Maximum number of sequences in final MSAs after filtering.
        sharding_pattern: Directory sharding pattern (e.g., "/0:2/").
        output_extension: Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).
        search_config: HHblits search configuration. If None, uses defaults.
        cpu: Number of CPU threads for HHblits (``-cpu``).
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    if search_config is None:
        search_config = HHblitsSearchConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize HHblits executable
    HHblits.get_or_initialize()

    start_time = time.time()
    for sequence in tqdm(sequences, desc="HHblits MSA generation", unit="seq"):
        seq_hash = hash_sequence(sequence)
        _hhblits_iterative_search_single(sequence, seq_hash, output_path, search_config, cpu=cpu)

    logger.info(f"Completed {len(sequences)} sequences in {time.time() - start_time:.1f}s with HHblits")

    # Organize MSAs using existing organization functionality
    org_config = MSAOrganizationConfig(
        input_extension=MSAFileExtension.A3M,
        output_extension=output_extension,
        sharding_pattern=sharding_pattern,
        copy_files=False,
    )

    logger.info("Organizing MSA files...")
    organize_msas(output_dir, output_dir, org_config)

    # Filter MSA files to reduce sequence count and redundancy
    if max_final_sequences is not None:
        filter_config = MSAFilterConfig(
            input_extension=output_extension,
            output_extension=output_extension,
            hhfilter=HHFilterConfig(max_sequences=max_final_sequences),
        )
        logger.info(f"Filtering MSA files to max {max_final_sequences} sequences...")
        filter_msas(output_dir, output_dir, filter_config)


def make_msas_mmseqs(
    sequences: str | list[str],
    output_dir: PathLike,
    gpu: bool = False,
    gpu_server: bool = False,
    use_local_temp_dir: bool = True,
    max_final_sequences: int = 10_000,
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
    search_config: MMseqs2SearchConfig | None = None,
    threads: int = 4,
) -> None:
    """Generate MSAs directly from protein sequences.

    Args:
        sequences: A single protein sequence string or list of protein sequences.
        output_dir: Path to the output directory where MSA files will be saved.
        gpu: Whether to use GPU acceleration.
        gpu_server: Whether to use GPU server (requires gpu=True).
        use_local_temp_dir: Whether to use local temporary directory for intermediate files.
        max_final_sequences: Maximum number of sequences in final MSAs after filtering.
        sharding_pattern: Directory sharding pattern (e.g., "/0:2/").
        output_extension: Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).
        search_config: Advanced MMseqs2 search configuration (includes ``num_iterations`` and ``max_seqs``).
        threads: Number of CPU threads for MMseqs2 search.

    Examples:
        .. code-block:: python

           make_msas_mmseqs(
               ["MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...", "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."], "output_msas/"
           )
    """
    # Ensure sequences is a list for unified processing
    if isinstance(sequences, str):
        sequences = [sequences]

    # Handle search config creation
    if search_config is None:
        search_config = MMseqs2SearchConfig()

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use tempfile to make temporary directory that should automatically be on local drive
    if use_local_temp_dir:
        intermediate_dir = tempfile.mkdtemp()
    else:
        intermediate_dir = output_dir

    # Initialize MMseqs2 executable and create FASTA file from sequences
    MMseqs2.get_or_initialize()
    fasta_file = Path(intermediate_dir) / "input_sequences.fasta"
    create_fasta_with_hashed_headers(sequences, fasta_file)
    _make_mmseqs_db_from_fasta(fasta_file, intermediate_dir)

    if gpu_server and not gpu:
        raise ValueError("gpu_server is True but gpu is False")

    # Get the database path, creating isolated symlinks if using GPU server
    # to avoid socket conflicts between concurrent jobs on the same node
    original_db_path = _get_database_path(gpu=gpu)
    isolated_db_dir = None
    if gpu_server:
        dbbase, isolated_db_dir = _create_isolated_db_path(original_db_path)
    else:
        dbbase = original_db_path

    start_time = time.time()
    try:
        _mmseqs_search_monomer(
            dbbase=dbbase,
            base=Path(intermediate_dir),
            uniref_db=Path(UNIREF30_DB_NAME),
            metagenomic_db=Path(COLABFOLD_DB_NAME),
            gpu=int(gpu),
            gpu_server=int(gpu_server),
            num_iterations=search_config.num_iterations,
            max_seqs=search_config.max_seqs,
            s=search_config.s,
            filter=search_config.filter,
            search_eval=search_config.search_eval,
            expand_eval=search_config.expand_eval,
            expand_max_seq_id=search_config.expand_max_seq_id,
            align_eval=search_config.align_eval,
            diff=search_config.diff,
            qsc=search_config.qsc,
            filter_qsc=search_config.filter_qsc,
            filter_max_seq_id=search_config.filter_max_seq_id,
            filter_min_enable=search_config.filter_min_enable,
            filter_qid=search_config.filter_qid,
            max_accept=search_config.max_accept,
            prefilter_mode=search_config.prefilter_mode,
            db_load_mode=search_config.db_load_mode,
        )
    finally:
        # Clean up isolated database symlinks
        if isolated_db_dir is not None and isolated_db_dir.exists():
            shutil.rmtree(isolated_db_dir)
            logger.info(f"Cleaned up isolated database path: {isolated_db_dir}")
    logger.info(
        f"Completed {len(sequences)} sequences in {time.time() - start_time} seconds with MMSeqs2 search and alignment"
    )

    # cleanup by removing any file or directory that isn't .a3m or .m8
    for file in Path(intermediate_dir).iterdir():
        if not file.name.endswith((".a3m", ".m8")):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

    if use_local_temp_dir:
        # copy over everything from intermediate_dir to output_dir
        for file in Path(intermediate_dir).iterdir():
            shutil.copy(file, Path(output_dir) / file.name)
        # remove the intermediate_dir
        shutil.rmtree(intermediate_dir)

    logger.info(f"MSA files saved to: {Path(output_dir).absolute()}")

    # Organize MSAs using existing organization functionality
    org_config = MSAOrganizationConfig(
        input_extension=MSAFileExtension.A3M,
        output_extension=output_extension,
        sharding_pattern=sharding_pattern,
        copy_files=False,  # Move files instead of copying
    )

    logger.info("Organizing MSA files...")
    organize_msas(output_dir, output_dir, org_config)

    # Filter MSA files to reduce sequence count and redundancy
    filter_config = MSAFilterConfig(
        input_extension=output_extension,
        output_extension=output_extension,
        hhfilter=HHFilterConfig(
            max_sequences=max_final_sequences,
        ),
    )

    if max_final_sequences is not None:
        logger.info(f"Filtering MSA files to max {max_final_sequences} sequences...")
        filter_msas(output_dir, output_dir, filter_config)


def make_msas_from_csv(
    csv_file: PathLike,
    output_dir: PathLike,
    sequence_column: str | None = None,
    config: MSAGenerationConfig | None = None,
) -> None:
    """Generate MSAs from sequences in a CSV file.

    Args:
        csv_file: Path to CSV file containing protein sequences.
        output_dir: Directory where organized MSA files will be saved.
        sequence_column: Name of column containing sequences. If None, CSV must have exactly one column.
        config: MSA generation configuration. If None, uses default config.

    Examples:
        Generate MSAs from single-column CSV:

        .. code-block:: python

           make_msas_from_csv("sequences.csv", "output_msas/")

        Generate MSAs from multi-column CSV:

        .. code-block:: python

           make_msas_from_csv("data.csv", "output_msas/", sequence_column="sequence")
    """

    df = pd.read_csv(csv_file)

    if sequence_column is None:
        if len(df.columns) != 1:
            raise ValueError(
                f"CSV has {len(df.columns)} columns. Either provide exactly 1 column or specify sequence_column parameter"
            )
        sequence_column = df.columns[0]

    sequences = df[sequence_column].dropna().unique().tolist()

    logger.info(f"Loaded {len(sequences)} unique sequences from {csv_file}")

    # Handle config creation
    if config is None:
        config = MSAGenerationConfig()

    # Filter existing sequences if requested
    if config.check_existing and config.backend != "mmseqs2_server":
        logger.info(f"Finding existing MSAs among {len(sequences)} sequences...")
        missing_sequences, _ = find_msas(
            sequences,
            msa_dirs=config.existing_msa_dirs,
        )
        sequences = missing_sequences
        logger.info(f"Found {len(sequences)} sequences needing MSA generation")

        if not sequences:
            logger.info("All sequences already have MSAs, skipping generation")
            return

    if config.backend == "mmseqs2_server":
        make_msas_mmseqs_server(
            sequences=sequences,
            output_dir=output_dir,
            config=config.server_config or MSAServerConfig(use_env=config.use_env),
            max_final_sequences=config.server_max_final_sequences,
            sharding_pattern=config.sharding_pattern,
            output_extension=config.output_extension,
            check_existing=config.check_existing,
            existing_msa_dirs=config.existing_msa_dirs,
        )
    elif config.backend == "hhblits":
        hhblits_config = (
            config.hhblits_search_config if config.hhblits_search_config is not None else HHblitsSearchConfig()
        )
        make_msas_hhblits(
            sequences=sequences,
            output_dir=output_dir,
            max_final_sequences=config.max_final_sequences,
            sharding_pattern=config.sharding_pattern,
            output_extension=config.output_extension,
            search_config=hhblits_config,
            cpu=config.threads,
        )
    else:
        make_msas_mmseqs(
            sequences=sequences,
            output_dir=output_dir,
            gpu=config.gpu,
            gpu_server=config.gpu_server,
            use_local_temp_dir=config.use_local_temp_dir,
            max_final_sequences=config.max_final_sequences,
            sharding_pattern=config.sharding_pattern,
            output_extension=config.output_extension,
            search_config=config.search_config,
            threads=config.threads,
        )
