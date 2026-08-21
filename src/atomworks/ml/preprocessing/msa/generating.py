"""Generate ColabFold-style multiple sequence alignments (MSAs) using MMseqs2.

Provides functions to generate MSAs from protein sequences using the MMseqs2 search pipeline adapted from ColabFold.
MSA files are automatically organized with proper hashing, sharding, and compression.

Examples:
    Generate MSAs with custom configuration:

    .. code-block:: python

       from atomworks.ml.preprocessing.msa.generation import make_msas_mmseqs, MSAGenerationConfig

       config = MSAGenerationConfig(gpu=True, threads=16, max_final_sequences=5000)
       sequences = ["MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...", "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."]
       make_msas_mmseqs(sequences, "output_msas/", config)

    Generate MSAs from a CSV file with defaults:

    .. code-block:: python

       from atomworks.ml.preprocessing.msa.generation import make_msas_from_csv

       make_msas_from_csv("sequences.csv", "sequence_column", "output_msas/")

References:
    * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
    * `ColabFold MMseqs2 Search Script`_ - Original implementation and documentation.

    .. _ColabFold MMseqs2 Search Script: https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
"""

import dataclasses
import logging
import os
import shutil
import subprocess
import tempfile
import time
from os import PathLike
from pathlib import Path
from typing import Literal

import pandas as pd

from atomworks.constants import _load_env_var
from atomworks.enums import MSAFileExtension
from atomworks.ml.executables.mmseqs2 import MMseqs2
from atomworks.ml.preprocessing.msa.filtering import HHFilterConfig, MSAFilterConfig, filter_msas
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

    Args:
        sequences: List of protein sequence strings.
        output_file: Path to the output FASTA file.
    """
    with open(output_file, "w") as f:
        for sequence_string in sequences:
            header = hash_sequence(sequence_string)
            f.write(f">{header}\n{sequence_string}\n")


MODULE_OUTPUT_POS = {
    "align": 4,
    "convertalis": 4,
    "expandaln": 5,
    "filterresult": 4,
    "lndb": 2,
    "mergedbs": 2,
    "mvdb": 2,
    "pairaln": 4,
    "result2msa": 4,
    "search": 3,
}


@dataclasses.dataclass
class MMseqs2SearchConfig:
    """Configuration for MMseqs2 search parameters.

    Default values match those used in the working ColabFold script (reference below).

    Args:
        filter: Whether to filter the MSA.
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
    search_eval: float = 0.1
    expand_eval: float = 1e-3
    expand_max_seq_id: float = 0.95
    align_eval: int = 10
    diff: int = 3000
    qsc: float = -20.0
    filter_qsc: float = 0.0
    filter_max_seq_id: float = 0.95
    filter_min_enable: int = 1000
    filter_qid: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    max_accept: int = 10_000
    prefilter_mode: int = 0
    s: float = 8.0  # Set to None to use k-score instead
    db_load_mode: int = 2


@dataclasses.dataclass
class MSAGenerationConfig:
    """Configuration for MSA generation using MMseqs2.

    This dataclass encapsulates all user-facing configuration options for MSA generation,
    providing a clean interface while maintaining full configurability based on the
    working ColabFold script parameters.

    Args:
        backend: Where the search runs. "local" uses the local MMseqs2 installation and ColabFold
            databases; "server" submits the sequences to a remote ColabFold-style MMseqs2 server.
        sharding_pattern: Directory sharding pattern for file organization.
        output_extension: File extension and compression for output files.
        use_env: Whether to include environmental (metagenomic) database. Applies to both backends;
            with backend="server" it overrides `server.use_env`.
        gpu: Whether to use GPU acceleration. Local backend only.
        gpu_server: Whether to use GPU server (requires gpu=True). Local backend only.
        num_iterations: Number of MMseqs2 search iterations. Local backend only.
        max_seqs: Maximum number of cluster centers (NOT total sequences) in the MSA. Local backend only.
        threads: Number of CPU threads to use. Local backend only.
        use_local_temp_dir: Whether to use local temporary directory for intermediate files. Local backend only.
        max_final_sequences: Maximum number of sequences in the final MSA after HHFilter, or None to skip
            filtering. Note that HHfilter is a local binary, which a "server" backend user may not have.
        check_existing: Whether to check for existing MSAs before generation.
        existing_msa_dirs: Directories to check for existing MSAs. If None, uses LOCAL_MSA_DIRS env var.
        search_config: Advanced MMseqs2 search configuration. Local backend only.
        server: Remote MSA server configuration. Server backend only.

    References:
        * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
    """

    backend: Literal["local", "server"] = "local"
    sharding_pattern: str = "/0:2/"
    output_extension: str = MSAFileExtension.A3M_GZ.value
    use_env: bool = True
    gpu: bool = False
    gpu_server: bool = False
    num_iterations: int = 3
    max_seqs: int = 10000
    threads: int = 32
    use_local_temp_dir: bool = True
    max_final_sequences: int | None = 10000
    check_existing: bool = False
    existing_msa_dirs: list[PathLike] | None = None
    search_config: MMseqs2SearchConfig = dataclasses.field(default_factory=lambda: MMseqs2SearchConfig())
    server: MSAServerConfig = dataclasses.field(default_factory=lambda: MSAServerConfig())

    def __post_init__(self):
        if self.backend not in ("local", "server"):
            raise ValueError(f"Unknown MSA backend: {self.backend!r}. Must be one of 'local', 'server'.")

        # If we're using GPU, also use the GPU server by default
        if self.gpu and not self.gpu_server:
            logger.info("GPU is enabled, setting gpu_server to True")
            self.gpu_server = True

        if self.backend == "server":
            # Keep the two backends' notion of which databases to search in sync
            self.server.use_env = self.use_env


def _get_database_path(gpu: bool = False) -> Path:
    """
    Determine which database path to use, falling back from local to network paths.

    Args:
        gpu: Whether to use GPU databases

    Returns:
        Path to the database directory
    """
    if gpu:
        # For GPU, try local GPU path first, then network path
        if Path(LOCAL_DB_PATH_GPU).exists():
            logger.info(f"Using local GPU database path: {LOCAL_DB_PATH_GPU}")
            return Path(LOCAL_DB_PATH_GPU)
        else:
            logger.info(f"Local GPU database path not found, using network path: {NET_DB_PATH_GPU}")
            return Path(NET_DB_PATH_GPU)
    else:
        # For CPU, try local CPU path first, then network path
        if Path(LOCAL_DB_PATH_CPU).exists():
            logger.info(f"Using local CPU database path: {LOCAL_DB_PATH_CPU}")
            return Path(LOCAL_DB_PATH_CPU)
        else:
            logger.info(f"Local CPU database path not found, using network path: {NET_DB_PATH_CPU}")
            return Path(NET_DB_PATH_CPU)


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
    subprocess.check_call([str(mmseqs2.get_bin_path()), "createdb", str(fasta_file), str(output_db)])
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
    filter_max_seq_id: float = 1.0,
    filter_min_enable: int = 1000,
    profile_input: str = "qdb",
    tmp_dir: str = "tmp",
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
        filter_max_seq_id: filterresult - Maximum sequence identity for filtering.
        filter_min_enable: filterresult - Minimum number of sequences to keep in each MSA block.
        profile_input: Profile input (usually qdb).
        tmp_dir: Temporary directory.

    References:
        * `ColabFold Paper`_ - MSA generation methodology

        .. _ColabFold Paper: https://www.nature.com/articles/s41592-022-01488-1
    """

    if "--gpu-server" in search_param:
        logger.info("Setting up GPU server...")
        gpu_server_process = _start_gpu_server(dbbase.joinpath(db_name), search_param[8], search_param[3], "1")
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

    if "--gpu-server" in search_param:
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
            str(filter_max_seq_id),
            "--filter-min-enable",
            str(filter_min_enable),
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
    expand_eval: float = 1e-3,
    expand_max_seq_id: float = 0.95,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    filter_qsc: float = 0.0,
    filter_max_seq_id: float = 0.95,
    filter_min_enable: int = 1000,
    filter_qid: str = "0.0,0.2,0.4,0.6,0.8,1.0",
    max_accept: int = 10_000,  # this was 1000000 in the original ColabFold script
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
        align_eval = 1e-3
        qsc = 0.8
        max_accept = 10_000

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
    expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        str(expand_eval),
        "--expand-filter-clusters",
        str(int(filter)),
        "--max-seq-id",
        str(expand_max_seq_id),
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
            expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
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
            expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
            profile_input="prof_res",
            tmp_dir="tmp3",
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


def make_msas_mmseqs(
    sequences: str | list[str],
    output_dir: PathLike,
    gpu: bool = False,
    gpu_server: bool = False,
    num_iterations: int = 3,
    max_seqs: int = 10_000,
    use_local_temp_dir: bool = True,
    max_final_sequences: int | None = 10_000,
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
    search_config: MMseqs2SearchConfig | None = None,
) -> None:
    """Generate MSAs directly from protein sequences.

    Args:
        sequences: A single protein sequence string or list of protein sequences.
        output_dir: Path to the output directory where MSA files will be saved.
        gpu: Whether to use GPU acceleration.
        gpu_server: Whether to use GPU server (requires gpu=True).
        num_iterations: Number of search iterations.
        max_seqs: Maximum number of cluster centers.
        use_local_temp_dir: Whether to use local temporary directory for intermediate files.
        max_final_sequences: Maximum number of sequences in final MSAs after filtering, or None to skip filtering.
        sharding_pattern: Directory sharding pattern (e.g., "/0:2/").
        output_extension: Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).
        search_config: Advanced MMseqs2 search configuration.

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

    start_time = time.time()
    _mmseqs_search_monomer(
        dbbase=_get_database_path(gpu=gpu),
        base=Path(intermediate_dir),
        uniref_db=Path(UNIREF30_DB_NAME),
        metagenomic_db=Path(COLABFOLD_DB_NAME),
        gpu=int(gpu),
        gpu_server=int(gpu_server),
        num_iterations=num_iterations,
        max_seqs=max_seqs,
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

    Dispatches to the local MMseqs2 pipeline or to a remote MMseqs2 server, depending on
    ``config.backend``.

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

        Generate MSAs with a remote MMseqs2 server instead of local databases:

        .. code-block:: python

           make_msas_from_csv("sequences.csv", "output_msas/", config=MSAGenerationConfig(backend="server"))
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

    if config.backend == "server":
        # The server backend does its own existence check: unlike the local backend it also looks in
        # `output_dir`, so it works without LOCAL_MSA_DIRS being set
        make_msas_mmseqs_server(
            sequences=sequences,
            output_dir=output_dir,
            config=config.server,
            sharding_pattern=config.sharding_pattern,
            output_extension=config.output_extension,
            max_final_sequences=config.max_final_sequences,
            check_existing=config.check_existing,
            existing_msa_dirs=config.existing_msa_dirs,
        )
        return

    # Filter existing sequences if requested
    if config.check_existing:
        logger.info(f"Finding existing MSAs among {len(sequences)} sequences...")
        missing_sequences, _ = find_msas(
            sequences,
            msa_dirs=config.existing_msa_dirs,
            shard_depths=[0, 1, 2, 3, 4],
            extensions=[MSAFileExtension.A3M, MSAFileExtension.A3M_GZ],
        )
        sequences = missing_sequences
        logger.info(f"Found {len(sequences)} sequences needing MSA generation")

        if not sequences:
            logger.info("All sequences already have MSAs, skipping generation")
            return

    make_msas_mmseqs(
        sequences=sequences,
        output_dir=output_dir,
        gpu=config.gpu,
        gpu_server=config.gpu_server,
        num_iterations=config.num_iterations,
        max_seqs=config.max_seqs,
        use_local_temp_dir=config.use_local_temp_dir,
        max_final_sequences=config.max_final_sequences,
        sharding_pattern=config.sharding_pattern,
        output_extension=config.output_extension,
        search_config=config.search_config,
    )
