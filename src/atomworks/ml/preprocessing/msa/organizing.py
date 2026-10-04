"""MSA file organization for AtomWorks.

Prepares arbitrary directory of MSA files into an AtomWorks-compatible structure.
"""

import concurrent.futures
import dataclasses
import functools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from atomworks.enums import MSAFileExtension
from atomworks.io.utils.compression import transfer_with_compression
from atomworks.io.utils.io_utils import apply_sharding_pattern, find_files_by_extension
from atomworks.ml.preprocessing.msa.finding import sequence_has_msa
from atomworks.ml.utils.io import open_file
from atomworks.ml.utils.misc import get_complex_id, hash_sequence

logger = logging.getLogger(__name__)


def validate_msa_file_extension(file_extension: str) -> None:
    """Validate that a file extension is supported for MSA processing.

    Args:
        file_extension: File extension to validate.

    Raises:
        ValueError: If file extension is not supported.
    """
    from atomworks.enums import SUPPORTED_MSA_FILE_EXTENSIONS

    if file_extension not in SUPPORTED_MSA_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file_extension: {file_extension}")


def extract_first_sequence_from_msa(file_path: Path) -> str:
    """Extract the first sequence from an MSA file.

    Args:
        file_path: Path to the MSA file. May be compressed (.gz).
            A3M and FASTA formats supported.

    Returns:
        First sequence string (without header).

    Raises:
        ValueError: If MSA file has no sequences or invalid format.
        FileNotFoundError: If file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"MSA file not found: {file_path}")

    with open_file(file_path) as f:
        found_header = False
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                found_header = True
                continue
            if found_header and line:
                return line

    raise ValueError(f"No sequence found in MSA file: {file_path}")


@dataclasses.dataclass
class MSAOrganizationConfig:
    """Configuration for MSA file organization and processing.

    Used by organize_msas to control how existing MSA files are organized in the output directory.
    Controls input and output file extensions with automatic compression/decompression.
    """

    input_extension: MSAFileExtension | str = MSAFileExtension.A3M_GZ
    output_extension: MSAFileExtension | str = MSAFileExtension.A3M_GZ
    sharding_pattern: str | None = "/0:2/"
    copy_files: bool = True
    num_workers: int | None = None
    execution_model: str = "process"  # Options: "process", "thread", or "auto"
    check_existing: bool = False
    existing_msa_dirs: list[PathLike] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate and convert input extension
        validate_msa_file_extension(self.input_extension)
        self.input_extension = MSAFileExtension(self.input_extension)

        # Validate and convert output extension
        validate_msa_file_extension(self.output_extension)
        self.output_extension = MSAFileExtension(self.output_extension)


def organize_msas(input_dir: PathLike, output_dir: PathLike, config: MSAOrganizationConfig | None = None) -> None:
    """Organize an arbitrary directory of MSA files into AtomWorks-compatible structure.

    After organization, files will be named by the SHA-256 hash of the first sequence, and can be loaded via the AtomWorks MSA loading and pairing Transforms.

    For each MSA file found in the source directory, this function:
    1. Computes the SHA-256 hash of the sequence, which will become the new filename (so we don't store duplicates)
    2. Shards files into subdirectories based on hash and directory structure
    3. Automatically handles compression/decompression based on input/output extensions

    Args:
        input_dir: Source directory containing MSA files, recursively searched.
        output_dir: Destination directory for organized files. Must contain sufficient space.
        config: MSA processing configuration. If None, uses defaults. See :py:class:`~MSAOrganizationConfig` for details.

    Raises:
        ValueError: If configuration is invalid.
        FileNotFoundError: If no MSA files are found or directories don't exist.

    Examples:
        Organize MSA files with default configuration:

        >>> from atomworks.ml.preprocessing.msa.organization import organize_msas
        >>> organize_msas("./raw_msas", "./organized_msas")

        Organize with custom configuration:

        >>> from atomworks.ml.preprocessing.msa.organization import organize_msas, MSAOrganizationConfig
        >>> from atomworks.enums import MSAFileExtension
        >>> config = MSAOrganizationConfig(
        ...     input_extension=MSAFileExtension.A3M, output_extension=MSAFileExtension.A3M_GZ, sharding_pattern="/0:2/"
        ... )
        >>> organize_msas("./raw_msas", "./organized_msas", config=config)
    """
    if config is None:
        # Use default configuration
        config = MSAOrganizationConfig()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Source directory not found: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Recursively searching for files with extension {config.input_extension} in {input_dir}")
    msa_files = find_files_by_extension(input_path, str(config.input_extension))

    logger.info(f"Found {len(msa_files)} files with extension {config.input_extension}")

    logger.info(f"Organizing files to {output_dir}...")

    # Configuration is complete, proceed with organization

    num_workers = config.num_workers or min(os.cpu_count() or 4, 16)
    execution_model = config.execution_model.lower()

    # Validate execution model
    valid_models = ["process", "thread", "auto"]
    if execution_model not in valid_models:
        logger.warning(f"Invalid execution_model '{execution_model}'. Using 'auto' instead.")
        execution_model = "auto"

    # Auto-select best execution model
    if execution_model == "auto":
        # Compression/decompression is CPU-bound, use process
        # If input and output extensions match (no compression/decompression needed), use thread
        is_compression_needed = str(config.input_extension) != str(config.output_extension)
        execution_model = "process" if is_compression_needed else "thread"

    logger.info(f"Processing with {num_workers} workers using {execution_model} parallelism")

    # Process files with progress bar
    organize_func = functools.partial(_organize_single_msa, output_path=output_path, config=config)

    # Choose appropriate executor based on task characteristics
    executor_class = ProcessPoolExecutor if execution_model == "process" else ThreadPoolExecutor

    # Use chunksize for process pool to reduce overhead
    chunksize = max(1, min(100, len(msa_files) // max(1, num_workers))) if execution_model == "process" else 1

    with executor_class(max_workers=num_workers) as executor:
        if execution_model == "process":
            # Process pool works better with imap for large datasets
            results = executor.map(organize_func, msa_files, chunksize=chunksize)
            # Wrap with tqdm for progress tracking
            list(tqdm(results, total=len(msa_files), desc="Organizing MSA files", unit="file"))
        else:
            # Thread pool works better with as_completed for I/O bound tasks
            futures = [executor.submit(organize_func, file) for file in msa_files]
            # Monitor progress with tqdm
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Organizing MSA files",
                unit="file",
            ):
                pass  # Progress is tracked by tqdm

    logger.info(f"MSA file organization complete! Saved processed files to: {output_dir}")


def organize_paired_msas(input_dir: PathLike, output_dir: PathLike, config: MSAOrganizationConfig | None = None) -> str:
    """Organize raw per-chain a3m files from ONE paired/complex query into AtomWorks'
    standard paired-MSA layout.

    Unlike `organize_msas` (which treats every file in `input_dir` as an independent,
    unrelated alignment -- one per unique sequence), this treats *all* files found in
    `input_dir` as belonging to the SAME complex: their first sequences, collectively,
    determine the complex_id (see
    :py:func:`~atomworks.ml.utils.misc.get_complex_id`) -- computed from file content,
    the same way `organize_msas` derives each file's identity from content, not
    passed in or tracked separately by the caller.

    Output layout: ``<output_dir>/<shard-of-complex_id>/<complex_id>/<chain_hash><extension>``
    -- flat within the complex's own directory (a complex has few chains), but
    complex_id itself is sharded the same way a sequence hash would be (a large
    fine-tuning run can have many complexes). Discoverable via
    :py:func:`~atomworks.ml.preprocessing.msa.finding.find_paired_msas` /
    :py:func:`~atomworks.ml.preprocessing.msa.finding.get_paired_msa_path`.

    Args:
        input_dir: Directory containing one raw a3m file per chain in the complex
            (recursively searched, same as `organize_msas`).
        output_dir: Destination directory.
        config: MSA processing configuration. `sharding_pattern` shards complex_id
            (not the files within it, which are always flat). `check_existing` /
            `existing_msa_dirs` / `num_workers` / `execution_model` are not used --
            a complex has too few files to need parallelism, and existing-MSA
            deduplication belongs at the sequence-submission level, before this is
            ever called.

    Returns:
        The complex_id this batch of files was organized under.

    Raises:
        FileNotFoundError: If no MSA files are found or `input_dir` doesn't exist.
        FileExistsError: If the complex's directory already exists (complete, or partial
            from an interrupted run -- delete it to regenerate).
    """
    if config is None:
        config = MSAOrganizationConfig()

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Source directory not found: {input_dir}")

    msa_files = find_files_by_extension(input_path, str(config.input_extension))
    if not msa_files:
        raise FileNotFoundError(f"No files with extension {config.input_extension} found in {input_dir}")

    chain_hash_by_file: dict[Path, str] = {}
    sequences: list[str] = []
    for file_path in msa_files:
        sequence = extract_first_sequence_from_msa(file_path)
        sequences.append(sequence)
        chain_hash_by_file[file_path] = hash_sequence(sequence)

    complex_id = get_complex_id(sequences)
    if config.sharding_pattern:
        complex_dir = Path(output_dir) / apply_sharding_pattern(complex_id, config.sharding_pattern)
    else:
        complex_dir = Path(output_dir) / complex_id
    # Paired files are row-aligned across chains, so never add to an existing complex
    # directory: a partial one left by an interrupted run would otherwise end up mixing
    # chains from two separate searches.
    complex_dir.mkdir(parents=True, exist_ok=False)

    for file_path, chain_hash in chain_hash_by_file.items():
        dest = complex_dir / f"{chain_hash}{config.output_extension}"
        transfer_with_compression(input_file=file_path, output_file=dest, move=not config.copy_files)

    logger.info(f"Paired MSA organization complete! complex_id={complex_id}, saved to: {complex_dir}")
    return complex_id


def organize_template_alignments(
    sequence_to_content: dict[str, str],
    output_dir: PathLike,
    sharding_pattern: str | None = "/0:2/",
    output_extension: str = ".m8",
) -> None:
    """Organize raw ColabFold template-alignment hit tables (``pdb70.m8`` rows) into
    AtomWorks' standard hash-sharded layout.

    Uses the same sequence-hash/sharding convention as `organize_msas`, so a
    sequence's template alignment and its unpaired MSA land at the corresponding
    shard. Unlike MSA files, template hit tables don't self-identify their query
    sequence (no FASTA-style header), so the sequence must be supplied explicitly
    rather than extracted from file content.

    Discoverable via :py:func:`~atomworks.ml.preprocessing.msa.finding.find_template_alignments`.

    Args:
        sequence_to_content: Maps each query sequence to its raw ``pdb70.m8`` hit
            lines (tab-separated text, one hit per line). Sequences with falsy
            content (no hits) are skipped -- no empty file is written.
        output_dir: Destination directory for organized files.
        sharding_pattern: Directory sharding pattern (e.g. "/0:2/"), matching
            `organize_msas`'s convention.
        output_extension: Output file extension.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for sequence, content in sequence_to_content.items():
        if not content:
            continue

        hashed_filename = Path(hash_sequence(sequence))
        if sharding_pattern:
            dest = output_path / apply_sharding_pattern(hashed_filename, sharding_pattern).with_suffix(output_extension)
        else:
            dest = output_path / hashed_filename.with_suffix(output_extension)

        if dest.exists():
            logger.debug(f"Output file already exists, skipping: {dest}")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    logger.info(f"Template alignment organization complete! Saved to: {output_dir}")


def _build_output_path(dest_dir: Path, hashed_filename: Path, config: MSAOrganizationConfig) -> Path:
    """Build the output file path using sharding pattern.

    Args:
        dest_dir: Destination directory base path.
        hashed_filename: SHA-256 hash of the sequence.
        config: Organization configuration.

    Returns:
        Path to the output file.
    """
    # Use the output_extension from config
    output_extension = config.output_extension

    # Apply sharding pattern to organize files
    if config.sharding_pattern:
        sharded_path = apply_sharding_pattern(hashed_filename, config.sharding_pattern)
        output_path = dest_dir / sharded_path.with_suffix(output_extension)
    else:
        output_path = dest_dir / hashed_filename.with_suffix(output_extension)

    return output_path


def _organize_single_msa(file_path: Path, output_path: Path, config: MSAOrganizationConfig) -> None:
    """Organize a single MSA file with automatic compression/decompression based on extensions.

    Args:
        file_path: Path to the MSA file to process.
        output_path: Destination directory for processed files.
        config: MSA organization configuration.
    """
    try:
        # Hash the first sequence to get the filename
        sequence = extract_first_sequence_from_msa(file_path)

        # Check if MSA already exists if check_existing is enabled
        if config.check_existing and sequence_has_msa(sequence, config.existing_msa_dirs):
            logger.debug(f"MSA already exists for sequence, skipping: {file_path}")
            return

        hashed_filename = Path(hash_sequence(sequence))

        # Build the final output path
        final_output_path = _build_output_path(output_path, hashed_filename, config)

        # Skip if output file already exists
        if final_output_path.exists():
            logger.debug(f"Output file already exists, skipping: {final_output_path}")
            return

        # Copy or move the file to the final location with automatic compression/decompression
        transfer_with_compression(
            input_file=file_path,
            output_file=final_output_path,
            move=not config.copy_files,
        )

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        raise
