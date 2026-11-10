"""MSA filtering for AtomWorks.

Handles filtering MSA files to reduce sequence count and redundancy using HHfilter.
"""

import dataclasses
import functools
import logging
import tempfile
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path

from atomworks.enums import MSAFileExtension
from atomworks.io.utils.compression import (
    compress_file,
    is_compressed_file,
    maybe_decompress_file,
    transfer_with_compression,
)
from atomworks.io.utils.io_utils import find_files_by_extension
from atomworks.ml.executables.hhfilter import HHFilter
from atomworks.ml.preprocessing.msa.organizing import validate_msa_file_extension

logger = logging.getLogger(__name__)


def count_sequences_in_msa(file_path: Path) -> int:
    """Count sequences in MSA file by counting lines starting with '>'.

    Uses bash commands for faster counting of large files.

    Args:
        file_path: Path to the MSA file.

    Returns:
        Number of sequences in the MSA.

    Raises:
        FileNotFoundError: If file does not exist.
        subprocess.CalledProcessError: If bash command fails.
    """
    import subprocess

    if not file_path.exists():
        raise FileNotFoundError(f"MSA file not found: {file_path}")

    if is_compressed_file(file_path):
        # For compressed files: use appropriate decompression tool
        if str(file_path).endswith(".zst"):
            cmd = f'zstdcat "{file_path}" | grep -c "^>"'
        else:
            # .gz or .gzip files
            cmd = f'zcat "{file_path}" | grep -c "^>"'
    else:
        # For regular files: grep -c "^>" file
        cmd = f'grep -c "^>" "{file_path}"'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


@dataclasses.dataclass
class HHFilterConfig:
    """Configuration for HHfilter MSA filtering."""

    max_sequences: int = 10_000
    max_identity_percent: float = 90.0
    min_coverage_percent: float = 50.0


@dataclasses.dataclass
class MSAFilterConfig:
    """Configuration for MSA file filtering."""

    input_extension: MSAFileExtension | str = MSAFileExtension.A3M_GZ
    output_extension: MSAFileExtension | str = MSAFileExtension.A3M_GZ
    hhfilter: HHFilterConfig = dataclasses.field(default_factory=HHFilterConfig)
    num_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if isinstance(self.input_extension, str):
            validate_msa_file_extension(self.input_extension)
            self.input_extension = MSAFileExtension(self.input_extension)

        if isinstance(self.output_extension, str):
            validate_msa_file_extension(self.output_extension)
            self.output_extension = MSAFileExtension(self.output_extension)


def filter_msas(input_dir: PathLike, output_dir: PathLike | None = None, config: MSAFilterConfig | None = None) -> None:
    """Filter MSA files using HHfilter to reduce sequence count and redundancy.

    Args:
        input_dir: Source directory containing MSA files, recursively searched.
        output_dir: Destination directory for filtered files. If None, performs in-place filtering.
        config: MSA filtering configuration. If None, uses defaults.

    Raises:
        FileNotFoundError: If no MSA files are found or directories don't exist.

    Examples:
        Filter MSA files in-place (modifying original files):

        >>> from atomworks.ml.preprocessing.msa.filtering import filter_msas, MSAFilterConfig, HHFilterConfig
        >>> filter_config = MSAFilterConfig(hhfilter=HHFilterConfig(max_sequences=5000, max_identity_percent=90))
        >>> filter_msas("./organized_msas", config=filter_config)  # No output_dir means in-place

        Filter to a new directory with different input/output extensions:

        >>> filter_config = MSAFilterConfig(input_extension=MSAFileExtension.A3M, output_extension=MSAFileExtension.A3M_GZ)
        >>> filter_msas("./organized_msas", "./filtered_msas", config=filter_config)
    """
    if config is None:
        config = MSAFilterConfig()

    input_path = Path(input_dir)
    output_path = input_path if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize HHfilter
    HHFilter.get_or_initialize()

    logger.info(f"Recursively searching for files with extension {config.input_extension} in {input_dir}")
    msa_files = find_files_by_extension(input_path, config.input_extension)
    logger.info(f"Found {len(msa_files)} files with extension {config.input_extension}")

    # Log the filtering configuration
    logger.info(
        f"HHfilter configuration: max_sequences={config.hhfilter.max_sequences}, "
        f"max_identity_percent={config.hhfilter.max_identity_percent}, "
        f"min_coverage_percent={config.hhfilter.min_coverage_percent}"
    )

    if output_dir is None and config.input_extension == config.output_extension:
        logger.info("Filtering files in-place (overwriting originals)!")

    # Set up multiprocessing
    num_workers = config.num_workers or min(cpu_count(), 16)
    chunksize = max(1, min(100, len(msa_files) // max(1, num_workers)))

    logger.info(f"Processing with {num_workers} workers, chunksize={chunksize}")

    filter_func = functools.partial(_filter_single_msa, input_dir=input_path, output_dir=output_path, config=config)
    with Pool(processes=num_workers) as pool:
        list(pool.imap(filter_func, msa_files, chunksize=chunksize))

    logger.info(f"MSA file filtering complete! Processed {len(msa_files)} files.")


def _filter_single_msa(input_file: Path, input_dir: Path, output_dir: Path, config: MSAFilterConfig) -> None:
    """Filter a single MSA file using HHfilter.

    Args:
        input_file: Path to the input MSA file to filter.
        input_dir: Base input directory for calculating relative paths.
        output_dir: Output directory for filtered files.
        config: MSA filtering configuration.
    """
    try:
        # Calculate relative path to preserve directory structure
        rel_path = input_file.relative_to(input_dir)

        # Remove input extension and add output extension
        base_name = input_file.name
        if base_name.endswith(str(config.input_extension)):
            base_name = base_name[: -len(str(config.input_extension))]
        base_name += str(config.output_extension)

        # Create output path
        output_file = output_dir / rel_path.parent / base_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        run_hhfilter(
            input_file=input_file,
            output_file=output_file,
            maxseq=config.hhfilter.max_sequences,
            id=config.hhfilter.max_identity_percent,
            cov=config.hhfilter.min_coverage_percent,
        )
    except Exception as e:
        logger.error(f"Failed to filter {input_file}: {e}")
        raise


def run_hhfilter(
    input_file: PathLike,
    output_file: PathLike,
    maxseq: int = 10_000,
    id: float = 90.0,
    cov: float = 50.0,
) -> None:
    """Run HHfilter on an MSA file with specified parameters.

    Args:
        input_file: Path to input MSA file (handles .gz compression if needed).
        output_file: Path to output filtered file (handles .gz compression if needed).
        maxseq: Maximum number of sequences to keep (default: 10000).
        id: Maximum pairwise sequence identity (%) (default: 90.0).
        cov: Minimum coverage with query (%) (default: 50.0).

    References:
      * `HHfilter Documentation`_ - MSA filtering tool

      .. _HHfilter Documentation: https://github.com/soedinglab/hh-suite/wiki#hhfilter--filter-an-msa
    """
    hhfilter = HHFilter.get_or_initialize()

    input_path = Path(input_file)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if count_sequences_in_msa(input_path) <= maxseq:
        # If already within limits, just copy the file
        logger.debug(f"File {input_path} has <= {maxseq} sequences, transferring to {output_path} without filtering.")
        if output_path and input_path != output_path:
            transfer_with_compression(input_path, output_path, move=False)
        return

    input_is_compressed = is_compressed_file(input_path)
    output_is_compressed = is_compressed_file(output_path)

    # Determine actual input and output files for HHfilter (which can't handle compressed files)
    effective_input = input_path
    effective_output = output_path
    temp_dir = None

    if input_is_compressed or output_is_compressed:
        # Create temp directory for decompressed files
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)

        if input_is_compressed:
            temp_input = temp_dir_path / "input.a3m"
            maybe_decompress_file(input_path, temp_input)
            effective_input = temp_input

        if output_is_compressed:
            # Use temporary output file, we'll compress later
            temp_output = temp_dir_path / "output.a3m"
            effective_output = temp_output

    try:
        # Run HHfilter with appropriate input/output
        hhfilter.run_command(
            "-maxseq",
            str(maxseq),
            "-id",
            str(id),
            "-cov",
            str(cov),
            "-i",
            str(effective_input),
            "-o",
            str(effective_output),
        )

        # Handle compression of output if needed
        if output_is_compressed and effective_output != output_path:
            compress_file(effective_output, output_path)

    except Exception as e:
        logger.error(f"HHfilter failed: {e}")
        raise

    finally:
        # Clean up temporary directory if used
        if temp_dir:
            temp_dir.cleanup()
