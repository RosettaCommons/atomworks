"""Find existing MSA files and their locations."""

import logging
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from atomworks.constants import _load_env_var
from atomworks.enums import MSAFileExtension
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger(__name__)


def get_msa_dirs_from_env(raise_if_not_set: bool = True) -> list[Path] | None:
    """Parse LOCAL_MSA_DIRS environment variable into Path objects.

    Returns:
        List of Path objects for MSA directories from environment variable.
        None if LOCAL_MSA_DIRS is not set or empty and raise_if_not_set is False.

    Raises:
        ValueError: If LOCAL_MSA_DIRS is not set or empty and raise_if_not_set is True.
    """
    local_msa_dirs = _load_env_var("LOCAL_MSA_DIRS")
    if not local_msa_dirs:
        if raise_if_not_set:
            raise ValueError("LOCAL_MSA_DIRS environment variable is not set or empty")
        else:
            return None

    dirs = [Path(dir_path.strip()) for dir_path in local_msa_dirs.split(",")]

    return dirs


def get_msa_depth_and_ext_from_folder(folder: Path, max_depth: int = 10) -> tuple[int, MSAFileExtension]:
    """Automatically detect the shard depth and file extension of an MSA folder.

    Goes down subdirectories one level at a time until finding MSA files.

    Args:
        folder: Top-level MSA directory to analyze.
        max_depth: Maximum depth to search (default: 10).

    Returns:
        Tuple of (shard_depth, extension) where:
        - shard_depth: Number of subdirectory levels (0 = files directly in folder)
        - extension: MSAFileExtension enum value for the file extension found

    Raises:
        ValueError: If no MSA files are found within max_depth levels.

    Examples:
        For structure like `/msa/ab/cd/abcd123.a3m.gz`:

           depth, ext = get_msa_depth_and_ext_from_folder(Path("/msa"))
           # Returns: (2, MSAFileExtension.A3M_GZ)
    """
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    # All possible MSA extensions to check
    msa_extensions = [ext.value for ext in MSAFileExtension]

    current_dir = folder
    depth = 0

    while depth <= max_depth:
        found_subdir = False
        # Check if current directory contains any MSA files
        for item in current_dir.iterdir():
            if item.is_file():
                for ext_str in msa_extensions:
                    if item.name.endswith(ext_str):
                        matching_ext = MSAFileExtension(ext_str)
                        return depth, matching_ext
            else:  # it's a directory
                next_subdir = item
                found_subdir = True

        if not found_subdir:
            break

        current_dir = next_subdir
        depth += 1

    raise ValueError(
        f"No MSA files found in {folder} within {max_depth} levels. "
        f"Searched for extensions: {', '.join(msa_extensions)}"
    )


def _build_msa_file_paths(
    sequence_hash: str, msa_dir: Path, shard_depths: list[int], extensions: list[str]
) -> list[Path]:
    """Build all possible MSA file paths for a sequence hash.

    Args:
        sequence_hash: Hash of the protein sequence.
        msa_dir: Base MSA directory.
        shard_depths: List of sharding depths to check.
        extensions: List of file extensions to check.

    Returns:
        List of possible file paths for the sequence.
    """
    possible_paths = []

    for shard_depth in shard_depths:
        # Build shard path like "ab/cd/" for depth 2 with hash "abcd123..."
        shard_path = "".join([f"{sequence_hash[(i * 2) : (i + 1) * 2]}/" for i in range(shard_depth)])

        for extension in extensions:
            file_path = msa_dir / shard_path / f"{sequence_hash}{extension}"
            possible_paths.append(file_path)

    return possible_paths


def sequence_has_msa(
    sequence: str,
    msa_dirs: list[PathLike] | None = None,
    shard_depths: list[int] | None = None,
    extensions: list[MSAFileExtension] | None = None,
) -> bool:
    """Check if a sequence has an existing MSA file in any directory.

    Args:
        sequence: Protein sequence to check.
        msa_dirs: Directories to search. If None, uses LOCAL_MSA_DIRS env var.
        shard_depths: Sharding levels to check. Defaults to [0, 1, 2, 3, 4].
        extensions: File extensions to check. Defaults to [A3M, A3M_GZ, A3M_ZST].

    Returns:
        True if MSA exists, False otherwise.
    """
    # Set defaults
    if msa_dirs is None:
        msa_dirs = get_msa_dirs_from_env()
    else:
        msa_dirs = [Path(d) for d in msa_dirs]

    if shard_depths is None:
        shard_depths = [0, 1, 2, 3, 4]

    if extensions is None:
        extensions = [MSAFileExtension.A3M, MSAFileExtension.A3M_GZ, MSAFileExtension.A3M_ZST]

    # Convert extensions to string list
    extension_strs = [ext.value if isinstance(ext, MSAFileExtension) else str(ext) for ext in extensions]

    # Filter existing directories
    existing_dirs = [d for d in msa_dirs if d.exists()]
    if not existing_dirs:
        logger.warning("No existing MSA directories found")
        return False

    sequence_hash = hash_sequence(sequence)

    for msa_dir in existing_dirs:
        possible_paths = _build_msa_file_paths(sequence_hash, msa_dir, shard_depths, extension_strs)

        # Return True as soon as we find the first existing file
        for path in possible_paths:
            if path.exists():
                logger.debug(f"Found existing MSA for sequence hash {sequence_hash}: {path}")
                return True

    return False


def find_msas(
    sequences: list[str],
    msa_dirs: list[PathLike] | None = None,
    shard_depths: list[int] | None = None,
    extensions: list[MSAFileExtension] | None = None,
) -> tuple[list[str], dict[str, Path]]:
    """Find existing MSA files for sequences and return missing sequences with MSA path mapping.

    Args:
        sequences: Protein sequences to find MSAs for.
        msa_dirs: Directories to search. If None, uses LOCAL_MSA_DIRS env var.
        shard_depths: Sharding levels to check. Defaults to [0, 1, 2, 3, 4].
        extensions: File extensions to check. Defaults to [A3M, A3M_GZ, A3M_ZST].

    Returns:
        Tuple of (missing_sequences, sequence_to_msa_path) where:
        - missing_sequences: List of sequences without existing MSA files
        - sequence_to_msa_path: Dict mapping sequences to their MSA file paths

    Examples:
        Find MSAs with default settings:

        .. code-block:: python

           sequences = ["MKKKEVE...", "MSYIWRQ..."]
           missing, found_paths = find_msas(sequences)

        Find MSAs in specific directories:

        .. code-block:: python

           from pathlib import Path

           dirs = [Path("/projects/msa/chembl"), Path("/projects/msa/mmseqs_gpu")]
           missing, found_paths = find_msas(sequences, msa_dirs=dirs)
    """
    logger.info(f"Finding MSAs for {len(sequences)} sequences")

    # Set defaults
    if msa_dirs is None:
        msa_dirs = get_msa_dirs_from_env()
    else:
        msa_dirs = [Path(d) for d in msa_dirs]

    if shard_depths is None:
        shard_depths = [0, 1, 2, 3, 4]

    if extensions is None:
        extensions = [MSAFileExtension.A3M, MSAFileExtension.A3M_GZ, MSAFileExtension.A3M_ZST]

    # Convert extensions to string list
    extension_strs = [ext.value if isinstance(ext, MSAFileExtension) else str(ext) for ext in extensions]

    # Filter existing directories
    existing_dirs = [d for d in msa_dirs if d.exists()]
    if not existing_dirs:
        logger.warning("No existing MSA directories found")
        return sequences.copy(), {}

    # Find MSAs for each sequence
    missing_sequences = []
    sequence_to_msa_path = {}

    for sequence in tqdm(sequences, desc="Finding existing MSAs", unit="seq"):
        sequence_hash = hash_sequence(sequence)
        found_path = None

        # Search for MSA file in all directories
        for msa_dir in existing_dirs:
            possible_paths = _build_msa_file_paths(sequence_hash, msa_dir, shard_depths, extension_strs)
            for path in possible_paths:
                if path.exists():
                    found_path = path
                    logger.debug(f"Found existing MSA for sequence hash {sequence_hash}: {path}")
                    break
            if found_path:
                break

        if found_path:
            sequence_to_msa_path[sequence] = found_path
        else:
            missing_sequences.append(sequence)

    found_count = len(sequence_to_msa_path)
    logger.info(f"Found {found_count} existing MSAs, {len(missing_sequences)} sequences need generation")

    return missing_sequences, sequence_to_msa_path
