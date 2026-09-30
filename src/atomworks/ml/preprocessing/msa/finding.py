"""Find existing MSA files and their locations."""

import logging
import warnings
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from atomworks.constants import _load_env_var
from atomworks.enums import MSAFileExtension
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger(__name__)


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
        For structure like ``/msa/ab/cd/abcd123.a3m.gz``:

        >>> depth, ext = get_msa_depth_and_ext_from_folder(Path("/msa"))
        >>> # Returns: (2, MSAFileExtension.A3M_GZ)
    """
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    # All possible MSA extensions to check
    msa_extensions = [ext.value for ext in MSAFileExtension]

    current_dir = folder
    depth = 0

    while depth <= max_depth:
        # Separate file detection from subdirectory selection for efficiency:
        # First check all files for MSA matches, then find first subdir if needed
        files = []
        subdirs = []
        for item in current_dir.iterdir():
            if item.is_file():
                files.append(item)
            else:
                subdirs.append(item)

        # Check files for MSA matches
        for item in files:
            for ext_str in msa_extensions:
                if item.name.endswith(ext_str):
                    matching_ext = MSAFileExtension(ext_str)
                    return depth, matching_ext

        # No MSA files found, descend into first subdirectory (sorted for determinism)
        if not subdirs:
            break

        current_dir = min(subdirs)  # Deterministic: pick lexicographically first
        depth += 1

    raise ValueError(
        f"No MSA files found in {folder} within {max_depth} levels. "
        f"Searched for extensions: {', '.join(msa_extensions)}"
    )


def _auto_detect_msa_dir_metadata(msa_dir: Path) -> dict | None:
    """Auto-detect MSA directory metadata (depth and extension).

    Args:
        msa_dir: Path to the MSA directory.

    Returns:
        Dict with 'dir', 'extension', 'directory_depth' keys, or None if
        detection fails or directory doesn't exist.
    """
    if not msa_dir.exists():
        logger.warning(f"MSA directory does not exist: {msa_dir}")
        return None
    try:
        depth, ext = get_msa_depth_and_ext_from_folder(msa_dir)
        return {
            "dir": str(msa_dir),
            "extension": ext.value,
            "directory_depth": depth,
        }
    except ValueError as e:
        logger.warning(f"Could not auto-detect MSA format for {msa_dir}: {e}")
        return None


def get_msa_dirs(
    env_var_name: str = "PROTEIN_MSA_DIRS",
    raise_if_not_set: bool = False,
) -> list[dict]:
    """Get MSA directories with auto-detected metadata from an environment variable.

    Args:
        env_var_name: Environment variable with comma-separated paths.
        raise_if_not_set: Raise if env var is unset. Defaults to ``False``.

    Returns:
        List of dicts with ``dir``, ``extension``, and ``directory_depth`` keys.
    """
    msa_dirs_str = _load_env_var(env_var_name)
    if not msa_dirs_str:
        if raise_if_not_set:
            raise ValueError(f"{env_var_name} environment variable is not set or empty")
        return []

    # Parse comma-separated paths
    msa_dirs = [Path(p.strip()) for p in msa_dirs_str.split(",") if p.strip()]

    result = []
    for msa_dir in msa_dirs:
        metadata = _auto_detect_msa_dir_metadata(msa_dir)
        if metadata is not None:
            result.append(metadata)

    return result


# Backwards compatibility alias
def get_msa_dirs_from_env(raise_if_not_set: bool = False) -> list[dict]:
    """Deprecated: Use :py:func:`get_msa_dirs` instead."""
    warnings.warn(
        "get_msa_dirs_from_env is deprecated, use get_msa_dirs('PROTEIN_MSA_DIRS') instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_msa_dirs("PROTEIN_MSA_DIRS", raise_if_not_set=raise_if_not_set)


def _build_msa_file_path(sequence_hash: str, msa_dir: str, depth: int, extension: str) -> Path:
    """Build the MSA file path for a sequence hash.

    Args:
        sequence_hash: Hash of the protein sequence.
        msa_dir: Base MSA directory.
        depth: Shard depth (0 = flat, 2 = "ab/cd/" style).
        extension: File extension (e.g., ".a3m.gz").

    Returns:
        Path to the MSA file.
    """
    # Build shard path like "ab/cd/" for depth 2 with hash "abcd123..."
    shard_path = "".join([f"{sequence_hash[(i*2):(i+1)*2]}/" for i in range(depth)])
    return Path(msa_dir) / shard_path / f"{sequence_hash}{extension}"


def _normalize_msa_dirs(msa_dirs: list[dict] | list[PathLike] | None) -> list[dict]:
    """Normalize MSA directories to list of dicts with metadata.

    Args:
        msa_dirs: MSA directories in one of three formats:
            - None: Load from PROTEIN_MSA_DIRS env var
            - list[PathLike]: Simple format, auto-detects metadata for each directory
            - list[dict]: Pre-computed format with 'dir', 'extension', 'directory_depth' keys

    Returns:
        List of dicts with 'dir', 'extension', 'directory_depth' keys.

    Raises:
        TypeError: If list contains mixed types (must be all dicts or all PathLike).
    """
    if msa_dirs is None:
        return get_msa_dirs(raise_if_not_set=False)

    if not msa_dirs:
        return []

    # Check if already in dict format - validate all elements are dicts
    if isinstance(msa_dirs[0], dict):
        if not all(isinstance(d, dict) for d in msa_dirs):
            raise TypeError("msa_dirs must be homogeneous: all elements must be dicts or all PathLike, not mixed")
        return msa_dirs

    # Simple PathLike format - auto-detect metadata for each directory
    result = []
    for msa_dir in msa_dirs:
        metadata = _auto_detect_msa_dir_metadata(Path(msa_dir))
        if metadata is not None:
            result.append(metadata)

    return result


def sequence_has_msa(
    sequence: str,
    msa_dirs: list[dict] | list[PathLike] | None = None,
) -> bool:
    """Check if a sequence has an existing MSA file in any directory.

    Args:
        sequence: Protein sequence to check.
        msa_dirs: Directories to search. Accepts:
            - None: Uses PROTEIN_MSA_DIRS env var
            - list[PathLike]: Simple format, auto-detects metadata
            - list[dict]: Pre-computed format with 'dir', 'extension', 'directory_depth' keys

    Returns:
        True if MSA exists, False otherwise.
    """
    normalized_dirs = _normalize_msa_dirs(msa_dirs)
    if not normalized_dirs:
        logger.warning("No MSA directories found")
        return False

    sequence_hash = hash_sequence(sequence)

    for dir_info in normalized_dirs:
        path = _build_msa_file_path(
            sequence_hash,
            dir_info["dir"],
            dir_info["directory_depth"],
            dir_info["extension"],
        )
        if path.exists():
            logger.debug(f"Found existing MSA for sequence hash {sequence_hash}: {path}")
            return True

    return False


def find_msas(
    sequences: list[str],
    msa_dirs: list[dict] | list[PathLike] | None = None,
) -> tuple[list[str], dict[str, Path]]:
    """Find existing MSA files for sequences and return missing sequences with MSA path mapping.

    Args:
        sequences: Protein sequences to find MSAs for.
        msa_dirs: Directories to search. Accepts:
            - None: Uses PROTEIN_MSA_DIRS env var
            - list[PathLike]: Simple format, auto-detects metadata
            - list[dict]: Pre-computed format with 'dir', 'extension', 'directory_depth' keys

    Returns:
        Tuple of (missing_sequences, sequence_to_msa_path) where:
        - missing_sequences: List of sequences without existing MSA files
        - sequence_to_msa_path: Dict mapping sequences to their MSA file paths

    Examples:
        Find MSAs with default settings (uses PROTEIN_MSA_DIRS env var):

        .. code-block:: python

           sequences = ["MKKKEVE...", "MSYIWRQ..."]
           missing, found_paths = find_msas(sequences)

        Find MSAs with explicit directories:

        .. code-block:: python

           dirs = get_msa_dirs("MY_MSA_DIRS")
           missing, found_paths = find_msas(sequences, msa_dirs=dirs)
    """
    logger.info(f"Finding MSAs for {len(sequences)} sequences")

    normalized_dirs = _normalize_msa_dirs(msa_dirs)
    if not normalized_dirs:
        logger.warning("No MSA directories found")
        return sequences.copy(), {}

    # Find MSAs for each sequence
    missing_sequences = []
    sequence_to_msa_path = {}

    for sequence in tqdm(sequences, desc="Finding existing MSAs", unit="seq"):
        sequence_hash = hash_sequence(sequence)
        found_path = None

        # Search for MSA file in all directories
        for dir_info in normalized_dirs:
            path = _build_msa_file_path(
                sequence_hash,
                dir_info["dir"],
                dir_info["directory_depth"],
                dir_info["extension"],
            )
            if path.exists():
                found_path = path
                logger.debug(f"Found existing MSA for sequence hash {sequence_hash}: {path}")
                break

        if found_path:
            sequence_to_msa_path[sequence] = found_path
        else:
            missing_sequences.append(sequence)

    found_count = len(sequence_to_msa_path)
    logger.info(f"Found {found_count} existing MSAs, {len(missing_sequences)} sequences need generation")

    return missing_sequences, sequence_to_msa_path
