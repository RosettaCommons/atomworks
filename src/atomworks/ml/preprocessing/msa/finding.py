"""Find existing MSA files and their locations."""

import logging
import warnings
from os import PathLike
from pathlib import Path

from tqdm import tqdm

from atomworks.constants import _load_env_var
from atomworks.enums import MSAFileExtension
from atomworks.io.utils.io_utils import apply_sharding_pattern, build_sharding_pattern
from atomworks.ml.utils.misc import get_complex_id, hash_sequence

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


def _normalize_template_alignment_dirs(template_dirs: list[dict] | list[PathLike] | None) -> list[dict]:
    """Normalize template-alignment directories to the same dict shape MSA dirs use.

    Unlike `_normalize_msa_dirs`, PathLike entries aren't auto-detected: template
    alignment files (``.m8`` hit tables) aren't a `MSAFileExtension` value, so a
    plain PathLike entry is assumed to match
    :py:func:`~atomworks.ml.preprocessing.msa.organizing.organize_template_alignments`'s
    defaults (``.m8``, sharded 2 chars deep).

    Args:
        template_dirs: Template alignment directories in one of three formats:
            - list[PathLike]: Simple format, auto-detects metadata for each directory
            - list[dict]: Pre-computed format with 'dir', 'extension', 'directory_depth' keys

    Returns:
        List of dicts with 'dir', 'extension', 'directory_depth' keys.
    """
    if not template_dirs:
        return []

    if isinstance(template_dirs[0], dict):
        if not all(isinstance(d, dict) for d in template_dirs):
            raise TypeError("template_dirs must be homogeneous: all elements must be dicts or all PathLike, not mixed")
        return template_dirs

    return [{"dir": str(d), "extension": ".m8", "directory_depth": 1} for d in template_dirs]


def find_template_alignments(
    sequences: list[str],
    template_dirs: list[dict] | list[PathLike] | None = None,
) -> tuple[list[str], dict[str, Path]]:
    """Find existing raw template-alignment (``pdb70.m8``) files for sequences.

    Mirrors :py:func:`find_msas`'s shape and hash/shard convention: template
    alignments are looked up by the same sequence hash as MSAs, just under a
    different directory/extension. To download the hit structures, see
    :py:mod:`~atomworks.ml.preprocessing.msa.template_structures`.

    Args:
        sequences: Protein sequences to find template alignments for.
        template_dirs: Directories to search. Accepts:
            - None: No directories (all sequences reported missing).
            - list[PathLike]: Auto-assumes ``.m8`` / directory_depth=2 (matching
              `organize_template_alignments`'s defaults).
            - list[dict]: Pre-computed format with 'dir', 'extension',
              'directory_depth' keys.

    Returns:
        Tuple of (missing_sequences, sequence_to_template_alignment_path), same
        shape as :py:func:`find_msas`.
    """
    logger.info(f"Finding template alignments for {len(sequences)} sequences")

    normalized_dirs = _normalize_template_alignment_dirs(template_dirs)
    if not normalized_dirs:
        logger.warning("No template alignment directories found")
        return sequences.copy(), {}

    missing_sequences = []
    sequence_to_template_alignment_path = {}

    for sequence in sequences:
        sequence_hash = hash_sequence(sequence)
        found_path = None

        for dir_info in normalized_dirs:
            path = _build_msa_file_path(
                sequence_hash, dir_info["dir"], dir_info["directory_depth"], dir_info["extension"]
            )
            if path.exists():
                found_path = path
                logger.debug(f"Found existing template alignment for sequence hash {sequence_hash}: {path}")
                break

        if found_path:
            sequence_to_template_alignment_path[sequence] = found_path
        else:
            missing_sequences.append(sequence)

    found_count = len(sequence_to_template_alignment_path)
    logger.info(f"Found {found_count} existing template alignments, {len(missing_sequences)} sequences need generation")

    return missing_sequences, sequence_to_template_alignment_path


def _normalize_paired_msa_dirs(msa_dirs: list[dict] | list[PathLike] | None) -> list[dict]:
    """Normalize paired-MSA directories to list of dicts with metadata.

    Paired MSAs sit one level deeper, under a per-``complex_id`` directory, so
    :py:func:`_normalize_msa_dirs`'s depth auto-detection over-counts by one. Plain
    paths instead assume the writer's default depth of 1 and are searched for every
    :py:class:`~atomworks.enums.MSAFileExtension`. Dicts (e.g. with an explicit
    ``directory_depth`` for other layouts) pass through unchanged.
    """
    if not msa_dirs or isinstance(msa_dirs[0], dict):
        return _normalize_msa_dirs(msa_dirs)

    return [{"dir": str(d), "extension": ext.value, "directory_depth": 1} for d in msa_dirs for ext in MSAFileExtension]


def _build_paired_msa_file_path(
    complex_id: str, sequence_hash: str, msa_dir: str, extension: str, complex_directory_depth: int = 1
) -> Path:
    """Build the paired-MSA file path for a sequence hash within a given complex.

    complex_id is sharded the same way an unpaired sequence hash is (default: one
    2-char directory level) -- a large fine-tuning run can have many complexes, so
    this avoids one huge flat directory of complex_ids. Chain files *within* a
    complex's own directory stay flat (a complex has few chains). See
    :py:func:`~atomworks.ml.preprocessing.msa.colabfold_server.make_msas_colabfold_server`
    for the matching writer -- both use the same `build_sharding_pattern`/
    `apply_sharding_pattern` helpers for the complex_id shard, so a
    `complex_directory_depth` here always corresponds to the writer's
    `sharding_pattern` at the same depth (unlike unpaired lookups' `directory_depth`
    vs. `organize_msas`'s `sharding_pattern`, which use different units -- see
    test_organizing.py).

    Args:
        complex_id: Identifier for the complex, from `get_complex_id`.
        sequence_hash: Hash of the chain's protein sequence.
        msa_dir: Base directory containing one (possibly sharded) subdirectory per complex_id.
        extension: File extension (e.g., ".a3m.gz").
        complex_directory_depth: Shard depth for complex_id (0 = flat, 1 = "ab/" style,
            matching `make_msas_colabfold_server`'s default `sharding_pattern="/0:2/"`).

    Returns:
        Path to the paired MSA file.
    """
    complex_shard_pattern = build_sharding_pattern(depth=complex_directory_depth, chars_per_dir=2)
    sharded_complex_path = apply_sharding_pattern(complex_id, complex_shard_pattern)
    return Path(msa_dir) / sharded_complex_path / f"{sequence_hash}{extension}"


def get_paired_msa_path(
    complex_id: str,
    sequence: str,
    msa_dirs: list[dict] | list[PathLike] | None = None,
) -> Path | None:
    """Retrieve the path to a chain's paired MSA file within a given complex.

    Companion to ``get_msa_path`` (in ``atomworks.ml.transforms.msa._msa_loading_utils``)
    for paired MSAs, which need two keys -- which complex, and which chain within it --
    rather than a sequence hash alone.

    Args:
        complex_id: Identifier for the complex, from `get_complex_id`.
        sequence: The chain's protein sequence, within that complex.
        msa_dirs: Directories to search, each containing one (possibly sharded)
            subdirectory per complex_id. Plain paths assume the writer's default layout
            (see :py:func:`_normalize_paired_msa_dirs`); dicts use the same format as
            :py:func:`find_msas`, where `directory_depth` shards complex_id (default 1,
            matching `make_msas_colabfold_server`'s default), not the chain hash --
            chain files within a complex directory are always flat.

    Returns:
        The MSA file path if found, else None.
    """
    normalized_dirs = _normalize_paired_msa_dirs(msa_dirs)
    sequence_hash = hash_sequence(sequence)

    for dir_info in normalized_dirs:
        complex_depth = dir_info.get("directory_depth", 1)
        path = _build_paired_msa_file_path(
            complex_id, sequence_hash, dir_info["dir"], dir_info["extension"], complex_depth
        )
        if path.exists():
            logger.debug(f"Found existing paired MSA for complex {complex_id}, sequence hash {sequence_hash}: {path}")
            return path

    return None


def find_paired_msas(
    sequences: list[str],
    msa_dirs: list[dict] | list[PathLike] | None = None,
) -> tuple[list[str], dict[str, Path]]:
    """Find the paired MSA files for one complex's full unique sequence set.

    Mirrors :py:func:`find_msas`'s shape, but for a paired/complex query:
    `sequences` is treated as the complete set of unique chains in ONE complex --
    the same unit `organize_paired_msas`/`make_msas_colabfold_server` operate on --
    and complex_id is derived from them automatically via `get_complex_id`, not
    something the caller computes or passes separately.

    Args:
        sequences: The full unique-sequence set of one complex.
        msa_dirs: Directories to search, each containing paired MSA output (e.g.
            `make_msas_colabfold_server`'s ``<output_dir>/paired``). Accepts the
            same formats as :py:func:`find_msas`.

    Returns:
        Tuple of (missing_sequences, sequence_to_msa_path), same shape as
        :py:func:`find_msas`.
    """
    logger.info(f"Finding paired MSAs for {len(sequences)} sequences (one complex)")

    normalized_dirs = _normalize_paired_msa_dirs(msa_dirs)
    if not normalized_dirs:
        logger.warning("No MSA directories found")
        return sequences.copy(), {}

    complex_id = get_complex_id(sequences)

    missing_sequences = []
    sequence_to_msa_path = {}
    for sequence in sequences:
        found_path = get_paired_msa_path(complex_id, sequence, normalized_dirs)
        if found_path:
            sequence_to_msa_path[sequence] = found_path
        else:
            missing_sequences.append(sequence)

    found_count = len(sequence_to_msa_path)
    logger.info(f"Found {found_count} existing paired MSAs, {len(missing_sequences)} sequences need generation")

    return missing_sequences, sequence_to_msa_path
