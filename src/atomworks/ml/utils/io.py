"""I/O utilities for ML components.

Provides functions for file operations, directory scanning, and data loading.
"""

import contextlib
import gzip
import hashlib
import io
import os
import time
from collections.abc import Callable, Iterator
from functools import cache, wraps
from os import PathLike
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import zstandard as zstd
from filelock import FileLock

from atomworks.common import string_to_md5_hash
from atomworks.io.utils.io_utils import apply_sharding_pattern, build_sharding_pattern
from atomworks.ml.utils.huggingface import download_hf_metadata
from atomworks.ml.utils.misc import logger

try:
    import boto3
    from botocore.client import BaseClient
    from botocore.config import Config
except ImportError:
    boto3 = None
    BaseClient = None


@cache
def _s3_client(endpoint_url: str | None = None) -> "BaseClient":
    """Process-wide cached boto3 S3 client.

    Region/credentials come from the ambient AWS config; ``endpoint_url`` defaults to ``$AWS_ENDPOINT_URL``.
    """
    if boto3 is None:
        raise ImportError("boto3 is required for s3:// paths. Install it with: uv pip install boto3")
    # virtual-host addressing, retry throttling with backoff, larger pool for DataLoader fan-out
    config = Config(
        max_pool_connections=50,
        retries={"max_attempts": 10, "mode": "standard"},
        s3={"addressing_style": "virtual"},
    )
    return boto3.client("s3", endpoint_url=endpoint_url or os.environ.get("AWS_ENDPOINT_URL"), config=config)


def read_s3_bytes(
    url: str, *, offset: int | None = None, length: int | None = None, endpoint_url: str | None = None
) -> bytes:
    """Read an ``s3://`` object (whole, or a ``[offset, offset+length)`` byte range) into memory."""
    bucket, key = url[len("s3://") :].split("/", 1)
    kwargs = {} if offset is None else {"Range": f"bytes={offset}-{offset + length - 1}"}
    return _s3_client(endpoint_url).get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_GZIP_MAGIC = b"\x1f\x8b"


def _decompress(raw: bytes) -> bytes:
    """Auto-detect and decompress zstd/gzip data, or return the raw bytes."""
    if raw[:4] == _ZSTD_MAGIC:
        # stream_reader (not .decompress) so frames without an embedded content size still read
        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as r:
            return r.read()
    if raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw)
    return raw


def open_file(filename: PathLike) -> TextIO:
    """Open a file, handling compressed files if necessary.

    Args:
        filename: The path to the file to open.

    Returns:
        A file-like object for reading.

    Raises:
        AssertionError: If the file does not exist.
    """
    # Pass an already-open text stream straight through (e.g. in-memory MSA bytes) — nothing on disk to open.
    if hasattr(filename, "read"):
        return filename
    filename = Path(filename)
    # ...assert that the file exists
    assert filename.exists(), f"File {filename} does not exist"
    # ...open the file for reading, accepting gzipped, zstd, or plaintext files
    if filename.suffix == ".gz":
        return gzip.open(filename, "rt")
    elif filename.suffix == ".zst":
        # Open zstd file and wrap in TextIOWrapper for text mode
        # Note: The file handle is managed by the TextIOWrapper/stream_reader
        dctx = zstd.ZstdDecompressor()
        fh = open(filename, "rb")  # noqa: SIM115
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return filename.open("r")


@contextlib.contextmanager
def opened_file(filename: PathLike) -> "Iterator[TextIO]":
    """Open `filename` for reading, closing it on exit only if this call opened it.

    A caller-supplied stream is passed through by `open_file` and stays the caller's to close.
    """
    stream = open_file(filename)
    try:
        yield stream
    finally:
        if stream is not filename:
            stream.close()


def scan_directory(dir_path: PathLike, max_depth: int) -> list[str]:
    """Fast, order-independent directory scan for files up to max_depth levels deep.

    Args:
        dir_path: The root directory to scan.
        max_depth: The maximum depth to scan. A max_depth of 1 means only the top-level directory.

    Returns:
        A list of file paths found within the specified directory and depth.
    """
    file_paths = []

    for root, dirs, files in os.walk(dir_path):
        current_depth = len(Path(root).relative_to(dir_path).parts)

        if current_depth >= max_depth:
            dirs.clear()
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


def cache_based_on_subset_of_args(cache_keys: list[str], maxsize: int | None = None) -> Callable:
    """Decorator to cache function results based on a subset of its keyword arguments.

    Most helpful when some arguments may be unhashable types (e.g., dictionaries, AtomArray).
    If the value of any of the cache keys is None, the function is executed and the result is not cached.

    Note:
        The wrapped function must use keyword arguments for those specified in cache_keys.
        Positional arguments are not supported for cache key extraction.

    Args:
        cache_keys: The names of the keyword arguments to use as the cache key.
        maxsize: The maximum number of entries to store in the cache.
            If None, the cache size is unlimited.

    Returns:
        A decorator that caches the function results based on the specified keyword arguments.

    Example:
        .. code-block:: python

            @cache_based_on_subset_of_args(["arg1"], maxsize=2)
            def function(*, arg1, arg2):
                return arg1 + arg2


            result1 = function(arg1=1, arg2=2)  # Caches with key 1
            result2 = function(arg1=1, arg2=3)  # Retrieves from cache
    """

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order: list[tuple[Any, ...]] = []  # To track the order of keys for eviction

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract the cache key values from kwargs
            key_values = tuple(kwargs.get(key) for key in cache_keys)

            # Check if any of the key values are None
            if None in key_values:
                # Bypass caching if any key value is None
                return func(*args, **kwargs)

            # Use the key values to form a unique cache key
            cache_key = tuple(key_values)

            if cache_key not in cache:
                # Evict the oldest entry if the cache is full
                if maxsize is not None and len(cache) >= maxsize:
                    oldest_key = cache_order.pop(0)
                    del cache[oldest_key]

                # Cache the result
                cache[cache_key] = func(*args, **kwargs)
                cache_order.append(cache_key)
            return cache[cache_key]

        return wrapper

    return decorator


def cache_to_disk_as_pickle(cache_dir: PathLike | None = None, *, directory_depth: int = 2) -> Callable:
    """
    Cache function results to disk as zstd-3 compressed pickle files.

    Creates a unique cached pickle file for each set of function arguments using an MD5 hash.
    If the cache file exists, the result is loaded from the file. Otherwise, the
    function is called, and the result is saved to the cache file.

    If `cache_dir` is `None`, caching is disabled and the function is always executed.

    Args:
        cache_dir (PathLike or None): The directory where cache files will be stored, or
            `None` to disable caching.
        directory_depth (int): The depth of the directory structure for sharding cache files.

    Returns:
        function: The wrapped function with optional disk caching enabled.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cache_dir is None:
                # If caching is disabled, always execute the function
                return func(*args, **kwargs)

            # ... create cache directory if it doesn't exist
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            # ... create a unique cache file path based on the MD5 hash of function arguments
            args_repr = f"{args}_{kwargs}"
            hash_hex = hashlib.md5(args_repr.encode()).hexdigest()
            sharding_pattern = build_sharding_pattern(depth=directory_depth, chars_per_dir=2)
            sharded_path = apply_sharding_pattern(hash_hex, sharding_pattern)
            cache_file = Path(cache_dir) / sharded_path.with_suffix(".pkl.zst")

            # ... check if cache file exists
            if cache_file.exists():
                try:
                    # ... try to load the result from cache file
                    return pd.read_pickle(cache_file)

                except Exception as e:
                    # (Fallback to executing the function, with a warning)
                    logger.error(f"Error loading cache file {cache_file}: {e}")

            # If cache file doesn't exist, execute the function
            result = func(*args, **kwargs)

            # ... save the result to cache file, creating directories if necessary
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(result, cache_file, compression={"method": "zstd", "level": 3})

            return result

        return wrapper

    return decorator


def to_parquet_with_metadata(df: pd.DataFrame, filepath: PathLike, **kwargs: Any) -> None:
    """Convenience wrapper around df.to_parquet that saves table-wide metadata (df.attrs) to the parquet file.

    Args:
        df: pandas DataFrame to save.
        filepath: Path where to save the parquet file.
        **kwargs: Additional arguments to pass to df.to_parquet.
    """
    # Use df.attrs as metadata
    metadata = df.attrs.copy() if hasattr(df, "attrs") else {}

    # Convert metadata dictionary to strings
    string_metadata = {str(key): str(value) for key, value in metadata.items()}

    # Convert pandas DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)

    # Add metadata to the table
    table_metadata = table.schema.metadata
    table_metadata.update({k.encode(): v.encode() for k, v in string_metadata.items()})

    # Create new table with updated metadata
    table = table.replace_schema_metadata(table_metadata)

    # Write to parquet
    pq.write_table(table, filepath, **kwargs)


def _readable_source(filepath: PathLike) -> Any:
    """Resolve a local path, cached HF metadata file, or downloaded S3 buffer for pandas/pyarrow."""
    if str(filepath).startswith("hf://"):
        return download_hf_metadata(str(filepath))
    return io.BytesIO(read_s3_bytes(str(filepath))) if str(filepath).startswith("s3://") else filepath


def read_csv(filepath: PathLike, **kwargs: Any) -> pd.DataFrame:
    """``pd.read_csv`` that also accepts an ``s3://`` URL (read via boto3, like :func:`read_parquet_with_metadata`)."""
    return pd.read_csv(_readable_source(filepath), **kwargs)


def read_parquet_with_metadata(filepath: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper around pd.read_parquet that preserves metadata.

    Args:
        filepath: Path to the parquet file.
        **kwargs: Additional arguments to pass to pd.read_parquet.

    Returns:
        pandas DataFrame with metadata in .attrs attribute

    ``filepath`` may be a local path or an ``s3://`` URL (downloaded once via boto3 — which reads the ambient
    AWS profile, incl. endpoint + addressing_style — since bare ``pq.read_schema`` does not accept ``s3://``).
    """
    src = _readable_source(filepath)

    # Read the parquet schema using pyarrow, then the DataFrame using pandas (from the same buffer for s3).
    schema = pq.read_schema(src)
    raw_metadata = schema.metadata or {}
    metadata_dict = {k.decode(): v.decode() for k, v in raw_metadata.items() if k not in (b"pandas", b"pyarrow_schema")}

    if isinstance(src, io.BytesIO):
        src.seek(0)
    df = pd.read_parquet(src, **kwargs)
    df.attrs = metadata_dict
    return df


def job_scoped_feather_path(
    name: str,
    *,
    local_drive_mount: str,
    job_id_env_var: str = "SLURM_JOB_ID",
    content_key: Any = None,
) -> str:
    """Return ``<local_drive_mount>/<job_id>/<name>[-<hash>].feather``, namespaced per job.

    Args:
        name: Human-readable dataset name; the leading path component.
        local_drive_mount: Root directory for feather files.
        job_id_env_var: Environment variable used to namespace files per job.
        content_key: Anything that decides what the feather *contains* (source path,
            filters, columns). Two datasets sharing a ``name`` within one job must not
            share a file unless this matches, so it is hashed into the filename.

    Examples:
        >>> a = job_scoped_feather_path("pdb", local_drive_mount="/tmp", content_key=["n_nuc == 0"])
        >>> b = job_scoped_feather_path("pdb", local_drive_mount="/tmp", content_key=[])
        >>> a != b
        True
    """
    job_id = os.environ.get(job_id_env_var, f"manual_{os.getpid()}_{int(time.time())}")
    # 16 hex chars, not 8: a collision here silently serves one dataset's rows to another,
    # which is the failure this key exists to prevent.
    suffix = f"-{string_to_md5_hash(repr(content_key), truncate=16)}" if content_key is not None else ""
    return os.path.join(local_drive_mount, job_id, f"{name}{suffix}.feather")


def build_feather_once(feather_path: str, build_table: Callable[[], pa.Table]) -> str:
    """Build ``feather_path`` once under a lock, publishing atomically; return the path.

    Ranks/workers race on the same path: the first to win the lock builds via ``build_table`` and
    ``os.replace``s it into place (atomic on the same filesystem); the rest see the finished file and
    skip. ``build_table`` is called only when the feather is missing, so the source read happens once.
    """
    os.makedirs(os.path.dirname(feather_path), exist_ok=True)
    with FileLock(feather_path + ".lock"):
        if not os.path.exists(feather_path):
            tmp_path = f"{feather_path}.tmp.{os.getpid()}"
            try:
                feather.write_feather(build_table(), tmp_path, compression="uncompressed")
                os.replace(tmp_path, feather_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)  # don't leave a partial temp behind on failure
                raise
    return feather_path
