"""I/O utilities for ML components.

Provides functions for file operations, directory scanning, and data loading.
"""

import gzip
import hashlib
import os
import pickle
import re
from collections.abc import Callable
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from atomworks.ml.utils.misc import (
    logger,
)


def open_file(filename: PathLike) -> TextIO:
    """Open a file, handling gzipped files if necessary.

    Args:
        filename: The path to the file to open.

    Returns:
        A file-like object for reading.

    Raises:
        AssertionError: If the file does not exist.
    """
    filename = Path(filename)
    # ...assert that the file exists
    assert filename.exists(), f"File {filename} does not exist"
    # ...open the file for reading, accepting either gzipped or plaintext files
    if filename.suffix == ".gz":
        return gzip.open(filename, "rt")
    return filename.open("r")


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


def cache_to_disk_as_pickle(
    cache_dir: PathLike | None = None, use_gzip: bool = True, directory_depth: int = 2
) -> Callable:
    """
    A decorator to cache the results of a function to disk as a pickle file.

    Creates a unique cached pickle file for each set of function arguments using an MD5 hash.
    If the cache file exists, the result is loaded from the file. Otherwise, the
    function is called, and the result is saved to the cache file.

    If `cache_dir` is `None`, caching is disabled and the function is always executed.

    Args:
        cache_dir (PathLike or None): The directory where cache files will be stored, or
            `None` to disable caching.
        use_gzip (bool): Whether to use gzip compression for the cache files.
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
            file_extension = ".pkl.gz" if use_gzip else ".pkl"
            cache_file = get_sharded_file_path(cache_dir, hash_hex, file_extension, directory_depth)

            # ... check if cache file exists
            open_func = gzip.open if use_gzip else open
            if cache_file.exists():
                try:
                    # ... try to load the result from cache file
                    with open_func(cache_file, "rb") as f:
                        result = pickle.load(f)
                    return result

                except Exception as e:
                    # (Fallback to executing the function, with a warning)
                    logger.error(f"Error loading cache file {cache_file}: {e}")

            # If cache file doesn't exist, execute the function
            result = func(*args, **kwargs)

            # ... save the result to cache file, creating directories if necessary
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open_func(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


def get_sharded_file_path(
    base_dir: Path,
    file_hash: str,
    extension: str,
    depth: int,
    chars_per_dir: int = 2,
    include_subdirectory: bool = False,
) -> Path:
    """Construct a nested file path based on the directory depth.

    Args:
        base_dir (Path): The base directory where the files are stored.
        file_hash (str): The hash of the file content or identifier.
        extension (str): The file extension.
        depth (int): The directory nesting depth.
        chars_per_dir (int): The number of characters to use for each directory level.
        include_subdirectory (bool): If True, creates an additional directory with the full hash name.

    Returns:
        Path: The constructed path to the file.

    Example:
        >>> get_sharded_file_path("/path/to/cache", "abcdef123456", ".pkl", 2)
        Path("/path/to/cache/ab/cd/abcdef123456.pkl")
        >>> get_sharded_file_path("/path/to/cache", "abcdef123456", ".pkl", 3, chars_per_dir=1)
        Path("/path/to/cache/a/b/c/abcdef123456.pkl")
        >>> get_sharded_file_path("/path/to/cache", "abcdef123456", ".pkl", 2, include_subdirectory=True)
        Path("/path/to/cache/ab/cd/abcdef123456/abcdef123456.pkl")
    """
    nested_path = Path(base_dir)
    for i in range(depth):
        start_idx = chars_per_dir * i
        end_idx = chars_per_dir * (i + 1)
        nested_path /= Path(file_hash[start_idx:end_idx])

    if include_subdirectory:
        nested_path /= file_hash

    return (nested_path / file_hash).with_suffix(extension)


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


def read_parquet_with_metadata(filepath: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper around pd.read_parquet that preserves metadata.

    Args:
        filepath: Path to the parquet file.
        **kwargs: Additional arguments to pass to pd.read_parquet.

    Returns:
        pandas DataFrame with metadata in .attrs attribute
    """
    # Read the parquet file using pyarrow
    table = pq.read_table(filepath)

    # Extract metadata
    raw_metadata = table.schema.metadata

    # Convert bytes keys and values back to strings
    metadata_dict = {k.decode(): v.decode() for k, v in raw_metadata.items() if k not in (b"pandas", b"pyarrow_schema")}

    # Read the DataFrame using pandas
    df = pd.read_parquet(filepath, **kwargs)

    # Attach metadata to DataFrame's attrs
    df.attrs = metadata_dict

    return df


def parse_sharding_pattern(sharding_pattern: str) -> list[tuple[int, int]]:
    """Parse a sharding pattern string into directory levels.

    Args:
        sharding_pattern: String like "/1:2/0:2/" where each /start:end/ defines a directory level
            - start:end defines the character range to use for that directory level
            - Example: "/1:2/0:2/" means use chars 1-2 for first dir, then chars 0-2 for second dir

    Returns:
        List of (start, end) tuples for each directory level
    """
    # Find all patterns like /start:end/ using a non-consuming lookahead
    pattern = r"/(\d+):(\d+)(?=/)"
    matches = []
    for match in re.finditer(pattern, sharding_pattern):
        matches.append((int(match.group(1)), int(match.group(2))))

    if not matches:
        raise ValueError(f"Invalid sharding pattern format: {sharding_pattern}. Expected format like '/1:2/0:2/'")

    return matches


def apply_sharding_pattern(path: str, sharding_pattern: str | None = None) -> Path:
    """Apply a sharding pattern to construct a file path.

    Args:
        path: The base path or identifier (e.g., PDB ID)
        sharding_pattern: Pattern for organizing files in subdirectories
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)

    Returns:
        Path: The constructed file path with sharding applied
    """
    if sharding_pattern and sharding_pattern.startswith("/"):
        # General sharding pattern: /start:end/start:end/...
        try:
            shard_levels = parse_sharding_pattern(sharding_pattern)
        except ValueError as e:
            raise ValueError(f"Invalid sharding pattern: {e}") from e

        # Build the sharded path
        current_path = Path()

        for start, end in shard_levels:
            if end > len(path):
                raise ValueError(f"Sharding range {start}:{end} exceeds path length {len(path)} for path '{path}'")
            shard_dir = path[start:end]
            current_path = current_path / shard_dir

        final_path = current_path / path
    else:
        # Default behavior: no sharding
        final_path = Path(path)

    return final_path
