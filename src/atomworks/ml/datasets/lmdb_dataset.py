"""Dataset for loading molecular structures from raw LMDB databases."""

import logging
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path

from atomworks.ml.datasets.loaders.cif import create_structure_loader
from atomworks.ml.datasets.metadata import MetadataIndexProtocol
from atomworks.ml.datasets.sharded_dataset import ShardedDataset
from atomworks.ml.utils.io import _decompress

try:
    import lmdb
except ImportError:
    warnings.warn(
        "lmdb library is required for LmdbDataset. Install it with: uv pip install lmdb",
        ImportWarning,
        stacklevel=2,
    )
    lmdb = None


def _default_key_encoder(i: int) -> bytes:
    """Encode an integer index as 8-byte little-endian bytes."""
    return i.to_bytes(8, "little")


logger = logging.getLogger(__name__)


class LmdbDataset(ShardedDataset):
    """Dataset for loading raw bytes from LMDB databases.

    Values are raw bytes (typically CIF, optionally zstd/gzip compressed).
    Supports the same metadata filtering as :class:`PandasDataset`, etc.
    """

    def __init__(
        self,
        *,
        lmdb_path: str,
        name: str,
        shard_extension: str = ".ciflmdb",
        key_encoder: Callable[[int], bytes] | None = None,
        loader: Callable | None = None,
        data: str | PathLike | None = None,
        id_column: str = "example_id",
        lmdb_idx_column: str = "lmdb_idx",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        parser_args: dict | None = None,
        transform: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        memory_map: bool = False,
        metadata: MetadataIndexProtocol | None = None,
    ):
        """Initialize LmdbDataset.

        Args:
            lmdb_path: Path to the LMDB database directory or single file.
            name: Descriptive name for this dataset.
            shard_extension: File extension used to discover shards in a directory.
                Defaults to ``".ciflmdb"``.
            key_encoder: Callable that converts an integer index to the LMDB key bytes.
                Defaults to 8-byte little-endian encoding (OMOL format).
            loader: Optional callable to process ``(raw_bytes, global_idx, metadata_row)``
                into a data dict.  Defaults to
                :py:func:`~atomworks.ml.datasets.loaders.cif.create_structure_loader`
                when ``shard_extension`` is ``".ciflmdb"``.  Must be provided explicitly
                for other extensions.
            data: Optional path to a parquet/CSV file with metadata for filtering.
            id_column: Column name for human-readable example IDs.
            lmdb_idx_column: Column name containing integer LMDB indices.
            filters: Optional list of pandas query strings to filter metadata.
            columns_to_load: Optional list of columns to load from metadata file.
            parser_args: Optional parser arguments passed to the default CIF bytes loader.
            transform: Transform pipeline to apply to loaded data.
            save_failed_examples_to_dir: Optional directory to save failed examples.
            memory_map: If ``True``, use :class:`ArrowMetadataIndex` for metadata.
            metadata: Optional pre-built metadata index.
        """
        # LMDB-specific shard state (set early so __del__ cleanup works)
        self._current_env = None
        self._current_txn = None

        if loader is None:
            if shard_extension == ".ciflmdb":
                loader = create_structure_loader(storage="bytes", parser_args=parser_args)
            else:
                raise ValueError(
                    f"No default loader for shard extension '{shard_extension}'. Please provide an explicit loader."
                )

        self._key_encoder = key_encoder or _default_key_encoder

        super().__init__(
            shard_path=lmdb_path,
            shard_extension=shard_extension,
            name=name,
            loader=loader,
            data=data,
            id_column=id_column,
            idx_column=lmdb_idx_column,
            filters=filters,
            columns_to_load=columns_to_load,
            transform=transform,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
            memory_map=memory_map,
            metadata=metadata,
        )

    def _get_shard_count(self, path: str) -> int:
        """Return the number of entries in an LMDB shard."""
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                return txn.stat()["entries"]
        finally:
            env.close()

    def _open_shard(self, shard_idx: int) -> None:
        """Open an LMDB shard."""
        path = self._filepaths[shard_idx]
        self._current_env = lmdb.open(path, subdir=False, readonly=True, lock=False)
        self._current_txn = self._current_env.begin()
        logger.debug(f"Loaded shard {shard_idx}: {Path(path).name}")

    def _close_shard(self) -> None:
        """Close the currently open LMDB shard."""
        if self._current_txn is not None:
            try:
                self._current_txn.abort()
            except Exception as e:
                logger.warning(f"Failed to abort txn for shard {self._current_shard_idx}: {e}")
            self._current_txn = None
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception as e:
                logger.warning(f"Failed to close shard {self._current_shard_idx}: {e}")
            self._current_env = None

    def _fetch_entry(self, local_idx: int) -> bytes:
        """Fetch and decompress an entry from the current LMDB shard."""
        key = self._key_encoder(local_idx)
        raw = self._current_txn.get(key)
        if raw is None:
            raise KeyError(f"Key {local_idx} not found in shard {self._current_shard_idx}")
        return _decompress(raw)
