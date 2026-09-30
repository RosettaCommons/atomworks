"""Dataset for loading molecular structures from ASE LMDB databases."""

import logging
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

from atomworks.ml.datasets.metadata import MetadataIndexProtocol
from atomworks.ml.datasets.sharded_dataset import ShardedDataset

try:
    import ase.db
    from ase_db_backends.aselmdb import LMDBDatabase  # noqa: F401
except ImportError:
    warnings.warn(
        "ASE and ase-db-backends are required for AseDBDataset. " "Install with: uv pip install ase ase-db-backends",
        ImportWarning,
        stacklevel=2,
    )
    ase = None

logger = logging.getLogger(__name__)


class AseDBDataset(ShardedDataset):
    """Dataset for loading molecular structures from ASE LMDB databases.

    Uses memory-efficient lazy shard loading - only one shard is kept
    open at a time (~50MB), instead of all shards (~50MB * num_shards).
    """

    def __init__(
        self,
        *,
        lmdb_path: str,
        name: str,
        loader: Callable | None = None,
        data: str | PathLike | None = None,
        id_column: str = "example_id",
        lmdb_idx_column: str = "lmdb_idx",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        transform: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        memory_map: bool = False,
        metadata: MetadataIndexProtocol | None = None,
    ):
        """Initialize AseDBDataset.

        Args:
            lmdb_path: Path to the ASE LMDB database directory.
            name: Descriptive name for this dataset.
            loader: Optional callable to process ``(atoms_row, global_idx, metadata_row)``
                into a data dict.
            data: Optional path to a parquet/CSV file with metadata for filtering.
            id_column: Column name for human-readable example IDs.
            lmdb_idx_column: Column name containing integer LMDB indices.
            filters: Optional list of pandas query strings to filter metadata.
            columns_to_load: Optional list of columns to load from metadata file.
            transform: Transform pipeline to apply to loaded data.
            save_failed_examples_to_dir: Optional directory to save failed examples.
            memory_map: If ``True``, use :class:`ArrowMetadataIndex` for metadata.
            metadata: Optional pre-built metadata index.
        """
        # ASE-specific shard state
        self._current_db = None
        self._current_ids: list | None = None

        super().__init__(
            shard_path=lmdb_path,
            shard_extension=".aselmdb",
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
        """Return the number of entries in an ASE shard."""
        db = ase.db.connect(path, type="aselmdb", readonly=True, use_lock_file=False)
        try:
            return len(db.ids)
        finally:
            db.close()

    def _open_shard(self, shard_idx: int) -> None:
        """Open an ASE database shard."""
        path = self._filepaths[shard_idx]
        self._current_db = ase.db.connect(path, type="aselmdb", readonly=True, use_lock_file=False)
        self._current_ids = self._current_db.ids
        logger.debug(f"Loaded shard {shard_idx}: {Path(path).name}")

    def _close_shard(self) -> None:
        """Close the currently open ASE shard."""
        if self._current_db is not None:
            try:
                self._current_db.close()
            except Exception as e:
                logger.warning(f"Failed to close shard {self._current_shard_idx}: {e}")
            self._current_db = None
            self._current_ids = None

    def _fetch_entry(self, local_idx: int) -> Any:
        """Fetch an atoms_row from the current ASE shard."""
        actual_shard_len = len(self._current_ids)
        if local_idx >= actual_shard_len:
            raise IndexError(
                f"Local index {local_idx} exceeds shard bounds. "
                f"Shard {self._current_shard_idx} has {actual_shard_len} entries."
            )
        lmdb_row_id = self._current_ids[local_idx]
        return self._current_db.get(id=lmdb_row_id)
