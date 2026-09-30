"""Base class for sharded molecular datasets (LMDB, ASE, etc.)."""

import logging
from abc import abstractmethod
from collections.abc import Callable
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from atomworks.ml.datasets.base import MolecularDataset
from atomworks.ml.datasets.metadata import (
    ArrowMetadataIndex,
    MetadataIndex,
    MetadataIndexProtocol,
    SequentialMetadataIndex,
)

logger = logging.getLogger(__name__)


class ShardedDataset(MolecularDataset):
    """Base class for sharded molecular datasets.

    Owns all shared logic: metadata setup, shard discovery, offset calculation,
    lazy shard loading, ``__getitem__`` skeleton, and context manager protocol.

    Subclasses implement four abstract methods to handle format-specific I/O:
    :meth:`_get_shard_count`, :meth:`_open_shard`, :meth:`_close_shard`,
    and :meth:`_fetch_entry`.
    """

    def __init__(
        self,
        *,
        shard_path: str,
        shard_extension: str,
        name: str,
        loader: Callable | None = None,
        data: str | PathLike | None = None,
        id_column: str = "example_id",
        idx_column: str = "lmdb_idx",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        transform: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        memory_map: bool = False,
        metadata: MetadataIndexProtocol | None = None,
    ):
        """Initialize ShardedDataset.

        Args:
            shard_path: Path to the shard directory or single shard file.
            shard_extension: File extension used to discover shards in a directory.
            name: Descriptive name for this dataset.
            loader: Optional callable to process ``(entry, global_idx, metadata_row)``
                into a data dict.
            data: Optional path to a parquet/CSV file with metadata for filtering.
            id_column: Column name for human-readable example IDs.
            idx_column: Column name containing integer shard indices.
            filters: Optional list of pandas query strings to filter metadata.
            columns_to_load: Optional list of columns to load from metadata file.
            transform: Transform pipeline to apply to loaded data.
            save_failed_examples_to_dir: Optional directory to save failed examples.
            memory_map: If ``True``, use :class:`ArrowMetadataIndex` for metadata.
            metadata: Optional pre-built metadata index.
        """
        super().__init__(
            name=name,
            loader=loader,
            transform=transform,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        self.shard_path = Path(shard_path)
        self._shard_extension = shard_extension

        # Discover shard files
        self._filepaths = self._discover_shards(self.shard_path)
        self._num_shards = len(self._filepaths)
        logger.info(f"Found {self._num_shards} shard(s) at {self.shard_path}")

        # Auto-detect shard sizes and build cumulative offset array
        shard_counts = [self._get_shard_count(p) for p in self._filepaths]
        self._shard_offsets = np.zeros(self._num_shards + 1, dtype=np.int64)
        self._shard_offsets[1:] = np.cumsum(shard_counts)
        self._total_entries = int(self._shard_offsets[-1])
        logger.info(f"Total entries across {self._num_shards} shard(s): {self._total_entries:,}")

        # Lazy shard state — only one shard open at a time
        self._current_shard_idx: int | None = None

        # Metadata index
        if metadata is not None:
            self._metadata = metadata
        elif data is not None:
            _index_cls = ArrowMetadataIndex if memory_map else MetadataIndex
            self._metadata = _index_cls(
                data=data,
                name=name,
                id_column=id_column,
                filters=filters,
                columns_to_load=columns_to_load,
            )
        else:
            self._metadata = SequentialMetadataIndex(
                n_entries=self._total_entries,
                idx_column=idx_column,
            )

        self._shard_indices = self._metadata.get_column_values(idx_column).astype(np.int64)
        logger.info(f"Dataset contains {len(self._metadata):,} examples")

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement these
    # ------------------------------------------------------------------

    def _discover_shards(self, path: Path) -> list[str]:
        """Discover shard files matching ``self._shard_extension``."""
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            filepaths = sorted(glob(str(path / f"*{self._shard_extension}")))
            if not filepaths:
                raise ValueError(f"No {self._shard_extension} files found in directory: {path}")
            return filepaths
        raise ValueError(f"Path does not exist: {path}")

    @abstractmethod
    def _get_shard_count(self, path: str) -> int:
        """Return the number of entries in a single shard file."""
        ...

    @abstractmethod
    def _open_shard(self, shard_idx: int) -> None:
        """Open the shard at ``shard_idx``, storing handles on ``self``."""
        ...

    @abstractmethod
    def _close_shard(self) -> None:
        """Close the currently open shard and reset handles on ``self``."""
        ...

    @abstractmethod
    def _fetch_entry(self, local_idx: int) -> Any:
        """Fetch the raw entry at ``local_idx`` within the currently open shard."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> MetadataIndexProtocol:
        """The metadata index."""
        return self._metadata

    @property
    def data(self) -> pd.DataFrame | pa.Table:
        """The underlying metadata table."""
        return self._metadata.data

    def _ensure_shard_loaded(self, shard_idx: int) -> None:
        """Load a shard if not already loaded, closing previous shard."""
        if self._current_shard_idx == shard_idx:
            return
        self._close_shard()
        self._open_shard(shard_idx)
        self._current_shard_idx = shard_idx

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._metadata)

    def __getitem__(self, idx: int) -> Any:
        """Load and transform an example by index."""
        dataset_len = len(self)
        if idx < 0:
            idx = dataset_len + idx
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Index {idx} out of range for dataset with {dataset_len} examples")

        global_idx = int(self._shard_indices[idx])
        example_id = self._metadata.get_example_id(idx)

        # Find which shard this index belongs to
        shard_idx = int(np.searchsorted(self._shard_offsets, global_idx, side="right") - 1)
        local_idx = global_idx - int(self._shard_offsets[shard_idx])

        if shard_idx >= self._num_shards or shard_idx < 0:
            raise IndexError(f"Global index {global_idx} exceeds dataset bounds ({self._total_entries})")

        self._ensure_shard_loaded(shard_idx)
        entry = self._fetch_entry(local_idx)

        # Fetch metadata row (shared logic)
        try:
            metadata_row = self._metadata.get_row(idx)
        except NotImplementedError:
            metadata_row = None

        data = self._apply_loader((entry, global_idx, metadata_row))

        if isinstance(data, dict):
            data["example_id"] = example_id

        return self._apply_transform(data, example_id=example_id, idx=idx)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self._metadata

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to dataset index(es)."""
        return self._metadata.id_to_idx(example_id)

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert dataset index(es) to example ID(s)."""
        return self._metadata.idx_to_id(idx)

    def close(self) -> None:
        """Explicitly close database connection."""
        self._close_shard()
        self._current_shard_idx = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        """Context manager exit with automatic cleanup."""
        self.close()
        return False

    def __del__(self):
        """Close database connections on deletion."""
        self.close()
