"""Pandas DataFrame-based dataset implementation."""

import logging
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from .base import MolecularDataset
from .metadata import ArrowMetadataIndex, MetadataIndex, MetadataIndexProtocol

logger = logging.getLogger("datasets")


class PandasDataset(MolecularDataset):
    """Dataset for tabular data stored as pandas DataFrames.

    Delegates all metadata, filtering, and ID-mapping logic to
    :py:class:`~atomworks.ml.datasets.metadata.MetadataIndex`.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        name: str,
        id_column: str = "example_id",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        # MolecularDataset parameters
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        load_kwargs: dict | tuple | None = None,
        # Metadata options
        memory_map: bool = False,
        metadata: MetadataIndexProtocol | None = None,
    ):
        """Initialize PandasDataset.

        Args:
            data: Either a pandas DataFrame or path to a CSV/Parquet file containing
                the tabular data. Each row represents one example.
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            id_column: Column name to use as the DataFrame index for
                example ID lookups.
            filters: Independent row-wise pandas predicates evaluated on the input data.
                Masks are combined in order for waterfall logging; rows are selected once.
            columns_to_load: Optional list of column names to load when reading
                from a file. If None, all columns are loaded. Can dramatically reduce
                memory usage and load time if loading from a columnar format like Parquet.
            transform: Transform pipeline to apply to loaded data.
            loader: Optional function to process raw DataFrame rows into Transform-ready format.
            save_failed_examples_to_dir: Optional directory path where failed examples
                will be saved for debugging. Includes RNG state and error information.
            load_kwargs: Additional keyword arguments passed to pandas' read functions
                (read_csv, read_parquet) when loading from file.
            memory_map: If ``True``, use :class:`ArrowMetadataIndex` for memory-mapped
                metadata storage instead of in-memory pandas. Reduces heap usage for
                large datasets.
            metadata: Optional pre-built metadata index. If provided, used directly
                instead of constructing a new index. Takes precedence over ``memory_map``.

        Examples:
            Load from DataFrame:
                >>> df = pd.DataFrame({"path": [...], "label": [...]})
                >>> dataset = PandasDataset(data=df, name="my_dataset")

            Load from file with memory-mapped metadata:
                >>> dataset = PandasDataset(data="data.parquet", name="big_dataset", memory_map=True)
        """
        super().__init__(
            name=name,
            transform=transform,
            loader=loader,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        if metadata is not None:
            self._metadata = metadata
        else:
            # Normalize load_kwargs to dict
            _load_kwargs = dict(load_kwargs) if isinstance(load_kwargs, tuple) else load_kwargs

            _index_cls = ArrowMetadataIndex if memory_map else MetadataIndex
            self._metadata = _index_cls(
                data=data,
                name=name,
                id_column=id_column,
                filters=filters,
                columns_to_load=columns_to_load,
                load_kwargs=_load_kwargs,
            )

    @property
    def metadata(self) -> MetadataIndexProtocol:
        """The metadata index."""
        return self._metadata

    @property
    def data(self) -> pd.DataFrame | pa.Table:
        """The underlying metadata table.

        Returns a :class:`pd.DataFrame` for :class:`MetadataIndex` or a
        :class:`pa.Table` for :class:`ArrowMetadataIndex`.

        Raises:
            AttributeError: If the metadata backend has no ``data`` attribute
                (e.g. :class:`SequentialMetadataIndex`).
        """
        return self._metadata.data

    def __getitem__(self, idx: int) -> Any:
        """Get an example by index, applying specified loader and Transforms.

        Args:
            idx: The index of the example to retrieve.

        Returns:
            Transformed data from the row.
        """
        raw_data = self.metadata.get_row(idx)
        example_id = self.metadata.get_example_id(idx)
        data = self._apply_loader(raw_data)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.metadata)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self.metadata

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID to the corresponding local index."""
        return self.metadata.id_to_idx(example_id)

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert a local index to the corresponding example ID."""
        return self.metadata.idx_to_id(idx)
