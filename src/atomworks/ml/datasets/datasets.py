"""AtomWorks Dataset classes and common APIs.

At a high level, to train models with AtomWorks, we need a Dataset class that:
    (1) Takes as input an item index and returns the corresponding example information; typically includes:
        a. Path to a structural file saved on disk (`/path/to/dataset/my_dataset_0.cif`)
        b. Additional item-specific metadata (e.g., class labels)
    (2) Pre-loads structural information from the returned example into an `AtomArray` and assembles inputs for the Transform pipeline
    (3) Feed the input dictionary through a Transform pipeline and return the result

Due to the heterogeneity of biomolecular data, in many cases, we may also want:
    (4) In the event of a failure during the Transform pipeline, fall back to a different example

For bespoke use cases, users may choose to write a custom Dataset that accomplish these steps; downstream code makes no assumptions.

To accelerate development, we also provide an off-the-shelf, composable approach following common patterns:
    - :class:`MolecularDataset`: Base class that handles pre-loading structural data and executing the Transform pipeline with error handling and debugging utilities
    - :class:`PandasDataset`: A subclass of MolecularDataset for tabular data stored as pandas DataFrames
    - :class:`FileDataset`: A subclass of MolecularDataset where each file is one example
"""

import copy
import os
import socket
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from atomworks.ml.datasets import logger
from atomworks.ml.preprocessing.constants import NA_VALUES
from atomworks.ml.transforms.base import TransformedDict
from atomworks.ml.utils.debug import save_failed_example_to_disk
from atomworks.ml.utils.io import read_parquet_with_metadata, scan_directory
from atomworks.ml.utils.rng import capture_rng_states


class ExampleIDMixin(ABC):
    """Mixin providing example ID functionality to a Dataset.

    Provides methods for converting between example IDs and indices, and checking
    if an example ID exists in the dataset.
    """

    @abstractmethod
    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID.

        Args:
            example_id: The ID to check for.

        Returns:
            True if the ID exists in the dataset.
        """
        pass

    @abstractmethod
    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to index(es).

        Args:
            example_id: Single ID or list of IDs to convert.

        Returns:
            Corresponding index or list of indices.
        """
        pass

    @abstractmethod
    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert index(es) to example ID(s).

        Args:
            idx: Single index or list of indices to convert.

        Returns:
            Corresponding ID or list of IDs.
        """
        pass


class MolecularDataset(Dataset):
    """Base class for AtomWorks molecular datasets.

    Handles Transform pipelines and loader functionality for molecular data.
    Subclasses implement :meth:`__getitem__` with their own data access patterns.
    """

    def __init__(
        self,
        *,
        name: str,
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        """Initialize MolecularDataset.

        Args:
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            transform: Transform function or pipeline to apply to loaded data.
                Should accept the output of the loader and return featurized data.
            loader: Optional function to process raw dataset output into Transform-ready
                format. For example, parsing structural files or gathering columns
                into structured data.
            save_failed_examples_to_dir: Optional directory path where failed examples
                will be saved for debugging. Includes RNG state and error information.
        """
        self.loader = loader

        self.transform = transform
        self.name = name
        self.save_failed_examples_to_dir = Path(save_failed_examples_to_dir) if save_failed_examples_to_dir else None

    def _apply_loader(self, raw_data: Any) -> Any:
        """Apply the loader function to raw data with timing and debugging.

        Args:
            raw_data: The raw data to process.

        Returns:
            Processed data ready for transforms.
        """
        if self.loader is None:
            return raw_data

        # Apply loader function with timing
        _start_load_time = time.time()
        data = self.loader(raw_data)
        _stop_load_time = time.time()

        # Add timing information if data supports it (preserving TransformDataset behavior)
        if isinstance(data, dict):
            data = TransformedDict(data)
            data.__transform_history__.append(
                {
                    "name": "apply loader",
                    "instance": hex(id(self.loader)),
                    "start_time": _start_load_time,
                    "end_time": _stop_load_time,
                    "processing_time": _stop_load_time - _start_load_time,
                }
            )

        return data

    def _apply_transform(self, data: Any, example_id: str | None = None, idx: int | None = None) -> Any:
        """Apply the Transform pipeline with error handling and debugging support.

        Args:
            data: The loaded data ready for transforms.
            example_id: Optional example ID for debugging purposes. If not provided,
                will generate one using dataset name and index.
            idx: Optional dataset index for error reporting.

        Returns:
            Transformed data.

        Raises:
            KeyboardInterrupt: Always re-raised if encountered.
            Exception: Any exception from the transform pipeline is re-raised.
        """
        if self.transform is None:
            return data

        # Generate default example_id from idx and dataset name if not provided
        if example_id is None and idx is not None:
            example_id = f"{self.name}_{idx}"

        # Get process id and hostname for debugging
        if example_id:
            logger.debug(f"({socket.gethostname()}:{os.getpid()}) Processing example: {example_id}")

        try:
            # Capture RNG state for reproducibility before applying Transforms
            rng_state_dict = capture_rng_states(include_cuda=False)
            data = self.transform(data)
            return data

        except KeyboardInterrupt:
            # Always re-raise keyboard interrupts
            raise
        except Exception as e:
            logger.error(e)

            if self.save_failed_examples_to_dir and example_id:
                save_failed_example_to_disk(
                    example_id=example_id,
                    error_msg=e,
                    rng_state_dict=rng_state_dict,
                    data={},  # We do not save the data by default, since it may be large
                    fail_dir=self.save_failed_examples_to_dir,
                )

            # Re-raise the original exception
            raise

    def __getitem__(self, index: int) -> Any:
        """Return a fully-featurized data example given an index.

        Subclasses should implement this method to:
            1. Query the underlying data source for raw data at the given index
            2. Optionally pre-process data to prepare for the Transform pipeline
            3. Feed the input dictionary through a Transform pipeline

        Typical output for an activity prediction network:
            Step 1: ``{"path": "/path/to/dataset", "class_label": 5}``
            Step 2: ``{"atom_array": AtomArray, "extra_info": {"class_label": 5}}``
            Step 3: ``{"features": torch.Tensor, "class_label": torch.Tensor}``

        Args:
            index: The index of the example to retrieve.

        Returns:
            Fully-featurized data example.
        """
        raise

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Returns:
            The dataset length.
        """
        pass


class FileDataset(MolecularDataset, ExampleIDMixin):
    """Dataset that loads molecular data from individual files.

    Each file represents one example in the dataset. If creating a dataset from a
    directory, use the :meth:`from_directory` class method instead of the default
    constructor.
    """

    def __init__(
        self,
        *,
        file_paths: list[str | PathLike],
        name: str,
        filter_fn: Callable[[PathLike], bool] | None = None,
        **kwargs: Any,
    ):
        """Initialize FileDataset.

        Args:
            file_paths: List of file paths for the dataset. Each file represents
                one example.
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            filter_fn: Optional function to filter file paths. Should return True
                for files to include.
            **kwargs: Additional arguments passed to :class:`MolecularDataset`.

        Examples:
            Create from explicit file list:
                >>> files = ["/path/to/file1.cif", "/path/to/file2.cif"]
                >>> dataset = FileDataset(file_paths=files, name="my_dataset")
        """
        super().__init__(name=name, **kwargs)

        self.filter_fn = filter_fn or (lambda x: True)

        # Convert to Path objects and filter
        file_paths = [Path(path) for path in file_paths if self.filter_fn(path)]
        if not file_paths:
            raise ValueError("No files found after applying filters")
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("File paths must be unique")

        # Sort for consistent id<>idx mapping
        file_paths.sort()
        self.file_paths = file_paths

        # Create ID mapping
        self.id_to_idx_map = {self._get_example_id(i): i for i, _ in enumerate(self.file_paths)}

        # Verify that all example IDs are unique
        if len(self.id_to_idx_map) != len(self.file_paths):
            raise ValueError("Example IDs must be unique. Found duplicate example IDs.")

    @classmethod
    def from_directory(
        cls,
        *,
        directory: PathLike,
        name: str,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> "FileDataset":
        """Create a FileDataset by scanning a directory for files.

        Args:
            directory: Path to directory to scan for files.
            name: Descriptive name for this dataset.
            max_depth: Maximum depth to scan for files in subdirectories.
            **kwargs: Additional arguments passed to :class:`FileDataset`.

        Returns:
            FileDataset instance with files discovered from the directory.

        Example:
            Create from directory:
                >>> dataset = FileDataset.from_directory(directory="/path/to/files", name="my_dataset", max_depth=2)
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        if not dir_path.is_dir():
            raise ValueError(f"Path {directory} is not a directory.")

        file_paths = scan_directory(dir_path=dir_path, max_depth=max_depth)
        return cls(file_paths=file_paths, name=name, **kwargs)

    @classmethod
    def from_file_list(
        cls,
        *,
        file_paths: list[str | PathLike],
        name: str,
        **kwargs: Any,
    ) -> "FileDataset":
        """Create a FileDataset from an explicit list of file paths.

        This is an alias for the main constructor for clarity and consistency
        with :meth:`from_directory`.

        Args:
            file_paths: List of file paths for the dataset. Each file represents one example.
            name: Descriptive name for this dataset.
            **kwargs: Additional arguments passed to :class:`FileDataset`.

        Returns:
            FileDataset instance with the provided file paths.
        """
        return cls(file_paths=file_paths, name=name, **kwargs)

    def __len__(self) -> int:
        """Return the number of files in the dataset."""
        return len(self.file_paths)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self.id_to_idx_map

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to index(es)."""
        if isinstance(example_id, list):
            return [self.id_to_idx_map[id] for id in example_id]
        return self.id_to_idx_map[example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert index(es) to example ID(s)."""
        if isinstance(idx, list):
            return [self._get_example_id(i) for i in idx]
        return self._get_example_id(idx)

    def __getitem__(self, idx: int) -> Any:
        """Load and transform an example by file index.

        Args:
            idx: The index of the file to load.

        Returns:
            Transformed data from the file.
        """
        file_path = str(self.file_paths[idx])
        example_id = self._get_example_id(idx)
        data = self._apply_loader(file_path)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def _get_example_id(self, idx: int) -> str:
        """Get example ID from index - returns filename stem without extensions.

        Args:
            idx: The index of the file.

        Returns:
            Filename stem without any extensions.
        """
        file_path = self.file_paths[idx]
        filename = Path(file_path).stem
        # If filename has multiple extensions (e.g., .cif.gz), remove them all
        while "." in filename:
            filename = Path(filename).stem
        return filename


class PandasDataset(MolecularDataset, ExampleIDMixin):
    """Dataset for tabular data stored as pandas DataFrames.

    Inherits all functionality from :class:`MolecularDataset` with additional
    DataFrame-specific features for filtering and ID-based access.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        name: str,
        id_column: str | None = "example_id",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        # MolecularDataset parameters
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
        load_kwargs: dict | tuple | None = None,
    ):
        """Initialize PandasDataset.

        Args:
            data: Either a pandas DataFrame or path to a CSV/Parquet file containing
                the tabular data. Each row represents one example.
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            id_column: Optional column name to use as the DataFrame index for
                example ID lookups. If provided, this column will be set as the index.
            filters: Optional list of pandas query strings to filter the data.
                Applied in order during initialization.
            columns_to_load: Optional list of column names to load when reading
                from a file. If None, all columns are loaded. Can dramatically reduce
                memory usage and load time if loading from a columnar format like Parquet.
            transform: Transform pipeline to apply to loaded data.
            loader: Optional function to process raw DataFrame rows into Transform-ready format.
            save_failed_examples_to_dir: Optional directory path where failed examples
                will be saved for debugging. Includes RNG state and error information.
            load_kwargs: Additional keyword arguments passed to pandas' read functions
                (read_csv, read_parquet) when loading from file.

        Examples:
            Load from DataFrame:
                >>> df = pd.DataFrame({"path": [...], "label": [...]})
                >>> dataset = PandasDataset(data=df, name="my_dataset")

            Load from file with filtering:
                >>> dataset = PandasDataset(data="data.csv", name="filtered_dataset", filters=["label > 0", "path.str.contains('.pdb')"])
        """
        super().__init__(
            name=name,
            transform=transform,
            loader=loader,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        # Load data from path if needed
        if isinstance(data, PathLike | str):
            data = self._load_from_path(data, columns_to_load, **(load_kwargs or {}))
        self.data = data

        # Apply filters
        self.filters = filters
        self._already_filtered = False
        if filters:
            self._apply_filters(filters)
        self._already_filtered = True

        # Set index column if specified
        if id_column is not None:
            assert id_column in self.data.columns, f"Column {id_column} not found in dataset."
            self.data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)

    def _load_from_path(
        self, path: PathLike | str, columns_to_load: list[str] | None = None, **load_kwargs: Any
    ) -> pd.DataFrame:
        """Load data from file path.

        Args:
            path: Path to the file to load.
            columns_to_load: Optional list of column names to load.
            **load_kwargs: Additional arguments for pandas read functions.

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If file type is unsupported.
        """
        path = Path(path)
        if path.suffix == ".csv":
            data = pd.read_csv(path, usecols=columns_to_load, keep_default_na=False, na_values=NA_VALUES, **load_kwargs)
        elif path.suffix == ".parquet":
            data = read_parquet_with_metadata(path, columns=columns_to_load, **load_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return data

    def __getitem__(self, idx: int) -> Any:
        """Get an example by index, applying specified loader and Transforms.

        Args:
            idx: The index of the example to retrieve.

        Returns:
            Transformed data from the row.
        """
        raw_data = self.data.iloc[idx]
        example_id = self._get_example_id(idx)
        data = self._apply_loader(raw_data)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.data)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self.data.index

    def _id_to_index_single(self, example_id: str) -> int:
        """Convert single example ID to index."""
        return self.data.index.get_loc(example_id)

    def _id_to_index_multiple(self, example_ids: list[str]) -> list[int]:
        """Convert multiple example IDs to indices."""
        idxs = np.arange(len(self.data))
        return [idxs[self.data.index.get_loc(example_id)] for example_id in example_ids]

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID to the corresponding local index."""
        if np.isscalar(example_id):
            return self._id_to_index_single(example_id)
        elif isinstance(example_id, list | np.ndarray | tuple):
            return self._id_to_index_multiple(example_id)
        else:
            raise ValueError(f"Invalid type for example_id: {type(example_id)}")

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert a local index to the corresponding example ID."""
        _return_single = False
        if np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.shape == ()):
            _return_single = True
            idx = idx.item() if isinstance(idx, np.ndarray) else idx
            idx = slice(idx, idx + 1)
        ids = self.data.iloc[idx].index.values
        return ids[0] if _return_single else ids

    def _apply_filters(self, filters: list[str]) -> pd.DataFrame:
        """Apply filters to the data based on the provided list of query strings.

        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters: List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Example:
            >>> queries = [
            >>>     "deposition_date < '2020-01-01'",
            >>>     "resolution < 2.5 and ~method.str.contains('NMR')",
            >>>     "cluster.notnull()",
            >>>     "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            >>> ]
            >>> dataset = PandasDataset(data="data.csv", name="filtered_dataset", filters=queries)
        """
        assert not self._already_filtered, "Filters cannot be applied after initialization."

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            self._apply_query(query)

    def _apply_query(self, query: str) -> None:
        """Apply a single query to the data.

        Args:
            query: The pandas query string to apply.
        """
        # Filter using query and validate impact
        original_num_rows = len(self.data)
        self.data = self.data.query(query)
        filtered_num_rows = len(self.data)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        """Validate the impact of the filter.

        Args:
            query: The query that was applied.
            original_num_rows: Number of rows before filtering.
            filtered_num_rows: Number of rows after filtering.

        Raises:
            ValueError: If the query removes all rows.
        """
        rows_removed = original_num_rows - filtered_num_rows
        percent_removed = (rows_removed / original_num_rows) * 100
        percent_remaining = (filtered_num_rows / original_num_rows) * 100

        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' on dataset {self.name} did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' on dataset {self.name} removed all rows.")
        else:
            logger.info(
                f"\n+-------------------------------------------+\n"
                f"Query '{query}' on dataset {self.name}:\n"
                f"  - Started with: {original_num_rows:,} rows\n"
                f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
                f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
                f"+-------------------------------------------+\n"
            )

    def _get_example_id(self, idx: int) -> str:
        """Get example ID from index - returns the index value from the DataFrame.

        Args:
            idx: The index of the row.

        Returns:
            The index value as a string.
        """
        return str(self.data.iloc[idx].name)  # .name gets the index value


class ConcatDatasetWithID(ConcatDataset):
    """Equivalent to :class:`torch.utils.data.ConcatDataset` but allows accessing examples by ID.

    Provides ID-based access across multiple datasets that implement :class:`ExampleIDMixin`.
    """

    # TODO: Do I need all of these _raise_if etc. etc. here? Can I just check that the wrapped datasets inherit somehow from ExampleIDMixin?

    datasets: list[ExampleIDMixin]

    def __init__(self, datasets: list[ExampleIDMixin]):
        """Initialize ConcatDatasetWithID.

        Args:
            datasets: List of datasets that implement ExampleIDMixin.
        """
        super().__init__(datasets)

        # Log the length of each dataset
        for i, dataset in enumerate(datasets):
            logger.info(f"Dataset {i} ({type(dataset)}): {len(dataset):,} examples")

    @cached_property
    def _can_convert_ids_and_idx(self) -> bool:
        """Check if all sub-datasets can convert between IDs and indices."""
        has_id_to_idx = all(hasattr(sub_dataset, "id_to_idx") for sub_dataset in self.datasets)
        has_idx_to_id = all(hasattr(sub_dataset, "idx_to_id") for sub_dataset in self.datasets)
        return has_id_to_idx and has_idx_to_id and self._can_check_contains

    @cached_property
    def _can_check_contains(self) -> bool:
        """Check if all sub-datasets support contains operations."""
        return all(hasattr(sub_dataset, "__contains__") for sub_dataset in self.datasets)

    def _raise_if_cannot_check_contains(self) -> None:
        """Raise error if dataset cannot check contains."""
        if not self._can_check_contains:
            raise ValueError("This dataset cannot check if it contains an example ID.")

    def _raise_if_cannot_convert_ids_and_idx(self) -> None:
        """Raise error if dataset cannot convert IDs and indices."""
        if not self._can_convert_ids_and_idx:
            raise ValueError("This dataset cannot convert example IDs to indices.")

    def _raise_if_idx_out_of_bounds(self, idx: int) -> None:
        """Raise error if index is out of bounds.

        Args:
            idx: The index to check.
        """
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of bounds for dataset of length {len(self)}.")

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID.

        Args:
            example_id: The ID to check for.

        Returns:
            True if the ID exists in any sub-dataset.
        """
        self._raise_if_cannot_check_contains()
        return any(example_id in sub_dataset for sub_dataset in self.datasets)

    def id_to_idx(self, example_id: str) -> int:
        """Retrieves the index corresponding to the example ID.

        Args:
            example_id: The ID to convert.

        Returns:
            The corresponding index.

        Raises:
            ValueError: If the example ID is not found.

        Warning:
            Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        # TODO: Generalize to list[str]
        self._raise_if_cannot_convert_ids_and_idx()
        offset = 0
        for sub_dataset in self.datasets:
            if example_id in sub_dataset:
                return offset + sub_dataset.id_to_idx(example_id)
            offset += len(sub_dataset)
        raise ValueError(f"Example ID {example_id} not found in any sub-dataset.")

    def idx_to_id(self, idx: int) -> str:
        """Retrieves the example ID corresponding to the index.

        Args:
            idx: The index to convert.

        Returns:
            The corresponding example ID.

        Raises:
            ValueError: If the index is out of bounds.
        """
        # TODO: Generalize to list[int]
        self._raise_if_cannot_convert_ids_and_idx()
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset.idx_to_id(idx)
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_idx(self, idx: int) -> Dataset:
        """Retrieves the dataset containing the index.

        Args:
            idx: The index to find.

        Returns:
            The sub-dataset containing the index.

        Raises:
            ValueError: If the index is out of bounds.
        """
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_id(self, example_id: str) -> Dataset:
        """Retrieves the dataset containing the example ID.

        Args:
            example_id: The ID to find.

        Returns:
            The sub-dataset containing the ID.

        Warning:
            Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        idx = self.id_to_idx(example_id)
        return self.get_dataset_by_idx(idx)


def get_row_and_index_by_example_id(dataset: ExampleIDMixin, example_id: str) -> dict:
    """Retrieve a row and its index from a nested dataset structure by its example ID.

    Args:
        dataset: The dataset or concatenated dataset to search.
            Must have the `id_to_idx` method.
        example_id: The example ID to search for.

    Returns:
        Dictionary containing the row and global index corresponding to the example ID.
    """
    assert hasattr(dataset, "id_to_idx"), "Dataset must have the `id_to_idx` method."
    idx = dataset.id_to_idx(example_id)

    _local_idx = copy.deepcopy(idx)
    while isinstance(dataset, ConcatDatasetWithID):
        dataset = dataset.get_dataset_by_idx(_local_idx)
        _local_idx = dataset.id_to_idx(example_id)

    row = dataset.data.loc[example_id]
    return {"row": row, "index": idx}


class FallbackDatasetWrapper(Dataset):
    """A wrapper around a dataset that allows for a fallback dataset to be used when an error occurs.

    Meant to be used with a FallbackSamplerWrapper.
    """

    def __init__(self, dataset: Dataset, fallback_dataset: Dataset):
        """Initialize FallbackDatasetWrapper.

        Args:
            dataset: The primary dataset to retrieve data from.
            fallback_dataset: The fallback dataset to use when an error occurs. This
                may be the same as the primary dataset, or a different one.
        """
        self.dataset = dataset
        self.fallback_dataset = fallback_dataset

    def __getitem__(self, idxs: tuple[int, ...]) -> Any:
        """Attempt to retrieve an item from the primary dataset, falling back to additional indices if errors occur.

        If all attempts fail, raises a RuntimeError containing all encountered exceptions.

        Args:
            idxs: Tuple of indices, where the first is for the primary dataset and the rest are for fallbacks.

        Returns:
            The retrieved item from the first successful dataset.

        Raises:
            KeyboardInterrupt: If interrupted.
            StopIteration: If iteration should stop.
            RuntimeError: If all attempts fail, with a list of all exceptions encountered.
        """
        error_list = []
        example_id_list = []

        for i, idx in enumerate(idxs):
            dataset = self.dataset if i == 0 else self.fallback_dataset
            dataset_name = "Primary dataset" if i == 0 else f"Fallback {i}/{len(idxs)-1}"

            try:
                return dataset[idx]
            except (KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                error_list.append(e)

                # Log the error
                example_id = f" ({dataset.idx_to_id(idx)})" if hasattr(dataset, "idx_to_id") else ""
                example_id_list.append(example_id)
                logger.error(f"({dataset_name}): Error ({e}) at index {idx}.{example_id}")

                # Log fallback attempt if not the last one
                if i < len(idxs) - 1:
                    logger.warning(f"({dataset_name}): Trying fallback index {idxs[i+1]}.{example_id}")

        # All attempts failed
        logger.error(
            f"(Exceeded all {len(idxs)-1} fallbacks. Training will crash now. Errors: {error_list} for examples: {example_id_list})"
        )
        raise RuntimeError(f"All attempts failed for indices {idxs}. See error_list for details.") from ExceptionGroup(
            "All fallback attempts failed", error_list
        )

    def __len__(self):
        """Return the length of the primary dataset."""
        return len(self.dataset)


# Backwards Compatibility
# TODO: Deprecate
def StructuralDatasetWrapper(  # noqa: N802
    dataset_parser: Callable,
    transform: Callable | None = None,
    dataset: PandasDataset | None = None,
    cif_parser_args: dict | None = None,
    save_failed_examples_to_dir: str | Path | None = None,
    **kwargs,
) -> PandasDataset:
    """Backwards-compatible wrapper for the deprecated StructuralDatasetWrapper.

    This function is deprecated and will be removed in a future version.
    Use :class:`PandasDataset` with the appropriate loader function instead.

    Args:
        dataset_parser: The dataset parser to use (e.g., PNUnitsDFParser, InterfacesDFParser).
        transform: Transform pipeline to apply to loaded data.
        dataset: The underlying PandasDataset containing the tabular data.
        cif_parser_args: Arguments to pass to the CIF parser.
        save_failed_examples_to_dir: Directory to save failed examples for debugging.
        **kwargs: Additional arguments passed to PandasDataset.

    Returns:
        PandasDataset instance configured with the deprecated parameters.

    Raises:
        ValueError: If dataset parameter is required but not provided.
    """
    from atomworks.ml.datasets.parsers import load_example_from_metadata_row

    warnings.warn(
        "StructuralDatasetWrapper is deprecated. Use PandasDataset with a loader function instead. "
        "See atomworks.ml.datasets.loaders for functional alternatives to dataset parsers.",
        DeprecationWarning,
        stacklevel=2,
    )

    if dataset is None:
        raise ValueError("dataset parameter is required for StructuralDatasetWrapper")

    # Create loader from deprecated parameters
    def loader(row: pd.Series) -> dict[str, Any]:
        return load_example_from_metadata_row(row, dataset_parser, cif_parser_args=cif_parser_args or {})

    # Create a new PandasDataset with the loader
    return PandasDataset(
        data=dataset.data,
        name=dataset.name if hasattr(dataset, "name") else "structural_dataset",
        transform=transform,
        loader=loader,
        save_failed_examples_to_dir=save_failed_examples_to_dir,
        **kwargs,
    )
