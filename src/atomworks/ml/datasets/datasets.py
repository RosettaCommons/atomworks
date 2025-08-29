import copy
import os
import socket
import time
from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from atomworks.common import default, exists
from atomworks.ml.datasets import logger
from atomworks.ml.datasets.parsers import MetadataRowParser, load_example_from_metadata_row
from atomworks.ml.preprocessing.constants import NA_VALUES
from atomworks.ml.transforms.base import Compose, Transform, TransformedDict
from atomworks.ml.utils.debug import save_failed_example_to_disk
from atomworks.ml.utils.io import read_parquet_with_metadata
from atomworks.ml.utils.rng import capture_rng_states

_USER = default(os.getenv("USER"), "")


class BaseDataset(Dataset):
    """
    Abstract base class for datasets. All dataset types (e.g., Pandas, Polars) should inherit from this class
    and implement its methods.

    In addition to the standard PyTorch Dataset methods (`__getitem__`, `__len__`), this class requires
    implementations for converting between example IDs and indices, which is necessary for our nested dataset structure.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        pass

    @abstractmethod
    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID or list of example IDs to the corresponding index or indices."""
        pass

    @abstractmethod
    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert an index or list of indices to the corresponding example ID or IDs."""
        pass


class FileDataset(BaseDataset):
    def __init__(
        self,
        source: PathLike | list[str | PathLike],
        filter_fn: Callable[[PathLike], bool] | None = None,
        max_depth: int = 3,
    ):
        """Initialize a FileDataset that loads files from a directory or uses a pre-provided list.

        Args:
            source: Either a directory path to scan for files, or a pre-built list of file paths
            filter_fn: Optional function that takes a file path and returns True if the file should be included
            max_depth: Maximum directory depth to scan (only used when source is a directory path)
        """
        if isinstance(source, str | Path):
            # Directory scanning mode
            self.dir_path = Path(source)
            assert self.dir_path.is_dir(), f"Directory {source} does not exist."

            # Default filter accepts all files
            self.filter_fn = filter_fn if filter_fn is not None else lambda x: True

            # Scan directory for any files below
            file_paths = self._scan_directory(max_depth=max_depth)

        elif isinstance(source, list):
            # Pre-provided file list mode
            self.dir_path = None

            # Convert to strings and apply filter if provided
            file_paths = [str(path) for path in source]
            if filter_fn is not None:
                file_paths = [path for path in file_paths if filter_fn(path)]
            self.filter_fn = filter_fn

        else:
            raise ValueError("source must be either a directory path (str/Path) or a list of file paths")

        # Sort paths alphabetically for id<>idx consistency
        file_paths.sort()

        self.file_paths = file_paths
        self.path_to_idx = {path: i for i, path in enumerate(file_paths)}

    def _scan_directory(self, max_depth: int) -> list[str]:
        """Fast directory scan without worrying about order."""
        file_paths = []

        for root, dirs, files in os.walk(self.dir_path):
            current_depth = len(Path(root).relative_to(self.dir_path).parts)

            if current_depth >= max_depth:
                dirs.clear()
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if self.filter_fn(file_path):
                    file_paths.append(file_path)

        return file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self.path_to_idx

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        if isinstance(example_id, list):
            return [self.path_to_idx[id] for id in example_id]
        return self.path_to_idx[example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        if isinstance(idx, list):
            return [self.file_paths[i] for i in idx]
        return self.file_paths[idx]

    def __getitem__(self, idx: int) -> Any:
        """Return the file path at the given index.

        Subclasses can override this to load and process the file content instead.
        """
        return self.file_paths[idx]


class StructuralFileDataset(FileDataset):
    """FileDataset with StructuralDatasetWrapper compatibility.

    Inherits all functionality from FileDataset but adds:
    - .data property that returns a pandas DataFrame for compatibility
    - __getitem__ returns pandas Series instead of just file paths
    - Optional name attribute for logging/debugging

    Allows integration with StructuralDatasetWrapper, samplers, and weight calculation.
    """

    def __init__(
        self,
        source: PathLike | list[str | PathLike],
        filter_fn: Callable[[PathLike], bool] | None = None,
        max_depth: int = 3,
        name: str | None = None,
    ):
        """
        Args:
            source: Either a directory path to scan for files, or a pre-built list of file paths
            filter_fn: Optional function that takes a file path and returns True if the file should be included
            max_depth: Maximum directory depth to scan (only used when source is a directory path)
            name: Optional name for the dataset (useful for logging and debugging)
        """
        super().__init__(source, filter_fn, max_depth)
        self.name = name if name is not None else f"StructuralFileDataset({source})"

        assert len(self.file_paths) == len(set(self.file_paths)), "File paths must be unique."

    @cached_property
    def data(self) -> pd.DataFrame:
        """Return a pandas DataFrame with file paths and generated example IDs.

        This property makes StructuralFileDataset compatible with StructuralDatasetWrapper
        and other components that expect a .data attribute.
        """
        # Generate example IDs from file paths (use filename without extension)
        example_ids = []
        for file_path in self.file_paths:
            filename = Path(file_path).stem  # filename without extension
            # If filename has multiple extensions (e.g., .cif.gz), remove them all
            while "." in filename:
                filename = Path(filename).stem
            example_ids.append(filename)

        # Create DataFrame with path and example_id columns
        df = pd.DataFrame(
            {
                "path": self.file_paths,
                "example_id": example_ids,
            }
        )

        # Set example_id as index for fast lookups
        df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)  # No duplicates allowed

        return df

    def __getitem__(self, idx: int) -> Any:
        return self.data.iloc[idx]


class StructuralDatasetWrapper(BaseDataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_parser: MetadataRowParser,
        cif_parser_args: dict | None = None,
        transform: Transform | Compose | None = None,
        return_key: str | None = None,
        save_failed_examples_to_dir: PathLike | str | None = None,
    ):
        """
        Decorator (wrapper) for an arbitrary Dataset (e.g., PandasDataset, PolarsDataset, etc.) to handle loading of structural data from PDB or CIF files,
        parsing, and applying a Transformation pipeline to the data.

        Designed to be used with a Transforms pipeline to process the data and a MetadataRowParser to convert the dataset rows into a common dictionary format.

        For more detail, see the README in the `datasets` directory.

        Args:
            dataset (Dataset): The dataset to wrap. For example, a PandasDataset, PolarsDataset, or standard PyTorch Dataset.
            dataset_parser (MetadataRowParser): Parser to convert dataset metadata rows into a common dictionary format. See `atomworks.ml.datasets.dataframe_parsers`.
            cif_parser_args (dict, optional): Arguments to pass to `atomworks.io.parse` (will override the defaults). Defaults to None.
            transform (Transform | Compose, optional): Transformation pipeline to apply to the data. See `atomworks.ml.transforms.base`.
            return_key (str, optional): Key to return from the data dictionary instead of the entire dictionary.
            save_failed_examples_to_dir (PathLike | str | None, optional): Directory to save failed examples.

        Example usage:
            ```python
            dataset = StructuralDatasetDecorator(dataset=PandasDataset(data="path/to/data.csv"), ...)
            dataset[0]  # Returns the processed data for the first example.
            ```
        """
        # ...basic assignments
        self.transform = transform
        self.return_key = return_key
        self.save_failed_examples_to_dir = (
            Path(save_failed_examples_to_dir) if exists(save_failed_examples_to_dir) else None
        )
        self.cif_parser_args = cif_parser_args
        self.dataset_parser = dataset_parser
        self.dataset = dataset

        # ...carry forward the data
        self.data = self.dataset.data

        # ...carry forward the name
        self.name = self.dataset.name if hasattr(self.dataset, "name") else repr(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """
        Performs the following steps:
            (1) Retrieve the row at the specified index from the dataset using the __getitem__ method.
            (2) Parse the row into a common dictionary format using the dataset parser.
            (3) Load the CIF file from the information in the common dictionary format (i.e., the "path" key).
            (4) Apply the transformation pipeline to the data which, at a minimum, contains the output of `atomworks.io` parsing.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The processed item.
        """

        # Capture example ID & current rng state (for reproducibility & debugging)
        if hasattr(self, "idx_to_id"):
            # ...if the dataset has a custom idx_to_id method, use it (e.g., for a PandasDataset)
            example_id = self.idx_to_id(idx)
        else:
            # ...otherwise, fallback to a the `id_column` or a string representation of the index
            example_id = self.dataset[idx][self.id_column] if self.id_column else f"row_{idx}"

        # Get process id and hostname (for debugging)
        logger.debug(f"({socket.gethostname()}:{os.getpid()}) Processing example ID: {example_id}")

        # Load the row, using the __getitem__ method of the dataset
        row = self.dataset[idx]

        # Process the row into a transform-ready dictionary with the given CIF and dataset parsers
        # We require the "data" dictionary output from `load_example_from_metadata_row` to contain, at a minimum:
        #   (a) An "id" key, which uniquely identifies the example within the dataframe; and,
        #   (b) The "path" key, which is the path to the CIF file
        _start_parse_time = time.time()
        data = load_example_from_metadata_row(row, self.dataset_parser, cif_parser_args=self.cif_parser_args)
        _stop_parse_time = time.time()

        # Manually add timing for cif-parsing
        data = TransformedDict(data)
        data.__transform_history__.append(
            {
                "name": "load_example_from_metadata_row",
                "instance": hex(id(load_example_from_metadata_row)),
                "start_time": _start_parse_time,
                "end_time": _stop_parse_time,
                "processing_time": _stop_parse_time - _start_parse_time,
            }
        )

        # Apply the transformation pipeline to the data
        if exists(self.transform):
            try:
                rng_state_dict = capture_rng_states(include_cuda=False)
                data = self.transform(data)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                # Log the error and save the failed example to disk (optional)
                logger.info(f"Error processing row {idx} ({example_id}): {e}")

                if exists(self.save_failed_examples_to_dir):
                    save_failed_example_to_disk(
                        example_id=example_id,
                        error_msg=e,
                        rng_state_dict=rng_state_dict,
                        data={},  # We do not save the data, since it may be large.
                        fail_dir=self.save_failed_examples_to_dir,
                    )
                raise e

        # Return the specified key or the entire data dict (i.e., only "feats" key from the Transform dictionary)
        if exists(self.return_key):
            return data[self.return_key]
        else:
            return data

    def __len__(self) -> int:
        """Pass through the length of the wrapped dataset."""
        return len(self.dataset)

    def __contains__(self, example_id: str) -> bool:
        """Pass through the contains method of the wrapped dataset."""
        return example_id in self.dataset

    def id_to_idx(self, example_id: str) -> int:
        """Pass through the id_to_idx method of the wrapped dataset."""
        return self.dataset.id_to_idx(example_id)

    def idx_to_id(self, idx: int) -> str:
        """Pass through the idx_to_id method of the wrapped dataset."""
        return self.dataset.idx_to_id(idx)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped dataset."""
        try:
            # `object.__getattribute__(self, "dataset")` bypasses the custom `__getattr__` and safely retrieves the attribute,
            # avoiding infinite recursion.
            dataset = object.__getattribute__(self, "dataset")
            return getattr(dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object (or its wrapped dataset) has no attribute '{name}'")  # noqa: B904


class PandasDataset(BaseDataset):
    """
    A wrapper around PyTorch's Dataset class that allows for easy loading, filtering, and indexing of datasets stored as Pandas DataFrames.
    The underlying DataFrame can be accessed via the `data` property.

    For example usage, see the tests in `tests/datasets/test_datasets.py`.

    Args:
        data (pd.DataFrame | PathLike): The dataset, either as a Pandas DataFrame or a path to a file.
        id_column (str | None, optional): The column to use as the index; must be unique within the DataFrame. Defaults to None.
            For example, we use the `example_id` column as the index in the `PDBDataset`. By setting the dataframe index to the `example_id`
            column, we can retrieve the row corresponding to a specific example ID by calling `dataset.data.loc[example_id]` in O(1) time.
        filters (list[str] | None, optional): A list of query strings to filter the data. Defaults to None. For examples on how to specify filters,
            see the docstring for `_apply_filters`.
        name (str | None, optional): The name of the dataset. Defaults to None. Useful for debugging and logging.
        columns_to_load (list[str] | None, optional): Specific columns to load if data is provided as a file path. Defaults to None. Helpful for
            large datasets where only a subset of columns is needed (if using `parquet` or other columnar storage formats).
        **load_kwargs (Any): Additional keyword arguments for loading the data.

    Attributes:
        data (pd.DataFrame): The underlying DataFrame, accessible via the `data` property.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        id_column: str | None = None,
        filters: list[str] | None = None,
        name: str | None = None,
        columns_to_load: list[str] | None = None,
        **load_kwargs: Any,
    ):
        if name is not None:
            self.name = name
        else:
            self.name = repr(self)

        # Load the data from the path, if provided (and load only the specified columns)
        if isinstance(data, PathLike | str):
            data = self._load_from_path(data, columns_to_load, **load_kwargs)
        self._data = data

        # Apply filters, if provided
        self.filters = filters
        self._already_filtered = False
        if exists(filters):
            self._apply_filters(filters)
        self._already_filtered = True

        if id_column is not None:
            assert id_column in self._data.columns, f"Column {id_column} not found in dataset."
            self._data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)

    def _load_from_path(
        self, path: PathLike | str, columns_to_load: list[str] | None = None, **load_kwargs: Any
    ) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".csv":
            data = pd.read_csv(path, usecols=columns_to_load, keep_default_na=False, na_values=NA_VALUES, **load_kwargs)
        elif path.suffix == ".parquet":
            data = read_parquet_with_metadata(path, columns=columns_to_load, **load_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return data

    @property
    def data(self) -> pd.DataFrame:
        """Expose underlying dataframe as property to discourage changing it (can lead to unexpected behavior with torch ConcatDatasets)."""
        return self._data

    def __getitem__(self, idx: int) -> Any:
        return self._data.iloc[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self._data.index

    def _id_to_index_single(self, example_id: str) -> int:
        return self._data.index.get_loc(example_id)

    def _id_to_index_multiple(self, example_ids: list[str]) -> list[int]:
        idxs = np.arange(len(self._data))
        return [idxs[self._data.index.get_loc(example_id)] for example_id in example_ids]

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
        ids = self._data.iloc[idx].index.values
        return ids[0] if _return_single else ids

    def _apply_filters(self, filters: list[str]) -> pd.DataFrame:
        """
        Apply filters to the data based on the provided list of query strings.
        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters (List[str]): List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Exampleelse:
            logger.info(
                f"Query '{query}' filtered dataset from {original_num_rows:,} to {filtered_num_rows:,} rows (dropped {original_num_rows - filtered_num_rows:,} rows)"
            ):
            queries = [
                "deposition_date < '2020-01-01'",
                "resolution < 2.5 and ~method.str.contains('NMR')",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            ]
        """
        assert not self._already_filtered, "Filters cannot be applied after initialization."

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            self._apply_query(query)

    def _apply_query(self, query: str) -> None:
        """
        Apply a single query to the data.

        Args:
            query (str): A query string to apply to the data.
        """
        # Filter using query and validate impact
        original_num_rows = len(self._data)
        self._data = self._data.query(query)
        filtered_num_rows = len(self._data)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        """
        Validate the impact of the filter.

        Args:
            query (str): The query string that was applied.
            original_num_rows (int): The number of rows before applying the filter.
            filtered_num_rows (int): The number of rows after applying the filter.

        Raises:
            Warning: If the filter did not remove any rows.
            ValueError: If the filter removed all rows.
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


class ConcatDatasetWithID(ConcatDataset):
    """Equivalent to `torch.utils.data.ConcatDataset` but allows accessing examples by ID."""

    datasets: list[Dataset]

    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)

        # Print the length of each dataset
        for i, dataset in enumerate(datasets):
            logger.info(f"Dataset {i} ({type(dataset)}): {len(dataset):,} examples")

    @cached_property
    def _can_convert_ids_and_idx(self) -> bool:
        has_id_to_idx = all(hasattr(sub_dataset, "id_to_idx") for sub_dataset in self.datasets)
        has_idx_to_id = all(hasattr(sub_dataset, "idx_to_id") for sub_dataset in self.datasets)
        return has_id_to_idx and has_idx_to_id and self._can_check_contains

    @cached_property
    def _can_check_contains(self) -> bool:
        return all(hasattr(sub_dataset, "__contains__") for sub_dataset in self.datasets)

    def _raise_if_cannot_check_contains(self) -> None:
        if not self._can_check_contains:
            raise ValueError("This dataset cannot check if it contains an example ID.")

    def _raise_if_cannot_convert_ids_and_idx(self) -> None:
        if not self._can_convert_ids_and_idx:
            raise ValueError("This dataset cannot convert example IDs to indices.")

    def _raise_if_idx_out_of_bounds(self, idx: int) -> None:
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of bounds for dataset of length {len(self)}.")

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        self._raise_if_cannot_check_contains()
        return any(example_id in sub_dataset for sub_dataset in self.datasets)

    def id_to_idx(self, example_id: str) -> int:
        """Retrieves the index corresponding to the example ID.

        WARNING: Assumes that the example ID is unique within the dataset. If not,
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
        """Retrieves the example ID corresponding to the index."""
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
        """Retrieves the dataset containing the index."""
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_id(self, example_id: str) -> Dataset:
        """Retrieves the dataset containing the example ID.

        WARNING: Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        idx = self.id_to_idx(example_id)
        return self.get_dataset_by_idx(idx)


def get_row_and_index_by_example_id(dataset: ConcatDatasetWithID, example_id: str) -> dict:
    """
    Retrieve a row and its index from a nested dataset structure by its example ID.

    Parameters:
        dataset (PandasDataset | ConcatDataset): The dataset or concatenated dataset to search.
            Must have the `id_to_idx` method.
        example_id (str): The example ID to search for.

    Returns:
        tuple: A tuple containing the row (pd.Series) and the (global)index (int) corresponding to the example ID.
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
    """
    A wrapper around a dataset that allows for a fallback dataset to be used when an error occurs.

    Meant to be used with a FallbackSamplerWrapper.
    """

    def __init__(self, dataset: Dataset, fallback_dataset: Dataset):
        """
        FallbackDatasetWrapper is a wrapper around a dataset that provides a fallback mechanism
        to another dataset in case of errors during data retrieval.

        Attributes:
            - dataset (Dataset): The primary dataset to retrieve data from.
            - fallback_dataset (Dataset): The fallback dataset to use when an error occurs. This
                may be the same as the primary dataset, or a different one.
        """
        self.dataset = dataset
        self.fallback_dataset = fallback_dataset

    def __getitem__(self, idxs: tuple[int, ...]) -> Any:
        """
        Attempt to retrieve an item from the primary dataset, falling back to additional indices if errors occur.
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
        return len(self.dataset)
