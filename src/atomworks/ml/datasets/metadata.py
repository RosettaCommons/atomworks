"""Metadata index for filtering and ID-mapping in molecular datasets."""

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from atomworks.common import as_list
from atomworks.constants import NA_VALUES
from atomworks.ml.utils.io import build_feather_once, job_scoped_feather_path, read_csv, read_parquet_with_metadata

logger = logging.getLogger("datasets")


@runtime_checkable
class MetadataIndexProtocol(Protocol):
    """Structural interface shared by all metadata index backends."""

    def __len__(self) -> int: ...
    def __contains__(self, example_id: str) -> bool: ...
    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]: ...
    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray: ...
    def get_row(self, idx: int) -> pd.Series: ...
    def get_column_values(self, column: str) -> np.ndarray: ...
    def get_example_id(self, idx: int) -> str: ...


class MetadataIndex:
    """Parquet/DataFrame metadata with filtering and ID lookup.

    Single source of truth for example IDs across all dataset backends.
    The ``id_column`` is set as the pandas index for O(1) ``.loc`` lookups.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        name: str,
        id_column: str = "example_id",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        load_kwargs: dict | None = None,
    ):
        """Initialize MetadataIndex.

        Args:
            data: Either a pandas DataFrame or path to a CSV/Parquet file.
            name: Descriptive name for logging.
            id_column: Column to use as the index for example ID lookups.
            filters: Independent row-wise pandas predicates evaluated on the input data.
            columns_to_load: Optional list of columns to load from file.
            load_kwargs: Additional keyword arguments for pandas read functions.
        """
        if isinstance(data, PathLike | str):
            data = _load_from_path(data, columns_to_load, **(load_kwargs or {}))

        assert id_column in data.columns, f"Column {id_column} not found. Available: {list(data.columns)}"

        if filters:
            data = _apply_filters(data, filters, name)

        self.name = name
        data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)
        self.data: pd.DataFrame = data

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self.data.index

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to positional index(es)."""
        if np.isscalar(example_id):
            return self.data.index.get_loc(example_id)
        return [self.data.index.get_loc(eid) for eid in example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert positional index(es) to example ID(s)."""
        _return_single = False
        if np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.shape == ()):
            _return_single = True
            idx = idx.item() if isinstance(idx, np.ndarray) else idx
            idx = slice(idx, idx + 1)
        ids = self.data.iloc[idx].index.values
        return ids[0] if _return_single else ids

    def get_row(self, idx: int) -> pd.Series:
        """Get metadata row by positional index."""
        return self.data.iloc[idx]

    def get_column_values(self, column: str) -> np.ndarray:
        """Get values of a column as numpy array."""
        return np.array(self.data[column])

    def get_example_id(self, idx: int) -> str:
        """Get example ID by positional index."""
        return str(self.data.iloc[idx].name)


class ArrowMetadataIndex:
    """Memory-mapped Arrow metadata with the same interface as :class:`MetadataIndex`.

    Converts the DataFrame to an uncompressed feather file on local storage and
    reads it back as a memory-mapped :class:`pyarrow.Table`, keeping Python heap
    usage minimal. ID lookups are O(N) — acceptable because they are not on the
    DataLoader hot path.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        name: str,
        id_column: str = "example_id",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        load_kwargs: dict | None = None,
        local_drive_mount: str = "/tmp",
        job_id_env_var: str = "SLURM_JOB_ID",
    ):
        """Initialize ArrowMetadataIndex.

        Args:
            data: Either a pandas DataFrame or path to a CSV/Parquet file.
            name: Descriptive name for logging.
            id_column: Column to use as the ID column.
            filters: Independent row-wise pandas predicates evaluated on the input data.
            columns_to_load: Optional list of columns to load from file.
            load_kwargs: Additional keyword arguments for pandas read functions.
            local_drive_mount: Root directory for feather files.
            job_id_env_var: Environment variable used to namespace feather files
                per job (e.g. ``"SLURM_JOB_ID"``).
        """
        self._id_column = id_column
        self.name = name
        self.feather_path = _build_feather_index(
            data=data,
            name=name,
            id_column=id_column,
            filters=filters,
            columns_to_load=columns_to_load,
            load_kwargs=load_kwargs,
            local_drive_mount=local_drive_mount,
            job_id_env_var=job_id_env_var,
        )
        self.data: pa.Table = feather.read_table(self.feather_path, memory_map=True)
        assert id_column in self.data.column_names, f"Column {id_column} not found. Available: {self.data.column_names}"

    def __getstate__(self) -> dict:
        """Pickle-friendly: store path instead of the full table."""
        return {
            "feather_path": self.feather_path,
            "_id_column": self._id_column,
            "name": self.name,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle by re-opening the feather file."""
        self.feather_path = state["feather_path"]
        self._id_column = state["_id_column"]
        self.name = state["name"]
        self.data = feather.read_table(self.feather_path, memory_map=True)

    def __len__(self) -> int:
        return self.data.num_rows

    def __contains__(self, example_id: str) -> bool:
        return pc.index(self.data.column(self._id_column), example_id).as_py() != -1

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to positional index(es). O(N) per call."""
        col = self.data.column(self._id_column)
        if np.isscalar(example_id):
            idx = pc.index(col, example_id).as_py()
            if idx == -1:
                raise KeyError(example_id)
            return idx
        mask = pc.is_in(col, value_set=pa.array(example_id))
        matching_rows = np.where(mask.to_numpy())[0]
        matching_ids = col.take(matching_rows).to_pylist()
        id_to_row = {str(v): int(r) for v, r in zip(matching_ids, matching_rows, strict=False)}
        return [id_to_row[eid] for eid in example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert positional index(es) to example ID(s)."""
        col = self.data.column(self._id_column)
        if np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.shape == ()):
            idx = idx.item() if isinstance(idx, np.ndarray) else idx
            return str(col[idx].as_py())
        return np.array([str(col[i].as_py()) for i in idx])

    def get_row(self, idx: int) -> pd.Series:
        """Get metadata row by positional index."""
        return self.data.slice(idx, 1).to_pandas().iloc[0]

    def get_column_values(self, column: str) -> np.ndarray:
        """Get values of a column as numpy array."""
        return self.data.column(column).to_numpy(zero_copy_only=False)

    def get_example_id(self, idx: int) -> str:
        """Get example ID by positional index."""
        return str(self.data.column(self._id_column)[idx].as_py())


class SequentialMetadataIndex:
    """Identity metadata mapping ``"0"`` … ``"n-1"`` to sequential indices.

    Same interface as :class:`MetadataIndex` / :class:`ArrowMetadataIndex`
    for datasets that don't need an external metadata file.
    """

    def __init__(self, *, n_entries: int, idx_column: str):
        self._n = n_entries
        self._idx_column = idx_column

    def __len__(self) -> int:
        return self._n

    def __contains__(self, example_id: str) -> bool:
        try:
            return 0 <= int(example_id) < self._n
        except (ValueError, TypeError):
            return False

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert string index(es) to int."""
        if np.isscalar(example_id):
            idx = int(example_id)
            if not 0 <= idx < self._n:
                raise KeyError(example_id)
            return idx
        return [self.id_to_idx(eid) for eid in example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert int index(es) to string."""
        if np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.shape == ()):
            return str(idx.item() if isinstance(idx, np.ndarray) else idx)
        return np.array([str(i) for i in idx])

    def get_column_values(self, column: str) -> np.ndarray:
        """Return sequential indices for ``idx_column``."""
        if column != self._idx_column:
            raise KeyError(f"Column {column!r} not available without a metadata file")
        return np.arange(self._n)

    def get_row(self, idx: int) -> pd.Series:
        """Not available — ``SequentialMetadataIndex`` has no row data."""
        raise NotImplementedError(
            "SequentialMetadataIndex has no row data. " "Provide a metadata file to use get_row()."
        )

    def get_example_id(self, idx: int) -> str:
        return str(idx)


def _build_feather_index(
    *,
    data: pd.DataFrame | PathLike | str,
    name: str,
    id_column: str,
    filters: list[str] | None,
    columns_to_load: list[str] | None,
    load_kwargs: dict | None,
    local_drive_mount: str,
    job_id_env_var: str,
) -> str:
    """Build once (under a lock) and return the path to a memory-mappable feather index."""
    # Key by name *and* content (filters, columns, source): one name can mean different rows.
    source_key = str(data) if isinstance(data, PathLike | str) else None  # in-memory frame: name only
    feather_path = job_scoped_feather_path(
        name,
        local_drive_mount=local_drive_mount,
        job_id_env_var=job_id_env_var,
        content_key=(
            source_key,
            id_column,
            filters,
            "cumulative_masks",
            columns_to_load,
            sorted((load_kwargs or {}).items()),
        ),
    )

    def build() -> pa.Table:
        # Read the source only here (feather missing) — not once per rank.
        df = _load_from_path(data, columns_to_load, **(load_kwargs or {})) if isinstance(data, PathLike | str) else data
        assert id_column in df.columns, f"Column {id_column} not found. Available: {list(df.columns)}"
        if filters:
            df = _apply_filters(df, filters, name)
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "N/A"
        except ImportError:
            rank = "N/A"
        logger.info(f"Rank {rank}: Converting {name} to feather at {feather_path}")
        return pa.Table.from_pandas(df)

    return build_feather_once(feather_path, build)


def _load_from_path(
    path: PathLike | str,
    columns_to_load: list[str] | None = None,
    **load_kwargs: Any,
) -> pd.DataFrame:
    """Load DataFrame from a CSV or Parquet file (local path or ``s3://`` URL)."""
    suffix = Path(str(path)).suffix
    if columns_to_load is not None:
        columns_to_load = as_list(columns_to_load)
    if suffix == ".csv":
        return read_csv(path, usecols=columns_to_load, keep_default_na=False, na_values=NA_VALUES, **load_kwargs)
    elif suffix == ".parquet":
        return read_parquet_with_metadata(path, columns=columns_to_load, **load_kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _apply_filters(
    data: pd.DataFrame,
    filters: list[str],
    name: str,
) -> pd.DataFrame:
    """Combine independent row-wise masks with cumulative waterfall counts, then select rows once."""
    initial_count = len(data)
    filtered_count = initial_count
    keep = np.ones(initial_count, dtype=bool)
    filter_results: list[tuple[str, int]] = []

    for query in filters:
        original_count = filtered_count
        mask = data.eval(query)
        if not isinstance(mask, pd.Series) or not pd.api.types.is_bool_dtype(mask.dtype):
            raise ValueError(f"Dataset filters require a boolean row-wise predicate, got {query!r}.")
        keep &= mask.to_numpy(dtype=bool, na_value=False)
        filtered_count = int(keep.sum())

        if filtered_count == 0:
            raise ValueError(f"Query '{query}' on dataset {name} removed all rows.")

        rows_removed = original_count - filtered_count
        filter_results.append((query, rows_removed))

    _log_filter_summary(name, initial_count, filtered_count, filter_results)
    return data.loc[keep]


def _log_filter_summary(
    name: str,
    initial_count: int,
    final_count: int,
    filter_results: list[tuple[str, int]],
    max_width: int = 120,
) -> None:
    """Log a waterfall summary of all applied filters."""
    for query, removed in filter_results:
        if removed == 0:
            logger.warning(f"Query '{query}' on dataset {name} did not remove any rows.")

    total_removed = initial_count - final_count
    total_pct = (total_removed / initial_count) * 100 if initial_count > 0 else 0.0

    header = f" {name}: {initial_count:,} \u2192 {final_count:,} rows ({total_pct:.1f}% removed) "

    bar_width = 22
    # inner_width = max_width - 2 (for border chars)
    inner_width = max_width - 2
    # Format: "  <query> <count> <pct> <bar>  "
    #          2 + query + 1 + 10 + 1 + 7 + 1 + bar + 2(padding) = inner_width
    max_query_width = inner_width - 2 - 1 - 10 - 1 - 7 - 1 - bar_width - 2

    filter_lines: list[str] = []
    for query, removed in filter_results:
        pct = (removed / initial_count) * 100 if initial_count > 0 else 0.0
        filled = round(pct / 100 * bar_width)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        count_str = f"-{removed:,}"
        pct_str = f"({pct:.1f}%)"
        if len(query) > max_query_width:
            query = query[: max_query_width - 3] + "..."
        filter_lines.append(f"  {query:<{max_query_width}s} {count_str:>10s} {pct_str:>7s} {bar}")

    top = f"\u250c{header:\u2500<{inner_width}}\u2510"
    bottom = f"\u2514{'\u2500' * inner_width}\u2518"
    empty = f"\u2502{' ' * inner_width}\u2502"

    lines = [top, empty]
    for fl in filter_lines:
        lines.append(f"\u2502{fl:<{inner_width}}\u2502")
    lines += [empty, bottom]

    logger.info("\n" + "\n".join(lines))
