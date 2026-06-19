"""ASE LMDB-backed dataset implementation."""

from bisect import bisect_right
from collections.abc import Callable, Sequence
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from atomworks.ml.datasets.base import ExampleIDMixin, MolecularDataset

LMDBReturnType = Literal["record", "atoms", "row"]

_ASE_IMPORT_ERROR = (
    "ASE LMDB dataset support requires the optional ASE dependencies. "
    'Install them with `pip install "atomworks[ase]"` or `uv pip install "atomworks[ase]"`.'
)


def _require_ase_connect() -> Callable[..., Any]:
    """Import ASE DB support lazily so base AtomWorks installs do not require ASE."""
    try:
        # Importing this package makes ASE's `aselmdb` backend available.
        import ase_db_backends.aselmdb  # noqa: F401
        from ase.db import connect
    except ImportError as exc:
        raise ImportError(_ASE_IMPORT_ERROR) from exc
    return connect


def _normalize_lmdb_paths(paths: str | PathLike | Sequence[str | PathLike]) -> list[Path]:
    """Normalize file and directory inputs into a sorted list of ``*.aselmdb`` files."""
    if isinstance(paths, str | PathLike):
        paths = [paths]

    normalized_paths: list[Path] = []
    for path_like in paths:
        path = Path(path_like).expanduser()
        if path.is_dir():
            normalized_paths.extend(sorted(path.rglob("*.aselmdb")))
        else:
            normalized_paths.append(path)

    if not normalized_paths:
        raise ValueError("No LMDB paths were provided.")

    missing_paths = [path for path in normalized_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"LMDB path(s) do not exist: {missing_paths}")

    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("LMDB paths must be unique.")

    return sorted(normalized_paths)


def _make_shard_ids(paths: Sequence[Path]) -> list[str]:
    """Create stable, parseable shard IDs from file stems."""
    counts: dict[str, int] = {}
    shard_ids: list[str] = []
    for i, path in enumerate(paths):
        stem = path.stem
        if stem in counts:
            counts[stem] += 1
            shard_id = f"{stem}-{counts[stem]}"
        else:
            counts[stem] = 0
            shard_id = stem
        if ":" in shard_id:
            shard_id = f"shard-{i}"
        shard_ids.append(shard_id)
    return shard_ids


def _get_row_data(row: Any) -> dict[str, Any]:
    """Return the ASE row's data dictionary, if present."""
    data = row.get("data", {}) if hasattr(row, "get") else {}
    return data if isinstance(data, dict) else {}


def _get_row_metadata_value(row: Any, key: str) -> Any:
    """Look up a metadata value across ASE row fields, key-value pairs, and data."""
    if key in row:
        return row[key]

    key_value_pairs = getattr(row, "key_value_pairs", {})
    if key in key_value_pairs:
        return key_value_pairs[key]

    data = _get_row_data(row)
    if key in data:
        return data[key]

    raise KeyError(f"Key {key!r} not found in ASE row metadata.")


def _get_calculator_results(row: Any) -> dict[str, Any]:
    """Extract ASE calculator result fields stored on a database row."""
    try:
        from ase.calculators.calculator import all_properties
    except ImportError as exc:
        raise ImportError(_ASE_IMPORT_ERROR) from exc

    return {property_name: row[property_name] for property_name in all_properties if property_name in row}


class ASELMDBDataset(MolecularDataset, ExampleIDMixin):
    """Dataset for ASE DB-compatible LMDB files such as OMol25 ``*.aselmdb`` shards.

    The dataset reads ASE ``AtomsRow`` entries lazily, supports one or more LMDB
    shards, and follows AtomWorks' loader/transform pipeline:

    1. Retrieve an ASE row by global index.
    2. Return the row as a record, an ASE ``Atoms`` object, or the raw row.
    3. Optionally pass that raw object through a loader and transforms.
    """

    def __init__(
        self,
        *,
        paths: str | PathLike | Sequence[str | PathLike],
        name: str,
        return_type: LMDBReturnType = "record",
        example_id_key: str | None = None,
        build_id_index: bool = False,
        readonly: bool = True,
        readahead: bool = False,
        include_data: bool = True,
        db_kwargs: dict[str, Any] | None = None,
        transform: Callable | None = None,
        loader: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        """Initialize an ASE LMDB dataset.

        Args:
            paths: A ``*.aselmdb`` path, a directory containing ``*.aselmdb`` files,
                or a sequence of either. Directories are scanned recursively.
            name: Descriptive dataset name used for logging/debugging.
            return_type: Raw object returned before the optional loader:
                ``"record"`` returns a dictionary with ASE atoms and metadata,
                ``"atoms"`` returns an ASE ``Atoms`` object, and ``"row"``
                returns the ASE ``AtomsRow``.
            example_id_key: Optional metadata key to use as the example ID.
                If omitted, IDs are generated as ``"<shard_id>:<ase_row_id>"``.
            build_id_index: Build an in-memory ID-to-index map. This is required
                for efficient ``id_to_idx()`` when ``example_id_key`` is set, but
                can be expensive for very large datasets.
            readonly: Open LMDB shards in read-only mode.
            readahead: LMDB readahead flag. For random-access training on large
                datasets, ``False`` is usually preferable.
            include_data: Whether to include ASE DB ``data`` payloads when reading rows.
            db_kwargs: Additional keyword arguments passed to ``ase.db.connect``.
            transform: Optional transform pipeline applied after loading.
            loader: Optional loader applied to the raw row/atoms/record.
            save_failed_examples_to_dir: Optional directory for failed examples.
        """
        super().__init__(
            name=name,
            transform=transform,
            loader=loader,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        if return_type not in ("record", "atoms", "row"):
            raise ValueError(f"Unsupported return_type: {return_type!r}")

        self.paths = _normalize_lmdb_paths(paths)
        self.return_type = return_type
        self.example_id_key = example_id_key
        self.readonly = readonly
        self.readahead = readahead
        self.include_data = include_data
        self.db_kwargs = db_kwargs or {}

        self.shard_ids = _make_shard_ids(self.paths)
        self._shard_id_to_idx = {shard_id: i for i, shard_id in enumerate(self.shard_ids)}
        self._dbs: list[Any | None] = [None] * len(self.paths)

        self._lengths = self._read_shard_lengths()
        self._cumulative_lengths = [0]
        for length in self._lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)

        self._id_to_idx_map = self._build_id_index() if build_id_index else None

    @classmethod
    def from_directory(
        cls,
        *,
        directory: str | PathLike,
        name: str,
        **kwargs: Any,
    ) -> "ASELMDBDataset":
        """Create an ASE LMDB dataset by recursively scanning a directory."""
        return cls(paths=directory, name=name, **kwargs)

    def __len__(self) -> int:
        """Return the total number of rows across all LMDB shards."""
        return self._cumulative_lengths[-1]

    def __contains__(self, example_id: str) -> bool:
        """Check whether an example ID exists in this dataset."""
        try:
            self.id_to_idx(example_id)
        except (KeyError, ValueError):
            return False
        return True

    def __getitem__(self, idx: int) -> Any:
        """Load, optionally process, and transform an ASE LMDB example."""
        shard_idx, local_idx = self._resolve_index(idx)
        row = self._get_row_by_local_index(shard_idx, local_idx)
        example_id = self._get_example_id(row, shard_idx)

        raw_data = self._format_row(row, shard_idx, local_idx, example_id)
        data = self._apply_loader(raw_data)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to global dataset index/indices."""
        if isinstance(example_id, list):
            return [self.id_to_idx(id_) for id_ in example_id]

        if self._id_to_idx_map is not None:
            return self._id_to_idx_map[example_id]

        if self.example_id_key is not None:
            raise ValueError(
                "id_to_idx() with example_id_key requires build_id_index=True. "
                "For large datasets, prefer generated IDs or build a custom external index."
            )

        return self._generated_id_to_idx(example_id)

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert global index/indices to example ID(s)."""
        if isinstance(idx, list):
            return [self.idx_to_id(i) for i in idx]

        shard_idx, local_idx = self._resolve_index(idx)
        if self.example_id_key is None:
            row_id = self._row_id_for_local_index(shard_idx, local_idx)
            return self._make_generated_example_id(shard_idx, row_id)

        row = self._get_row_by_local_index(shard_idx, local_idx)
        return self._get_example_id(row, shard_idx)

    def close(self) -> None:
        """Close any open LMDB handles held by this dataset instance."""
        for i, db in enumerate(self._dbs):
            if db is None:
                continue
            with suppress(Exception):
                db.close()
            self._dbs[i] = None

    def __enter__(self) -> "ASELMDBDataset":
        """Return this dataset as a context manager."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        """Close LMDB handles when leaving a context manager."""
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Drop live LMDB handles before pickling for DataLoader workers."""
        state = self.__dict__.copy()
        for db in state.get("_dbs", []):
            if db is not None:
                with suppress(Exception):
                    db.close()
        state["_dbs"] = [None] * len(self.paths)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore dataset state after pickling."""
        self.__dict__.update(state)
        self._dbs = [None] * len(self.paths)

    def __del__(self) -> None:
        """Best-effort cleanup for open LMDB handles."""
        with suppress(Exception):
            self.close()

    def _read_shard_lengths(self) -> list[int]:
        """Open each shard once to read its length."""
        lengths = []
        for shard_idx in range(len(self.paths)):
            db = self._connect_db(shard_idx)
            try:
                lengths.append(len(db))
            finally:
                with suppress(Exception):
                    db.close()
        return lengths

    def _connect_db(self, shard_idx: int) -> Any:
        """Open one ASE LMDB shard."""
        connect = _require_ase_connect()
        kwargs = {
            "type": "aselmdb",
            "readonly": self.readonly,
            "readahead": self.readahead,
        }
        kwargs.update(self.db_kwargs)
        return connect(self.paths[shard_idx], **kwargs)

    def _get_db(self, shard_idx: int) -> Any:
        """Return an open DB handle for one shard, opening lazily if needed."""
        db = self._dbs[shard_idx]
        if db is None:
            db = self._connect_db(shard_idx)
            self._dbs[shard_idx] = db
        return db

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        """Resolve a global dataset index to ``(shard_idx, local_idx)``."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}.")

        shard_idx = bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[shard_idx]
        return shard_idx, local_idx

    def _get_row_by_local_index(self, shard_idx: int, local_idx: int) -> Any:
        """Read one ASE row by its local index within a shard."""
        db = self._get_db(shard_idx)
        if hasattr(db, "_get_row_by_index"):
            return db._get_row_by_index(local_idx, include_data=self.include_data)

        row_id = self._row_id_for_local_index(shard_idx, local_idx)
        return db.get(id=row_id, include_data=self.include_data)

    def _row_id_for_local_index(self, shard_idx: int, local_idx: int) -> int:
        """Return the ASE database row ID for a local shard index."""
        db = self._get_db(shard_idx)
        return int(db.ids[local_idx])

    def _format_row(self, row: Any, shard_idx: int, local_idx: int, example_id: str) -> Any:
        """Format an ASE row according to ``return_type``."""
        if self.return_type == "row":
            return row

        atoms = self._row_to_atoms(row, shard_idx, local_idx, example_id)
        if self.return_type == "atoms":
            return atoms

        key_value_pairs = dict(getattr(row, "key_value_pairs", {}))
        data = _get_row_data(row)
        calculator_results = _get_calculator_results(row)
        extra_info = key_value_pairs | data

        return {
            "example_id": example_id,
            "path": self.paths[shard_idx],
            "db_path": self.paths[shard_idx],
            "db_shard": shard_idx,
            "db_index": local_idx,
            "db_id": int(row.id),
            "shard_id": self.shard_ids[shard_idx],
            "atoms": atoms,
            "key_value_pairs": key_value_pairs,
            "data": data,
            "calculator_results": calculator_results,
            "extra_info": extra_info,
        }

    def _row_to_atoms(self, row: Any, shard_idx: int, local_idx: int, example_id: str) -> Any:
        """Convert an ASE row to an ASE Atoms object and flatten useful metadata into ``atoms.info``."""
        atoms = row.toatoms(add_additional_information=True)
        data = _get_row_data(row)
        key_value_pairs = dict(getattr(row, "key_value_pairs", {}))
        atoms.info.update(key_value_pairs)
        atoms.info.update(data)
        atoms.info.update(
            {
                "example_id": example_id,
                "db_path": str(self.paths[shard_idx]),
                "db_shard": shard_idx,
                "db_index": local_idx,
                "db_id": int(row.id),
                "shard_id": self.shard_ids[shard_idx],
            }
        )
        return atoms

    def _get_example_id(self, row: Any, shard_idx: int) -> str:
        """Return the configured example ID for one row."""
        if self.example_id_key is None:
            return self._make_generated_example_id(shard_idx, int(row.id))

        return str(_get_row_metadata_value(row, self.example_id_key))

    def _make_generated_example_id(self, shard_idx: int, row_id: int) -> str:
        """Create a stable, reversible example ID from a shard ID and ASE row ID."""
        return f"{self.shard_ids[shard_idx]}:{row_id}"

    def _generated_id_to_idx(self, example_id: str) -> int:
        """Convert a generated ``<shard_id>:<row_id>`` ID back to a global index."""
        try:
            shard_id, row_id_str = example_id.rsplit(":", maxsplit=1)
            row_id = int(row_id_str)
        except ValueError as exc:
            raise KeyError(f"Invalid generated ASE LMDB example ID: {example_id!r}") from exc

        if shard_id not in self._shard_id_to_idx:
            raise KeyError(f"Unknown ASE LMDB shard ID: {shard_id!r}")

        shard_idx = self._shard_id_to_idx[shard_id]
        db = self._get_db(shard_idx)
        try:
            local_idx = db.ids.index(row_id)
        except ValueError as exc:
            raise KeyError(f"ASE row ID {row_id} not found in shard {shard_id!r}") from exc

        return self._cumulative_lengths[shard_idx] + local_idx

    def _build_id_index(self) -> dict[str, int]:
        """Build an in-memory map from example ID to global index."""
        id_to_idx_map: dict[str, int] = {}
        for idx in range(len(self)):
            example_id = self.idx_to_id(idx)
            if example_id in id_to_idx_map:
                raise ValueError(f"Duplicate example ID found while indexing ASE LMDB dataset: {example_id!r}")
            id_to_idx_map[example_id] = idx
        return id_to_idx_map


LMDBDataset = ASELMDBDataset
"""Backward-compatible alias for ASE-compatible LMDB datasets."""
