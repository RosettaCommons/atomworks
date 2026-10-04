"""CIF-based dataset loaders."""

import functools
import io
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from atomworks.io.config import ParseConfig
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.utils.io_utils import infer_pdb_file_type
from atomworks.ml.utils.blob_store import BlobIndex, BlobStore

from .base import _construct_metadata_hierarchy, _construct_structure_path

ColumnMapping = Mapping[str, str | Sequence[str] | None]


def _warn_deprecated(name: str, replacement: str) -> None:
    warnings.warn(f"{name} is deprecated; use {replacement}", DeprecationWarning, stacklevel=3)


def _normalize_altloc_seed(value: Any) -> int | None:
    """Convert a scalar metadata seed to a Python integer, preserving missing values."""
    return None if pd.isna(value) else int(value)


def _apply_column_mapping(
    result: dict[str, Any],
    row: pd.Series | None,
    column_mapping: ColumnMapping | None,
) -> dict[str, Any]:
    """Promote selected metadata columns into named loader outputs."""
    if not column_mapping:
        return result
    if row is None:
        raise ValueError("column_mapping requires a metadata row")

    collisions = result.keys() & column_mapping.keys()
    if collisions:
        raise ValueError(f"column_mapping cannot replace loader outputs: {sorted(collisions)}")

    for output_name, source_columns in column_mapping.items():
        if source_columns is None:
            result[output_name] = None
            continue
        columns = [source_columns] if isinstance(source_columns, str) else list(source_columns)
        result[output_name] = (
            row[columns[0]] if isinstance(source_columns, str) else [row[column] for column in columns]
        )
        for column in columns:
            result["extra_info"].pop(column, None)
    return result


def _resolve_parse_config(
    parser_args: ParseConfig | dict | None,
    assembly_id: str,
    source: Path | io.BytesIO | io.StringIO | Callable[[], io.StringIO],
    file_type: str | None = None,
    altloc_seed: int | None = None,
) -> ParseConfig:
    """Resolve ``parser_args`` to a :py:class:`ParseConfig`, stamping ``build_assembly`` from ``assembly_id`` for CIF/bCIF inputs.

    Accepts a ``ParseConfig`` (preferred) or a legacy dict of kwargs (merged
    with :py:data:`STANDARD_PARSER_ARGS` via :py:meth:`ParseConfig.from_dict`).

    When ``altloc_seed`` is provided, altloc selection is switched to
    ``"random_clash_aware"`` with the given seed for deterministic sampling.
    """
    if not isinstance(parser_args, ParseConfig):
        # Legacy dict support: merge with STANDARD_PARSER_ARGS and convert to ParseConfig
        parser_args = ParseConfig.from_dict({**STANDARD_PARSER_ARGS, **(parser_args or {})})

    if file_type is not None:
        # Overwrite file_type in parser_args if explicitly provided (e.g. for CIF bytes loader)
        parser_args = parser_args.replace(file_type=file_type)

    if (parser_args.file_type or infer_pdb_file_type(source)) in ("cif", "bcif", "mmjson"):
        parser_args = parser_args.replace(build_assembly=(assembly_id,))

        if altloc_seed is not None:
            # For CIF inputs, if altloc_seed is provided, switch to random_clash_aware altloc selection with the given seed
            parser_args = parser_args.replace(altloc="random_clash_aware", altloc_seed=altloc_seed)

    return parser_args


def _base_loader_function(
    row: pd.Series,
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    attrs: dict | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    parser_args: ParseConfig | dict | None = None,
    column_mapping: ColumnMapping | None = None,
) -> dict[str, Any]:
    """Base loader function (picklable when used with functools.partial)."""
    # Prepare loader-specific attributes
    loader_attrs = (attrs or {}).copy()
    if base_path and "base_path" not in loader_attrs:
        loader_attrs["base_path"] = base_path
    if extension and "extension" not in loader_attrs:
        loader_attrs["extension"] = extension

    extra_info = _construct_metadata_hierarchy(row, loader_attrs)

    assembly_id = row[assembly_id_colname] if assembly_id_colname is not None and assembly_id_colname in row else "1"
    altloc_seed = _normalize_altloc_seed(row[altloc_seed_colname]) if altloc_seed_colname is not None else None

    path = _construct_structure_path(
        row[path_colname], extra_info.get("base_path"), extra_info.get("extension"), sharding_pattern
    )
    result_dict = parse(path, config=_resolve_parse_config(parser_args, assembly_id, path, altloc_seed=altloc_seed))

    # Remove used columns from extra_info
    exclude_cols = (
        [example_id_colname, path_colname]
        + ([assembly_id_colname] if assembly_id_colname else [])
        + ([altloc_seed_colname] if altloc_seed_colname else [])
        + ["base_path", "extension"]
    )
    extra_info = {k: v for k, v in extra_info.items() if k not in exclude_cols}

    result = {
        "example_id": row[example_id_colname],
        "path": path,
        "assembly_id": assembly_id,
        "altloc_seed": altloc_seed,
        "extra_info": extra_info,
        "atom_array": result_dict["assemblies"][assembly_id][0],
        "atom_array_stack": result_dict["assemblies"][assembly_id],
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    return _apply_column_mapping(result, row, column_mapping)


def create_base_loader(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    attrs: dict | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Deprecated; use :func:`create_structure_loader`."""
    _warn_deprecated("create_base_loader", 'create_structure_loader(storage="filesystem")')
    return create_structure_loader(
        storage="filesystem",
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )


def create_loader_with_query_pn_units(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    pn_unit_iid_colnames: str | list[str] | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Deprecated; use :func:`create_structure_loader` with ``column_mapping``."""
    _warn_deprecated("create_loader_with_query_pn_units", "create_structure_loader(column_mapping=...)")

    if isinstance(pn_unit_iid_colnames, str):
        pn_unit_iid_colnames = [pn_unit_iid_colnames]
    pn_unit_iid_colnames = pn_unit_iid_colnames or []

    return create_structure_loader(
        storage="filesystem",
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
        column_mapping={"query_pn_unit_iids": pn_unit_iid_colnames},
    )


def create_loader_with_interfaces_and_pn_units_to_score(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    interfaces_to_score_colname: str | None = "interfaces_to_score",
    pn_units_to_score_colname: str | None = "pn_units_to_score",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Deprecated; use :func:`create_structure_loader` with ``column_mapping``."""
    _warn_deprecated(
        "create_loader_with_interfaces_and_pn_units_to_score", "create_structure_loader(column_mapping=...)"
    )

    return create_structure_loader(
        storage="filesystem",
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
        column_mapping={
            "interfaces_to_score": interfaces_to_score_colname,
            "pn_units_to_score": pn_units_to_score_colname,
        },
    )


def _cif_bytes_loader_function(
    raw_data: tuple,
    parser_args: ParseConfig | dict | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    column_mapping: ColumnMapping | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Loader for CIF bytes (picklable when used with functools.partial)."""
    cif_bytes, global_idx, metadata_row = raw_data

    # Extract assembly_id from metadata row when available
    assembly_id = "1"
    if metadata_row is not None and assembly_id_colname is not None and assembly_id_colname in metadata_row.index:
        assembly_id = str(metadata_row[assembly_id_colname])

    altloc_seed = None
    if metadata_row is not None and altloc_seed_colname is not None and altloc_seed_colname in metadata_row.index:
        altloc_seed = _normalize_altloc_seed(metadata_row[altloc_seed_colname])

    source = cif_bytes if callable(cif_bytes) else io.StringIO(cif_bytes.decode("utf-8"))
    result = parse(
        source,
        config=_resolve_parse_config(parser_args, assembly_id, source, file_type="cif", altloc_seed=altloc_seed),
        cache_key=cache_key,
    )
    # Build extra_info from metadata row
    extra_info: dict[str, Any] = {}
    if metadata_row is not None:
        extra_info = _construct_metadata_hierarchy(metadata_row, {})
        # Remove columns already represented in the output
        exclude_cols = {"example_id"}
        if assembly_id_colname:
            exclude_cols.add(assembly_id_colname)
        extra_info = {k: v for k, v in extra_info.items() if k not in exclude_cols}

    result = {
        "assembly_id": assembly_id,
        "extra_info": extra_info,
        "atom_array": result["assemblies"][assembly_id][0],
        "atom_array_stack": result["assemblies"][assembly_id],
        "chain_info": result["chain_info"],
        "ligand_info": result["ligand_info"],
        "metadata": result["metadata"],
    }
    return _apply_column_mapping(result, metadata_row, column_mapping)


def create_cif_bytes_loader(
    parser_args: ParseConfig | dict | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
) -> Callable:
    """Deprecated; use :func:`create_structure_loader` with ``storage="bytes"``."""
    _warn_deprecated("create_cif_bytes_loader", 'create_structure_loader(storage="bytes")')
    return create_structure_loader(
        storage="bytes",
        parser_args=parser_args,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
    )


def _blob_cif_loader_function(
    row: pd.Series,
    store: BlobStore,
    index: BlobIndex,
    id_column: str,
    parser_args: ParseConfig | dict | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    example_id_colname: str | None = None,
    column_mapping: ColumnMapping | None = None,
) -> dict[str, Any]:
    """Fetch a structure record from the blob store and retain the sampled example identity."""
    example_id_colname = example_id_colname or id_column
    if id_column in row.index:
        record_id = row[id_column]
    elif id_column == example_id_colname:
        record_id = row.name
    else:
        raise KeyError(f"Record ID column {id_column!r} is missing from the metadata row")
    if example_id_colname in row.index:
        example_id = row[example_id_colname]
    elif id_column == example_id_colname:
        example_id = row.name
    else:
        raise KeyError(f"Example ID column {example_id_colname!r} is missing from the metadata row")
    location = index.lookup(str(record_id))
    out = _cif_bytes_loader_function(
        (lambda: io.StringIO(store.get_bytes(*location).decode("utf-8")), None, row),
        parser_args,
        assembly_id_colname,
        altloc_seed_colname,
        column_mapping,
        cache_key=json.dumps(["blob", store.endpoint_url, store.data_dir, *location]),
    )
    if id_column != example_id_colname:
        out["extra_info"].pop(id_column, None)
    out["example_id"] = example_id
    return out


def create_blob_cif_loader(
    blob_dir: str,
    *,
    endpoint_url: str | None = None,
    example_id_colname: str = "example_id",
    record_id_colname: str | None = None,
    assembly_id_colname: str | None = "assembly_id",
    altloc_seed_colname: str | None = None,
    parser_args: ParseConfig | dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Deprecated; use :func:`create_structure_loader` with ``storage="blob"``."""
    _warn_deprecated("create_blob_cif_loader", 'create_structure_loader(storage="blob")')
    return create_structure_loader(
        storage="blob",
        blob_dir=blob_dir,
        endpoint_url=endpoint_url,
        example_id_colname=example_id_colname,
        record_id_colname=record_id_colname,
        parser_args=parser_args,
        assembly_id_colname=assembly_id_colname,
        altloc_seed_colname=altloc_seed_colname,
    )


def create_structure_loader(
    storage: Literal["filesystem", "bytes", "blob"] = "filesystem",
    *,
    column_mapping: ColumnMapping | None = None,
    **kwargs: Any,
) -> Callable:
    """Create a structure loader with consistent metadata projection across storage backends.

    ``column_mapping`` maps output names to one metadata column or an ordered sequence of columns.
    Remaining arguments are forwarded to the selected backend factory.
    Blob loaders check the parsed cache before fetching records; shard paths must be immutable
    and globally identify their contents. Row metadata is applied after caching.
    """
    if storage in {"filesystem", "bytes"}:
        loader = _base_loader_function if storage == "filesystem" else _cif_bytes_loader_function
        return functools.partial(loader, column_mapping=column_mapping, **kwargs)

    if storage == "blob":
        blob_dir = kwargs.pop("blob_dir").rstrip("/")
        endpoint_url = kwargs.pop("endpoint_url", None)
        s3_config = kwargs.pop("s3_config", None)
        store = BlobStore(f"{blob_dir}/data", endpoint_url=endpoint_url, s3_config=s3_config)
        example_id_colname = kwargs.pop("example_id_colname", "example_id")
        record_id_colname = kwargs.pop("record_id_colname", None) or example_id_colname
        return functools.partial(
            _blob_cif_loader_function,
            store=store,
            index=BlobIndex(f"{blob_dir}/index.parquet", id_column=record_id_colname, s3_config=store.s3_config),
            id_column=record_id_colname,
            example_id_colname=example_id_colname,
            column_mapping=column_mapping,
            **kwargs,
        )

    raise ValueError(f"Unsupported structure storage: {storage!r}")
