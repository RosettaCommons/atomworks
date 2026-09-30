"""Entrypoint for parsing and preparing atomic-level structures with AtomWorks.

We provide three public functions to cover the main use cases:

* :py:func:`parse` — load a file and return a full result dictionary
  (``chain_info``, ``assemblies``, ``metadata``, etc.).
* :py:func:`parse_atom_array` — process an existing AtomArray and return a
  result dictionary matching :py:func:`parse` output.
* :py:func:`prepare_atom_array` — process an existing AtomArray with AtomWorks' common
  annotations and return the processed atoms directly, without the surrounding metadata.

To control the options for parsing and preparing, either:
(a) Use the ``config`` argument to pass a configuration object (e.g. :py:class:`~atomworks.io.config.ParseConfig`) or preset name (e.g. ``"rcsb"``).
(b) (Legacy) Pass bare keyword arguments to :py:func:`parse` (e.g. ``add_missing_atoms=True``)
"""

import contextlib
import hashlib
import io
import json
import logging
import os
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import biotite
import pandas as pd
from biotite.structure import AtomArray, AtomArrayStack

import atomworks
from atomworks.io._loaders import load_cif, load_pdb
from atomworks.io._pipeline import _assemble_parse_result, _maybe_promote_to_plus, _prepare_atom_array_or_stack
from atomworks.io.config import ParseConfig, PrepareConfig, _resolve_config, get_config
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack
from atomworks.io.utils.catcif import resolve_catcif_source
from atomworks.io.utils.ccd import (
    build_ccd_entries_from_cif_block,
    custom_ccd_residues,
    get_custom_ccd_entries,
    snapshot_custom_ccd_registry,
)
from atomworks.io.utils.extra_fields import ExtraFieldsType
from atomworks.io.utils.io_utils import infer_pdb_file_type
from atomworks.io.utils.standard_annotations.serialization import (
    _deserialize_standard_annotations,
    _handle_legacy_standard_annotations,
)

logger = logging.getLogger("atomworks.io")

__all__ = ["ParseConfig", "get_config", "parse", "parse_atom_array", "prepare_atom_array"]

STANDARD_PARSER_ARGS = get_config("rcsb").to_dict()
STANDARD_PARSER_ARGS["model"] = None
"""Common parser arguments for many biomolecular use cases (deprecated, use ``get_config("rcsb")``). Will be removed in a future version."""


def _restore_legacy_annotations_in_result(result: dict[str, Any]) -> None:
    """Restore legacy condition targets in each model's final assembly frame."""
    result["asym_unit"] = _maybe_promote_to_plus(result["asym_unit"])
    _handle_legacy_standard_annotations(result["asym_unit"])
    for assembly_id, assembly in result["assemblies"].items():
        assembly = _maybe_promote_to_plus(assembly)
        _handle_legacy_standard_annotations(assembly)
        result["assemblies"][assembly_id] = assembly


def _build_cache_file_path(
    cache_dir: Path,
    source: os.PathLike | io.StringIO | io.BytesIO | Callable,
    config: ParseConfig,
    cache_key: str | None = None,
) -> Path:
    """Key complete parse results by source, parsing options, and package versions.

    ``cache_key`` supplies an immutable source identity without reading or invoking
    ``source``; it must change when the source contents change. Otherwise, use the
    resolved file path or hash the buffer contents, type, and position.
    """
    options = config.to_dict()
    for name in ("cache_dir", "load_from_cache", "save_to_cache"):
        options.pop(name)
    identity = {
        "atomworks": atomworks.__version__,
        "biotite": biotite.__version__,
        "options": options,
    }
    if cache_key is not None:
        identity["cache_key"] = cache_key
        # The explicit identity replaces content hashing; updating the digest with empty bytes is a no-op.
        content = b""
    elif isinstance(source, io.StringIO | io.BytesIO):
        identity.update(buffer_type=type(source).__name__, position=source.tell())
        content = source.getvalue()
        content = content.encode("utf-8") if isinstance(content, str) else content
    else:
        identity["path"] = str(Path(source).resolve())
        content = b""
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode())
    digest.update(content)
    key = digest.hexdigest()
    return cache_dir / "parse-v2" / key[:2] / key[2:4] / f"{key}.pkl.zst"


def _make_cache_dirs_world_writable(path: Path) -> None:
    """Create ``path`` and any missing parents, each world-writable (``0o777``)."""
    newly_created = []
    cursor = path
    while not cursor.exists():
        newly_created.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in newly_created:
        with contextlib.suppress(PermissionError, FileNotFoundError):
            directory.chmod(0o777)


def _atomic_write_pickle(obj: Any, path: Path) -> None:
    """Write a zstd-3 pickle atomically so concurrent readers never see a partial file."""
    _make_cache_dirs_world_writable(path.parent)
    # Keep the codec suffix; same-directory replacement is atomic.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.tmp.", suffix=path.suffix)
    os.close(fd)
    try:
        pd.to_pickle(obj, tmp_name, compression={"method": "zstd", "level": 3})
        os.chmod(tmp_name, 0o666)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)  # don't leave a partial temp behind on failure
        raise


def _attach_live_ccd_registry(results: list[dict[str, Any]]) -> None:
    """Snapshot the currently-scoped custom CCD registry and attach it to each result's ``asym_unit``."""
    ccd_registry = snapshot_custom_ccd_registry()

    if not ccd_registry:
        # No custom CCD entries currently registered
        return

    # Attach to all AtomArrayPlus-type objects
    for r in results:
        if isinstance(r["asym_unit"], AtomArrayPlus | AtomArrayPlusStack):
            r["asym_unit"]._custom_ccd_registry = ccd_registry
        for assembly in r["assemblies"].values():
            if isinstance(assembly, AtomArrayPlus | AtomArrayPlusStack):
                assembly._custom_ccd_registry = ccd_registry


def parse(
    source: os.PathLike | io.StringIO | io.BytesIO | Callable[[], io.StringIO | io.BytesIO] | None = None,
    *,
    config: str | ParseConfig | None = None,
    filename: os.PathLike | io.StringIO | io.BytesIO | None = None,
    cache_key: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse structural files into an AtomArrayStack with standardized annotations and metadata.

    Processing behaviour is controlled by :py:class:`~atomworks.io.config.ParseConfig`.
    Legacy bare keyword arguments (e.g. ``add_missing_atoms=True``) are still accepted
    but will emit a :py:class:`DeprecationWarning`; prefer passing a ``config`` object.

    Args:
        source: Path or buffer to the structure file. May be any format of
            atomic-level structure (e.g. .cif, .bcif, .pdb), optionally compressed
            with gzip (.gz/.gzip) or Zstandard (.zst),
            although .cif files are strongly recommended. A callable returning a fresh buffer
            is evaluated only on cache misses and requires ``cache_key`` and ``config.file_type``.
        config: Processing configuration. Pass a preset name (``"default"``,
            ``"rcsb"``, ``"lightweight"``, ``"minimal"``), a
            :py:class:`~atomworks.io.config.ParseConfig` instance, or ``None``
            for defaults.
        filename: Deprecated alias for ``source``. Cannot be used together with ``source``.
        cache_key: Immutable source identity for deferred reads, replacing content hashing.
            Must change whenever source contents change; parser settings and package versions
            are included automatically. Cached results still honor the config's cache flags.

    Returns:
        dict: A dictionary containing the following keys:
            chain_info
                A dictionary mapping chain ID to sequence, type (as an IntEnum), RCSB entity,
                EC number, and other information.
            ligand_info
                A dictionary containing ligand of interest information.
            asym_unit
                An AtomArrayStack instance representing the asymmetric unit.
            assemblies
                A dictionary mapping assembly IDs to AtomArrayStack instances.
            metadata
                A dictionary containing metadata about the structure
                (e.g., resolution, deposition date, etc.).
            extra_info
                A dictionary with information for cross-compatibility and caching.
                Should typically not be used directly.
    """
    config, kwargs = _resolve_config(config, kwargs, cls=ParseConfig)

    if filename is not None:
        if source is not None:
            raise TypeError("Cannot pass both 'source' and 'filename'")
        warnings.warn(
            "The 'filename' parameter is deprecated; use 'source' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        source = filename
    if source is None:
        raise TypeError("parse() requires a source (file path or buffer)")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
    if callable(source) and (not cache_key or not config.file_type):
        raise ValueError("Deferred sources require cache_key and config.file_type")

    # Resolve catcif paths before any file-type inference or caching logic
    source, catcif_file_type = resolve_catcif_source(source)

    # Dispatch: file input
    file_type = config.file_type or catcif_file_type or infer_pdb_file_type(source)
    is_buffer = isinstance(source, io.StringIO | io.BytesIO)
    build_assembly = config.build_assembly
    extra_fields = config.extra_fields

    cache_file_path = None
    if (
        (config.load_from_cache or config.save_to_cache)
        and not get_custom_ccd_entries()
        and not (config.altloc.startswith("random") and config.altloc_seed is None)
    ):
        cache_file_path = _build_cache_file_path(
            Path(config.cache_dir), source, config.replace(file_type=file_type), cache_key
        )
        if config.load_from_cache and cache_file_path.exists():
            try:
                result, position = pd.read_pickle(cache_file_path)
                if is_buffer:
                    source.seek(position)
                if config.load_standard_annotations:
                    for entry in result if isinstance(result, list) else [result]:
                        _restore_legacy_annotations_in_result(entry)
                return result
            except Exception as exc:
                raise RuntimeError(f"Error loading parse cache: {cache_file_path}") from exc

    # +------ Uncached route: loading and processing ------+
    if callable(source):
        source = source()
        is_buffer = isinstance(source, io.StringIO | io.BytesIO)
    cif_block = None
    model_ids = None
    if file_type == "pdb":
        if config.load_standard_annotations:
            raise ValueError("load_standard_annotations=True is not supported for PDB files. Use CIF format instead.")
        if config.altloc != "first":
            raise ValueError(
                f"altloc='{config.altloc}' is not supported for PDB files. "
                "PDB parsing always uses altloc='first'. Use CIF format for altloc selection."
            )
        atoms, metadata = load_pdb(source, model=config.model)
        extra_fields = None
        if build_assembly not in ("all", None):
            logger.warning(
                "PDB files always build all assemblies; ignoring build_assembly=%r",
                build_assembly,
            )
        build_assembly = "all"
    elif file_type in ("cif", "bcif", "mmjson"):
        atoms, _cif_file, cif_block, metadata, model_ids = load_cif(
            source,
            file_type=file_type,
            model=config.model,
            extra_fields=extra_fields,
            load_standard_annotations=config.load_standard_annotations,
            altloc=config.altloc,
            altloc_seed=config.altloc_seed,
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")

    # Build CCD entries from the CIF's chem_comp* categories and scope them
    # to this parse call so all downstream lookups (types, bonds, templates)
    # go through the registry.
    cif_ccd_entries = (
        build_ccd_entries_from_cif_block(cif_block, on_mismatch=config.cif_ccd_on_mismatch)
        if cif_block is not None
        else {}
    )
    ctx = custom_ccd_residues(cif_ccd_entries) if cif_ccd_entries else contextlib.nullcontext()

    with ctx:
        multiple_models = isinstance(atoms, list)
        models = (
            zip(atoms, model_ids, strict=False) if multiple_models else [(atoms, model_ids[0] if model_ids else None)]
        )
        results = []
        for arr, model_num in models:
            arr, chain_info = _prepare_atom_array_or_stack(
                arr,
                cif_block=cif_block,
                config=config,
                extra_fields=extra_fields,
            )
            if config.load_standard_annotations:
                arr = _deserialize_standard_annotations(arr, cif_block, model_num=model_num, restore_legacy=False)
            results.append(
                _assemble_parse_result(
                    atoms=arr,
                    chain_info=chain_info,
                    cif_block=cif_block,
                    config=config,
                    build_assembly=build_assembly,
                    metadata=metadata,
                    keep_cif_block=config.keep_cif_block,
                )
            )
        _attach_live_ccd_registry(results)
        result = results if multiple_models else results[0]

    if config.load_standard_annotations:
        for entry in results:
            _restore_legacy_annotations_in_result(entry)

    if config.save_to_cache and cache_file_path is not None:
        for entry in results:
            entry.setdefault("metadata", {}).update(
                {"parse_arguments": config.to_dict(), "atomworks.version": atomworks.__version__}
            )
        _atomic_write_pickle((result, source.tell() if is_buffer else None), cache_file_path)

    return result


def prepare_atom_array(
    source: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    *,
    config: str | PrepareConfig | None = None,
    cif_block: Any | None = None,
    extra_fields: ExtraFieldsType = None,
) -> AtomArray | AtomArrayStack:
    """Perform standard AtomWorks preparation of an AtomArray, returning the processed atoms directly.

    Runs the same core processing as :py:func:`parse` (standardize, add missing
    atoms, infer bonds, annotate) but returns the processed atoms instead of a
    full result dictionary. Single-model results are squeezed to
    :py:class:`~biotite.structure.AtomArray` for conciseness.

    For example, useful to add entity/molecule annotations to an AtomArray that already has a complete set of atoms and bonds.

    Args:
        source: The structure to process.
        config: Preset name (``"default"``, ``"rcsb"``, ``"lightweight"``),
            :py:class:`~atomworks.io.config.PrepareConfig`, or ``None`` for defaults.
        cif_block: Optional CIF block for richer processing (struct_conn bonds,
            entity categories, custom bonds).
        extra_fields: Extra CIF fields to preserve through processing.
    """
    config, _ = _resolve_config(config, {})
    atoms, _ = _prepare_atom_array_or_stack(
        source,
        config=config,
        cif_block=cif_block,
        extra_fields=extra_fields,
    )

    if config.return_atom_array_plus:
        atoms = _maybe_promote_to_plus(atoms)

    if isinstance(atoms, AtomArrayStack) and atoms.stack_depth() == 1:
        return atoms[0]

    return atoms


def parse_atom_array(
    source: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    *,
    config: str | PrepareConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Mimic of :py:func:`parse` that operates on an AtomArray or AtomArrayStack instead of a file input.

    Returns identical result dict to :py:func:`parse`, but takes an AtomArray or AtomArrayStack as input instead of a file path or buffer.

    Args:
        source: The AtomArray or AtomArrayStack to process.
        config: Preset name (``"default"``, ``"rcsb"``, ``"lightweight"``),
            :py:class:`~atomworks.io.config.PrepareConfig`, or ``None`` for defaults.
    """
    config, kwargs = _resolve_config(config, kwargs)

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    atoms, chain_info = _prepare_atom_array_or_stack(
        source,
        cif_block=None,
        config=config,
    )

    return _assemble_parse_result(
        atoms=atoms,
        chain_info=chain_info,
        cif_block=None,
        config=config,
        build_assembly="all",
        metadata={},
    )
