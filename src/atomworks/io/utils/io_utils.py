"""General utility functions for working with CIF files in Biotite."""

__all__ = [
    "CIFWriteConfig",
    "apply_sharding_pattern",
    "build_sharding_pattern",
    "get_structure",
    "load_any",
    "parse_sharding_pattern",
    "read_any",
    "suppress_logging_messages",
    "to_cif_buffer",
    "to_cif_file",
    "to_cif_string",
    "to_pdb_buffer",
    "to_pdb_string",
]
import contextlib
import io
import json
import logging
import os
import random
import re
import warnings
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import biotite.structure.io.pdb as biotite_pdb
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import mol, pdbx
from biotite.structure.io.pdbx import CIFColumn
from biotite.structure.io.pdbx.component import MaskValue
from biotite.structure.io.pdbx.convert import COMP_BOND_TYPE_TO_ORDER

import atomworks.io.transforms.atom_array as ta  # to avoid circular import
from atomworks.common import exists
from atomworks.constants import (
    ALTLOC_DEFAULT_IDS,
    ATOMIC_NUMBER_TO_ELEMENT,
    STANDARD_CIF_ANNOTATIONS,
    STANDARD_POLYMER_RESIDUES,
)
from atomworks.io.utils.altloc import has_multiple_altlocs_per_residue, select_altlocs_clash_aware
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack
from atomworks.io.utils.bonds import remap_intra_residue_coordination_bonds
from atomworks.io.utils.catcif import resolve_catcif_source
from atomworks.io.utils.ccd import atom_array_from_ccd_code, custom_ccd_residues
from atomworks.io.utils.chain_info import build_chain_info
from atomworks.io.utils.compression import is_compressed_file, open_compressed
from atomworks.io.utils.extra_fields import (
    ExtraFieldsType,
    apply_extra_field_dtypes,
    filter_extra_fields,
    normalize_extra_fields,
)
from atomworks.io.utils.selection import get_annotation, get_residue_starts
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from atomworks.io.utils.standard_annotations.serialization import (
    _LEGACY_MASK_FIELDS,
    _canonicalize_standard_annotation_aliases,
    _get_scalar_1body_sa_names,
    _serialize_standard_annotation_categories,
)
from atomworks.io.utils.testing import has_ambiguous_annotation_set

logger = logging.getLogger("atomworks.io")


CIF_LIKE_EXTENSIONS = {
    base + compression for base in (".cif", ".pdb", ".bcif", ".mmjson") for compression in ("", ".gz", ".gzip", ".zst")
}
CIFFileType = Literal["cif", "bcif", "cif.gz", "bcif.gz", "cif.gzip", "bcif.gzip", "cif.zst", "bcif.zst"]

# Solution text for chain disambiguation methods
CHAIN_DISAMBIGUATION_SOLUTIONS = """To resolve, choose a disambiguation method:
  (1) chain_disambiguation='transformation_id' (AtomWorks-only, preserves chain_id)
  (2) chain_disambiguation='chain_iid' (universal compatibility, modifies chain_id)"""


@contextmanager
def suppress_logging_messages(logger_name: str, message_pattern: str) -> Generator[None, None, None]:
    """Temporarily suppress logging messages matching a pattern.

    Args:
        logger_name: Name of the logger to filter.
        message_pattern: String pattern to match in log messages (substring match).

    Examples:
        >>> with suppress_logging_messages("atomworks.io", "not found"):
        ...     # Code that generates "not found" warnings
        ...     pass
    """
    target_logger = logging.getLogger(logger_name)

    def filter_func(record: logging.LogRecord) -> bool:
        return message_pattern not in record.getMessage()

    target_logger.addFilter(filter_func)
    try:
        yield
    finally:
        target_logger.removeFilter(filter_func)


def _get_logged_in_user() -> str:
    """Get the logged in user.

    Returns:
        The username of the logged in user, or "unknown_user" if unavailable.
    """
    try:
        return os.getlogin()
    except OSError:
        return "unknown_user"


@dataclass
class CIFWriteConfig:
    """Configuration for CIF/BCIF writing operations.

    Consolidates options used when writing structures to CIF/BCIF format.

    Attributes:
        id: Entry ID for the data block. Defaults to "unknown_id".
        author: Author name for metadata. Defaults to current user.
        date: Date string (YYYY-MM-DD format). Auto-generated if None.
        time: Time string (HH:MM:SS format). Auto-generated if None.
        include_entity_categories: Write entity categories (_entity, _entity_poly, etc.).
            Defaults to False for buffer/string operations, True for file operations.
        include_chem_comp: Write _chem_comp CIF category. Defaults to True.
        include_nan_coords: Include atoms with NaN coordinates. Defaults to True.
        extra_fields: Additional AtomArray annotations to write. Use "all" to include all
            non-standard annotations. Defaults to empty list.
        extra_categories: Additional CIF categories as dict of {category_name: {column_name: value}}.
            Defaults to None.
        chain_disambiguation: Method for disambiguating chains in multi-transformation assemblies.

            - "transformation_id": Stores transformation_id in struct_conn symmetry fields.
              Preserves original chain_id. Requires AtomWorks to read.
            - "chain_iid": Writes chain_iid as chain_id to make chains unique.
              Better compatibility with external software.
            - None: No disambiguation. Raises error if chains are ambiguous.

            Defaults to None.
        save_standard_annotations: Whether to serialize registered
            :py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase` annotations
            (conditions, ``atomize``, etc.) present on the structure into custom CIF
            categories. Defaults to True.
        ccd_entries: Optional dict mapping residue name to CCD template AtomArray.
            When provided, these templates are used for ``chem_comp_atom`` and
            ``chem_comp_bond`` emission. Falls back to the structure's
            ``_custom_ccd_registry`` attribute (if available, only for AtomArrayPlus types),
            then bundled CCD.
        chem_comp_source: Source of ``chem_comp_atom`` / ``chem_comp_bond`` rows:

            - ``"array"`` — always the AtomArray itself (regardless of where it came from)
            - ``"ccd"`` — always the CCD we have access to (AtomArray fallback if no template).
            - ``"ccd_non_canonicals"`` — AtomArray for canonical residues, CCD for the rest.

        warn_on_ccd_without_registry: Warn on CCD write from a plain ``AtomArray``
            (no parse-time CCD snapshot). Set ``False`` to silence. Defaults to ``True``.

    Note:
        ``atom_site`` always includes an ``atom_array_index`` column (0, 1, …, n-1).

    See Also:
        :py:func:`to_cif_buffer`
        :py:func:`to_cif_string`
        :py:func:`to_cif_file`
    """

    id: str = "unknown_id"
    author: str = field(default_factory=_get_logged_in_user)
    date: str | None = None
    time: str | None = None
    include_entity_categories: bool = False
    include_nan_coords: bool = True
    include_chem_comp: bool = True
    extra_fields: list[str] | Literal["all"] = field(default_factory=list)
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None
    chain_disambiguation: Literal["transformation_id", "chain_iid"] | None = None
    save_standard_annotations: bool = True
    ccd_entries: dict[str, struc.AtomArray] | None = None
    chem_comp_source: Literal["array", "ccd", "ccd_non_canonicals"] = "ccd"
    warn_on_ccd_without_registry: bool = True


def load_any(
    file_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif", "catcif", "mmjson"] | None = None,
    *,
    extra_fields: ExtraFieldsType = None,
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "random_per_chain", "random_clash_aware"] | str = "first",
    altloc_seed: int | None = None,
    infer_dtypes: bool = True,
) -> AtomArrayStack | AtomArrayPlusStack | AtomArray | AtomArrayPlus:
    """Convenience function for loading a structure from a file or buffer.

    NOTE: This function is deprecated! Please use :py:func:`atomworks.io.parse`
    with `config="minimal"`

    By default, bonds are NOT loaded here - they are added later in
    :py:func:`~atomworks.io.parser.parse_atom_array` using CCD templates for consistency.

    Args:
        file_or_buffer: Path to the file or buffer to load the structure from.
        file_type: Type of the file to load. If None, it will be inferred.
        extra_fields: Extra fields to include as AtomArray annotations. Can be:

            - List of field names: ``["annot1", "annot2"]``
            - Dict with specs: ``{"my_field": {"default": 0, "dtype": np.int32}}``
            - ``"all"``: Include all fields from the atom_site category
        include_bonds: Whether to include bond information from biotite's parser.
            Defaults to False. Bonds are typically added later in parse_atom_array()
            using CCD templates for consistency.
        model: The model number to use for loading the structure. If None, all models
            will be loaded.
        altloc: The altloc ID to use for loading the structure.
        altloc_seed: Seed for the random number generator when ``altloc="random_per_chain"``
            or ``altloc="random_clash_aware"``.
        infer_dtypes: Whether to automatically infer dtypes for extra fields from their values.
            For example, fields containing "1", "2", "3" will be converted to int.

    Returns:
        The loaded structure.

    References:
        `Biotite Structure I/O <https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure>`_
        `mmCIF Format Specification <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/>`_
    """

    warnings.warn(
        "load_any() is deprecated and will be removed in a future release. "
        "Please use atomworks.io.parse() with config='minimal' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    file_obj = read_any(file_or_buffer, file_type=file_type)
    atom_array = get_structure(
        file_obj,
        extra_fields=extra_fields,
        include_bonds=include_bonds,
        model=model,
        altloc=altloc,
        altloc_seed=altloc_seed,
        infer_dtypes=infer_dtypes,
    )

    # Ensure chem_comp_type is always present
    ensure_annotations(atom_array, "chem_comp_type")

    return atom_array


def _validate_label_seq_ids(cif_block: pdbx.CIFBlock | pdbx.BinaryCIFBlock) -> None:
    """Reject inconsistent CIF entity and polymer sequence identity."""
    atom_site = cif_block["atom_site"]
    if not {"label_asym_id", "label_seq_id"}.issubset(atom_site):
        return

    asym_ids = atom_site["label_asym_id"].as_array(str)
    seq_ids = atom_site["label_seq_id"].as_array(str)
    missing = np.isin(seq_ids, (".", "?"))
    entity_ids = atom_site["label_entity_id"].as_array(str) if "label_entity_id" in atom_site else None

    for asym_id in np.unique(asym_ids):
        mask = asym_ids == asym_id
        if missing[mask].any() and (~missing[mask]).any():
            raise ValueError(
                f"CIF label_asym_id '{asym_id}' mixes defined and missing label_seq_id values. "
                "Polymer members, including noncanonical monomers, require label_seq_id; standalone "
                "non-polymers must use a separate label_asym_id."
            )
        if entity_ids is not None and len(np.unique(entity_ids[mask])) != 1:
            raise ValueError(f"CIF label_asym_id '{asym_id}' maps to multiple label_entity_id values.")

    entity = cif_block.get("entity")
    if entity_ids is None or entity is None or not {"id", "type"}.issubset(entity):
        return
    entity_types = dict(zip(entity["id"].as_array(str), entity["type"].as_array(str), strict=True))
    invalid_polymer = missing & np.array(
        [entity_types.get(entity_id, "").strip().lower() == "polymer" for entity_id in entity_ids]
    )
    if invalid_polymer.any():
        asym_id = asym_ids[np.flatnonzero(invalid_polymer)[0]]
        raise ValueError(
            f"CIF polymer label_asym_id '{asym_id}' has missing label_seq_id values. Every polymer member, "
            "including noncanonical monomers, must have a sequence ID."
        )


def get_structure(
    file_obj: pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile | pdbx.CIFBlock,
    *,
    extra_fields: ExtraFieldsType = None,
    include_bonds: bool = False,
    model: int | None = None,
    altloc: Literal["first", "random_per_chain", "random_clash_aware"] | str = "first",
    altloc_seed: int | None = None,
    infer_dtypes: bool = True,
) -> AtomArrayStack | AtomArray:
    """Load structure into Biotite's AtomArray or AtomArrayStack.

    By default, bonds are NOT loaded here - they are added later in
    :py:func:`~atomworks.io.parser.parse_atom_array` using CCD templates for consistency.

    Note:
        For CIF files, ``transformation_id`` is automatically included. This field is
        required for assembly processing.

    Args:
        file_obj: The file object to load with Biotite.
        extra_fields: Extra fields to include as AtomArray annotations. Can be:

            - List of field names: ``["b_factor", "occupancy"]``
            - Dict with specs: ``{"my_field": {"default": 0, "dtype": np.int32}}``
            - ``"all"``: Include all fields from the atom_site category
        include_bonds: Whether to include bond information. Passed to biotite's parser.
            Defaults to False.
        model: The model number to use for loading the structure.
        altloc (Literal["first", "random_per_chain", "random_clash_aware"] | str): The altloc ID to use
            for loading the structure.
            ``"first"`` selects the alphabetically-first altloc letter per chain.
            ``"random_per_chain"`` selects a random altloc independently for each chain.
            ``"random_clash_aware"`` selects one altloc per residue, avoiding steric clashes between
            nearby residues via constraint satisfaction.
            A specific altloc ID (e.g., ``"A"``, ``"B"``) can also be provided.
            If a specific altloc ID is not present in the file, an error will be raised.
        altloc_seed: Seed for the random number generator when ``altloc="random_per_chain"`` or
            ``altloc="random_clash_aware"``.
            If ``None`` (default), selection is non-deterministic. Provide an integer
            for reproducible results.
        infer_dtypes: Whether to automatically infer dtypes for extra fields from their
            values.

    Returns:
        The loaded structure with bonds from the CIF file.

    Reference:
        `Biotite documentation <https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure>`_
    """
    # Normalize extra_fields and extract field names for biotite
    extra_field_specs = normalize_extra_fields(extra_fields) if extra_fields not in (None, "all") else {}
    extra_field_names = "all" if extra_fields == "all" else list(extra_field_specs.keys())

    # For CIF files, auto-inject transformation_id into extra fields (which we need for assemblies)
    if (
        isinstance(file_obj, pdbx.CIFFile | pdbx.BinaryCIFFile | pdbx.CIFBlock)
        and extra_field_names != "all"
        and "transformation_id" not in extra_field_names
    ):
        extra_field_names.append("transformation_id")
        if "transformation_id" not in extra_field_specs:
            extra_field_specs["transformation_id"] = {}

    cif_block = None
    match type(file_obj):
        case pdbx.CIFFile | pdbx.BinaryCIFFile | pdbx.CIFBlock:
            # Filter extra annotations to fields that are actually present in the file
            if not isinstance(file_obj, pdbx.CIFBlock):
                cif_block = file_obj.block
            else:
                cif_block = file_obj
            _validate_label_seq_ids(cif_block)
            atom_site = cif_block["atom_site"]
            if extra_field_names == "all":
                extra_field_names = list(atom_site.keys())
            extra_field_names = filter_extra_fields(extra_field_names, atom_site)
            # Biotite requires label_alt_id for altloc="all", even without alternates.
            load_altlocs = altloc if include_bonds else "all"
            if "label_alt_id" not in atom_site:
                load_altlocs = "first"
            atom_array_stack = pdbx.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_field_names,
                use_author_fields=False,
                altloc=load_altlocs,
                include_bonds=include_bonds,
            )

            # Normalize insertion codes to empty string for CIF files
            # (ins_code is a legacy PDB artifact; CIF files typically don't use it meaningfully)
            if "ins_code" in atom_array_stack.get_annotation_categories():
                atom_array_stack.ins_code[:] = ""

        case biotite_pdb.PDBFile:
            atom_array_stack = biotite_pdb.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_field_names if extra_field_names != "all" else [],
                altloc="all" if not include_bonds else altloc,
                include_bonds=include_bonds,
            )
        case _:
            raise ValueError(f"Unsupported file type: {type(file_obj)}. Must be a CIFFile, BinaryCIFFile, or PDBFile.")

    # Apply altloc filtering (only when we loaded all altlocs; when include_bonds=True, Biotite already filtered altlocs at load time)
    if not include_bonds:
        if altloc == "random_clash_aware":
            atom_array_stack = select_altlocs_clash_aware(atom_array_stack, seed=altloc_seed, cif_block=cif_block)
        else:
            altloc_ids = get_annotation(atom_array_stack, "altloc_id")
            if altloc_ids is not None:
                defaults = np.isin(altloc_ids, ALTLOC_DEFAULT_IDS)

                if altloc in ("first", "random_per_chain"):
                    # Keep one altloc per chain: first observed ("first") or random ("random_per_chain")
                    mask = defaults.copy()
                    chain_ids = get_annotation(atom_array_stack, "chain_id")
                    rng = random.Random(altloc_seed) if altloc == "random_per_chain" else None
                    for chain_id in np.unique(chain_ids):
                        chain_mask = chain_ids == chain_id
                        letters = [a for a in np.unique(altloc_ids[chain_mask]) if a.isalpha()]
                        if letters:
                            chosen = rng.choice(letters) if rng else letters[0]
                            mask |= chain_mask & (altloc_ids == chosen)
                else:
                    # Keep global altloc
                    if altloc not in altloc_ids:
                        available = [a for a in np.unique(altloc_ids) if a.isalpha()]
                        raise ValueError(f"Altloc '{altloc}' not found. Available: {available}")
                    mask = defaults | (altloc_ids == altloc)

                atom_array_stack = atom_array_stack[..., mask]

    # Ensure charge is always present; some files (e.g. AF3 outputs) omit pdbx_formal_charge
    if "charge" not in atom_array_stack.get_annotation_categories():
        atom_array_stack.set_annotation("charge", np.zeros(atom_array_stack.array_length(), dtype=np.int8))

    # Apply dtype inference/conversion for extra fields
    if infer_dtypes and extra_field_specs:
        atom_array_stack = apply_extra_field_dtypes(atom_array_stack, extra_field_specs)
    elif infer_dtypes and extra_field_names:
        # Auto-infer dtypes even without explicit specs
        atom_array_stack = apply_extra_field_dtypes(atom_array_stack, {}, extra_field_names)

    return atom_array_stack


def infer_pdb_file_type(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
) -> Literal["cif", "pdb", "bcif", "sdf", "mmjson"]:
    """
    Infer the file type of a PDB file or buffer.
    """
    # Convert string paths to Path objects
    if isinstance(path_or_buffer, str | os.PathLike):
        path_or_buffer = Path(path_or_buffer)

    # Determine file type and open context
    if isinstance(path_or_buffer, io.BytesIO):
        return "bcif"
    elif isinstance(path_or_buffer, io.StringIO):
        # ... if second line starts with '#', it is very likely a cif file
        path_or_buffer.seek(0)
        path_or_buffer.readline()  # Skip the first line
        second_line = path_or_buffer.readline().strip()
        path_or_buffer.seek(0)
        return "cif" if second_line.startswith("#") else "pdb"
    elif isinstance(path_or_buffer, Path):
        if is_compressed_file(path_or_buffer):
            path_or_buffer = path_or_buffer.with_suffix("")
        inferred_file_type = path_or_buffer.suffix.lstrip(".").lower()
    else:
        raise TypeError(f"Expected a path, StringIO, or BytesIO, got {type(path_or_buffer).__name__}")

    # Canonicalize the file type
    if inferred_file_type in ("cif", "mmcif", "pdbx"):
        return "cif"
    elif inferred_file_type in ("pdb", "pdb1", "ent"):
        return "pdb"
    elif inferred_file_type == "bcif":
        return "bcif"
    elif inferred_file_type == "sdf":
        return "sdf"
    elif inferred_file_type == "mmjson":
        return "mmjson"
    else:
        raise ValueError(f"Unsupported file type: {inferred_file_type}")


def _read_mmjson(stream: io.TextIOBase | io.BytesIO) -> pdbx.CIFFile:
    """Read mmJSON categories into Biotite's CIF representation.

    JSON nulls collapse CIF's unknown and inapplicable markers, so restore the
    inapplicable marker for the two ``atom_site`` fields that require it.
    """
    data = json.load(stream)
    cif_file = pdbx.CIFFile()
    for block_name, block_data in data.items():
        cif_block = pdbx.CIFBlock()
        for category_name, category_data in block_data.items():
            category = pdbx.CIFCategory()
            for column_name, values in category_data.items():
                missing = (
                    "." if category_name == "atom_site" and column_name in ("label_alt_id", "label_seq_id") else "?"
                )
                category[column_name] = pdbx.CIFColumn(
                    [str(value) if value is not None else missing for value in values]
                )
            cif_block[category_name] = category
        cif_file[block_name] = cif_block
    return cif_file


def read_any(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "pdb", "bcif", "sdf", "catcif", "mmjson"] | None = None,
) -> pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile:
    """
    Reads any of the allowed file types into the appropriate Biotite file object.

    Args:
        path_or_buffer (PathLike | io.StringIO | io.BytesIO): The path to the file or a buffer to read from.
            If a buffer, it's highly recommended to specify the file_type.
            Paths may be plain, gzip-compressed (``.gz``/``.gzip``), or Zstandard-compressed (``.zst``).
        file_type: Type of the file, including ``"mmjson"``.
            If None, it will be inferred from the file extension. When using a buffer, the file type must be specified.

    Returns:
        pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile: The loaded file object.

    Raises:
        ValueError: If the file type is unsupported or cannot be determined.
    """
    # Resolve catcif paths before any file-type inference
    resolved, catcif_file_type = resolve_catcif_source(path_or_buffer)
    if catcif_file_type is not None:
        if file_type not in (None, "catcif"):
            raise ValueError(f"catcif path passed but file_type doesn't agree: {file_type}")
        path_or_buffer, file_type = resolved, catcif_file_type

    if file_type == "catcif" and catcif_file_type is None:
        raise ValueError("file_type='catcif' requires a path containing '.catcif:' (e.g. 'archive.catcif:entry_id')")

    # Determine file type
    if file_type is None:
        file_type = infer_pdb_file_type(path_or_buffer)

    open_mode = "rb" if file_type == "bcif" else "rt"

    # Select the appropriate file reader.
    if file_type == "cif":
        reader = pdbx.CIFFile.read
    elif file_type == "pdb":
        reader = biotite_pdb.PDBFile.read
    elif file_type == "bcif":
        reader = pdbx.BinaryCIFFile.read
    elif file_type == "sdf":
        reader = mol.SDFile.read
    elif file_type == "mmjson":
        reader = _read_mmjson
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Load the file content
    if isinstance(path_or_buffer, str | os.PathLike):
        with open_compressed(path_or_buffer, open_mode) as stream:
            return reader(stream)
    return reader(path_or_buffer)


def _rows_to_columns(
    rows: list[dict],
    *,
    on_key_mismatch: Literal["raise", "intersection", "union"] = "raise",
) -> dict[str, list]:
    """Convert list of row dicts to dict of column lists.

    Args:
        rows: List of dictionaries, each representing a row.
        on_key_mismatch: How to handle rows with differing key sets.

            * ``"raise"`` (default): require all rows to have identical keys;
              raise :class:`ValueError` otherwise.
            * ``"intersection"``: emit only columns present in every row;
              column order follows ``rows[0]``.
            * ``"union"``: emit the union of columns (first-seen order);
              missing values are filled with ``"?"``.

    Returns:
        Dictionary mapping column names to lists of values.

    Examples:
        >>> rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        >>> _rows_to_columns(rows)
        {'a': [1, 3], 'b': [2, 4]}
    """
    if not rows:
        return {}

    if on_key_mismatch == "raise":
        first_keys = rows[0].keys()
        for i, row in enumerate(rows):
            if row.keys() != first_keys:
                raise ValueError(f"Row {i} keys {set(row)} differ from row 0 keys {set(first_keys)}")
        return {key: [row[key] for row in rows] for key in rows[0]}

    if on_key_mismatch == "intersection":
        common = set(rows[0]).intersection(*(row.keys() for row in rows[1:]))
        return {key: [row[key] for row in rows] for key in rows[0] if key in common}

    # union
    all_keys = list(dict.fromkeys(key for row in rows for key in row))
    return {key: [row.get(key, "?") for row in rows] for key in all_keys}


def _template_to_chem_comp_atom_rows(comp_id: str, template: struc.AtomArray) -> list[dict]:
    """Convert a CCD template AtomArray to chem_comp_atom rows."""
    annotations = template.get_annotation_categories()
    has_charge = "charge" in annotations
    has_aromatic = "is_aromatic" in annotations
    has_leaving = "is_leaving_atom" in annotations

    rows = []
    for i in range(len(template)):
        rows.append(
            {
                "comp_id": comp_id,
                "atom_id": template.atom_name[i],
                "type_symbol": template.element[i],
                "charge": int(template.charge[i]) if has_charge else 0,
                "pdbx_aromatic_flag": ("Y" if template.is_aromatic[i] else "N") if has_aromatic else "N",
                "pdbx_leaving_atom_flag": ("Y" if template.is_leaving_atom[i] else "N") if has_leaving else "N",
                "pdbx_model_Cartn_x_ideal": float(template.coord[i, 0]),
                "pdbx_model_Cartn_y_ideal": float(template.coord[i, 1]),
                "pdbx_model_Cartn_z_ideal": float(template.coord[i, 2]),
            }
        )

    return rows


def _template_to_chem_comp_bond_rows(comp_id: str, template: struc.AtomArray) -> list[dict]:
    """Convert a CCD template AtomArray's bonds to chem_comp_bond rows."""
    if template.bonds is None:
        return []

    rows = []
    for idx1, idx2, bond_type_int in template.bonds.as_array():
        bond_type = struc.BondType(bond_type_int)
        order, aromatic = COMP_BOND_TYPE_TO_ORDER.get(bond_type, ("SING", "N"))
        rows.append(
            {
                "comp_id": comp_id,
                "atom_id_1": template.atom_name[idx1],
                "atom_id_2": template.atom_name[idx2],
                "value_order": order,
                "pdbx_aromatic_flag": aromatic,
            }
        )

    return rows


def _atom_array_chem_comp_bond_rows(atom_array: struc.AtomArray, res_names: set[str]) -> list[dict]:
    """Emit chem_comp_bond rows for ``res_names`` from ``atom_array.bonds``.

    Delegates to biotite's :py:func:`_set_intra_residue_bonds` on the subset
    of atoms in ``res_names``.
    """
    if not res_names:
        return []

    subset = atom_array[np.isin(atom_array.res_name, np.asarray(sorted(res_names)))]
    if subset.bonds is None or len(subset.bonds.as_array()) == 0:
        return []

    # _build_chem_comp_bond receives structures[0] (unpreprocessed), not s, so the upstream
    # remap_intra_residue_coordination_bonds hasn't run on this input. Required because
    # biotite's _set_intra_residue_bonds KeyErrors on COORDINATION bonds.
    subset.bonds.convert_bond_type(struc.BondType.COORDINATION, struc.BondType.SINGLE)

    category = pdbx.convert._set_intra_residue_bonds(subset, pdbx.CIFCategory())
    if category is None:
        return []

    # Drop columns that are re-derived per call and/or are missing
    cols = {k: category[k].as_array() for k in category if k not in ("pdbx_ordinal", "pdbx_stereo_config")}
    n = len(next(iter(cols.values()))) if cols else 0

    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


def _atom_array_chem_comp_atom_rows(atom_array: struc.AtomArray, res_name: str) -> list[dict]:
    """Emit chem_comp_atom rows for ``res_name`` from its first-occurrence residue slice."""
    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    res_names_at_starts = atom_array.res_name[res_starts[:-1]]
    first = np.where(res_names_at_starts == res_name)[0]
    if len(first) == 0:
        return []
    i = first[0]
    sl = atom_array[res_starts[i] : res_starts[i + 1]]

    annotations = sl.get_annotation_categories()
    has_charge = "charge" in annotations
    has_aromatic = "is_aromatic" in annotations
    has_leaving = "is_leaving_atom" in annotations

    rows = []
    for k in range(len(sl)):
        rows.append(
            {
                "comp_id": res_name,
                "atom_id": str(sl.atom_name[k]),
                "type_symbol": str(sl.element[k]),
                "charge": int(sl.charge[k]) if has_charge else 0,
                "pdbx_aromatic_flag": ("Y" if sl.is_aromatic[k] else "N") if has_aromatic else "N",
                "pdbx_leaving_atom_flag": ("Y" if sl.is_leaving_atom[k] else "N") if has_leaving else "N",
                "pdbx_model_Cartn_x_ideal": float(sl.coord[k, 0]),
                "pdbx_model_Cartn_y_ideal": float(sl.coord[k, 1]),
                "pdbx_model_Cartn_z_ideal": float(sl.coord[k, 2]),
            }
        )
    return rows


def _build_chem_comp(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """Build the chem_comp category for a CIF file.

    Args:
        atom_array: AtomArray containing structure data with polymer chain information.
    """
    if isinstance(atom_array, struc.AtomArrayStack):
        atom_array = atom_array[0]  # Choose any model; annotations are shared across models.

    ensure_annotations(atom_array, "chem_comp_type")

    chem_comp_rows = []

    res_starts_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    res_names_at_starts = atom_array.res_name[res_starts_ends[:-1]]
    for res_name in np.unique(atom_array.res_name):
        first_occurrence = np.where(res_names_at_starts == res_name)[0][0]

        current_row = {
            "id": res_name,
            "name": res_name,
            "type": atom_array.chem_comp_type[res_starts_ends[first_occurrence]],
            "formula": "?",
            "mon_nstd_parent_comp_id": "?",
            "pdbx_synonyms": "?",
            "formula_weight": "?",
        }

        chem_comp_rows.append(current_row)

    return {"chem_comp": _rows_to_columns(chem_comp_rows)}


def _build_chem_comp_atom(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    *,
    source: Literal["array", "ccd", "ccd_non_canonicals"] = "ccd",
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """Build the chem_comp_atom category for a CIF file.

    Args:
        atom_array: Structure data.
        source: Per-residue row source:

            - ``"array"`` — always the AtomArray itself.
            - ``"ccd"`` — always the CCD (AtomArray fallback if no template).
            - ``"ccd_non_canonicals"`` — AtomArray for canonical residues, CCD for the rest.
    """
    if isinstance(atom_array, struc.AtomArrayStack):
        # Use first model; annotations should be shared across models
        atom_array = atom_array[0]

    canonicals = STANDARD_POLYMER_RESIDUES if source == "ccd_non_canonicals" else ()

    chem_comp_atom_rows = []
    array_only_res_names: list[str] = []
    for res_name in np.unique(atom_array.res_name):
        res_name_str = str(res_name)
        use_ccd = source == "ccd" or (source == "ccd_non_canonicals" and res_name_str not in canonicals)
        if not use_ccd:
            array_only_res_names.append(res_name_str)
            continue
        try:
            template = atom_array_from_ccd_code(res_name)
        except (ValueError, KeyError):
            # Fallback: synthesize rows from the AtomArray itself for residues without
            # a CCD template (e.g. SMILES / SDF ligands with synthetic codes like "L:0").
            # Without this, downstream readers cannot register them as custom CCD
            # entries on roundtrip.
            array_only_res_names.append(res_name_str)
            continue
        chem_comp_atom_rows.extend(_template_to_chem_comp_atom_rows(res_name, template))

    for res_name in array_only_res_names:
        chem_comp_atom_rows.extend(_atom_array_chem_comp_atom_rows(atom_array, res_name))

    return {"chem_comp_atom": _rows_to_columns(chem_comp_atom_rows)}


def _build_chem_comp_bond(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    *,
    source: Literal["array", "ccd", "ccd_non_canonicals"] = "ccd",
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """Build the chem_comp_bond category for a CIF file.

    Array-sourced residues (including SMILES/SDF ligands like ``L:0`` that have
    no CCD template) have their bonds emitted from ``atom_array.bonds``, so
    they aren't silently dropped when this category overwrites biotite's
    auto-emitted one.

    Args:
        atom_array: Structure data.
        source: Per-residue row source. See :py:func:`_build_chem_comp_atom`.
    """
    if isinstance(atom_array, struc.AtomArrayStack):
        # Use first model to determine chem_comp_bond; annotations are shared across models.
        atom_array = atom_array[0]

    canonicals = STANDARD_POLYMER_RESIDUES if source == "ccd_non_canonicals" else ()

    chem_comp_bond_rows = []
    array_only_res_names: set[str] = set()
    for res_name in np.unique(atom_array.res_name):
        res_name_str = str(res_name)
        use_ccd = source == "ccd" or (source == "ccd_non_canonicals" and res_name_str not in canonicals)
        if not use_ccd:
            array_only_res_names.add(res_name_str)
            continue
        try:
            template = atom_array_from_ccd_code(res_name)
        except (ValueError, KeyError):
            array_only_res_names.add(res_name_str)
            continue
        chem_comp_bond_rows.extend(_template_to_chem_comp_bond_rows(res_name, template))

    chem_comp_bond_rows.extend(_atom_array_chem_comp_bond_rows(atom_array, array_only_res_names))

    if not chem_comp_bond_rows:
        return {}

    return {"chem_comp_bond": _rows_to_columns(chem_comp_bond_rows)}


def _build_entity_categories(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """Build ``_entity``, ``_entity_poly``, and ``_entity_poly_seq`` categories for a CIF file.

    Iterates over unique entities once to produce all entity-related categories within standard CIF files.
    Many CIF-reading programs (e.g., OpenStructure) expect these categories to be present for polymers.

    Non-polymer entities only get an ``_entity`` row; polymers get all three.

    Args:
        atom_array: AtomArray containing structure data with polymer chain information.

    Returns:
        Dict mapping category names to column dicts, suitable for
        :pyfunc:`_write_categories_to_block`.
    """
    if isinstance(atom_array, struc.AtomArrayStack):
        atom_array = atom_array[0]  # Choose any model

    atom_array = _ensure_label_entity_id(atom_array)

    assert (
        "chain_id" in atom_array.get_annotation_categories()
    ), "`chain_id` annotation is required to build entity categories."
    chain_info = build_chain_info(atom_array)

    # Build entity→chains mapping
    res_starts = get_residue_starts(atom_array)
    chain_ids_at_res = atom_array.chain_id[res_starts]
    entity_ids_at_res = atom_array.label_entity_id[res_starts]

    entity_to_chains = {}
    for chain_id in chain_info:
        # Get entity for this chain from atom_array annotation
        entity_id = entity_ids_at_res[chain_ids_at_res == chain_id][0]
        entity_to_chains.setdefault(entity_id, []).append(chain_id)

    # Helper for wrapping long sequences
    wrap_every_n = lambda text, n: "\n".join(text[i : i + n] for i in range(0, len(text), n))  # noqa: E731

    # Build categories as rows
    entity_rows = []
    entity_poly_rows = []
    entity_poly_seq_rows = []

    for entity_id in sorted(entity_to_chains.keys()):
        chain_ids = entity_to_chains[entity_id]
        representative_chain = chain_ids[0]
        info = chain_info[representative_chain]

        # Add entity row (all entities)
        entity_rows.append(
            {
                "id": str(entity_id),
                "type": info["chain_type"].to_entity_type(),
                "pdbx_description": ".",
            }
        )

        if not info["is_polymer"]:
            continue  # Skip polymer-specific categories

        # Check for non-standard monomers
        has_non_standard = not np.all(np.isin(info["res_name"], STANDARD_POLYMER_RESIDUES))

        # Add entity_poly row (polymers only)
        entity_poly_rows.append(
            {
                "entity_id": str(entity_id),
                "type": info["chain_type"].to_mmcif_string(),
                "nstd_linkage": "no",
                "nstd_monomer": "yes" if has_non_standard else "no",
                "pdbx_seq_one_letter_code": wrap_every_n(info["processed_entity_non_canonical_sequence"], 80),
                "pdbx_seq_one_letter_code_can": wrap_every_n(info["processed_entity_canonical_sequence"], 80),
                "pdbx_strand_id": ",".join(str(cid) for cid in chain_ids),
                "pdbx_target_identifier": "?",
            }
        )

        # Add entity_poly_seq rows (one per residue)
        entity_poly_seq_rows.extend(
            [
                {
                    "entity_id": str(entity_id),
                    "hetero": "n",
                    "mon_id": res_name,
                    "num": res_id,
                }
                for res_id, res_name in zip(info["res_id"], info["res_name"], strict=False)
            ]
        )

    # Convert rows to columns (CIF format)
    result = {"entity": _rows_to_columns(entity_rows)}
    if entity_poly_rows:
        result["entity_poly"] = _rows_to_columns(entity_poly_rows)
        result["entity_poly_seq"] = _rows_to_columns(entity_poly_seq_rows)

    return result


def _write_categories_to_block(
    block: "pdbx.Block", categories: dict[str, dict[str, float | int | str | list | np.ndarray]]
) -> None:
    """Write a set of categories to a CIF block"""
    Category = block.subcomponent_class()  # noqa: N806
    Column = Category.subcomponent_class()  # noqa: N806
    for category_name, category_data in categories.items():
        category = Category()
        for key, value in category_data.items():
            # ... skip empty columns
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                continue
            category[key] = Column(value)

        # ... skip empty categories
        if len(category) == 0:
            continue
        block[category_name] = category


def _cif_to_bcif(cif_file: pdbx.CIFFile | pdbx.BinaryCIFFile) -> pdbx.BinaryCIFFile:
    """Convert to BCIF while preserving column values, missing-value masks, and ordering."""
    compressed_file = pdbx.BinaryCIFFile()
    for block_name, block in cif_file.items():
        compressed_block = pdbx.BinaryCIFBlock()
        for category_name, category in block.items():
            compressed_category = pdbx.BinaryCIFCategory()
            for column_name, column in category.items():
                data = pdbx.BinaryCIFData(column.data.array.copy())
                if np.issubdtype(data.array.dtype, np.str_) or np.issubdtype(data.array.dtype, np.integer):
                    data = pdbx.compress(data)
                elif np.issubdtype(data.array.dtype, np.floating) and data.array.dtype.itemsize > 8:
                    raise ValueError(f"BinaryCIF cannot losslessly represent {data.array.dtype} in {column_name!r}")
                mask = column.mask.array.copy() if column.mask is not None else None
                compressed_category[column_name] = pdbx.BinaryCIFColumn(data, mask)
            compressed_block[category_name] = compressed_category
        compressed_file[block_name] = compressed_block
    return compressed_file


def _disambiguate_via_transformation_id(extra_fields: list[str] | Literal["all"]) -> list[str] | Literal["all"]:
    """Ensure transformation_id is in extra_fields for disambiguation."""
    if isinstance(extra_fields, list) and "transformation_id" not in extra_fields:
        return [*list(extra_fields), "transformation_id"]
    return extra_fields


def _disambiguate_via_chain_iid(
    structure: AtomArray,
) -> AtomArray:
    """Replace chain_id with chain_iid to make chains unique."""
    assert (
        "chain_iid" in structure.get_annotation_categories()
    ), "chain_iid annotation must be present for chain_iid disambiguation."
    structure.chain_id = structure.chain_iid
    return structure


def _preprocess_structure_for_cif(
    structure: AtomArray,
    config: CIFWriteConfig,
) -> tuple[AtomArray, list[str]]:
    """Preprocess a single AtomArray for CIF writing.

    Returns:
        Tuple of (preprocessed structure, resolved extra_fields list).
    """
    structure = structure.copy()
    structure.coord = np.round(structure.coord, 3)

    extra_fields = config.extra_fields
    if has_ambiguous_annotation_set(structure):
        if config.chain_disambiguation is None:
            raise ValueError(
                "Ambiguous chain annotations detected. Multiple atoms share the same "
                "(chain_id, res_id, res_name, atom_name, ins_code) identifier.\n\n"
                f"{CHAIN_DISAMBIGUATION_SOLUTIONS}"
            )
        elif config.chain_disambiguation == "transformation_id":
            # Mode 1: Keep original chain_id, store transformation_id in struct_conn symmetry fields
            # (Only can be loaded with AtomWorks, not other software)
            if "transformation_id" not in structure.get_annotation_categories():
                raise ValueError(
                    "chain_disambiguation='transformation_id' specified, but structure lacks "
                    "'transformation_id' annotation.\n\n"
                    f"{CHAIN_DISAMBIGUATION_SOLUTIONS}"
                )
            logger.debug("Disambiguating chains via transformation_id; original chain_id will be preserved.")
            extra_fields = _disambiguate_via_transformation_id(extra_fields)

        elif config.chain_disambiguation == "chain_iid":
            # Mode 2: Replace chain_id with chain_iid to make chains unique
            # (Can be loaded with any software, but loses original chain_id)
            if "chain_iid" not in structure.get_annotation_categories():
                raise ValueError(
                    "chain_disambiguation='chain_iid' specified, but structure lacks "
                    "'chain_iid' annotation.\n\n"
                    f"{CHAIN_DISAMBIGUATION_SOLUTIONS}"
                )
            logger.debug("Disambiguating chains via chain_iid; chain_id will be replaced with chain_iid.")
            structure = _disambiguate_via_chain_iid(structure)

    # Standardization: If elements are given as atomic numbers, convert them to (uppercase) element symbols
    structure.element = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x))(structure.element)

    # If altloc information is present but no altloc id is given, set all to "."
    if "altloc_id" in structure.get_annotation_categories() and structure.altloc_id[0].strip() == "":
        structure.altloc_id = ["."] * structure.array_length()

    canonical_names = set(STANDARD_ANNOTATIONS.get_field_names())
    accepted_names = set(STANDARD_ANNOTATIONS.get_field_names(include_aliases=True))
    legacy_names = (accepted_names - canonical_names) | set(_LEGACY_MASK_FIELDS.values())
    if extra_fields != "all":
        unsupported_names = sorted(set(extra_fields) & legacy_names)
        if unsupported_names:
            raise ValueError(
                f"Explicit StandardAnnotation aliases and legacy names in extra_fields are not supported: "
                f"{unsupported_names}. "
                'Omit these fields and use save_standard_annotations=True, or use extra_fields="all"; '
                "StandardAnnotations are saved under their canonical names."
            )

    if config.save_standard_annotations:
        _canonicalize_standard_annotation_aliases(structure)
        scalar_sa_names = _get_scalar_1body_sa_names(structure)
        if extra_fields != "all" and scalar_sa_names:
            existing = set(extra_fields)
            extra_fields = list(extra_fields) + [f for f in scalar_sa_names if f not in existing]

    # Set the structure in the CIF file
    # (Biotite writes the standard CIF categories by default)
    if extra_fields == "all":
        extra_fields = list(set(structure.get_annotation_categories()) - STANDARD_CIF_ANNOTATIONS)

        if config.save_standard_annotations:
            extra_fields = [field for field in extra_fields if field not in legacy_names]

    if not config.include_nan_coords:
        structure = ta.remove_nan_coords(structure)

    structure = _ensure_label_entity_id(structure)

    structure = remap_intra_residue_coordination_bonds(structure)

    # Always include atom_array_index in atom_site so readers can correlate atom_site rows
    # with idx0/idx1 values in other CIF categories (e.g. nonscalar StandardAnnotation blocks).
    structure.set_annotation("atom_array_index", np.arange(structure.array_length()))
    if "atom_array_index" not in extra_fields:
        extra_fields = [*list(extra_fields), "atom_array_index"]

    return structure, extra_fields


def _atom_site_from_array(structure: AtomArray, extra_fields: list[str]) -> pdbx.CIFCategory:
    """Build an atom_site CIFCategory for one AtomArray via a temporary CIFFile."""
    tmp = pdbx.CIFFile()
    pdbx.set_structure(tmp, structure, data_block="_tmp", extra_fields=extra_fields)
    return tmp["_tmp"]["atom_site"]


def _concatenate_atom_sites(atom_sites: list[pdbx.CIFCategory]) -> pdbx.CIFCategory:
    """Concatenate atom_site categories that may have different column sets.

    Missing columns are filled with zeros (numeric) or ``"?"`` (string) and
    marked as ``MaskValue.MISSING``.
    """
    if len(atom_sites) == 1:
        return atom_sites[0]

    seen: set[str] = set()
    all_keys: list[str] = []
    for site in atom_sites:
        for k in site:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    lengths = [len(site[next(iter(site))].as_array()) for site in atom_sites]

    result = pdbx.CIFCategory()
    for key in all_keys:
        ref_col = next(site[key] for site in atom_sites if key in site)
        ref_data = ref_col.as_array()
        ref_dtype = ref_data.dtype

        data_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []

        for site, n in zip(atom_sites, lengths, strict=False):
            if key in site:
                col = site[key]
                data_parts.append(col.as_array())
                if col.mask is not None:
                    mask_parts.append(col.mask._array)
                else:
                    mask_parts.append(np.zeros(n, dtype=np.int8))
            else:
                if ref_dtype.kind in ("U", "S", "O"):
                    data_parts.append(np.full(n, "?", dtype=ref_dtype))
                else:
                    data_parts.append(np.zeros(n, dtype=ref_dtype))
                mask_parts.append(np.full(n, MaskValue.MISSING, dtype=np.int8))

        combined_mask = np.concatenate(mask_parts)
        # Only attach a mask if any entry is non-PRESENT to keep output clean
        if np.any(combined_mask != MaskValue.PRESENT):
            result[key] = pdbx.CIFColumn(np.concatenate(data_parts), combined_mask)
        else:
            result[key] = pdbx.CIFColumn(np.concatenate(data_parts))

    return result


def _ensure_label_entity_id(atom_array: AtomArray) -> AtomArray:
    """Return a copy of ``atom_array`` with ``label_entity_id`` set (1-indexed).

    Ensures the entity table and ``label_entity_id`` on ``atom_site`` reference
    matching IDs during CIF writing. ``chain_entity`` (0-indexed, from
    :py:func:`annotate_entities`) is computed if missing but not modified.
    Returns the input unchanged if ``label_entity_id`` is already set.
    """
    if "label_entity_id" in atom_array.get_annotation_categories():
        return atom_array
    annotated = atom_array.copy()
    if "chain_entity" not in annotated.get_annotation_categories():
        annotated, _ = ta.annotate_entities(
            atom_array=annotated,
            level="chain",
            lower_level_id="res_id",
            lower_level_entity="res_name",
            add_inter_level_bond_hash=False,
        )
    annotated.set_annotation("label_entity_id", (annotated.chain_entity + 1).astype(np.int8))
    return annotated


def _to_cif_or_bcif(
    structure: AtomArray | list[AtomArray],
    config: CIFWriteConfig,
    *,
    model_ids: list[int] | None = None,
    as_bcif: bool = False,
) -> pdbx.CIFFile | pdbx.BinaryCIFFile:
    """Convert an AtomArray (or list thereof) to a CIF or BCIF file.

    When ``structure`` is a list, each element is written as a separate model.
    Models may differ in atom count and annotation set; StandardAnnotation
    categories are concatenated across models with a ``model_num`` column
    that records the originating model ID.
    Entity categories are only written for single-structure input.

    Args:
        structure: Single structure or list of structures.
        config: Configuration for the CIF write operation.
        model_ids: Model IDs for list input. Defaults to ``[1, 2, …]``.
        as_bcif: Return binary CIF format. Defaults to False.
    """
    is_list = isinstance(structure, list)
    structures: list[AtomArray] = structure if is_list else [structure]

    if not structures:
        raise ValueError("structures list must not be empty")
    if model_ids is None:
        model_ids = list(range(1, len(structures) + 1))
    if len(model_ids) != len(structures):
        raise ValueError(f"model_ids length {len(model_ids)} != structures length {len(structures)}")

    # Set label_entity_id once on the shared structure so the entity table and
    # atom_site reference matching IDs.
    structures = [_ensure_label_entity_id(s) for s in structures]

    date = config.date if exists(config.date) else datetime.now().strftime("%Y-%m-%d")
    time_str = config.time if exists(config.time) else datetime.now().strftime("%H:%M:%S")

    cif_file = pdbx.CIFFile()
    block = pdbx.convert._get_or_create_block(cif_file, block_name=config.id)

    # Extract CCD entries: prefer explicit config, then AtomArrayPlus attr, then empty
    ccd_entries = config.ccd_entries or getattr(structures[0], "_custom_ccd_registry", None) or {}

    metadata: dict = {"entry": {"id": config.id, "author": config.author, "date": date, "time": time_str}}
    if not is_list:
        for flag, build_func in [
            (config.include_entity_categories, _build_entity_categories),
            (config.include_chem_comp, _build_chem_comp),
        ]:
            if flag:
                try:
                    metadata.update(build_func(structures[0]))
                except Exception as e:
                    logger.warning(f"Failed to build `{build_func.__name__}`: {e}")
    _write_categories_to_block(block, metadata)

    all_extra_categories: dict[str, dict] = {}

    if is_list:
        all_atom_sites: list[pdbx.CIFCategory] = []
        # Accumulate SA data per category name across models. Each category
        # gets a `model_num` column so the reader can filter rows by model,
        # mirroring how atom_site uses pdbx_PDB_model_num.
        per_sa_category: dict[str, list[dict]] = defaultdict(list)
        for s, model_id in zip(structures, model_ids, strict=False):
            s, extra_fields = _preprocess_structure_for_cif(s, config)
            atom_site = _atom_site_from_array(s, extra_fields)
            atom_site["pdbx_PDB_model_num"] = np.full(s.array_length(), model_id, dtype=np.int32)
            all_atom_sites.append(atom_site)
            if config.save_standard_annotations:
                for key, val in _serialize_standard_annotation_categories(s).items():
                    n_rows = len(next(iter(val.values())))
                    val["model_num"] = np.full(n_rows, model_id, dtype=np.int32)
                    per_sa_category[key].append(val)

        # Concatenate each SA category across models into a single CIF category
        for key, parts in per_sa_category.items():
            all_columns = sorted(set().union(*(p.keys() for p in parts)))
            merged: dict[str, np.ndarray] = {}
            for col in all_columns:
                merged[col] = np.concatenate([p[col] for p in parts])
            all_extra_categories[key] = merged

        combined = _concatenate_atom_sites(all_atom_sites)
        total_atoms = sum(len(site["group_PDB"].as_array()) for site in all_atom_sites)
        combined["id"] = np.arange(1, total_atoms + 1, dtype=np.int32)
        block["atom_site"] = combined
    else:
        s, extra_fields = _preprocess_structure_for_cif(structures[0], config)
        pdbx.set_structure(cif_file, s, data_block=config.id, extra_fields=extra_fields)

        # Biotite hardcodes label_alt_id="." for all atoms — restore actual values,
        # but only when multiple conformers genuinely coexist per residue. After altloc
        # selection (one conformer per residue), keeping a non-"." label_alt_id would
        # break downstream CIF consumers (e.g. PyMOL struct_conn atom matching)
        if "label_alt_id" in s.get_annotation_categories():
            alt_id = s.label_alt_id
            if has_multiple_altlocs_per_residue(s):
                if isinstance(s, AtomArrayStack):
                    alt_id = np.tile(alt_id, len(s))
                block["atom_site"]["label_alt_id"] = CIFColumn(alt_id)

        if config.save_standard_annotations:
            all_extra_categories.update(_serialize_standard_annotation_categories(s))

    extra_categories = {**all_extra_categories, **(config.extra_categories or {})}
    if extra_categories:
        _write_categories_to_block(block, extra_categories)

    # Emit chem_comp_atom + chem_comp_bond inside context manager
    if not is_list and config.include_chem_comp:
        source = config.chem_comp_source
        uses_ccd = source in ("ccd", "ccd_non_canonicals")
        if uses_ccd and config.warn_on_ccd_without_registry and not isinstance(structures[0], AtomArrayPlus):
            logger.warning(
                "Writing chem_comp with chem_comp_source=%r from a plain AtomArray. "
                "A plain AtomArray has no preserved CCD context, so known component codes resolve to the bundled CCD. "
                "If this structure came from a CIF, parse it with return_atom_array_plus=True to preserve its CCD "
                "registry. If it was constructed or modified and a component slice is the complete intended CCD "
                "definition, write it with ccd_entries={code: component_template}; run any earlier CCD-dependent "
                "processing inside `with custom_ccd_residues(ccd_entries):`. "
                "Silence this warning with CIFWriteConfig(warn_on_ccd_without_registry=False).",
                source,
            )
        ctx = custom_ccd_residues(ccd_entries) if uses_ccd and ccd_entries else contextlib.nullcontext()
        with ctx:
            post_categories: dict = {}
            try:
                post_categories.update(_build_chem_comp_atom(structures[0], source=source))
            except Exception as e:
                logger.warning(f"Failed to build _build_chem_comp_atom: {e}")
            try:
                post_categories.update(_build_chem_comp_bond(structures[0], source=source))
            except Exception as e:
                logger.warning(f"Failed to build _build_chem_comp_bond: {e}")
            if post_categories:
                _write_categories_to_block(block, post_categories)

    if as_bcif:
        cif_file = _cif_to_bcif(cif_file)

    return cif_file


def to_cif_buffer(
    structure: AtomArray | list[AtomArray],
    *,
    model_ids: list[int] | None = None,
    config: CIFWriteConfig | None = None,
    as_bcif: bool = False,
) -> io.StringIO | io.BytesIO:
    """Convert an AtomArray (or list thereof) to a CIF formatted buffer.

    Args:
        structure: The atomic structure to be converted.  Pass a list to write
            each element as a separate model in a multi-model CIF.
        model_ids: Model IDs for each element when ``structure`` is a list.
            Defaults to ``[1, 2, …]``.
        config: Write configuration. If None, uses default CIFWriteConfig().
        as_bcif: Return binary CIF format. Defaults to False.

    Returns:
        StringIO or BytesIO buffer containing the CIF/BCIF formatted structure.

    See Also:
        :py:class:`CIFWriteConfig`
        :py:func:`to_cif_string`
        :py:func:`to_cif_file`
    """
    if config is None:
        config = CIFWriteConfig()

    file_obj = _to_cif_or_bcif(structure, config, model_ids=model_ids, as_bcif=as_bcif)
    buffer = io.BytesIO() if as_bcif else io.StringIO()
    file_obj.write(buffer)
    buffer.seek(0)
    return buffer


def to_cif_string(
    structure: AtomArray | list[AtomArray],
    *,
    model_ids: list[int] | None = None,
    config: CIFWriteConfig | None = None,
    as_bcif: bool = False,
) -> str | bytes:
    """Convert an AtomArray (or list thereof) to a CIF formatted string.

    Args:
        structure: The atomic structure to be converted.  Pass a list to write
            each element as a separate model in a multi-model CIF.
        model_ids: Model IDs for each element when ``structure`` is a list.
            Defaults to ``[1, 2, …]``.
        config: Write configuration. If None, uses default CIFWriteConfig().
        as_bcif: Return binary CIF format. Defaults to False.

    Returns:
        String or bytes containing the CIF/BCIF formatted structure.
    """
    return to_cif_buffer(structure, model_ids=model_ids, config=config, as_bcif=as_bcif).getvalue()


def _infer_file_type_from_path(path: str | os.PathLike) -> str:
    """Infer CIF file type from path extension."""
    path = Path(path)
    compression = path.suffix.lower() if is_compressed_file(path) else ""
    structure_path = path.with_suffix("") if compression else path
    if structure_path.suffix.lower() in (".cif", ".bcif"):
        return structure_path.suffix.lower().lstrip(".") + compression
    raise ValueError(
        f"Could not infer file type from path: {path}. Use .cif or .bcif, optionally with .gz, .gzip, or .zst"
    )


def _to_cif_file(
    file_obj: pdbx.CIFFile | pdbx.BinaryCIFFile,
    path: os.PathLike,
    file_type: CIFFileType,
) -> str:
    # turn any relative path into an absolute path
    path = str(os.path.abspath(path))

    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    path_suffix = "." + file_type
    if _infer_file_type_from_path("structure" + path_suffix) != file_type:
        raise ValueError(f"Unsupported CIF file type: {file_type}")
    open_mode = "wb" if file_type.startswith("bcif") else "wt"

    # ... check that the file ends with the correct suffix, otherwise mutate the path to end with the correct suffix
    if not path.lower().endswith(path_suffix):
        path = path + path_suffix

    with open_compressed(path, mode=open_mode) as f:
        file_obj.write(f)

    return path


def to_cif_file(
    structure: AtomArray | list[AtomArray],
    path: os.PathLike,
    *,
    model_ids: list[int] | None = None,
    file_type: CIFFileType | None = None,
    id: str | None = None,
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool | None = None,
    include_entity_categories: bool = True,
    include_chem_comp: bool = True,
    include_nan_coords: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    chain_disambiguation: Literal["transformation_id", "chain_iid"] | None = None,
    save_standard_annotations: bool = True,
    include_bonds: bool | None = None,
    ccd_entries: dict[str, struc.AtomArray] | None = None,
) -> os.PathLike:
    """Convert an AtomArray structure to a CIF/BCIF formatted file.

    Args:
        structure: The atomic structure to be converted.
        path: The file path where the CIF formatted structure will be saved.
        file_type: The file type to save the structure as.
            ``cif`` or ``bcif``, optionally suffixed with ``.gz``, ``.gzip``, or ``.zst``.
            If None, inferred from the path. Zstandard uses level 3; gzip uses level 9.
        id: The ID of the entry. This will be used as the data block name.
            If None, the data block name will be inferred from the path.
        author: The author of the entry. Defaults to current user.
        date: The date of the entry (YYYY-MM-DD). Auto-generated if None.
        time: The time of the entry (HH:MM:SS). Auto-generated if None.
        include_entity_poly: **DEPRECATED**. Use ``include_entity_categories`` instead.
        include_entity_categories: Write entity categories (_entity, _entity_poly, etc.).
            Defaults to True for file operations.
        include_chem_comp: Write _chem_comp CIF category. Defaults to True.
        include_nan_coords: Write atoms with NaN coordinates. Defaults to True.
        extra_fields: Additional atom_array annotations to include in the CIF file.
            Use "all" to include all non-standard annotations.
            Explicit StandardAnnotation aliases and legacy names are unsupported. With
            ``save_standard_annotations=True``, "all" silently filters them and saves canonical names.
        extra_categories: Additional CIF categories to include in data block.
            Must be dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}
        chain_disambiguation: Method for disambiguating chains in multi-transformation assemblies.
            Defaults to "chain_iid".

            - "transformation_id": Stores transformation_id in struct_conn symmetry fields.
              Preserves original chain_id. Requires AtomWorks to read.
            - "chain_iid": Writes chain_iid as chain_id to make chains unique.
              Better compatibility with external software.

        save_standard_annotations: Whether to serialize registered
            :py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase` annotations
            (conditions, ``atomize``, etc.) present on the structure. Defaults to True.
        include_bonds: **DEPRECATED**. This parameter has no effect and will be removed in a future version.
        ccd_entries: Optional dict mapping residue name to CCD template AtomArray for
            ``chem_comp_atom`` and ``chem_comp_bond`` emission. Falls back to the
            structure's ``_custom_ccd_registry``, then bundled CCD.

    Returns:
        The absolute file path where the CIF formatted structure was saved.

    Raises:
        IOError: If there's an issue writing to the specified file path.

    See Also:
        :py:class:`CIFWriteConfig`
        :py:func:`to_cif_buffer`
        :py:func:`to_cif_string`
    """
    # Handle deprecated parameter
    if include_entity_poly is not None:
        warnings.warn(
            "Parameter 'include_entity_poly' is deprecated and will be removed in a future version. "
            "Use 'include_entity_categories' instead.",
            FutureWarning,
            stacklevel=2,
        )
        # If new parameter not explicitly set, use deprecated value
        if include_entity_categories and not include_entity_poly:
            include_entity_categories = include_entity_poly

    if include_bonds is not None:
        warnings.warn(
            "Parameter 'include_bonds' has been deprecated in Biotite and will be removed in a future version. "
            "This parameter has no effect.",
            FutureWarning,
            stacklevel=2,
        )

    # Turn any relative path into an absolute path
    path = str(os.path.abspath(path))
    file_name = os.path.basename(path)

    if file_type is None:
        file_type = _infer_file_type_from_path(path)

    # Build config from parameters
    config = CIFWriteConfig(
        id=id or file_name,
        author=author,
        date=date,
        time=time,
        include_entity_categories=include_entity_categories,
        include_chem_comp=include_chem_comp,
        include_nan_coords=include_nan_coords,
        extra_fields=extra_fields if extra_fields else [],
        extra_categories=extra_categories,
        chain_disambiguation=chain_disambiguation,
        save_standard_annotations=save_standard_annotations,
        ccd_entries=ccd_entries,
    )

    file_obj = _to_cif_or_bcif(structure, config, model_ids=model_ids, as_bcif="bcif" in file_type)
    return _to_cif_file(file_obj, path, file_type=file_type)


def to_pdb_buffer(
    structure: AtomArray,
) -> io.StringIO:
    """Convert an AtomArray structure to a PDB formatted StringIO buffer.

    NOTE: It's recommended to use `to_cif_buffer` instead of this function. That function
    is more flexible and can handle extra annotations and metadata that PDB does not support.

    Args:
        - structure (AtomArray): The atomic structure to be converted.

    Returns:
        StringIO: The PDB formatted StringIO buffer of the structure.
    """
    # Create a PDBFile object
    pdb_file = biotite_pdb.PDBFile()

    if has_ambiguous_annotation_set(structure):
        raise ValueError(
            "Ambiguous bond annotations detected. This happens when there are atoms that "
            "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
            "This happens for example when you have a bio-assembly with multiple copies "
            "of a chain that only differ by `transformation_id`.\n"
            "You can fix this for example by re-naming the chains to be named uniquely."
        )

    # Set the structure and bonds
    pdb_file.set_structure(structure)

    # Convert to string
    buffer = io.StringIO()
    pdb_file.write(buffer)
    return buffer


def to_pdb_string(
    structure: AtomArray,
) -> str:
    """
    Convert an AtomArray structure to a PDB formatted string.

    NOTE: It's recommended to use `to_cif_string` instead of this function. That function
    is more flexible and can handle extra annotations and metadata that PDB does not support.

    Args:
        - structure (AtomArray): The atomic structure to be converted.

    Returns:
        str: The PDB formatted string representation of the structure.
    """
    return to_pdb_buffer(structure).getvalue()


def find_files_by_extension(input_dir: Path, extension: str) -> list[Path]:
    """Recursively find files with the specified extension in a directory."""
    files = [f for f in input_dir.rglob(f"*{extension}") if str(f).endswith(extension)]

    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found in {input_dir}")

    return files


def build_sharding_pattern(depth: int, chars_per_dir: int = 2) -> str:
    """Build a sharding pattern string from depth and characters per directory.

    Args:
        depth: Number of directory levels.
        chars_per_dir: Number of characters to use for each directory level.

    Returns:
        Sharding pattern string.

    Examples:
        >>> build_sharding_pattern(2, 2)
        '/0:2/2:4/'
        >>> build_sharding_pattern(3, 1)
        '/0:1/1:2/2:3/'
    """
    if depth == 0:
        return ""

    parts = []
    for i in range(depth):
        start = i * chars_per_dir
        end = start + chars_per_dir
        parts.append(f"/{start}:{end}")

    return "".join(parts) + "/"


def parse_sharding_pattern(sharding_pattern: str) -> list[tuple[int, int]]:
    """Parse a sharding pattern string into directory levels.

    Args:
        sharding_pattern: String like ``"/1:2/0:2/"`` where each ``/start:end/`` defines a directory level.
            ``start:end`` defines the character range to use for that directory level.

    Returns:
        List of (start, end) tuples for each directory level.

    Examples:
        >>> parse_sharding_pattern("/1:2/0:2/")
        [(1, 2), (0, 2)]
    """
    # Find all patterns like /start:end/ using a non-consuming lookahead
    pattern = r"/(\d+):(\d+)(?=/)"
    matches = []
    for match in re.finditer(pattern, sharding_pattern):
        matches.append((int(match.group(1)), int(match.group(2))))

    if not matches:
        raise ValueError(f"Invalid sharding pattern format: {sharding_pattern}. Expected format like '/1:2/0:2/'")

    return matches


def apply_sharding_pattern(path: os.PathLike, sharding_pattern: str | None = None) -> Path:
    """Apply a sharding pattern to construct a file path.

    Args:
        path: The base path or identifier (e.g., PDB ID).
        sharding_pattern: Pattern for organizing files in subdirectories. Examples:
            - ``"/0:2/"``: Use first two characters for first directory level
            - ``"/0:2/2:4/"``: Use chars 0-2 for first dir, then chars 2-4 for second dir
            - ``None``: No sharding (default)

    Returns:
        The constructed file path with sharding applied.

    Examples:
        >>> apply_sharding_pattern("12as", "/0:2/1:3/")
        Path("12/2a/12as")
    """
    path_str = str(path)
    assert path_str and path_str != ".", "Path cannot be empty"

    if not sharding_pattern:
        return Path(path_str)

    if not sharding_pattern.startswith("/"):
        raise ValueError(f"Sharding pattern must start with '/': {sharding_pattern}")

    try:
        shard_ranges = parse_sharding_pattern(sharding_pattern)
    except ValueError as e:
        raise ValueError(f"Invalid sharding pattern '{sharding_pattern}': {e}") from e

    # Validate all ranges before building path
    for start, end in shard_ranges:
        if end > len(path_str):
            raise ValueError(f"Sharding range {start}:{end} exceeds path length {len(path_str)} for '{path_str}'")

    # Build directory components from sharding ranges
    directory_parts = [path_str[start:end] for start, end in shard_ranges]

    # Construct final path: directories + filename
    return Path(*directory_parts, path_str)
