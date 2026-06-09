"""General utility functions for working with CIF files in Biotite."""

__all__ = [
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
]

import gzip
import io
import json
import logging
import os
import re
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import biotite.structure.io.pdb as biotite_pdb
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.bonds import connect_via_residue_names
from biotite.structure.io import mol, pdbx

import atomworks.io.transforms.atom_array as ta  # to avoid circular import
from atomworks.common import exists
from atomworks.constants import ATOMIC_NUMBER_TO_ELEMENT, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.enums import ChainType
from atomworks.io.template import add_inter_residue_bonds
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.selection import get_annotation
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from atomworks.io.utils.testing import has_ambiguous_annotation_set

logger = logging.getLogger("atomworks.io")

CIF_LIKE_EXTENSIONS = {
    ".cif",
    ".pdb",
    ".bcif",
    ".cif.gz",
    ".pdb.gz",
    ".bcif.gz",
    ".json",
    ".json.gz",
    ".mmjson",
    ".mmjson.gz",
}


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


def load_any(
    file_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif", "mmjson"] | None = None,
    *,
    extra_fields: list[str] | Literal["all"] = [],
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "occupancy", "all"] = "occupancy",
) -> AtomArrayStack | AtomArray:
    """Convenience function for loading a structure from a file or buffer.

    Args:
        file_or_buffer: Path to the file or buffer to load the structure from.
        file_type: Type of the file to load. If None, it will be inferred.
        extra_fields: List of extra fields to include as AtomArray annotations.
            If "all", all fields in the 'atom_site' category of the file will be included.
        include_bonds: Whether to include bonds in the structure.
        model: The model number to use for loading the structure. If None, all models will be loaded.
        altloc: The altloc ID to use for loading the structure.

    Returns:
        The loaded structure with the specified fields and assumptions.

    References:
        `Biotite Structure I/O <https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure>`_
        `mmCIF Format Specification <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/>`_
    """
    file_obj = read_any(file_or_buffer, file_type=file_type)
    return get_structure(
        file_obj,
        extra_fields=extra_fields,
        include_bonds=include_bonds,
        model=model,
        altloc=altloc,
    )


def _add_bonds(
    atom_array: AtomArray | AtomArrayStack,
    cif_block: pdbx.CIFBlock,
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    fix_bond_types: bool = True,
) -> AtomArray | AtomArrayStack:
    """Add bonds to the AtomArray and filter by a given altloc strategy.

    Avoids the issue where spurious bonds are added due to uninformative label_seq_ids.

    Args:
        atom_array: The AtomArray to add bonds to. Must contain `auth_seq_id` annotation.
        cif_block: The CIFBlock containing the structure data.
        add_bond_types_from_struct_conn: A list of bond types to add to the structure
            from the `struct_conn` category. Defaults to `["covale"]`. This means that we will only
            add covalent bonds to the structure (excluding metal coordination and disulfide bonds).
        fix_bond_types: Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.

    Returns:
        AtomArray | AtomArrayStack: The AtomArray or AtomArrayStack with bonds and filtered by altloc.
    """

    # If there are no uninformative res_ids, we can skip a few steps
    contains_uninformative_res_ids = np.any(atom_array.res_id == -1)

    if contains_uninformative_res_ids and not hasattr(atom_array, "auth_seq_id"):
        raise ValueError(
            "To ensure that bonds are added correctly when there are uninformative `label_seq_id` values present "
            "(occurs for non-polymers), the `auth_seq_id` annotation must be given in the `AtomArray`."
            "This error should not occur if the `AtomArray` was loaded with `atomworks.io`, but may "
            "occur if biotite was used directly. Please re-load the structure from CIF using `atomworks.io`."
        )

    # Compute intra-residue bonds as specified in the CIF, or fallback to CCD
    custom_bond_dict = None
    if "chem_comp_bond" in cif_block:
        try:
            custom_bond_dict = pdbx.convert._parse_intra_residue_bonds(cif_block["chem_comp_bond"])
        except KeyError:
            warnings.warn(
                "The 'chem_comp_bond' category has missing columns, "
                "falling back to using Chemical Component Dictionary",
                UserWarning,
                stacklevel=2,
            )
            custom_bond_dict = None
    intra_residue_bonds: struc.bonds.BondList = connect_via_residue_names(
        atom_array, custom_bond_dict=custom_bond_dict, inter_residue=False
    )

    if contains_uninformative_res_ids:
        # Detect spurious inter-residue bonds - at this point all bonds should be intra-residue
        bonds_array = intra_residue_bonds.as_array()
        auth_seq_ids = atom_array.auth_seq_id.astype(int)
        is_spurious = auth_seq_ids[bonds_array[:, 0]] != auth_seq_ids[bonds_array[:, 1]]
        # Remove spurious inter-residue bonds
        intra_residue_bonds._bonds = intra_residue_bonds._bonds[~is_spurious]

        # Temporarily replace uninformative label_seq_ids with auth_seq_ids
        to_replace = atom_array.res_id == -1
        atom_array.set_annotation("_to_replace", to_replace)  # To preserve info through leaving atom removal
        atom_array.res_id[to_replace] = atom_array.auth_seq_id[to_replace]

    # Add intra-residue bonds
    atom_array.bonds = intra_residue_bonds

    # Correctly add inter-residue bonds for each AtomArray
    if isinstance(atom_array, AtomArrayStack):
        processed_arrays = []
        for array in atom_array:
            array = add_inter_residue_bonds(
                array,
                struct_conn_dict=category_to_dict(cif_block, "struct_conn"),
                add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
                fix_formal_charges=False,  # Only works if we have True (not inferred) hydrogens
                fix_bond_types=fix_bond_types,
            )
            processed_arrays.append(array)
        atom_array = struc.stack(processed_arrays)
    else:
        atom_array = add_inter_residue_bonds(
            atom_array,
            struct_conn_dict=category_to_dict(cif_block, "struct_conn"),
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            fix_formal_charges=False,  # Only works if we have True (not inferred) hydrogens
            fix_bond_types=fix_bond_types,
        )

    if contains_uninformative_res_ids:
        # Revert back to the original label_seq_id
        atom_array.res_id[atom_array._to_replace] = -1
        atom_array.del_annotation("_to_replace")

    return atom_array


def get_structure(
    file_obj: pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile | pdbx.CIFBlock,
    *,
    extra_fields: list[str] | Literal["all"] = [],
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "occupancy", "all"] | str = "first",
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    fix_bond_types: bool = True,
) -> AtomArrayStack | AtomArray:
    """
    Load example structure into Biotite's AtomArray or AtomArrayStack using the specified fields and assumptions.

    Args:
        - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object to load with Biotite.
        - extra_fields (list | Literal["all"]): List of extra fields to include as AtomArray annotations.
            If "all", all fields in the 'atom_site' category of the file will be included.
        - include_bonds (bool): Whether to include bonds in the structure. These will not be affected by the issue
            where spurious bonds are added due to uninformative label_seq_ids.
        - model (int): The model number to use for loading the structure.
        - altloc (Literal["first", "occupancy", "all"]): The altloc ID to use for loading the structure.
            If a string is provided, it will be used as the altloc ID to filter the structure by and it is assumed
            that that altloc ID is present in the file. If it is not present, an error will be raised.
        - add_bond_types_from_struct_conn (list, optional): A list of bond types to add to the structure
            from the `struct_conn` category. Defaults to `["covale"]`. This means that we will only
            add covalent bonds to the structure (excluding metal coordination and disulfide bonds).
        - fix_bond_types (bool, optional): Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.

    Returns:
        AtomArray | AtomArrayStack: The loaded structure with the specified fields and assumptions.

    Reference:
        `Biotite documentation <https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure>`_
    """
    tmp_altloc = altloc if altloc in {"first", "occupancy", "all"} else "all"

    match type(file_obj):
        case pdbx.CIFFile | pdbx.BinaryCIFFile | pdbx.CIFBlock:
            # auth_seq_id must be included for the spurious bonds issue to be avoided
            # This will be removed later if it wasn't requested
            remove_auth_seq_id = False
            if include_bonds and extra_fields != "all" and "auth_seq_id" not in extra_fields:
                remove_auth_seq_id = True
                extra_fields = ["auth_seq_id", *extra_fields]

            # Filter extra annotations to fields that are actually present in the file
            if not isinstance(file_obj, pdbx.CIFBlock):
                cif_block = file_obj.block
            else:
                cif_block = file_obj
            if extra_fields == "all":
                extra_fields = list(cif_block["atom_site"].keys())
            extra_fields = _filter_extra_fields(extra_fields, cif_block["atom_site"])

            atom_array_stack = pdbx.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_fields,
                use_author_fields=False,
                altloc="first" if "occupancy" not in extra_fields else tmp_altloc,
                include_bonds=False,
            )

            # Add bonds and filter by altloc if requested, avoiding the spurious bonds issue
            if include_bonds:
                atom_array_stack = _add_bonds(
                    atom_array_stack,
                    cif_block,
                    add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
                    fix_bond_types=fix_bond_types,
                )

                # Remove the auth_seq_id annotation if it was not requested
                if remove_auth_seq_id:
                    atom_array_stack.del_annotation("auth_seq_id")

        case biotite_pdb.PDBFile:
            atom_array_stack = biotite_pdb.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_fields,
                altloc=tmp_altloc,
                include_bonds=include_bonds,  # PDB files contain only auth_seq_ids so the biotite issue does not arise
            )
        case _:
            raise ValueError(f"Unsupported file type: {type(file_obj)}. Must be a CIFFile, BinaryCIFFile, or PDBFile.")

    # Filter down to specified altloc if requested
    if altloc != tmp_altloc:
        altloc_ids = get_annotation(atom_array_stack, "altloc_id", default=[])
        existing_altloc_ids = np.unique(altloc_ids)
        if altloc not in existing_altloc_ids:
            raise ValueError(
                f"Altloc ID '{altloc}' not found. "
                f"Available altloc IDs: {existing_altloc_ids}. "
                f"If you are using a PDB file, please ensure that the altloc ID is present in the file."
            )
        to_keep = np.isin(altloc_ids, [altloc, ".", "?"])  # always keep default altloc IDs
        atom_array_stack = atom_array_stack[..., to_keep]

    return atom_array_stack


def _infer_file_type_from_buffer(buffer: io.BytesIO | io.StringIO) -> str:
    """Infer file type from buffer contents."""
    if isinstance(buffer, io.BytesIO):
        return "bcif"

    # StringIO - peek at contents to determine format
    buffer.seek(0)
    first_char = buffer.read(1)
    buffer.readline()  # finish first line
    second_line = buffer.readline()
    buffer.seek(0)

    if first_char == "{":
        return "mmjson"
    if second_line.startswith("#"):
        return "cif"
    return "pdb"


def infer_pdb_file_type(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
) -> Literal["cif", "pdb", "bcif", "sdf", "mmjson"]:
    """
    Infer the file type of a PDB file or buffer.
    """
    # Convert string paths to Path objects
    if isinstance(path_or_buffer, str):
        path_or_buffer = Path(path_or_buffer)

    # Determine file type and open context
    if isinstance(path_or_buffer, io.StringIO | io.BytesIO):
        return _infer_file_type_from_buffer(path_or_buffer)
    elif isinstance(path_or_buffer, Path):
        if path_or_buffer.suffix in (".gz", ".gzip"):
            inferred_file_type = Path(path_or_buffer.stem).suffix.lstrip(".")
        else:
            inferred_file_type = path_or_buffer.suffix.lstrip(".")

    # Canonicalize the file type
    if inferred_file_type in ("cif", "mmcif", "pdbx"):
        return "cif"
    elif inferred_file_type in ("pdb", "pdb1", "ent"):
        return "pdb"
    elif inferred_file_type == "bcif":
        return "bcif"
    elif inferred_file_type == "sdf":
        return "sdf"
    elif inferred_file_type in ("json", "mmjson"):
        return "mmjson"
    else:
        raise ValueError(f"Unsupported file type: {inferred_file_type}")


def _read_mmjson(file_obj: io.StringIO | io.BytesIO | io.TextIOWrapper) -> pdbx.CIFFile:
    """Read an mmjson file into a CIFFile object."""
    data = json.load(file_obj)
    cif_file = pdbx.CIFFile()
    for block_name, block_data in data.items():
        cif_block = pdbx.CIFBlock()
        for cat_name, cat_data in block_data.items():
            cif_category = pdbx.CIFCategory()
            for col_name, col_data in cat_data.items():
                # Convert None to "?" and ensure all elements are strings
                processed_data = [str(x) if x is not None else "?" for x in col_data]
                cif_category[col_name] = pdbx.CIFColumn(processed_data)
            cif_block[cat_name] = cif_category
        cif_file[block_name] = cif_block
    return cif_file


def read_any(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "pdb", "bcif", "sdf", "mmjson"] | None = None,
) -> pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile:
    """
    Reads any of the allowed file types into the appropriate Biotite file object.

    Args:
        path_or_buffer (PathLike | io.StringIO | io.BytesIO): The path to the file or a buffer to read from.
            If a buffer, it's highly recommended to specify the file_type.
        file_type (Literal["cif", "pdb", "bcif", "mmjson"], optional): Type of the file.
            If None, it will be inferred from the file extension. When using a buffer, the file type must be specified.

    Returns:
        pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile: The loaded file object.

    Raises:
        ValueError: If the file type is unsupported or cannot be determined.
    """
    # Determine file type
    if file_type is None:
        file_type = infer_pdb_file_type(path_or_buffer)

    open_mode = "rb" if file_type == "bcif" else "rt"

    # Convert string paths to Path objects and decompress if necessary
    if isinstance(path_or_buffer, str | Path):
        path_or_buffer = Path(path_or_buffer)
        if path_or_buffer.suffix in (".gz", ".gzip"):
            with gzip.open(path_or_buffer, open_mode) as f:
                buffer_type = io.StringIO if open_mode == "rt" else io.BytesIO
                path_or_buffer = buffer_type(f.read())

    # Determine the appropriate file object based on file type
    if file_type == "cif":
        file_cls = pdbx.CIFFile
    elif file_type == "pdb":
        file_cls = biotite_pdb.PDBFile
    elif file_type == "bcif":
        file_cls = pdbx.BinaryCIFFile
    elif file_type == "sdf":
        file_cls = mol.SDFile
    elif file_type == "mmjson":
        # Special handling for mmjson
        if isinstance(path_or_buffer, io.StringIO | io.BytesIO):
            return _read_mmjson(path_or_buffer)
        else:
            with open(path_or_buffer) as f:
                return _read_mmjson(f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Load the file content
    file_obj = file_cls.read(path_or_buffer)

    return file_obj


def _build_entity_poly(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """
    Build the entity_poly category for a CIF file from an AtomArray.

    This function processes polymer entities in the structure and generates their sequence information
    in both canonical and non-canonical forms.

    Args:
        - atom_array: AtomArray containing the structure data with polymer chain information.

    Returns:
        A dictionary containing the entity_poly category with the following fields:
        - entity_id: List of entity identifiers
        - type: List of polymer types (polypeptide, polynucleotide, etc.)
        - nstd_linkage: List indicating presence of non-standard linkages
        - nstd_monomer: List indicating presence of non-standard monomers
        - pdbx_seq_one_letter_code: List of sequences in one-letter code
        - pdbx_seq_one_letter_code_can: List of canonical sequences
        - pdbx_strand_id: List of chain identifiers
        - pdbx_target_identifier: List of target identifiers
    """
    _entity_poly_categories = (
        "entity_id",
        "type",
        "nstd_linkage",
        "nstd_monomer",
        "pdbx_seq_one_letter_code",
        "pdbx_seq_one_letter_code_can",
        "pdbx_strand_id",
        "pdbx_target_identifier",
    )

    if isinstance(atom_array, struc.AtomArrayStack):
        atom_array = atom_array[0]  # Choose any model

    # ... get index of the first atom of each chain
    chain_starts = struc.get_chain_starts(atom_array)

    # ... get chain ids, iids, entity ids, and chain types
    chain_ids = atom_array.chain_id[chain_starts]
    chain_iids = atom_array.chain_iid[chain_starts]
    entity_ids = atom_array.chain_entity[chain_starts]
    is_polymer = atom_array.is_polymer[chain_starts]
    chain_types = atom_array.chain_type[chain_starts]

    if not np.any(is_polymer):
        return {}

    unique_polymer_entity_ids = np.unique(entity_ids[is_polymer])
    entity_poly = {cat: [] for cat in _entity_poly_categories}
    for entity_id in unique_polymer_entity_ids:
        # ... get all relevant chain ids
        example_chain_ids = np.unique(chain_ids[entity_ids == entity_id])

        # ... get chain type
        chain_type = ChainType.as_enum(chain_types[entity_ids == entity_id][0])

        # ... get sequence
        example_chain_iid = chain_iids[entity_ids == entity_id][0]
        res_starts = struc.get_residue_starts(atom_array[atom_array.chain_iid == example_chain_iid])
        seq = atom_array.res_name[res_starts]
        wrap_every_n = lambda text, n: "\n".join(text[i : i + n] for i in range(0, len(text), n))  # noqa: E731
        processed_entity_non_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=False) for ccd_code in seq
        )
        processed_entity_non_canonical_sequence = wrap_every_n(processed_entity_non_canonical_sequence, 80)
        processed_entity_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=True) for ccd_code in seq
        )
        processed_entity_canonical_sequence = wrap_every_n(processed_entity_canonical_sequence, 80)

        # ... check for non-standard monomers
        has_non_standard_monomer = ~np.all(np.isin(seq, STANDARD_AA + STANDARD_RNA + STANDARD_DNA))

        # ... add to entity_poly
        entity_poly["entity_id"].append(entity_id)
        entity_poly["type"].append(chain_type.to_string().lower())
        entity_poly["nstd_linkage"].append("no")
        entity_poly["nstd_monomer"].append("yes" if has_non_standard_monomer else "no")
        entity_poly["pdbx_seq_one_letter_code"].append(processed_entity_non_canonical_sequence)
        entity_poly["pdbx_seq_one_letter_code_can"].append(processed_entity_canonical_sequence)
        entity_poly["pdbx_strand_id"].append(",".join(example_chain_ids))
        entity_poly["pdbx_target_identifier"].append("?")
    return {"entity_poly": entity_poly}


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
    """Convert a given CIF file to an optimized BCIF file."""
    from biotite.setup_ccd import _concatenate_blocks_into_category

    compressed_file = pdbx.BinaryCIFFile()
    for block_name, block in cif_file.items():
        compressed_block = pdbx.BinaryCIFBlock()
        for category_name in block:
            _tmp_cif_file = pdbx.CIFFile()
            _tmp_cif_file[block_name] = block
            compressed_block[category_name] = pdbx.compress(
                _concatenate_blocks_into_category(_tmp_cif_file, category_name)
            )
        compressed_file[block_name] = compressed_block
    return compressed_file


def _to_cif_or_bcif(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    as_bcif: bool = False,
    _allow_ambiguous_bond_annotations: bool = False,
) -> pdbx.CIFFile | pdbx.BinaryCIFFile:
    structure = structure.copy()
    cif_file = pdbx.CIFFile()

    if not exists(date):
        date = datetime.now().strftime("%Y-%m-%d")
    if not exists(time):
        time = datetime.now().strftime("%H:%M:%S")

    if not _allow_ambiguous_bond_annotations and has_ambiguous_annotation_set(structure):
        raise ValueError(
            "Ambiguous bond annotations detected. This happens when there are atoms that "
            "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
            "This happens for example when you have a bio-assembly with multiple copies "
            "of a chain that only differ by `transformation_id`.\n"
            "You can fix this for example by re-naming the chains to be named uniquely."
        )

    # If elements are given as atomic numbers, convert them to (uppercase) element symbols
    structure.element = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x))(structure.element)

    # If altloc information is present but no altloc id is given, set all to "."
    if "altloc_id" in structure.get_annotation_categories() and structure.altloc_id[0].strip() == "":
        structure.altloc_id = ["."] * structure.array_length()

    block = pdbx.convert._get_or_create_block(cif_file, block_name=id)

    # Build metadata
    metadata = {"entry": {"id": id, "author": author, "date": date, "time": time}}
    for flag, build_func in [
        (include_entity_poly, _build_entity_poly),
    ]:
        if flag:
            try:
                metadata.update(build_func(structure))
            except Exception as e:
                logger.warning(f"Failed to build `{build_func.__name__}`: {e}")
    # Write metadata to block
    _write_categories_to_block(block, metadata)

    # Set the structure in the CIF file
    if extra_fields == "all":
        _standard_cif_annotations = frozenset(
            {
                "chain_id",
                "res_id",
                "res_name",
                "atom_name",
                "atom_id",
                "element",
                "ins_code",
                "hetero",
                "altloc_id",
                "charge",
                "occupancy",
                "b_factor",
            }
        )
        extra_fields = list(set(structure.get_annotation_categories()) - _standard_cif_annotations)

    if not include_nan_coords:
        structure = ta.remove_nan_coords(structure)

    if include_bonds and structure.bonds is not None:
        # TODO: Switch to using the `convert_bond_type` method once we upgrade to Biotite v1.4.0
        # structure.bonds.convert_bond_type(struc.bonds.BondType.COORDINATION, struc.bonds.BondType.SINGLE)
        mask = structure.bonds._bonds[:, 2] == struc.bonds.BondType.COORDINATION
        structure.bonds._bonds[mask, 2] = struc.bonds.BondType.SINGLE

    pdbx.set_structure(cif_file, structure, data_block=id, include_bonds=include_bonds, extra_fields=extra_fields)

    # Add extra categories if provided
    extra_categories = extra_categories or {}
    if extra_categories:
        _write_categories_to_block(block, extra_categories)

    if as_bcif:
        cif_file = _cif_to_bcif(cif_file)
    return cif_file


def to_cif_buffer(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    _allow_ambiguous_bond_annotations: bool = False,
    as_bcif: bool = False,
) -> io.StringIO | io.BytesIO:
    """Convert an AtomArray structure to a CIF formatted StringIO buffer.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        id (str): The ID of the entry. This will be used as the data block name.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}
        _allow_ambiguous_bond_annotations (bool, optional): Private argument, not meant for public use.
            If True, allows ambiguous bond annotations.

    Returns:
        StringIO | BytesIO: A buffer containing the CIF/BCIF formatted string/bytes representation of the structure.
    """
    file_obj = _to_cif_or_bcif(
        structure,
        id=id,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        as_bcif=as_bcif,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
    )
    buffer = io.BytesIO() if as_bcif else io.StringIO()
    file_obj.write(buffer)
    buffer.seek(0)
    return buffer


def to_cif_string(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    as_bcif: bool = False,
    _allow_ambiguous_bond_annotations: bool = False,
) -> str | bytes:
    """Convert an AtomArray structure to a CIF formatted string.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        id (str): The ID of the entry. This will be used as the data block name.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        str | bytes: The CIF/BCIF formatted string/bytes representation of the structure.
    """
    return to_cif_buffer(
        structure,
        id=id,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
        as_bcif=as_bcif,
    ).getvalue()


def _to_cif_file(
    file_obj: pdbx.CIFFile | pdbx.BinaryCIFFile,
    path: os.PathLike,
    file_type: Literal["cif", "bcif", "cif.gz"] | None = None,
) -> str:
    # turn any relative path into an absolute path
    path = str(os.path.abspath(path))

    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _file_type_map = {
        # suffix: (open_func, open_mode, path_suffix)
        "cif": (open, "wt", ".cif"),
        "bcif": (open, "wb", ".bcif"),
        "cif.gz": (gzip.open, "wt", ".cif.gz"),
        "bcif.gz": (gzip.open, "wb", ".bcif.gz"),
        # ... default if no suffix is provided
        "": (gzip.open, "wt", ".cif.gz"),
    }
    if file_type is None:
        file_name = os.path.basename(path)
        file_type = file_name.split(".")[-1] if "." in file_name else ""
        # ... with gz suffix by fetching the pre-gz suffix
        file_type = file_name.split(".")[-2] + ".gz" if file_type == "gz" else file_type

    open_func, open_mode, path_suffix = _file_type_map[file_type]

    # ... check that the file ends with the correct suffix, otherwise mutate the path to end with the correct suffix
    if not path.endswith(path_suffix):
        path = path + path_suffix
    # ... get the name minus the suffix
    file_name = os.path.basename(path).replace(path_suffix, "")

    with open_func(path, mode=open_mode) as f:
        file_obj.write(f)

    return path


def to_cif_file(
    structure: AtomArray,
    path: os.PathLike,
    *,
    file_type: Literal["cif", "bcif", "cif.gz"] | None = None,
    id: str | None = None,
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = True,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    _allow_ambiguous_bond_annotations: bool = False,
) -> os.PathLike:
    """Convert an AtomArray structure to a CIF/BCIF formatted file.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        path (os.PathLike): The file path where the CIF formatted structure will be saved.
        file_type (Literal["cif", "bcif", "cif.gz"] | None): The file type to save the structure as.
            If None, the file type will be inferred from the path.
        id (str | None): The ID of the entry. This will be used as the data block name.
            If None, the data block name will be inferred from the path.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        str: The file path where the CIF formatted structure was saved.

    Raises:
        IOError: If there's an issue writing to the specified file path.
    """
    # turn any relative path into an absolute path
    path = str(os.path.abspath(path))
    file_name = os.path.basename(path)

    if file_type is None:
        if str(path).endswith(".cif"):
            file_type = "cif"
        elif str(path).endswith(".cif.gz"):
            file_type = "cif.gz"
        elif str(path).endswith(".bcif"):
            file_type = "bcif"
        elif str(path).endswith(".bcif.gz"):
            file_type = "bcif.gz"
        else:
            raise ValueError(f"Could not infer file type from path: {path}")

    file_obj = _to_cif_or_bcif(
        structure,
        id=id or file_name,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
        as_bcif=".bcif" in path,
    )

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


def _filter_extra_fields(extra_fields: list[str], atom_site: pdbx.CIFCategory) -> list[str]:
    """
    Filter the extra fields to only include fields that are actually present in the file.
    """
    _translate_builtin_fields = {
        "atom_id": "id",
        "charge": "pdbx_formal_charge",
        "b_factor": "B_iso_or_equiv",
        "occupancy": "occupancy",
    }
    _fields_with_default = {
        "label_entity_id",
        "auth_seq_id",
    }

    filtered_extra_fields = []
    for field in extra_fields:
        if field in _fields_with_default:
            filtered_extra_fields.append(field)
            continue
        if _translate_builtin_fields.get(field, field) in atom_site:
            filtered_extra_fields.append(field)
        else:
            logger.warning(f"Field {field} not found in file, ignoring.")

    return filtered_extra_fields


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
