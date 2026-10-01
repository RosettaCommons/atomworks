"""Entrypoint for parsing atomic-level structure files into Biotite-compatible data structures.

This module provides functionality for parsing PDB, CIF, and other structure files
into Biotite-compatible data structures with various processing options.

References:
    `Biotite Structure I/O <https://www.biotite-python.org/apidoc/biotite.structure.io.html>`_
    `mmCIF Format Specification <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/>`_
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.file import InvalidFileError
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx
from toolz import keyfilter

import atomworks.io.transforms.atom_array as ta
from atomworks.common import exists, string_to_md5_hash
from atomworks.constants import CCD_MIRROR_PATH, CRYSTALLIZATION_AIDS, WATER_LIKE_CCDS
from atomworks.io import template
from atomworks.io.transforms.categories import (
    category_to_dict,
    extract_crystallization_details,
    get_ligand_of_interest_info,
    get_metadata_from_category,
    initialize_chain_info_from_category,
    load_monomer_sequence_information_from_category,
)
from atomworks.io.utils.assembly import build_assemblies_from_asym_unit
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack, stack_any
from atomworks.io.utils.bonds import get_struct_conn_dict_from_atom_array
from atomworks.io.utils.ccd import check_ccd_codes_are_available
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import (
    apply_sharding_pattern,
    build_sharding_pattern,
    get_structure,
    infer_pdb_file_type,
    read_any,
)
from atomworks.io.utils.non_rcsb import (
    get_identity_assembly_gen_category,
    get_identity_op_expr_category,
    initialize_chain_info_from_atom_array,
)

logger = logging.getLogger("atomworks.io")

__all__ = ["parse"]

STANDARD_PARSER_ARGS = {
    "add_missing_atoms": True,
    "add_id_and_entity_annotations": True,
    "add_bond_types_from_struct_conn": ("covale",),
    "remove_ccds": tuple(CRYSTALLIZATION_AIDS),
    "remove_waters": True,
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "fix_formal_charges": True,
    "fix_bond_types": True,
    "convert_mse_to_met": True,  # Changed from False to True vs. atomworks.io.parser.parse default
    "hydrogen_policy": "keep",
    "model": None,  # all models
}
"""Common cif parser arguments for `atomworks.io.parse` for many biomolecular use cases.

Similar to the defaults below but additionally converts selenomethionine (MSE) residues to methionine (MET) residues,
which is desirable for many practical applications but would not be appropriate as a universal default.

This dictionary exists to provide a convenient import for the standard parameters.
"""

# Cache sharding configuration (internal, not exposed to parse() to avoid complexity)
_CACHE_SHARDING_DEPTH = 2  # Use 2-level sharding by default (e.g., ab/cd/abcdef123456/)
_CACHE_SHARDING_CHARS_PER_DIR = 2  # Number of characters per directory level


def _get_atomworks_version() -> str:
    """Lazy import of atomworks version to avoid circular imports."""
    try:
        from atomworks import __version__

        return __version__
    except ImportError:
        return "unknown"


def _parse_args_to_hash(parse_arguments: dict[str, Any], truncate: int = 8) -> str:
    """Compute hash from parse arguments with sorted keys."""
    args_string = ",".join(str(parse_arguments[k]) for k in sorted(parse_arguments.keys()))
    return string_to_md5_hash(args_string, truncate=truncate)


def _build_cache_file_path(
    cache_dir: Path,
    args_hash: str,
    filename: os.PathLike,
    assembly_info: str,
) -> Path:
    """Build sharded cache file path for parsed structure."""
    structure_id = Path(filename).stem

    # Pad structure ID to minimum required length for sharding
    min_length = _CACHE_SHARDING_DEPTH * _CACHE_SHARDING_CHARS_PER_DIR
    structure_id_padded = structure_id.ljust(min_length, "_")

    # Build sharded path
    sharding_pattern = build_sharding_pattern(depth=_CACHE_SHARDING_DEPTH, chars_per_dir=_CACHE_SHARDING_CHARS_PER_DIR)
    sharded_path = apply_sharding_pattern(structure_id_padded, sharding_pattern)

    return cache_dir / args_hash / sharded_path / f"{structure_id}_assembly_{assembly_info}.pkl.gz"


def parse(
    filename: os.PathLike | io.StringIO | io.BytesIO,
    *,
    file_type: Literal["cif", "pdb", "mmjson"] | None = None,
    ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH,
    cache_dir: os.PathLike | None = None,
    save_to_cache: bool = False,
    load_from_cache: bool = False,
    add_missing_atoms: bool = True,
    add_id_and_entity_annotations: bool = True,
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    remove_ccds: list[str] | None = None,
    remove_waters: bool = True,
    fix_ligands_at_symmetry_centers: bool = True,
    fix_arginines: bool = True,
    fix_formal_charges: bool = True,
    fix_bond_types: bool = True,
    convert_mse_to_met: bool = False,
    hydrogen_policy: Literal["keep", "remove", "infer"] = "keep",
    model: int | None = None,
    build_assembly: Literal["first", "all"] | list[str] | tuple[str] | None = "all",
    extra_fields: list[str] | Literal["all"] | None = None,
    keep_cif_block: bool = False,
) -> dict[str, Any]:
    """Entrypoint for general parsing of atomic-level structure files.

    Can either:
        - Directly load structure from file, using the specified keyword arguments;
        - Load the structure from a cached directory, re-building bioassemblies on-the-fly if necessary; or
        - Perform analogous cleaning/processing steps on an existing AtomArray or AtomArrayStack.

    We categorize arguments into two groups:
        - Wrapper arguments: Arguments that are used within the wrapping parse method (e.g., caching)
        - CIF parsing arguments: Arguments that control structure parsing and are ultimately are passed
            to the _parse_from_atom_array method (regardless of file type, we convert to an AtomArray before parsing)

    Args:
        filename (PathLike | io.StringIO | io.BytesIO): Either a Path or buffer to the file. This may be any format of
            atomic-level structure (e.g. .cif, .bcif, .cif.gz, .pdb), although .cif files are strongly recommended.

        **Wrapper arguments:**
        file_type (Literal["cif", "pdb", "mmjson"] | None, optional): The file type of the structure file.
            If not provided, the file type will be inferred automatically.
        load_from_cache (bool, optional): Whether to load pre-compiled results from cache. Defaults to False.
        cache_dir (PathLike, optional): Directory path to save pre-compiled results. Defaults to None.
        save_to_cache (bool, optional): Whether to save the results to cache when building the structure. Defaults to False.

        **Parsing arguments:**
        ccd_mirror_path (str, optional): Path to the local mirror of the Chemical Component Dictionary (recommended).
            If not provided, Biotite's built-in CCD will be used.
        add_missing_atoms (bool, optional): Whether to add missing atoms to the
            structure (from entirely or partially unresolved residues). Defaults to True.
        add_id_and_entity_annotations (bool, optional): Whether to add identifier and entity
            annotations to the structure. Defaults to True.
        add_bond_types_from_struct_conn (list, optional): A list of bond types to add to the structure
            from the `struct_conn` category. Defaults to `["covale"]`. This means that we will only
            add covalent bonds to the structure (excluding metal coordination and disulfide bonds).
        remove_ccds (list, optional): A list of CCD codes (e.g. `ALA`, `HEM`, ...) to remove from
            the structure. Defaults to crystallization aids. NOTE: Exclusion of polymer
            residues and common multi-chain ligands must be done with care to avoid sequence gaps.
        remove_waters (bool, optional): Whether to remove water molecules from the
            structure. Defaults to True.
        fix_ligands_at_symmetry_centers (bool, optional): Whether to patch non-polymer residues
            at symmetry centers that clash with themselves when transformed. Defaults to True.
        fix_arginines (bool, optional): Whether to fix arginine naming ambiguity, see the
            AF-3 supplement for details. Defaults to True.
        fix_formal_charges (bool, optional): Whether to fix formal charges on atoms involved in inter-residue bonds.
            Defaults to True.
        fix_bond_types (bool, optional): Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.
            Defaults to True.
        convert_mse_to_met (bool, optional): Whether to convert selenomethionine (MSE)
            residues to methionine (MET) residues. Defaults to False.
        hydrogen_policy (Literal, optional): Whether to keep, remove or infer hydrogens using
            biotite-hydride (will remove existing hydrogens and infer fresh).
            Defaults to "keep". Options: "keep", "remove", "infer".
        model (int, optional): The model number to parse for files with multiple models (e.g., NMR).
            Defaults to all models (None).
        build_assembly (string, list, or tuple, optional): Specifies which assembly to build, if any. Options are None
            (e.g., asymmetric unit), "first", "all", or a list or tuple of assembly IDs. Defaults to "all".
        extra_fields (list, optional): A list of extra fields to include in the AtomArrayStack. Defaults to None. "all" includes all fields.
            Only supports mmCIF files.
        keep_cif_block (bool, optional): Whether to keep the CIF block in the result. Defaults to False.

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
    if ccd_mirror_path and not os.path.exists(ccd_mirror_path):
        raise FileNotFoundError(
            f"Local mirror of the Chemical Component Dictionary does not exist: {ccd_mirror_path}. "
            "To use Biotite's built-in CCD, set `ccd_mirror_path` to None."
        )

    # Set default value for remove_ccds if None
    if remove_ccds is None:
        remove_ccds = CRYSTALLIZATION_AIDS

    # Argument validation
    check_ccd_codes_are_available(remove_ccds, ccd_mirror_path=ccd_mirror_path, mode="warn")

    if load_from_cache and not cache_dir:
        raise ValueError("Must provide a cache directory to load from cache")

    if save_to_cache and not cache_dir:
        raise ValueError("Must provide a cache directory to save to cache")

    if fix_formal_charges and not add_missing_atoms:
        logger.warning(
            "We can't fix formal charges without building from templates, as we need to know the true number of "
            "hydrogens bonded to a given atom, not the inferred number. This may lead to occasional inaccuracies "
            "after adding inter-residue bonds. To avoid this and fix formal charges, set `add_missing_atoms = True`."
        )

    file_type = file_type or infer_pdb_file_type(filename)
    is_buffer = isinstance(filename, io.StringIO | io.BytesIO)

    # Only load from / save to cache if we are not using a buffer
    if cache_dir and not is_buffer:
        # Build the cache file path, if necessary
        cache_dir = Path(cache_dir)

        # Prepare readable arguments dict for metadata
        parse_arguments = {
            "ccd_mirror_path": ccd_mirror_path,
            "add_missing_atoms": add_missing_atoms,
            "add_id_and_entity_annotations": add_id_and_entity_annotations,
            "add_bond_types_from_struct_conn": add_bond_types_from_struct_conn,
            "remove_ccds": remove_ccds,
            "remove_waters": remove_waters,
            "fix_ligands_at_symmetry_centers": fix_ligands_at_symmetry_centers,
            "fix_arginines": fix_arginines,
            "fix_formal_charges": fix_formal_charges,
            "fix_bond_types": fix_bond_types,
            "convert_mse_to_met": convert_mse_to_met,
            "hydrogen_policy": hydrogen_policy,
        }
        args_hash = _parse_args_to_hash(parse_arguments)

        # ... generate assembly info
        assembly_info = ",".join(build_assembly) if isinstance(build_assembly, list | tuple) else build_assembly

        # ... construct the full cache file path with sharding
        cache_file_path = _build_cache_file_path(cache_dir, args_hash, filename, assembly_info)

        # If we are loading from cache, try to load the result from the cache
        if load_from_cache:
            try:
                # Try to load the result from the cache
                if cache_file_path.exists():
                    # Load the result from the cache
                    result = pd.read_pickle(cache_file_path)

                    # Build assemblies
                    asym_unit = result["asym_unit"]
                    extra_info = result["extra_info"]
                    if "assembly_gen_category" in extra_info:
                        assemblies = build_assemblies_from_asym_unit(
                            assembly_gen_category=extra_info["assembly_gen_category"],
                            struct_oper_category=extra_info["struct_oper_category"],
                            asym_unit_atom_array_stack=asym_unit,
                            build_assembly=build_assembly,
                            fix_symmetry_centers=fix_ligands_at_symmetry_centers,
                        )
                    else:
                        assemblies = asym_unit

                    # Return updated result
                    result["assemblies"] = assemblies
                    return result
            except Exception as e:
                raise RuntimeError(f"Error loading from cache: {e}, tried path: {cache_file_path}") from e

    if file_type == "pdb":
        result = _parse_from_pdb(
            filename=filename,
            ccd_mirror_path=ccd_mirror_path,
            add_missing_atoms=add_missing_atoms,
            add_id_and_entity_annotations=add_id_and_entity_annotations,
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            remove_ccds=remove_ccds,
            remove_waters=remove_waters,
            fix_ligands_at_symmetry_centers=fix_ligands_at_symmetry_centers,
            fix_arginines=fix_arginines,
            fix_formal_charges=fix_formal_charges,
            fix_bond_types=fix_bond_types,
            convert_mse_to_met=convert_mse_to_met,
            hydrogen_policy=hydrogen_policy,
            model=model,
            build_assembly=build_assembly,
            extra_fields=extra_fields,
        )
    elif file_type in ("cif", "bcif", "mmjson"):
        result = _parse_from_cif(
            filename=filename,
            file_type=file_type,
            ccd_mirror_path=ccd_mirror_path,
            add_missing_atoms=add_missing_atoms,
            add_id_and_entity_annotations=add_id_and_entity_annotations,
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            remove_ccds=remove_ccds,
            remove_waters=remove_waters,
            fix_ligands_at_symmetry_centers=fix_ligands_at_symmetry_centers,
            fix_arginines=fix_arginines,
            fix_formal_charges=fix_formal_charges,
            fix_bond_types=fix_bond_types,
            convert_mse_to_met=convert_mse_to_met,
            hydrogen_policy=hydrogen_policy,
            model=model,
            build_assembly=build_assembly,
            extra_fields=extra_fields,
            keep_cif_block=keep_cif_block,
        )
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    if not is_buffer and save_to_cache and cache_dir and (not cache_file_path.exists()):
        # We want our cache to include:
        #   (1) All keys in `result` except the assemblies; and,
        #   (2) The information needed to rebuild the assembly(s), which is stored in `result["extra_info"]`; and,
        #   (3) The parse_arguments and atomworks version

        # Add parse_arguments and version to metadata before saving
        result.setdefault("metadata", {}).update(
            {"parse_arguments": parse_arguments, "atomworks.version": _get_atomworks_version()}
        )

        # Ensure all parent directories exist
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the result to the cache, excluding the assemblies
        result_to_cache = {k: v for k, v in result.items() if k != "assemblies"}
        pd.to_pickle(result_to_cache, cache_file_path)

    return result


def parse_atom_array(
    atom_array_or_stack: AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack,
    data_dict: dict | None = None,
    _cif_file: pdbx.CIFFile | pdbx.BinaryCIFFile | None = None,
    ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH,
    add_missing_atoms: bool = True,
    add_id_and_entity_annotations: bool = True,
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    remove_ccds: list[str] | None = CRYSTALLIZATION_AIDS,
    remove_waters: bool = True,
    fix_ligands_at_symmetry_centers: bool = True,
    fix_arginines: bool = True,
    fix_formal_charges: bool = True,
    fix_bond_types: bool = True,
    convert_mse_to_met: bool = False,
    hydrogen_policy: Literal["keep", "remove", "infer"] = "keep",
    build_assembly: Literal["first", "all"] | list[str] | tuple[str] | None = "all",
    extra_fields: list[str] | Literal["all"] | None = None,
) -> dict[str, Any]:
    """Parse, clean and augment an AtomArray or AtomArrayStack.

    AtomArrayPlus and AtomArrayPlusStack inputs are also supported, with some restrictions (see Notes).

    Args:
        atom_array_or_stack (AtomArray | AtomArrayStack): The AtomArray or AtomArrayStack to parse.
        data_dict (dict | None, optional): A dictionary to store the results of the parsing. If None, a new data_dict
            will be created.
        _cif_file (pdbx.CIFFile | pdbx.BinaryCIFFile | None, optional): The biotite CIF file object to use for parsing.
            Intended for internal use only. Defaults to None, corresponding to direct AtomArray parsing.
        build_assembly: Specifies which assembly to build. Options:
            - ``None``: Creates a single identity assembly (ID "1") with instance ID annotations
              (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``).
            - ``"first"``: Build only the first assembly defined in the file.
            - ``"all"``: Build all assemblies defined in the file.
            - ``list | tuple``: Build specific assemblies by their IDs (e.g., ``["1", "2"]``).
        **additional_kwargs: See `parse` documentation for details.

    Returns:
        Dictionary containing chain information, residue information, atom array, assemblies, and metadata.
        This method performs all aspects of ``_parse_from_cif`` that do not require a CIF file input.

    Note:
        When using AtomArrayPlus or AtomArrayPlusStack inputs, the following
        restrictions apply:

        - add_missing_atoms must be False
        - hydrogen_policy cannot be "infer"
        - convert_mse_to_met must be False
        - _cif_file must be None

        These restrictions ensure that 2D annotations remain aligned with atom indices.
    """

    # TODO: Support more arguments with AtomArrayPlus
    if isinstance(atom_array_or_stack, AtomArrayPlus | AtomArrayPlusStack):
        if exists(_cif_file):
            raise ValueError(
                "Providing a CIF file is not supported when parsing an AtomArrayPlus or AtomArrayPlusStack. "
                "Consider using parse() instead, which accepts CIF files directly."
            )
        if add_missing_atoms:
            raise ValueError(
                "Adding missing atoms is not supported when parsing an AtomArrayPlus or AtomArrayPlusStack. "
                "Convert to AtomArray using atom_array.as_atom_array() first, or pass add_missing_atoms=False."
            )
        if hydrogen_policy == "infer":
            raise ValueError(
                "Hydrogen inference is not supported when parsing an AtomArrayPlus or AtomArrayPlusStack. "
                "Convert to AtomArray using atom_array.as_atom_array() first, or use hydrogen_policy='keep' or 'remove'."
            )
        if convert_mse_to_met:
            raise ValueError(
                "MSE to MET conversion is not supported when parsing an AtomArrayPlus or AtomArrayPlusStack. "
                "Convert to AtomArray using atom_array.as_atom_array() first, or pass convert_mse_to_met=False."
            )

    if build_assembly == "_spoof":
        import warnings

        warnings.warn(
            "build_assembly='_spoof' is deprecated. Use build_assembly='all' or None instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        build_assembly = "all"

    # ... ensure that the input AtomArray or AtomArrayStack has a BondList
    if atom_array_or_stack.bonds is None:
        atom_array_or_stack.bonds = struc.BondList(atom_array_or_stack.array_length())

    # (Handle default lists to avoid mutable default arguments)
    remove_ccds = [] if remove_ccds is None else remove_ccds

    # We must perform argument validation if the function was called directly without a top-level call to `parse`
    if _cif_file is None:
        # CCD mirror
        if ccd_mirror_path and not os.path.exists(ccd_mirror_path):
            raise FileNotFoundError(
                f"Local mirror of the Chemical Component Dictionary does not exist: {ccd_mirror_path}. "
                "To use Biotite's built-in CCD, set `ccd_mirror_path` to None."
            )

        # Argument validation
        check_ccd_codes_are_available(remove_ccds, ccd_mirror_path=ccd_mirror_path, mode="warn")

    if not exists(data_dict):
        # (Default running dictionary, which we will populate through a series of Transforms)
        data_dict = {"extra_info": {}}

    if not exists(_cif_file):
        # Assemble spoofed metadata for formatting consistency
        data_dict["metadata"] = {}
        for key in ["method", "deposition_date", "release_date", "resolution", "extra_metadata"]:
            data_dict["metadata"][key] = None

        # Add an informative name
        data_dict["metadata"]["id"] = f"Parsed from AtomArray on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Initialize unused keys
        data_dict["cif_block"] = None
        data_dict["assemblies"] = {}
        data_dict["ligand_info"] = {"has_ligand_of_interest": False, "ligand_of_interest": []}

    if exists(extra_fields) and not exists(_cif_file):
        logger.warning("The `extra_fields` argument will be ignored if there is no CIF file input.")

    if "label_entity_id" not in atom_array_or_stack.get_annotation_categories():
        if "chain_entity" in atom_array_or_stack.get_annotation_categories():
            atom_array_or_stack.set_annotation("label_entity_id", np.copy(atom_array_or_stack.chain_entity))
        else:
            atom_array_or_stack.set_annotation(
                "label_entity_id", pdbx.convert._determine_entity_id(atom_array_or_stack.chain_id)
            )

    # If occupancy is not an annotation, add it, defaulting to 1.0
    if "occupancy" not in atom_array_or_stack.get_annotation_categories():
        atom_array_or_stack.set_annotation("occupancy", np.ones(atom_array_or_stack.array_length()))

    # ... ensure we have an atom array stack (e.g., if we selected a specific model, we may get an AtomArray)
    asym_unit_stack = ta.ensure_atom_array_stack(atom_array_or_stack)

    # ... remove any explicitly excluded residues (e.g., crystallization solvents, waters)
    if remove_ccds or remove_waters:
        # NOTE: If the excluded residues are part of a polymer chain, or part of a
        #  multi-chain ligand, this may create sequence gaps!
        # ... remove the residues we don't want to keep
        remove_ccds = set(map(str.upper, remove_ccds))
        if remove_waters:
            remove_ccds.update(WATER_LIKE_CCDS)

        asym_unit_stack = ta.remove_ccd_components(asym_unit_stack, remove_ccds)

    # ... initialize chain information from the first model (uses atom_array to build chain list)
    if exists(_cif_file) and "entity" in _cif_file.block and "entity_poly" in _cif_file.block:
        # We can get the chain entity-level information directly from the CIF file
        data_dict["chain_info"] = initialize_chain_info_from_category(_cif_file.block, asym_unit_stack[0])
    else:
        if "auth_seq_id" in asym_unit_stack.get_annotation_categories():
            # ... replace negative res_ids with auth_seq_id (as they are sometimes from AF-3 predictions)
            asym_unit_stack = ta.replace_negative_res_ids_with_auth_seq_id(asym_unit_stack)
        # ... infer the chain information from the AtomArray residue names (useful for inference; should not be used for RCSB files)
        data_dict["chain_info"] = initialize_chain_info_from_atom_array(
            asym_unit_stack[0], infer_chain_type=True, infer_chain_sequences=True
        )

    if "auth_seq_id" in asym_unit_stack.get_annotation_categories():
        # ... replace non-polymeric chain sequence ids with author sequence ids (since the non-polymer sequence ID's are not informative)
        asym_unit_stack = ta.update_nonpoly_seq_ids(asym_unit_stack, data_dict["chain_info"])

    if exists(_cif_file) and "entity_poly_seq" in _cif_file.block:
        # Use the `entity_poly_seq` category as ground-truth sequence for polymers, and the AtomArray as ground-truth for non-polymers
        data_dict["chain_info"] = load_monomer_sequence_information_from_category(
            cif_block=_cif_file.block,
            chain_info_dict=data_dict["chain_info"],
            atom_array=asym_unit_stack,
            ccd_mirror_path=ccd_mirror_path,
        )

    # Handle sequence heterogeneity by selecting the residue that appears last
    asym_unit_stack = ta.keep_last_residue(asym_unit_stack)

    # ... add the is_polymer annotation to the AtomArray
    asym_unit_stack = ta.add_polymer_annotation(asym_unit_stack, data_dict["chain_info"])

    # ... add the ChainType annotation to the AtomArray
    asym_unit_stack = ta.add_chain_type_annotation(asym_unit_stack, data_dict["chain_info"])

    # (Most examples, except NMR studies and small molecules, will not have any hydrogens)
    if hydrogen_policy == "keep":
        pass
    elif hydrogen_policy == "remove":
        asym_unit_stack = ta.remove_hydrogens(asym_unit_stack)
    elif hydrogen_policy == "infer":
        # infer hydrogens using biotite-hydride, will replace existing hydrogens
        asym_unit_stack = ta.add_hydrogen_atom_positions(asym_unit_stack)
    else:
        raise ValueError(f"Invalid hydrogen policy: {hydrogen_policy}. Must be 'keep', 'remove', or 'infer'.")

    models = []
    for model_idx in range(asym_unit_stack.stack_depth()):
        atom_array = asym_unit_stack[model_idx]

        # ... add any atoms that should be there based on the sequence information
        #     but may not be resolved. These will have occupancy 0.0 and `nan` coords.
        if add_missing_atoms:
            if extra_fields is not None:
                logger.warning(
                    "Adding missing atoms will erase extra fields. If you just want to load a structure with the given extra fields, "
                    "you should probably use the much faster 'load_any' function from atomworks.io.utils.io_utils instead of 'parse'. "
                    "Parse is meant for cleaning up structures from the RCSB PDB."
                )

            if exists(_cif_file):
                struct_conn_dict = category_to_dict(_cif_file.block, "struct_conn")
            else:
                struct_conn_dict = get_struct_conn_dict_from_atom_array(atom_array)

            atom_array = template.add_missing_atoms(
                atom_array,
                chain_info_dict=data_dict["chain_info"],
                struct_conn_dict=struct_conn_dict,
                add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
                remove_hydrogens=hydrogen_policy == "remove",
                use_ccd_charges=True,
                fix_formal_charges=fix_formal_charges,
                fix_bond_types=fix_bond_types,
            )

        # ... resolve arginine naming ambiguity
        if fix_arginines:
            atom_array = ta.resolve_arginine_naming_ambiguity(atom_array, raise_on_error=False)

        # ... convert MSE to MET
        if convert_mse_to_met:
            atom_array = ta.mse_to_met(atom_array)

        # ... add identifiers and entity annotations
        if add_id_and_entity_annotations:
            atom_array = ta.add_id_and_entity_annotations(atom_array)

        models.append(atom_array)

    # ... create an AtomArrayStack from the list of AtomArrays
    asym_unit_stack = stack_any(models)

    # ... add the atomic number annotation (vs. element, which is a string)
    asym_unit_stack = ta.add_atomic_number_annotation(asym_unit_stack)

    if "msa_path" in asym_unit_stack.get_annotation_categories():
        # ... add the MSA information to the chain info dictionary
        logger.info("MSA paths attribute detected in AtomArray. Adding to chain information...")

        for chain in data_dict["chain_info"]:
            msa_paths_in_chain = asym_unit_stack[asym_unit_stack.chain_id == chain].msa_path
            unique_path_in_chain = np.unique(msa_paths_in_chain)
            if len(unique_path_in_chain) > 1:
                raise ValueError(f"Multiple distinct MSA paths found for chain {chain}.")
            msa_path = unique_path_in_chain[0]
            if msa_path != "":
                data_dict["chain_info"][chain]["msa_path"] = Path(msa_path)

    # ... build assemblies and add assembly-specific annotations (instance IDs like `chain_iid`, `pn_unit_iid`, `molecule_iid`)
    if exists(build_assembly):
        assert build_assembly in ["first", "all"] or isinstance(
            build_assembly, list | tuple
        ), "Invalid `build_assembly` option. Must be 'first', 'all', or a list/tuple of assembly IDs as strings."

    # Determine assembly categories: use CIF data if build_assembly is set, otherwise identity operations
    if exists(build_assembly) and exists(_cif_file) and "pdbx_struct_assembly" in data_dict["cif_block"]:
        assembly_gen_category = data_dict["cif_block"]["pdbx_struct_assembly_gen"]
        struct_oper_category = data_dict["cif_block"]["pdbx_struct_oper_list"]
    else:
        assembly_gen_category = get_identity_assembly_gen_category(list(data_dict["chain_info"].keys()))
        struct_oper_category = get_identity_op_expr_category()

    # When build_assembly=None with identity ops, "all" builds the single identity assembly (ID "1")
    data_dict["assemblies"] = build_assemblies_from_asym_unit(
        assembly_gen_category=assembly_gen_category,
        struct_oper_category=struct_oper_category,
        asym_unit_atom_array_stack=asym_unit_stack,
        build_assembly=build_assembly if build_assembly is not None else "all",
        fix_symmetry_centers=fix_ligands_at_symmetry_centers,
    )

    # Store the assembly generation and struct oper categories in extra_info for caching and future reference
    data_dict["extra_info"]["assembly_gen_category"] = assembly_gen_category
    data_dict["extra_info"]["struct_oper_category"] = struct_oper_category

    # Handle instances where ph information is included in crystallization conditions
    if exists(_cif_file) and "exptl_crystal_grow" in _cif_file.block:
        crystal_key = "exptl_crystal_grow"
        crystal_dict = category_to_dict(_cif_file.block, crystal_key)
        data_dict["metadata"]["crystallization_details"] = extract_crystallization_details(crystal_dict)
    else:
        # No crystal growth section available in the CIF
        data_dict["metadata"]["crystallization_details"] = {"pH": None}

    if not exists(_cif_file):
        # Remove temporary annotations from the asym_unit_stack
        data_dict = _remove_tmp_annotations(data_dict, asym_unit_stack)

        # ... subset to only the keys we want to return, verbosely for clarity
        _keep_keys = {"chain_info", "ligand_info", "asym_unit", "assemblies", "metadata", "extra_info"}
        data_dict = keyfilter(lambda k: k in _keep_keys, data_dict)
    else:
        # If more processing may be performed, we just save the asym_unit_stack temporarily
        data_dict["asym_unit"] = asym_unit_stack

    return data_dict


def _parse_from_cif(
    filename: os.PathLike | io.StringIO | io.BytesIO, file_type: str | None = None, **kwargs
) -> dict[str, Any]:
    """Parse the CIF file.

    Return chain information, residue information, atom array, and metadata.
    See `parse` for details on the arguments and return values.

    NOTE: This method is not intended to be called directly; use `parse` instead.
    """
    # (Default running dictionary, which we will populate through a series of Transforms)
    data_dict = {"extra_info": {}}

    # ... read the CIF file into the dictionary (we will clean up the dictionary before returning)
    cif_file = read_any(filename, file_type=file_type)
    data_dict["cif_block"] = cif_file.block

    # ... load metadata into "metadata" key (either from RCSB standard fields, or from the custom `extra_metadata` field)
    if isinstance(filename, io.StringIO | io.BytesIO):
        fallback_filename = next(iter(cif_file.keys()))
    else:
        fallback_filename = Path(filename).stem
    data_dict["metadata"] = get_metadata_from_category(cif_file.block, fallback_id=fallback_filename)

    # ... load structure into the "asym_unit" key using the RCSB labels for sequence ids, and later update for non-polymers
    common_extra_fields = [
        "label_entity_id",
        "auth_seq_id",  # for non-polymer residue indexing
        "atom_id",
        "b_factor",
        "occupancy",
        "charge",
    ]

    if kwargs["extra_fields"] is not None:
        if kwargs["extra_fields"] != "all":
            common_extra_fields += kwargs["extra_fields"]
        else:
            common_extra_fields = "all"

    try:
        asym_unit_stack = get_structure(
            cif_file,
            extra_fields=common_extra_fields,
            model=kwargs["model"],
            add_bond_types_from_struct_conn=kwargs["add_bond_types_from_struct_conn"],
            fix_bond_types=kwargs["fix_bond_types"],
        )
    except InvalidFileError:
        logger.info("Invalid file error encountered; loading with only one model")
        # Try again, choosing only the first model
        asym_unit_stack = get_structure(
            cif_file,
            extra_fields=common_extra_fields,
            model=1,
            add_bond_types_from_struct_conn=kwargs["add_bond_types_from_struct_conn"],
            fix_bond_types=kwargs["fix_bond_types"],
        )

    # process the asym_unit_stack according to the given keyword arguments
    kwargs_to_pass = {k: v for k, v in kwargs.items() if k not in ["model", "file_type", "keep_cif_block"]}
    data_dict = parse_atom_array(asym_unit_stack, data_dict=data_dict, _cif_file=cif_file, **kwargs_to_pass)

    # Extract the asym_unit_stack from the returned data_dict
    asym_unit_stack = data_dict.pop("asym_unit")

    # ... get ligand of interest information
    data_dict["ligand_info"] = get_ligand_of_interest_info(data_dict["cif_block"])

    if "msa_paths_by_chain_id" in cif_file.block:
        # ... add the MSA information to the chain info dictionary
        logger.info("MSA paths detected in CIF file. Adding to chain information...")
        msa_paths_by_chain_id = category_to_dict(cif_file.block, "msa_paths_by_chain_id")
        for chain_id, msa_path in msa_paths_by_chain_id.items():
            data_dict["chain_info"][chain_id]["msa_path"] = Path(msa_path.item())

    # Remove temporary annotations from the asym_unit_stack
    data_dict = _remove_tmp_annotations(data_dict, asym_unit_stack)

    # ... subset to only the keys we want to return, verbosely for clarity
    _keep_keys = {"chain_info", "ligand_info", "asym_unit", "assemblies", "metadata", "extra_info"}
    if kwargs.get("keep_cif_block", False):
        _keep_keys.add("cif_block")
    data_dict = keyfilter(lambda k: k in _keep_keys, data_dict)

    return data_dict


def _parse_from_pdb(filename: os.PathLike, **parse_from_cif_kwargs) -> dict[str, Any]:
    """Parse a PDB file and return chain information, residue information, atom array, metadata, and legacy data.

    WARNING: We require that a single chain contains either polymer or non-polymer residues, but not both. Thus, if
    the PDB file contains a chain with both polymer and non-polymer residues, the non-polymer
    residues will be named with "$" appended to the chain ID (to not conflict with existing chains).

    WARNING: We assume that all residues are resolved (e.g., as is the case for computationally predicted structures). If not, use CIF files.

    NOTE: This method is not intended to be called directly; use `parse` instead.
    """
    # ...read the PDB file into a CIF block
    pdb_file = read_any(filename)
    atom_array_stack = pdb_file.get_structure(
        model=parse_from_cif_kwargs["model"],
        altloc="first",
        extra_fields=["b_factor", "occupancy", "charge", "atom_id"],
        include_bonds=True,
    )

    # ...if we have polymer and non-polymers on the same chain (as given by the HETATM field), we need to separate them for processing
    assert "hetero" in atom_array_stack.get_annotation_categories()

    hetero_atom_mask = atom_array_stack.get_annotation("hetero")
    if np.any(atom_array_stack.get_annotation("hetero")):
        # ...loop through chains and ensure the chain contains either polymer or non-polymer residues, but not both (as required by CIF files)
        original_chain_ids = np.unique(atom_array_stack.chain_id)
        chain_id_generator = create_chain_id_generator(unavailable_chain_ids=original_chain_ids)
        chain_ids = list(original_chain_ids)  # Creates a copy
        for chain_id in original_chain_ids:
            # ...check if we have blended `hetero` annotations in the chain
            chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
            if np.any(chain_hetero_annotations) and np.any(~chain_hetero_annotations):
                hetero_chain_id = next(chain_id_generator)
                logger.warning(
                    f"Chain {chain_id} contains both polymer and non-polymer residues; separating them for processing, naming the non-polymer residues as {hetero_chain_id}."
                )
                atom_array_stack.chain_id[(atom_array_stack.chain_id == chain_id) & hetero_atom_mask] = hetero_chain_id

                # Add the newly created chain ID to the list to avoid conflicts in future iterations
                chain_ids.append(hetero_chain_id)

            # ...ensure we don't have blended `hetero` annotations
            updated_chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
            assert np.all(updated_chain_hetero_annotations) or np.all(~updated_chain_hetero_annotations)

    # ... parse the CIF block into a dictionary
    parse_from_cif_kwargs["file_type"] = "pdb"
    parse_from_cif_kwargs["extra_fields"] = None
    # PDB files use identity assembly, so "all" builds just the single identity assembly
    parse_from_cif_kwargs["build_assembly"] = "all"

    kwargs_to_pass = {k: v for k, v in parse_from_cif_kwargs.items() if k not in ["model", "file_type"]}
    data_dict = parse_atom_array(atom_array_stack, _cif_file=None, **kwargs_to_pass)
    data_dict["metadata"]["id"] = Path(filename).stem.lower()

    return data_dict


# Helper functions
def _remove_annotation_if_exists(atom_array: struc.AtomArray | struc.AtomArrayStack, annotation: str) -> None:
    """Safely remove an annotation from an AtomArray or AtomArrayStack if it exists."""
    if annotation in atom_array.get_annotation_categories():
        atom_array.del_annotation(annotation)


def _remove_tmp_annotations(data_dict: dict, asym_unit_stack: AtomArrayStack) -> dict:
    """Clean the asym_unit_stack by removing unnecessary annotations.

    Also adds the clean asym_unit_stack to the data dictionary.
    """

    # ... remove annotations that are no longer needed to save memory
    _remove_annotations = {
        "leaving_atom_flag",
        "is_leaving_atom",
        "is_n_terminal_atom",
        "is_c_terminal_atom",
        "index",
    }
    for annotation in _remove_annotations:
        _remove_annotation_if_exists(asym_unit_stack, annotation)
        if "assemblies" in data_dict:
            for assembly in data_dict["assemblies"].values():
                _remove_annotation_if_exists(assembly, annotation)

    # ... add the asym_unit_stack to the data dict
    data_dict["asym_unit"] = asym_unit_stack

    return data_dict
