"""Functional loader implementations for AtomWorks datasets.

Loaders are functions that process raw dataset output (e.g., pandas Series) into a Transform-ready format.
E.g., converts what may be dataset-specific metadata into a standard format for use in AtomWorks Transform pipelines.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from toolz import keyfilter

from atomworks.io.parser import parse
from atomworks.ml.utils.io import apply_sharding_pattern


def _construct_metadata_hierarchy(row: pd.Series, attrs: dict | None = None) -> dict[str, Any]:
    """Construct metadata dictionary with precedence hierarchy.

    Assembles metadata from multiple sources with the following precedence (lowest to highest priority):
        1. DataFrame-level attributes (row.attrs)
        2. Row-level data (row.to_dict())
        3. Loader-specific attributes (attrs parameter)

    Args:
        row: pandas Series representing one dataset example
        attrs: Optional loader-specific attributes to merge with highest precedence

    Returns:
        Dictionary containing merged metadata with proper hierarchy
    """
    # Start with DataFrame-level attributes (lowest precedence)
    extra_info = row.attrs.copy() if hasattr(row, "attrs") else {}

    # Add row-level data (middle precedence)
    extra_info.update(row.to_dict())

    # Add loader-specific attributes (highest precedence)
    extra_info.update(attrs or {})

    return extra_info


def _construct_structure_path(
    path: str, base_path: str | None, extension: str | None, sharding_pattern: str | None = None
) -> Path:
    """Construct file path with optional base_path, extension, and sharding pattern.

    Args:
        path: The base path or identifier (e.g., PDB ID)
        base_path: Base directory to prepend
        extension: File extension to add/replace
        sharding_pattern: Pattern for organizing files in subdirectories
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)
    """
    sharded_path = apply_sharding_pattern(path, sharding_pattern)

    if base_path:
        final_path = Path(base_path) / sharded_path
    else:
        final_path = sharded_path

    if extension:
        final_path = final_path.with_suffix(extension)

    return final_path


def _load_structure_from_path(path: Path, assembly_id: str, parser_args: dict | None = None) -> dict[str, Any]:
    """Load structure from file path using the CIF parser."""
    result_dict = parse(
        filename=path,
        build_assembly=(assembly_id,),
        **(parser_args or {}),
    )
    return result_dict


def create_base_loader(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    attrs: dict | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a base loader with common logic for many AtomWorks datasets.

    Args:
        example_id_colname: Name of column containing unique example identifiers
        path_colname: Name of column containing paths to structure files
        assembly_id_colname: Optional column name containing assembly IDs.
            If None, assembly_id defaults to "1" for all examples.
        attrs: Additional attributes to merge with highest precedence into the metadata hierarchy
            (and ultimately included in the output dictionary's "extra_info" key).
        base_path: Base path to prepend to file paths if not included in path column
        extension: File extension to add/replace if not included in path column
        sharding_pattern: Pattern for how files are organized in subdirectories, if not specified in the path
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)
        parser_args: Optional dictionary of arguments to pass to the CIF parser when loading the structure file.

    Returns:
        A function that takes a pandas Series and returns a dictionary of the loaded structure.
    """

    def loader_function(row: pd.Series) -> dict[str, Any]:
        # Prepare loader-specific attributes
        loader_attrs = attrs.copy() if attrs else {}
        if base_path and "base_path" not in loader_attrs:
            loader_attrs["base_path"] = base_path
        if extension and "extension" not in loader_attrs:
            loader_attrs["extension"] = extension

        extra_info = _construct_metadata_hierarchy(row, loader_attrs)

        assembly_id = (
            row[assembly_id_colname] if assembly_id_colname is not None and assembly_id_colname in row else "1"
        )
        path = _construct_structure_path(
            row[path_colname], extra_info.get("base_path"), extra_info.get("extension"), sharding_pattern
        )
        result_dict = _load_structure_from_path(path, assembly_id, parser_args)

        # Remove used columns from extra_info to avoid duplication in the output dictionary
        exclude_cols = (
            [example_id_colname, path_colname]
            + ([assembly_id_colname] if assembly_id_colname else [])
            + ["base_path", "extension"]
        )
        extra_info = keyfilter(lambda k: k not in exclude_cols, extra_info)

        return {
            # ... from the row and metadata hierarchy
            "example_id": row[example_id_colname],
            "path": path,
            "assembly_id": assembly_id,
            "extra_info": extra_info,
            # ... from the CIF parser
            "atom_array": result_dict["assemblies"][assembly_id][0],  # First model
            "atom_array_stack": result_dict["assemblies"][assembly_id],  # All models
            "chain_info": result_dict["chain_info"],
            "ligand_info": result_dict["ligand_info"],
            "metadata": result_dict["metadata"],
        }

    return loader_function


def create_loader_with_query_pn_units(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    pn_unit_iid_colnames: str | list[str] | None = None,
    assembly_id_colname: str | None = "assembly_id",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a generic loader for pipelines with query pn_units (chains).

    For instance, in the interfaces dataset, each sampled row contains two pn_unit instance IDs
    that should be included in the cropped structure.

    Examples:
        Interfaces dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"], assembly_id_colname="assembly_id"
            ... )

        Chains dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames="pn_unit_iid", base_path="/data/structures", extension=".cif.gz"
            ... )
    """
    # Normalize pn_unit_iid_colnames to list format
    if isinstance(pn_unit_iid_colnames, str):
        pn_unit_iid_colnames = [pn_unit_iid_colnames]
    pn_unit_iid_colnames = pn_unit_iid_colnames or []

    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    def loader_function(row: pd.Series) -> dict[str, Any]:
        # Get base loader dictionary with common functionality
        result = base_loader(row)
        result["extra_info"] = keyfilter(lambda k: k not in pn_unit_iid_colnames, result["extra_info"])

        # Add query-specific fields
        query_pn_unit_iids = [row[colname] for colname in pn_unit_iid_colnames]
        result["query_pn_unit_iids"] = query_pn_unit_iids

        return result

    return loader_function


def create_loader_with_interfaces_and_pn_units_to_score(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    interfaces_to_score_colname: str | None = "interfaces_to_score",
    pn_units_to_score_colname: str | None = "pn_units_to_score",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a loader that adds interfaces and pn_units to score for validation datasets.

    Example:
        >>> loader = create_loader_with_interfaces_and_pn_units_to_score(
        ...     interfaces_to_score_colname="interfaces_to_score", pn_units_to_score_colname="pn_units_to_score"
        ... )
    """
    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    def loader_function(row: pd.Series) -> dict[str, Any]:
        # Get base loader dictionary with common functionality
        result = base_loader(row)
        result["extra_info"] = keyfilter(
            lambda k: k not in [interfaces_to_score_colname, pn_units_to_score_colname], result["extra_info"]
        )

        # Add validation-specific fields
        interfaces_to_score = row[interfaces_to_score_colname] if interfaces_to_score_colname is not None else None
        pn_units_to_score = row[pn_units_to_score_colname] if pn_units_to_score_colname is not None else None

        result.update(
            {
                "interfaces_to_score": interfaces_to_score,
                "pn_units_to_score": pn_units_to_score,
            }
        )

        return result

    return loader_function
