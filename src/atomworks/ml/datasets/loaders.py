"""Functional loader implementations for AtomWorks datasets.

Loaders are functions that process raw dataset output (e.g., pandas Series) into a Transform-ready format.
E.g., converts what may be dataset-specific metadata into a standard format for use in AtomWorks Transform pipelines.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from toolz import keyfilter


def build_metadata_hierarchy(row: pd.Series, attrs: dict | None = None) -> dict[str, Any]:
    """Build up metadata dictionary with precedence hierarchy.

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


def build_structure_path(path: str, base_path: str | None, extension: str | None) -> Path:
    """Construct final structure file path with optional base_path and extension.

    Args:
        path: The base file path as a string
        base_path: Optional base path to prepend to the file path
        extension: Optional file extension to add or replace
    """
    # Start with the raw path
    final_path = Path(path)

    # Prepend base_path if specified
    if base_path:
        final_path = Path(base_path) / final_path

    # Add or replace extension if specified
    if extension:
        final_path = final_path.with_suffix(extension)

    return final_path


def loader_with_query_pn_units(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    pn_unit_iid_colnames: str | list[str] | None = None,
    assembly_id_colname: str | None = None,
    base_path: str = "",
    extension: str = "",
    attrs: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a generic loader for pipelines with query pn_units (chains).

    For instance, in the interfaces dataset, each sampled row contains two pn_unit instance IDs
    that should be included in the cropped structure.

    The returned function can be passed to any AtomWorks Dataset, e.g., PandasDataset(loader=...).

    Args:
        example_id_colname: Name of column containing unique example identifiers
        path_colname: Name of column containing paths to structure files
        pn_unit_iid_colnames: Column name(s) containing pn_unit instance IDs for cropping.
            Can be a string, list of strings, or None for random cropping.
        assembly_id_colname: Optional column name containing assembly IDs.
            If None, assembly_id defaults to "1" for all examples.
        base_path: Base path to prepend to file paths if not included in path column
        extension: File extension to add/replace if not included in path column
        attrs: Additional attributes to merge with highest precedence

    Returns:
        Loader function that processes pandas Series into Transform-ready dict format

    Examples:
        Basic usage (e.g., no query chains that we want to include in crop; no assembly_id specified):
            >>> loader = loader_with_query_pn_units()
            >>> dataset = PandasDataset(data=df, loader=loader, name="my_dataset")

        Interfaces dataset:
            >>> loader = loader_with_query_pn_units(
            ...     pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"], assembly_id_colname="assembly_id"
            ... )

        Chains dataset:
            >>> loader = loader_with_query_pn_units(
            ...     pn_unit_iid_colnames="pn_unit_iid", base_path="/data/structures", extension=".cif.gz"
            ... )
    """
    # Normalize pn_unit_iid_colnames to list format
    if isinstance(pn_unit_iid_colnames, str):
        pn_unit_iid_colnames = [pn_unit_iid_colnames]
    pn_unit_iid_colnames = pn_unit_iid_colnames or []

    # Prepare loader-specific attributes
    loader_attrs = attrs.copy() if attrs else {}
    if base_path and "base_path" not in loader_attrs:
        loader_attrs["base_path"] = base_path
    if extension and "extension" not in loader_attrs:
        loader_attrs["extension"] = extension

    def loader_function(row: pd.Series) -> dict[str, Any]:
        # Build metadata hierarchy
        extra_info = build_metadata_hierarchy(row, loader_attrs)

        # Extract query pn_unit IDs for cropping
        query_pn_unit_iids = [row[colname] for colname in pn_unit_iid_colnames]

        # Get assembly ID, defaulting to "1" if not specified
        assembly_id = row[assembly_id_colname] if assembly_id_colname is not None else "1"

        path = build_structure_path(row[path_colname], extra_info.get("base_path"), extra_info.get("extension"))

        # Filter out used columns from extra_info
        exclude_cols = set(
            pn_unit_iid_colnames
            + [example_id_colname, path_colname]
            + ([assembly_id_colname] if assembly_id_colname else [])
            + ["base_path", "extension"]
        )
        filtered_extra_info = keyfilter(lambda k: k not in exclude_cols, extra_info)

        return {
            "example_id": row[example_id_colname],
            "path": path,
            "assembly_id": assembly_id,
            "query_pn_unit_iids": query_pn_unit_iids,
            "extra_info": filtered_extra_info,
        }

    return loader_function
