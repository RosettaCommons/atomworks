"""MetadataRowParser implementations for chain- and interface-based datasets."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from atomworks.common import as_list
from atomworks.constants import PDB_MIRROR_PATH
from atomworks.ml.datasets.parsers import MetadataRowParser


def build_path_from_template(path_template: str, **kwargs) -> Path:
    """Build a path using a template string with variable substitution and slicing.

    Args:
        path_template: Template string with {var} and {var[start:end]} patterns
        **kwargs: Variables to substitute

    Examples:
        >>> build_path_from_template("{base_dir}/{pdb_id[1:3]}/{pdb_id}", base_dir="/data", pdb_id="3usg")
        Path("/data/us/3usg")
    """
    # Find all template variables: {var} or {var[start:end]}
    pattern = r"\{([^}]+)\}"
    matches = re.finditer(pattern, path_template)

    # Initialize result with the template string (we will replace matches with the substituted values)
    result = path_template

    for match in reversed(list(matches)):
        var_expr = match.group(1)  # e.g., "pdb_id[1:3]" or "pdb_id"

        # ... extract variable name and slice group
        var_match = re.match(r"([^[]+)(?:\[([^\]]+)\])?", var_expr)
        assert var_match is not None, f"Invalid variable expression: {var_expr}"

        var_name = var_match.group(1)  # e.g., "pdb_id"
        slice_part = var_match.group(2)  # e.g., "1:3"

        assert var_name in kwargs, f"Variable '{var_name}' not found in kwargs"
        var_value = kwargs[var_name]

        if slice_part:
            if ":" in slice_part:
                # Slice notation (e.g., "1:3")
                start_str, end_str = slice_part.split(":")
                start = int(start_str) if start_str else None
                end = int(end_str) if end_str else None
                result_value = var_value[start:end]
            else:
                # Single index (e.g., "0")
                index = int(slice_part)
                result_value = var_value[index]
        else:
            # No slicing, use full variable
            result_value = var_value

        # Replace the match with the result
        result = result[: match.start()] + str(result_value) + result[match.end() :]

    return Path(result)


def find_existing_file_path(
    base_dirs: Sequence[Path | str], file_extensions: Sequence[str], path_templates: Sequence[str], pdb_id: str
) -> Path:
    """Find the first existing file path by trying corresponding base_dirs, file_extensions, and path_templates in order."""
    assert len(base_dirs) == len(file_extensions) == len(path_templates), "All lists must have the same length"

    for base_dir, file_extension, path_template in zip(base_dirs, file_extensions, path_templates, strict=False):
        base_dir = Path(base_dir)
        candidate_path = build_path_from_template(
            path_template, pdb_id=pdb_id, base_dir=base_dir, file_extension=file_extension
        )
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"No file found for pdb_id {pdb_id} using any of the provided path templates. Last tried: {candidate_path}"
    )


class PNUnitsDFParser(MetadataRowParser):
    # TODO: Deprecate in favor of GenericDFParser

    """Parser for pn_units DataFrame rows.

    In addition to standard fields (example_id, path), this parser also includes:
        - The query pn_unit instance ID, which is used to center the crop.
        - The assembly ID, which is used to load the correct assembly from the CIF file.
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(
        self,
        base_dir: Path | str | list[Path | str] | tuple[Path | str, ...] = PDB_MIRROR_PATH,
        file_extension: str | list[str] | tuple[str, ...] = ".cif.gz",
        path_template: str | list[str] | tuple[str, ...] = "{base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}",
    ):
        """Initialize the PNUnitsDFParser.

        Args:
            base_dir: Base directory for PDB files. Can be a single path or a list/tuple of paths to try in order.
            file_extension: File extension for PDB files. Can be a single extension or a list/tuple of extensions corresponding to the base_dir(s).
            path_template: Template string for path construction (perfect for Hydra configs). Can be a single template or a list/tuple of templates corresponding to the base_dir(s) and file_extension(s).
                Example: "{base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}" (default)
        """
        self.base_dirs = [Path(bd) for bd in as_list(base_dir)]
        self.file_extensions = as_list(file_extension)
        self.path_templates = as_list(path_template)

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # For the Query DF, the query pn_unit is the only pn_unit in the query
        query_pn_unit_iids = [row["q_pn_unit_iid"]]

        # Build the path to the CIF file
        pdb_id = str(row["pdb_id"])  # Ensure we get a string from the Series

        # Find the first existing file path
        path = find_existing_file_path(self.base_dirs, self.file_extensions, self.path_templates, pdb_id)

        # Put the full row in the extra info dictionary
        extra_info = row.to_dict()

        return {
            "example_id": row["example_id"],
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": query_pn_unit_iids,  # Where to center the crop; if more than one, center the crop on the interface
            "extra_info": extra_info,
        }


class InterfacesDFParser(MetadataRowParser):
    # TODO: Deprecate in favor of GenericDFParser

    """Parser for interfaces DataFrame rows.

    In addition to standard fields (example_id, path), this parser also includes:
        - The two query pn_unit instance IDs, as a list, which are used to sample the interface during cropping.
        - The assembly ID, which is used to load the correct assembly from the CIF file.
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(
        self,
        base_dir: Path | str | list[Path | str] | tuple[Path | str, ...] = PDB_MIRROR_PATH,
        file_extension: str | list[str] | tuple[str, ...] = ".cif.gz",
        path_template: str | list[str] | tuple[str, ...] = "{base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}",
    ):
        """Initialize the InterfacesDFParser.

        Args:
            base_dir: Base directory for PDB files. Can be a single path or a list/tuple of paths to try in order.
            file_extension: File extension for PDB files. Can be a single extension or a list/tuple of extensions corresponding to the base_dir(s).
            path_template: Template string for path construction (perfect for Hydra configs). Can be a single template or a list/tuple of templates corresponding to the base_dir(s) and file_extension(s).
                Example: "{base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}" (default)
        """
        self.base_dirs = [Path(bd) for bd in as_list(base_dir)]
        self.file_extensions = as_list(file_extension)
        self.path_templates = as_list(path_template)

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # For the Interfaces DF, there are two query pn_units
        query_pn_unit_iids = [row["pn_unit_1_iid"], row["pn_unit_2_iid"]]

        # Build the path to the CIF file
        pdb_id = str(row["pdb_id"])  # Ensure we get a string from the Series

        # Find the first existing file path
        path = find_existing_file_path(self.base_dirs, self.file_extensions, self.path_templates, pdb_id)

        # Put the full row in the extra info dictionary
        extra_info = row.to_dict()

        return {
            "example_id": row["example_id"],
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": query_pn_unit_iids,  # Where to center the crop; if more than one, center the crop on the interface
            "extra_info": extra_info,
        }


class GenericDFParser(MetadataRowParser):
    """Generic dataframe parser for training or validation dataframes.

    We parse an input row (e.g., a Pandas Series) and return a dictionary containing pertinent information for the Transform pipeline.

    Args:
        example_id_colname: Name of the column containing a unique identifier for each example (across ALL datasets, not just this dataset).
            By convention, the columns values should be generated with ``atomworks.ml.common.generate_example_id``. Default: "example_id"
        path_colname: Name of the column containing paths (relative or absolute) to the relevant structure files. Default: "path"
        pn_unit_iid_colnames: The name(s) of the column(s) containing the CIFUtils pn_unit_iid(s); used for cropping.
            If given as a list, should contain one element for a monomers dataset and two for an interfaces dataset.
            Default: None (crop randomly)
        assembly_id_colname: Optional parameter giving the name of the column containing the assembly ID.
            If None, the assembly ID will be set to "1" for all examples. Default: None
        base_path: The base path to the files, if not included in the path.
        extension: The file extension of the structure files, if not included in the path.
        attrs: Additional attributes to be merged with the dataframe-level attributes stored in the DF (if present). Attributes
            in this dictionary will take precedence over those in the dataset-level attributes and will be returned in the "extra_info" key.

    Returns:
        dict: A dictionary containing:

            example_id
                The unique identifier for the example. Must be unique across all datasets.
            path
                The composed path to the structure file, including the base path and extension if specified.
            query_pn_unit_iids
                The pn_unit_iid(s) that inform where to crop the structure.
                During TRAINING, we typically want to specify the chain(s) or interface at which to center our crop. If not given (i.e., None),
                then we will crop the structure at a random location, if a crop is required.
                During VALIDATION, then we do not crop, and query_pn_unit_iids should be None.
            assembly_id
                The assembly ID. Used to load the correct assembly from the CIF file. If not given, the assembly ID will be set to "1".
            extra_info
                A dictionary containing all additional information that should be passed to the Transform pipeline. Contains, in order of precedence:

                - Any additional key-value pairs specified by the ``attrs`` parameter
                - All unused dataframe columns (i.e., those not used for example_id, path, query_pn_unit_iids, or assembly_id)
                - Dataset-level attributes (if present), found in the ``attrs`` attribute of the Dataframe (or Series)
                For example, the "extra_info" key could contain information about which chain(s) to score during validation, metadata for specific metrics, etc.

    Note:
        We must avoid duplication of interfaces due to order inversion. If not using the preprocessing
        scripts in ``atomworks.ml``, ensure that the interfaces dataframe has been checked for duplicates.
        For example, [A, B] and [B, A] should be considered the same interface.

    Example:
        Example dataframe:
        example_id                      path                      pn_unit_1_iid  pn_unit_2_iid
        {['my-dataset']}{ex_1}{1}{[A_1,B_1]}  /path/to/structure_1.cif  A_1            B_1
        {['my-dataset']}{ex_2}{2}{[C_1,B_1]}  /path/to/structure_2.cif  C_1            B_1
    """

    def __init__(
        self,
        example_id_colname: str = "example_id",
        path_colname: str = "path",
        pn_unit_iid_colnames: str | list[str] | None = None,
        assembly_id_colname: str | None = None,
        base_path: str = "",
        extension: str = "",
        attrs: dict | None = None,
    ):
        # Columns to extract
        self.example_id_colname = example_id_colname
        self.path_colname = path_colname

        if isinstance(pn_unit_iid_colnames, str):
            self.pn_unit_iid_colnames = [pn_unit_iid_colnames]
        elif pn_unit_iid_colnames is None:
            self.pn_unit_iid_colnames = []
        else:
            self.pn_unit_iid_colnames = pn_unit_iid_colnames

        self.assembly_id_colname = assembly_id_colname

        self.attrs = attrs.copy() if attrs else {}
        # (For clarity, we explicitly expose base_path and extension, but just treat them as additional attributes)
        if base_path and "base_path" not in self.attrs:
            self.attrs["base_path"] = base_path
        if extension and "extension" not in self.attrs:
            self.attrs["extension"] = extension

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # Compose the metadata (extra_info) dictionary:
        # ... dataframe-level attributes; lowest precedence
        extra_info = row.attrs.copy() if hasattr(row, "attrs") else {}
        # ... row-level attributes
        extra_info.update(row.to_dict())
        # ... parser attributes; highest precedence
        extra_info.update(self.attrs or {})

        # Assemble input pn_units (to inform cropping)
        query_pn_unit_iids = [row[colname] for colname in self.pn_unit_iid_colnames]

        # Get the assembly ID if specified, otherwise default to "1"
        assembly_id = row[self.assembly_id_colname] if self.assembly_id_colname is not None else "1"

        # Compose the path to the structure file, including the base path and extension if specified
        path = Path(str(row[self.path_colname]))
        if extra_info.get("base_path"):
            path = Path(extra_info["base_path"]) / path
        if extra_info.get("extension"):
            path = path.with_suffix(extra_info["extension"])

        # (Exclude columns that we've already used from the extra_info dictionary)
        exclude_cols = set(
            self.pn_unit_iid_colnames
            + [self.example_id_colname, self.path_colname]
            + ([self.assembly_id_colname] if self.assembly_id_colname else [])
            + ["base_path", "extension"]
        )

        return {
            "example_id": row[self.example_id_colname],
            "path": path,
            "assembly_id": assembly_id,
            "query_pn_unit_iids": query_pn_unit_iids,
            "extra_info": {k: v for k, v in extra_info.items() if k not in exclude_cols},
        }
