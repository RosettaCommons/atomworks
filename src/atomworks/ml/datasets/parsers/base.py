from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from atomworks.constants import CRYSTALLIZATION_AIDS
from atomworks.io import parse

DEFAULT_PARSER_ARGS = {
    "add_missing_atoms": True,
    "add_id_and_entity_annotations": True,
    "add_bond_types_from_struct_conn": ["covale"],
    "remove_ccds": CRYSTALLIZATION_AIDS,
    "remove_waters": True,
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "fix_formal_charges": True,
    "fix_bond_types": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "model": None,  # all models
}
"""Default cif parser arguments for `atomworks.io.parse`.
This dictionary exists to provide a convenient import for the default parameters.
"""


class MetadataRowParser(ABC):
    """Abstract base class for MetadataRowParsers.

    A MetadataRowParser is a class that parses a row from a DataFrame on disk into a format digestible by the `load_example_from_metadata_row` function.

    In the common case that a model is trained on multiple datasets, each with their own dataframe and base data format,
    we must ensure that the data pipeline receives a consistent input format. By way of example, when training an
    AF-3-style model, we might have a "PDB Chains" dataset of `mmCIF` files, a "PDB Interfaces" dataset of `mmCIF`
    files, and a `distillation` dataset of computationally-generated `PDB` files, and many others.

    We enforce the following common schema for all datasets:
        - "example_id": A unique identifier for the example within the dataset.
        - "path": The path to the data file (which we will load with CIFUtils).

    WARNING: For many transforms, additional keys are required. For example:
        - For cropping, the `query_pn_unit_iids` field is used to center the crop on the interface or pn_unit.
          If not provided, the AF-3-style crop transforms will crop randomly.
        - For loading templates, the "pdb_id" is required to load the correct template from disk (at least with the legacy code).
    """

    required_schema: ClassVar[dict[str, type]] = {
        "example_id": str,
        "path": Path,
        "extra_info": dict,
    }

    def parse(self, row: pd.Series) -> dict[str, Any]:
        """Wrapper to parse and validate a DataFrame row."""
        output = self._parse(row)

        # Validate output
        self.validate_output(output)
        return output

    @abstractmethod
    def _parse(self, row: pd.Series) -> dict[str, Any]:
        """Abstract method to be implemented by subclasses for parsing a DataFrame row.

        Args:
            row (pd.Series): DataFrame row to parse.

        Returns:
            dict[str, Any]: Parsed output dictionary, including required keys.
        """
        pass

    def validate_output(self, output: dict[str, Any]) -> None:
        """Validate the output dictionary for required keys and their types."""
        for key in self.required_schema:
            if key not in output:
                if key == "extra_info":
                    output[key] = {}  # Default to an empty dictionary
                else:
                    raise ValueError(f"Missing key in output: {key}")

        for key, expected_type in self.required_schema.items():
            if not isinstance(output[key], expected_type):
                raise TypeError(f"Key '{key}' has incorrect type: expected {expected_type}, got {type(output[key])}")


def load_example_from_metadata_row(
    metadata_row: pd.Series,
    metadata_row_parser: MetadataRowParser,
    *,
    cif_parser_args: dict | None = None,
) -> dict:
    """Load training/validation example from a DataFrame row into a common format using the given metadata row parsing function
    and CIF parser arguments.

    Performs the following steps:
        (1) Parse the row into a common dictionary format using the provided row parsing function and metadata row.
        (2) Load the CIF file from the information in the common dictionary format (i.e., the "path" key).
        (3) Combine the parsed row data and the loaded CIF data into a single dictionary.

    Args:
        metadata_row (pd.Series): The DataFrame row to parse.
        metadata_row_parser (MetadataRowParser): The parser to use for converting the row into a dictionary format.
        cif_parser_args (dict, optional): Additional arguments for the CIF parser. Defaults to None.

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    # Load common outputs from the dataframe using the provided parsing function
    # See the `MetadataRowParser` class for more details on the expected output schema
    parsed_row = metadata_row_parser.parse(metadata_row)

    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    # TODO: Move to DEFAULT_CIF_PARSER_ARGS, but set to False by default not True
    if cif_parser_args.get("cache_dir"):
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    # Merge DEFAULT_CIF_PARSER_ARGS with cif_parser_args, overriding with the keys present in cif_parser_args
    merged_cif_parser_args = {**DEFAULT_PARSER_ARGS, **cif_parser_args}

    # Use the parse function with the merged CIF parser arguments
    result_dict = parse(
        filename=parsed_row["path"],
        build_assembly=(parsed_row["assembly_id"],),  # Convert list to tuple (make hashable)
        **merged_cif_parser_args,
    )

    # Combine the PDB output and the parsed output into our clean representation
    data = {
        # ...from the data frame (at a minimum, example_id and path)
        **parsed_row,
        # ...from the CIF parser
        "atom_array": result_dict["assemblies"][parsed_row["assembly_id"]][0],  # First model
        "atom_array_stack": result_dict["assemblies"][parsed_row["assembly_id"]],  # All models
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }

    return data
