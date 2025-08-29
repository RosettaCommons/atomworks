"""Row parser for non-standard metadata dataframes"""

from pathlib import Path
from typing import Any

import pandas as pd

from atomworks.constants import PDB_MIRROR_PATH
from atomworks.ml.datasets.parsers import MetadataRowParser


class AF2FB_DistillationParser(MetadataRowParser):  # noqa: N801
    # TODO: Deprecate in favor of GenericDFParser

    """
    DEPRECATION WARNING: This parser is deprecated and will be removed in a future release.
    We should use the GenericDFParser instead, providing `path` and `example_id` columns.

    Parser for AF2FB distillation metadata.

    The AF2FB distillation dataset is provided courtesy of Meta/Facebook.
    It contains ~7.6 Mio AF2 predicted structures from UniRef50.

    Metadata (i.e. which sequences, which cluster identities @ 30% seq.id,
    whether a sequence has an msa & template, sequence_hash etc.) are stored
    in the `af2_distillation_facebook.parquet` dataframe.

    The parquet has the following columns:
        - example_id
        - n_atoms
        - n_res
        - mean_plddt
        - min_plddt
        - median_plddt
        - sequence_hash
        - has_msa
        - msa_depth
        - has_template
        - cluster_id
        - seq (!WARNING: this is a relatively data-heavy column)
    """

    def __init__(self, base_dir: str, file_extension: str = ".cif"):
        """
        Initialize the AF2FB_DistillationParser.

        This parser is designed to handle the AF2FB distillation dataset, which contains
        approximately 7.6 million AlphaFold2 predicted structures from UniRef50.

        Args:
            - base_dir (str): The base directory where the AF2FB distillation dataset is stored.
                Defaults to "/squash/af2_distillation_facebook", which is stored on `tukwila` for
                ML model training.
            - file_extension (str): The file extension of the structure files. Defaults to ".cif".

        Raises:
            - AssertionError: If the specified dataset directory does not exist.
        """
        self.dataset_dir = Path(base_dir)
        self.file_extension = file_extension
        assert self.dataset_dir.exists(), f"Dataset directory {self.dataset_dir} does not exist."

    @staticmethod
    def _get_shard_from_hash(hash_value: str) -> str:
        """Due to the size of the AF2FB dataset, we store it with 2-level sharding.

        The two layers of sharding is an optimization technique for faster filesystem
        performance. (Do not put more than 10k files in any directory).

        Example:
            - example_id: UniRef50_A0A1S3ZVX8
            - sequence_hash: f771c39dfbf

        therefore the two level shard is `f7/71/` and the files can be found at
            -  ./cif/f7/71/UniRef50_A0A1S3ZVX8.cif
            -  ./msa/f7/71/f771c39dfbf.a3m
            -  ./template/f7/71/f771c39dfbf.atab
        """
        return f"{hash_value[:2]}/{hash_value[2:4]}/"

    def _parse(self, row: pd.Series) -> dict:
        example_id = row["example_id"]
        sequence_hash = row["sequence_hash"]

        path = (
            self.dataset_dir / "cif" / self._get_shard_from_hash(sequence_hash) / f"{example_id}{self.file_extension}"
        )

        return {
            "example_id": example_id,
            "path": path,
            "assembly_id": "1",  # just default to the first assembly (=identity if none given)
            "sequence_hash": sequence_hash,
        }


class ValidationDFParserLikeAF3(MetadataRowParser):
    # TODO: Deprecate in favor of GenericDFParser

    """
    Parser for AF-3-style validation DataFrame rows.

    As output, we give:
        - pdb_id: The PDB ID of the structure.
        - assembly_id: The assembly ID of the structure, required to load the correct assembly from the CIF file.
        - path: The path to the CIF file.
        - example_id: An identifier that combines the pdb_id and assembly_id.
        - ground_truth: A dictionary containing non-feature information for loss and validation. For validation, we initialize with the following:
            - interfaces_to_score: A list of tuples like (pn_unit_iid_1, pn_unit_iid_2, interface_type), which represent low-homology interfaces to score.
            - pn_units_to_score: A list of tuples like (pn_unit_iid, pn_unit_type), which represent low-homology pn_units to score.
    """

    def __init__(self, base_dir: Path = PDB_MIRROR_PATH, file_extension: str = ".cif.gz"):
        self.base_dir = base_dir
        self.file_extension = file_extension

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # Build the path to the CIF file
        pdb_id = row["pdb_id"]
        path = Path(f"{self.base_dir}/{pdb_id[1:3]}/{pdb_id}{self.file_extension}")

        # Extract the interfaces and pn_units to score

        # Example: [(A_1, B_1, "protein-protein"), (B_1, C_1, "protein-ligand")]
        interfaces_to_score = (
            eval(row["interfaces_to_score"])
            if isinstance(row["interfaces_to_score"], str)
            else [eval(interface) for interface in row["interfaces_to_score"]]
        )
        # Example: [(A_1, "protein"), (B_1, "DNA")]
        pn_units_to_score = (
            eval(row["pn_units_to_score"])
            if isinstance(row["pn_units_to_score"], str)
            else [eval(unit) for unit in row["pn_units_to_score"]]
        )

        return {
            "example_id": row["example_id"],
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "ground_truth": {
                "interfaces_to_score": interfaces_to_score,
                "pn_units_to_score": pn_units_to_score,
            },
        }
