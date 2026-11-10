"""Prepare protein-ligand interface metadata for the model-building tutorial.

Example:
    python data_cleaning_script.py \
        data/pdb_metadata/shared/interfaces_df.parquet \
        --pdb-mirror /path/to/pdb_mirror \
        --output-dir splits
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from atomworks.enums import ChainType

REQUIRED_INTERFACE_COLUMNS = {
    "pdb_id",
    "assembly_id",
    "altloc_seed",
    "pn_unit_1_iid",
    "pn_unit_2_iid",
    "pn_unit_1_type",
    "pn_unit_2_type",
    "pn_unit_1_is_polymer",
    "pn_unit_2_is_polymer",
    "involves_loi",
    "is_inter_molecule",
    "involves_metal",
    "involves_covalent_modification",
    "example_id",
    "path",
}
PN_UNIT_COLUMNS = {
    "pdb_id",
    "assembly_id",
    "altloc_seed",
    "q_pn_unit_iid",
    "q_pn_unit_num_resolved_residues",
    "cluster",
}
# This tutorial excludes cyclic pseudo-peptides from the broader protein grouping.
PROTEIN_CHAIN_TYPES = {int(ChainType.POLYPEPTIDE_D), int(ChainType.POLYPEPTIDE_L)}
MAX_RESOLVED_RESIDUES = 200


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="Path to the interface metadata Parquet file.")
    parser.add_argument(
        "--pn-units", type=Path, help="PN-unit metadata Parquet file; inferred from the interface filename by default."
    )
    parser.add_argument(
        "--pdb-mirror",
        type=Path,
        default=os.environ.get("PDB_MIRROR_PATH"),
        help="PDB mirror root. Defaults to PDB_MIRROR_PATH.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"), help="Directory for output Parquet files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to shuffle protein clusters.")
    args = parser.parse_args()
    if args.pdb_mirror is None:
        parser.error("set PDB_MIRROR_PATH or pass --pdb-mirror")
    if args.pn_units is None:
        if "interfaces" not in args.metadata.name:
            parser.error("pass --pn-units when the interface filename does not contain interfaces")
        args.pn_units = args.metadata.with_name(args.metadata.name.replace("interfaces", "pn_units", 1))
    return args


def load_metadata(interfaces_path: Path, pn_units_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the interfaces and PN-unit tables, checking the interfaces schema first."""
    missing = REQUIRED_INTERFACE_COLUMNS.difference(pq.read_schema(interfaces_path).names)
    assert not missing, f"{interfaces_path} is missing required columns: {sorted(missing)}"

    interfaces = pd.read_parquet(interfaces_path)
    pn_units = pd.read_parquet(pn_units_path)
    return interfaces, pn_units


def merge_pn_unit_info(interfaces: pd.DataFrame, pn_units: pd.DataFrame) -> pd.DataFrame:
    """Merge resolved-residue counts and cluster labels in for each side of the interface."""
    pn_columns = pn_units[
        ["pdb_id", "assembly_id", "altloc_seed", "q_pn_unit_iid", "q_pn_unit_num_resolved_residues", "cluster"]
    ]

    side_1 = pn_columns.rename(
        columns={
            "q_pn_unit_iid": "pn_unit_1_iid",
            "q_pn_unit_num_resolved_residues": "pn_unit_1_num_resolved_residues",
            "cluster": "pn_unit_1_cluster",
        }
    )
    side_2 = pn_columns.rename(
        columns={
            "q_pn_unit_iid": "pn_unit_2_iid",
            "q_pn_unit_num_resolved_residues": "pn_unit_2_num_resolved_residues",
            "cluster": "pn_unit_2_cluster",
        }
    )

    df = interfaces.merge(side_1, on=["pdb_id", "assembly_id", "altloc_seed", "pn_unit_1_iid"], how="left")
    df = df.merge(side_2, on=["pdb_id", "assembly_id", "altloc_seed", "pn_unit_2_iid"], how="left")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep small, noncovalent protein-ligand interfaces and drop duplicates."""
    df = df[df["involves_loi"].eq(True)]
    df = df[df["is_inter_molecule"].eq(True)]
    df = df[df["involves_metal"].ne(True)]
    df = df[df["involves_covalent_modification"].ne(True)]

    exactly_one_polymer = df["pn_unit_1_is_polymer"] ^ df["pn_unit_2_is_polymer"]
    df = df[exactly_one_polymer]

    protein_side_type = np.where(df["pn_unit_1_is_polymer"], df["pn_unit_1_type"], df["pn_unit_2_type"])
    df = df[pd.Series(protein_side_type, index=df.index).isin(PROTEIN_CHAIN_TYPES)]

    residue_columns = ["pn_unit_1_num_resolved_residues", "pn_unit_2_num_resolved_residues"]
    df = df.dropna(subset=residue_columns)
    df = df[df[residue_columns].sum(axis="columns") < MAX_RESOLVED_RESIDUES]

    df = df.drop_duplicates(subset=["example_id"])
    assert df["example_id"].is_unique
    return df


def resolve_structure_paths(df: pd.DataFrame, pdb_mirror: Path) -> pd.DataFrame:
    """Overwrite `path` with each structure's location in the local PDB mirror."""
    pdb_mirror = pdb_mirror.expanduser().resolve()

    def resolve_path(pdb_id: str) -> str:
        pdb_id = pdb_id.lower()
        return str(pdb_mirror / pdb_id[1:3] / f"{pdb_id}.cif.gz")

    df = df.copy()
    df["path"] = df["pdb_id"].map(resolve_path)
    return df


def assign_cluster_splits(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Add a `protein_cluster` column and assign each cluster to one 80/10/10 split."""
    df = df.copy()
    df["protein_cluster"] = np.where(df["pn_unit_1_is_polymer"], df["pn_unit_1_cluster"], df["pn_unit_2_cluster"])
    df = df[df["protein_cluster"].notna()].reset_index(drop=True)

    unique_clusters = df["protein_cluster"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(unique_clusters)

    n = len(unique_clusters)
    n_train = int(0.8 * n)
    n_validation = int(0.1 * n)
    # test gets the remainder to avoid off-by-one gaps

    train_clusters = set(unique_clusters[:n_train])
    validation_clusters = set(unique_clusters[n_train : n_train + n_validation])
    test_clusters = set(unique_clusters[n_train + n_validation :])

    def assign_split(cluster: object) -> str:
        if cluster in train_clusters:
            return "train"
        if cluster in validation_clusters:
            return "validation"
        if cluster in test_clusters:
            return "test"
        return "unassigned"  # rows where cluster was null

    df["split"] = df["protein_cluster"].map(assign_split)
    assert df.groupby("protein_cluster")["split"].nunique().eq(1).all()
    return df


def write_splits(df: pd.DataFrame, output_dir: Path) -> None:
    """Write cleaned metadata and individual split files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "all.parquet", index=False)
    for split in ("train", "validation", "test"):
        df[df["split"] == split].reset_index(drop=True).to_parquet(output_dir / f"{split}.parquet", index=False)


def main() -> None:
    """Run the metadata preparation workflow."""
    args = parse_args()
    interfaces, pn_units = load_metadata(args.metadata, args.pn_units)
    df = merge_pn_unit_info(interfaces, pn_units)
    df = clean_data(df)
    df = resolve_structure_paths(df, args.pdb_mirror)
    df = assign_cluster_splits(df, seed=args.seed)
    write_splits(df, args.output_dir)

    counts = df["split"].value_counts()
    print(f"Wrote {len(df):,} examples to {args.output_dir}")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {counts.get(split, 0):,}")


if __name__ == "__main__":
    main()
