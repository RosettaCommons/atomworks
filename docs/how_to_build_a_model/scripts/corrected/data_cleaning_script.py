"""
Script to read in parquet files and clean the data needed for the
"How to Build a Model Using AtomWorks" tutorial. This script is meant to be run
from the command line and takes the path to the parquet files and the path to
your PDB mirror as arguments. It saves the cleaned data and the train/val/test
splits as parquet files for future use in the tutorial.

Example usage:
    python data_cleaning_script.py /path/to/parquet/files /path/to/pdb_mirror

Last edited: July 9, 2026
"""

import os
import sys

import numpy as np  # BUGFIX: np was used (np.where, np.random) but never imported.
import pandas as pd
import numpy as np


def read_in_parquet_file(path_to_parquets: str | os.PathLike, parquet_file: str) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a parquet file.
    :param path_to_parquets: Directory containing the parquet files.
    :param parquet_file: Name of the parquet file (without the .parquet extension).
    :return: A pandas DataFrame with the parquet contents.
    """
    file_path = os.path.join(path_to_parquets, (parquet_file + ".parquet"))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    return pd.read_parquet(file_path)


def assign_split(cluster):
    """Assign a protein cluster to a split, using the cluster sets defined below."""
    if cluster in train_clusters:
        return "train"
    if cluster in val_clusters:
        return "val"
    if cluster in test_clusters:
        return "test"
    return "unassigned"  # Should already be filtered out; useful as a debug flag.


# ── Parse command-line arguments ─────────────────────────────────────────────
if len(sys.argv) > 2:
    path_to_parquets = sys.argv[1]
    PDB_MIRROR_PATH = sys.argv[2]
else:
    print(
        "To use this script, please provide the path to the parquet files and "
        "the path to your PDB mirror as command line arguments."
    )
    sys.exit(1)

# ── Read the parquet files ───────────────────────────────────────────────────
interfaces = read_in_parquet_file(path_to_parquets, "interfaces")
pn_units = read_in_parquet_file(path_to_parquets, "pn_units")

# ── Merge PN-unit metadata onto each interface (once per side) ───────────────
# pn_unit_iid is only unique within a (pdb_id, assembly_id) scope, so merge on
# all three keys, twice (once for each PN unit in the interface).
u1_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()
u2_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()

u1_cols = u1_cols.rename(columns={
    "pn_unit_iid": "pn_unit_1_iid",
    "is_polymer": "u1_is_polymer",
    "num_resolved_residues": "u1_num_resolved_residues",
})
u2_cols = u2_cols.rename(columns={
    "pn_unit_iid": "pn_unit_2_iid",
    "is_polymer": "u2_is_polymer",
    "num_resolved_residues": "u2_num_resolved_residues",
})

df = interfaces.merge(u1_cols, on=["pdb_id", "assembly_id", "pn_unit_1_iid"], how="inner")
df = df.merge(u2_cols, on=["pdb_id", "assembly_id", "pn_unit_2_iid"], how="inner")

# ── Filter to drug-like protein-ligand interfaces ────────────────────────────
df = df[df["involves_loi"]]
df = df[df["is_inter_molecule"]]
df = df[~df["involves_metal"]]
df = df[~df["involves_covalent_modification"]]
df = df[df["u1_is_polymer"] != df["u2_is_polymer"]]
df = df[(df["u1_num_resolved_residues"] + df["u2_num_resolved_residues"]) < 200]

# Remove duplicate interface rows.
df = df.drop_duplicates(subset=["pdb_id", "assembly_id", "pn_unit_1_iid", "pn_unit_2_iid"])

# ── Add a unique example identifier ──────────────────────────────────────────
df["example_id"] = (
    df["pdb_id"] + "_" +
    df["assembly_id"].astype(str) + "_" +
    df["pn_unit_1_iid"] + "_" +
    df["pn_unit_2_iid"]
)
assert df["example_id"].nunique() == len(df), "example_id is not unique!"

# ── Add the path to each structure in the PDB mirror ─────────────────────────
# BUGFIX: the original overwrote PDB_MIRROR_PATH (from sys.argv[2]) with an
# os.environ lookup that fell back to "/PATH/TO/pdb_mirror", silently ignoring
# the command-line argument. Use the argument the user actually passed in.
df["path"] = df["pdb_id"].str.lower().map(
    lambda x: f"{PDB_MIRROR_PATH}/{x[1:3]}/{x}.cif.gz"
)

df.to_parquet("cleaned_data.parquet")

# ── Split into train/val/test by protein cluster ─────────────────────────────
protein_clusters = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "protein_cluster_30"]].copy()

df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_1_iid",
        "protein_cluster_30": "u1_cluster",
    }),
    on=["pdb_id", "assembly_id", "pn_unit_1_iid"],
    how="left",
)
df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_2_iid",
        "protein_cluster_30": "u2_cluster",
    }),
    on=["pdb_id", "assembly_id", "pn_unit_2_iid"],
    how="left",
)

# The polymer side carries the cluster; the ligand side is null. Take whichever
# side is the polymer.
df["protein_cluster"] = np.where(df["u1_is_polymer"], df["u1_cluster"], df["u2_cluster"])

# Drop interfaces with no assigned cluster (RNA/DNA, short peptides, low quality).
df = df[df["protein_cluster"].notna()].reset_index(drop=True)

# Shuffle the unique clusters (seeded for reproducibility) and split 80/10/10.
unique_clusters = df["protein_cluster"].unique()
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_clusters)

n = len(unique_clusters)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
# test gets the remainder to avoid off-by-one gaps

train_clusters = set(unique_clusters[:n_train])
val_clusters = set(unique_clusters[n_train:n_train + n_val])
test_clusters = set(unique_clusters[n_train + n_val:])

df["split"] = df["protein_cluster"].map(assign_split)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val = df[df["split"] == "val"].reset_index(drop=True)
df_test = df[df["split"] == "test"].reset_index(drop=True)

os.makedirs("splits", exist_ok=True)
df_train.to_parquet("splits/train.parquet", index=False)
df_val.to_parquet("splits/val.parquet", index=False)
df_test.to_parquet("splits/test.parquet", index=False)
