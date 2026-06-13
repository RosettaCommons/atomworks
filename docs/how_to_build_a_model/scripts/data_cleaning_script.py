"""
Script to read in parquet files and clean the data needed for the "How to Build a Model Using AtomWorks" tutorial. This script is meant to be run from the command line and takes in the path to the parquet files as an argument. It saves the cleaned data as a parquet file for future use in the tutorial.

Example usage:
    python data_cleaning_script.py /path/to/parquet/files

Last edited: April 14, 2026
"""

import pandas as pd
import os
import sys

def read_in_parquet_file(path_to_parquets: str | os.PathLike, parquet_file: str) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from a parquet file.
    :param path_to_parquets: The path to the directory containing the parquet files.
    :param parquet_file: The name of the parquet file (without the .parquet extension).
    :return: A pandas DataFrame containing the data from the parquet file.
    """                                                                  
    file_path = os.path.join(path_to_parquets, (parquet_file + ".parquet"))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    mydf = pd.read_parquet(file_path)

    return mydf  

path_to_parquets = None
PDB_MIRROR_PATH = None

if len(sys.argv) > 2:
    path_to_parquets = sys.argv[1]
    PDB_MIRROR_PATH = sys.argv[2]
else:
    print("To use this script, please provide the path to the parquet files and the path to your PDB mirror as command line arguments.")
    sys.exit(1)

def assign_split(cluster):
    """
    Helper function to assign each cluster to a split (train, val, or test) based on the sets of clusters we defined earlier.
    :param cluster: The protein cluster to assign a split to.
    :return: The split that the cluster belongs to (train, val, test, or unassigned).
    """
    # TODO: shouldn't we already have removed any unassigned clusters in the previous step? 
    if cluster in train_clusters: return "train"
    if cluster in val_clusters:   return "val"
    if cluster in test_clusters:  return "test"
    return "unassigned"  # rows where cluster was null. For this tutorial, these should already be filtered out, but it's a good debugging tool!


####################################################
# Reading in the parquet files and cleaning the data.
####################################################

interfaces = read_in_parquet_file(path_to_parquets, "interfaces")
pn_units = read_in_parquet_file(path_to_parquets, "pn_units")

# We need to combine the information in the interfaces and pn_units datasets to get the data for each pn_unit involved in each interface. 
# However, the label "pn_unit_iid" in the pn_units dataset is not unique - it is only unique for a given pdb_id and assembly_id. 
# So when we merge these dataframes we need to merge on all three keys and we need to do it twice - once for the first pn_unit in each interface and once for the second pn_unit in each interface.

# Saving the particular columns we want to merge for the first set of pn_units
# We also keep the "is_polymer" column to filter our data later.
u1_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()

u2_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()

# Renaming the columns in these new dataframes to merge on the pn_unit_iid labels (they are called "pn_unit_1_iid" and "pn_unit_2_iid" in the interfaces dataset) and to distinguish the other columns for the first and second pn_units in each interface.
u1_cols = u1_cols.rename(columns={"pn_unit_iid": "pn_unit_1_iid",
    "is_polymer": "u1_is_polymer",
    "num_resolved_residues": "u1_num_resolved_residues"})

u2_cols = u2_cols.rename(columns={"pn_unit_iid": "pn_unit_2_iid",
    "is_polymer": "u2_is_polymer",
    "num_resolved_residues": "u2_num_resolved_residues"})

# Merge these three dataframes together to get the data for each pn_unit involved in each interface. 

df = interfaces.merge(u1_cols, on=["pdb_id", "assembly_id", "pn_unit_1_iid"], how="inner")
df = df.merge(u2_cols, on=["pdb_id", "assembly_id", "pn_unit_2_iid"], how="inner")

# Now we will filter the data to only keep drug-like protein-ligand pockets: 
# involves_loi==True so that at least one of the pn_units is a "ligand of interest"
df = df[df["involves_loi"]]
# is_inter_molecule==True to ensure the two pn_units belong to different molecules
df = df[df["is_inter_molecule"]]
# involves_metal==False to avoid interfaces that are metal-mediated
df = df[~df["involves_metal"]]
# involves_covalent_modification==False to avoid covalently modified residues
df = df[~df["involves_covalent_modification"]]
# only keep interfaces where one pn_unit is a polymer and the other is not (should be a small molecule ligand). This wil avoid protein-protein, ligand-ligand, and RNA/DNA interfaces.
df = df[df["u1_is_polymer"] != df["u2_is_polymer"]]
# total of num_resolved_residues < 200 to keep the size tractable for training
df = df[(df["u1_num_resolved_residues"] + df["u2_num_resolved_residues"]) < 200]

# Remove duplicate rows
df = df.drop_duplicates(subset=["pdb_id", "assembly_id", "pn_unit_1_iid", "pn_unit_2_iid"])

# Add a unique identifier for each interface. This will be useful later in the tutorial. 
df["example_id"] = (
    df["pdb_id"] + "_" +
    df["assembly_id"].astype(str) + "_" +
    df["pn_unit_1_iid"] + "_" +
    df["pn_unit_2_iid"]
)
assert df["example_id"].nunique() == len(df), "example_id is not unique!"

# Add path information to match each entry to the location of the files in your PDB mirror.
PDB_MIRROR_PATH = os.environ.get("PDB_MIRROR_PATH", "/PATH/TO/pdb_mirror")
df["path"] = df["pdb_id"].str.lower().map(
    lambda x: f"{PDB_MIRROR_PATH}/{x[1:3]}/{x}.cif.gz"
)

# Save the cleaned data as a parquet file for future use in the tutorial.
df.to_parquet("cleaned_data.parquet")

####################################################
# Splitting the data into training, testing, and validation sets. 
####################################################
# There are many ways to do this, but for the tutorial we have decided to use the information stored in the column "protein_cluster_30" to split the data. This column groups proteins by 30% sequence identity. We will use this information to ensure that all interfaces involving proteins from the same cluster are in the same set (training, testing, or validation). This will help us to better evaluate the generalizability of our model to new proteins.

protein_clusters = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "protein_cluster_30"]].copy()

# merge this with u1 and u2 since either could be the protein side of the interface. We us a left merge for this to only keep the keys from the "left" (df) dataframe. We will keep all rows in df and just add the cluster information where it exists. 
# Note: ligands will have a null cluster.

# Merge cluster for u1 side
df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_1_iid","protein_cluster_30": "u1_cluster"}),
    on=["pdb_id", "assembly_id", "pn_unit_1_iid"],
    how="left"
)
# Merge cluster for u2 side
df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_2_iid",
        "protein_cluster_30": "u2_cluster"}),
    on=["pdb_id", "assembly_id", "pn_unit_2_iid"],
    how="left"
)

# Get the protein_cluster information from whichever side is the polymer: 
df["protein_cluster"] = np.where(
    df["u1_is_polymer"],
    df["u1_cluster"],
    df["u2_cluster"]
)
# TODO ask Hope about this step. It seems like the condition is only for if u1 is a polymer - what if u2 is the polymer?

# There may be cases where no clusters were assigned, for example low-quality entries. Let's remove these from our dataset:
df = df[df["protein_cluster"].notna()].reset_index(drop=True)

# Now we can finally split the data based on the protein_cluster information. We will use 80% of the clusters for training, 10% for testing, and 10% for validation.
# Shuffle the the clusters randomly. A seed is specified for reproducibility.
unique_clusters = df["protein_cluster"].dropna().unique()
# TODO: didn't we already drop these in the previous step? 
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_clusters)

# split
n = len(unique_clusters)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
# test gets the remainder to avoid off-by-one gaps

train_clusters = set(unique_clusters[:n_train])
val_clusters   = set(unique_clusters[n_train : n_train + n_val])
test_clusters  = set(unique_clusters[n_train + n_val :])

# create datasets for each
df["split"] = df["protein_cluster"].map(assign_split)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val   = df[df["split"] == "val"].reset_index(drop=True)
df_test  = df[df["split"] == "test"].reset_index(drop=True)

# Save datasets as parquet files in separate directories
os.makedirs("splits", exist_ok=True)
df_train.to_parquet("splits/train.parquet", index=False)
df_val.to_parquet("splits/val.parquet",     index=False)
df_test.to_parquet("splits/test.parquet",   index=False)





