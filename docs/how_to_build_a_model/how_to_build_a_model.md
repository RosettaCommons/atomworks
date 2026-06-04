# How to Build a Model Using AtomWorks
## Part 1: Cleaning the Data

### Table of Contents

(aw_build_model_p1_intro)=
### Introduction
This is the first in a series of tutorials that walks you through how to use AtomWorks to build a machine learning model for protein design from start to finish. 

**In this installment, you will learn how to use the [IO functionalities in AtomWorks] to prepare your data for use in a machine learning model.**

By the end of this tutorial series you will have cleaned data and built a graph neural network to create plausible bound poses between a ligand and a protein pocket. 

```{important}
This tutorial will walk you through creating a script using the AtomWorks API. 

For those that want to use the tutorial text as structure and hints to write your own script, we have hidden the code in collapsible cells. 

If you would like to see the full script, it is provided in the tutorial files. <!-- TODO: Create a tutorials file folder and upload once ready. -->
```

(aw_build_model_p1_prereq)=
### Prerequisites
Before starting this tutorial it is assumed that you have:
- An intermediate knowledge of Python and the Pandas library.
- A working installation of AtomWorks. Only the `io` side will be used for this part of the How to Build a Model tutorial series, however other parts will require the `ml` side.
- At least 1GB of space for storage of the parquet files
- Having the PDB mirror set up requires ~100GB of space

```{note}
If you do not have over 100GB of space on your computing system, you can use a subset of the PDB instead of the full PDB mirror. See [Data Mirrors](../mirrors.rst) for how to only download specific PDB IDs.
```

(aw_build_model_p1_setup)=
### Setup 
AtomWorks provides a few parquet files that already contains various pieces of metadata about the various structures included in the PDB. We will use these as our starting point.

Download and decompress the parquet files via: 
```bash
wget https://files.ipd.uw.edu/pub/atomworks/dfs/pdb/2026_01_06.tar.gz
tar -xvf 2026_01_06.tar.gz
```
After decompressing the folder you should see three parquet files: 
- **`assemblies.parquet`:** There are multiple bio assemblies stored in a single PDB, each assembly will have multiple chains, interfaces, etc. 
- **`interfaces.parquet`:** Contains metadata for all binary interfaces in the PDB
- **`pn_units.parquet`:** Contains metadata for each PN unit <!-- TODO: define --> in the [PDB](https://www.rcsb.org/)

We will only use `interfaces.parquet` and `pn_units.parquet` in this tutorial.

<!-- TODO: ask hope for the code that generated this -->

Later on you will also need a mirror to at least a subset of the PDB so that AtomWorks is able to find the related structure when training your model. You can learn how to set up a PDB mirror using AtomWorks [here](../mirrors.rst). This document also provides further information on the parquet files. 

(aw_build_model_p1_tutorial)=
### Creating Cleaned Parquet Files

(aw_build_model_p1_load)=
#### Loading the parquet file using Pandas
Let's first take a look at the information contained in the parquet file. Parquet files are not human parsable, but we can use [Pandas](https://pandas.pydata.org/) to inspect it. 

````{dropdown} Click to see the code
Load in the datasets: 
```python
import pandas as pd

interfaces = pd.read_parquet("2026_01_06/interfaces.parquet")
pn_units = pd.read_parquet("2026_01_06/pn_units.parquet")
```
View the dataset columns: 
```python
interfaces.columns
pn_units.columns
```
````

Let's take a closer look at a few of the columns in the interfaces parquet: <!-- TODO: check if these descriptions are true-->
- `pdb_id`: The identifier for the specific structure in the PDB.
- `assembly_id`: Integer label for what assembly the given interface belongs to. There may be multiple assemblies in a single PDB structure.
- `pn_unit_1_iid`: Label for the first PN unit in the interface.
- `pn_unit_2_iid`: Label for the second PN unit in the interface.
- `involves_loi`: Boolean for if the LOI (Ligand of Interest) is part of the interface.
- `is_inter_molecule`: Boolean for if the interfaces is between two molecules (True) or within the same molecule (False).
- `involves_metal`: Boolean for if the interface involves a metal atom.
- `involves_covalent_modification`: Boolean for if the interface involves a covalent modification (e.g. glycosylation).
- `num_contacts`: Number of contacts between the two PN units that create the interface.
- `min_distance`: Minimum distance between the contacts that create the interface.

Let's also look at a few columns of interest from the PN units parquet file: 
- `pn_unit_iid`: Label for the specific PN unit in the structure that the row corresponds to. 
- `is_polymer`: Boolean for whether the PN unit is a polymer
- `num_resolved_residues`: Number of resolved residues in the structure.

We encourage you to take a closer look at these datasets on your own. For some suggestions on what to do, see the collapsible group below: 

````{dropdown} Click to see the code
See the first 5 rows of specific columns in a dataset:
```python
interfaces[["pdb_id","assembly_id", "pn_unit_1_iid", "pn_unit_2_iid"]].head()
```
See the unique values of a given column: 
```python
interfaces["assembly_id"].unique()
```
Determine the datatype of the data stored in a particular column: 
```python
interfaces[["is_inter_molecule"]].dtype
```
````

(aw_build_model_p1_merge)=
#### Merging Datasets
While the interfaces parquet has most of the data we need to train our model to predict poses for ligands binding to protein pockets, we need the information stored in the `is_polymer` and `num_resolved_residues` columns in the PN units dataset as well. We will use the information in `is_polymer` to ensure that the interfaces we are looking at are between a protein (polymer) and ligand (non-polymer). We will use the information in `num_resolved_residues` to make sure our dataset remains small enough to train our model on a single GPU. 

We will need to merge this information with the interfaces dataset twice, one for each PN unit involved in each interface. Keep in mind that you need the information in `pdb_id`, `assembly_id`, and `pn_unit_iid` to uniquely identify a structure!

````{dropdown} Click to see how to merge these datasets.
Create two copies of the relevant columns in the PN units dataset, one for each PN unit involved in the interface.  
```python
u1_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()
u2_cols = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "is_polymer", "num_resolved_residues"]].copy()
```
Rename the columns so that they match what is in the interfaces dataset and to ensure the new information is distinguishable between the two PN units.
```python
u1_cols = u1_cols.rename(columns={
    "pn_unit_iid": "pn_unit_1_iid",
    "is_polymer": "u1_is_polymer",
    "num_resolved_residues": "u1_num_resolved_residues"
})
u2_cols = u2_cols.rename(columns={
    "pn_unit_iid": "pn_unit_2_iid",
    "is_polymer": "u2_is_polymer",
    "num_resolved_residues": "u2_num_resolved_residues"
})
```
Actually merge the three datasets together: 
```python
df = interfaces.merge(u1_cols, on=["pdb_id", "assembly_id", "pn_unit_1_iid"], how="inner")
df = df.merge(u2_cols, on=["pdb_id", "assembly_id", "pn_unit_2_iid"], how="inner")
```
````

Check that the merge occurred correctly by printing out the columns of the new data frame, inspecting the first few rows of the new data frame, etc. 

(aw_build_model_p1_clean)=
#### Cleaning the Data
For the purposes of this tutorial, we want to remove rows where `involves_covalent_modification` is True and where the interface involves non-protein and non-ligand chains since we're looking for interfaces that are between a protein and a ligand.

This means we only want to keep rows where:
- `involves_loi` is True
- `is_inter_molecule` is True
- `involves_covalent_modification` is False
- We want one PN unit involved in the interface to be a polymer and the other to be non-polymer
- The total number of resolved residues for the pocket/ligand combination should be less than 200

We also want to make sure we remove any duplicates. The `u1` and `u2` labels are arbitrary, so it's possible for two rows to be identical except for which PN unit was labeled as 1 or 2. 

````{dropdown} Click to see the code
Keep only rows where at least one PN unit is a ligand of interest:
```python
df = df[df["involves_loi"]==True]
```
Keep only rows where the interfaces are intermolecular:
```python
df = df[df["is_inter_molecule"]==True]
```
Remove metal mediated-interfaces: 
```python
df = df[df["involves_metal"]!=True]
```
Remove covalently modified residues
```python
df = df[df["involves_covalent_modification"] != True]
```
Keep only interfaces where one PN unit is a polymer (protein) and one is not (ligand):
```python
df = df[df["u1_is_polymer"] != df["u2_is_polymer"]]
```
Keep only small examples:
df = df[(df["u1_num_resolved_residues"] + df["u2_num_resolved_residues"]) < 200 ]

Remove duplicates
df = df.drop_duplications(subset["pdb_id", "assembly_id", "pn_unit_1_iid", "pn_unit_2_iid"])
````

You can check to make sure these filters are actually being applied to your data frame by checking the `len` of the data frame before and after applying each filter.

To make sure the filter is doing what you expect, you can try running this procedure on a small subset of the data or locating specific rows in the larger dataset that should/should not be impacted by each filtering step. 

(aw_build_model_p1_new_cols)=
#### Adding New Columns
It will be useful later on if one row contains unique labels for each remaining interface. Right now information from four rows (`pdb_id`, `assembly_id`, `pn_unit_1_iid`, and `pn_unit_2_iid`) are required to uniquely identify an interface in our dataset. Let's add a new column to our dataframe and store our custom lable there. 

````{dropdown} Click to see the code.
```python
df["example_id"] = (df["pdb_id"] + "_" + 
    df["assembly_id"] + "_" + 
    df["pn_unit_1_iid"] + "_" + 
    df["pn_unit_2_iid"])
```
````

It will also be useful during training if our dataset already contains the path to the structure file in our PDB mirror. Add this information as a new column to your dataset. 
````{dropdown} Click to see the code. 
```python
PDB_MIRROR_PATH = os.environ.get("PDB_MIRROR_PATH", "/PATH/TO/pdb_mirror")
df["path"] = df["pdb_id"].str.lower().map(
    lambda x: f"{PDB_MIRROR_PATH}/{x[1:3]}/{x}.cif.gz"
)
```
Replace `"/PATH/TO/pdb_mirror"` if this environment variable is not already set. If it is set, you can leave this second argument blank.
````

To check that your code works correctly, make sure these new columns are present when you run `df.columns` and check the first few values in each. Are they what you expected? 

You can also add a test to ensure that all the values in `example_id` are unique:

````{dropdown} Click to see the code.
```python
assert df["example_id"].nunique() == len(df), "example_id is not unique!"
```
````

(aw_build_model_p1_split)=
#### Training, Testing, and Validation Sets
Now that we have the data, we need to split it up into three sets: `test`, `train`, and `val` (short for validation). There are many ways to do this and which is best will depend on your data and what you are trying to accomplish with your model. 

Here, we will use the `protein_cluster_30` column in the PN units data frame to split up our data. This column groups proteins by 30% sequence identity - it contains hash-based IDs that uniquely identify a cluster of proteins sharing more than 30% sequence identity. We will do an 80/10/10 split - 80% of the data will be in training, 10% in test, and 10% in validation. <!-- TODO: what algorithm was used to calculated these values -->

We will use this column to split the data by cluster, instead of individual rows. This will prevent the model from seeing near-identical protein pockets between the training and testing sets. 

Isolate this column and the data from the PN units data frame that uniquely identifies each row and merge it with the data frame. Remember that we will need to perform the merge twice, since there are two PN unites in each interface. 

````{dropdown} Click to see the code.
```python
protein_clusters = pn_units[["pdb_id", "assembly_id", "pn_unit_iid", "protein_cluster_30"]].copy()

df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_1_iid",
        "protein_cluster_30": "u1_cluster"
    }),
    on=["pdb_id", "assembly_id", "pn_unit_1_iid"],
    how="left"
)

df = df.merge(
    protein_clusters.rename(columns={
        "pn_unit_iid": "pn_unit_2_iid",
        "protein_cluster_30": "u2_cluster"
    }),
    on=["pdb_id", "assembly_id", "pn_unit_2_iid"],
    how="left"
)
```
````

Only the side of the interface that corresponds to the polymer will have a value for `protein_cluster_30`, the ligand will always have a `Null` value. Instead of having to check both `u2_cluster` and `u1_cluster` to determine which cluster our interface belongs to, let's put the information all in one column: 

````{dropdown} Click to see the code.
```python
df["protein_cluster"] = np.where(
    df["u1_is_polymer"],
    df["u1_cluster"],
    df["u2_cluster"]
)
```
````

Check to see if there are any cases where no cluster was assigned. 
````{dropdown} Click to see the code.
```python
# There are several ways to check, here we will just count the number of Null values:
len(df[df["protein_cluster"].isna()])
```
````

If any exist, we want to remove them from our dataset. They likely point to RNA/DNA chains, very short peptides, or low quality entries. 
````{dropdown} Click to see the code.
```python
df = df[df["protein_cluster"].notna()].reset_index(drop=True)
```
````

Before splitting the data up, let's shuffle the unique clusters. We use a see of 42 for reproducibility. Use this seed if you want to exactly replicate what was produced in this segment of the tutorial. 
````{dropdown} Click to see the code.
```python
unique_clusters = df["protein_cluster"].unique()
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_clusters)
```
````

Now we can finally spit the data into separate datasets and save them as parquet files for future use: 
````{dropdown} Click to see the code.
```python
n = len(unique_clusters)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
# test gets the remainder to avoid off-by-one gaps

train_clusters = set(unique_clusters[:n_train])
val_clusters   = set(unique_clusters[n_train : n_train + n_val])
test_clusters  = set(unique_clusters[n_train + n_val :])

def assign_split(cluster):
    if cluster in train_clusters: return "train"
    if cluster in val_clusters:   return "val"
    if cluster in test_clusters:  return "test"
    return "unassigned"  # rows where cluster was null

df["split"] = df["protein_cluster"].map(assign_split)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val   = df[df["split"] == "val"].reset_index(drop=True)
df_test  = df[df["split"] == "test"].reset_index(drop=True)

os.makedirs("splits", exist_ok=True)
df_train.to_parquet("splits/train.parquet", index=False)
df_val.to_parquet("splits/val.parquet",     index=False)
df_test.to_parquet("splits/test.parquet",   index=False)
```
````

You now have created the datasets you need to train, test, and validate the machine learning model you'll create as you continue to go through the **How to Build a Model Using AtomWorks** tutorial series. 

(aw_build_model_p1_next)=
### What Next? 
<!-- TODO  -->
<!-- You can find the next portion of the tutorial here. -->

(aw_build_model_p1_glossary)=
### Glossary

parquet
PN units - this is actually in the docs [glossary](https://rosettacommons.github.io/atomworks/latest/glossary.html#chains-pn-units-and-molecules)



