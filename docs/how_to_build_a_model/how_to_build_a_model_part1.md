# Part 1: Prepare the Data

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Download current metadata and structures](#download-current-metadata-and-structures)
- [Inspect the metadata tables](#inspect-the-metadata-tables)
- [Merge the datasets](#merge-the-datasets)
- [Clean the data](#clean-the-data)
- [Resolve structure paths](#resolve-structure-paths)
- [Split by protein cluster](#split-by-protein-cluster)
- [Save the results](#save-the-results)
- [Check the outputs](#check-the-outputs)
- [What Next?](#what-next)
- [Glossary](#glossary)

## Introduction

This is the first in a series of tutorials that walks you through how to use AtomWorks to build a machine learning model for determining the pose of a ligand in a protein binding pocket, from start to finish.

**In this installment, you will learn how to use the [IO functionalities in AtomWorks](../io.rst) to prepare your data for use in a machine learning model.**

By the end of this first part of the **How to Build a Model with AtomWorks** tutorial series, you will have written a script that collects, cleans, and splits the data we need to train our model. You'll end up with four Parquet files:
- `all.parquet`: every selected example and its assigned split
- `train.parquet`: the examples used to train the model
- `validation.parquet`: the examples used to validate the model
- `test.parquet`: the examples used to test the model

```{important}
For those that want to use the tutorial text as structure and hints to write your own script, we have hidden the code in collapsible cells.

The full script, `data_cleaning_script.py`, is provided in the [tutorial files](./scripts/index.rst).
```

## Prerequisites

Before starting this tutorial it is assumed that you have:
- An intermediate knowledge of Python and the [Pandas](https://pandas.pydata.org/) library.
- A working installation of [AtomWorks](https://rosettacommons.github.io/atomworks/latest/). Only the `io` side will be used for this first part of the How to Build a Model tutorial series, however other parts will require the `ml` side.
- At least 1GB of space for storage of the parquet files.
- At least a portion of the PDB mirrored - the full mirror takes ~100GB of space.

```{note}
If you do not have over 100GB of space on your computing system, you can use a subset of the PDB instead of the full PDB mirror. See [Data Mirrors](../mirrors.rst) for how to only download specific PDB IDs. Make sure you have the `PDB_MIRROR_PATH` environment variable set to wherever you put it before moving on.
```
````

(aw_build_model_p1_setup)=
## Setup

AtomWorks provides a few Parquet files that contain metadata about the structures included in the [PDB](https://www.rcsb.org/). We will use these as our starting point.

AtomWorks provides a few parquet files that already contain metadata about the various structures included in the PDB. We will use these as our starting point.

Download the parquet files with the AtomWorks CLI:
```bash
atomworks setup metadata data/pdb_metadata
```
This creates the `data/pdb_metadata` directory (if it doesn't already exist) and populates it with several parquet files, including `shared/interfaces_df.parquet` and `shared/pn_units_df.parquet`. We will only use those two files in this tutorial.

## Inspect the metadata tables

Let's first take a look at the information contained in the parquet files. Parquet files are not human parsable, but we can use [Pandas](https://pandas.pydata.org/) to inspect them.

````{dropdown} Click to see the code
Load in the datasets:
```python
import pandas as pd

interfaces = pd.read_parquet("data/pdb_metadata/shared/interfaces_df.parquet")
pn_units = pd.read_parquet("data/pdb_metadata/shared/pn_units_df.parquet")
```
View the dataset columns:
```python
interfaces.columns
pn_units.columns
```
````

Let's take a closer look at a few of the columns in the interfaces parquet:
- `pdb_id`: The identifier for the specific structure in the PDB.
- `assembly_id`: Integer label for what assembly the given interface belongs to. There may be multiple assemblies in a single PDB structure.
- `altloc_seed`: Seed used to resolve which alternate locations were kept when the structure was processed.
- `pn_unit_1_iid` / `pn_unit_2_iid`: Labels for the two PN units in the interface.
- `pn_unit_1_type` / `pn_unit_2_type`: Chain type of each PN unit (see AtomWorks' `ChainType` enum) - for example polypeptide, DNA, RNA, or non-polymer.
- `pn_unit_1_is_polymer` / `pn_unit_2_is_polymer`: Boolean for whether each PN unit is a polymer.
- `involves_loi`: Boolean for if the LOI (Ligand of Interest) is part of the interface.
- `is_inter_molecule`: Boolean for if the interface is between two molecules (True) or within the same molecule (False).
- `involves_metal`: Boolean for if the interface involves a metal atom.
- `involves_covalent_modification`: Boolean for if the interface involves a covalent modification (e.g. glycosylation).
- `num_contacts`: Number of contacts between the two PN units that create the interface.
- `example_id`: A label that already uniquely identifies the interface.
- `path`: Path to the structure file AtomWorks used to produce this row (we'll replace this with a path into your own PDB mirror in [Resolve structure paths](#resolve-structure-paths)).

Let's also look at a few columns of interest from the PN units parquet file:
- `q_pn_unit_iid`: Label for the specific PN unit in the structure that the row corresponds to.
- `q_pn_unit_num_resolved_residues`: Number of resolved residues in the structure.
- `cluster`: Sequence-identity cluster the PN unit belongs to. Only polymers are clustered; non-polymers (ligands) have a `Null` value here.

```{note}
PN means "polymer or non-polymer." A PN unit is usually a polymer chain or a
small molecule, though covalently connected components may be grouped together.
```

We encourage you to take a closer look at these datasets on your own. For some suggestions on what to do, see the collapsible group below:

````{dropdown} Click to see the code
See the first 5 rows of specific columns in a dataset:
```python
interfaces[["pdb_id", "assembly_id", "pn_unit_1_iid", "pn_unit_2_iid"]].head()
```
See the unique values of a given column:
```python
interfaces["pn_unit_1_type"].unique()
```
Determine the datatype of the data stored in a particular column:
```python
interfaces[["is_inter_molecule"]].dtypes
```
````

Since `interfaces_df.parquet` and `pn_units_df.parquet` are generated by AtomWorks itself and can change between releases, it's good practice to double-check that the columns you're about to rely on are actually present before writing a long script around them:

````{dropdown} Click to see the code
```python
import pyarrow.parquet as pq

required_interface_columns = {
    "pdb_id", "assembly_id", "altloc_seed", "pn_unit_1_iid", "pn_unit_2_iid",
    "pn_unit_1_type", "pn_unit_2_type", "pn_unit_1_is_polymer", "pn_unit_2_is_polymer",
    "involves_loi", "is_inter_molecule", "involves_metal",
    "involves_covalent_modification", "example_id", "path",
}
missing = required_interface_columns.difference(pq.read_schema("data/pdb_metadata/shared/interfaces_df.parquet").names)
assert not missing, f"interfaces_df is missing columns: {missing}"
```
````

## Merge the datasets

The interfaces table already tells us whether each PN unit is a polymer and what type of chain it is, but it doesn't contain resolved-residue counts or cluster labels - those live in the PN units table. We'll need to merge that information in twice, once for each PN unit involved in an interface. Keep in mind that you need `pdb_id`, `assembly_id`, `altloc_seed`, *and* `pn_unit_iid` together to uniquely identify a PN unit, since a structure can have more than one alternate-location seed.

The code that follows in the rest of this tutorial can be found in `data_cleaning_script.py` in the [tutorial files](./scripts/index.rst).

````{dropdown} Click here to see how to merge these datasets.
Create two renamed copies of the relevant PN-unit columns, one for each side of the interface:
```python
pn_columns = pn_units[["pdb_id", "assembly_id", "altloc_seed", "q_pn_unit_iid", "q_pn_unit_num_resolved_residues", "cluster"]]

side_1 = pn_columns.rename(columns={
    "q_pn_unit_iid": "pn_unit_1_iid",
    "q_pn_unit_num_resolved_residues": "pn_unit_1_num_resolved_residues",
    "cluster": "pn_unit_1_cluster",
})
side_2 = pn_columns.rename(columns={
    "q_pn_unit_iid": "pn_unit_2_iid",
    "q_pn_unit_num_resolved_residues": "pn_unit_2_num_resolved_residues",
    "cluster": "pn_unit_2_cluster",
})
```
Merge both sides into the interfaces dataframe:
```python
df = interfaces.merge(side_1, on=["pdb_id", "assembly_id", "altloc_seed", "pn_unit_1_iid"], how="left")
df = df.merge(side_2, on=["pdb_id", "assembly_id", "altloc_seed", "pn_unit_2_iid"], how="left")
```
````

Check that the merge occurred correctly by printing out the columns of the new data frame, inspecting the first few rows, and checking `len(df)` before and after.

## Clean the data

For the purposes of this tutorial, we want to keep only interfaces that are between a protein and a ligand, aren't covalently modified, and are small enough to train our model on quickly. This means we only want to keep rows where:
- `involves_loi` is `True`.
- `is_inter_molecule` is `True`.
- `involves_metal` is `False`.
- `involves_covalent_modification` is `False`.
- Exactly one PN unit involved in the interface is a polymer, and that polymer is a protein (not DNA or RNA).
- The total number of resolved residues for the pocket/ligand combination is less than 200, to keep training time relatively short.

We also want to remove any rows whose residue counts didn't merge and any duplicate examples.

````{dropdown} Click to see the code
Keep only rows where the LOI is part of the interface:
```python
df = df[df["involves_loi"].eq(True)]
```
Keep only rows where the interfaces are intermolecular:
```python
df = df[df["is_inter_molecule"].eq(True)]
```
Remove metal-mediated interfaces:
```python
df = df[df["involves_metal"].ne(True)]
```
Remove covalently modified interfaces:
```python
df = df[df["involves_covalent_modification"].ne(True)]
```
Keep only interfaces where exactly one PN unit is a polymer:
```python
exactly_one_polymer = df["pn_unit_1_is_polymer"] ^ df["pn_unit_2_is_polymer"]
df = df[exactly_one_polymer]
```
`is_polymer` alone isn't quite enough here, though - DNA and RNA chains are polymers too, and we specifically want proteins. Use the `pn_unit_type` columns together with AtomWorks' `ChainType` enum to check that the polymer side is a D- or L-polypeptide:
```python
import numpy as np
from atomworks.enums import ChainType

protein_chain_types = {int(ChainType.POLYPEPTIDE_D), int(ChainType.POLYPEPTIDE_L)}
protein_side_type = np.where(df["pn_unit_1_is_polymer"], df["pn_unit_1_type"], df["pn_unit_2_type"])
df = df[pd.Series(protein_side_type, index=df.index).isin(protein_chain_types)]
```
Drop rows where the residue counts didn't merge, then keep only small examples:
```python
residue_columns = ["pn_unit_1_num_resolved_residues", "pn_unit_2_num_resolved_residues"]
df = df.dropna(subset=residue_columns)
df = df[df[residue_columns].sum(axis="columns") < 200]
```
Remove duplicates:
```python
df = df.drop_duplicates(subset=["example_id"])
```
````

You can check to make sure these filters are actually being applied to your data frame by checking the `len` of the data frame before and after applying each filter.

To make sure a filter is doing what you expect, you can try running it on a small subset of the data, or locate specific rows in the larger dataset that should/should not be impacted by that filtering step.

You can also add a test to make sure `example_id` is still unique after dropping duplicates:
````{dropdown} Click to see the code.
```python
assert df["example_id"].is_unique
```
````

(aw_build_model_p1_new_cols)=
### Adding New Columns

It will be useful during training for our dataset to contain the path to each structure file in your PDB mirror. The `path` column that came with `interfaces_df.parquet` may point at internally processed `.cif` files, so we need to overwrite it with a path into your own mirror instead. The CLI's PDB mirror stores structures divided into two-letter subdirectories as gzipped mmCIF files:

````{dropdown} Click to see the code.
```python
import os
from pathlib import Path

pdb_mirror = Path(os.environ["PDB_MIRROR_PATH"]).expanduser().resolve()

def resolve_path(pdb_id: str) -> str:
    pdb_id = pdb_id.lower()
    return str(pdb_mirror / pdb_id[1:3] / f"{pdb_id}.cif.gz")

df["path"] = df["pdb_id"].map(resolve_path)
```
````

To check that this worked, look at a few values in `df["path"]` and confirm the files actually exist on disk, e.g. with `Path(df["path"].iloc[0]).exists()`.

## Split by protein cluster

Now that we have the data, we need to split it up into three sets: `train`, `validation`, and `test`. There are many ways to do this and which is best will depend on your data and what you are trying to accomplish with your model.

Here, we will use the `cluster` column we merged in from the PN units data frame to split up our data. This column groups proteins by sequence identity, meaning it contains hash-based IDs that uniquely identify a cluster of related proteins. We will do an 80/10/10 split: 80% of the data will be in training, 10% in validation, and 10% in test.

We use this column to split the data by cluster, instead of by individual row. This prevents the model from seeing near-identical protein pockets in both the training and evaluation sets.

Only the side of the interface that corresponds to the polymer will have a value for `cluster` - the ligand side will always be `Null`. Instead of having to check both `pn_unit_1_cluster` and `pn_unit_2_cluster` to determine which cluster an interface belongs to, let's put that information in one column:

````{dropdown} Click to see the code.
```python
df["protein_cluster"] = np.where(
    df["pn_unit_1_is_polymer"],
    df["pn_unit_1_cluster"],
    df["pn_unit_2_cluster"],
)
```
````

Check to see if there are any cases where no cluster was assigned:
````{dropdown} Click to see the code.
```python
# There are several ways to check, here we will just count the number of Null values:
len(df[df["protein_cluster"].isna()])
```
````

If any exist, we want to remove them from our dataset. They likely point to very short peptides or low-quality entries that didn't get clustered.
````{dropdown} Click to see the code.
```python
df = df[df["protein_cluster"].notna()].reset_index(drop=True)
```
````

Before splitting the data up, let's shuffle the unique clusters. We use a seed of 42 for reproducibility - use this seed if you want to exactly replicate what was produced in this segment of the tutorial.
````{dropdown} Click to see the code.
```python
unique_clusters = df["protein_cluster"].drop_duplicates().to_numpy(copy=True)
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_clusters)
```
````

Now we can finally split the data into separate datasets and save them as parquet files for future use:
````{dropdown} Click to see the code.
```python
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
```
````

As a final sanity check, make sure that one protein cluster never shows up in more than one split - this is the key leakage check:
````{dropdown} Click to see the code.
```python
assert df.groupby("protein_cluster")["split"].nunique().eq(1).all()
```
````

#### Save the results

Now we can save each split as its own parquet file for future use, alongside one file containing everything:
````{dropdown} Click to see the code.
```python
import os

os.makedirs("splits", exist_ok=True)
df.to_parquet("splits/all.parquet", index=False)
for split in ("train", "validation", "test"):
    df[df["split"] == split].reset_index(drop=True).to_parquet(f"splits/{split}.parquet", index=False)
```
````
````

Run the complete script with the path to the interface metadata you downloaded earlier:
```bash
python docs/how_to_build_a_model/scripts/data_cleaning_script.py \
  data/pdb_metadata/shared/interfaces_df.parquet \
  --pdb-mirror "$PDB_MIRROR_PATH" \
  --output-dir splits
```
The script prints the number of examples assigned to each split.

#### Check the outputs

````{dropdown} Click to see the code.
```python
train = pd.read_parquet("splits/train.parquet")
validation = pd.read_parquet("splits/validation.parquet")
test = pd.read_parquet("splits/test.parquet")

assert set(train["protein_cluster"]).isdisjoint(validation["protein_cluster"])
assert set(train["protein_cluster"]).isdisjoint(test["protein_cluster"])
assert set(validation["protein_cluster"]).isdisjoint(test["protein_cluster"])
```
````

You now have created the datasets you need to train, test, and validate the machine learning model you'll build as you continue through the **How to Build a Model with AtomWorks** tutorial series.

## What Next?
[How to Build a Model with AtomWorks Part 2](how_to_build_a_model_part2.md)

## Glossary

**Parquet:** a columnar binary file format for storing large tabular datasets efficiently.

**PN unit:** AtomWorks' unit of "Polymer or Non-polymer"; a single chain, ligand, or other discrete molecular entity within a structure.

**Interface:** a pair of PN units in contact within a bio assembly, characterized by columns like `num_contacts`.

**LOI (Ligand of Interest):** the ligand being tracked for a given interface; used to identify protein-ligand interfaces in the dataset.

**Sequence identity cluster:** a group of protein chains that share a threshold percentage of sequence identity; used to split data so that similar proteins don't leak across train/validation/test sets.
