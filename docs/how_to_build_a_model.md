# How to Build a Model Using AtomWorks

## Table of Contents

(aw_build_model_intro)=
## Introduction
In this tutorial you will combine several of the functionalities of AtomWorks to build a fully functional machine learning model to predict how ligands will attach to protein pockets. <!-- TODO: Is this correct? What type of model are we having them build? -->

```{important}
This tutorial will walk you through creating a script using the AtomWorks API. 

For those that want to use the tutorial text as structure and hints to write your own script, we have hidden the code in collapsible cells. 

If you would like to see the full script, it is provided in the tutorial files. <!-- TODO: Create a tutorials file folder and upload once ready. -->
```

## Prerequisites
- The parquet files take up about 674 MB of space
- Pandas (though I think this comes with an AtomWorks installation)
    - link to a good tutorial: https://www.w3schools.com/python/pandas/default.asp
- Having the PDB mirror set up requires ~100GB of space

(aw_build_model_setup)=
## Setup 
Before we start building a model, we need to organize the data we want to train on. For this tutorial we will be using subsets of existing parquets – sets of parsed PDBs – to train our model. 

Download and decompress the parquet files via: 
```bash
wget https://files.ipd.uw.edu/pub/atomworks/dfs/pdb/2026_01_06.tar.gz
tar -xvf 2026_01_06.tar.gz
```
After decompressing the folder you should see three parquet files: 
- **assemblies.parquet:** There are multiple bio assemblies stored in a single PDB, each assembly will have multiple chains, interfaces, etc. 
- **interfaces.parquet:** Contains metadata for all binary interfaces in the PDB
- **pn_units.parquet:** Contains metadata for each PN unit <!-- TODO: define --> in the [PDB](https://www.rcsb.org/)

<!-- TODO: ask hope for the code that generated this -->

For more information about these parquet files, see the [Data Mirroring Documentation](mirrors.rst)

(aw_build_model_organize_data)=
## Organizing the Data <!-- TO DO: change this -->

```{note}
You will need to have a PDB mirror set up to follow this portion of the tutorial. 
If you do not have this you can: 
1. follow the instructions [here](mirrors.rst)
2. Use the pre-cleaned parquet file and skip to section <!-- TODO: add section information. -->
```

We will use the **interfaces** and **pn_units** parquet files for this tutorial. 

```{important}
```

### Loading the parquet file using Pandas
Let's first take a look at the information contained in the parquet file. Parquet files are not human parsable, but we can use [Pandas](https://pandas.pydata.org/) to inspect it. 

```{dropdown} Click to see the code
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

Let's take a closer look at a few of the columns in the interfaces parquet: <!-- TODO: check if these descriptions are true-->
- `pdb_id`: The identifier for the specific structure in the PDB
- `assembly_id`: <!-- TODO: ask Nate what type of information this stores, I know it's an integer, but I'm not sure what the number represents. Ranges from 1 to 44 -->
- `pn_unit_1_iid`: Label for the first PN unit in the interface
- `pn_unit_2_iid`: Label for the second PN unit in the interface
- `involves_loi`: Boolean for if the LOI (Ligand of Interest) is part of the interface
- `is_inter_molecule`: Boolean for if the interfaces is between two molecules (True) or within the same molecule (False)
- `involves_metal`: Boolean for if the interface involves a metal atom
- `involves_covalent_modification`: Boolean for if the interface involves a covalent modification (e.g. glycosylation)
- `num_contacts`: Number of contacts between the two PN units that create the interface
- `min_distance`: Minimum distance between the contacts that create the interface

And at a few of the columns in the PN units parquet file: 
- pn_unit_iid
- is_polymer
- num_resolved_residues
- pn_unit_type

For the purposes of this tutorial, we want to remove rows where `involves_covalent_modification` is True and where the interface involves non-protein or non-ligand chains <!-- TODO: why? and which rows store this data? Remove if involves_metal, involves_covalent_modification, ??-->

```{dropdown} Click to see the code
```python
interfaces_no_covalent = interfaces[interfaces["involves_covalent_modification"] != True]
```

<!-- TODO: does the PN units dataset also need to be cleaned??-->

## Glossary

parquet
PN units - this is actually in the docs [glossary](https://rosettacommons.github.io/atomworks/latest/glossary.html#chains-pn-units-and-molecules)



