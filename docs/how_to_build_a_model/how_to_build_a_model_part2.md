# Part 2: Creating a Transform Pipeline

## Table of Contents

- {ref}`aw_build_model_p2_intro`
- {ref}`aw_build_model_p2_prereq`
- {ref}`aw_build_model_p2_goal`
- {ref}`aw_build_model_p2_wire`
- {ref}`aw_build_model_p2_compose`
  - {ref}`aw_build_model_p2_crop`
  - {ref}`aw_build_model_p2_featurize`
- {ref}`aw_build_model_p2_smoke`
- {ref}`aw_build_model_p2_next`
- {ref}`aw_build_model_p2_glossary`

(aw_build_model_p2_intro)=
## Introduction
This is the second tutorial in the **How to Build a Model with AtomWorks** series. In [Part 1](how_to_build_a_model_part1.md) you cleaned the PDB metadata and saved `train`/`val`/`test` parquet splits.

**In this installment, you will learn how to transform each parquet file into tensors that can be used in a machine learning model.**

More specifically, you will learn how to: 
- create custom transforms
- create a transform pipeline
- utilize the `PandasDataset` class to go from parquets to trainable tensors

By the end of this part of the tutorial series you will have two python files: 
- `transforms.py`: Definition of custom tensors
- `smoke_test.py`: Testing the pipeline by loading a single example

```{important}
This tutorial will walk you through the creation of these two scripts. 

For those who want to use the tutorial text as structure and hints to write your own code, the solutions are hidden in collapsible cells. The new code in cells with repeated code is highlighted. 

If you would like to see the full scripts, they are provided in the tutorial files. <!-- TODO: link the tutorial files -->
```

(aw_build_model_p2_prereq)=
## Prerequisites
Before starting this part it is assumed that you have:
- Completed [Part 1](how_to_build_a_model_part1.md) and saved `splits/train.parquet`, `splits/val.parquet`, and `splits/test.parquet`.
- A working installation of AtomWorks, including the `ml` side.
- A (partial) PDB mirror set up so the loader can find the structure file for each row (see [Data Mirrors](../mirrors.rst)).
- Familiarity with `biotite.structure.AtomArray`, [NumPy](https://numpy.org/doc/stable/index.html), and [SciPy](https://scipy.org/).

```{note}
The transforms you write here operate on a `biotite` `AtomArray`. If you have not worked with `AtomArray` objects before, it helps to skim the [Biotite structure documentation](https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html) so the coordinate and annotation access patterns below feel familiar.
```

(aw_build_model_p2_goal)=
## The Goal of the Pipeline
Our task is pose generation: given a protein pocket and a small-molecule ligand (from the parquet files we generated in [Part 1](how_to_build_a_model_part1.md)), predict plausible bound cartesian coordinates for every ligand atom. To get there, each raw structure needs to be reduced to just the binding pocket and converted into tensors.

We will create a transform pipeline that can take protein/ligand pairs from the parquet files we created in [Part 1](how_to_build_a_model_part1.md), reduce the information to just the binding pocket, and convert them into tensors that can be used in the training of our machine learning model. 

We will apply four transforms in order, two that already exist in AtomWorks and two that we will write ourselves:

- **`RemoveHydrogens`** *(exists already)*: drops hydrogen atoms.
- **`RemoveUnresolvedAtoms`** *(exists already)*: drops atoms with no resolved coordinates.
- **`CropToPocket`** *(new)*: spatially crops the structure to the atoms within a radius of the ligand.
- **`FeaturizeForDocking`** *(new)*: converts the `AtomArray` into the tensors our model needs.

A fifth transform, `ConvertToTorch`, converts the NumPy features into `torch` tensors. We add it in [Part 4](how_to_build_a_model_part4.md) when we build the training loop; for now we work in NumPy so the outputs are easy to inspect.

(aw_build_model_p2_wire)=
## Wiring Up the Dataset and Loader
Before writing any transforms, let's confirm we can load a single example. AtomWorks provides two pieces we need:

- {py:class}`~atomworks.ml.datasets.PandasDataset` wraps a parquet/DataFrame and applies a loader and transform to each row.
- {py:func}`~atomworks.ml.datasets.loaders.create_loader_with_query_pn_units` returns a picklable loader that parses the CIF file for each row and attaches the *query* PN unit IIDs to the example, so we know which chain(s) the interface of interest involves.

It is worth reading the documentation for both (linked above) before using them. You can search the API docs, or use `help()` at a Python prompt. There is also more detail on `PandasDataset` in the {ref}`sphx_glr_auto_examples_dataset_exploration.py` example.

````{dropdown} Click to see how to inspect the documentation in a Python prompt
```python
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units

help(PandasDataset)
help(create_loader_with_query_pn_units)
```
````

Let's create our PandasDataset. We need to supply it with
- `data`: let's use train.parquet
- `name`: however you want to label this structure, in the tutorial we'll call it `docking_train`
- `id_column`: The column that has the unique identifier for each of piece of data, `example_id`
- `loader`: we will use `create_loader_with_query_pn_units` here

As a quick check that everything fits together, write a script (`smoke_test.py`) that reads `train.parquet`, builds a `PandasDataset` from it, and uses `create_loader_with_query_pn_units` as the loader.

````{dropdown} Click to see the code
```python
import pandas as pd
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units

df_train = pd.read_parquet("splits/train.parquet")

dataset = PandasDataset(
    data=df_train,
    name="docking_train",
    id_column="example_id",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    ),
)
```
Note that loader is told which columns hold the query PN unit IIDs.
````

Now load one example and inspect it. This tells us what keys <!-- TODO replace keys with something more descriptive --> the loader attaches and confirms the parquet, the loader, and your PDB mirror are all talking to each other.

````{dropdown} Click to see the code
```python
print(f"Dataset size: {len(dataset)}")

# Load a single example
example = dataset[0]
print("\nLoaded one sample successfully.")
print("Sample type:", type(example))

if hasattr(example, "keys"):
    print("Sample keys:", list(example.keys()))
    for k, v in example.items():
        print(f"{k}: {type(v)}")
else:
    print(example)
```
````

You should see that the example `type` is `<class 'atomworks.ml.transforms.base.TransformedDict'>`. 

Among the keys you should see `atom_array`, `query_pn_unit_iids`, and `chain_info`. These are the pieces of information that At
omWorks has pulled from the parquet files that we will use in our transforms pipeline:

- **`atom_array`** is the {py:class}`~biotite.structure.AtomArray` for this structure. Every atom carries a `pn_unit_iid` annotation (`atom_array.pn_unit_iid`), tagging it with the PN unit it belongs to, using the same string format as `query_pn_unit_iids` below.
- **`query_pn_unit_iids`** is a list of PN unit IID strings, one per column named in `pn_unit_iid_colnames`. Each IID has the form `"{chain_id}_{transformation_id}"` (e.g. `"A_1"`), or a comma-joined list of such tokens for a covalently-linked, multi-chain unit (e.g. `"G_1,R_1"`). Since we asked for two columns (`pn_unit_1_iid` and `pn_unit_2_iid`), this list has two entries: the two sides of the interface we are studying.
- **`chain_info`** is a dictionary keyed by `chain_id` (e.g. `"A"`, note this is just the chain id, not the full PN unit IID), where each value is a dict of per-chain metadata parsed from the CIF file, including `is_polymer` (bool), `chain_type`, and the chain's sequence. `CropToPocket` will use `is_polymer` to work out which side of the interface is the ligand.

```{tip}
The loader also attaches other keys to `example`, such as `example_id`, `metadata`, and `ligand_info`, however the three listed above are the ones this tutorial's transforms rely on.
```

(aw_build_model_p2_compose)=
## Building the Transform Pipeline
We can start creating our pipeline by adding to the code in our `smoke_test.py` file. We will use the {py:class}`~atomworks.ml.transforms.base.Compose` class to store the transforms, in an object we'll call `pipe`, in the order we want them to be applied and then add that object to our `dataset`.

Let's start with the two transforms that are native to AtomWorks, {py:class}`~atomworks.ml.transforms.filters.RemoveHydrogens` and {py:class}`~atomworks.ml.transforms.filters.RemoveUnresolvedAtoms`:


````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 4-5,9-12,21

import pandas as pd
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import Compose

df_train = pd.read_parquet("splits/train.parquet")

pipe = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
])

dataset = PandasDataset(
    data=df_train,
    name="docking_train",
    id_column="example_id",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    ),
    transform=pipe,
)
```
````

### Creating Custom Transforms

Now we need the two transforms that do not yet exist: `CropToPocket` and `FeaturizeForDocking`. The {ref}`sphx_glr_auto_examples_pocket_conditioning_transform.py` example is a great place to start.

```{tip}
Write your transforms in a separate `transforms.py` file. Each transform is a class whose `forward()` method calls a standalone function of the same (snake_case) name. Keeping the logic in a standalone function makes it easy to test and reuse outside the transform machinery.
```

(aw_build_model_p2_crop)=
#### `CropToPocket`
The `CropToPocket` transform will take the protein-lingand interfaces from our dataset and crop them to just the pocket. We will build `CropToPocket` up incrementally, confirming it plugs into `Compose` at each stage.

##### Start with a Shell
Begin with a class that has an `__init__(radius=10.0)` and a `forward()` that just returns the input `data` unchanged. Confirm it plugs into `Compose()`.

````{dropdown} Click to see the code
In `transforms.py`:
```python
from atomworks.ml.transforms.base import Transform

class CropToPocket(Transform):
    def __init__(self, radius: float = 10.0):
        super().__init__()
        self.radius = radius

    def forward(self, data: dict) -> dict:
        return data
```
Back in your `smoke_test.py` file, make sure to add this transform to your pipeline:
```{code-block} python
:emphasize-lines: 1,6

from transforms import CropToPocket
...
pipe = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
    CropToPocket(radius=10.0),
])
...
```
````
##### Input Validation
We need to add input validation to our class, add a `check_input()` method. In `check_input()`, assert that `data` contains the three keys the transform needs: `atom_array`, `query_pn_unit_iids`, and `chain_info`.

````{dropdown} Click to see the code
In the `CropToPocket` class:
```python
    def check_input(self, data: dict) -> None:
        assert "atom_array" in data, "Missing atom_array"
        assert "query_pn_unit_iids" in data, "Missing query_pn_unit_iids"
        assert "chain_info" in data, "Missing chain_info"
```
````

##### Write the `crop_to_pocket` Function
Add a standalone `crop_to_pocket()` function that `forward()` calls. It should take `atom_array`, `query_pn_unit_iids`, `chain_info`, and `radius` as inputs. For now, just copy the `AtomArray` and return it.

````{dropdown} Click to see the code
In `transforms.py`:
```{code-block} python
:emphasize-lines: 4-11,15-22

import numpy as np
from biotite.structure import AtomArray

def crop_to_pocket(
    atom_array: AtomArray,
    query_pn_unit_iids: list,
    chain_info: dict,
    radius: float = 10.0,
) -> AtomArray:
    atom_array = atom_array.copy()
    return atom_array

class CropToPocket(Transform):
    ...
    def forward(self, data: dict) -> dict:
        data["atom_array"] = crop_to_pocket(
            data["atom_array"],
            query_pn_unit_iids=data["query_pn_unit_iids"],
            chain_info=data["chain_info"],
            radius=self.radius,
        )
        return data
```
````

Each interface has two query PN units, a ligand and a protein (polymer). Add to the `crop_to_pocket` function to use `chain_info` to correctly assign the `iid`s to `ligand_iid` and `protien_iid`. If exactly one side is a polymer, the non-polymer side is the ligand. If the flags are ambiguous, fall back to a heuristic: the side with fewer atoms is treated as the ligand.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 9-31

def crop_to_pocket(
    atom_array: AtomArray,
    query_pn_unit_iids: list,
    chain_info: dict,
    radius: float = 10.0,
) -> AtomArray:
    atom_array = atom_array.copy()

    iid_a, iid_b = query_pn_unit_iids
    chain_a = iid_a.split("_")[0]
    chain_b = iid_b.split("_")[0]

    # chain_info is a dictionary of metadata about the chains
    a_is_polymer = chain_info.get(chain_a, {}).get("is_polymer", True)
    b_is_polymer = chain_info.get(chain_b, {}).get("is_polymer", True)

    if not a_is_polymer:
        ligand_iid, protein_iid = iid_a, iid_b
    elif not b_is_polymer:
        ligand_iid, protein_iid = iid_b, iid_a
    else:
        mask_a = atom_array.pn_unit_iid == iid_a
        mask_b = atom_array.pn_unit_iid == iid_b
        if mask_a.sum() <= mask_b.sum():
            ligand_iid, protein_iid = iid_a, iid_b
        else:
            ligand_iid, protein_iid = iid_b, iid_a

    print("ligand_iid:", ligand_iid)
    print("protein_iid:", protein_iid)
    return atom_array
```
````

Create boolean masks to make sure that the PN unit actually contains atoms whose identity match the `ligand_iid` or `protein_iid`. Raise a clear error if either side is empty. Failing early with a descriptive message makes debugging bad structures much easier later.

````{dropdown} Click to see the code
```python
    ligand_mask = atom_array.pn_unit_iid == ligand_iid
    protein_mask = atom_array.pn_unit_iid == protein_iid

    print("ligand atoms:", int(ligand_mask.sum()))
    print("protein atoms:", int(protein_mask.sum()))

    if ligand_mask.sum() == 0:
        raise ValueError(f"Ligand {ligand_iid} has no atoms")
    if protein_mask.sum() == 0:
        raise ValueError(f"Protein {protein_iid} has no atoms")
    return atom_array
```
````

##### Add the Spatial Crop with a KD-tree
Now for the actual crop. We keep every ligand atom plus every protein atom within a given `radius` (in angstroms) of any ligand atom.

```{note}
A **KD-tree** (k-dimensional tree) is a data structure that partitions points in space so you can answer "which points are within radius *r* of this query point?" efficiently, without comparing every pair of atoms. SciPy's [`cKDTree.query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.cKDTree.query_ball_point) returns, for each ligand atom, the indices of all protein atoms within `radius`.
```

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 1,30-55

from scipy.spatial import cKDTree

def crop_to_pocket(
    atom_array: AtomArray,
    query_pn_unit_iids: list,
    chain_info: dict,
    radius: float = 10.0,
) -> AtomArray:
    atom_array = atom_array.copy()

    iid_a, iid_b = query_pn_unit_iids
    chain_a = iid_a.split("_")[0]
    chain_b = iid_b.split("_")[0]

    a_is_polymer = chain_info.get(chain_a, {}).get("is_polymer", True)
    b_is_polymer = chain_info.get(chain_b, {}).get("is_polymer", True)

    if not a_is_polymer:
        ligand_iid, protein_iid = iid_a, iid_b
    elif not b_is_polymer:
        ligand_iid, protein_iid = iid_b, iid_a
    else:
        mask_a = atom_array.pn_unit_iid == iid_a
        mask_b = atom_array.pn_unit_iid == iid_b
        if mask_a.sum() <= mask_b.sum():
            ligand_iid, protein_iid = iid_a, iid_b
        else:
            ligand_iid, protein_iid = iid_b, iid_a

    ligand_mask = atom_array.pn_unit_iid == ligand_iid
    protein_mask = atom_array.pn_unit_iid == protein_iid

    ligand_coords = atom_array.coord[ligand_mask]
    protein_coords = atom_array.coord[protein_mask]

    if len(ligand_coords) == 0:
        raise ValueError(f"Ligand {ligand_iid} has no atoms")
    if len(protein_coords) == 0:
        raise ValueError(f"Protein {protein_iid} has no atoms")

    tree = cKDTree(protein_coords)
    neighbor_indices = tree.query_ball_point(ligand_coords, r=radius)

    total_neighbors = sum(len(n) for n in neighbor_indices)
    if total_neighbors == 0:
        raise ValueError(f"No protein atoms found within {radius}A of ligand {ligand_iid}")

    pocket_local_indices = np.unique(np.concatenate(neighbor_indices).astype(int))
    protein_global_indices = np.where(protein_mask)[0]
    pocket_global_indices = protein_global_indices[pocket_local_indices]
    ligand_global_indices = np.where(ligand_mask)[0]

    keep = np.sort(np.concatenate([pocket_global_indices, ligand_global_indices]))
    cropped = atom_array[keep]
    return cropped
```
The `cKDTree` is built from the protein coordinates; `query_ball_point` returns indices *local* to `protein_coords`, so we map them back to global indices via `protein_global_indices` before slicing the `AtomArray`.
````

Confirm the crop runs in your test:

````{dropdown} Click to see the code
```python
example = dataset[0]
cropped = example["atom_array"]
print("\nLoaded one sample successfully.")
print("Cropped atom count:", len(cropped))
```
````

##### Add the `is_ligand` annotation
The featurizer (and later analysis) needs to know which cropped atoms are ligand atoms. Annotate the cropped `AtomArray` with a boolean `is_ligand` array.

```{important}
This annotation is easy to forget: it is computed from `keep` and `ligand_global_indices`, but you must actually attach it to the array with `set_annotation`. If you skip this line, `FeaturizeForDocking` will fail because `atom_array.is_ligand` will not exist.
```

````{dropdown} Click to see the code
Add these two lines just before returning, so the end of `crop_to_pocket` reads:
```{code-block} python
:emphasize-lines: 4-5

    keep = np.sort(np.concatenate([pocket_global_indices, ligand_global_indices]))
    cropped = atom_array[keep]

    is_ligand = np.isin(keep, ligand_global_indices)
    cropped.set_annotation("is_ligand", is_ligand)
    return cropped
```
````

Quick check in your test:

````{dropdown} Click to see the code
```python
example = dataset[0]
cropped = example["atom_array"]
print("\nLoaded one sample successfully.")
print("Cropped atom count:", len(cropped))
print("Has is_ligand annotation:", hasattr(cropped, "is_ligand"))
print("Ligand atoms kept:", int(cropped.is_ligand.sum()))
print("Pocket atoms kept:", int((~cropped.is_ligand).sum()))

assert hasattr(cropped, "is_ligand")
assert cropped.is_ligand.any(), "No ligand atoms were kept"
assert (~cropped.is_ligand).any(), "No pocket atoms were kept"
print("CropToPocket sanity checks passed.")
```
````

##### Declare which transforms must run first
`CropToPocket` assumes hydrogens and unresolved atoms are already gone. Declare this ordering requirement so AtomWorks can enforce it.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 2

class CropToPocket(Transform):
    requires_previous_transforms = ["RemoveHydrogens", "RemoveUnresolvedAtoms"]
    ...
```
````

At this point you can assemble a full `smoke_test.py`:

````{dropdown} Click to see the full Pyhton file:
```python
import pandas as pd
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import Compose
from transforms import CropToPocket

df_train = pd.read_parquet("splits/train.parquet")

pipe = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
    CropToPocket(radius=10.0),
])

dataset = PandasDataset(
    data=df_train,
    name="docking_train",
    id_column="example_id",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    ),
    transform=pipe,
)

print(f"Dataset size: {len(dataset)}")
example = dataset[0]
cropped = example["atom_array"]
print("\nLoaded one sample successfully.")
print("Sample keys:", list(example.keys()))
print("Cropped atom count:", len(cropped))
print("Has is_ligand annotation:", hasattr(cropped, "is_ligand"))
print("Ligand atoms kept:", int(cropped.is_ligand.sum()))
print("Pocket atoms kept:", int((~cropped.is_ligand).sum()))

assert hasattr(cropped, "is_ligand")
assert cropped.is_ligand.any(), "No ligand atoms were kept"
assert (~cropped.is_ligand).any(), "No pocket atoms were kept"
print("\nCropToPocket smoke test passed.")
```
````

(aw_build_model_p2_featurize)=
#### `FeaturizeForDocking`
We follow the same incremental procedure. `FeaturizeForDocking` reads the cropped `AtomArray` and returns a dictionary of tensors that we merge into the example.

##### Start with a Shell
````{dropdown} Click to see the code
In `transforms.py`:
```python
class FeaturizeForDocking(Transform):
    def forward(self, data: dict) -> dict:
        return data
```
Add it to `Compose()` in `smoke_test.py`:
```python
from transforms import CropToPocket, FeaturizeForDocking

pipe = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
    CropToPocket(radius=10.0),
    FeaturizeForDocking(),
])
```
````

##### Declare the dependency on `CropToPocket`
`FeaturizeForDocking` depends on the `is_ligand` annotation created by `CropToPocket`, so declare it as a required previous transform and validate the annotation in `check_input()`.

````{dropdown} Click to see the code
```python
from atomworks.ml.transforms._checks import check_atom_array_annotation

class FeaturizeForDocking(Transform):
    requires_previous_transforms = ["CropToPocket"]

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["is_ligand"])
```
````

##### Write the `featurize_for_docking` Function
Instead of starting with a copy of the `AtomArray`, start with just returning an empty dictionary. 

Also update the `forward()` function. Have it update the `data` dictionary with the dictionary features the `featureize_for_docking` function will eventually create. 

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 1-2,7-8

def featurize_for_docking(atom_array: AtomArray) -> dict:
    return {}

class FeaturizeForDocking(Transform):
    ...
    def forward(self, data: dict) -> dict:
        features = featurize_for_docking(data["atom_array"])
        data.update(features)
        return data
```
````

##### Add the `is_ligand` Mask and Target Coordinates
`target_coords` are the ground-truth coordinates the model must reproduce.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 2-6

def featurize_for_docking(atom_array: AtomArray) -> dict:
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)
    return {
        "is_ligand": is_ligand,
        "target_coords": target_coords,
    }
```
Check the new features in your smoke test:
```python
example = dataset[0]
print("Sample keys:", list(example.keys()))
print("is_ligand shape:", example["is_ligand"].shape)
print("ligand atoms:", int(example["is_ligand"].sum()))
assert "is_ligand" in example
assert example["is_ligand"].dtype == bool
print("target_coords shape:", example["target_coords"].shape)
print("target_coords dtype:", example["target_coords"].dtype)
assert example["target_coords"].ndim == 2
assert example["target_coords"].shape[1] == 3
```
````

##### Add `input_coords` with the ligand zeroed out
This is the heart of the task. We keep the real pocket coordinates but zero out the ligand coordinates: the model must learn to place the ligand.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 5-6,11

def featurize_for_docking(atom_array: AtomArray) -> dict:
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)

    input_coords = target_coords.copy()
    input_coords[is_ligand] = 0.0

    return {
        "is_ligand": is_ligand,
        "target_coords": target_coords,
        "input_coords": input_coords,
    }
```
````

##### Add `atomic_numbers`
Encode each atom's element as an integer atomic number using the AtomWorks lookup table. These integers will feed an embedding layer in the model.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 1,10-13,16

from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
...
def featurize_for_docking(atom_array: AtomArray) -> dict:
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)

    input_coords = target_coords.copy()
    input_coords[is_ligand] = 0.0

    atomic_numbers = np.array(
        [ELEMENT_NAME_TO_ATOMIC_NUMBER.get(e.upper(), 0) for e in atom_array.element],
        dtype=np.int64,
    )

    return {
        "atomic_numbers": atomic_numbers,
        "is_ligand": is_ligand,
        "target_coords": target_coords,
        "input_coords": input_coords,
    }
```
Elements not found in the lookup table map to `0`, which acts as an "unknown atom" index in the embedding table.
````

##### Add Bonds Via an `edge_index` 
`edge_index` is the bond graph in COO (coordinate) format: a `[2, <number of bonds>]` integer array where each column is a bonded pair of atoms.

````{dropdown} Click to see the code
```{code-block} python
:emphasize-lines: 13-14,20

def featurize_for_docking(atom_array: AtomArray) -> dict:
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)

    input_coords = target_coords.copy()
    input_coords[is_ligand] = 0.0

    atomic_numbers = np.array(
        [ELEMENT_NAME_TO_ATOMIC_NUMBER.get(e.upper(), 0) for e in atom_array.element],
        dtype=np.int64,
    )

    bonds = atom_array.bonds.as_array()
    # we only need the first two columns, the last one is the number of bonds
    edge_index = bonds[:, :2].T.astype(np.int64)

    return {
        "atomic_numbers": atomic_numbers,
        "input_coords": input_coords,
        "target_coords": target_coords,
        "edge_index": edge_index,
        "is_ligand": is_ligand,
    }
```
`atom_array.bonds.as_array()` returns an array whose first two columns are the indices of the bonded atoms; we take those two columns and transpose to get the `[2, E]` shape.
````

Check the edge graph in your smoke test:

````{dropdown} Click to see the code
```python
example = dataset[0]
print("edge_index shape:", example["edge_index"].shape)
print("edge_index dtype:", example["edge_index"].dtype)
assert example["edge_index"].ndim == 2
assert example["edge_index"].shape[0] == 2
assert example["edge_index"].max() < example["atomic_numbers"].shape[0]
```
````

### The final `FeaturizeForDocking` class
````{dropdown} Click to see the code
```python
class FeaturizeForDocking(Transform):
    requires_previous_transforms = ["CropToPocket"]

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["is_ligand"])

    def forward(self, data: dict) -> dict:
        features = featurize_for_docking(data["atom_array"])
        data.update(features)
        return data
```
````

(aw_build_model_p2_smoke)=
## The Complete Smoke Test
Your final `smoke_test.py` should build the full pipeline and validate every tensor:

````{dropdown} Click to see the complete file
```python
import pandas as pd
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import Compose
from transforms import CropToPocket, FeaturizeForDocking

df_train = pd.read_parquet("splits/train.parquet")

pipe = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
    CropToPocket(radius=10.0),
    FeaturizeForDocking(),
])

dataset = PandasDataset(
    data=df_train,
    name="docking_train",
    id_column="example_id",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    ),
    transform=pipe,
)

print(f"Dataset size: {len(dataset)}")
example = dataset[0]
print("\nLoaded one sample successfully.")
print("Sample keys:", list(example.keys()))
print("atomic_numbers:", example["atomic_numbers"].shape, example["atomic_numbers"].dtype)
print("input_coords:", example["input_coords"].shape, example["input_coords"].dtype)
print("target_coords:", example["target_coords"].shape, example["target_coords"].dtype)
print("edge_index:", example["edge_index"].shape, example["edge_index"].dtype)
print("is_ligand:", example["is_ligand"].shape, example["is_ligand"].dtype)

assert example["atomic_numbers"].ndim == 1
assert example["target_coords"].ndim == 2
assert example["target_coords"].shape[1] == 3
assert example["input_coords"].shape == example["target_coords"].shape
assert example["edge_index"].ndim == 2
assert example["edge_index"].shape[0] == 2
assert example["edge_index"].max() < example["atomic_numbers"].shape[0]
assert (example["input_coords"][example["is_ligand"]] == 0).all(), \
    "Ligand coordinates should be zeroed"
assert (example["input_coords"][~example["is_ligand"]] != 0).any(), \
    "Pocket coordinates should not all be zero"
print("\nFeaturizeForDocking smoke test passed.")
```
````

When this passes, every example your dataset yields carries five arrays: `atomic_numbers`, `input_coords`, `target_coords`, `edge_index`, and `is_ligand`.

(aw_build_model_p2_next)=
## What Next?
With a working transform pipeline, you are ready to write the model that consumes these tensors. Continue to [Part 3: Write a Model](how_to_build_a_model_part3.md).

(aw_build_model_p2_glossary)=
## Glossary

**KD-tree:** a space-partitioning data structure for fast nearest-neighbor and radius queries.

**COO format:** "coordinate" sparse format; here, a `[2, E]` array listing the endpoints of each bond.

**Transform:** a reusable step that reads and rewrites the example dictionary; chained together with `Compose`.
