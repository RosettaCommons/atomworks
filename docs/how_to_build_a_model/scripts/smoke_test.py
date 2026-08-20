"""
Applies a loader and transform pipeline to each example and runs a smoke test.
Created in part 2 of the How to Build a Model with AtomWorks tutorial series.

Transforms applied:
- RemoveHydrogens
- RemoveUnresolvedAtoms
- CropToPocket
- FeaturizeForDocking
"""

import pandas as pd
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import Compose

# NOTE: import assumes you run this script from the directory containing transforms.py. 

from transforms import CropToPocket, FeaturizeForDocking

# Load in the training data as a pandas DataFrame
df_train = pd.read_parquet("splits/train.parquet")

# Define the transform pipeline
transforms_pipeline = Compose([
    RemoveHydrogens(),
    RemoveUnresolvedAtoms(),
    CropToPocket(radius=10.0),
    FeaturizeForDocking(),
])

# Build the AtomWorks PandasDataset
dataset = PandasDataset(
    data=df_train,
    name="docking_train",
    id_column="example_id",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]),
    transform=transforms_pipeline,
)

################################
# Print statements for testing #
################################
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
