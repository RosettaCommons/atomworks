"""Load one Part 1 example through the Part 2 transform pipeline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transforms import CropToPocket, FeaturizeForDocking

from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_structure_loader
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms


def build_dataset(split_path: Path) -> PandasDataset:
    """Build the tutorial dataset from a Part 1 Parquet split."""
    pipeline = Compose(
        [
            RemoveHydrogens(),
            RemoveUnresolvedAtoms(),
            CropToPocket(radius=10.0),
            FeaturizeForDocking(),
        ]
    )
    return PandasDataset(
        data=pd.read_parquet(split_path),
        name="docking_train",
        id_column="example_id",
        loader=create_structure_loader(
            altloc_seed_colname="altloc_seed",
            column_mapping={
                "query_pn_unit_iids": ["pn_unit_1_iid", "pn_unit_2_iid"],
                "query_is_polymer": ["pn_unit_1_is_polymer", "pn_unit_2_is_polymer"],
            },
        ),
        transform=pipeline,
    )


def main() -> None:
    """Print shapes and check the basic feature contract for one example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, nargs="?", default=Path("splits/train.parquet"))
    parser.add_argument("--index", type=int, default=0, help="Row to load from the split.")
    args = parser.parse_args()

    dataset = build_dataset(args.split)
    example = dataset[args.index]
    atom_count = len(example["atomic_numbers"])
    assert example["target_coords"].shape == (atom_count, 3)
    assert example["input_coords"].shape == (atom_count, 3)
    assert example["is_ligand"].shape == (atom_count,)
    assert example["edge_index"].shape[0] == 2
    assert not example["edge_index"].size or example["edge_index"].max() < atom_count
    assert np.all(example["input_coords"][example["is_ligand"]] == 0)
    assert np.array_equal(
        example["input_coords"][~example["is_ligand"]],
        example["target_coords"][~example["is_ligand"]],
    )
    assert example["is_ligand"].any() and (~example["is_ligand"]).any()

    print(f"Loaded {example['example_id']} from {len(dataset)} examples")
    for name in ("atomic_numbers", "input_coords", "target_coords", "edge_index", "is_ligand"):
        value = example[name]
        print(f"{name}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
