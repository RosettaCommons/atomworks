"""Predict ligand coordinates from a Part 4 checkpoint and test split."""

import argparse
from pathlib import Path

import torch
from model import PocketDockGNN
from train import CONFIG, build_dataloader, build_dataset


def main() -> None:
    """Load one usable example and predict its ligand coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", type=Path, default=Path("splits/test.parquet"))
    args = parser.parse_args()
    dataset = build_dataset(args.split, "docking_inference", CONFIG["pocket_radius"], CONFIG["max_test"])
    loader = build_dataloader(dataset, shuffle=False)
    batch = next((example for example in loader if example is not None), None)
    if batch is None:
        raise ValueError("No usable examples in inference split")

    model = PocketDockGNN.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    with torch.no_grad():
        predictions = model(
            batch["atomic_numbers"].squeeze(0),
            batch["input_coords"].squeeze(0),
            batch["edge_index"].squeeze(0),
            batch["is_ligand"].squeeze(0),
        )
    ligand_coords = predictions[batch["is_ligand"].squeeze(0)]
    assert ligand_coords.shape[1] == 3 and torch.isfinite(ligand_coords).all()
    print(f"Predicted coordinates for {len(ligand_coords)} ligand atoms:")
    print(ligand_coords)


if __name__ == "__main__":
    main()
