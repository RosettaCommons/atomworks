import sys

import torch

from docs.how_to_build_a_model.scripts.model import PocketDockGNN
from docs.how_to_build_a_model.scripts.train import CONFIG, build_dataloader, build_dataset


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py /path/to/checkpoint.ckpt")
        return

    ckpt_path = sys.argv[1]

    test_dataset = build_dataset(
        "splits/test.parquet", "docking_test", CONFIG["pocket_radius"], CONFIG["max_test"]
    )
    test_loader = build_dataloader(test_dataset, shuffle=False)

    model = PocketDockGNN.load_from_checkpoint(ckpt_path)
    model.eval()

    batch = next(iter(test_loader))
    atomic_numbers = batch["atomic_numbers"].squeeze(0)
    input_coords = batch["input_coords"].squeeze(0)
    edge_index = batch["edge_index"].squeeze(0)
    is_ligand = batch["is_ligand"].squeeze(0)

    with torch.no_grad():
        pred_coords = model(atomic_numbers, input_coords, edge_index)

    predicted_ligand_coords = pred_coords[is_ligand]
    print(f"Predicted coordinates for {predicted_ligand_coords.shape[0]} ligand atoms:")
    print(predicted_ligand_coords)


if __name__ == "__main__":
    main()
