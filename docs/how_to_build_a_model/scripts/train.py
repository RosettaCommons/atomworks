import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset

from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import ConvertToTorch, Compose

from docs.how_to_build_a_model.scripts.transforms import CropToPocket, FeaturizeForDocking
from docs.how_to_build_a_model.scripts.model import PocketDockGNN

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)

CONFIG = {
    "hidden_dim": 128,
    "num_layers": 3,
    "learning_rate": 1e-3,
    "batch_size": 1,
    "max_epochs": 5,
    "pocket_radius": 10.0,
    "num_workers": 0,
    "max_train": 100,
    "max_val": 20,
    "max_test": 20,
}

class RobustDataset(Dataset):
    def __init__(self, dataset: PandasDataset):
        self.dataset = dataset
        self.failed = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception:
            self.failed.append(idx)
            return None
        

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        k: torch.stack([example[k] for example in batch])
        for k in TENSOR_KEYS
        if k in batch[0]
    }


def build_pipeline(radius: float) -> Compose:
    return Compose([
        RemoveHydrogens(),
        RemoveUnresolvedAtoms(),
        CropToPocket(radius=radius),
        FeaturizeForDocking(),
        ConvertToTorch(keys=TENSOR_KEYS),
    ])

def build_dataset(parquet_path: str, name: str, radius: float, max_examples: int = None):
    df = pd.read_parquet(parquet_path)
    if max_examples is not None:
        df = df.head(max_examples).reset_index(drop=True)

    loader = create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    )

    dataset = PandasDataset(
        data=df,
        name=name,
        id_column="example_id",
        loader=loader,
        transform=build_pipeline(radius),
        save_failed_examples_to_dir="failed_examples/",
    )
    return RobustDataset(dataset)

def build_dataloader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        persistent_workers=False,
    )

print("Building datasets...")
train_dataset = build_dataset(
    "splits/train.parquet", "docking_train", CONFIG["pocket_radius"], CONFIG["max_train"]
)
val_dataset = build_dataset(
    "splits/val.parquet", "docking_val", CONFIG["pocket_radius"], CONFIG["max_val"]
)
test_dataset = build_dataset(
    "splits/test.parquet", "docking_test", CONFIG["pocket_radius"], CONFIG["max_test"]
)

print(f"  Train: {len(train_dataset):,} examples")
print(f"  Val:   {len(val_dataset):,} examples")
print(f"  Test:  {len(test_dataset):,} examples")

train_loader = build_dataloader(train_dataset, shuffle=True)
val_loader = build_dataloader(val_dataset, shuffle=False)
test_loader = build_dataloader(test_dataset, shuffle=False)


model = PocketDockGNN(
    hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"],
    learning_rate=CONFIG["learning_rate"],
)

periodic_checkpoint = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="pocketdockgnn-{epoch:02d}-{step}",
    every_n_train_steps=50,
    save_last=True,
    verbose=True,
)

best_checkpoint = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="pocketdockgnn-best-{epoch:02d}-{val/loss:.4f}",
    monitor="val/loss",
    mode="min",
    save_top_k=3,
    verbose=True,
)

callbacks = [
    best_checkpoint,
    periodic_checkpoint,
    EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True,
    ),
]

trainer = pl.Trainer(
    max_epochs=CONFIG["max_epochs"],
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    log_every_n_steps=10,
    val_check_interval=50,
    enable_progress_bar=True,
)
print("\nStarting training...")
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

print(f"\nFailed examples during training: {len(train_dataset.failed)}")
print(f"Failed examples during val:      {len(val_dataset.failed)}")
print(f"Best checkpoint:                {best_checkpoint.best_model_path}")

print("\nEvaluating on test set...")
trainer.test(
    model,
    dataloaders=test_loader,
    ckpt_path=best_checkpoint.best_model_path,
)

print("\nDone.")


