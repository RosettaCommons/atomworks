# Part 4: Train a Model

## Table of Contents

(aw_build_model_p4_intro)=
## Introduction
This is the fourth and final tutorial in the **How to Build a Model Using AtomWorks** series. By now you have:

- Parquet files for your `train`/`val`/`test` splits ([Part 1](how_to_build_a_model.md)).
- A loader that turns each parquet row into a protein-ligand structure ([Part 2](how_to_build_a_model_part2.md)).
- A transform pipeline that crops the pocket and converts it into tensors ([Part 2](how_to_build_a_model_part2.md)).
- A `PocketDockGNN` model in `model.py` ([Part 3](how_to_build_a_model_part3.md)).

**In this installment, you will write `train.py`, which wires all of these pieces together to train, validate, checkpoint, and test the model.**

```{important}
This tutorial completes the training script using the AtomWorks API, PyTorch, and PyTorch Lightning. The full solution is available in the tutorial files, and the code below is hidden in collapsible cells so you can attempt each step yourself first.
```

(aw_build_model_p4_prereq)=
## Prerequisites
Before starting this part it is assumed that you have:
- Completed [Parts 1–3](how_to_build_a_model.md), with `transforms.py` and `model.py` in place.
- A working installation of PyTorch and PyTorch Lightning.
- Access to a GPU (the example trainer is configured for a single GPU, but you can change the accelerator).

(aw_build_model_p4_overview)=
## What the Training Script Does
`train.py` wires together four pieces:

1. Read the split parquets.
2. Load and transform each example into model inputs.
3. Instantiate your model (`PocketDockGNN`).
4. Train, validate, checkpoint, and test.

The model sees a pocket-centered graph with five tensors:

- **`atomic_numbers`** — atomic identity for each atom.
- **`input_coords`** — protein pocket coordinates are kept, ligand coordinates are zeroed out.
- **`target_coords`** — the true coordinates the model should predict.
- **`edge_index`** — bond graph connectivity.
- **`is_ligand`** — a boolean mask for which atoms belong to the ligand.

In other words, the task is: *given the protein pocket context and the ligand atoms, predict the ligand's 3D placement.*

(aw_build_model_p4_imports)=
## Imports and Global Settings
Import PyTorch, PyTorch Lightning, the AtomWorks dataset utilities, the transforms you wrote in Part 2, and the model class from Part 3.

````{dropdown} Click to see the code
```python
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset

from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnresolvedAtoms
from atomworks.ml.transforms.base import ConvertToTorch, Compose

from transforms import CropToPocket, FeaturizeForDocking
from model import PocketDockGNN
```
Note that we now also import `ConvertToTorch`, which turns the NumPy features from `FeaturizeForDocking` into `torch` tensors.
````

Add two small global settings.

````{dropdown} Click to see the code
```python
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)
```
`torch.set_float32_matmul_precision("medium")` is a practical speed/precision trade-off for training. `pl.seed_everything(42)` makes runs more reproducible.
````

(aw_build_model_p4_config)=
## Configuration
Put the tunable settings in one place near the top of the file.

````{dropdown} Click to see the code
```python
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
```
````

```{note}
The `max_*` values cap how many examples we use so the run stays small and fast to troubleshoot. For actual training, remove these caps (set them to `None`) and train on the full splits.
```

(aw_build_model_p4_keys)=
## List the Tensor Keys
Your featurization step produces five tensors. Write those keys down once so the rest of the file can reuse the same list. This list is used both to tell `ConvertToTorch` what to convert and to tell the collate function what to stack.

````{dropdown} Click to see the code
```python
TENSOR_KEYS = [
    "atomic_numbers",
    "input_coords",
    "target_coords",
    "edge_index",
    "is_ligand",
]
```
````

```{important}
`TENSOR_KEYS` must be defined before the functions that reference it (`collate_fn` and `build_pipeline`). Keep it near the top of the file with `CONFIG`.
```

(aw_build_model_p4_robust)=
## Make the Dataset Robust to Bad Examples
If you train on real structures — which we are — some examples will fail. A ligand may have no resolved coordinates, or the pocket crop may remove everything useful. You do not want one bad structure to crash the entire run.

Wrap `PandasDataset` in a small `RobustDataset`. If a single example raises during loading or transform, it records the index and returns `None` instead of killing training.

````{dropdown} Click to see the code
```python
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
```
````

(aw_build_model_p4_collate)=
## Write a Custom `collate_fn`
The default PyTorch collator is not a good fit here: the examples are variable-size graphs, the dataset may return `None`, and we only want to batch the specific tensors the model needs (ignoring any extra fields in the example dictionary). The collate function should do two things: drop failed examples, and stack only the tensor keys the model expects.

````{dropdown} Click to see the code
```python
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        k: torch.stack([example[k] for example in batch])
        for k in TENSOR_KEYS
        if k in batch[0]
    }
```
With `batch_size=1`, this stacks a single example and adds a leading dimension of size 1, which the model's `_shared_step` removes with `.squeeze(0)`.
````

(aw_build_model_p4_pipeline)=
## Rebuild the Transform Pipeline
Write a helper that reconstructs the same transform sequence you tested in Part 2, now with `ConvertToTorch` appended so the features come out as `torch` tensors.

````{dropdown} Click to see the code
```python
def build_pipeline(radius: float) -> Compose:
    return Compose([
        RemoveHydrogens(),
        RemoveUnresolvedAtoms(),
        CropToPocket(radius=radius),
        FeaturizeForDocking(),
        ConvertToTorch(keys=TENSOR_KEYS),
    ])
```
````

(aw_build_model_p4_factories)=
## Dataset and Dataloader Factories
Write one function that takes a split parquet path and returns a ready-to-use dataset (this is essentially what you did in `smoke_test.py`, wrapped in `RobustDataset`).

````{dropdown} Click to see the code
```python
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
```
`save_failed_examples_to_dir` writes any example that fails inside the AtomWorks pipeline to disk so you can inspect it later.
````

Then write one small helper to build dataloaders consistently. This keeps the script tidy and reuses the same settings for train, validation, and test.

````{dropdown} Click to see the code
```python
def build_dataloader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        persistent_workers=False,
    )
```
````

(aw_build_model_p4_build)=
## Build the Datasets and Dataloaders
Now the main script body. Start by building the three datasets.

````{dropdown} Click to see the code
```python
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
```
````

Then build the dataloaders.

````{dropdown} Click to see the code
```python
train_loader = build_dataloader(train_dataset, shuffle=True)
val_loader = build_dataloader(val_dataset, shuffle=False)
test_loader = build_dataloader(test_dataset, shuffle=False)
```
````

(aw_build_model_p4_model)=
## Instantiate the Model
Instantiate the network with the hyperparameters from `CONFIG`.

````{dropdown} Click to see the code
```python
model = PocketDockGNN(
    hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"],
    learning_rate=CONFIG["learning_rate"],
)
```
Optionally, print the parameter count as a sanity check that you built the expected network:
```python
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
```
````

(aw_build_model_p4_callbacks)=
## Checkpointing and Early Stopping
Before creating the trainer, define the callbacks. Use `periodic_checkpoint` to give recovery points during training, and `best_checkpoint` to keep the best models by `val/loss`.

````{dropdown} Click to see the code
```python
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
```
The `EarlyStopping` callback stops training if validation loss stops improving for `patience` checks.
````

(aw_build_model_p4_trainer)=
## Create the Trainer
Now define the trainer itself with the PyTorch Lightning `Trainer`.

````{dropdown} Click to see the code
```python
trainer = pl.Trainer(
    max_epochs=CONFIG["max_epochs"],
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    log_every_n_steps=10,
    val_check_interval=50,
    enable_progress_bar=True,
)
```
````

```{tip}
If you do not have a GPU available, set `accelerator="cpu"` (and drop `devices=1`) to run the small capped example on CPU.
```

(aw_build_model_p4_fit)=
## Train and Evaluate
Call `fit()` to train.

````{dropdown} Click to see the code
```python
print("\nStarting training...")
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
```
````

After training, print some diagnostics: how many examples were skipped, and which checkpoint Lightning considered best.

````{dropdown} Click to see the code
```python
print(f"\nFailed examples during training: {len(train_dataset.failed)}")
print(f"Failed examples during val:      {len(val_dataset.failed)}")
print(f"Best checkpoint:                {best_checkpoint.best_model_path}")
```
````

Finally, evaluate on the held-out test split using the best checkpoint.

````{dropdown} Click to see the code
```python
print("\nEvaluating on test set...")
trainer.test(
    model,
    dataloaders=test_loader,
    ckpt_path=best_checkpoint.best_model_path,
)

print("\nDone.")
```
````

(aw_build_model_p4_wrap)=
## Wrapping Up
You have now built a complete, if simplified, machine-learning pipeline with AtomWorks: from raw PDB metadata to a trained pose-generation model. As next steps you might:

- Remove the `max_*` caps and train on the full splits.
- Add symmetric edges or an equivariant architecture to remove the model's dependence on absolute coordinates (see the warning in [Part 3](how_to_build_a_model_part3.md)).
- Track additional metrics, such as per-example ligand RMSD distributions, or visualize predicted poses.

(aw_build_model_p4_glossary)=
## Glossary

collate function — a function that assembles a list of examples into a single batch; here it also drops failed (`None`) examples.

checkpoint — a saved snapshot of model weights (and optimizer state) that can be reloaded to resume training or run evaluation.

early stopping — halting training once a monitored metric (here `val/loss`) stops improving, to save compute and reduce overfitting.
