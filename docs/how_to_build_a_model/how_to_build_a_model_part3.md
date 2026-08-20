# Part 3: Write a Model

## Table of Contents

- {ref}`aw_build_model_p3_intro`
- {ref}`aw_build_model_p3_prereq`
- {ref}`aw_build_model_p3_choice`
- {ref}`aw_build_model_p3_goal`
- {ref}`aw_build_model_p3_imports`
- {ref}`aw_build_model_p3_init`
- {ref}`aw_build_model_p3_layers`
- {ref}`aw_build_model_p3_forward`
- {ref}`aw_build_model_p3_step`
- {ref}`aw_build_model_p3_hooks`
- {ref}`aw_build_model_p3_optim`
- {ref}`aw_build_model_p3_shape`
- {ref}`aw_build_model_p3_next`
- {ref}`aw_build_model_p3_glossary`

(aw_build_model_p3_intro)=
## Introduction
This is the third tutorial in the **How to Build a Model Using AtomWorks** series. So far you have:

- Parquet files for your `train`/`val`/`test` splits ([Part 1](how_to_build_a_model_part1.md)).
- A loader that turns each parquet row into a protein-ligand structure ([Part 2](how_to_build_a_model_part2.md#wiring-up-the-dataset-and-loader)).
- A transform pipeline that crops the pocket and converts it into tensors ([Part 2](how_to_build_a_model_part2.md#building-the-transform-pipeline)).

**In this installment, you will write the neural network that consumes those tensors and predicts 3D coordinates for every atom.**


By the end of this part you will have a `model.py` containing a trainable `PocketDockGNN` `LightningModule`.

```{important}
This tutorial continues to build a script using the AtomWorks API and PyTorch. The full solution is available in the [tutorial files](./scripts/index.rst); the code here is also hidden in collapsible cells so you can attempt each step yourself.
```

(aw_build_model_p3_prereq)=
## Prerequisites
Before starting this part of the tutorial series it is assumed that you have:
- Completed [Part 2](how_to_build_a_model_part2.md), with a working `transforms.py` and `smoke_test.py`.
- A working installation of [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index).
- Familiarity with basic neural-network building blocks (`Linear`, `Embedding`, `LayerNorm`) and the idea of message passing on a graph.

(aw_build_model_p3_choice)=
## Choosing an Architecture
Before writing any code, we have to decide what architecture to use. For this tutorial we use a simple message-passing **graph neural network (GNN)**. A GNN is a natural choice because our data is already a graph: the bond graph is stored in `edge_index`, and each atom is a node with an atomic-number feature.

```{note}
We use a GNN as an example model architecture. This tutorial focuses on using AtomWorks to build and train a model, rather than on the details of how GNNs work.
```

```{warning}
This model is **not** rotation/translation invariant. It sees raw XYZ coordinates, which means it can learn to "cheat" based on absolute position rather than on geometry. Equivariant architectures (for example, those built on relative displacements or SE(3)-equivariant layers) address this, but they add substantial complexity. We keep things simple here so the pipeline is easy to follow; treat the resulting model as a teaching example rather than a production docking model.
```

A few other decisions to make before writing:

- **How many GNN layers?** More layers propagate information further across the bond graph, at the cost of compute and the risk of over-smoothing.
- **What hidden dimension?** Wider layers can represent more, but use more memory.
- **What loss function?** This should reflect what "a good pose" means.

To keep the example small and trainable on a single GPU, we use:

- **3 GNN layers**: enough to propagate information a few hops through the bond graph.
- **128 hidden dimensions**: small enough to train on a single GPU.
- **Mean squared error (MSE)** between predicted and target coordinates.

(aw_build_model_p3_goal)=
## The Goal of `model.py`
We will write the model in `model.py`. It needs to:

- Take atom types, coordinates, and bond edges as input.
- Pass information along the atom graph.
- Predict 3D coordinates for each atom.
- Expose train/val/test steps through a `LightningModule`.

(aw_build_model_p3_imports)=
## Imports and Class Definition
Import PyTorch, Lightning, and the basic building blocks. [`torch`](https://pytorch.org/) provides tensors and tensor ops; [`torch.nn`](https://docs.pytorch.org/docs/2.13/nn.html) provides layers like `Linear`, `Embedding`, and `LayerNorm`; [`pytorch_lightning`](https://lightning.ai/docs/pytorch/stable/index) provides `LightningModule`, which packages the model, loss, logging, and optimizer setup into one class.

Lets start off `model.py` by importing these modules and begining our `PocketDockGNN` class that inherits from `Lightning Module`:
````{dropdown} Click to see the code
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class PocketDockGNN(pl.LightningModule):
    ...
```
````

(aw_build_model_p3_init)=
## `__init__` and Hyperparameters
Write `__init__` and save the hyperparameters so Lightning can restore them from a checkpoint.

````{dropdown} Click to see the code
```python
    def __init__(
        self,
        num_atom_types: int = 119,
        hidden_dim: int = 128,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
```
````

These arguments mean:

- **`num_atom_types=119`** - one embedding-table entry for each atomic number from 0 to 118. (We are using 0 to denote an unknown atom type.)
- **`hidden_dim`** - the width of the learned per-atom representation.
- **`num_layers`** - how many rounds of message passing to run.
- **`learning_rate`** - step size for the [Adam optimizer](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/).

(aw_build_model_p3_layers)=
## Building the Layers
We build the layers in the order the data flows through them.

### Embed Atom Identities
Each atom arrives as an integer atomic number. A neural network works better with learned vectors, so we add an embedding table. This lets the model learn different behavior for carbon, oxygen, nitrogen, and so on without us hand-coding any chemistry rules.

````{dropdown} Click to see the code
```python
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
```
````

### Combine atom type with input coordinates
For each atom, the model combines its learned atom-type embedding with its three input coordinates. We concatenate these features and project the resulting vector back to hidden_dim.

````{dropdown} Click to see the code
```python
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
```
````

### Build the message-passing blocks
The graph structure comes from `edge_index`, which says which atoms are bonded. To use that graph, add a stack of message and update blocks, plus layer norms.

````{dropdown} Click to see the code
```python
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
```
````

These three lists work together:

- **`conv_layers`**: transform each source atom into a message.
- **`update_layers`**: combine an atom's current state with the aggregated neighbor message.
- **`layer_norms`**: stabilize training after each update.

### Project back to 3D coordinates
After message passing, each atom has a learned hidden representation. The output head maps that vector to an (x, y, z) prediction.

````{dropdown} Click to see the code
```python
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
```
````

### Define the loss
For the tutorial we use mean squared error between predicted and target coordinates.

````{dropdown} Click to see the code
```python
        self.loss_fn = nn.MSELoss()
```
````

(aw_build_model_p3_forward)=
## The Forward Pass
Now write `forward()`. It does four things in order:

1. Embed the atomic numbers.
2. Concatenate those embeddings with the input coordinates and project.
3. Run message passing over the bond graph. For each layer, compute a message for every atom, route each source message to its destination atom using `edge_index`, sum the incoming messages per destination, and update each atom with a residual connection.
4. Project to coordinates.

````{dropdown} Click to see the code
```python
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        input_coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.atom_embedding(atomic_numbers)
        x = self.input_proj(torch.cat([x, input_coords], dim=-1))

        src, dst = edge_index[0], edge_index[1]

        for conv, update, norm in zip(self.conv_layers, self.update_layers, self.layer_norms):
            messages = conv(x)
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), messages[src])
            x = x + norm(update(torch.cat([x, agg], dim=-1)))

        pred_coords = self.output_proj(x)
        return pred_coords
```
`scatter_add_` sums each source atom's message into the row of `agg` belonging to its destination atom, which implements neighbor aggregation. The residual connection (`x = x + ...`) keeps gradients well-behaved across layers.
````

```{note}
`atom_array.bonds.as_array()` lists each bond once, so as written, messages flow from `src` to `dst` in a single direction per bond. If you want symmetric message passing (information flowing both ways along every bond), you can duplicate and flip the edges when you build `edge_index`, or concatenate `[src, dst]` with `[dst, src]`. This is optional for the tutorial but worth knowing.
```

(aw_build_model_p3_step)=
## One Shared Train/Val/Test Step
Lightning calls separate methods for training, validation, and test, but the logic is almost identical, so write it once in a helper. Skip `None` batches (these come from failed examples, which we handle in [Part 4](how_to_build_a_model_part4.md)), remove the leading batch dimension added by the `DataLoader` (we use `batch_size=1`), run the model, and compute the coordinate loss.

The full-coordinate MSE is fine for optimization, but docking quality is best judged on the ligand atoms alone, so we also log a ligand RMSD.

````{dropdown} Click to see the code
```python
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        if batch is None:
            return None
        # Remove the batch dimension added by DataLoader (batch_size=1)
        atomic_numbers = batch["atomic_numbers"].squeeze(0)   # (N,)
        input_coords   = batch["input_coords"].squeeze(0)     # (N, 3)
        target_coords  = batch["target_coords"].squeeze(0)    # (N, 3)
        edge_index     = batch["edge_index"].squeeze(0)       # (2, E)
        is_ligand      = batch["is_ligand"].squeeze(0)        # (N,)

        pred_coords = self(atomic_numbers, input_coords, edge_index)
        loss = self.loss_fn(pred_coords, target_coords)

        with torch.no_grad():
            ligand_rmsd = torch.sqrt(
                ((pred_coords[is_ligand] - target_coords[is_ligand]) ** 2)
                .sum(dim=-1).mean()
            )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/ligand_rmsd", ligand_rmsd, prog_bar=True)

        return loss
```
````

(aw_build_model_p3_hooks)=
## Connect Lightning's Step Methods
The stage-specific methods become tiny wrappers around the shared step.

````{dropdown} Click to see the code
```python
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "test")
```
````

(aw_build_model_p3_optim)=
## Configure the Optimizer
Tell Lightning which optimizer to use. Here we use [Adam](https://docs.pytorch.org/docs/2.13/generated/torch.optim.Adam.html).

````{dropdown} Click to see the code
```python
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```
````

(aw_build_model_p3_shape)=
## The Shape of `model.py`
Your final `model.py` should have this shape:

````{dropdown} Click to see the outline
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class PocketDockGNN(pl.LightningModule):
    def __init__(...):
        ...
    def forward(...):
        ...
    def _shared_step(...):
        ...
    def training_step(...):
        ...
    def validation_step(...):
        ...
    def test_step(...):
        ...
    def configure_optimizers(...):
        ...
```
````

At this point you should have a trainable model class.

(aw_build_model_p3_next)=
## What Next?
With `transforms.py` and `model.py` in place, you are ready to wire everything into a training loop. Continue to [Part 4: Train a Model](how_to_build_a_model_part4.md).

(aw_build_model_p3_glossary)=
## Glossary

**GNN (graph neural network):** a network that operates on graph-structured data by passing messages between connected nodes.

**Message passing:** the process of computing a message from each node, aggregating messages at each destination node, and updating node states.

**RMSD (root-mean-square deviation):**: the square root of the mean squared distance between predicted and true atom positions; a standard structural-accuracy metric.
