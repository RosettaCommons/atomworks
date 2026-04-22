import torch
import torch.nn as nn
import pytorch_lightning as pl

class PocketDockGNN(pl.LightningModule):
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
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.loss_fn = nn.MSELoss()

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
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    
    




