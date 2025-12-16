from typing import Any, ClassVar

import torch

from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising


def sample_t_edm(sigma_data: float, diffusion_batch_size: int) -> torch.Tensor:
    """
    Sample timesteps following the EDM paper.

    Args:
        sigma_data (float): The sigma data parameter for scaling.
        diffusion_batch_size (int): The size of the batch for diffusion. We will sample this many timesteps.

    Returns:
        torch.Tensor: A tensor of shape (diffusion_batch_size,) containing sampled time values.
    """
    # Reference for h-params: NVIDIA EDM Paper (https://arxiv.org/pdf/2206.00364)
    t = sigma_data * torch.exp(-1.2 + 1.5 * torch.normal(mean=0, std=1, size=(diffusion_batch_size,)))
    return t


def sample_noise_edm(t: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """
    Based on the timestep t, sample noise for the diffusion process.

    Args:
        t (torch.Tensor): A tensor of shape (diffusion_batch_size,) containing time values.
        num_atoms (int): The number of atoms.

    Returns:
        torch.Tensor: A tensor of shape (diffusion_batch_size, num_atoms, 3) containing sampled noise.

    """
    t_tiled = t[:, None, None].tile(1, num_atoms, 3)
    return torch.normal(mean=0, std=1, size=t_tiled.shape) * t_tiled


class SampleEDMNoise(Transform):
    requires_previous_transforms: ClassVar[list[str | Transform]] = [BatchStructuresForDiffusionNoising]

    def __init__(self, sigma_data: float, diffusion_batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.sigma_data = sigma_data
        self.diffusion_batch_size = diffusion_batch_size

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        assert (
            data["coord_atom_lvl_to_be_noised"].shape[0] == self.diffusion_batch_size
        ), "Must batch coordinates to be noised before applying this transform"

    def forward(self, data: dict) -> dict:
        """
        Apply EDM noise sampling to the coordinates that are to be noised.

        Args:
            data (Dict[str, Any]): The input data dictionary containing the coordinates to be noised.

        Returns:
            Dict[str, Any]: The input data dictionary with the added keys "t" and "noise" containing the sampled timesteps and noise.
                - t (torch.Tensor): A tensor of shape (diffusion_batch_size,) containing sampled time values.
                - noise (torch.Tensor): A tensor of shape (diffusion_batch_size, num_atoms, 3) containing sampled noise for each atom.
        """
        t = sample_t_edm(self.sigma_data, self.diffusion_batch_size)
        noise = sample_noise_edm(t, data["coord_atom_lvl_to_be_noised"].shape[1])
        data["t"] = t
        data["noise"] = noise
        return data
