from typing import ClassVar

from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation


class CenterRandomAugmentation(Transform):
    """Centers coordinates and then randomly rotates and translates the input coordinates.

    Args:
        batch_size (int): Number of samples in the batch.
        scale (int): Scaling factor for the random rotation and translation. Default is 1.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [BatchStructuresForDiffusionNoising]

    def __init__(self, batch_size: int, scale: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.scale = scale

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        check_contains_keys(data["ground_truth"], ["coord_atom_lvl", "mask_atom_lvl"])

        assert (
            data["coord_atom_lvl_to_be_noised"].shape[0] == self.batch_size
        ), "Must batch coordinates to be noised before applying this transform"

    def forward(self, data: dict) -> dict:
        centered_coord_atom_lvl_to_be_noised = data["coord_atom_lvl_to_be_noised"]  # (batch_size, n_atoms, 3)
        mask_atom_lvl_expanded = data["ground_truth"]["mask_atom_lvl"].expand(
            centered_coord_atom_lvl_to_be_noised.shape[0], -1
        )
        centered_coord_atom_lvl_to_be_noised = masked_center(
            centered_coord_atom_lvl_to_be_noised, mask_atom_lvl_expanded
        )
        centered_coord_atom_lvl_to_be_noised = random_rigid_augmentation(
            centered_coord_atom_lvl_to_be_noised, batch_size=self.batch_size, s=self.scale
        )
        data["coord_atom_lvl_to_be_noised"] = centered_coord_atom_lvl_to_be_noised
        return data
