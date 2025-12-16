from typing import Any, ClassVar

from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Transform


class BatchStructuresForDiffusionNoising(Transform):
    """
    Tiles the ground truth structures to match the diffusion batch size.

    In AF-3, we first batch input structures (broadcast the ground truth down the batch dimension),
    and then perform data augmentations such as differentially noising and rotating each structure.

    Precise behavior depends on whether the data dictionary already contains the key `coord_atom_lvl_to_be_noised`:
        - If the data dictionary already contains the key `coord_atom_lvl_to_be_noised`, we will batch the coordinates found in that key.
        - Otherwise, we will batch the coordinates found in `ground_truth.coord_atom_lvl`

    Performs the following transformation: (n_atoms, 3) -> (diffusion_batch_size, n_atoms, 3)

    Args:
        batch_size (int): The size of the diffusion batch.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "BatchStructuresForDiffusionNoising"
    ]  # Can only be applied once

    def __init__(self, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["ground_truth", "atom_array"])
        check_contains_keys(data["ground_truth"], ["coord_atom_lvl", "mask_atom_lvl"])

        if "coord_atom_lvl_to_be_noised" in data:
            assert len(data["coord_atom_lvl_to_be_noised"]) == len(
                data["atom_array"]
            ), "structure must not be batched yet"

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if "coord_atom_lvl_to_be_noised" in data:
            # Key already exists; we will batch the coordinates found in that key
            data["coord_atom_lvl_to_be_noised"] = data["coord_atom_lvl_to_be_noised"].repeat(self.batch_size, 1, 1)
        else:
            # Key does not exist; we will batch the coordinates found in `ground_truth.coord_atom_lvl`, and store the result in `coord_atom_lvl_to_be_noised`
            # (NOTE: `repeat` creates a new tensor; modifying `coord_atom_lvl_to_be_noised` will not affect `coord_atom_lvl`)
            data["coord_atom_lvl_to_be_noised"] = data["ground_truth"]["coord_atom_lvl"].repeat(self.batch_size, 1, 1)

        return data
