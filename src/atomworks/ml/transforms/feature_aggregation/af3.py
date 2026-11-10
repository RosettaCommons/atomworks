from typing import Any, ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from biotite.structure import AtomArray
from einops import rearrange

from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_af3_token_representative_idxs, get_token_starts


class AggregateFeaturesLikeAF3(Transform):
    """
    Aggregates features into the correct places, and shapes with the names for AlphaFold 3.

    This transform combines various features from the input data into the format
    expected by the AlphaFold 3 model. It processes MSA features, ground truth
    structures, and other relevant data.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AtomizeByCCDName",
        "FeaturizeMSALikeAF3",
        "EncodeAF3TokenLevelFeatures",
    ]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AggregateFeaturesLikeAF3"]

    def check_input(self, data: dict[str, Any]) -> None:
        """
        Checks if the input data contains the required keys and types.

        Args:
            data (Dict[str, Any]): The input data dictionary.

        Raises:
            KeyError: If a required key is missing from the input data.
            TypeError: If a value in the input data is not of the expected type.
        """
        check_contains_keys(data, ["msa_features", "atom_array"])
        check_is_instance(data, "msa_features", dict)
        check_is_instance(data, "atom_array", AtomArray)

        # Check MSA features
        msa_features = data["msa_features"]
        check_contains_keys(msa_features, ["msa_features_per_recycle_dict", "msa_static_features_dict"])
        check_is_instance(msa_features, "msa_features_per_recycle_dict", dict)
        check_is_instance(msa_features, "msa_static_features_dict", dict)

        # Check specific MSA feature keys
        msa_per_recycle = msa_features["msa_features_per_recycle_dict"]
        check_contains_keys(msa_per_recycle, ["msa", "has_insertion", "insertion_value"])
        msa_static = msa_features["msa_static_features_dict"]
        check_contains_keys(msa_static, ["profile", "insertion_mean"])

        # Check atom array annotations
        check_atom_array_annotation(data, ["coord_to_be_noised", "chain_iid", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Aggregates features into the format expected by AlphaFold 3.

        This method processes the input data, combining MSA features, ground truth
        structures, and other relevant information into a standardized format.

        Args:
            data (Dict[str, Any]): The input data dictionary containing MSA features,
                atom array, and other relevant information.

        Returns:
            Dict[str, Any]: The processed data dictionary with aggregated features.
        """
        # Initialize feats dictionary if not present
        if "feats" not in data:
            data["feats"] = {}

        # Aggregate and stack MSA features
        msa_feats = data["msa_features"]

        msa_stacked_by_recycle = torch.stack(
            [msa.float() for msa in msa_feats["msa_features_per_recycle_dict"]["msa"]]
        )  # [n_recycles, n_sequences, n_tokens_across_chains, n_types_of_tokens]
        has_deletion_stacked_by_recycle = torch.stack(
            msa_feats["msa_features_per_recycle_dict"]["has_insertion"]
        )  # [n_recycles, n_sequences, n_tokens_across_chains]
        deletion_value_stacked_by_recycle = torch.stack(
            msa_feats["msa_features_per_recycle_dict"]["insertion_value"]
        )  # [n_recycles, n_sequences, n_tokens_across_chains]

        data["feats"]["msa_stack"] = torch.concatenate(
            [
                msa_stacked_by_recycle,
                rearrange(has_deletion_stacked_by_recycle, "... -> ... 1"),
                rearrange(deletion_value_stacked_by_recycle, "... -> ... 1"),
            ],
            dim=-1,
        )  # [n_recycles, n_msa, n_tokens_across_chains, n_types_of_tokens + 2] (float)

        # Add pairing information if present
        if "residue_is_paired" in msa_feats["msa_features_per_recycle_dict"]:
            residue_is_paired_stacked_by_recycle = torch.stack(
                msa_feats["msa_features_per_recycle_dict"]["residue_is_paired"]
            )
            data["feats"]["msa_stack"] = torch.concatenate(
                [
                    data["feats"]["msa_stack"],
                    rearrange(residue_is_paired_stacked_by_recycle, "... -> ... 1"),
                ],
                dim=-1,
            )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, n_types_of_tokens + 3] (float)

        data["feats"] |= {
            "profile": msa_feats["msa_static_features_dict"]["profile"],
            "deletion_mean": msa_feats["msa_static_features_dict"]["insertion_mean"],
        }

        # NOTE: Each atom name is encoded as `ord(c) - 32`, which shifts the character values to create a
        # more compact one-hot encoding (as the first 32 Unicode characters will not occur in an atom name)
        data["feats"]["ref_atom_name_chars"] = F.one_hot(data["feats"]["ref_atom_name_chars"].long(), num_classes=64)

        # NOTE: the ref element is one-hot encoded by element number up to 128 (more than the known number of elements)
        data["feats"]["ref_element"] = F.one_hot(data["feats"]["ref_element"].long(), num_classes=128)

        # handle case where reference conformer was not able to be made and is currently nan
        data["feats"]["ref_pos"] = torch.nan_to_num(data["feats"]["ref_pos"], nan=0.0)

        # Process ground truth structure
        atom_array = data["atom_array"]
        coord_atom_lvl = atom_array.coord
        mask_atom_lvl = atom_array.occupancy > 0.0

        _token_rep_idxs = get_af3_token_representative_idxs(atom_array)
        coord_token_lvl = atom_array.coord[_token_rep_idxs]
        mask_token_lvl = atom_array.occupancy[_token_rep_idxs] > 0.0

        # ...get chain_iid for each token (needed in validation for scoring)
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        chain_iid_token_lvl = token_level_array.chain_iid

        # (We may already have ground_truth in the data, i.e., during validation, when we pass extra information for evaluation)
        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"].update(
            {
                "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                "coord_token_lvl": torch.tensor(coord_token_lvl),  # [n_tokens, 3], using the representative tokens
                "mask_token_lvl": torch.tensor(mask_token_lvl),  # [n_tokens], using the representative tokens
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
                "rep_atom_idxs": torch.tensor(_token_rep_idxs),  # [n_tokens]
            }
        )

        # data for symmetry resolution
        if "symmetry_resolution" not in data:
            data["symmetry_resolution"] = {}

        if "crop_info" not in data:
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["atom_array"].molecule_iid),
                    "crop_mask": torch.arange(data["atom_array"].shape[0]),
                    "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                    "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                }
            )
        else:
            token_starts = get_token_starts(data["crop_info"]["atom_array"])
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["crop_info"]["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["crop_info"]["atom_array"].molecule_iid),
                    "crop_mask": torch.tensor(data["crop_info"]["crop_atom_idxs"]),
                    "coord_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].coord),
                    "mask_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].occupancy > 0.0),
                }
            )

        # Add atom-level features for noising
        data["coord_atom_lvl_to_be_noised"] = torch.tensor(atom_array.coord_to_be_noised)

        return data
