import logging
from typing import Any, ClassVar

import numpy as np
import torch
from biotite.structure import AtomArray

from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger("atomworks.ml")


def featurize_atom_level_embeddings(
    atom_array: AtomArray,
    cached_residue_level_data: dict,
    residue_conformer_indices: dict[int, np.ndarray],
    ignore_res_names: list[str] | None = None,
    global_std: torch.Tensor | np.ndarray | None = None,
    global_mean: torch.Tensor | np.ndarray | None = None,
    threshold: float = 1e3,
    embedding_dim: int | None = None,
    n_conformers: int | None = None,
    p_dropout_atom_level_embeddings: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return atom-level embeddings and mask for each atom in the atom_array.

    For each atom, looks up its embedding by residue name and uses the selected conformer indices
    to concatenate embeddings along the batch dimension.

    If the global mean and standard deviation are provided, the embeddings are normalized to have zero mean and unit variance.

    Args:
        atom_array: AtomArray to featurize.
        cached_residue_level_data: Dict of cached data by residue name.
        residue_conformer_indices: Dict mapping global residue ID to selected conformer indices.
        ignore_res_names: List of residue names to ignore. If None, no residues are ignored.
        global_std: Global standard deviation of the embeddings (e.g., across all conformers of all residues). If None, no normalization is performed.
        global_mean: Global mean of the embeddings (e.g., across all conformers of all residues). If None, no normalization is performed.
        threshold: Maximum absolute value for descriptors. If any descriptor exceeds this threshold, the entire residue is ignored.
        embedding_dim: Dimensionality of the atom-level embeddings. If None, the dimensionality is inferred from the first available descriptors.
        n_conformers: Number of conformers to sample. If None, the number of conformers is inferred from the first available descriptors.
        p_dropout_atom_level_embeddings: Probability of dropping out the atom-level embeddings.

    Returns:
        dict: {'atom_level_embedding': (n_conformers, L, D), 'has_atom_level_embedding': (L,), 'mean_atom_level_embedding': (L, D)}
    """

    L = len(atom_array)
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    global_res_ids = atom_array.res_id_global

    # Infer dimensions from first available descriptors
    try:
        first_descriptors = next(
            res_data["descriptors"]
            for res_data in cached_residue_level_data.values()
            if res_data.get("descriptors") is not None
        )
        embedding_dim = embedding_dim or first_descriptors.shape[-1]
        # Get number of conformers from first residue instance
        n_conformers = n_conformers or (
            len(next(iter(residue_conformer_indices.values()))) if residue_conformer_indices else 0
        )
        _has_descriptors = True
    except (StopIteration, ValueError):
        assert (
            embedding_dim is not None and n_conformers is not None
        ), "embedding_dim and n_conformers must be provided if no descriptors are available"
        _has_descriptors = False

    default_return = {
        "atom_level_embedding": np.zeros((n_conformers, L, embedding_dim), dtype=np.float32),
        "has_atom_level_embedding": np.zeros(L, dtype=bool),
        "mean_atom_level_embedding": np.zeros((L, embedding_dim), dtype=np.float32),
    }

    if not _has_descriptors:
        return default_return

    if (p_dropout_atom_level_embeddings > 0.0) and (np.random.random() < p_dropout_atom_level_embeddings):
        # With probability p_dropout_atom_level_embeddings, drop out the atom-level embeddings (all 0's)
        return default_return

    # Initialize embeddings with shape (n_conformers, L, embedding_dim)
    embeddings = np.full((n_conformers, L, embedding_dim), np.nan, dtype=np.float32)
    has_embedding = np.zeros(L, dtype=bool)

    for i, (res_name, atom_name, global_res_id) in enumerate(zip(res_names, atom_names, global_res_ids, strict=False)):
        # (Skip checks)
        if ignore_res_names is not None and res_name in ignore_res_names:
            continue
        if res_name not in cached_residue_level_data or global_res_id not in residue_conformer_indices:
            continue

        res_data = cached_residue_level_data[res_name]
        if res_data.get("descriptors") is None or res_data.get("atom_names") is None:
            # ... no descriptors or atom names
            continue

        try:
            atom_idx = list(res_data["atom_names"]).index(atom_name)
        except ValueError:
            # ... atom name not found in atom names list
            continue

        conformer_indices = residue_conformer_indices[global_res_id]
        selected_descriptors = res_data["descriptors"][conformer_indices, atom_idx, :embedding_dim]

        # Check if any descriptor exceeds the threshold (diverged - likely a bad reference conformer)
        if np.any(np.abs(selected_descriptors) > threshold):
            continue

        if global_std is not None and global_mean is not None:
            global_std = global_std.numpy() if isinstance(global_std, torch.Tensor) else global_std
            global_mean = global_mean.numpy() if isinstance(global_mean, torch.Tensor) else global_mean

            # Normalize the descriptors to have unit variance and zero mean
            selected_descriptors = (selected_descriptors - global_mean[:embedding_dim]) / global_std[:embedding_dim]

        # Pad or truncate to match n_conformers
        n_selected = len(conformer_indices)
        if n_selected <= n_conformers:
            embeddings[:n_selected, i, :] = selected_descriptors
        else:
            embeddings[:, i, :] = selected_descriptors[:n_conformers]

        has_embedding[i] = True

    return {
        "atom_level_embedding": np.nan_to_num(embeddings),  # (n_conformers, L, D)
        "has_atom_level_embedding": has_embedding,  # (L,)
        "mean_atom_level_embedding": np.nan_to_num(embeddings.mean(axis=0)),  # (L, D)
    }


class FeaturizeAtomLevelEmbeddings(Transform):
    """Featurizes atom-level embeddings from cached data and adds them to the "feats" key.

    Uses cached residue-level data and conformer indices to create atom-level embeddings
    with a batch dimension for multiple conformers per residue.

    See `featurize_atom_level_embeddings` for details.

    Args:
        ignore_res_names (list[str] | None): List of residue names to ignore. If None, no residues are ignored.
        mask_rdkit_conformers (bool): Whether to mask the RDKit conformers where the atom level embedding IS present.
        threshold (float): Maximum absolute value for descriptors. If any descriptor exceeds this threshold, the entire residue is ignored.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "LoadCachedResidueLevelData",
        "RandomSubsampleCachedConformers",
        "AddGlobalResIdAnnotation",
    ]

    def __init__(
        self,
        ignore_res_names: list[str] | None = None,
        mask_rdkit_conformers: bool = False,
        threshold: float = 1e3,
        p_dropout_atom_level_embeddings: float = 0.0,
        embedding_dim: int | None = None,
        n_conformers: int | None = None,
    ):
        self.ignore_res_names = ignore_res_names
        self.mask_rdkit_conformers = mask_rdkit_conformers
        self.threshold = threshold
        self.p_dropout_atom_level_embeddings = p_dropout_atom_level_embeddings
        self.embedding_dim = embedding_dim
        self.n_conformers = n_conformers

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["res_id_global"])
        check_contains_keys(data, ["cached_residue_level_data", "residue_conformer_indices"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]
        assert "residues" in data["cached_residue_level_data"], "cached_residue_level_data must contain 'residues' key"

        cached_residue_level_data = data["cached_residue_level_data"]["residues"]
        residue_conformer_indices = data["residue_conformer_indices"]

        mean = None
        std = None
        if data["cached_residue_level_data"].get("metadata"):
            std = data["cached_residue_level_data"]["metadata"].get("std")
            mean = data["cached_residue_level_data"]["metadata"].get("mean")

        result = featurize_atom_level_embeddings(
            atom_array,
            cached_residue_level_data,
            residue_conformer_indices,
            ignore_res_names=self.ignore_res_names,
            global_std=std,
            global_mean=mean,
            threshold=self.threshold,
            p_dropout_atom_level_embeddings=self.p_dropout_atom_level_embeddings,
            embedding_dim=self.embedding_dim,
            n_conformers=self.n_conformers,
        )

        feats = data.setdefault("feats", {})
        feats.update(result)

        # (Optional) Mask the RDKit conformers where the atom level embedding IS present
        if self.mask_rdkit_conformers:
            assert "ref_pos" in feats
            mask = feats["has_atom_level_embedding"]
            feats["ref_pos"][mask] = 0.0
            feats["ref_mask"][mask] = 0.0

        assert all(
            key in feats for key in ("atom_level_embedding", "has_atom_level_embedding", "mean_atom_level_embedding")
        )

        return data
