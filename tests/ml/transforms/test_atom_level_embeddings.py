import numpy as np
import pytest

from atomworks.ml.transforms.atom_array import AddGlobalResIdAnnotation
from atomworks.ml.transforms.atom_level_embeddings import (
    FeaturizeAtomLevelEmbeddings,
    featurize_atom_level_embeddings,
)
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.cached_residue_data import (
    LoadCachedResidueLevelData,
    RandomSubsampleCachedConformers,
)
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def subsampled_cached_residue_data(cache_dir):
    """Fixture providing sample data with cached residue data and subsampled conformers."""
    data = cached_parse("1crn", hydrogen_policy="remove")

    pipeline = Compose(
        [
            AddGlobalResIdAnnotation(),
            LoadCachedResidueLevelData(dir=cache_dir, sharding_depth=1),
            RandomSubsampleCachedConformers(n_conformers=3),
        ]
    )

    return pipeline(data)


def test_featurize_with_unknown_residue(cache_dir, subsampled_cached_residue_data):
    """Test featurization with unknown residue names - should have False mask for unknown residues."""
    atom_array = subsampled_cached_residue_data["atom_array"].copy()

    # Find the first residue and change its name to something unknown
    first_res_id = atom_array.res_id_global[0]
    res_mask = atom_array.res_id_global == first_res_id
    unknown_res_name = "L:0"  # This should not exist in cached data
    atom_array.res_name[res_mask] = unknown_res_name

    # Featurize with the modified atom array
    assert "residues" in subsampled_cached_residue_data["cached_residue_level_data"]
    result = featurize_atom_level_embeddings(
        atom_array,
        subsampled_cached_residue_data["cached_residue_level_data"]["residues"],
        subsampled_cached_residue_data["residue_conformer_indices"],
    )

    has_embedding = result["has_atom_level_embedding"]

    # All atoms in the modified residue should have False in the mask
    assert not has_embedding[
        res_mask
    ].any(), f"Unknown residue {unknown_res_name} should have False mask for all its atoms"

    # Verify embeddings for unknown residue are all zeros
    embeddings = result["atom_level_embedding"]  # Shape: (n_conformers, L, D)
    unknown_embeddings = embeddings[:, res_mask]  # Index along the L dimension
    assert np.allclose(unknown_embeddings, 0), "Embeddings for unknown residue should be all zeros"


def test_featurize_atom_level_embeddings_transform(cache_dir, subsampled_cached_residue_data):
    """Test the FeaturizeAtomLevelEmbeddings transform class."""
    featurize_transform = FeaturizeAtomLevelEmbeddings()
    data = featurize_transform(subsampled_cached_residue_data)

    # Verify embeddings were added to feats
    assert "feats" in data
    assert "atom_level_embedding" in data["feats"]
    assert "has_atom_level_embedding" in data["feats"]

    embeddings = data["feats"]["atom_level_embedding"]
    has_embedding = data["feats"]["has_atom_level_embedding"]

    L = len(subsampled_cached_residue_data["atom_array"])
    assert embeddings.shape[1] == L  # Shape: (n_conformers, L, D)
    assert embeddings.ndim == 3
    assert has_embedding.shape[0] == L
    assert has_embedding.dtype == bool

    if has_embedding.sum() > 0:
        # Select atoms that have embeddings and check they're not all zeros
        non_zero_embeddings = embeddings[:, has_embedding]  # Shape: (n_conformers, n_atoms_with_embeddings, D)
        assert not np.allclose(non_zero_embeddings, 0)


def test_featurize_with_ignore_residues(cache_dir, subsampled_cached_residue_data):
    """Test featurization with ignored residue names."""
    # Get a residue name that exists in the data
    existing_res_names = list(subsampled_cached_residue_data["cached_residue_level_data"]["residues"].keys())
    if not existing_res_names:
        pytest.skip("No cached residue data available")

    ignore_res_name = existing_res_names[0]

    result = featurize_atom_level_embeddings(
        subsampled_cached_residue_data["atom_array"],
        subsampled_cached_residue_data["cached_residue_level_data"]["residues"],
        subsampled_cached_residue_data["residue_conformer_indices"],
        ignore_res_names=[ignore_res_name],
    )

    has_embedding = result["has_atom_level_embedding"]

    # Atoms with ignored residue names should have False in the mask
    atom_array = subsampled_cached_residue_data["atom_array"]
    ignored_mask = atom_array.res_name == ignore_res_name

    if ignored_mask.any():
        assert not has_embedding[
            ignored_mask
        ].any(), f"Ignored residue {ignore_res_name} should have False mask for all its atoms"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", __file__])
