import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from biotite.structure import AtomArray, residue_iter
from toolz import keyfilter

from atomworks.common import exists
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.io import get_sharded_file_path

logger = logging.getLogger("atomworks.ml")


def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


FILE_LOADERS: dict[str, Callable[[Path], Any]] = {
    ".pt": lambda path: torch.load(path, map_location="cpu", weights_only=False),
    ".pkl": _load_pkl,
    ".json": _load_json,
}


def load_cache_level_metadata(dir: str | Path, metadata_file: str | None) -> dict | None:
    """Load metadata file from the residue cache directory.

    Args:
        dir: Root directory containing cached files.
        metadata_file: Path to metadata file. Can be absolute or relative to the cache directory.
            If None, no metadata is loaded.
    """
    if metadata_file is None:
        return None

    # Determine if metadata_file is an absolute path or relative path
    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        # If relative, join with the cache directory
        metadata_path = Path(dir) / metadata_file

    if metadata_path.suffix not in FILE_LOADERS:
        raise ValueError(
            f"Unsupported metadata file extension: {metadata_path.suffix}. Supported: {', '.join(FILE_LOADERS.keys())}"
        )

    return FILE_LOADERS[metadata_path.suffix](metadata_path) if metadata_path.exists() else None


def load_cached_residue_level_data(
    atom_array: AtomArray,
    dir: str | Path,
    sharding_depth: int = 1,
    keys_to_load: list[str] | None = None,
    file_extension: str = ".pt",
    metadata_file: str | None = "global_stats.pt",
) -> dict[str, Any]:
    """Load cached residue-level data from sharded directory structure.

    Example directory structure:
    /path/to/cache/
    ├── A/
    │   ├── A_1.pt
    │   ├── A_2.pt
    │   └── ...
    ├── B/
    │   ├── B_1.pt
    │   ├── B_2.pt
    │   └── ...
    └── global_stats.pt

    Args:
        atom_array: AtomArray to extract residue names from.
        dir: Root directory containing cached files.
        sharding_depth: Depth of sharding (default=1).
        keys_to_load: List of keys to load from each file. If None, loads all available keys.
        file_extension: File extension for cached files (default=".pt"). Supports ".pt", ".pkl", and ".json".
        metadata_file: Path to metadata file (default="global_stats.pt"). Can be absolute or relative to the cache directory.
            Supports ".pt", ".pkl", and ".json". If None, no metadata is loaded.

    Returns:
        dict: Contains the following keys:
            - "residues": Maps residue name to loaded data dict containing the requested keys
            - "metadata": Loaded metadata dict, or None if metadata_file is None or file doesn't exist
    """
    if not exists(dir):
        return {
            "residues": {},
            "metadata": None,
        }

    if file_extension not in FILE_LOADERS:
        supported = ", ".join(FILE_LOADERS.keys())
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported: {supported}")

    loader = FILE_LOADERS[file_extension]

    metadata = load_cache_level_metadata(dir, metadata_file)

    unique_res_names = np.unique(np.array(atom_array.res_name))
    cached_data_by_res_name = {}

    for res_name in unique_res_names:
        file_path = get_sharded_file_path(
            base_dir=Path(dir),
            file_hash=res_name,
            extension=file_extension,
            depth=sharding_depth,
            chars_per_dir=1,
            include_subdirectory=True,
        )

        if not file_path.exists():
            logger.warning(f"Cached data not found for {res_name} at {file_path}")
            continue

        # Load and optionally filter the cached data
        cached_data = loader(file_path)
        if keys_to_load is not None:
            cached_data = keyfilter(lambda k: k in keys_to_load, cached_data)

        cached_data_by_res_name[res_name] = cached_data

    return {
        "residues": cached_data_by_res_name,
        "metadata": metadata,
    }


class LoadCachedResidueLevelData(Transform):
    """Load cached residue-level data from sharded directory structure.

    See `load_cached_residue_level_data` for details.
    """

    def __init__(
        self,
        dir: str | Path,
        sharding_depth: int = 1,
        keys_to_load: list[str] | None = None,
        file_extension: str = ".pt",
        metadata_file: str | None = "global_stats.pt",
    ):
        self.dir = dir
        self.sharding_depth = sharding_depth
        self.keys_to_load = keys_to_load
        self.file_extension = file_extension
        self.metadata_file = metadata_file

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]
        result = load_cached_residue_level_data(
            atom_array,
            dir=self.dir,
            sharding_depth=self.sharding_depth,
            keys_to_load=self.keys_to_load,
            file_extension=self.file_extension,
            metadata_file=self.metadata_file,
        )

        data["cached_residue_level_data"] = result
        return data


def random_subsample_cached_conformers(
    atom_array: AtomArray,
    cached_residue_level_data: dict,
    n_conformers: int,
    seed: int | None = None,
    sample_with_replacement: bool = False,
) -> dict[int, np.ndarray]:
    """Randomly subsample n conformers/descriptors per residue instance from cached data.

    For each residue instance in the AtomArray, randomly selects n conformer indices from the
    corresponding cached RDKit conformer data.

    Different instances of the same residue type can sample different conformers.

    Args:
        atom_array: AtomArray to extract residue instances from.
        cached_residue_level_data: Dict of cached data by residue name.
        n_conformers: Number of conformer indices to randomly select per residue instance.
        seed: Random seed for reproducibility. If None, uses random sampling.
        sample_with_replacement: Whether to sample with replacement (True) or without replacement (False).
            When False, ensures all sampled conformers are unique for each residue instance.

    Returns:
        dict: Maps global residue ID to sampled conformer indices (within the RDKit mol object).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    residue_conformer_indices = {}

    for residue in residue_iter(atom_array):
        # Get residue information from the first atom in the residue
        res_id = residue.res_id[0]
        res_name = residue.res_name[0]
        global_res_id = residue.res_id_global[0]

        # Skip if residue data not cached
        if res_name not in cached_residue_level_data:
            logger.debug(f"Skipping residue {res_name}_{res_id}: not in cached data")
            continue

        res_data = cached_residue_level_data[res_name]

        # Skip if no mol object
        if "mol" not in res_data or res_data["mol"] is None:
            logger.debug(f"Skipping residue {res_name}_{res_id}: no mol object found")
            continue

        n_available = res_data["mol"].GetNumConformers()
        if n_available > 0:
            # Randomly select conformer indices for this residue instance
            if n_available <= n_conformers:
                selected_indices = rng.permutation(n_available)
            else:
                selected_indices = rng.choice(n_available, size=n_conformers, replace=sample_with_replacement)

            # Store conformer indices mapped by global residue ID
            residue_conformer_indices[int(global_res_id)] = selected_indices

    logger.info(f"Subsampled conformer indices for {len(residue_conformer_indices)} residue instances")
    return residue_conformer_indices


class RandomSubsampleCachedConformers(Transform):
    """Randomly subsample n conformers/descriptors per residue instance from cached data.

    See `random_subsample_cached_conformers` for details.

    Args:
        n_conformers (int): Number of conformer indices to randomly select per residue instance.
        seed (int | None): Random seed for reproducibility. If None, uses random sampling.
        sample_with_replacement (bool): Whether to sample with replacement. Defaults to True.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "LoadCachedResidueLevelData",
        "AddGlobalResIdAnnotation",
    ]

    def __init__(
        self,
        n_conformers: int,
        seed: int | None = None,
        sample_with_replacement: bool = True,
    ):
        self.n_conformers = n_conformers
        self.seed = seed
        self.sample_with_replacement = sample_with_replacement

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["res_id_global"])
        check_contains_keys(data, ["cached_residue_level_data"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]
        assert "residues" in data["cached_residue_level_data"], "cached_residue_level_data must contain 'residues' key"
        cached_residue_level_data = data["cached_residue_level_data"]["residues"]

        residue_conformer_indices = random_subsample_cached_conformers(
            atom_array,
            cached_residue_level_data,
            n_conformers=self.n_conformers,
            seed=self.seed,
            sample_with_replacement=self.sample_with_replacement,
        )

        data["residue_conformer_indices"] = residue_conformer_indices
        return data
