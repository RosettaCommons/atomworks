import itertools
import logging
import math
from collections.abc import Iterator, Sequence
from operator import add

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from toolz import accumulate
from torch.utils.data import Dataset, DistributedSampler, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


def _calculate_af3_example_weights(df: pd.DataFrame, alphas: dict[str, float], beta: float) -> pd.Series:
    """Determines the weight of each example in the DataFrame using a methodology inspired by AF-3.

    In AF-3, the weight of a given example is a function of:
        (1) The size of the cluster to which the example belongs (specific for interfaces vs. chains)
        (2) The number of proteins / nucleic acids / ligands in the example
        (3) Whether the example is an interface or a chain

    Specifically, AF3 gives the following formula (Section 2.5.1 from the AF-3 Supplementary Information):
        w ∝ (β_r / N_clust) * (a_prot * n_prot + a_nuc * n_nuc + a_ligand * n_ligand)

    Where:
        - w is the weight of the example
        - β_r is a weighting hyperparameter that is distinct for interfaces and chains
        - N_clust is the number of examples in the cluster
        - a_prot, a_nuc, and a_ligand are the interface weight hyperparameters for proteins, nucleic acids, and ligands, respectively
        - n_prot, n_nuc, and n_ligand are the number of proteins, nucleic acids, and ligands in the example

    We make the following modifications to the original AF-3 formula:
        - We introduce n_peptide and a_peptide to better control the sampling over peptides (which were being over-sampled). We define peptides
        as proteins with fewer than PEPTIDE_MAX_RESIDUES residues (see `atomworks.constants`).
        - We introduce an incremental a_loi weight to control the sampling of ligands of interests (LOI), also described as Subject of Investigation.

    Thus, our full formula is:
        w ∝ (β_r / N_clust) * (a_prot * n_prot + a_peptide * n_peptide + a_nuc * n_nuc + a_ligand * n_ligand + a_loi * is_loi)

    Args:
        df (pd.DataFrame): DataFrame containing the PN unit or interface data
        alphas (dict): Dictionary containing the weight hyperparameters for proteins, nucleic acids, ligands, and possibly peptides (common across interfaces and chains)
        beta (float): Weighting hyperparameter (distinct for interfaces and chains)

    Returns:
        pd.Series: A Series containing the calculated weights for each row in the DataFrame
    """
    # Extract relevant columns with default handling
    n_prot = df["n_prot"]
    n_nuc = df["n_nuc"]
    n_ligand = df["n_ligand"]
    n_peptide = df["n_peptide"]
    cluster_size = df["cluster_size"]

    is_loi = (df["involves_loi"] if "involves_loi" in df.columns else df["q_pn_unit_is_loi"]).astype(int)

    # Assert that all cluster sizes are greater than 0
    assert all(cluster_size > 0), "All cluster sizes must be greater than 0"

    # Warn if not all cluster sizes are less than the dataframe length
    if not all(cluster_size < len(df)):
        logger.warning(
            "Some cluster sizes are greater than the DataFrame length. "
            "This is unexpected, unless you are running with a very "
            "restricted dataframe for debugging. If you aren't, please check!"
        )

    # If we're missing any of the alphas, or any of the counts, log a warning
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_peptide", "a_nuc", "a_ligand", "a_loi"}
    missing_counts = {"n_prot", "n_peptide", "n_nuc", "n_ligand"} - set(df.columns)

    if missing_alphas:
        logger.warning(f"Missing alphas from configuration file: {missing_alphas}; defaulting to 0")
    if missing_counts:
        logger.warning(f"Missing chain within dataframe counts: {missing_counts}; defaulting to 0")
        logger.warning(f"Columns in dataframe: {df.columns}")

    logger.info(f"Calculating weights for AF-3 examples using alphas={alphas}, beta={beta}")

    # Vectorized calculation of the weights
    weights = (beta / cluster_size) * (
        alphas.get("a_prot", 0) * n_prot
        + alphas.get("a_peptide", 0) * n_peptide
        + alphas.get("a_nuc", 0) * n_nuc
        + alphas.get("a_ligand", 0) * n_ligand
        + alphas.get("a_loi", 0) * is_loi
    )

    return weights


def _col_to_series(table: pa.Table | pd.DataFrame, column: str) -> pd.Series:
    """Extract a single column from a pandas DataFrame or PyArrow Table as a pandas Series.

    Only the requested column is materialised, keeping memory usage low when the
    backing table is a PyArrow Table held in columnar format.
    """
    col = table[column]
    return col.to_pandas() if hasattr(col, "to_pandas") else col


def _get_effective_cluster_sizes(
    dataset_df: pa.Table | pd.DataFrame,
    cluster_column: str,
    altloc_weights: pd.Series | None = None,
) -> pd.Series:
    """Return a per-row Series of effective cluster sizes.

    Without altloc weights, effective size is simply the row count per cluster.
    With altloc weights, effective size is the sum of altloc weights per cluster,
    which equals the number of unique examples (e.g. pdb_ids / assembly id / pn_unit)
    since each unique example's altloc weights sum to exactly 1.0.

    Args:
        dataset_df: DataFrame or PyArrow Table containing the data.
        cluster_column: Column identifying sequence clusters.
        altloc_weights: If provided, summed per cluster to get effective size.
            Defaults to ``None`` (use raw row counts).

    Returns:
        Per-row Series of effective cluster sizes.
    """
    cluster_col = _col_to_series(dataset_df, cluster_column)
    if altloc_weights is None:
        cluster_to_size = cluster_col.value_counts().to_dict()
    else:
        tmp = pd.DataFrame({"cluster": cluster_col, "altloc_weight": altloc_weights})
        cluster_to_size = tmp.groupby("cluster")["altloc_weight"].sum().to_dict()
    return cluster_col.map(cluster_to_size)


def _get_altloc_weights(
    dataset_df: pa.Table | pd.DataFrame,
    altloc_seed_column: str,
    pdb_id_column: str,
) -> pd.Series:
    """Return a per-row Series of ``1 / num_altlocs`` for altloc-aware sampling.

    ``num_altlocs`` is the number of unique values in ``altloc_seed_column`` per
    ``pdb_id``.  ``NaN`` counts as one unique value (structures with no altlocs
    have a single row with ``altloc_seed=None``, so ``num_altlocs=1``).
    """
    pdb_id_col = _col_to_series(dataset_df, pdb_id_column)
    altloc_col = _col_to_series(dataset_df, altloc_seed_column)
    tmp = pd.DataFrame({pdb_id_column: pdb_id_col, altloc_seed_column: altloc_col})
    pdb_id_to_num_altlocs = tmp.groupby(pdb_id_column)[altloc_seed_column].nunique(dropna=False).to_dict()
    return 1.0 / pdb_id_col.map(pdb_id_to_num_altlocs)


def calculate_weights_for_pdb_dataset_df(
    dataset_df: pa.Table | pd.DataFrame,
    alphas: dict[str, float],
    beta: float,
    cluster_column: str = "cluster",
    altloc_seed_column: str | None = None,
    pdb_id_column: str = "pdb_id",
) -> torch.Tensor:
    """Calculate weights based on the AF-3 methodology, optionally adjusted for altlocs.

    Base weight per row: ``(beta / cluster_size) * (a_prot * n_prot + ...)``.
    If ``altloc_seed_column`` is provided, weights are additionally multiplied by
    ``1 / num_altlocs`` so that each altloc variant of a structure is sampled uniformly.

    Args:
        dataset_df: DataFrame or PyArrow Table containing the PN unit or interface data.
        alphas: Alpha hyperparameters for the AF-3 weighting formula.
        beta: Beta hyperparameter (distinct for interfaces vs. chains).
        cluster_column: Column identifying sequence clusters. Defaults to ``"cluster"``.
        altloc_seed_column: If provided, weights are divided by the number of unique
            altloc seeds per ``pdb_id``. Defaults to ``None`` (no altloc adjustment).
        pdb_id_column: Column identifying structures, used only when
            ``altloc_seed_column`` is set. Defaults to ``"pdb_id"``.

    Returns:
        Tensor of per-row weights with shape ``(len(dataset_df),)``.
    """
    col_names = dataset_df.schema.names if isinstance(dataset_df, pa.Table) else list(dataset_df.columns)

    required_columns = [cluster_column, "n_prot", "n_nuc", "n_ligand", "n_peptide"]
    assert all(col in col_names for col in required_columns), (
        "Missing required columns in the (loaded) table. "
        f"Please ensure the table contains the following columns: {required_columns}. "
        "Also ensure that the columns to include are specified in the Hydra configuration file."
    )
    assert "involves_loi" in col_names or "q_pn_unit_is_loi" in col_names, (
        "Missing column for 'involves_loi' or 'q_pn_unit_is_loi'. "
        f"Please check the columns in the table: {col_names}, "
        "and the columns to include specified in the Hydra configuration file."
    )

    loi_col = "involves_loi" if "involves_loi" in col_names else "q_pn_unit_is_loi"
    needed_cols = [cluster_column, "n_prot", "n_nuc", "n_ligand", "n_peptide", loi_col]
    df = pd.DataFrame({col: _col_to_series(dataset_df, col) for col in needed_cols})

    altloc_weights = (
        _get_altloc_weights(dataset_df, altloc_seed_column, pdb_id_column) if altloc_seed_column is not None else None
    )
    df["cluster_size"] = _get_effective_cluster_sizes(dataset_df, cluster_column, altloc_weights)
    assert not df["cluster_size"].isnull().any(), "Cluster sizes must not be NaN"

    weights = _calculate_af3_example_weights(df, alphas, beta).values
    if altloc_weights is not None:
        weights = weights * altloc_weights.values
    return torch.tensor(weights)


def calculate_weights_by_inverse_cluster_size(
    dataset_df: pa.Table | pd.DataFrame,
    cluster_column: str = "cluster",
    altloc_seed_column: str | None = None,
    pdb_id_column: str = "pdb_id",
) -> torch.Tensor:
    """Calculate weights as the inverse of cluster size, optionally adjusted for altlocs.

    Base weight per row: ``1 / cluster_size``.
    If ``altloc_seed_column`` is provided, weights are additionally multiplied by
    ``1 / num_altlocs`` for uniform three-level sampling: cluster → example → altloc.

    Args:
        dataset_df: DataFrame or PyArrow Table containing the PN unit or interface data.
        cluster_column: Column identifying sequence clusters. Defaults to ``"cluster"``.
        altloc_seed_column: If provided, weights are divided by the number of unique
            altloc seeds per ``pdb_id``. Defaults to ``None`` (no altloc adjustment).
        pdb_id_column: Column identifying structures, used only when
            ``altloc_seed_column`` is set. Defaults to ``"pdb_id"``.

    Returns:
        Tensor of per-row weights with shape ``(len(dataset_df),)``.
    """
    altloc_weights = (
        _get_altloc_weights(dataset_df, altloc_seed_column, pdb_id_column) if altloc_seed_column is not None else None
    )
    effective_cluster_sizes = _get_effective_cluster_sizes(dataset_df, cluster_column, altloc_weights)
    weights = 1.0 / effective_cluster_sizes
    if altloc_weights is not None:
        weights = weights * altloc_weights
    return torch.tensor(weights.values, dtype=torch.float64)


def set_sampler_epoch(sampler: Sampler, epoch: int, add_random_offset: bool = False) -> None:
    """Control the random seed for a sampler."""
    if add_random_offset:
        epoch += torch.randint(-int(1e12), int(1e12), (1,)).item()

    logger.info(f"Setting epoch for sampler {sampler} to {epoch}")

    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    elif hasattr(sampler, "generator"):
        if sampler.generator is None:
            sampler.generator = torch.Generator()
        sampler.generator.manual_seed(epoch)
    else:
        logger.warning(
            f"Sampler {sampler} does not have a set_epoch method or generator attribute, so epoch cannot be set."
        )


class DistributedMixedSampler(Sampler):
    """Custom DistributedSampler implementation that samples from an arbitrary list of samplers with specified probabilities.

    Child samplers can be any type of non-distributed sampler, including a MixedSampler.
    After gathering all indices, shards the samples across nodes, ensuring each node receives a unique slice of the dataset.

    Child samplers must not shard their own output. For 100 examples with probabilities
    0.8 and 0.2, collect 80 and 20 samples respectively, then shard across replicas.
    The order of ``datasets_info`` must match the associated ``ConcatDataset``.

    Args:
        datasets_info: List of dictionaries, where each dictionary must contain at a minimum:
            - "sampler": Sampler object for the dataset
            - "dataset": Dataset object associated with the sampler
            - "probability": Probability of sampling from this dataset
        num_replicas: Number of replicas (nodes) in the distributed setting
        rank: Rank of the current node
        n_examples_per_epoch: Number of examples in an epoch. Effectively, the "length" of the sampler (since we often sample with replacement).
            May be None, in which case the number of examples per epoch must be set dynamically by a parent sampler.
        shuffle: Whether to shuffle the indices. If False, the iterator will return all sampled indices from the first dataset, then the second, etc.
        drop_last: Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size

    Returns:
        iter: An iterator over indices of the dataset for the current process (of length n_samples, not n_examples_per_epoch)

    Reference:
        `PyTorch DistributedSampler <https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L68>`_
    """

    def __init__(
        self,
        datasets_info: list[dict[str, any]],
        num_replicas: int,
        rank: int,
        n_examples_per_epoch: int | None,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.datasets_info = datasets_info
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.epoch = 0  # Initialize epoch to 0
        self.samplers = [info["sampler"] for info in datasets_info]  # ordered
        self.probabilities = [info["probability"] for info in datasets_info]  # ordered
        self.dataset_lengths = [len(info["dataset"]) for info in datasets_info]  # ordered

        # Calculate cumulative lengths of datasets (so we can map local dataset indices to ConcatDataset indices)
        self.cumulative_lengths = [0, *list(accumulate(add, self.dataset_lengths))]  # ordered
        # Remove the last element to match the other list shapes
        self.cumulative_lengths = self.cumulative_lengths[:-1]

        # Assert that:
        # ... the number of samplers, probabilities, and datasets match
        assert len(self.samplers) == len(self.probabilities) == len(self.dataset_lengths)
        # ... the probabilities sum to 1
        assert abs(sum(self.probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1"
        # ... the datasets_info contains keys for "sampler", "probability", and "dataset"
        assert "sampler" in datasets_info[0] and "probability" in datasets_info[0] and "dataset" in datasets_info[0]

        if n_examples_per_epoch is not None:
            self._set_num_examples_per_epoch(n_examples_per_epoch)

    def _set_num_examples_per_epoch(self, n_examples_per_epoch: int) -> None:
        """Set the number of examples per epoch, and update the number of examples per epoch for each sampler.

        Allows for dynamic setting and propagation of the number of examples per epoch.

        Args:
            n_examples_per_epoch: Number of examples in an epoch. Effectively, the "length" of the sampler
        """
        self.n_examples_per_epoch = n_examples_per_epoch

        # If the number of examples per epoch is not evenly divisible by the number of replicas, there
        # is no need to drop any data, since the examples will be split equally.
        if self.drop_last and self.n_examples_per_epoch % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when using this Sampler.
            self.n_samples = math.ceil((self.n_examples_per_epoch - self.num_replicas) / self.num_replicas)
        else:
            self.n_samples = math.ceil(self.n_examples_per_epoch / self.num_replicas)

        self.epoch = 0  # Initialize epoch to 0
        self.total_size = (
            self.n_samples * self.num_replicas
        )  # May be greater than n_examples_per_epoch, which we will handle in __iter__

        # Create a list representing the number of items to sample from each dataset (sampler)
        self.n_examples_per_dataset = [math.ceil(prob * self.total_size) for prob in self.probabilities]  # ordered

        for sampler, n_examples in zip(self.samplers, self.n_examples_per_dataset, strict=False):
            # Set the `n_examples_per_epoch` for each sampler, if they allow it...
            # NOTE: Required for MixedSamplers, which must continue propagating the number of examples per epoch
            if hasattr(sampler, "_set_num_examples_per_epoch"):
                sampler._set_num_examples_per_epoch(n_examples)

            # ... override the `num_samples` attribute if it exists (e.g., for WeightedRandomSampler)
            if hasattr(sampler, "num_samples"):
                sampler.num_samples = n_examples

            # ... and assert that either we have more than n_examples_per_epoch examples or we are sampling with replacement
            sampler_has_enough_data = len(sampler) >= n_examples
            sampler_is_replacement = getattr(sampler, "replacement", False)
            assert (
                sampler_has_enough_data or sampler_is_replacement
            ), "Must either have enough data or be sampling with replacement"

    def __iter__(self):
        # Trigger the __iter__ of each sampler upfront (generates a list of local indices based on the sampling scheme)
        sampler_iters = [iter(sampler) for sampler in self.samplers]

        # Take the first n_examples_per_dataset indices from each sampler
        indices = [
            list(itertools.islice(sampler_iter, n))
            for sampler_iter, n in zip(sampler_iters, self.n_examples_per_dataset, strict=False)
        ]

        # Convert to global indices
        for i in range(1, len(indices)):
            indices[i] = [index + self.cumulative_lengths[i] for index in indices[i]]

        # Flatten the list of local indices
        indices = [index for sublist in indices for index in sublist]

        padding_size = self.total_size - len(indices)
        if not self.drop_last and padding_size > 0:
            # Add extra samples to make it evenly divisible
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size, f"Expected {self.total_size} indices, got {len(indices)}"

        # Randomly permute the global indices (otherwise, we will sample one dataset first, then the next, etc.)
        if self.shuffle:
            # Set the seed based on the epoch
            indices = torch.tensor(indices)
            g = torch.Generator()
            g.manual_seed(self.epoch)

            # Randomly permute the global indices
            permuted_indices = torch.randperm(len(indices), generator=g)
            indices = indices[permuted_indices]

            # Back to list
            indices = indices.tolist()

        # Subsample
        # This samples [0, num_replicas, 2*num_replicas, ...] for node 0,
        # [1, num_replicas+1, 2*num_replicas+1...] for node 1, and so on
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.n_samples

        return iter(indices)

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for sampler in self.samplers:
            set_sampler_epoch(sampler, epoch)


class MixedSampler(DistributedMixedSampler):
    """A non-distributed sampler that samples from an arbitrary list of samplers with specified probabilities.

    This class acts like a DistributedMixedSampler with `rank=0` and `num_replicas=1`.

    Args:
        datasets_info: List of dictionaries, where each dictionary must contain at a minimum:
            - "sampler": Sampler object for the dataset
            - "dataset": Dataset object associated with the sampler
            - "probability": Probability of sampling from this dataset
        n_examples_per_epoch: Number of examples in an epoch. Effectively, the "length" of the sampler.
        shuffle: Whether to shuffle the indices. If False, the iterator will return all sampled indices from the first dataset, then the second, etc.
    """

    def __init__(
        self,
        datasets_info: list[dict[str, any]],
        n_examples_per_epoch: int | None = None,
        shuffle: bool = True,
    ):
        super().__init__(
            datasets_info=datasets_info,
            num_replicas=1,
            rank=0,
            n_examples_per_epoch=n_examples_per_epoch,
            shuffle=shuffle,
        )


class FallbackSamplerWrapper(Sampler):
    """A wrapper around a sampler that allows for a fallback sampler to be used when an error occurs.

    Meant to be used with a FallbackDatasetWrapper.
    """

    def __init__(self, sampler: Sampler, fallback_sampler: Sampler, n_fallback_retries: int = 2):
        self.sampler = sampler
        self.fallback_sampler = fallback_sampler
        self.n_fallback_retries = n_fallback_retries

    def __iter__(self):
        # Create a list of iterators, each of which will yield the next n_fallback_retries indices from the fallback sampler
        fallback_iterators = [itertools.cycle(iter(self.fallback_sampler)) for _ in range(self.n_fallback_retries)]
        iterators = [iter(self.sampler), *fallback_iterators]
        return zip(*iterators, strict=False)

    def __len__(self):
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        set_sampler_epoch(self.sampler, epoch)
        set_sampler_epoch(self.fallback_sampler, epoch, add_random_offset=True)


class LazyWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
        prefetch_buffer_size: int = 1,
    ) -> None:
        assert replacement, "LazyWeightedRandomSampler only supports replacement=True"
        super().__init__(weights, num_samples, replacement, generator)
        self.prefetch_buffer_size = prefetch_buffer_size

        # We cannot use torch.multinomial with > 2^24 categories (and MGnify validation has more than this)
        # precompute sampling probabilities
        weights_np = self.weights.cpu().numpy() if self.weights.is_cuda else self.weights.numpy()
        self.cumsum = np.cumsum(weights_np, dtype=np.float64)
        self.cumsum = self.cumsum / self.cumsum[-1]  # Normalize to [0, 1]

    def __iter__(self):
        prefetch_buffer = []

        for _ in range(self.num_samples):
            if not prefetch_buffer:
                # Pull another buffer of length `prefetch_buffer_size`
                # Use inverse transform sampling with precomputed CDF
                random_values = torch.rand(self.prefetch_buffer_size, generator=self.generator).cpu().numpy()
                prefetch_buffer = np.searchsorted(self.cumsum, random_values).tolist()

            yield prefetch_buffer.pop(0)


class LoadBalancedDistributedSampler(DistributedSampler):
    """DistributedSampler that balances large examples across replicas.

    Helpful for validation, where we don't want GPUs to be idle while waiting for the slowest replica to finish.

    For example, we may want to avoid the scenario where one GPU receives many large examples that are slow to process,
    while another GPU receives many small examples that are quick to process.

    NOTE: Only useful for validation, as the order of the examples is deterministic.

    Args:
        dataset: Dataset used for sampling.
        key_to_balance: Key in the dataset data dataframe that contains the length (size) of each example.
            The dataset must have a data attribute that can be accessed like a dataframe.
            For example, if the dataset has a data attribute that is a pandas DataFrame, the key_to_balance
            should be a column in that DataFrame (i.e., "n_tokens").
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        key_to_balance: str,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,  # No shuffling when we try and balance across replicas
            drop_last=drop_last,
        )
        self.length_key = key_to_balance

    def __iter__(self) -> Iterator[int]:
        # Extract sizes from the dataset
        sizes = self.dataset.data[self.length_key]
        indices = list(range(len(sizes)))

        # Sort indices by example size
        indices.sort(key=lambda x: sizes[x], reverse=True)

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[-padding_size:]  # Add from the end of the list, which are the smallest examples
                else:
                    indices += indices[-1:] * padding_size
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
