import itertools
import logging
import math
from collections.abc import Iterator, Sequence
from operator import add

import numpy as np
import pandas as pd
import torch
from toolz import accumulate
from torch.utils.data import Dataset, DistributedSampler, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


def calculate_af3_example_weights(df: pd.DataFrame, alphas: dict[str, float], beta: float) -> pd.Series:
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
        as proteins with fewer than PEPTIDE_MAX_RESIDUES residues (see `atomworks.ml.preprocessing.constants`).
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
    required_columns = ["n_prot", "n_nuc", "n_ligand", "cluster_size"]
    assert all(col in df.columns for col in required_columns), (
        "Missing required columns in the (loaded) DataFrame. "
        f"Please ensure the DataFrame contains the following columns: {required_columns}"
        "Also ensure that the columns to include are specified in the Hydra configuration file."
    )
    # Extract relevant columns with default handling
    n_prot = df["n_prot"]
    n_nuc = df["n_nuc"]
    n_ligand = df["n_ligand"]
    n_peptide = df["n_peptide"]
    cluster_size = df["cluster_size"]

    # For interfaces, the column "involves_loi" indicates whether the interface involves a ligand of interest
    # For pn_units, the column "q_pn_unit_is_loi" indicates whether the query PN Unit is a ligand of interest
    # (1 = True, 0 = False)
    assert "involves_loi" in df.columns or "q_pn_unit_is_loi" in df.columns, (
        "Missing column for 'involves_loi' or 'q_pn_unit_is_loi'. "
        "Please check the columns in the DataFrame: {df.columns}, "
        "and the columns to include specified in the Hydra configuration file."
    )
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


def get_cluster_sizes(df: pd.DataFrame, cluster_column: str = "cluster") -> dict[str, int]:
    """Generate a mapping between cluster alphanumeric IDs and the number of PN units/interfaces in each cluster.

    Args:
        df (pd.DataFrame): DataFrame containing the PN unit or interface data
        cluster_column (str): Name of the column containing the cluster alphanumeric IDs

    Returns:
        dict: A dictionary where the keys are unique cluster IDs and the values are the counts of occurrences.
    """
    # Use the value_counts method to count occurrences of each unique value in the cluster column
    cluster_counts = df[cluster_column].value_counts()

    # Convert the Series to a dictionary and return
    return cluster_counts.to_dict()


def calculate_weights_for_pdb_dataset_df(
    dataset_df: pd.DataFrame, alphas: dict[str, float], beta: float, cluster_column: str = "cluster"
) -> torch.Tensor:
    """Calculate weights for each row in the DataFrame based on the cluster size and the AF-3 weighting methodology.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing the PN unit or interface data
        alphas (dict[str, float]): Dictionary containing alpha values for the weighting calculation (common across interfaces and chains/pn_units)
        beta (float): Beta value for the weighting calculation (distinct for interfaces and chains/pn_units)

    Returns:
        torch.Tensor: A tensor containing the calculated weights for each row in the DataFrame
    """
    # Generate the cluster sizes...
    cluster_id_to_size_map = get_cluster_sizes(dataset_df, cluster_column=cluster_column)

    # ...map the cluster sizes to the DataFrame
    dataset_df["cluster_size"] = dataset_df[cluster_column].map(cluster_id_to_size_map)

    # ... assert no NaN cluster sizes
    assert not dataset_df["cluster_size"].isnull().any(), "Cluster sizes must not be NaN"

    # ...calculate weights using vectorized operations
    weights = calculate_af3_example_weights(dataset_df, alphas, beta).values

    # ...and return the weights as a tensor
    return torch.tensor(weights)


def calculate_weights_by_inverse_cluster_size(
    dataset_df: pd.DataFrame, cluster_column: str = "cluster"
) -> torch.Tensor:
    """Calculate weights for each row in the DataFrame as the inverse of its cluster size.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing the PN unit or interface data
        cluster_column (str): Column name in `dataset_df` corresponding to the cluster info. Default is "cluster".

    Returns:
        torch.Tensor: A tensor containing the calculated weights for each row in the DataFrame
    """
    # Generate the cluster sizes...
    cluster_id_to_size_map = get_cluster_sizes(dataset_df, cluster_column=cluster_column)

    # ... map the cluster sizes to the DataFrame
    dataset_df["cluster_size"] = dataset_df[cluster_column].map(cluster_id_to_size_map)

    # ... calculate weights as the inverse of the cluster size
    weights = 1 / dataset_df["cluster_size"].values

    # ... and return the weights as a tensor
    return torch.tensor(weights)


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

    Example:
        Imagine we have the following sampling tree:

                DistributedMixedSampler
                           |
                -------------------------
                |                       |
               0.8                     0.2
             Sampler1              MixedSampler
                                    /       \
                                   0.9       0.1
                                Sampler2   Sampler3

        If we initialized DistributedMixedSampler with `n_examples_per_epoch=100` and `num_replicas=2`, it would collect 80 samples
        from Sampler1 and 20 samples from the MixedSampler. The MixedSampler would in turn collect 18 samples from Sampler2 and 2 samples from Sampler3.
        After collecting those 100 samples, the DistributedMixedSampler would shard the samples across the two nodes, ensuring each node receives a unique slice
        of 50 examples.

        If any of the child samplers were distributed samples, then the DistributedMixedSampler would not receive n_examples_per_epoch indices,
        and we would raise an error.

    NOTE: The order of the datasets in datasets_info MUST match the order of the datasets in the ConcatDataset associated with this MixedSampler.

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
