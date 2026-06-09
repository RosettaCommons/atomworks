"""Dataset concatenation and wrapper utilities."""

import copy
import logging
from functools import cached_property
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from .base import ExampleIDMixin

logger = logging.getLogger("datasets")


class ConcatDatasetWithID(ConcatDataset):
    """Equivalent to :class:`torch.utils.data.ConcatDataset` but allows accessing examples by ID.

    Provides ID-based access across multiple datasets that implement :class:`ExampleIDMixin`.
    """

    # TODO: Do I need all of these _raise_if etc. etc. here? Can I just check that the wrapped datasets inherit somehow from ExampleIDMixin?

    datasets: list[ExampleIDMixin]

    def __init__(self, datasets: list[ExampleIDMixin]):
        """Initialize ConcatDatasetWithID.

        Args:
            datasets: List of datasets that implement ExampleIDMixin.
        """
        super().__init__(datasets)

        # Log the length of each dataset
        for i, dataset in enumerate(datasets):
            logger.info(f"Dataset {i} ({type(dataset)}): {len(dataset):,} examples")

    @cached_property
    def _can_convert_ids_and_idx(self) -> bool:
        """Check if all sub-datasets can convert between IDs and indices."""
        has_id_to_idx = all(hasattr(sub_dataset, "id_to_idx") for sub_dataset in self.datasets)
        has_idx_to_id = all(hasattr(sub_dataset, "idx_to_id") for sub_dataset in self.datasets)
        return has_id_to_idx and has_idx_to_id and self._can_check_contains

    @cached_property
    def _can_check_contains(self) -> bool:
        """Check if all sub-datasets support contains operations."""
        return all(hasattr(sub_dataset, "__contains__") for sub_dataset in self.datasets)

    def _raise_if_cannot_check_contains(self) -> None:
        """Raise error if dataset cannot check contains."""
        if not self._can_check_contains:
            raise ValueError("This dataset cannot check if it contains an example ID.")

    def _raise_if_cannot_convert_ids_and_idx(self) -> None:
        """Raise error if dataset cannot convert IDs and indices."""
        if not self._can_convert_ids_and_idx:
            raise ValueError("This dataset cannot convert example IDs to indices.")

    def _raise_if_idx_out_of_bounds(self, idx: int) -> None:
        """Raise error if index is out of bounds.

        Args:
            idx: The index to check.
        """
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of bounds for dataset of length {len(self)}.")

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID.

        Args:
            example_id: The ID to check for.

        Returns:
            True if the ID exists in any sub-dataset.
        """
        self._raise_if_cannot_check_contains()
        return any(example_id in sub_dataset for sub_dataset in self.datasets)

    def id_to_idx(self, example_id: str) -> int:
        """Retrieves the index corresponding to the example ID.

        Args:
            example_id: The ID to convert.

        Returns:
            The corresponding index.

        Raises:
            ValueError: If the example ID is not found.

        Warning:
            Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        # TODO: Generalize to list[str]
        self._raise_if_cannot_convert_ids_and_idx()
        offset = 0
        for sub_dataset in self.datasets:
            if example_id in sub_dataset:
                return offset + sub_dataset.id_to_idx(example_id)
            offset += len(sub_dataset)
        raise ValueError(f"Example ID {example_id} not found in any sub-dataset.")

    def idx_to_id(self, idx: int) -> str:
        """Retrieves the example ID corresponding to the index.

        Args:
            idx: The index to convert.

        Returns:
            The corresponding example ID.

        Raises:
            ValueError: If the index is out of bounds.
        """
        # TODO: Generalize to list[int]
        self._raise_if_cannot_convert_ids_and_idx()
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset.idx_to_id(idx)
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_idx(self, idx: int) -> Dataset:
        """Retrieves the dataset containing the index.

        Args:
            idx: The index to find.

        Returns:
            The sub-dataset containing the index.

        Raises:
            ValueError: If the index is out of bounds.
        """
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_id(self, example_id: str) -> Dataset:
        """Retrieves the dataset containing the example ID.

        Args:
            example_id: The ID to find.

        Returns:
            The sub-dataset containing the ID.

        Warning:
            Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        idx = self.id_to_idx(example_id)
        return self.get_dataset_by_idx(idx)


def get_row_and_index_by_example_id(dataset: ExampleIDMixin, example_id: str) -> dict:
    """Retrieve a row and its index from a nested dataset structure by its example ID.

    Args:
        dataset: The dataset or concatenated dataset to search.
            Must have the `id_to_idx` method.
        example_id: The example ID to search for.

    Returns:
        Dictionary containing the row and global index corresponding to the example ID.
    """
    assert hasattr(dataset, "id_to_idx"), "Dataset must have the `id_to_idx` method."
    idx = dataset.id_to_idx(example_id)

    _local_idx = copy.deepcopy(idx)
    while isinstance(dataset, ConcatDatasetWithID):
        dataset = dataset.get_dataset_by_idx(_local_idx)
        _local_idx = dataset.id_to_idx(example_id)

    row = dataset.data.loc[example_id]
    return {"row": row, "index": idx}


class FallbackDatasetWrapper(Dataset):
    """A wrapper around a dataset that allows for a fallback dataset to be used when an error occurs.

    Meant to be used with a FallbackSamplerWrapper.
    """

    def __init__(self, dataset: Dataset, fallback_dataset: Dataset):
        """Initialize FallbackDatasetWrapper.

        Args:
            dataset: The primary dataset to retrieve data from.
            fallback_dataset: The fallback dataset to use when an error occurs. This
                may be the same as the primary dataset, or a different one.
        """
        self.dataset = dataset
        self.fallback_dataset = fallback_dataset

    def __getitem__(self, idxs: tuple[int, ...]) -> Any:
        """Attempt to retrieve an item from the primary dataset, falling back to additional indices if errors occur.

        If all attempts fail, raises a RuntimeError containing all encountered exceptions.

        Args:
            idxs: Tuple of indices, where the first is for the primary dataset and the rest are for fallbacks.

        Returns:
            The retrieved item from the first successful dataset.

        Raises:
            KeyboardInterrupt: If interrupted.
            StopIteration: If iteration should stop.
            RuntimeError: If all attempts fail, with a list of all exceptions encountered.
        """
        error_list = []
        example_id_list = []

        for i, idx in enumerate(idxs):
            dataset = self.dataset if i == 0 else self.fallback_dataset
            dataset_name = "Primary dataset" if i == 0 else f"Fallback {i}/{len(idxs) - 1}"

            try:
                return dataset[idx]
            except (KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                error_list.append(e)

                # Log the error
                example_id = f" ({dataset.idx_to_id(idx)})" if hasattr(dataset, "idx_to_id") else ""
                example_id_list.append(example_id)
                logger.error(f"({dataset_name}): Error ({e}) at index {idx}.{example_id}")

                # Log fallback attempt if not the last one
                if i < len(idxs) - 1:
                    logger.warning(f"({dataset_name}): Trying fallback index {idxs[i + 1]}.{example_id}")

        # All attempts failed
        logger.error(
            f"(Exceeded all {len(idxs) - 1} fallbacks. Training will crash now. Errors: {error_list} for examples: {example_id_list})"
        )
        raise RuntimeError(f"All attempts failed for indices {idxs}. See error_list for details.") from ExceptionGroup(
            "All fallback attempts failed", error_list
        )

    def __len__(self):
        """Return the length of the primary dataset."""
        return len(self.dataset)
