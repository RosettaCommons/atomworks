"""File-based dataset implementation."""

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

from atomworks.ml.utils.io import scan_directory

from .base import MolecularDataset


def _always_true(x: PathLike) -> bool:
    return True


class FileDataset(MolecularDataset):
    """Dataset that loads molecular data from individual files.

    Each file represents one example in the dataset. If creating a dataset from a
    directory, use the :meth:`from_directory` class method instead of the default
    constructor.
    """

    def __init__(
        self,
        *,
        file_paths: list[str | PathLike],
        name: str,
        filter_fn: Callable[[PathLike], bool] | None = None,
        **kwargs: Any,
    ):
        """Initialize FileDataset.

        Args:
            file_paths: List of file paths for the dataset. Each file represents
                one example.
            name: Descriptive name for this dataset. Used for debugging and some
                downstream functions when using nested datasets.
            filter_fn: Optional function to filter file paths. Should return True
                for files to include.
            **kwargs: Additional arguments passed to :class:`MolecularDataset`.

        Examples:
            Create from explicit file list:
                >>> files = ["/path/to/file1.cif", "/path/to/file2.cif"]
                >>> dataset = FileDataset(file_paths=files, name="my_dataset")
        """
        super().__init__(name=name, **kwargs)

        self.filter_fn = filter_fn or _always_true

        # Convert to Path objects and filter
        file_paths = [Path(path) for path in file_paths if self.filter_fn(path)]
        if not file_paths:
            raise ValueError("No files found after applying filters")
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("File paths must be unique")

        # Sort for consistent id<>idx mapping
        file_paths.sort()
        self.file_paths = file_paths

        # Create ID mapping
        self.id_to_idx_map = {self._get_example_id(i): i for i, _ in enumerate(self.file_paths)}

        # Verify that all example IDs are unique
        if len(self.id_to_idx_map) != len(self.file_paths):
            raise ValueError("Example IDs must be unique. Found duplicate example IDs.")

    @classmethod
    def from_directory(
        cls,
        *,
        directory: PathLike,
        name: str,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> "FileDataset":
        """Create a FileDataset by scanning a directory for files.

        Args:
            directory: Path to directory to scan for files.
            name: Descriptive name for this dataset.
            max_depth: Maximum depth to scan for files in subdirectories.
            **kwargs: Additional arguments passed to :class:`FileDataset`.

        Returns:
            FileDataset instance with files discovered from the directory.

        Example:
            Create from directory:
                >>> dataset = FileDataset.from_directory(directory="/path/to/files", name="my_dataset", max_depth=2)
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        if not dir_path.is_dir():
            raise ValueError(f"Path {directory} is not a directory.")

        file_paths = scan_directory(dir_path=dir_path, max_depth=max_depth)
        return cls(file_paths=file_paths, name=name, **kwargs)

    @classmethod
    def from_file_list(
        cls,
        *,
        file_paths: list[str | PathLike],
        name: str,
        **kwargs: Any,
    ) -> "FileDataset":
        """Create a FileDataset from an explicit list of file paths.

        This is an alias for the main constructor for clarity and consistency
        with :meth:`from_directory`.

        Args:
            file_paths: List of file paths for the dataset. Each file represents one example.
            name: Descriptive name for this dataset.
            **kwargs: Additional arguments passed to :class:`FileDataset`.

        Returns:
            FileDataset instance with the provided file paths.
        """
        return cls(file_paths=file_paths, name=name, **kwargs)

    def __len__(self) -> int:
        """Return the number of files in the dataset."""
        return len(self.file_paths)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self.id_to_idx_map

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to index(es)."""
        if isinstance(example_id, list):
            return [self.id_to_idx_map[id] for id in example_id]
        return self.id_to_idx_map[example_id]

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert index(es) to example ID(s)."""
        if isinstance(idx, list):
            return [self._get_example_id(i) for i in idx]
        return self._get_example_id(idx)

    def __getitem__(self, idx: int) -> Any:
        """Load and transform an example by file index.

        Args:
            idx: The index of the file to load.

        Returns:
            Transformed data from the file.
        """
        file_path = str(self.file_paths[idx])
        example_id = self._get_example_id(idx)
        data = self._apply_loader(file_path)
        return self._apply_transform(data, example_id=example_id, idx=idx)

    def _get_example_id(self, idx: int) -> str:
        """Get example ID from index - returns filename stem without extensions.

        Args:
            idx: The index of the file.

        Returns:
            Filename stem without any extensions.
        """
        file_path = self.file_paths[idx]
        filename = Path(file_path).stem
        # If filename has multiple extensions (e.g., .cif.gz), remove them all
        while "." in filename:
            filename = Path(filename).stem
        return filename
