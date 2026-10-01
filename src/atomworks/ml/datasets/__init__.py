import logging

# HACK: Re-export all dataset classes for backward compatibility
# In the future, we will just import from their respective modules (e.g., from .pandas_dataset import PandasDataset)
from .base import ExampleIDProtocol, MolecularDataset
from .concat_dataset import ConcatDatasetWithID, FallbackDatasetWrapper, get_row_and_index_by_example_id
from .file_dataset import FileDataset
from .metadata import ArrowMetadataIndex, MetadataIndex, MetadataIndexProtocol, SequentialMetadataIndex
from .pandas_dataset import PandasDataset

logger = logging.getLogger("datasets")

__all__ = [
    "ArrowMetadataIndex",
    "ConcatDatasetWithID",
    "ExampleIDProtocol",
    "FallbackDatasetWrapper",
    "FileDataset",
    "MetadataIndex",
    "MetadataIndexProtocol",
    "MolecularDataset",
    "PandasDataset",
    "SequentialMetadataIndex",
    "get_row_and_index_by_example_id",
    "logger",
]
