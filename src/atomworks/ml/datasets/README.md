# AtomWorks Datasets

This README provides an overview of how to work with datasets for training structure-based models using AtomWorks.

## Quick Start

### Simple File-Based Dataset

```python
from atomworks.ml.datasets import FileDataset
from atomworks.io import parse

# Define a simple loader function
def simple_loader(file_path):
    result = parse(file_path)
    return {"atom_array": result["assemblies"]["1"][0]}

# Create dataset from a directory
dataset = FileDataset.from_directory(
    directory="path/to/structures",
    name="my_structures",
    loader=simple_loader
)
```

### Tabular Dataset with Transforms

```python
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.crop import CropSpatialLikeAF3

# Create dataset from parquet file
dataset = PandasDataset(
    data="path/to/metadata.parquet",
    name="interfaces",
    loader=create_loader_with_query_pn_units(
        pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
    ),
    transform=Compose([
        CropSpatialLikeAF3(crop_size=384),
        # ... additional transforms
    ])
)
```

### ASE LMDB Dataset

```python
from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import create_ase_atoms_loader, create_ase_materials_loader

# OMol25/OMat24-style datasets are ASE DB-compatible LMDB files (*.aselmdb).
dataset = ASELMDBDataset.from_directory(
    directory="/path/to/omol25/train",
    name="omol25_train",
    loader=create_ase_atoms_loader(),  # Optional: adds an AtomArray for AtomWorks transforms
)

example = dataset[0]
atoms = example["atoms"]          # ASE Atoms object with flattened atoms.info metadata
atom_array = example["atom_array"]  # Biotite AtomArray created by the loader

materials_dataset = ASELMDBDataset.from_directory(
    directory="/path/to/omat24/rattled-300-subsampled",
    name="omat24_rattled",
    loader=create_ase_materials_loader(),
)
material = materials_dataset[0]
fractional_coordinates = material["fractional_coordinates"]
lattice_lengths = material["lattice_lengths"]
lattice_angles = material["lattice_angles"]
space_group = material["space_group"]
```

## Core Concepts

### The Three-Step Pipeline

All AtomWorks datasets follow a consistent three-step process:

1. **Raw Data Retrieval**: Get dataset metadata (e.g., file path, labels) by index
2. **Loading**: Convert raw data into an `AtomArray` using a loader function
3. **Transformation**: Apply ML feature engineering via Transform pipelines

```python
# What happens when you call dataset[0]:
raw_data = dataset._get_raw_data(0)           # Step 1: {"path": "/data/3ne2.cif", "label": 5}
loaded = dataset._apply_loader(raw_data)       # Step 2: {"atom_array": AtomArray, "label": 5}
transformed = dataset._apply_transform(loaded) # Step 3: {"features": Tensor, "label": Tensor}
```

### Dataset Classes

#### `MolecularDataset` (Base Class)

The abstract base class that handles loader and transform execution. All concrete datasets inherit from this.

**Key Features:**
- Executes loader functions with timing/debugging
- Applies Transform pipelines with error handling
- Saves failed examples for debugging (optional)

**Parameters:**
- `name`: Descriptive name for debugging and logging
- `loader`: Function to convert raw data to Transform-ready format
- `transform`: Transform pipeline to apply
- `save_failed_examples_to_dir`: Optional directory for debugging failures

#### `FileDataset`

For datasets where each file is one training example.

```python
# From explicit file list
dataset = FileDataset(
    file_paths=["file1.cif", "file2.cif"],
    name="my_files",
    loader=my_loader_fn
)

# From directory scan
dataset = FileDataset.from_directory(
    directory="path/to/structures",
    name="my_structures",
    max_depth=3,  # How deep to search subdirectories
    filter_fn=lambda path: path.suffix == ".cif"  # Optional filter
)
```

**ID Mapping:** Uses filename stem (without extensions) as example ID.

#### `PandasDataset`

For tabular datasets stored as DataFrames or Parquet/CSV files.

```python
dataset = PandasDataset(
    data="path/to/data.parquet",  # Or a DataFrame
    name="my_dataset",
    id_column="example_id",  # Column to use for ID-based access
    filters=[                # Optional pandas query filters
        "resolution < 2.5",
        "method == 'X-RAY_DIFFRACTION'"
    ],
    columns_to_load=["path", "label"],  # Load subset of columns (efficient for Parquet)
    loader=my_loader_fn,
    transform=my_transform_pipeline
)
```

**Filtering:** Filters are applied sequentially during initialization. Each filter logs its impact on dataset size.

**ID-Based Access:** Set an `id_column` to enable `dataset.id_to_idx()` and `idx_to_id()` methods.

#### `ASELMDBDataset`

For ASE DB-compatible LMDB shards, including FAIR Chemistry datasets such as OMol25, OMat24, and OPoly26.

```python
dataset = ASELMDBDataset(
    paths="/data/omol25/train",  # Directory scanned recursively for *.aselmdb files
    name="omol25_train",
    return_type="record",        # "record" (default), "atoms", or "row"
    readonly=True,
    readahead=False,
)
```

**Record output:** The default output is a dictionary containing `atoms` (ASE `Atoms`), `key_value_pairs`, `data`, `calculator_results`, and `extra_info`.

**AtomArray loading:** Use `create_ase_atoms_loader()` when ASE molecule records need to flow through AtomWorks transform pipelines.

**ID Mapping:** By default, IDs are generated as `<shard_id>:<ase_row_id>` for fast reversible lookup. To use an OMol/OPoly metadata field such as `sid`, pass `example_id_key="sid"` and `build_id_index=True`.

**Materials loading:** Use `create_ase_materials_loader()` for periodic materials datasets. It adds `fractional_coordinates`, `lattice_vectors`, `lattice_lengths`, `lattice_angles`, `cell_parameters`, `space_group`, and `parent_space_group`.

**Optional dependencies:** Install with `atomworks[ase]` to enable ASE LMDB support.

### Loader Functions

Loaders are functions that convert dataset-specific raw data into a standard format for Transforms.

#### Factory Pattern

Use loader factory functions to create loaders with common patterns:

```python
from atomworks.ml.datasets.loaders import create_base_loader, create_loader_with_query_pn_units

# Basic loader for simple datasets
loader = create_base_loader(
    example_id_colname="example_id",
    path_colname="structure_path",
    assembly_id_colname="assembly_id",
    base_path="/data/pdb",           # Prepended to paths
    extension=".cif.gz",              # Added if missing
    sharding_pattern="/1:2/",         # e.g., "3ne2" → "3n/3ne2.cif.gz"
    parser_args={"remove_waters": True}
)

# Loader with query pn_units (for cropping)
loader = create_loader_with_query_pn_units(
    pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"],  # For interfaces
    base_path="/data/pdb",
    extension=".cif.gz"
)
```

**Output Format:** Loaders return dictionaries with standardized keys:
- `example_id`: Unique identifier
- `path`: Full path to structure file
- `assembly_id`: Assembly ID
- `atom_array`: First model as AtomArray
- `atom_array_stack`: All models
- `chain_info`, `ligand_info`, `metadata`: From CIF parser
- `extra_info`: Additional metadata from DataFrame
- `query_pn_unit_iids`: (Optional) For cropping transforms

#### Custom Loaders

You can write custom loaders as simple functions:

```python
def my_custom_loader(row: pd.Series) -> dict:
    """Load structure and add custom metadata."""
    result = parse(row["path"])

    return {
        "example_id": row["my_id"],
        "atom_array": result["assemblies"]["1"][0],
        "custom_label": row["label"],
        "extra_info": {"dataset": "custom"}
    }

dataset = PandasDataset(data=df, name="custom", loader=my_custom_loader)
```

## Hierarchical Datasets

For complex training schemes, combine multiple datasets with `ConcatDatasetWithID`:

```python
from atomworks.ml.datasets import ConcatDatasetWithID

# Create individual datasets
chains_dataset = PandasDataset(data="chains.parquet", name="chains", ...)
interfaces_dataset = PandasDataset(data="interfaces.parquet", name="interfaces", ...)
distillation_dataset = PandasDataset(data="distillation.parquet", name="distillation", ...)

# Combine hierarchically
pdb_data = ConcatDatasetWithID([chains_dataset, interfaces_dataset])
all_data = ConcatDatasetWithID([pdb_data, distillation_dataset])
```

**Hierarchical Structure Example:**

```plaintext
                 ConcatDatasetWithID
                        |
        ---------------------------------
        |                               |
 FB Distillation                ConcatDatasetWithID
(PandasDataset)                        |
                                       |
                                -----------------------
                                |                     |
                        Interfaces Dataset    PN Units Dataset
                        (PandasDataset)      (PandasDataset)
```

**ID-Based Access:** `ConcatDatasetWithID` provides `id_to_idx()`, `idx_to_id()`, and `__contains__()` methods that work across all nested datasets.

## Error Handling and Fallbacks

### Failed Example Debugging

```python
dataset = PandasDataset(
    data="data.parquet",
    name="debug_dataset",
    save_failed_examples_to_dir="/tmp/failed_examples"
)
```

When a Transform fails, AtomWorks saves:
- Example ID and error message
- RNG state for reproducibility
- Timing information

### Fallback Dataset Wrapper

For robust training, use `FallbackDatasetWrapper` with `FallbackSamplerWrapper`:

```python
from atomworks.ml.datasets import FallbackDatasetWrapper

# Wrap dataset to enable fallback on errors
dataset_with_fallback = FallbackDatasetWrapper(
    dataset=my_dataset,
    fallback_dataset=my_dataset  # Can be same or different dataset
)
```

The wrapper attempts to load examples from fallback indices when errors occur, preventing training crashes on bad data.

## Sampling Strategies

AtomWorks provides sophisticated sampling utilities in `atomworks.ml.samplers`:

```python
from atomworks.ml.samplers import calculate_weights_for_pdb_dataset_df
import torch
from torch.utils.data import WeightedRandomSampler

# Calculate AF3-style weights (inverse cluster size + composition)
weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=df,
    alphas={"a_prot": 1.0, "a_nuc": 1.0, "a_ligand": 2.0, "a_loi": 5.0},
    beta=1.0
)

# Create weighted sampler
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
```

See `atomworks.ml.samplers` for additional weighting strategies and distributed samplers.
