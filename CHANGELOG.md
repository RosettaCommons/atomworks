# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0]

### Compatibility

- Require Python 3.12 or newer, NumPy 2.1 or newer, and Biotite 2.0.0.
- Move command-line dependencies to the `scripts` extra; install `atomworks[scripts]` to use `aw`.
- Remove the retired RF2AA pipeline and feature adapters.
- `atom_array_from_file` now uses minimal parsing, including first-conformer selection and normalized identifiers. `to_cif_file` is an alias with the `atom_array_to_file` signature; `to_cif_buffer` and `to_cif_string` are removed.

### Added and changed

- Add `atom_array_from_file` and `atom_array_to_file` for basic structure I/O, including standard annotations and catcif input.
- Reduce core dependencies and group optional ML, script, and documentation dependencies into extras.
- Add explicit policies for unknown CCD charges and preserve formal charges through ASE conversions.
- Improve annotated CIF loading and assembly performance, and update link chemistry, entity annotation, and Biotite compatibility.

See the [migration guide](docs/migration/migration-long.md) for API and behavior changes.

## [2.0.0] - 2025-11-29

### Performance

- **Parser 2-3x faster**: Significant optimizations to structure parsing, especially for symmetric assemblies
- **Cache loading 3-5x faster**: Improved pickle/gzip cache handling with 2-level directory sharding for better filesystem performance
- **Vectorized annotations**: `add_pn_unit_iid_annotation()` now uses boolean masks instead of expensive subarray operations (10-100x speedup on symmetric assemblies)

### Breaking Changes

#### Dataset Module Restructuring

The dataset module has been restructured to align with TorchVision/TorchAudio and HuggingFace conventions, using a dataset/loader pattern:

- **Removed `dataset.dataset` nesting**: Datasets are now flat; access data directly from the dataset object
- **MetadataRowParser deprecated**: The `StructuralDatasetWrapper` + `dataset_parser` pattern is replaced with a `loader` parameter directly on datasets (backwards-compatible but deprecated)

**Migration example:**
```python
# Old (deprecated)
from atomworks.ml.datasets import StructuralDatasetWrapper, PandasDataset
from atomworks.ml.datasets.parsers import PNUnitsDFParser

dataset = StructuralDatasetWrapper(
    dataset=PandasDataset(data="df.parquet"),
    dataset_parser=PNUnitsDFParser(...)
)

# New
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_base_loader

dataset = PandasDataset(
    data="df.parquet",
    loader=create_base_loader(
        example_id_colname="example_id",
        path_colname="path",
    )
)
```

#### Parser Changes

- **CCD mirror path validation**: `ccd_mirror_path` now raises `FileNotFoundError` if the path doesn't exist. Pass `None` explicitly to use Biotite's bundled CCD
- **`build_assembly="_spoof"` removed**: Use `"all"` instead (raises deprecation warning)
- **`convert_mse_to_met` default changed**: Now `True` by default (was `False`)
- **`STANDARD_PARSER_ARGS` renamed**: Was `DEFAULT_PARSE_KWARGS`; now uses tuples instead of lists for hashability

#### Environment Changes

- **Removed automatic `.env` loading**: `dotenv` is no longer auto-loaded on import. Call `load_dotenv()` explicitly if needed:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```

#### Removed Exports

- `monkey_patch_atomarray` removed from top-level exports. Use `from atomworks.biotite_patch import monkey_patch_biotite` instead

### Added

#### New Modules

- `atomworks.ml.conditions` - Unified conditioning management for model training
- `atomworks.ml.preprocessing.msa` - MSA preprocessing (organize, filter, generate)
- `atomworks.ml.executables` - External executable management (hbplus, hhfilter, mmseqs2, x3dna)
- `atomworks.ml.transforms.design_task` - Design task transforms
- `atomworks.ml.transforms.mask_generator` - Mask generation for training
- `atomworks.ml.utils.condition` - Condition utilities
- `atomworks.io.utils.compression` - Compression utilities (zstd support)

#### New Dataset Classes

- `FileDataset` - Each file is one example (extracted from old monolithic datasets.py)
- `PandasDataset` - DataFrame-backed dataset with loader support

#### New Loader Functions

- `create_base_loader()` - Standard CIF loading
- `create_loader_with_query_pn_units()` - Loading with PN unit queries
- `create_loader_with_interfaces_and_pn_units_to_score()` - Interface scoring loader

#### New Constants

- `PROTEIN_BACKBONE_ATOM_NAMES` - Backbone atoms including OXT
- `RNA_BACKBONE_ATOM_NAMES` - Sugar-phosphate + 2' hydroxyl atoms
- `DNA_BACKBONE_ATOM_NAMES` - Sugar-phosphate atoms
- `NUCLEIC_ACID_BACKBONE_ATOM_NAMES` - Union of RNA+DNA backbones
- `MASKED` - Token code for masked positions
- `MSAFileExtension` enum - Supported MSA file formats
- Expanded `METAL_ELEMENTS` - Now includes lanthanides and actinides

#### New Features

- `AtomArrayPlus` support in parser - Extended atom array with additional metadata
- Spawn multiprocessing support for data loading
- zstd compression support for MSA files
- Atom37 encoding with atomization support
- JSON-level atom selection for bonds argument

### Fixed

- Residue starts bug with dependent functions
- SASA calculation for empty amino acid arrays
- Null handling in A3M files
- Design tasks with zero frequency now handled gracefully instead of erroring
- Non-uniform shard sizes handling
- Pickling during data loading with spawn multiprocessing

### Changed

- Loaders module restructured from `loaders.py` to `loaders/` subpackage (imports still work via `__init__.py`)
- Parser cache structure now uses 2-level sharding (old caches automatically regenerated)

### Deprecated

- `atomworks.ml.datasets.parsers` module - Use loaders instead
- `StructuralDatasetWrapper` - Use loader parameter on datasets directly

## [1.0.3] - 2025-10-01

Initial public release.

[2.0.0]: https://github.com/RosettaCommons/atomworks/compare/v1.0.3...v2.0.0
[1.0.3]: https://github.com/RosettaCommons/atomworks/releases/tag/v1.0.3
