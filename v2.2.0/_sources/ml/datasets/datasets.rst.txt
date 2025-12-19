Dataset Architecture
====================

AtomWorks provides a modern, composable dataset architecture that separates data loading, processing, and transformation concerns. This approach replaces the legacy parser-based system with functional loaders and transform pipelines.

.. warning::
   The metadata parser system (``atomworks.ml.datasets.parsers``) is **deprecated** and will be removed in a future version. 
   Use the new loader-based approach with ``FileDataset`` and ``PandasDataset`` instead.

Modern Dataset Architecture
---------------------------

The current AtomWorks dataset system consists of three main components:

1. **Datasets**: Container classes that manage data access and indexing
2. **Loaders**: Functions that process raw data into transform-ready format  
3. **Transforms**: Pipelines that convert loaded data into model inputs

Dataset Classes
---------------

.. automodule:: atomworks.ml.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Functional Loaders
------------------

Loaders are functions that process raw dataset output (e.g., pandas Series) into a Transform-ready format.
They replace the legacy parser classes with a more flexible, functional approach.

.. automodule:: atomworks.ml.datasets.loaders
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

**File-based datasets** (replacing simple file parsers):

.. code-block:: python

   from atomworks.ml.datasets import FileDataset
   from atomworks.io import parse
   
   def simple_loading_fn(raw_data) -> dict:
       """Simple loading function that parses structural data."""
       parse_output = parse(raw_data)
       return {"atom_array": parse_output["assemblies"]["1"][0]}
   
   dataset = FileDataset.from_directory(
       directory="/path/to/structures", 
       name="my_dataset", 
       loader=simple_loading_fn
   )

**Tabular datasets** (replacing metadata parsers):

.. code-block:: python

   from atomworks.ml.datasets import PandasDataset
   from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units

   dataset = PandasDataset(
       data="metadata.parquet",
       name="interfaces_dataset",
       loader=create_loader_with_query_pn_units(
           pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
       )
   )

**Custom loaders** for specialized use cases:

.. code-block:: python

   def custom_loader(row: pd.Series) -> dict:
       """Custom loader with specific processing logic."""
       # Load structure
       structure_path = Path(row["path"])
       parse_output = parse(structure_path)
       
       # Extract specific metadata
       metadata = {
           "resolution": row.get("resolution", None),
           "method": row.get("method", "unknown"),
           "custom_field": row.get("custom_field", "default_value")
       }
       
       return {
           "atom_array": parse_output["assemblies"]["1"][0],
           "extra_info": metadata,
           "example_id": row["example_id"]
       }
   
   dataset = PandasDataset(
       data=my_dataframe,
       name="custom_dataset", 
       loader=custom_loader
   )

Common Loader Patterns
~~~~~~~~~~~~~~~~~~~~~~

**Base loader** for standard structure loading:

.. code-block:: python

   from atomworks.ml.datasets.loaders import create_base_loader

   loader = create_base_loader(
       example_id_colname="example_id",
       path_colname="path",
       assembly_id_colname="assembly_id",
       base_path="/data/structures",
       extension=".cif"
   )

**Interface loader** for protein-protein interfaces:

.. code-block:: python

   from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units

   loader = create_loader_with_query_pn_units(
       pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"],
       base_path="/data/pdb",
       extension=".cif.gz"
   )

**Validation loader** with scoring targets:

.. code-block:: python

   from atomworks.ml.datasets.loaders import create_loader_with_interfaces_and_pn_units_to_score

   loader = create_loader_with_interfaces_and_pn_units_to_score(
       interfaces_to_score_colname="interfaces_to_score",
       pn_units_to_score_colname="pn_units_to_score"
   )

Integration with Transform Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loaders work seamlessly with AtomWorks transform pipelines. The loader output becomes the input to the transform pipeline:

.. code-block:: python

   from atomworks.ml.transforms.base import Compose
   from atomworks.ml.transforms.crop import CropSpatialLikeAF3
   from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
   
   # Create a transform pipeline
   transform_pipeline = Compose([
       AddGlobalAtomIdAnnotation(),
       CropSpatialLikeAF3(crop_size=256),
   ])
   
   # Create dataset with both loader and transforms
   dataset = PandasDataset(
       data="metadata.parquet",
       name="my_dataset",
       loader=loader_with_query_pn_units(
           pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"]
       ),
       transform=transform_pipeline
   )
   
   # Access processed data
   example = dataset[0]  # Returns transformed data ready for model input

Data Flow
~~~~~~~~~

The complete data flow in the new architecture is:

1. **Raw Data**: File paths or DataFrame rows
2. **Loader**: Processes raw data into standardized format with ``AtomArray``
3. **Transform Pipeline**: Converts loaded data into model-ready tensors
4. **Model Input**: Final processed data ready for training/inference

This separation allows for:
- **Reusable loaders** across different datasets
- **Composable transforms** that can be mixed and matched
- **Easy testing** of individual components
- **Clear debugging** when issues arise
