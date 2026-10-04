Public S3 datasets
==================

Install the ``ml`` and ``s3`` extras to read metadata and structure records from
an S3-compatible object store. Metadata is downloaded once, the blob index is
cached locally, and each structure is fetched with a byte-range request.
Use an immutable dataset prefix so cached records retain their identity.

Pass the same connection configuration to the dataset's ``load_kwargs`` and to
its structure loader. ``anonymous=True`` reads public objects without consulting
AWS credentials. Signed access remains the default. The endpoint defaults to
``AWS_ENDPOINT_URL`` when omitted; addressing may be ``virtual``, ``path``, or
``auto``. Blob stores support local paths and ``s3://`` URLs. Other metadata
sources retain the underlying pandas/PyArrow readers' behavior.

.. code-block:: python

   from atomworks.io.config import ParseConfig
   from atomworks.ml.datasets import PandasDataset
   from atomworks.ml.datasets.loaders import create_structure_loader
   from atomworks.ml.utils.io import S3ReadConfig

   root = (
       "s3://rfd4-proteina-public-e04a/train_datasets/streaming-latest/"
       "synthetic/2026_09_09_tcr_af2"
   )
   connection = S3ReadConfig(endpoint_url="https://cwobject.com", anonymous=True)
   parser = ParseConfig.from_preset(
       "annotations_only", altloc="first",
       load_standard_annotations=True, return_atom_array_plus=True,
   )
   loader = create_structure_loader(
       storage="blob", blob_dir=f"{root}/blob", s3_config=connection,
       record_id_colname="path", parser_args=parser,
   )
   dataset = PandasDataset(
       name="public-tcr", data=f"{root}/examples.parquet",
       loader=loader, load_kwargs={"s3_config": connection},
   )
   example = dataset[0]
   print(example["example_id"], len(example["atom_array"]))

The sampled ``example_id`` and stored record ID can differ: several interfaces
may refer to one CIF. This dataset uses ``path`` as its blob index key; other
datasets may use ``example_id``. Keep that choice consistent with the index schema.
Parser settings and training transforms remain specific to each dataset.

For YAML-driven applications, ``s3_config`` also accepts a mapping with
``endpoint_url``, ``anonymous``, and ``addressing_style`` keys. Share that mapping
between ``loader.s3_config`` and ``load_kwargs.s3_config``.

Tiny local subset
-----------------

Download :download:`public_s3_example.py <public_s3_example.py>` and run:

.. code-block:: bash

   python public_s3_example.py --count 2 --output-dir tiny-tcr

This writes an ``examples.parquet`` plus the selected CIF files, retaining the
source record IDs and all example metadata. To use the files locally, supply
``base_path="tiny-tcr"`` to a filesystem structure loader and retain the parser
configuration above. This is a data-loading smoke example; downstream models
must also select their task transforms, data split, and fine-tuning settings.

.. literalinclude:: public_s3_example.py
   :language: python
   :start-at: PREFIX =
