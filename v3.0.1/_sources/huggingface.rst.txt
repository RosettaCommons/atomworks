Hugging Face dataset repositories
=================================

Install ``atomworks[ml,hf]`` to read an indexed blob dataset from a private or
public HF dataset repository. Authenticate with ``hf auth login`` or ``HF_TOKEN``
for private data. Pin every URL to the same full commit SHA:

.. code-block:: python

   from atomworks.ml.utils.blob_store import BlobIndex, BlobStore
   from atomworks.ml.utils.io import read_parquet_with_metadata

   root = "hf://datasets/org/repo@<40-character-sha>/protein"
   examples = read_parquet_with_metadata(f"{root}/examples.parquet")
   index = BlobIndex(f"{root}/blob/index.parquet", id_column="path")
   store = BlobStore(f"{root}/blob/data")
   cif = store.get_bytes(*index.lookup(examples.iloc[0]["path"]))

Set ``id_column`` to the identifier used by your index. Sampling tables and
indices use the HF file cache; blob records use exact ranged reads and are not
cached. For repeated training, download the required shards once and use local
paths. The existing S3 and local readers are unchanged.
