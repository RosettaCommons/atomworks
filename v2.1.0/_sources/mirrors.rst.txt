Data Mirrors
============

AtomWorks uses local mirrors of the PDB and CCD databases for parsing structures and training models. This page explains how to set up these mirrors.

Setting Up Mirrors
------------------

PDB Mirror (~100 GB)
^^^^^^^^^^^^^^^^^^^^

The PDB mirror contains mmCIF structure files. We use mmCIF as the recommended format.

.. code-block:: bash

   # Download the entire PDB
   atomworks pdb sync /path/to/pdb_mirror

   # Or download specific PDB IDs only
   atomworks pdb sync /path/to/pdb_mirror --pdb-id 1A0I --pdb-id 7XYZ

   # Or from a file of IDs (one per line)
   atomworks pdb sync /path/to/pdb_mirror --pdb-ids-file /path/to/ids.txt

This creates a carbon-copy of the PDB with the same sharding pattern as RCSB (e.g., ``1a2b`` → ``/path/to/pdb_mirror/a2/1a2b.cif.gz``).

CCD Mirror (~2 GB)
^^^^^^^^^^^^^^^^^^

The CCD (Chemical Component Dictionary) mirror contains ligand definitions used for parsing non-polymer entities.

.. code-block:: bash

   atomworks ccd sync /path/to/ccd_mirror

If no CCD mirror is provided, AtomWorks falls back to Biotite's internal CCD. You can also add custom ligand definitions by placing CIF files in the mirror following the CCD pattern (e.g., ``/path/to/ccd_mirror/M/MYLIGAND/MYLIGAND.cif``).

Configuring Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once mirrors are created, configure the paths in a ``.env`` file in your repository root:

.. code-block:: bash

   PDB_MIRROR_PATH=/path/to/pdb_mirror
   CCD_MIRROR_PATH=/path/to/ccd_mirror

You can copy ``.env.sample`` as a starting point. See :doc:`installation` for more details on environment setup.

----

Training on the PDB
===================

Step 1 — Mirror the PDB (mmCIFs)
--------------------------------

To train on the PDB, you need access to the structure files. Follow the instructions above to set up your PDB mirror.
  
Step 2 — Get PDB metadata (PN units and interfaces)
---------------------------------------------------

To calculate sampling probabilities and filter examples for splits, we pre-process the PDB with metadata for each PDB entry. 
To save you the work, we provide pre-computed metadata (dated July 15/2025) for downloading:

.. code-block:: bash

  atomworks setup metadata /path/to/metadata  # This will download the metadata (as .tar.gz) and extract it to the specified directory.


This produces parquet files at:

* ``/path/to/metadata/pn_units_df.parquet`` — Contains metadata for each **PN unit** in the PDB. The term **pn unit** is shorthand for ``polymer XOR non-polymer unit`` and behaves for almost all purposes like the ``chain`` in a PDB file. The only difference is that a ligand composed of multiple covalently bonded ligands is considered a single PN unit (whilst it would be multiple chains in a PDB file). Effectively this ``.parquet`` is a large table of all individual chains, ligands, etc (to be precise, it has one entry per  pn unit) in the PDB that includes helpful metadata for filtering and sampling.
* ``/path/to/metadata/interfaces_df.parquet`` — Contains metadata for each interface in the PDB. This ``.parquet`` is a large table of all binary interfaces in the PDB. It lists each interface as (pn_unit_1, pn_unit_2) pairs and includes helpful metadata for filtering and sampling.

  Alternatively, you can generate fresher metadata yourself (scripts will be uploaded in the coming weeks).

Step 3 — Configure an AF3-style dataset (example: train only on D-polypeptides)
-------------------------------------------------------------------------------

Next we need to use the metadata to configure a dataset that we would like to sample from. This includes e.g. training cut-off, filters, transforms to apply, etc.
Here's a simple example that:

* Filters to D-polypeptide and L-polypeptide chains only (`POLYPEPTIDE_D` and `POLYPEPTIDE_L` -- to include additional chain types, replace the lists with the appropriate IDs (see [mapping](./src/atomworks/enums.py#L31-L45) in comments).
* Excludes ligands in the AF3 list of excluded ligands, available at [`atomworks.io.constants.AF3_EXCLUDED_LIGANDS_REGEX`](./src/atomworks/constants.py#L350).

.. code-block:: yaml

  # NOTE: The below is a hydra config and the _target_ fields are the hydra syntax for instantiating a class.
  #  You can use this without hyrda, but will then instead need to provide the corresponding arguments for the
  #  _target_ objects directly.

   Chain type ids used below (from atomworks.enums.ChainType):
  # 0=CyclicPseudoPeptide, 1=OtherPolymer, 2=PeptideNucleicAcid,
  # 3=DNA, 4=DNA_RNA_HYBRID, 5=POLYPEPTIDE_D, 6=POLYPEPTIDE_L, 7=RNA,
  # 8=NON_POLYMER, 9=WATER, 10=BRANCHED, 11=MACROLIDE

  af3_pdb_dataset:
    _target_: atomworks.ml.datasets.datasets.ConcatDatasetWithID
    datasets:
      # Single PN units
      - _target_: atomworks.ml.datasets.datasets.StructuralDatasetWrapper
        dataset_parser:
          _target_: atomworks.ml.datasets.parsers.PNUnitsDFParser
        transform:
          _target_: atomworks.ml.pipelines.af3.build_af3_transform_pipeline
          is_inference: false
          n_recycles: 5  # This means that we will subsample 5 random sets from the MSA for each example.
          crop_size: 256
          crop_contiguous_probability: 0.3333333333333333
          crop_spatial_probability: 0.6666666666666666
          diffusion_batch_size: 32
          # Optional templates (if available)
          template_lookup_path: ${paths.shared}/template_lookup.csv
          template_base_dir: ${paths.shared}/template
          # Optional MSAs (see Step 4)
          # protein_msa_dirs:
          #   - { dir: /path/to/msa, extension: .a3m.gz, directory_depth: 2 }
          # rna_msa_dirs:
          #   - { dir: /path/to/msa, extension: .afa, directory_depth: 0 }
        dataset:
          _target_: atomworks.ml.datasets.datasets.PandasDataset
          name: pn_units
          id_column: example_id
          data: /path/to/metadata/pn_units_df.parquet
          filters:
            - "deposition_date < '2022-01-01'"
            - "resolution < 5.0 and ~method.str.contains('NMR')"
            - "num_polymer_pn_units <= 20"
            - "cluster.notnull()"
            - "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            # Train only on D-polypeptides:
            - "q_pn_unit_type in [5, 6]"  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            # Exclude ligands from AF3 excluded set:
            - "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
          columns_to_load: null
        save_failed_examples_to_dir: null

      # Binary interfaces
      - _target_: atomworks.ml.datasets.datasets.StructuralDatasetWrapper
        dataset_parser:
          _target_: atomworks.ml.datasets.parsers.InterfacesDFParser
        transform:
          _target_: atomworks.ml.pipelines.af3.build_af3_transform_pipeline
          is_inference: false
          n_recycles: 5
          crop_size: 256
          crop_spatial_probability: 1.0
          crop_contiguous_probability: 0.0
          diffusion_batch_size: 32
          template_lookup_path: ${paths.shared}/template_lookup.csv
          template_base_dir: ${paths.shared}/template
          # Optional MSAs (see Step 4)
          # protein_msa_dirs:
          #   - { dir: /path/to/msa, extension: .a3m.gz, directory_depth: 2 }
          # rna_msa_dirs:
          #   - { dir: /path/to/msa, extension: .afa, directory_depth: 0 }
        dataset:
          _target_: atomworks.ml.datasets.datasets.PandasDataset
          name: interfaces
          id_column: example_id
          data: /path/to/metadata/interfaces_df.parquet
          filters:
            - "deposition_date < '2022-01-01'"
            - "resolution < 5.0 and ~method.str.contains('NMR')"
            - "num_polymer_pn_units <= 20"
            - "cluster.notnull()"
            - "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            # Train only on D-polypeptide interfaces:
            - "pn_unit_1_type in [5, 6]"  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            - "pn_unit_2_type in [5, 6]"  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            - "~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
            - "~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
          columns_to_load: null
        cif_parser_args:
          cache_dir: null
        save_failed_examples_to_dir: null

Step 4 — Train a model
----------------------

You now have a full fledged dataset that you can use to train models on! If you want to just try this out without having to download the whole PDB and the metdatada, you can instead run our tests which have a mini-mockup of the pipeline with real pdb files, metadata, distillation data, templates and MSAs for the example of AF3. You can download all this relevant metadata via the atomworks CLI:

.. note:: 

  Make sure you are in the AtomWorks root directory when you run the following command, otherwise a new tests/data folder will be created in your current working directory.

.. code-block::bash
  atomworks setup tests  # This will download the test pack to `tests/data` and unpack it there (~500 MB). 

You will now have a mini PDB at `tests/data/pdb` and a mini custom CCD at `tests/data/ccd`. The distillation and metadata are in `data/ml/af2_distillation`, `data/ml/pdb_pn_units` and `data/ml/pdb_interfaces`. A dataset that uses all of these is [for example here](./tests/ml/conftest.py#L300).

To run the tests for the various datasets, you can run the following command:

.. code-block:: bash
  
  # Make sure you have the correct environment activated, and set your paths correctly in the .env file / shell environment variables (see points above)
  pytest tests/ml/pipelines/test_data_loading_pipelines.py

