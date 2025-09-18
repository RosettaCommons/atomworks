[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://baker-laboratory.github.io/atomworks-dev/latest/index.html)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<div align="center">
  <img src="docs/_static/atomworks_logo_color.svg" width="450" alt="atomworks logo">
</div>

**atomworks** is an open-source platform that maximizes research velocity for biomolecular modeling tasks. Much like how [Torchvision](https://docs.pytorch.org/vision/stable/index.html) enables rapid prototyping within the vision domain, and [Torchaudio](https://docs.pytorch.org/audio/main/) within the audio domain, AtomWorks aims to accelerate development and experimentation within biomolecular modeling.

> **⚠️ Notice:** We are currently finalizing some cleanup work within our repositories. Please expect the APIs (e.g., function and class names, inputs and outputs) to stabilize within the next one week. Thank you for your patience!

If you're looking for the models themselves (e.g., RF3, MPNN) that integrate with AtomWorks rather than the underlying framework, check out [ModelForge](https://github.com/RosettaCommons/modelforge)

> **💡 Note:** Not sure where to start? We've made some [examples in the AtomWorks documentation](https://rosettacommons.github.io/atomworks/latest/auto_examples/index.html) that work through several helpful scenarios; a full tutorial is under construction!

AtomWorks is composed of two symbiotic libraries:

- `atomworks.io`: A universal Python toolkit for parsing, cleaning, manipulating, and converting biological data (structures, sequences, small molecules). Built on the [biotite](https://www.biotite-python.org/) API, it seamlessly loads and exports between standard formats like mmCIF, PDB, FASTA, SMILES, MOL, and more. Broadly useful for anyone who works with structural data for biomolecules.
- `atomworks.ml`: Advanced dataset featurization and sampling for deep learning workflows that uses `atomworks.io` as its structural backbone. We provide a comprehensive, pre-built and well-tested set of `Transforms` for common tasks that can be easily composed into full deep-learning pipelines; users may also create their own `Transforms` for custom operations.

For more detail on the motivation for and applications of AtomWorks, please see the [preprint](https://doi.org/10.1101/2025.08.14.670328). 

AtomWorks is built atop [biotite](https://www.biotite-python.org/): We are grateful to the Biotite developers for maintaining such a high-quality and flexible toolkit, and hope that our package will prove a helpful addition to the broader `biotite` community.

---

## atomworks.io

> *A general-purpose Python toolkit for cleaning, standardizing, and manipulating with biomolecular structure files - built atop [biotite](https://www.biotite-python.org/):

**atomworks.io** lets you:

- Parse, convert, and clean any common biological file (structure or sequence). For example, identifying and removing leaving groups, correcting bond order after nucleophilic addition, fixing charges, parsing covalent geometries, and appropriate treatment of structures with multiple occupancies and ligands at symmetry centers
- Transform all data to a consistent `AtomArray` representation for further analysis or machine learning applications, regardless of initial source
- Model missing atoms (those implied by the sequence but not represented in the coordinates) and initialize entity- and instance-level annotations (see the [glossary]() for more detail on our composable naming conventions)

We have found `atomworks.io` to be generally useful to a broad bioinformatics and protein design audience; in many cases, `atomworks.io` can replace bespoke scripts and manual curation, enabling researchers to spend more time testing hypothesis and less time juggling dozens of tools and dependencies.

---

## atomworks.ml

> *Modular, component-based library for dataset featurization within biomolecular deep learning workflows*

**atomworks.ml** provides:

- A library of pre-built, well-tested `Transforms` that can be slotted into novel pipelines
- An extensible framework, integrated with `atomworks.io`, to write `Transforms` for arbitrary use cases
- Pre-built datasets and samplers suitable for most model training scenarios

Within the AtomWorks paradigm, the output of each `Transform` is not an opaque dictionary with model-specific tensors but instead an updated version of our atom-level structural representation (Biotite's `AtomArray`). Operations within – and between – pipelines thus maintain a common vocabulary of inputs and outputs.

We have found that `atomworks.ml` **dramatically** reduces the overhead of starting, and completing, many ML projects; research topics that once took months now achieve signal within weeks if not days, accelerating the pace of innovation.

---

## Installation

```shell
pip install atomworks # base installation version without torch (for only atomworks.io)
pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
pip install "atomworks[dev]" # with development dependencies
pip install "atomworks[ml,dev]" # with all dependencies
```

If you are using [uv](https://docs.astral.sh/uv/reference/policies/versioning/) for package management, you can install atomworks with:

```shell
uv pip install "atomworks[ml,openbabel,dev]"
```

For more advanced setup options (including how to run workflows via apptainers) see the [full documentation](https://baker-laboratory.github.io/atomworks-dev/latest).

---

## Getting started

### 1. When to use `atomworks.io` vs `atomworks.ml`?

- Use `atomworks.io` when you:
  - Need to parse/clean/convert between biological file formats (mmCIF, PDB, FASTA, etc.)
  - Want a unified structural representation to plug into any downstream analysis or modeling
  - Need structural operations like adding missing atoms, filtering ligands/solvents, or assembly generation

- Use `atomworks.ml` when you:
  - Need to featurize entire datasets for deep learning
  - Want ready-made sampling and batching utilities for training pipelines
  - Already use `atomworks.io` and want a seamless bridge to ML-ready feature engineering

### 2. Quick Start

To parse a pdb file (parse = load, clean, annotate relevant metadata such as entities, molecules, etc) you can use the `parse` function:

```python

from atomworks.io.parser import parse

result = parse(filename="3nez.cif.gz")

asym_unit: AtomArrayStack = result["asym_unit"]
assemblies: dict[str, AtomArrayStack] = result["assemblies"]

for chain_id, info in result["chain_info"].items():
    print(chain_id, info["sequence"])

```

The output of `parse` includes:

- **chain_info** — Sequences/metadata for each chain
- **ligand_info** — Ligand annotation & metrics
- **asym_unit** — Structure (`AtomArrayStack`)
- **assemblies** — Built biological assemblies (each are their own `AtomArrayStack`)
- **metadata** — Experimental and source information

See [usage examples](https://baker-laboratory.github.io/atomworks-dev/latest/auto_examples/) for more details.

If you just want to load a file, you can use the `load_any` function:

```python
from atomworks.io.utils.io_utils import load_any

atom_array: AtomArray = load_any("3nez.cif.gz", model=1)  # model=1 means that we want to load the model 1 (i.e. the first model) rather than a stack of all models in the file
```

### 3. Training on the PDB

> ⚠️ **Disclaimer:** Documentation for this section is currently under construction. Please check back soon for updates!

**Step 1 — Mirror the PDB (mmCIFs)**
  To train on the PDB, you first need to make sure you have access to the samples form the PDB. We use `mmCIF` files as the highly recommended format for training.
  For convenience, we provide a command to mirror the PDB:

  ```bash
  # Full mirror (~100 GB)
  atomworks pdb sync /path/to/pdb_mirror  # This will create a carbon-copy of the PDB, dated today, in the specified directory. It will download the .mmcif files in the same sharding pattern as the original PDB and keep them gzipped for efficiency.

#   # If, for some reason you only want to download specific IDs, the CLI also supports this:
#   atomworks pdb sync /path/to/pdb_mirror --pdb-id 1A0I --pdb-id 7XYZ  # This will only download the specified PDB IDs.
#   # or
#   atomworks pdb sync /path/to/pdb_mirror --pdb-ids-file /path/to/ids.txt  # This will download the PDB IDs listed in the file, one per line. Each line should be a PDB ID (e.g. '6lyz') and separated by a newline.
  ```

  Once the mirror is created, set the environment variable:

  ```bash
  export PDB_MIRROR_PATH=/path/to/pdb_mirror
  ```

  To have this more permanent, you can add it to a `.env` file in your home directory. Here is an [example of a `.env`](.env.sample) file structure that you can copy, rename to `.env` and edit with your own paths.

**Step 2 — Get PDB metadata (PN units and interfaces)**
    To calculate sampling probabilities and filter examples for splits, we pre-process the PDB with metadata for each PDB entry. 
    To save you the work, we provide pre-computed metadata (dated July 15/2025) for downloading:

  ```bash
  atomworks setup metadata /path/to/metadata  # This will download the metadata (as .tar.gz) and extract it to the specified directory.
  ```

  This produces parquet files at:

- `/path/to/metadata/pn_units_df.parquet` — Contains metadata for each *PN unit* in the PDB. The term *pn unit* is shorthand for `polymer XOR non-polymer unit` and behaves for almost all purposes like the `chain` in a PDB file. The only difference is that a ligand composed of multiple covalently bonded ligands is considered a single PN unit (whilst it would be multiple chains in a PDB file). Effectively this `.parquet` is a large table of all individual chains, ligands, etc (to be precise, it has one entry per  pn unit) in the PDB that includes helpful metadata for filtering and sampling.
- `/path/to/metadata/interfaces_df.parquet` — Contains metadata for each interface in the PDB. This `.parquet` is a large table of all binary interfaces in the PDB. It lists each interface as (pn_unit_1, pn_unit_2) pairs and includes helpful metadata for filtering and sampling.

  Alternatively, you can generate fresher metadata yourself (scripts will be uploaded in the coming weeks).

**Step 3 — Configure an AF3-style dataset (example: train only on D-polypeptides)**
Next we need to use the metadata to configure a dataset that we would like to sample from. This includes e.g. training cut-off, filters, transforms to apply, etc.
Here's a simple example that:

- Filters to D-polypeptide and L-polypeptide chains only (`POLYPEPTIDE_D` and `POLYPEPTIDE_L` -- to include additional chain types, replace the lists with the appropriate IDs (see [mapping](./src/atomworks/enums.py#L31-L45) in comments).
- Excludes ligands in the AF3 list of excluded ligands, available at [`atomworks.io.constants.AF3_EXCLUDED_LIGANDS_REGEX`](./src/atomworks/io/constants.py#L350).

```yaml
# NOTE: The below is a hydra config and the _target_ fields are the hydra syntax for instantiating a class.
#  You can use this without hyrda, but will then instead need to provide the corresponding arguments for the
#  _target_ objects directly.

# Chain type ids used below (from atomworks.enums.ChainType):
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
```

**Step 4 — MSAs (optional)**
We are working on a way to make MSAs accessible to the public, but due to the large storage requirements (multiple TB) we are still working on this. If your organization has interest & capacity to host the MSAs, please contact us. In the meantime, if you have MSAs (e.g., from OpenProteinSet) you can configure the pipeline to use them like so:

```yaml
    protein_msa_dirs:
      - { dir: /path/to/msa, extension: .a3m.gz, directory_depth: 2 }
    rna_msa_dirs:
      - { dir: /path/to/msa, extension: .afa, directory_depth: 0 }
```

Or alternatively not use MSAs.

**Step 5 — Train a model**
You now have a full fledged dataset that you can use to train models on! If you want to just try this out without having to download the whole PDB and the metdatada, you can instead run our tests which have a mini-mockup of the pipeline with real pdb files, metadata, distillation data, templates and MSAs for the example of AF3. You can download all this relevant metadata via the atomworks CLI:

```bash
atomworks setup tests  # This will download the test pack to `tests/data` and unpack it there (~500 MB)
```

You will now have a mini PDB at `tests/data/pdb` and a mini custom CCD at `tests/data/ccd`. MSA and template data is in `tests/data/shared` and the distillation and metadata are in `data/ml/af2_distillation`, `data/ml/pdb_pn_units` and `data/ml/pdb_interfaces`. A dataset that uses all of these is [for example here](./tests/ml/conftest.py#L300).

To run the tests for the various datasets, you can run the following command:

```bash
# Make sure you have the correct environment activated, and set your paths correctly in the .env file / shell environment variables (see points above)
pytest tests/ml/test_data_loading_pipelines.py
```

---

## Contribution

We welcome improvements!  
Please see the [full documentation](https://baker-laboratory.github.io/atomworks-dev/latest/index.html) for contribution guidelines.

## Citation

If you make use of AtomWorks in your research, please cite:

> N. Corley\*, S. Mathis\*, R. Krishna\*, M. S. Bauer, T. R. Thompson, W. Ahern, M. W. Kazman, R. I. Brent, K. Didi, A. Kubaney, L. McHugh, A. Nagle, A. Favor, M. Kshirsagar, P. Sturmfels, Y. Li, J. Butcher, B. Qiang, L. L. Schaaf, R. Mitra, K. Campbell, O. Zhang, R. Weissman, I. R. Humphreys, Q. Cong, J. Funk, S. Sonthalia, P. Lio, D. Baker, F. DiMaio,
> "Accelerating Biomolecular Modeling with AtomWorks and RF3," bioRxiv, August 2025. doi: [10.1101/2025.08.14.670328](https://doi.org/10.1101/2025.08.14.670328)

If you use bibtex, here's the GoogleScholar formatted citation:

```bibtex
@article{corley2025accelerating,
  title={Accelerating Biomolecular Modeling with AtomWorks and RF3},
  author={Corley, Nathaniel and Mathis, Simon and Krishna, Rohith and Bauer, Magnus S and Thompson, Tuscan R and Ahern, Woody and Kazman, Maxwell W and Brent, Rafael I and Didi, Kieran and Kubaney, Andrew and others},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
