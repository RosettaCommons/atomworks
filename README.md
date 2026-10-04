[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/RosettaCommons/atomworks/actions/workflows/release_and_docs.yaml/badge.svg?branch=release%2Fatomworks-3-0)](https://github.com/RosettaCommons/atomworks/actions/workflows/release_and_docs.yaml?query=branch%3Arelease%2Fatomworks-3-0)
[![Codecov coverage](https://codecov.io/gh/RosettaCommons/atomworks/branch/release%2Fatomworks-3-0/graph/badge.svg)](https://codecov.io/gh/RosettaCommons/atomworks/tree/release%2Fatomworks-3-0)
[![PyPI version](https://img.shields.io/pypi/v/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue?logo=python&logoColor=white)](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/pyproject.toml)
[![Documentation](https://img.shields.io/badge/docs-read-blue.svg)](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/index.rst)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/LICENSE.md)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RosettaCommons/atomworks/release/atomworks-3-0/docs/_static/atomworks_logo_dark.svg">
    <img src="https://raw.githubusercontent.com/RosettaCommons/atomworks/release/atomworks-3-0/docs/_static/atomworks_logo_color.svg" width="450" alt="AtomWorks logo">
  </picture>
</div>

**AtomWorks** is an open-source toolkit for biomolecular data processing and machine learning,
built on [Biotite](https://www.biotite-python.org/). It brings structure parsing, chemical
standardization, annotations, and composable learning pipelines into a shared atom-level representation.

For models built with AtomWorks, see [Foundry](https://github.com/RosettaCommons/foundry)
(RF3, RFD3, and MPNN) and [RFD4-Proteína](https://github.com/RosettaCommons/RFD4-Proteina).

Start with the [tutorials](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/tutorial/index.rst), [examples](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/examples), or
[model-building tutorial](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/how_to_build_a_model/index.rst).
Upgrading from 2.2 or earlier? See the [AtomWorks 3.0 migration guide](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/migration/migration-short.md)
for parser configuration, covalent chemistry, alternate conformers, and CIF round trips.

AtomWorks is composed of two symbiotic libraries:

- `atomworks.io`: A universal Python toolkit for parsing, cleaning, manipulating, and converting biological data (structures, sequences, small molecules). Built on the [biotite](https://www.biotite-python.org/) API, it seamlessly loads and exports between standard formats like mmCIF, PDB, FASTA, SMILES, MOL, and more. Broadly useful for anyone who works with structural data for biomolecules.
- `atomworks.ml`: Advanced dataset featurization and sampling for deep learning workflows that uses `atomworks.io` as its structural backbone. We provide a comprehensive, pre-built and well-tested set of `Transforms` for common tasks that can be easily composed into full deep-learning pipelines; users may also create their own `Transforms` for custom operations.

For more detail on the motivation for and applications of AtomWorks, please see the [preprint](https://doi.org/10.1101/2025.08.14.670328). 

AtomWorks is built atop [biotite](https://www.biotite-python.org/): We are grateful to the Biotite developers for maintaining such a high-quality and flexible toolkit, and hope that our package will prove a helpful addition to the broader `biotite` community.

---

## atomworks.io

> *A general-purpose Python toolkit for cleaning, standardizing, and manipulating biomolecular structure files, built on [Biotite](https://www.biotite-python.org/).*

**atomworks.io** lets you:

- Parse, convert, and clean any common biological file (structure or sequence). For example, identifying and removing leaving groups, correcting bond order after nucleophilic addition, fixing charges, parsing covalent geometries, and appropriate treatment of structures with multiple occupancies and ligands at symmetry centers
- Transform all data to a consistent `AtomArray` representation for further analysis or machine learning applications, regardless of initial source
- Model missing atoms (those implied by the sequence but not represented in the coordinates) and initialize entity- and instance-level annotations (see the [glossary](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/glossary.rst) for more detail on our composable naming conventions)

We have found `atomworks.io` to be generally useful to a broad bioinformatics and protein design audience; in many cases, `atomworks.io` can replace bespoke scripts and manual curation, enabling researchers to spend more time testing hypotheses and less time juggling dozens of tools and dependencies.

---

## atomworks.ml

> *Modular, component-based library for dataset featurization within biomolecular deep learning workflows.*

**atomworks.ml** provides:

- A library of pre-built, well-tested `Transforms` that can be slotted into novel pipelines
- An extensible framework, integrated with `atomworks.io`, to write `Transforms` for arbitrary use cases
- Pre-built datasets and samplers suitable for most model training scenarios

Transforms consume and return dictionaries containing structural data, annotations, and features.
Biotite's `AtomArray` provides the shared atom-level representation, and model-specific pipelines
can convert these features into tensors. Operations within and between pipelines share a common
vocabulary of inputs and outputs.

We have found that `atomworks.ml` **dramatically** reduces the overhead of starting, and completing, many ML projects; research topics that once took months now achieve signal within weeks if not days, accelerating the pace of innovation.

---

## When to use `atomworks.io` vs `atomworks.ml`?

- Use `atomworks.io` when you:
  - Need to parse/clean/convert between biological file formats (mmCIF, PDB, FASTA, etc.)
  - Want a unified structural representation to plug into any downstream analysis or modeling
  - Need structural operations like adding missing atoms, filtering ligands/solvents, or assembly generation

- Use `atomworks.ml` when you:
  - Need to featurize entire datasets for deep learning
  - Want ready-made sampling and batching utilities for training pipelines
  - Already use `atomworks.io` and want a seamless bridge to ML-ready feature engineering

---

## Installation

AtomWorks requires **Python 3.11 or newer**. Pip installs the core dependencies automatically,
including `python-dotenv` and the compatible Biotite version.

Install the latest published package from [PyPI](https://pypi.org/project/atomworks/):

```shell
pip install atomworks                     # Core structure IO; no PyTorch
pip install "atomworks[ml]"               # IO plus PyTorch and ML pipelines
pip install "atomworks[ml,openbabel,dev]"  # ML, Open Babel, and development tools
```

To install the **AtomWorks 3.0 release branch** represented by this README directly with pip:

```shell
pip install "atomworks[ml] @ git+https://github.com/RosettaCommons/atomworks.git@release/atomworks-3-0"
```

The PyPI badge reports the published package version. Development of 3.0 uses
`release/atomworks-3-0`.
With [uv](https://docs.astral.sh/uv/), use `uv pip install` in place of `pip install`.

Optional extras can be combined: `ml` (PyTorch), `openbabel` (Open Babel), `s3` (S3 storage),
`ase` (ASE databases), `posebusters` (structure validation), `catcif` (CIF archives),
`dev` (development tools), and `docs` (documentation builds).
See the [installation guide](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/installation.rst) and [mirror setup](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/mirrors.rst)
for development environments and large-scale data access.

---

## Getting started

Download a structure from the PDB and parse it into a standardized representation with
sequence, chain, ligand, and assembly metadata. In AtomWorks 3.0, use `ParseConfig` to
make parsing choices explicit:

```python
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from biotite.database import rcsb
from biotite.structure import AtomArrayStack

structure_file = rcsb.fetch("3nez", format="cif", target_path=".")
config = ParseConfig.from_preset("rcsb", build_assembly="first")
result = parse(structure_file, config=config)

asym_unit: AtomArrayStack = result["asym_unit"]
assemblies: dict[str, AtomArrayStack] = result["assemblies"]

for chain_id, info in result["chain_info"].items():
    print(chain_id, info["processed_entity_canonical_sequence"])

```

The output of `parse` includes:

- **chain_info** — Sequences/metadata for each chain
- **ligand_info** — Ligand annotation & metrics
- **asym_unit** — Structure (`AtomArrayStack`)
- **assemblies** — Built biological assemblies (each are their own `AtomArrayStack`)
- **metadata** — Experimental and source information
- **extra_info** — Cache and compatibility information

See the [examples](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/examples) and [parser API reference](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/io/parser.rst)
for more parsing workflows and configuration options.

If you just want to load a file, you can use the `load_any` function:

```python
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray

atom_array: AtomArray = load_any(structure_file, model=1)  # Load the first model.
```

---

## Contribution

We welcome improvements!

Please see the [contributor guide](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/contributor_guide.rst) for contribution guidelines.

## Acknowledgments

We thank Hope Woods and Rachel Clune from the Rosetta Commons for their partnership and collaboration on the codebase, documentation, tutorials, and examples.

## Citation

If you make use of AtomWorks in your research, please cite:

> N. Corley\*, S. Mathis\*, R. Krishna\*, M. S. Bauer, T. R. Thompson, W. Ahern, M. W. Kazman, R. I. Brent, K. Didi, A. Kubaney, L. McHugh, A. Nagle, A. Favor, M. Kshirsagar, P. Sturmfels, Y. Li, J. Butcher, B. Qiang, L. L. Schaaf, R. Mitra, K. Campbell, O. Zhang, R. Weissman, I. R. Humphreys, Q. Cong, H. Jiang, J. Funk, S. Sonthalia, P. Lio, D. Baker, F. DiMaio,
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
