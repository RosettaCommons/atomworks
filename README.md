[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rosettacommons.github.io/atomworks/latest/)
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
- Model missing atoms (those implied by the sequence but not represented in the coordinates) and initialize entity- and instance-level annotations (see the [glossary](https://rosettacommons.github.io/atomworks/latest/glossary.html) for more detail on our composable naming conventions)

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
> Note: AtomWorks requires Python >= 3.11 and [`dotenv`](https://pypi.org/project/python-dotenv/#file-format)

```shell
pip install atomworks # base installation version without torch (for only atomworks.io)
pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
pip install "atomworks[dev]" # with development dependencies
pip install "atomworks[openbabel]" # with [Open Babel](https://openbabel.org/) and its dependencies
pip install "atomworks[ml,openbabel,dev]" # with all dependencies
```
*Running multiple of these installations will just add to the installed dependencies and will not install multiple installations of atomworks.*

If you are using [uv](https://docs.astral.sh/uv/reference/policies/versioning/) for package management, you can install atomworks with:

```shell
uv pip install "atomworks[ml,openbabel,dev]"
```

For more advanced setup options (including how to run workflows via apptainers) see the [full documentation](https://rosettacommons.github.io/atomworks/latest/index.html).

---

## Getting started

This section contains information for how to get atomworks set up and a quick guide for using some of the features of atomworks.io to parse PDB files. To learn more about the features in atomworks.io and atomworks.ml, see the [external documentation](https://rosettacommons.github.io/atomworks/latest/). 

To parse a pdb file (parse = load, clean, annotate relevant metadata such as entities, molecules, etc) you can use the `parse` function:

> Note: To run the code in this section you will need to download the 3nez.cif.gz file yourself. See the [examples](https://rosettacommons.github.io/atomworks/latest/auto_examples/index.html) for how to download files from the PDB within a Python script. 

```python

from atomworks.io.parser import parse
from biotite.structure import AtomArrayStack

result = parse(filename="3nez.cif.gz")

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

See [usage examples](https://rosettacommons.github.io/atomworks/latest/auto_examples/index.html) for more examples of the use of `parse()`. All of the provided examples make use of this method. 
See [API reference documentation](https://rosettacommons.github.io/atomworks/latest/io/parser.html) for more information on this method.

If you just want to load a file, you can use the `load_any` function:

```python
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray

atom_array: AtomArray = load_any("3nez.cif.gz", model=1)  # model=1 means that we want to load the model 1 (i.e. the first model) rather than a stack of all models in the file
```

---

## Contribution

We welcome improvements!

Please see the [contributors guide in the full documentation](https://rosettacommons.github.io/atomworks/latest/contributor_guide.html) for contribution guidelines.

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
