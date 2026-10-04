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

| Library | Use it for |
| --- | --- |
| `atomworks.io` | Parse and standardize structures, sequences and molecules as Biotite `AtomArray` objects; preserve chemical annotations and build assemblies. |
| `atomworks.ml` | Compose dataset transforms, featurization, sampling and batching for model training. Requires the `ml` extra. |

For the motivation and applications, see the [preprint](https://doi.org/10.1101/2025.08.14.670328).

## Installation

Requires **Python 3.11 or newer**.

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

See the [installation guide](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/installation.rst) and [mirror setup](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/mirrors.rst)
for optional extras, development environments and large-scale data access.

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

The result contains the asymmetric unit and assemblies as `AtomArrayStack` objects,
plus `chain_info`, `ligand_info`, `metadata` and `extra_info` dictionaries.

See the [examples](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/examples) and [parser API reference](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/io/parser.rst)
for more parsing workflows and configuration options.

For minimal processing, use the same parser with the `minimal` preset:

```python
from atomworks.io import parse

atom_array = parse(structure_file, config="minimal")["asym_unit"][0]
```

---

## Contributing

See the [contributor guide](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/docs/contributor_guide.rst) for contribution guidelines.

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
