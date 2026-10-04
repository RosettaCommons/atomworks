# AtomWorks 2.2 and earlier → 3.0: short migration guide

This guide covers the main I/O changes when moving from AtomWorks 2.2 and earlier to 3.0: parser configuration, charges and bonds, alternate conformers, and CIF writing. See the [detailed guide](migration-long.md) for examples and compatibility details.

## 1. Put parser options in `ParseConfig`

The result still contains `asym_unit`, `assemblies`, `chain_info`, `ligand_info`, `metadata`, and `extra_info` for ordinary inputs. Use `source` (or the first positional argument), and make processing choices explicit:

```python
from atomworks.io import parse
from atomworks.io.config import ParseConfig

config = ParseConfig.from_preset(
    "rcsb",
    hydrogen_policy="remove",
    build_assembly="first",
)
result = parse("structure.cif", config=config)
```

Surviving bare keyword options and `filename=` remain accepted with deprecation warnings. Remove `fix_formal_charges` and `fix_bond_types`: these flags no longer exist; completing missing atoms now includes chemistry sanitization. `hydrogen_policy="infer"` is no longer a supported parser policy; use explicit protonation if needed. [Parser and configuration](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py).

## 2. Check changed charges and covalent chemistry

Covalent linkage processing corrects leaving atoms, bond orders, hydrogen counts and formal charges, including incorrectly charged amide nitrogens. Link validation also checks whether connections fit the selected conformer and supported chemistry; ambiguous or unsupported products can now raise an error and require explicit chemical definitions. A model trained on 2.2 and earlier features can be sensitive to these changed inputs. Keep its old preprocessing environment available, compare features on representative covalent complexes, and evaluate the checkpoint before switching production inference. There is no parser switch that reproduces all historical chemistry. See the [detailed chemistry guidance](migration-long.md#covalent-links-reaction-states-and-unsupported-chemistry) for affected cases and how to handle them.

## 3. Choose alternate conformers deliberately

CIF parsing supports `altloc="first"` (default), a specific letter, `"random_per_chain"`, or `"random_clash_aware"`. Set `altloc_seed` for reproducible random selection. This selects a conformer; it does not return an exhaustive ensemble. PDB parsing only supports `"first"`. Keep `"first"` for an initial migration comparison; adopt random selection as a separate data change. Record the seed with the parsed data. [Options](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py#L106).

## 4. Use CIF to preserve structure chemistry and annotations

The writer and parser now support substantially richer AtomArray → CIF → AtomArray round trips, including custom component chemistry. Use `to_cif_file()`, retain component/entity categories, and reload with `add_missing_atoms=False` when preserving an already prepared structure. For registered StandardAnnotations, also set `return_atom_array_plus=True, load_standard_annotations=True` on reload. Assemblies with repeated chain identities need an explicit `chain_disambiguation` policy.

Round trips are structural, not byte-for-byte identity: coordinates are rounded to three decimals, order/dtypes and some derived IDs may change, and arbitrary annotations require explicit handling. See the [worked example and limits](migration-long.md#cif-round-trips).

## 5. Refresh the environment and parser caches

Install into a fresh environment and regenerate your dependency lock: AtomWorks 3.0 requires exactly `biotite==1.6.0` (2.2.1 used 1.4.0), `pyarrow>=23.0.1`, and Python ≥3.11. Rebuild or separate parser caches when adopting the new chemistry and conformer policies. [Dependencies](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/pyproject.toml).

Before switching, compare a small fixed set of structures under both environments: selected model/assembly, atom identities and counts, bonds, charges and implicit hydrogens. If these structures feed an existing model, check its predictions with the new parser output.
