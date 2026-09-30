# AtomWorks 2.x → 3.0: short migration guide

This guide highlights the changes most likely to affect users and agents moving from AtomWorks 2.x to 3.0. See the [detailed guide](migration-long.md) for examples and compatibility details.

## 1. Put parser options in `ParseConfig`

The result still contains `asym_unit`, `assemblies`, `chain_info`, `ligand_info`, `metadata`, and `extra_info` for ordinary inputs. Use `source` (or the first positional argument), and make processing choices explicit:

```python
from atomworks.io import parse
from atomworks.io.config import ParseConfig

config = ParseConfig.from_preset(
    "rcsb",                    # Includes MSE → MET, like STANDARD_PARSER_ARGS
    hydrogen_policy="remove",
    build_assembly="first",
)
result = parse("structure.cif", config=config)
```

Surviving bare keyword options and `filename=` remain accepted with deprecation warnings. Remove `fix_formal_charges` and `fix_bond_types`: these flags no longer exist; completing missing atoms now includes chemistry sanitization. `hydrogen_policy="infer"` is no longer a supported parser policy; use explicit protonation if needed. `ParseConfig()` does **not** enable MSE → MET; the `"rcsb"` preset does. [Parser and configuration](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py).

## 2. Revalidate existing models against changed features

Covalent linkage processing corrects leaving atoms, bond orders, hydrogen counts and formal charges, including incorrectly charged amide nitrogens. A model trained on 2.x features can be sensitive to these corrected inputs. Keep its old preprocessing environment available, compare features on representative covalent complexes, and evaluate the checkpoint before switching production inference. There is no parser switch that reproduces all historical chemistry.

Also check your encoding: `UNIFIED_ATOM37_ENCODING` adds gap class 33 (34 classes total), and terminal `OP3` in slot 36 for standard RNA/DNA. `atom_array_to_encoded_resnames()` now defaults to element/atomic-number tokens for atomized atoms and requires an `atomize` annotation. Pass `atomize_token="<A>"` when that is the checkpoint's intended token convention. An unchanged tensor shape alone does not establish compatible feature meanings. [Chemistry cases](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/tests/io/utils/test_io.py#L417), [encoding](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/encoding_definitions.py).

## 3. Choose alternate conformers deliberately

CIF parsing supports `altloc="first"` (default), a specific letter, `"random_per_chain"`, or `"random_clash_aware"`. Set `altloc_seed` for reproducible random selection. This selects a conformer; it does not return an exhaustive ensemble. PDB parsing only supports `"first"`. Keep `"first"` for an initial migration comparison; adopt random selection as a separate data change. Preserve the seed in training metadata. [Options](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py#L106).

## 4. Use CIF to preserve structure chemistry and annotations

The writer and parser now support substantially richer AtomArray → CIF → AtomArray round trips, including custom component chemistry. Use `to_cif_file()`, retain component/entity categories, and reload with `add_missing_atoms=False` when preserving an already prepared structure. For registered conditions/StandardAnnotations, also set `return_atom_array_plus=True, load_standard_annotations=True` on reload. Assemblies with repeated chain identities need an explicit `chain_disambiguation` policy.

Round trips are structural, not byte-for-byte identity: coordinates are rounded to three decimals, order/dtypes and some derived IDs may change, and arbitrary annotations require explicit handling. See the [worked example and limits](migration-long.md#cif-round-trips).

## 5. Update the surrounding data pipeline

Use `create_structure_loader(column_mapping=...)` instead of specialized loader factories. Pass `ParseConfig` through `parser_args`. A metadata `altloc_seed` can select a deterministic clash-aware conformer; enable this intentionally. Rebuild/version preprocessed tables and caches after adopting the new chemistry or conformer policies. Conditions now store target values through StandardAnnotations; migrate code that directly reads or writes legacy `mask_*` fields.

Install into a fresh environment and regenerate your dependency lock: AtomWorks 3.0 requires exactly `biotite==1.6.0` (2.2.1 used 1.4.0), `pyarrow>=23.0.1`, and Python ≥3.11. Include `[ml]` for Torch workflows. [Dependencies](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/pyproject.toml).

Before adopting 3.0, compare a small fixed dataset under both environments: selected model/assembly, atom identities and counts, bonds/charges/implicit hydrogens, token IDs and masks, then your model's metrics.
