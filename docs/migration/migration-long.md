# AtomWorks 2.x → 3.0: detailed migration guide

Start with the I/O changes below: parser configuration, chemical corrections, alternate conformers, CIF round trips and parser caches. The [short guide](migration-short.md) covers the essential steps; additional ML migration notes follow the I/O sections.

## Parser calls and defaults

For 2.x:

```python
from atomworks.io import parse
from atomworks.io.parser import STANDARD_PARSER_ARGS

options = {**STANDARD_PARSER_ARGS, "hydrogen_policy": "remove"}
result = parse(filename="structure.cif", **options)
```

For 3.0:

```python
from atomworks.io import parse
from atomworks.io.config import ParseConfig

config = ParseConfig.from_preset("rcsb", hydrogen_policy="remove")
result = parse("structure.cif", config=config)
```

`ParseConfig` and `PrepareConfig` are frozen dataclasses. Derive a changed configuration with `config.replace(...)`; serialize it with `to_dict()`. Prefer an explicit constructor when migrating old dictionaries: `from_dict()` deliberately drops unknown keys, which can silently discard obsolete chemistry flags or misspellings. Bare surviving options still work with deprecation warnings; obsolete names passed directly to `parse()` produce `TypeError`. [Configuration](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py), [parser](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/parser.py).

| 2.x usage or assumption | 3.0 action |
|---|---|
| `filename=path` | Use positional `source` or `source=path`; alias is deprecated. |
| `STANDARD_PARSER_ARGS` | Use `ParseConfig.from_preset("rcsb")`; dictionary remains deprecated. |
| `fix_formal_charges` / `fix_bond_types` | Remove; completing atoms includes integrated sanitization. There is no independent legacy-equivalence switch. |
| `hydrogen_policy="infer"` | Choose `"keep"`/`"remove"`; run explicit protonation separately when needed. |
| `parse_atom_array(atoms, **options)` | Use `parse_atom_array(atoms, config=PrepareConfig(...))` for a result dictionary. |
| Need only prepared atoms | Use `prepare_atom_array(atoms, config=PrepareConfig(...))`; single-model output is squeezed to AtomArray. |
| `remove_ccds=None` as old default sentinel | Omit the option for default crystallization-aid removal; use `remove_ccds=()` to keep them. |
| Assume all multi-model files produce one stack | Same-topology models can stack; variable-topology CIFs can return a **list of result dictionaries**. Handle this explicitly. |

The ordinary parser result retains `asym_unit`, `assemblies`, `chain_info`, `ligand_info`, `metadata`, and `extra_info`. Do not infer guaranteed rank from the result-key names; choose models deliberately and inspect AtomArray versus AtomArrayStack. Models use Biotite positional conventions (positive positions are one-based); do not mistake a position for an arbitrary deposited model ID. For multi-model inputs, establish the chosen model before calling code requiring a single AtomArray.

Important defaults remain intentional: all models (`model=None`), all assemblies, missing-atom completion enabled, hydrogen policy `"keep"`, waters removed. Default `ParseConfig()` retains MSE; `"rcsb"` converts MSE to MET. `"lightweight"` disables missing-atom completion; `"minimal"` additionally keeps waters/excluded components and disables several annotations/corrections. These are distinct processing choices, not interchangeable performance settings. `long_bond_policy="warn"` and `struct_conn_distance_policy="filter"` can expose or remove problematic connections. Inspect changed bond sets for unusual inputs. [Presets and policies](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py).

## Chemical corrections and existing checkpoints

With missing-atom completion enabled, the pipeline resolves leaving atoms and overvalence, corrects relevant inter-residue formal charges/bond orders, and neutralizes charged amide nitrogens. Hydrogen counts are used in these corrections. With completion disabled, this sanitization is disabled too; selecting that mode solely to suppress changed charges also changes completeness and is not a reproduction of 2.x. [Pipeline](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/_pipeline.py), [link chemistry](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/utils/link_chemistry.py).

The release tests identify concrete cases worth including in a downstream migration set:

| Structures | Behavior to compare |
|---|---|
| `1ivo`, `4js1`, `1qfe` | Charges/hydrogen counts at covalent modifications and amide nitrogens. |
| `1a3g` | Schiff-base linkage bond order and carbon charge. |
| `1j8z`, `1a1e`, `1d9d` | Leaving hydrogen handling and resulting charges. |
| `1rcq` | Alternate conformers, `struct_conn`, and lysine charge. |
| `4ndz`, `6h9v` | Multi-residue ligand bonds and beta-peptide connectivity. |
| `1twr`, `6q9t` | Authored component chemistry and unusual atom names. |

These are documented regression/round-trip cases, not a claim that every structure in a production corpus is chemically validated. [Case list](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/tests/io/utils/test_io.py#L417).

For an existing model, preserve the old locked environment and a small set of its exact features. Compare by atom identity including assembly instance, not array offset alone. Review atom counts, charge/implicit-hydrogen annotations, bond topology and orders, masks and atomization. Evaluate the checkpoint on both feature sets and relevant scientific metrics. If necessary, retain the versioned 2.x preprocessing path for that checkpoint until retraining/fine-tuning or an explicitly validated adapter is available. Avoid silently mixing newly generated features with old cached training data.

### Covalent links, reaction states and unsupported chemistry

Link processing proposes connectivity, resolves the linked product, then validates its valence with RDKit. Recognized explicit `struct_conn` bond orders are retained; missing multiple-bond orders are constrained by both component templates. A candidate single bond is not rejected solely because an isolated CCD reactant is saturated: product formation may displace atoms or change internal bonds and charges. An unsupported explicit order raises instead of silently becoming a single bond. A double bond in one component's leaving group does not by itself authorize a double bond to the other component.

Sequence adjacency also does not authorize arbitrary backbone chemistry. Polymer linking preserves an existing attachment to the adjacent residue and allows side-chain crosslinks alongside backbone bonds; the resulting product must still pass chemistry validation. Metal coordination is excluded from covalent leaving-group removal and valence accounting; a coordination bond alone should not remove a donor's hydrogens or change its charge.

Selected alternate-location labels are retained through missing-atom rebuilding so that authored links remain associated with the selected reaction state. In `6dp5`, the B conformer retains an OP3–magnesium coordination contact, while C has the O3′–P covalent link; rebuilding must not combine these states. Distance filtering also recognizes authored glycosylation: links identified by a glycosylation role, or covalent links between carbohydrates, can be retained through 2.4 Å with a warning when above the usual threshold. This includes the 2.211 Å ASN–NAG link in `2odp`; it is not a general relaxation for every covalent link.

Sanitization handles supported leaving-group substitution, addition across multiple bonds, hydrogen transfer, resonance changes and product charges while preserving observed atoms. Unsupported products still raise `ValueError`; for example, `1vq7` requires an invalid phosphorus bond-order sum of seven. Supply the product's component definitions and explicit connectivity/bond orders when required; do not treat disabling completion or ignoring the exception as a chemical repair. Custom components need consistent `chem_comp_atom` and `chem_comp_bond` definitions.

Extend the migration set with cases that exercise these decisions:

| Structures | Behavior to compare |
|---|---|
| `1rcq`, `1dpn` | Inferred link order and the resulting linked product's bond orders and valence. |
| `3n95`, `4aah` | Sequence-defined terminal caps and coexistence of backbone bonds with side-chain crosslinks. |
| `6dp5` | Conformer-specific covalent versus coordination links after rebuilding. |
| `2odp` | Retention of an authored glycosylation link above the usual distance threshold. |
| `1n4e`, `8qia`, `6n0a` | Thymine photodimer formation, addition across C=N with hydrogen transfer, and substitution of an unresolved ASN nitrogen. |
| `6w13`, `1vq7` | Preservation of observed phosphate oxygen and rejection of an invalid phosphorus product. |

Compare successful outputs and rejected inputs: matching atom counts alone do not establish matching bonds, charges, hydrogen counts or conformer annotations. [Link validation and resolution](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/src/atomworks/io/utils/link_chemistry.py), [product regressions](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/tests/io/utils/test_link_product_regressions.py), [rebuilding pipeline](https://github.com/RosettaCommons/atomworks/blob/release/atomworks-3-0/src/atomworks/io/_pipeline.py).

### Explicit protonation

3.0 provides experimental pH-aware protonation via RDKit/Dimorphite-DL. It is a separate chemical operation and will not numerically reproduce the old Hydride path:

```python
from atomworks.experimental.protonation import add_hydrogens

# atoms is one prepared AtomArray with bonds and charge annotations.
protonated = add_hydrogens(atoms.copy(), ph=7.4)
```

Use a single AtomArray with bonds and charges. `assign_hydrogens()` determines charges and hydrogen counts; `place_hydrogens()` builds those hydrogens, and `add_hydrogens()` runs both. This can change charges, atom count and ordering, so recompute external masks/features afterwards. The old `atomworks.io.utils.protonation.ensure_hydrogens` API is removed in 3.0; use the experimental API above rather than importing its former vendored dependencies.

## Alternate conformers

```python
config = ParseConfig.from_preset(
    "rcsb",
    altloc="random_clash_aware",
    altloc_seed=42,
)
result = parse("structure.cif", config=config)
```

`"first"` remains the default; `"random_per_chain"` selects a letter per chain; `"random_clash_aware"` chooses per-residue alternatives while considering nearby clashes and correlations. A specific letter is also accepted. Selection is not exhaustive conformer enumeration and is not occupancy-weighted ensemble sampling. The clash-aware algorithm can reject unresolved conflicts and can fall back to a modified residue's parent component; this can affect atom identities. It uses the first model's coordinates when selecting from a stack. PDB inputs reject altloc choices other than `"first"`. [Configuration](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/config.py), [selection algorithm](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/utils/altloc.py).

Record the source, policy, seed, parser version and CCD inputs. Unseeded random parsing bypasses the parse cache. Retaining `"first"` makes an initial version comparison easier; switching to random selection should be evaluated separately. A fixed seed establishes reproducibility within the selected implementation, not an eternal cross-version conformer identity guarantee.

**Candidate edge case to resolve before broad claims:** `load_cif()` fallback paths for variable-topology models and some invalid requested-model reads do not forward the requested `altloc`/`altloc_seed`. Exercise these combinations explicitly before promising consistent random selection for all multi-model CIFs. [Loader source](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/_loaders.py).

## CIF round trips

For an already prepared single AtomArray, retaining chemistry and registered annotations:

```python
from atomworks.io.utils.io_utils import to_cif_file

# atoms is an already prepared AtomArray/AtomArrayPlus.
to_cif_file(
    atoms,
    "prepared.cif",
    include_entity_categories=True,
    include_chem_comp=True,
    chain_disambiguation="transformation_id",
)
restored = parse(
    "prepared.cif",
    config=ParseConfig(
        add_missing_atoms=False,
        fix_arginines=False,
        remove_waters=False,
        remove_ccds=(),
        build_assembly=None,
        extra_fields="all",
        return_atom_array_plus=True,
        load_standard_annotations=True,
    ),
)
```

Choose `chain_disambiguation="transformation_id"` to preserve original chain IDs through AtomWorks; repeated chains require the transformation annotation. Choose `"chain_iid"` for more interoperable unique chain names, accepting renamed chains. **Pass this explicitly**: the actual default is `None` and ambiguous identities raise, despite a writer docstring claiming the default is `"chain_iid"`. The writer accepts multiple models as a list of AtomArrays with optional `model_ids`; do not assume every writer accepts a stack directly. [Writer](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/utils/io_utils.py).

The new implementation emits/reloads richer component and bond chemistry, including authored custom components and standard annotations. Tests cover asymmetric units with and without CCD access and assemblies under both disambiguation modes. Scope the phrase “complete round trip” carefully:

- Coordinates are rounded to three decimals by the writer.
- Atom order, annotation dtypes, some regenerated IDs and alternate-location labels are not guaranteed identical. Compare structural identity and values after alignment.
- Request additional ordinary annotations explicitly by name when writing; registered StandardAnnotations use their serializer. In this candidate, writer `extra_fields="all"` can fail on parsed structures because `label_entity_id` conflicts with a reserved CIF column (reproduced on 6LYZ). The example uses writer defaults; it does not promise retention of every ordinary annotation. Arbitrary Python attributes are not a general persistence mechanism.
- StandardAnnotation loading requires AtomArrayPlus and forbids missing-atom completion and MSE conversion, which change indices. PDB files cannot carry this path.
- Serialization of two-body standard annotations has a 10,000-row limit. Large pairwise annotations may need another storage design.
- CIF custom chemistry is scoped to the parse; AtomArrayPlus can retain the custom CCD registry for later writing. Supply `ccd_entries` explicitly if working outside that path.
- `cif_ccd_on_mismatch="error_heavy"` rejects heavy-atom name mismatches in authored component templates by default. Investigate malformed/custom templates; `"ignore"` intentionally skips CCD supplementation for mismatching heavy-atom names rather than repairing them.

[Round-trip tests](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/tests/io/utils/test_io.py), [annotation serialization](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/utils/standard_annotations/serialization.py), [custom CCD handling](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/utils/ccd.py).

## Parser caching

The parser now caches complete results as Zstandard-compressed pickles in a `parse-v2` namespace. Settings, package versions and source identity enter the key; random unseeded conformers and active external CCD overrides bypass caching. **Paths are keyed by resolved path, not file-content hash.** Replacing a file in place can reuse a stale result; keep source paths immutable or supply a changed immutable `cache_key`. An explicit key must change with source contents. Isolate old/new cache directories for a controlled migration. Benchmark cached and uncached performance separately. [Cache code](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/io/parser.py).

## I/O environment and validation

Python ≥3.11 remains supported; the candidate pins Biotite exactly to 1.6.0 and checks it at import time (2.2.1 pins 1.4.0). PyArrow minimum rises to 23.0.1, core adds jaxtyping, and Hydride is no longer a core dependency. Optional S3 and catcif integrations have extras. Regenerate a locked environment instead of upgrading Biotite in isolation. [Dependency manifest](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/pyproject.toml).

Lock both environments, migrate parser calls and configuration, then compare a fixed set of structures covering covalent modifications, alternate conformers and assemblies. Review atom identities, charges, bond orders and hydrogen counts before adopting the new parser or regenerating cached outputs.

## Additional ML migration notes

The remaining sections apply when using AtomWorks for model features or datasets. They are not required for a standalone I/O migration. Install the `[ml]` extra for Torch workflows; it now includes numba.

### Model encodings and conditions

`UNIFIED_ATOM37_ENCODING` retains 37 atom slots but adds terminal phosphate `OP3` in slot 36 for standard DNA/RNA and a new gap token `<G>` at class 33. The vocabulary therefore grows from 33 to 34 classes relative to 2.2.1. Check embedding/head sizes, saved class tables, labels, loss masks and atom-slot masks. A matching 37-wide coordinate tensor can still carry changed semantics. [Definition](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/encoding_definitions.py).

`atom_array_to_encoded_resnames()` changes its default `atomize_token` from `"<A>"` to `None` (atomic-number tokens). It requires `atomize`; a one-atom residue is no longer automatically classified as atomized merely by length. For a checkpoint using generic atom tokens, explicitly supply `atomize_token="<A>"`, and check the encoding contains that token. `EncodeAtomArray` also exposes `atomize_token`/`atomize_atom_name` for explicitly matching its encoding. Keep the checkpoint's saved encoding definition when reproducing an old model rather than assuming the current global constant is compatible. [Encoding functions](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/transforms/encoding.py).

Conditions now specialize the StandardAnnotation framework. Sequence and coordinate conditions contain target residue names and coordinates; derive masks through their API rather than treating the annotation itself as a boolean. To mark existing coordinates as ground truth, for example:

```python
from atomworks.ml.conditions import C_CRD

# selected is a boolean array of length len(atoms).
C_CRD.set_annotation_from_ground_truth(atoms, selected)
conditioned_mask = C_CRD.mask(atoms)
```

For external target values, set explicit condition annotations with the condition API. Calling `set_annotation_from_ground_truth` intentionally replaces targets with values from the current structure. Legacy module aliases and some legacy CIF masks are supported, but this is a specific compatibility list (sequence, coordinate, index), not blanket support for arbitrary historical fields. Custom condition subclasses should follow the new definitions and StandardAnnotation contract. [Condition definitions](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/conditions/definitions.py), [base API](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/conditions/base.py).

### Datasets and preprocessing

For a previous loader created with `create_loader_with_query_pn_units(pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"])`:

```python
from atomworks.ml.datasets.loaders.cif import create_structure_loader

loader = create_structure_loader(
    parser_args=ParseConfig.from_preset("rcsb", hydrogen_policy="remove"),
    column_mapping={
        "query_pn_unit_iids": ["pn_unit_1_iid", "pn_unit_2_iid"],
        "query_is_polymer": ["pn_unit_1_is_polymer", "pn_unit_2_is_polymer"],
    },
)
```

Provide these columns in the metadata table. A string mapping produces a scalar; a list produces an ordered list. Specialized factories remain deprecated wrappers. `storage="filesystem"`, `"bytes"`, and `"blob"` share the factory but have backend-specific arguments. Passing `altloc_seed_colname="altloc_seed"` selects `random_clash_aware` for nonmissing CIF seeds, even if the supplied parser configuration had another altloc policy. Pass it only when the table is meant to define conformer sampling. [Loaders](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/datasets/loaders/cif.py).

Preprocessing is reorganized around `PreprocessConfig` and assembly/PN-unit/interface records; the old `get_pn_unit_data_from_structure` module is removed. Treat new preprocessing tables as versioned datasets. Audit column names/types and row meaning against the release tutorial and your training pipeline rather than assuming old Parquet files are interchangeable. For multiple conformer rows, preserve seeds and consider the optional altloc weighting in `calculate_weights_for_pdb_dataset_df(..., altloc_seed_column=...)` so extra conformers do not accidentally overweight a structure. This option is explicit, not automatic. [Preprocessing](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/preprocessing/preprocess.py), [samplers](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/samplers.py).

`PandasDataset` delegates to a metadata index and can use Arrow with `memory_map=True`. Its `.data` is then an Arrow Table, so pandas-only operations must be adapted. Dataset filters are independent row predicates evaluated on the input and combined; filters that calculate aggregate thresholds from the progressively filtered table may change meaning. Use a stable ID column rather than relying on the old optional `id_column=None` pattern. [Dataset](https://github.com/RosettaCommons/atomworks/blob/df50559731c0ba43cc29a82b50c87a60d1a0a951/src/atomworks/ml/datasets/pandas_dataset.py).
