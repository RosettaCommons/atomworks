# AGENTS.md

This is the working guide for AI coding agents in AtomWorks. It is model-agnostic and must work in a standalone checkout. When this repository is vendored by another project, also follow the parent repository's instructions; neither guide replaces the other.

Keep this file concise enough to scan, but update it when commands, layout, or invariants change. The README and `docs/` are the user-facing source of truth. Executable code, tests, `pyproject.toml`, and CI define current behavior when documentation disagrees.

## Repository and Git safety

- Work from the repository root, which contains `pyproject.toml` and `Makefile`.
- Check `git status --short` before editing. Preserve existing user changes and keep unrelated changes out of the task.
- Do not discard work with `git reset`, `git clean`, broad `git checkout`/`git restore`, or equivalent destructive commands unless the user explicitly authorizes the exact operation.
- Do not create or switch branches, stage, commit, amend, rebase, stash, or push unless the user explicitly asks. When authorized, stage only intended paths and inspect the staged diff before committing.
- Do not work directly on a shared integration branch such as `dev`, `staging`, `main`, or `production`. Use the branch and PR target requested by the user.
- Use focused commits that form one logical unit. Conventional prefixes used here include `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, and `style`.
- PRs normally target the branch from which the work was based. The internal repository commonly uses `dev`; public 3.0 release PRs target `release/atomworks-3-0`. Verify the intended remote and base rather than guessing.
- Before a PR, review every changed line, verify the target and changed-file list, remove debug artifacts and temporary files, and report exactly which checks ran.

For PRs explicitly targeting `staging`, preserve the repository's promotion boundary. The PR diff must not include:

- any `AGENTS.md` file;
- `.ipd/` or its contents;
- `.env`;
- `src/atomworks/ml/databases/` or its contents.

Check with `git diff origin/staging...HEAD --name-only`. Do not rewrite, reset, or cherry-pick user history merely to remove an excluded file without explicit permission.

## Standalone and vendored environments

AtomWorks requires Python 3.11 or newer; CI and Ruff target Python 3.12. For a standalone checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
make install
```

`make install` installs `.[dev,ml,openbabel]`. Install narrower or additional extras directly when appropriate:

```bash
uv pip install -e ".[dev]"
uv pip install -e ".[dev,ml,openbabel,docs]"
```

When working from the RFProteína submodule at `lib/atomworks`, use the parent environment prepared by the parent repository:

```bash
source ../../.venv/bin/activate
uv pip install -e ".[dev,ml,openbabel,docs]"
```

Do not assume the parent checkout or virtualenv exists in standalone AtomWorks. Conversely, do not invent a second submodule-local environment when the parent workflow owns the environment.

Dependency rules:

- `pyproject.toml` is the source of truth.
- AtomWorks requires exactly `biotite==1.6.0`; import-time patches depend on that version. Do not relax or duplicate the pin without validating all affected consumers.
- Torch-dependent functionality belongs behind the `ml` extra.
- S3, ASE, Open Babel, PoseBusters, catcif, PyMOL, and documentation support are optional. Core imports and ordinary tests must not accidentally require unrelated extras.
- Environment variables such as `PDB_MIRROR_PATH` and `CCD_MIRROR_PATH` point to external data. Never commit secrets or machine-specific paths.

## Architecture map

AtomWorks is a structural-biology data toolkit centered on Biotite `AtomArray` and `AtomArrayStack` objects.

- `src/atomworks/io/`: parsing, configuration, components, transforms, structural utilities, annotations, chemistry, and format conversion.
- `src/atomworks/io/parser.py`: public `parse()` entry point. Parsing is configured with `ParseConfig`; mmCIF is preferred for complete structural metadata.
- `src/atomworks/ml/`: reusable ML datasets, parsers/loaders, preprocessing, samplers, conditions, pipelines, transforms, and utilities.
- `src/atomworks/ml/transforms/base.py`: `Transform`, `Compose`, transform-history tracking, validation, and pipeline error behavior.
- `src/atomworks/ml/pipelines/`: reusable model-oriented transform compositions.
- `src/atomworks_cli/`: Typer applications exposed as `atomworks` and `aw`.
- `tests/io/` and `tests/ml/`: tests mirror the two principal library layers; speed tests live under `tests/io/speed/`.

Use `atomworks.io` for model-independent structure behavior. Use `atomworks.ml` for generally reusable dataset and featurization behavior. Keep downstream-model-specific schemas, conditions, and training policy in the downstream project when they are not broadly useful.

Important concepts:

- AtomWorks patches selected Biotite behavior at import time. Check `src/atomworks/biotite_patch.py` before changing code that relies on residue boundaries, annotations, bonds, or assembly identity.
- A PN unit is a "polymer XOR non-polymer unit." It behaves roughly like a structural unit: a polymer chain is one PN unit, while covalently connected non-polymer chains may form one PN unit.
- Transforms consume and return dictionaries, often containing an `atom_array`, and may mutate or replace values. Their required keys, produced keys, mutation, ordering requirements, and incompatible predecessors are part of the contract.
- Pipelines are ordered transform compositions. Validate behavior in the composition, not only in isolated transforms.

## Structural and scientific invariants

- Preserve `AtomArray` length and alignment across coordinates, annotations, bonds, masks, and parallel arrays unless the operation deliberately filters or expands atoms.
- Preserve chain, entity, PN-unit, molecule, instance, model, and transformation identity. Symmetry copies can share ordinary residue identifiers, so do not identify residues by `chain_id` and `res_id` alone.
- Parsed results conventionally contain `asym_unit`, `assemblies`, `chain_info`, `ligand_info`, `metadata`, and `extra_info`. Confirm the current parser contract and tests before changing this schema.
- Preserve bond indices and chemical annotations through slicing, concatenation, atomization, missing-atom handling, and file round trips.
- Standard annotations and encodings are shared vocabulary. Extend centralized definitions rather than creating alternate spellings in individual transforms or downstream projects.
- State assumptions about model selection, resolved atoms, polymer types, coordinate units, required annotations, and mutation whenever they are not evident from the API.
- Avoid implicit copies or in-place mutation of shared structures. Copy deliberately when isolation is required and test whether annotations, bonds, and custom CCD data survive.
- Chemical cleanup, leaving atoms, protonation, chirality, automorphisms, covalent modifications, symmetry, and CCD lookup are scientific behavior. Changes require focused edge-case tests.

### Efficient residue and segment operations

Use patched residue-boundary utilities and contiguous slices rather than rebuilding a full boolean mask for every residue:

```python
residue_bounds = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
for start, stop in zip(residue_bounds[:-1], residue_bounds[1:], strict=True):
    residue = atom_array[start:stop]
```

The patched boundary logic accounts for available assembly identity such as `transformation_id`. Use selection helpers in `atomworks.io.utils.selection` when their semantics fit. Before optimizing, verify that the array is sorted into contiguous units and that slicing preserves intended identity and bond behavior.

## Implementation conventions

- Survey nearby code and tests before adding an abstraction. Follow established parser, transform, pipeline, and CLI patterns.
- Keep changes focused. Apply DRY to genuinely shared behavior and YAGNI to speculative flexibility; do not create a framework for one call site.
- Prefer module-level functions for stateless behavior. Use classes when state, a transform/parser protocol, or inheritance makes them appropriate.
- Add type annotations to production function signatures. Use `jaxtyping` for public or non-obvious tensor contracts where named axes improve correctness; do not force it onto trivial or genuinely dynamic internals.
- Preserve device, dtype, shape, and batch semantics in ML code.
- Use descriptive `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, and a leading underscore for private members. Standard domain abbreviations such as `pdb_id` and `msa` are fine.
- Let clear native Python errors propagate. Add explicit validation for user-facing APIs, silent-corruption risks, scientific invariants, or materially clearer context. Catch exceptions only to recover, clean up resources, or add actionable information; do not use broad catches with silent defaults.
- Error messages should identify the missing key, annotation, dependency, file, or invalid scientific value.
- Avoid lazy imports unless an optional dependency, import cycle, or meaningful startup cost requires them.
- Comments and docstrings describe only current behavior, current invariants, and current interfaces. Do not document old bugs, former implementations, migration history, or superseded reasoning.
- Keep consecutive `#` comment blocks to at most two lines. Move longer API or behavioral detail into a concise docstring and broad design material into `docs/`. Prefer clear names and structure over explanatory comments.

## Testing

Prefer the smallest relevant test first, then expand according to risk. New functionality and bug fixes should normally include tests.

```bash
# One test or module
pytest tests/io/path/to/test_file.py::test_name -q
pytest tests/ml/path/to/test_file.py -q

# Full suite with coverage
make test

# Full suite in parallel
make parallel_test

# Common local selection without slow tests
PDB_MIRROR_PATH=tests/data/pdb pytest tests -m "not slow"

# Parser performance benchmarks
pytest tests/io/speed --benchmark-time-unit=s --benchmark-warmup=False --benchmark-min-rounds=3
```

Testing principles:

- Assert observable behavior and scientific invariants, not private implementation details.
- Prefer a small number of meaningful parameterized cases over many near-identical tests.
- For transforms, test input requirements, output keys, mutation/copy behavior, annotation and bond alignment, empty/minimal inputs, and composition with neighboring transforms.
- Use property- or invariant-oriented tests for filtering, cropping, permutation, atomization, and round trips. Examples include length/alignment, idempotence where promised, bond validity, and preservation of untouched annotations.
- Test parsing and chemistry across representative polymers, ligands, covalent modifications, alternate locations, assemblies, missing atoms, and symmetry when relevant.
- Use fixtures for shared synthetic structures and `cached_parse()` from testing utilities for repository structures.
- Name tests `test_<behavior>_<scenario>` and apply existing markers accurately. Do not make an ordinary unit test depend on a network service or unavailable optional extra.
- Do not add tests for trivial getters, obvious delegation, or implementation details already covered by a stronger integration test.

### Test data and regression policy

Download the public parser test pack when necessary:

```bash
atomworks setup tests
```

The downloaded pack does not necessarily contain every PDB ID referenced by the entire suite. Prefer configured readable PDB/CCD mirrors when available. A `FileNotFoundError` below the selected mirror indicates missing test data; it is not permission to skip the test or loosen its assertions.

Regression baselines define intended scientific behavior. Never make a regression test pass by filtering mismatches, lowering overlap or numeric thresholds, excluding annotations, widening tolerances without scientific justification, or adding special-case skips. Determine whether the implementation, expectation, or environment is wrong. Regenerate stored baselines only with explicit user intent and explain the behavioral reason.

When AtomWorks is vendored by another project, validate both layers: the smallest relevant AtomWorks tests and a downstream integration test that exercises the changed contract.

## Formatting and documentation

Non-mutating checks:

```bash
ruff format --check .
ruff check src tests
```

Apply formatting only when authorized to modify the affected files:

```bash
make format
```

`make format` rewrites files. Review its diff and avoid formatting unrelated user work. Ruff is configured in this repository's `pyproject.toml` with a 120-column limit and Python 3.12 target.

Public APIs need concise Google-style docstrings:

- A clear one-line summary is enough for a simple function.
- Add elaboration only for behavior, mutation, structural assumptions, tensor axes, coordinate frames, or units that are not obvious.
- Use `Args`, `Returns`/`Yields`, `Raises`, `Examples`, `References`, and `See Also` only when they add information. Do not repeat types already expressed by annotations.
- Document only exceptions that the function intentionally raises or that callers must meaningfully handle.
- Use examples for non-obvious workflows, not every helper.
- Use Sphinx/reStructuredText roles such as `:py:func:`, `:py:class:`, and `:py:meth:` for API cross-references.

Build the documentation with:

```bash
uv pip install -e ".[ml,docs]"
make -C docs html
```

Treat Sphinx warnings, broken references, and failed autodoc imports as problems to investigate. When building both AtomWorks and a vendoring project's docs, separate environments may be needed if their documentation requirements select incompatible dependency versions.

## Completion and PR review

Before handing off or opening a PR:

1. Review `git status`, all unstaged changes, and the exact staged diff if staging was authorized.
2. Confirm the implementation is focused, typed, and consistent with nearby APIs; remove debug prints, breakpoints, commented-out code, accidental TODOs, backups, and generated outputs.
3. Confirm public behavior and non-obvious structural contracts are documented without historical commentary or long comment blocks.
4. Run targeted tests, non-mutating Ruff checks, and broader tests/docs in proportion to the change. Do not claim checks that were not run.
5. Verify no regression expectation was weakened and no baseline was regenerated without explicit intent.
6. Verify the PR remote, base branch, title, changed-file list, and any `staging` exclusions.
7. Summarize behavior changed, validation performed, and anything skipped because data, optional dependencies, hardware, or time was unavailable.

## Where to look first

- Installation and test data: `docs/installation.rst`, `docs/mirrors.rst`
- Contributor and PR workflow: `docs/contributor_guide.rst`, `.github/pull_request_template.md`
- Dependency and Ruff configuration: `pyproject.toml`
- CI behavior: `.github/workflows/`, `.github/ci/`
- Parsing contract: `src/atomworks/io/parser.py`, `src/atomworks/io/config.py`, parser tests
- Structural utilities and annotations: `src/atomworks/io/utils/`, `src/atomworks/constants.py`
- Transform protocol: `src/atomworks/ml/transforms/base.py`, transform and pipeline tests
- CLI behavior: `src/atomworks_cli/`

If this guide disagrees with current code or tests, verify intended public behavior in README/docs and update this file as part of the same change.
