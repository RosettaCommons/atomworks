"""Configuration dataclasses for AtomWorks structure loading.

We define two main dataclasses for configuration:
- :py:class:`PrepareConfig`: Options that can be applied to any structure, regardless of input format. This includes things like adding missing atoms, adding bonds, etc.
- :py:class:`ParseConfig`: Configuration for the :py:func:`~atomworks.io.parser.parse` function, which extends :py:class:`PrepareConfig` with file/parse-specific fields.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import warnings
from typing import Literal, TypeVar

from atomworks.constants import CCD_MIRROR_PATH, CRYSTALLIZATION_AIDS
from atomworks.io.utils.extra_fields import ExtraFieldsType

# fmt: off
LongBondPolicyType = Literal["keep", "filter", "warn", "raise", "filter_nonstandard_only", "warn_nonstandard_only", "raise_nonstandard_only"]
# fmt: on


@dataclasses.dataclass(frozen=True)
class PrepareConfig:
    """Configuration for the AtomWorks processing pipeline.

    Use :py:func:`get_config` or :py:meth:`from_preset` to construct from a named
    preset with optional overrides::

        config = PrepareConfig.from_preset("rcsb")
        config = PrepareConfig.from_preset("lightweight", fix_arginines=False)

    Args:
        add_missing_atoms: Whether to add missing atoms to the structure (from entirely or
            partially unresolved residues). Implies sanitization of bonds and charges after
            bond inference (removing leaving atoms, fixing bond orders for nucleophilic
            additions, correcting formal charges, and fixing charged amide nitrogens), which
            is only well-defined once templates supply the true hydrogen count per atom.
            When ``False``, links without CCD templates retain unknown bond orders.
            Defaults to ``True``.
        add_bond_types_from_struct_conn: Bond types to add from the ``struct_conn``
            CIF category. Defaults to ``("covale",)``, meaning only covalent bonds are
            added (excluding metal coordination and disulfide bonds).
        remove_ccds: CCD codes (e.g. ``"GOL"``, ``"EDO"``) to remove unless covalently
            connected to a non-excluded residue. Defaults to common crystallization aids.
            Coordination does not protect candidates; candidate-only molecules are removed.
        remove_waters: Whether to remove water molecules. Defaults to ``True``.
        fix_ligands_at_symmetry_centers: Whether to patch non-polymer residues at
            symmetry centers that clash with themselves when transformed. Defaults to ``True``.
        fix_arginines: Whether to fix arginine naming ambiguity (ensuring NH1 is always
            closer to CD than NH2); see the AF-3 supplement for details. Defaults to ``True``.
        long_bond_policy: How to handle bonds whose length exceeds expected thresholds.
            Options: ``"keep"``, ``"filter"``, ``"warn"``, ``"raise"``,
            ``"filter_nonstandard_only"``, ``"warn_nonstandard_only"``,
            ``"raise_nonstandard_only"``. Defaults to ``"warn"``.
        add_id_and_entity_annotations: Whether to add identifier and entity annotations
            (``pn_unit_id``, ``molecule_id``, ``chain_entity``, ``pn_unit_entity``,
            ``molecule_entity``) to the structure. Defaults to ``True``.
        convert_mse_to_met: Whether to convert selenomethionine (MSE) residues to
            methionine (MET), as described in the AF-3 supplement. Defaults to ``False``.
        hydrogen_policy: Whether to keep or remove hydrogens. Options: ``"keep"``,
            ``"remove"``. Defaults to ``"keep"``.
        ccd_mirror_path: Path to a local mirror of the Chemical Component Dictionary.
            If ``None``, Biotite's built-in CCD will be used (if the environment is configured).
        return_atom_array_plus: If ``True``, promote all AtomArray / AtomArrayStack objects in the
            result dict to their AtomArrayPlus equivalents.
    """

    add_missing_atoms: bool = True
    add_bond_types_from_struct_conn: tuple[str, ...] = ("covale",)
    remove_ccds: tuple[str, ...] = tuple(CRYSTALLIZATION_AIDS)
    remove_waters: bool = True
    fix_ligands_at_symmetry_centers: bool = True
    fix_arginines: bool = True
    long_bond_policy: LongBondPolicyType = "warn"
    struct_conn_distance_policy: LongBondPolicyType = "filter"
    add_id_and_entity_annotations: bool = True
    convert_mse_to_met: bool = False
    hydrogen_policy: Literal["keep", "remove"] = "keep"
    ccd_mirror_path: str | None = CCD_MIRROR_PATH
    return_atom_array_plus: bool = False

    def __post_init__(self):
        if self.ccd_mirror_path and not os.path.exists(self.ccd_mirror_path):
            raise FileNotFoundError(
                f"Local mirror of the Chemical Component Dictionary provided but does not exist: {self.ccd_mirror_path}. "
                "To use Biotite's built-in CCD, set `ccd_mirror_path` to None (and ensure the environment is configured)."
            )

    def replace(self, **kwargs) -> PrepareConfig:
        """Return a new config with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> PrepareConfig:
        """Construct from a dict, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def to_dict(self) -> dict:
        """Convert to a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_preset(cls, preset: str = "default", **overrides) -> PrepareConfig:
        """Build from a named preset with optional overrides."""
        return get_config(preset, cls=cls, **overrides)


@dataclasses.dataclass(frozen=True)
class ParseConfig(PrepareConfig):
    """Configuration for :py:func:`~atomworks.io.parser.parse` when loading from files.

    Extends :py:class:`PrepareConfig` with file/parse-specific fields.
    Use :py:func:`get_config` or :py:meth:`from_preset` to construct from a named
    preset with optional overrides::

        config = ParseConfig.from_preset("rcsb")
        config = ParseConfig.from_preset("rcsb", model=0, build_assembly="first")

    Args:
        model: Model number to parse for files with multiple models (e.g., NMR).
            ``None`` loads all models. Defaults to ``None``.
        build_assembly: Which biological assembly to build. Options: ``None``
            (asymmetric unit only), ``"first"``, ``"all"``, or a list/tuple of
            assembly IDs as strings. Defaults to ``"all"``.
        extra_fields: Extra CIF fields to include in the AtomArrayStack. ``None``
            includes no extra fields; ``"all"`` includes all fields. Only supported
            for CIF files. Defaults to ``None``.
        file_type: File type of the structure file. If ``None``, inferred
            automatically from the file extension. Defaults to ``None``.
        altloc: Which alternate conformer to select. ``"first"`` selects the
            alphabetically-first altloc per chain, ``"random_per_chain"`` selects a random altloc per
            chain, ``"random_clash_aware"`` selects one altloc per residue while
            avoiding steric clashes between nearby residues, or a specific letter
            (e.g. ``"A"``). Only supported for CIF files; PDB files always use
            ``"first"``. Defaults to ``"first"``.
        altloc_seed: Seed for deterministic random altloc selection when
            ``altloc="random_per_chain"`` or ``altloc="random_clash_aware"``.
            If ``None`` (default), selection is non-deterministic.
        load_standard_annotations: Whether to deserialize StandardAnnotations from
            the CIF block. Incompatible with ``add_missing_atoms=True`` and PDB files. Requires
            ``return_atom_array_plus=True``.
            Defaults to ``False``.
        keep_cif_block: Whether to include the raw CIF block in the result
            dictionary. Defaults to ``False``.
        cache_dir: Cache directory; auto-defaults to ``<tempdir>/atomworks_parse_cache`` when caching is enabled. Defaults to ``None``.
        save_to_cache: Whether to save results as zstd-3 compressed pickles after parsing.
            Defaults to ``False``.
        load_from_cache: Whether to load complete parsed results from cache. Keys include parsing
            options, package versions, and the file path or buffer contents. Unseeded
            random altlocs and active external CCD overrides bypass caching. Defaults to ``False``.
        cif_ccd_on_mismatch: How authored ``chem_comp_atom`` templates treat atoms absent
            from the available CCD template: ``"error_heavy"`` (default) rejects unmatched
            heavy atoms; ``"error"`` rejects any unmatched atom; ``"ignore"`` keeps the
            component without CCD supplementation if heavy-atom names mismatch.
            Heavy atoms must consistently use standard or alternate CCD names.
    """

    model: int | None = None
    build_assembly: Literal["first", "all"] | list[str] | tuple[str] | None = "all"
    extra_fields: ExtraFieldsType = None
    file_type: Literal["cif", "pdb", "bcif", "mmjson"] | None = None
    altloc: Literal["first", "random_per_chain", "random_clash_aware"] | str = "first"
    altloc_seed: int | None = None
    load_standard_annotations: bool = False
    keep_cif_block: bool = False
    cache_dir: str | None = None
    save_to_cache: bool = False
    load_from_cache: bool = False
    cif_ccd_on_mismatch: Literal["ignore", "error_heavy", "error"] = "error_heavy"

    def __post_init__(self):
        super().__post_init__()
        if self.load_standard_annotations and self.add_missing_atoms:
            raise ValueError(
                "load_standard_annotations=True is incompatible with add_missing_atoms=True. "
                "Standard annotations cannot be preserved through add_missing_atoms because atom "
                "indices change. Load with add_missing_atoms=False, or set "
                "load_standard_annotations=False and reconstruct annotations afterward."
            )
        if self.load_standard_annotations and self.convert_mse_to_met:
            raise ValueError(
                "load_standard_annotations=True is incompatible with convert_mse_to_met=True. "
                "MSE to MET conversion changes atom indices, which invalidates standard annotations. "
                "Set convert_mse_to_met=False when loading standard annotations."
            )
        if self.load_standard_annotations and not self.return_atom_array_plus:
            raise ValueError("load_standard_annotations=True requires return_atom_array_plus=True.")
        if (self.load_from_cache or self.save_to_cache) and not self.cache_dir:
            # Default to node-local temp (fast NVMe on GPU nodes); frozen dataclass -> setattr.
            object.__setattr__(self, "cache_dir", os.path.join(tempfile.gettempdir(), "atomworks_parse_cache"))


_PRESETS: dict[str, dict] = {
    "default": {},
    "rcsb": {"convert_mse_to_met": True},
    "lightweight": {
        "add_missing_atoms": False,
    },
    "annotations_only": {
        "add_missing_atoms": False,
        "fix_arginines": False,
        "long_bond_policy": "keep",
    },
    "minimal": {
        "add_missing_atoms": False,
        "fix_arginines": False,
        "add_id_and_entity_annotations": False,
        "long_bond_policy": "keep",
        "remove_waters": False,
        "remove_ccds": (),
    },
}

T = TypeVar("T", bound=PrepareConfig)


def get_config(preset: str = "default", *, cls: type[T] = PrepareConfig, **overrides) -> T:
    """Build a config from a named preset with optional overrides.

    Args:
        preset: Preset name. Available: ``"default"``, ``"rcsb"``, ``"lightweight"``,
            ``"annotations_only"``, ``"minimal"``.
        cls: Config class to instantiate. Defaults to :py:class:`PrepareConfig`.
        **overrides: Fields to override on the preset.
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(_PRESETS)}")
    return cls(**{**_PRESETS[preset], **overrides})


def _resolve_config(
    config: str | T | None,
    kwargs: dict,
    cls: type[T] = PrepareConfig,
) -> tuple[T, dict]:
    """Separate config fields from non-config kwargs. Warns on bare config kwargs."""
    config_fields = {f.name for f in dataclasses.fields(cls)}
    config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}

    if config_kwargs:
        warnings.warn(
            "Passing processing kwargs directly is deprecated; use config instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    if isinstance(config, str):
        config = get_config(config, cls=cls, **config_kwargs)
    elif config is None:
        config = cls(**config_kwargs) if config_kwargs else cls()
    elif config_kwargs:
        config = config.replace(**config_kwargs)

    return config, kwargs
