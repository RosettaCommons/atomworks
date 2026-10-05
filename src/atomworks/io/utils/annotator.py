"""
A module for lazily adding annotations to `AtomArray` objects.

This module provides a framework for defining, registering, and applying annotations
to `AtomArray` instances. Annotations are numpy arrays of the same
length as the atom array, providing information about each atom (e.g., `is_protein`,
`is_ligand`).

The core components are:
- `ANNOTATOR_REGISTRY`: A global dictionary that maps annotation names to their
  generator functions. This registry is populated automatically at import time.
- `_register_lazy_annotator`: A decorator used to register a new annotation
  generator. The decorated function should accept an `AtomArray` and return a
  `numpy.ndarray` with the annotation values.
- `ensure_annotations`: A function to ensure that one or more annotations are
  present on an `AtomArray`. If an annotation is missing, its registered
  generator function is called to compute and add it.
- `remove_annotations` and `clear_generated_annotations`: Utility functions to
  remove specific or all registered annotations from an `AtomArray`.

Example:
    To define a new annotation, create a function and decorate it:

    >>> @_register_lazy_annotator("is_hydrophobic")
    ... def is_hydrophobic(atom_array: AtomArray) -> np.ndarray:
    ...     hydrophobic_res = ["ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET"]
    ...     return np.isin(atom_array.res_name, hydrophobic_res)

    To apply this and other annotations to an `AtomArray` in-place:

    >>> from atomworks.ml.utils.testing import cached_parse
    >>> data = cached_parse("1L2Y")
    >>> atom_array = data["atom_array"]
    >>> ensure_annotations(atom_array, "is_hydrophobic", "is_protein")
    >>> print(atom_array.get_annotation("is_hydrophobic"))
    [ True  True  True ... False False False]
"""

import contextlib
import functools
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Int

from atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    MASKED,
    MAX_CHEM_TYPE_LENGTH,
    METAL_ELEMENTS,
    NUCLEIC_ACID_BACKBONE_ATOM_NAMES,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA,
    STANDARD_AA_TIP_ATOM_NAMES,
    STANDARD_DNA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_ATOMIC_NUMBER,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType
from atomworks.io.transforms.atom_array import add_chain_type_annotation
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.ccd import get_chem_comp_type
from atomworks.io.utils.chain_info import build_chain_info
from atomworks.io.utils.scatter import apply_and_spread_segment_wise
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS

Array = np.ndarray
"""Alias for numpy.ndarray"""

ANNOTATOR_REGISTRY: dict[str, Callable[[AtomArray], None]] = {}
"""
Registry of annotation generators.

NOTE: This is a global registry and will auto-populate the annotation generator
functions as long as they are decorated with `register_lazy_annotator`. These
registration functions get called at import time.
"""


# General tooling for annotating atom arrays with simple annotations
def _register_lazy_annotator(annot_name: str) -> Callable:
    """
    Decorator that adds an annotation to AtomArray if it doesn't already exist.
    Also registers the annotation in the ANNOTATION_GENERATORS.

    Args:
        annot_name: Name of the annotation to check/add

    Returns:
        Decorator function
    """

    def decorator(fn: Callable[[AtomArray], Array]) -> Callable[[AtomArray], None]:
        @functools.wraps(fn)
        def wrapper(atom_array: AtomArray) -> None:
            if annot_name not in get_annotation_categories(atom_array, n_body="all"):
                values = fn(atom_array)
                if isinstance(values, AnnotationList2D):
                    atom_array.set_annotation(annot_name, values, n_body=2)
                elif np.ndim(values) == 0:
                    atom_array.set_annotation(annot_name, values, n_body=0)
                else:
                    atom_array.set_annotation(annot_name, values)

        # Register the annotation in the ANNOTATOR_REGISTRY
        ANNOTATOR_REGISTRY[annot_name] = wrapper

        return wrapper

    return decorator


def _requires_annotations(*annot_names: str) -> Callable:
    """
    Decorator that ensures required annotations exist before function execution.
    Required annotations must be registered in the ANNOTATOR_REGISTRY.

    NOTE: When using this in conjunction with `_register_lazy_annotator`, the
    annotation order has to be:
    ```python
    @_register_lazy_annotator("is_XXX")
    @_requires_annotations("is_YYY", "is_ZZZ")
    def is_XXX(atom_array: AtomArray) -> Bool[Array, "n_atoms"]: ...
    ```
    Otherwise, the required `is_YYY` and `is_ZZZ` annotations will not be generated
    before `is_XXX` is called.

    Args:
        *annot_names: Names of required annotations

    Returns:
        Decorator function
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(atom_array: AtomArray, *args, **kwargs) -> Any:
            # Generate missing annotations
            for annot_name in annot_names:
                _ensure_single_annotation(atom_array, annot_name)

            # Call the original function
            return fn(atom_array, *args, **kwargs)

        return wrapper

    return decorator


def _resolve_standard_annotation(name: str) -> type | None:
    """Return the StandardAnnotation matching a registry name, canonical name, or alias."""
    for annotation_cls in STANDARD_ANNOTATIONS:
        if name in (annotation_cls.name, annotation_cls.full_name, *annotation_cls.aliases):
            return annotation_cls
    return None


def _ensure_single_annotation(atom_array: AtomArray, name: str) -> None:
    """Ensure a single annotation exists on ``atom_array``, generating it if absent.

    Checks ``ANNOTATOR_REGISTRY`` first, then falls back to
    :py:class:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase` subclasses
    registered in :py:attr:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationMeta._registry`.

    Raises:
        ValueError: If no generator is found for ``name``.
    """
    if name not in get_annotation_categories(atom_array, n_body="all"):
        if name in ANNOTATOR_REGISTRY:
            ANNOTATOR_REGISTRY[name](atom_array)
        else:
            result = _resolve_standard_annotation(name)
            if result is not None:
                result.set_annotation(atom_array, result.annotation(atom_array, default="generate"))
            else:
                raise ValueError(f"No generator found for annotation: {name}")


def ensure_annotations(atom_array: AtomArray, *annotation_names: str) -> None:
    """
    Ensure that specified annotations exist on the AtomArray.
    If an annotation does not exist, it will be generated according to the
    generator function registered in the `ANNOTATOR_REGISTRY`, or by the
    default generation method of a registered :py:class:`StandardAnnotationBase`
    subclass, and added to the `AtomArray` in-place.

    StandardAnnotation names resolve to canonical storage fields, preserving existing alias values.

    Args:
        atom_array: The AtomArray to annotate
        *annotation_names: Names of annotations to ensure. Must be
            registered in ``ANNOTATOR_REGISTRY`` or correspond to a
            registered StandardAnnotation's registry name, ``full_name``, or accepted alias.

    Raises:
        ValueError: If a requested annotation has no generator
    """
    for name in annotation_names:
        _ensure_single_annotation(atom_array, name)


def remove_annotations(atom_array: AtomArray, *annotation_names: str) -> None:
    """
    Remove annotations from the AtomArray.

    Args:
        atom_array: The AtomArray to modify
        *annotation_names: Names of annotations to remove

    Note:
        Silently skips annotations that don't exist.
    """
    existing = {n_body: get_annotation_categories(atom_array, n_body=n_body) for n_body in (1, 2, 0)}
    for name in annotation_names:
        for n_body, categories in existing.items():
            if name in categories:
                if n_body == 1:
                    atom_array.del_annotation(name)
                else:
                    atom_array.del_annotation(name, n_body=n_body)
                break


def clear_generated_annotations(atom_array: AtomArray) -> None:
    """
    Remove all annotations that were generated by this module.

    Args:
        atom_array: The AtomArray to modify
    """
    remove_annotations(atom_array, *ANNOTATOR_REGISTRY)


# Custom annotation generators
@_register_lazy_annotator("chain_type")
def chain_type(atom_array: AtomArray) -> Array:
    """Annotate chain type using the ChainType enum."""
    chain_info = build_chain_info(atom_array)
    modified_atom_array = add_chain_type_annotation(atom_array.copy(), chain_info)  # avoid inplace operations
    return modified_atom_array.chain_type


@_register_lazy_annotator("is_protein")
@_requires_annotations("chain_type")
def is_protein(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein chains."""
    return np.isin(atom_array.chain_type, ChainType.get_proteins())


@_register_lazy_annotator("is_nucleic_acid")
@_requires_annotations("chain_type")
def is_nucleic_acid(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to nucleic acid chains."""
    return np.isin(atom_array.chain_type, ChainType.get_nucleic_acids())


@_register_lazy_annotator("is_dna")
@_requires_annotations("chain_type")
def is_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to DNA chains."""
    return np.isin(atom_array.chain_type, ChainType.DNA)


@_register_lazy_annotator("is_rna")
@_requires_annotations("chain_type")
def is_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to RNA chains."""
    return np.isin(atom_array.chain_type, ChainType.RNA)


@_register_lazy_annotator("is_standard_aa")
def is_standard_aa(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard amino acids."""
    return np.isin(atom_array.res_name, STANDARD_AA)


@_register_lazy_annotator("is_standard_or_unknown_aa")
def is_standard_or_unknown_aa(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown amino acid residue.

    NOTE: May be different than the chain-level "is_protein" in the case of mixed-type chains (e.g., a protein with a non-canonical amino acid).
    """
    return np.isin(atom_array.res_name, [*STANDARD_AA, UNKNOWN_AA])


@_register_lazy_annotator("is_standard_unknown_or_masked_aa")
@_requires_annotations("chain_type", "is_standard_or_unknown_aa")
def is_standard_unknown_or_masked_aa(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard, unknown, or masked amino acid residue.

    NOTE: May be different than the chain-level "is_protein" in the case of mixed-type chains (e.g., a protein with a non-canonical amino acid).
    """
    standard_and_unknown = atom_array.get_annotation("is_standard_or_unknown_aa")
    masked = (atom_array.res_name == MASKED) & (atom_array.chain_type == ChainType.POLYPEPTIDE_L)
    return standard_and_unknown | masked


@_register_lazy_annotator("is_protein_backbone")
@_requires_annotations("is_protein")
def is_protein_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein backbone."""

    is_protein = atom_array.get_annotation("is_protein")
    is_backbone_atom = np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    return is_protein & is_backbone_atom


@_register_lazy_annotator("is_protein_sidechain")
@_requires_annotations("is_protein")
def is_protein_sidechain(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to protein sidechain."""
    is_protein = atom_array.get_annotation("is_protein")
    is_sidechain_atom = ~np.isin(atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    return is_protein & is_sidechain_atom


@_register_lazy_annotator("is_rna_backbone")
@_requires_annotations("is_rna")
def is_rna_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to RNA backbone (sugar-phosphate)."""
    is_rna = atom_array.get_annotation("is_rna")
    is_backbone_atom = np.isin(atom_array.atom_name, RNA_BACKBONE_ATOM_NAMES)
    return is_rna & is_backbone_atom


@_register_lazy_annotator("is_dna_backbone")
@_requires_annotations("is_dna")
def is_dna_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to DNA backbone (sugar-phosphate)."""
    is_dna = atom_array.get_annotation("is_dna")
    is_backbone_atom = np.isin(atom_array.atom_name, DNA_BACKBONE_ATOM_NAMES)
    return is_dna & is_backbone_atom


@_register_lazy_annotator("is_nucleic_acid_backbone")
@_requires_annotations("is_nucleic_acid")
def is_nucleic_acid_backbone(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to nucleic acid backbone (sugar-phosphate)."""
    is_na = atom_array.get_annotation("is_nucleic_acid")
    is_backbone_atom = np.isin(atom_array.atom_name, NUCLEIC_ACID_BACKBONE_ATOM_NAMES)
    return is_na & is_backbone_atom


@_register_lazy_annotator("is_standard_rna")
def is_standard_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard RNA."""
    return np.isin(atom_array.res_name, STANDARD_RNA)


@_register_lazy_annotator("is_standard_dna")
def is_standard_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard DNA."""
    return np.isin(atom_array.res_name, STANDARD_DNA)


@_register_lazy_annotator("is_standard_or_unknown_dna")
def is_standard_or_unknown_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown DNA residue.

    NOTE: May be different than the chain-level "is_dna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    return np.isin(atom_array.res_name, [*STANDARD_DNA, UNKNOWN_DNA])


@_register_lazy_annotator("is_standard_unknown_or_masked_dna")
@_requires_annotations("chain_type", "is_standard_or_unknown_dna")
def is_standard_unknown_or_masked_dna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard, unknown, or masked DNA residue.

    NOTE: May be different than the chain-level "is_dna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    standard_and_unknown = atom_array.get_annotation("is_standard_or_unknown_dna")
    masked = (atom_array.res_name == MASKED) & (atom_array.chain_type == ChainType.DNA)
    return standard_and_unknown | masked


@_register_lazy_annotator("is_standard_or_unknown_rna")
def is_standard_or_unknown_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard or unknown RNA residue.

    NOTE: May be different than the chain-level "is_rna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    return np.isin(atom_array.res_name, [*STANDARD_RNA, UNKNOWN_RNA])


@_register_lazy_annotator("is_standard_unknown_or_masked_rna")
@_requires_annotations("chain_type", "is_standard_or_unknown_rna")
def is_standard_unknown_or_masked_rna(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to standard, unknown, or masked RNA residue.

    NOTE: May be different than the chain-level "is_rna" in the case of mixed-type chains (e.g., DNA/RNA hybrids).
    """
    standard_and_unknown = atom_array.get_annotation("is_standard_or_unknown_rna")
    masked = (atom_array.res_name == MASKED) & (atom_array.chain_type == ChainType.RNA)
    return standard_and_unknown | masked


@_register_lazy_annotator("is_pyrimidine")
def is_pyrimidine(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to pyrimidine residues."""
    return np.isin(atom_array.res_name, STANDARD_PYRIMIDINE_RESIDUES)


@_register_lazy_annotator("is_purine")
def is_purine(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to purine residues."""
    return np.isin(atom_array.res_name, STANDARD_PURINE_RESIDUES)


@_register_lazy_annotator("is_tip_atom")
def is_tip_atom(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Identify tip atoms in standard amino acids."""
    is_tip_atom = np.zeros(atom_array.array_length(), dtype=bool)

    for res_name, tip_atom_names in STANDARD_AA_TIP_ATOM_NAMES.items():
        mask = (atom_array.res_name == res_name) & np.isin(atom_array.atom_name, tip_atom_names)
        is_tip_atom |= mask

    return is_tip_atom


@_register_lazy_annotator("is_res_start")
def is_res_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Mark the first atom of each residue."""
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)
    is_res_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_res_start[res_starts] = True
    return is_res_start


@_register_lazy_annotator("is_chain_start")
def is_chain_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to chain starts."""
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=False)
    is_chain_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_chain_start[chain_starts] = True
    return is_chain_start


@_register_lazy_annotator("is_polymer")
@_requires_annotations("chain_type")
def is_polymer(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to polymers."""
    return np.isin(atom_array.chain_type, ChainType.get_polymers())


@_register_lazy_annotator("is_ligand")
@_requires_annotations("chain_type")
def is_ligand(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to ligands."""
    return np.isin(atom_array.chain_type, ChainType.get_non_polymers())


@_register_lazy_annotator("is_metal")
def is_metal(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to metal ions."""
    return np.isin(atom_array.res_name, METAL_ELEMENTS)


@_register_lazy_annotator("is_carbohydrate")
def is_carbohydrate(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if atoms belong to carbohydrates."""
    return struc.filter_carbohydrates(atom_array)


@_register_lazy_annotator("res_min_occupancy")
@_requires_annotations("is_res_start")
def res_min_occupancy(atom_array: AtomArray) -> Float[Array, "n_atoms"]:  # noqa: F821
    """Calculate minimum occupancy for each residue."""
    is_res_start = atom_array.get_annotation("is_res_start")
    res_start_idxs = np.where(is_res_start)[0]
    res_segments = np.concatenate([res_start_idxs, [atom_array.array_length()]])
    return apply_and_spread_segment_wise(res_segments, atom_array.occupancy, np.min)


@_register_lazy_annotator("res_has_tip_atom")
@_requires_annotations("is_tip_atom", "is_res_start")
def res_has_tip_atom(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
    """Check if each residue contains at least one tip atom."""
    is_tip_atom = atom_array.get_annotation("is_tip_atom")
    is_res_start = atom_array.get_annotation("is_res_start")
    res_start_idxs = np.where(is_res_start)[0]
    res_segments = np.concatenate([res_start_idxs, [atom_array.array_length()]])
    return apply_and_spread_segment_wise(res_segments, is_tip_atom, np.any)


@_register_lazy_annotator("atomic_number")
def atomic_number(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F821
    """Get atomic numbers for each atom. Unknown elements map to ``UNKNOWN_ATOMIC_NUMBER``."""
    atomic_numbers = [
        ELEMENT_NAME_TO_ATOMIC_NUMBER.get(elem, UNKNOWN_ATOMIC_NUMBER) for elem in np.char.upper(atom_array.element)
    ]
    return np.array(atomic_numbers, dtype=np.int8)


@_register_lazy_annotator("chem_comp_type")
def chem_comp_type(atom_array: AtomArray) -> np.ndarray:
    """Get chemical component types for each atom (``<U{MAX_CHEM_TYPE_LENGTH}`` dtype).

    Residues not in the CCD or custom CCD registry are set to ``OTHER``.
    """
    # Cache by unique res_name
    res_name_to_type: dict[str, str] = {}
    residue_bounds = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    residue_sizes = np.diff(residue_bounds)
    chem_comp_types = np.empty(len(residue_sizes), dtype=f"<U{MAX_CHEM_TYPE_LENGTH}")
    for i, (start, stop) in enumerate(pairwise(residue_bounds)):
        res_name = atom_array.res_name[start]

        comp_type = res_name_to_type.get(res_name)
        if comp_type is None:
            comp_type = get_chem_comp_type(res_name, atom_names=set(atom_array.atom_name[start:stop]))
            res_name_to_type[res_name] = comp_type

        chem_comp_types[i] = comp_type

    return np.repeat(chem_comp_types, residue_sizes)


# If available, ensure that all annotators in `atomworks.ml` get registered in the `ANNOTATOR_REGISTRY`.
# This import must come AFTER all definitions in this module to avoid circular import issues:
# ml/conditions/annotator.py imports _register_lazy_annotator from this module, so this module
# must be fully initialized before ml/conditions/annotator.py can be imported.
with contextlib.suppress(ImportError):
    import atomworks.ml.utils.annotator as _annotator_ml  # noqa: F401
