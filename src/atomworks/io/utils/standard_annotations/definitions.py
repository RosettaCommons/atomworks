"""Non-condition standard annotations.

Defines standard annotations that benefit from the registry-driven save/load
and default-generation pattern, but are NOT model-input conditions.
"""

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils.standard_annotations.base import Level, StandardAnnotationBase


class ExpandableSegmentMinLength(StandardAnnotationBase):
    """Per-residue minimum length for expandable segment sentinels.

    A value of ``-1`` indicates the residue is not an expandable segment sentinel.
    Values ``>= 0`` indicate the sentinel's minimum realized segment length.
    """

    name = "expsegmin"
    n_body = 1
    level = Level.RESIDUE
    dtype = int
    aliases = ("S_SEGMIN",)
    default_value = -1

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != -1


class ExpandableSegmentMaxLength(StandardAnnotationBase):
    """Per-residue maximum length for expandable segment sentinels.

    A value of ``-1`` indicates the residue is not an expandable segment sentinel.
    Values ``>= 0`` indicate the sentinel's maximum realized segment length.
    """

    name = "expsegmax"
    n_body = 1
    level = Level.RESIDUE
    dtype = int
    aliases = ("S_SEGMAX",)
    default_value = -1

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != -1


class Atomize(StandardAnnotationBase):
    """Per-atom flag indicating whether a residue should be treated atomistically.

    Unlike standard polymer residues (which are tokenized at the residue level),
    atomized residues are represented at the individual atom level. This includes
    non-polymers and non-canonical polymer residues by default.

    The canonical field name remains ``"atomize"`` for backward compatibility,
    so ``atom_array.atomize`` continues to work.
    """

    name = "atomize"
    n_body = 1
    level = Level.ATOM
    dtype = bool
    aliases = ("S_ATM",)

    @classmethod
    def get_full_name(cls) -> str:
        return "atomize"

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: True for non-polymers and non-canonical polymer residues."""
        # Lazy to avoid a module-load cycle with `annotator`
        from atomworks.io.utils.annotator import ensure_annotations

        ensure_annotations(atom_array, "is_polymer", "is_standard_aa", "is_standard_rna", "is_standard_dna")
        is_standard_polymer = atom_array.is_polymer & (
            atom_array.is_standard_aa
            | atom_array.is_standard_rna
            | atom_array.is_standard_dna
            | ExpandableSegmentMinLength.mask(atom_array, default="generate")
        )
        return ~is_standard_polymer
