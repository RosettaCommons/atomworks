"""Concrete implementations of common molecular design conditions."""

import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Shaped

from atomworks.constants import MASKED
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.scatter import apply_and_spread_segment_wise
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.standard_annotations.base import Level
from atomworks.ml.conditions.base import ConditionBase

Array = np.ndarray


class Sequence(ConditionBase):
    name = "sequence"
    n_body = 1
    level = Level.RESIDUE
    dtype = str
    aliases = ("C_SEQ",)
    default_value = MASKED

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return annotation != MASKED

    @classmethod
    def ground_truth_value(cls, atom_array: AtomArray) -> Shaped[Array, "n_atoms"]:  # noqa: F821
        # The conditioned identity is the residue name.
        return atom_array.res_name


class Coordinate(ConditionBase):
    name = "coordinate"
    n_body = 1
    level = Level.ATOM
    dtype = float
    aliases = ("C_CRD",)
    value_shape = (3,)
    default_value = np.nan

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray) -> np.ndarray:
        return np.isfinite(annotation).any(axis=1)

    @classmethod
    def ground_truth_value(cls, atom_array: AtomArray) -> Float[Array, "n_atoms 3"]:  # noqa: F722
        # The conditioned target is the atom's coordinates.
        return atom_array.coord


class Index(ConditionBase):
    name = "index"
    n_body = 1
    level = Level.RESIDUE
    dtype = bool
    aliases = ("C_IDX",)
    default_value = True


class Distance(ConditionBase):
    name = "distance"
    n_body = 2
    level = Level.ATOM
    is_symmetric = True
    dtype = float
    aliases = ("C_DIS",)

    @classmethod
    def mask_from_annotation(cls, annotation: AnnotationList2D) -> AnnotationList2D:
        return AnnotationList2D(
            n_atoms=annotation.n_atoms,
            pairs=annotation.pairs,
            values=annotation.values > 0,
        )

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=np.array([], dtype=int),
            values=np.array([], dtype=cls.storage_dtype()),
        )


class NTerminus(ConditionBase):
    name = "n-terminus"
    n_body = 1
    level = Level.RESIDUE
    dtype = bool
    aliases = ("mask_n-terminus_1_residue", "C_NTR")

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for N-terminus atoms
        is_n_terminus = atom_array.is_polymer & (atom_array.within_chain_res_idx == 0) & atom_array.is_chain_start

        # ... spread to full residue
        residue_segments = get_residue_starts(atom_array, add_exclusive_stop=True)
        is_n_terminus = apply_and_spread_segment_wise(residue_segments, is_n_terminus, np.any)

        return is_n_terminus


class CTerminus(ConditionBase):
    name = "c-terminus"
    n_body = 1
    level = Level.RESIDUE
    dtype = bool
    aliases = ("mask_c-terminus_1_residue", "C_CTR")

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F821
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for C-terminus atoms
        annotations = atom_array.get_annotation_categories()
        chain_ids = atom_array.chain_iid if "chain_iid" in annotations else atom_array.chain_id
        # ... find max within_chain_res_idx for each chain
        is_max_within_chain_res_idx = np.zeros(atom_array.array_length(), dtype=bool)
        for chain_id in np.unique(chain_ids):
            is_this_chain = chain_ids == chain_id
            max_chain_idx = np.max(atom_array.get_annotation("within_chain_res_idx")[is_this_chain])
            is_max_idx = atom_array.get_annotation("within_chain_res_idx") == max_chain_idx
            is_max_within_chain_res_idx |= is_this_chain & is_max_idx

        # ... spread to full residue
        is_c_terminus = is_max_within_chain_res_idx & atom_array.is_polymer
        residue_segments = get_residue_starts(atom_array, add_exclusive_stop=True)
        is_c_terminus = apply_and_spread_segment_wise(residue_segments, is_c_terminus, np.any)

        return is_c_terminus


class Chain(ConditionBase):
    name = "chain"
    n_body = 2
    level = Level.CHAIN
    is_symmetric = True
    dtype = bool
    aliases = ("mask_chain_2_chain", "C_CHA")

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotations = atom_array.get_annotation_categories()
        chain_iid_annotation = "chain_iid" if "chain_iid" in annotations else "chain_id"
        chain_instance = atom_array.get_annotation(chain_iid_annotation)
        is_same_chain = chain_instance[None, :] == chain_instance[:, None]
        pairs = np.stack(np.where(is_same_chain), axis=0).T
        values = np.ones(pairs.shape[0], dtype=bool)
        return AnnotationList2D(atom_array.array_length(), pairs=pairs, values=values)
