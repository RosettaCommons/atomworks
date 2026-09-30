"""Transforms to handle covalent modifications"""

from typing import ClassVar

from biotite.structure import AtomArray

from atomworks.io.transforms.atomize import flag_and_reassign_covalent_modifications
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform


class AnnotateCovalentModifications(Transform):
    """Handles covalent modifications within the AtomArray.

    Covalent modifications, e.g., glycosylation, are handled by the following algorithm:

    for polymer residues with atoms covalently bound to a NON-POLYMER:
        for ALL atoms in the polymer residue:
            set the pn_unit_iid and pn_unit_id identifying annotations to that of the NON-POLYMER polymer/non-polymer unit
            set atomize = true (thus, this transform must be run before the Atomize transform)
            set is_covalent_modification = true (for the entire pn_unit)

    Side-chain-modified residues are atomized with probability ``p_reassign_modified_residue``,
    sampled independently per residue. Backbone modifications (cyclizations, crosslinks, caps) are atomized when
    ``atomize_backbone_modifications``. The non-polymer is flagged and atomized either way, so
    ``is_covalent_modification`` is always annotated.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, "AddGlobalTokenIdAnnotation"]

    def __init__(self, p_reassign_modified_residue: float = 1.0, atomize_backbone_modifications: bool = True):
        """Initialize the transform.

        Args:
          p_reassign_modified_residue: Per-residue probability of atomizing side-chain-modified
            residues, between 0 and 1 inclusive. Defaults to ``1.0``.
          atomize_backbone_modifications: Atomize the residue when the modifier attaches at its
            polymerization atom (a cyclization, crosslink, or cap). Defaults to ``True``.
        """
        super().__init__()
        if not 0.0 <= p_reassign_modified_residue <= 1.0:
            raise ValueError("p_reassign_modified_residue must be between 0 and 1 inclusive.")
        self.p_reassign_modified_residue = p_reassign_modified_residue
        self.atomize_backbone_modifications = atomize_backbone_modifications

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = flag_and_reassign_covalent_modifications(
            data["atom_array"],
            p_reassign_modified_residue=self.p_reassign_modified_residue,
            atomize_backbone_modifications=self.atomize_backbone_modifications,
        )
        return data


# Backwards-compatible alias for the pre-rename class name.
FlagAndReassignCovalentModifications = AnnotateCovalentModifications
