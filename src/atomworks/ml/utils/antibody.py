"""Dataclasses and constants for antibody/TCR Ig chain annotation.

Defines the chain-type classification constants and the :class:`IgAnnotation`
family of dataclasses used to represent ANARCII-numbered CDR/framework boundaries.
"""

import json
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

#### Constants ####

"""
IG constants for antibody preprocessing and transform pipelines
"""
ANTIBODY_CHAIN_TYPES: frozenset[str] = frozenset({"H", "K", "L"})
TCR_CHAIN_TYPES: frozenset[str] = frozenset({"A", "B", "G", "D"})
ALL_IG_CHAIN_TYPES: frozenset[str] = ANTIBODY_CHAIN_TYPES | TCR_CHAIN_TYPES

IG_CHAIN_TYPE_NAMES: dict[str, str] = {
    "H": "heavy",
    "K": "light",
    "L": "light",
    "A": "alpha",
    "B": "beta",
    "G": "gamma",
    "D": "delta",
}

# =============================================================================
# North CDR definitions under IMGT numbering (inclusive on both ends).
# =============================================================================
NORTH_CDR_RANGES_IMGT: dict[str, dict[str, tuple[int, int]]] = {
    "heavy": {"cdr1": (24, 40), "cdr2": (55, 66), "cdr3": (105, 117), "cdr4": (80, 87)},
    "light": {"cdr1": (24, 40), "cdr2": (55, 69), "cdr3": (105, 117), "cdr4": (80, 87)},
}

# IMGT variable-domain position range (inclusive). IMGT V-domain numbering is 1-128.
VARIABLE_DOMAIN_IMGT_RANGE: tuple[int, int] = (1, 128)

# Map a normalized Ig chain-type name (see :data:`IG_CHAIN_TYPE_NAMES`) to its North
# CDR group. TCR chains reuse the antibody definitions by heavy/light analogy:
# beta/delta -> heavy-like, alpha/gamma -> light-like.
CDR_CHAIN_GROUP: dict[str, str] = {
    "heavy": "heavy",
    "light": "light",
    "beta": "heavy",
    "delta": "heavy",
    "alpha": "light",
    "gamma": "light",
}

###################


@dataclass
class CdrRegion:
    """CDR sequence region with 0-indexed slice boundaries into the canonical sequence."""

    start_seq_idx: int
    end_seq_idx: int
    sequence: str


@dataclass
class IgDomain:
    """One numbered Ig variable domain (VH, VL, VA, VB, etc.)."""

    chain_type: str
    scheme: str
    fw_start_seq_idx: int
    fw_end_seq_idx: int
    cdrs: dict[str, CdrRegion]


@dataclass
class IgAnnotation:
    """Full Ig annotation for one pn_unit, potentially multi-domain (e.g. scFv).

    Examples:
      Round-trip through JSON:

      >>> ann = IgAnnotation.from_json(json_str)
      >>> ann.concatenated_cdr_sequence
      'GFTFSSYGISHWVRQAPGKGLEWVA...'
    """

    ig_type: str
    domains: list[IgDomain]
    concatenated_cdr_sequence: str

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "ig_type": self.ig_type,
            "domains": [
                {
                    "chain_type": d.chain_type,
                    "scheme": d.scheme,
                    "fw_start_seq_idx": d.fw_start_seq_idx,
                    "fw_end_seq_idx": d.fw_end_seq_idx,
                    "cdrs": {
                        k: {
                            "start_seq_idx": v.start_seq_idx,
                            "end_seq_idx": v.end_seq_idx,
                            "sequence": v.sequence,
                        }
                        for k, v in d.cdrs.items()
                    },
                }
                for d in self.domains
            ],
            "concatenated_cdr_sequence": self.concatenated_cdr_sequence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IgAnnotation":
        """Reconstruct an :class:`IgAnnotation` from its serialized dict form."""
        domains = [
            IgDomain(
                chain_type=d["chain_type"],
                scheme=d["scheme"],
                fw_start_seq_idx=d["fw_start_seq_idx"],
                fw_end_seq_idx=d["fw_end_seq_idx"],
                cdrs={
                    k: CdrRegion(
                        start_seq_idx=v["start_seq_idx"],
                        end_seq_idx=v["end_seq_idx"],
                        sequence=v.get("sequence", ""),
                    )
                    for k, v in d["cdrs"].items()
                },
            )
            for d in data["domains"]
        ]
        return cls(
            ig_type=data["ig_type"],
            domains=domains,
            concatenated_cdr_sequence=data["concatenated_cdr_sequence"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "IgAnnotation":
        """Deserialize a single :class:`IgAnnotation` from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_masks(self, atom_array: "AtomArray", pn_unit_iid: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return per-CDR masks and the Ig chain mask for this pn_unit.

        Extracts atom-level masks by mapping stored sequence indices to residues in
        atom_array. Verifies that the residue sequence at each CDR's stored indices
        matches the stored CDR sequence before accepting it; CDRs with a mismatching
        sequence (e.g. due to cropping) are silently skipped.

        Args:
            atom_array: Structure with a ``pn_unit_iid`` annotation.
            pn_unit_iid: The pn_unit instance ID for this Ig chain.

        Returns:
            Tuple ``(cdr_masks, ig_chain_mask)``:

            - ``cdr_masks``: dict mapping CDR name (e.g. ``"cdr1"``) to a bool
              atom-level mask. When multiple domains share a CDR name their masks
              are OR-combined.
            - ``ig_chain_mask``: bool atom-level mask spanning all framework + CDR
              residues for this pn_unit.

            Both arrays have length ``atom_array.array_length()``.

        Raises:
            ValueError: If ``pn_unit_iid`` is absent from ``atom_array``.

        Examples:
            >>> cdr_masks, ig_chain_mask = annotation.get_masks(atom_array, "H_1")
            >>> cdr3_mask = cdr_masks.get("cdr3", np.zeros(len(atom_array), dtype=bool))
        """
        from biotite.sequence import ProteinSequence

        n = atom_array.array_length()
        pn_unit_atom_mask = atom_array.pn_unit_iid == pn_unit_iid
        if not pn_unit_atom_mask.any():
            raise ValueError(f"pn_unit_iid '{pn_unit_iid}' not found in atom_array")

        pn_unit_atoms = atom_array[pn_unit_atom_mask]
        global_indices = np.where(pn_unit_atom_mask)[0]

        # res_starts has shape (n_res + 1,): last element is len(pn_unit_atoms)
        res_starts = struc.get_residue_starts(pn_unit_atoms, add_exclusive_stop=True)
        n_res = len(res_starts) - 1

        def _one_letter(local_res_idx: int) -> str:
            atom_idx = res_starts[local_res_idx]
            try:
                return ProteinSequence.convert_letter_3to1(pn_unit_atoms.res_name[atom_idx])
            except Exception:
                return "X"

        def _global_atom_indices(local_res_idx: int) -> np.ndarray:
            return global_indices[res_starts[local_res_idx] : res_starts[local_res_idx + 1]]

        cdr_masks: dict[str, np.ndarray] = {}
        ig_chain_mask = np.zeros(n, dtype=bool)

        for domain in self.domains:
            fw_s = domain.fw_start_seq_idx
            fw_e = domain.fw_end_seq_idx  # stored as exclusive

            # Mark Ig chain atoms over the full framework span
            for ri in range(fw_s, min(fw_e, n_res)):
                ig_chain_mask[_global_atom_indices(ri)] = True

            for cdr_name, cdr_region in domain.cdrs.items():
                cs = cdr_region.start_seq_idx
                ce = cdr_region.end_seq_idx
                expected = cdr_region.sequence

                if cs >= n_res:
                    continue

                # Try inclusive end, then exclusive end (handles off-by-one in source data)
                seq_inc = "".join(_one_letter(i) for i in range(cs, min(ce + 1, n_res)))
                seq_exc = "".join(_one_letter(i) for i in range(cs, min(ce, n_res)))

                if seq_inc == expected:
                    cdr_res_range = range(cs, min(ce + 1, n_res))
                elif seq_exc == expected:
                    cdr_res_range = range(cs, min(ce, n_res))
                else:
                    continue  # sequence mismatch after crop or numbering change

                cdr_atom_mask = np.zeros(n, dtype=bool)
                for ri in cdr_res_range:
                    cdr_atom_mask[_global_atom_indices(ri)] = True

                if cdr_name in cdr_masks:
                    cdr_masks[cdr_name] |= cdr_atom_mask
                else:
                    cdr_masks[cdr_name] = cdr_atom_mask

        return cdr_masks, ig_chain_mask
