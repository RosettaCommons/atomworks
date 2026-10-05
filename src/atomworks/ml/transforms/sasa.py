import logging
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_single

from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger("atomworks.ml")

# User-extension dict for VdW radii of elements not known to biotite.
# Checked before biotite's vdw_radius_single in _get_element_radii and
# _get_protor_element_fallback_radii. Populate at runtime to override or
# extend biotite's built-in Mantina 2009 table for exotic elements.
_ELEMENT_VDW_RADII: dict[str, float] = {}


def _resolve_element_radius(elem: str) -> float | None:
    """Look up VdW radius for a single element string."""
    return _ELEMENT_VDW_RADII.get(elem) or vdw_radius_single(elem)


def _get_element_radii(atom_array: AtomArray) -> np.ndarray:
    """Get VdW radii for all atoms based on element type (Mantina 2009).

    Checks ``_ELEMENT_VDW_RADII`` first (for user overrides), then falls back
    to biotite's ``vdw_radius_single`` which covers the full periodic table.
    Raises ValueError only if neither source has a radius.

    Vectorized: resolves each unique element once, then broadcasts.
    """
    elements = atom_array.element
    unique_elems = np.unique(elements)
    elem_to_radius: dict[str, float] = {}
    for e in unique_elems:
        r = _resolve_element_radius(e)
        if r is None:
            idx = int(np.where(elements == e)[0][0])
            raise ValueError(
                f"No VdW radius found for element '{e}' at atom index {idx} "
                f"(atom_name='{atom_array.atom_name[idx]}', res_name='{atom_array.res_name[idx]}'). "
                f"Neither _ELEMENT_VDW_RADII nor biotite's vdw_radius_single cover this element. "
                f"To add support, update _ELEMENT_VDW_RADII in atomworks.ml.transforms.sasa."
            )
        elem_to_radius[e] = r

    radii = np.array([elem_to_radius[e] for e in elements], dtype=np.float32)
    return radii


def _resolve_protor_radius(res_name: str, atom_name: str) -> float | None:
    """Look up ProtOr VdW radius for a single (res_name, atom_name) pair."""
    try:
        return struc.info.radii.vdw_radius_protor(res_name, atom_name)
    except Exception:
        return None


def _get_protor_element_fallback_radii(atom_array: AtomArray) -> np.ndarray:
    """Get ProtOr radii where available, falling back to element radii otherwise.

    Best of both worlds: uses ProtOr's implicit-H-aware radii for standard
    protein residues and element-based Mantina 2009 radii for ligands, metals,
    and non-standard residues.

    Vectorized: resolves each unique (res_name, atom_name, element) triple once,
    then broadcasts.
    """
    triples = list(zip(atom_array.res_name, atom_array.atom_name, atom_array.element, strict=False))
    unique_triples = set(triples)
    triple_to_radius: dict[tuple[str, str, str], float] = {}
    for rn, an, elem in unique_triples:
        r = _resolve_protor_radius(rn, an)
        if r is not None:
            triple_to_radius[(rn, an, elem)] = r
            continue
        r_elem = _resolve_element_radius(elem)
        if r_elem is None:
            idx = int(
                np.where((atom_array.res_name == rn) & (atom_array.atom_name == an) & (atom_array.element == elem))[0][
                    0
                ]
            )
            raise ValueError(
                f"No radius found for atom at index {idx} "
                f"(element='{elem}', atom_name='{an}', res_name='{rn}'). "
                f"Neither ProtOr nor element radii cover this atom."
            )
        triple_to_radius[(rn, an, elem)] = r_elem

    radii = np.array([triple_to_radius[t] for t in triples], dtype=np.float32)
    return radii


def calculate_atomwise_sasa(
    atom_array: AtomArray,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "auto",
    point_number: int = 100,
    atom_filter: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the SASA for each atom in `atom_array`, excluding those
    with NaN coordinates. The output will have the same length as the
    input AtomArray, with NaN values for excluded (invalid) atoms.

    Args:
        atom_array: The input AtomArray containing the atomic coordinates.
        probe_radius: Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (water).
        atom_radii: Atom radii to use. Options:
            - "auto": ProtOr where available, element-based fallback for ligands/metals/
              non-standard residues. Best accuracy for mixed protein-ligand structures. Default.
            - "element": Element-based VdW radii (Mantina 2009). Works for all atom types
              including ligands and metals.
            - "ProtOr": Protein-only radii that account for implicit hydrogens (Tsai 1999).
              Only covers standard protein residues; ions and non-standard residues
              receive NaN SASA (biotite default). Use "auto" for mixed structures.
            - np.ndarray: Custom radii array matching atom_array length.
        point_number: Number of points in the Shrake-Rupley algorithm. Defaults to 100.
        atom_filter: Boolean mask for atoms to include. Defaults to None (all atoms).
    """
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)
    if atom_filter is None:
        atom_filter = has_resolved_coordinates
    else:
        atom_filter = atom_filter & has_resolved_coordinates

    valid_atom_array = atom_array[atom_filter]

    if len(valid_atom_array) == 0:
        return np.full(atom_array.array_length(), np.nan, dtype=float)

    if atom_radii in ("element", "auto"):
        if atom_radii == "auto":
            radii = _get_protor_element_fallback_radii(valid_atom_array)
        else:
            radii = _get_element_radii(valid_atom_array)
        valid_sasa = struc.sasa(
            valid_atom_array,
            probe_radius=probe_radius,
            vdw_radii=radii,
            point_number=point_number,
            ignore_ions=False,
        )
    else:
        vdw_radii_arg = atom_radii[atom_filter] if isinstance(atom_radii, np.ndarray) else atom_radii
        valid_sasa = struc.sasa(
            valid_atom_array, probe_radius=probe_radius, vdw_radii=vdw_radii_arg, point_number=point_number
        )

    full_sasa = np.full(atom_array.array_length(), np.nan, dtype=float)
    full_sasa[atom_filter] = valid_sasa

    return full_sasa


def calculate_atomwise_rasa(
    atom_array: AtomArray,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "auto",
    point_number: int = 100,
    atom_filter: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the Relative Solvent-Accessible Surface Area (RASA) for each atom.

    RASA = SASA_in_complex / SASA_extended_conformation, where the extended
    conformation SASA is computed from the isolated atom's VdW radius.

    Args:
        atom_array: The input AtomArray containing the atomic coordinates.
        probe_radius: Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4.
        atom_radii: Atom radii to use. "auto" (default) uses ProtOr where available,
            element fallback for the rest. "element" works for all atom types.
            "ProtOr" only works for standard protein residues (Tsai 1999).
        point_number: Number of Shrake-Rupley sample points. Defaults to 100.
        atom_filter: Boolean mask for atoms to include. Defaults to None.
    """
    try:
        sasa = calculate_atomwise_sasa(
            atom_array,
            atom_filter=atom_filter,
            probe_radius=probe_radius,
            atom_radii=atom_radii,
            point_number=point_number,
        )
    except Exception as e:
        logger.error(f"Error calculating SASA: {e}. Defaulting to NaN.")
        return np.full(atom_array.array_length(), np.nan, dtype=float)

    try:
        if atom_radii == "element":
            vdw_radii_arr = _get_element_radii(atom_array)
        elif atom_radii in ("ProtOr", "auto"):
            vdw_radii_arr = _get_protor_element_fallback_radii(atom_array)
        else:
            vdw_radii_arr = atom_radii
    except (ValueError, KeyError) as e:
        logger.error(f"Error getting VdW radii for RASA normalization: {e}. Defaulting to NaN.")
        return np.full(atom_array.array_length(), np.nan, dtype=float)

    max_value = 4 * np.pi * (vdw_radii_arr + probe_radius) ** 2
    rasa = sasa / max_value
    return rasa


class CalculateSASA(Transform):
    """Transform for calculating Solvent-Accessible Surface Area (SASA) for each atom in an AtomArray."""

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: Literal["ProtOr", "element", "auto"] | np.ndarray = "auto",
        point_number: int = 100,
    ):
        """
        Initialize the CalculateSASA transform.

        Args:
            probe_radius: Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4.
            atom_radii: Atom radii to use. "auto" (default) uses ProtOr where available,
                element fallback for the rest. "element" works for all atom types.
                "ProtOr" only works for standard protein residues (Tsai 1999).
            point_number: Number of Shrake-Rupley sample points. Defaults to 100.
        """
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict, key_to_add_sasa_to: str = "atom_array") -> dict:
        """Calculates SASA and adds it to the data dictionary under the key "atom_array"."""
        atom_array: AtomArray = data[key_to_add_sasa_to]
        sasa = calculate_atomwise_sasa(
            atom_array,
            self.probe_radius,
            self.atom_radii,
            self.point_number,
        )
        atom_array.set_annotation("sasa", sasa)
        data[key_to_add_sasa_to] = atom_array
        return data
