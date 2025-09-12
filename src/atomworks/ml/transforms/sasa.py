import logging
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger("atomworks.ml")


def calculate_atomwise_sasa(
    atom_array: AtomArray, probe_radius: float = 1.4, atom_radii: str | np.ndarray = "ProtOr", point_number: int = 100
) -> np.ndarray:
    """
    Calculate the SASA for each atom in `atom_array`, excluding those
    with NaN coordinates. The output will have the same length as the
    input AtomArray, with NaN values for excluded (invalid) atoms.

     Args:
        probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
        atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
        point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.

    """
    # 1) Create a boolean vector for valid atoms (no NaNs in their coordinates)
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)

    # 2) Slice the array to keep only valid atoms
    valid_atom_array = atom_array[has_resolved_coordinates]

    # 3) Compute SASA on only the valid atoms
    valid_sasa = struc.sasa(
        valid_atom_array, probe_radius=probe_radius, vdw_radii=atom_radii, point_number=point_number
    )

    # 4) Create a full-length result array, fill with NaNs
    full_sasa = np.full(atom_array.array_length(), np.nan, dtype=float)

    # 5) Place valid SASA values back into their original positions
    full_sasa[has_resolved_coordinates] = valid_sasa

    return full_sasa


def calculate_atomwise_rasa(
    atom_array: AtomArray,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "ProtOr",
    point_number: int = 100,
) -> np.ndarray:
    """
    Calculate the Relative Solvent-Accessible Surface Area (RASA) for each atom in `atom_array`.

    The RASA is defined as the ratio of the SASA of a residue in a protein structure
    to the SASA of the same residue in an extended conformation.

    The output will have the same length as the input AtomArray, with NaN values for excluded (invalid) atoms.

    Args:
        atom_array (AtomArray): The input AtomArray containing the atomic coordinates.
        probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
        atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
        point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
    """
    default_vdw_radius = 1.8
    # 1) Calculate the SASA for each atom in the atom array
    try:
        sasa = calculate_atomwise_sasa(
            atom_array,
            probe_radius=probe_radius,
            atom_radii=atom_radii,
            point_number=point_number,
        )
    except Exception as e:
        logger.error(f"Error calculating SASA: {e}. Defaulting to NaN.")
        return np.full(atom_array.array_length(), np.nan, dtype=float)

    # 2) Calculate the SASA for each atom in an extended conformation
    max_value = np.zeros(atom_array.array_length(), dtype=float)
    for i, row in enumerate(atom_array):
        # get the residue name and atom name
        res_name = row.res_name
        atom_name = row.atom_name
        # get the vdw radius
        try:
            vdw_radius = struc.info.radii.vdw_radius_protor(res_name, atom_name)
        except Exception:
            # if the residue name and atom name are not found, set vdw_radius to 1.8
            vdw_radius = default_vdw_radius
        if vdw_radius is None:
            # if the vdw radius is None, set it to 1.8
            vdw_radius = default_vdw_radius
        # calculate the extended conformation
        extended_conformation = 4 * np.pi * (vdw_radius + probe_radius) ** 2
        # set the extended conformation to the sasa
        max_value[i] = extended_conformation
    # 3) Calculate the RASA
    rasa = sasa / max_value
    return rasa


class CalculateSASA(Transform):
    """Transform for calculating Solvent-Accessible Surface Area (SASA) for each atom in an AtomArray."""

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: Literal["ProtOr"] | np.ndarray = "ProtOr",
        point_number: int = 100,
    ):
        """
        Initialize the CalculateSASA transform.

        Args:
            probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
            atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
            point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
        """
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict, key_to_add_sasa_to: str = "atom_array") -> dict:
        """Calculates SASA and adds it to the data dictionary under the key "atom_array".

        Args:
            data: A dictionary containing the input data atomarray.
            key_to_add_sasa_to: The key in the data dictionary to add the SASA values to.

        Returns:
            The data dictionary with SASA values added.
        """
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
