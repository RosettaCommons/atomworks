"""Transforms relating to stereochemistry (chirality)"""

import logging
from itertools import combinations
from typing import Any, ClassVar

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit.Chem import Mol

from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_af3_token_center_coords, get_token_count, spread_token_wise

logger = logging.getLogger(__name__)

_IDEAL_DIHEDRAL_ANGLE = np.arcsin(1 / 3**0.5)
"""
The ideal dihedral angle (in radians) between a side of the tetrahedron and
a plane that contains two atoms of the tetrahedral side and the
center of mass of the tetrahedron (= the chiral center).
This is ~35.26 degrees.
"""


def get_dih(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Copied from rf2aa.kinematics.get_dih to decouple the transform from the rf2aa package.

    Args:
        a,b,c,d : PyTorch tensors of shape [batch,nres,3] that store Cartesian coordinates of four sets of atoms

    Returns:
        dih : pytorch tensor of shape [batch,nres] that stores resulting the dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1n = b1 / (torch.norm(b1, dim=-1, keepdim=True) + eps)

    v = b0 - torch.sum(b0 * b1n, dim=-1, keepdim=True) * b1n
    w = b2 - torch.sum(b2 * b1n, dim=-1, keepdim=True) * b1n

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1n, v, dim=-1) * w, dim=-1)

    return torch.atan2(y + eps, x + eps)


def _get_plane_pair_keys_for_planes_between_chiral_center_and_tetrahedral_side(
    chiral_center: int, bonded_atoms: list[int], take_first_chiral_subordering: bool = True
) -> list[tuple[int, int, int, int]]:
    """
    Get the unique keys that define all pairs of planes that can be formed between:
    (1) a side of the tetrahedron and
    (2) a plane that contains two atoms of the tetrahedral side and the chiral center.

    The `plane_pair_key` (c, i, j, k) encodes the pair of planes (cij) and (ijk)
    where c is the chiral center and (ijk) are the points defining a side of the tetrahedron.
    NOTE: the order of i and j is irrelevant, and we use the convention that i < j.

    Args:
        chiral_center (int): The atom ID of the chiral center.
        bonded_atoms (list[int]): A list of atom IDs bonded to the chiral center.
            Must contain exactly 3 or 4 elements. In the case of 3 elements,
            the 4th element is assumed to be an implicit hydrogen / lone pair.
        take_first_chiral_subordering (bool): If True, only the first subordering is considered (when four
            bonded non-hydrogen atoms are present). If False, all suborderings are considered (leading to
            12 unique plane pairs in the case of 4 bonded atoms, or 3 unique plane pairs in the case of 3 bonded
            atoms).


    Returns:
        list[tuple[int, int, int, int]]: A list of tuples, each representing one of
            the unique keys that define all pairs of planes that can be formed between a side of the tetrahedron and
            a plane that contains two atoms of the tetrahedral side and the chiral center.

    Raises:
        AssertionError: If the length of `bonded_atoms` is not 4.

    Reference:
        `RF2AA supplementary notes figure S1 <https://www.science.org/doi/10.1126/science.adl2528#supplementary-materials>`_

    Example:
        >>> chiral_center = 1
        >>> bonded_atoms = [2, 3, 4, 5]
        >>> get_keys_for_adjacent_planes_from_chiral_center(chiral_center, bonded_atoms)
        [
            (1, 2, 3, 4),  # 3 pairs with side (234)
            (1, 2, 4, 3),
            (1, 3, 4, 2),
            (1, 2, 3, 5),  # 3 pairs with side (235)
            (1, 2, 5, 3),
            (1, 3, 5, 2),
            (1, 2, 4, 5),  # 3 pairs with side (245)
            (1, 2, 5, 4),
            (1, 4, 5, 2),
            (1, 3, 4, 5),  # 3 pairs with side (345)
            (1, 3, 5, 4),
            (1, 4, 5, 3)
        ]
    """

    assert len(bonded_atoms) in (
        3,
        4,
    ), f"Expected 3 bonded heavy atoms (implict hydrogen assumed) or 4 bonded atoms, got {bonded_atoms}"
    bonded_atoms = sorted(bonded_atoms)  # sort to ensure i < j < k < l

    # get unique keys that define all pairs of planes that can be formed between
    #   (1) a side of the tetrahedron and
    #   (2) a plane that contains two atoms of the tetrahedral side and the chiral center
    # The `plane_pair` key (c, i, j, k) encodes the pair of planes (cij) and (ijk)
    #  where c is the chiral center and (ijk) are the points defining a side of the tetrahedron.
    #  NOTE: the order of i and j is irrelevant, and we use the convention that i < j
    plane_pair_keys = []

    # ... iterate over all 4 sides of the tetrahedron (i,j,k,l):
    #  There are 4 such sides: (ijk), (ijl), (ikl), (jkl)
    for tetrahedral_side in combinations(bonded_atoms, 3):
        # If there are four bonded (non-hydrogen) atoms, and we indicated that we want don't want to
        # enumerate all sub-orders, simply take the first tetrahedral side
        if len(bonded_atoms) == 4 and take_first_chiral_subordering:
            plane_pair_keys.append((chiral_center, *tetrahedral_side))
        else:
            # ... iterate over all pairs of atoms in the tetrahedral side that
            #  form a plane together with the chiral center. There are 3 such
            #  pairs for each tetrahedral side (ijk): (ij), (ik), (jk)
            for atom_pair_in_plane_with_chiral_center in combinations(tetrahedral_side, 2):
                # ... get the remaining atom of the tetrahedral side that is not in the plane with the chiral center
                atom_remaining = next(i for i in tetrahedral_side if i not in atom_pair_in_plane_with_chiral_center)

                # The `plane_pair` key (c, i, j, k) encodes the pair of planes (cij) and (ijk)
                #  where c is the chiral center and (ijk) are the points defining a side of the tetrahedron.
                #  NOTE: the order of i and j is irrelevant, and we use the convention that i < j
                plane_pair_key = (chiral_center, *atom_pair_in_plane_with_chiral_center, atom_remaining)

                # add the plane key to the list
                plane_pair_keys.append(plane_pair_key)

    return plane_pair_keys


def get_rf2aa_chiral_features(
    chiral_centers: list[dict], coords: np.ndarray, take_first_chiral_subordering: bool = True
) -> torch.Tensor:
    """Extracts chiral centers and featurize them for RF2AA.

    NOTE: Each row of output features contains the indices of the plane pairs and the signed ideal
        dihedral angle for each chiral center. For example, the entry:
            [c, i, j, k, angle]
        means that the atom at index c is a chiral center with atoms at indices (i, j, k) bonded
        to it. The signed dihedral angle angle is the signed angle between the planes (cij) and
        (ijk). The sign of the angle determines the chirality of the chiral center.

    NOTE: Each chiral center will result in more than one feature. In particular:

        - 3 features if one of the 4 atoms bonded to the chiral center is an implicit hydrogen (as
            we do not look at any pair of planes where one plane contains an implicit hydrogen).
        - 12 features if all 4 atoms bonded to the chiral center are explicit atoms.

    In RF2AA this is used to compute the angles for all unique pairs of planes in the center that are
    explicitly modeled (hydrogens are implicit), measure their error from the ideal tetrahedron in
    the unit sphere, and pass the gradients of the error in predicted angles with respect to the
    predicted coordinates into the subsequent blocks as vector input features in the SE(3)-Transformer
    which breaks the symmetry over reflections present in the rest of the network and allows the
    network to iteratively refine predictions to match ideal tetrahedral geometry.

    Args:
        chiral_centers: A list of dictionaries, of the form:
            {"chiral_center_idx": int, "bonded_explicit_atom_idxs": list[int]}
            where chiral_center_idx is the index of the chiral center atom, and bonded_explicit_atom_idxs
            is a list of the indices of the atoms bonded to the chiral center (excluding implicit hydrogens).
        coords: A numpy array of atomic coordinates.
        take_first_chiral_subordering: If True, only the first subordering is considered (when four
            bonded non-hydrogen atoms are present). If False, all orderings are considered (leading to
            12 unique plane pairs in the case of 4 bonded atoms, or 3 unique plane pairs in the case of 3 bonded
            atoms).

    Returns:
        A tensor of shape [n_chirals, 5] where each row contains the indices of the plane pairs
                      and the signed ideal dihedral angle for each chiral center. The sign of the dihedral
                      angle determines the chirality of the chiral center (+1 for clockwise, -1 for counterclockwise).
                      If no stereocenters are found, returns an empty tensor of shape [0, 5].
    """

    # iterate over all tetrahedral stereo centers and record the plane pairs that define the tetrahedral side
    dihedral_plane_pair_idxs = []
    for chiral_center_info in chiral_centers:
        chiral_center: int = chiral_center_info["chiral_center_idx"]
        bonded_atoms: list[int] = chiral_center_info["bonded_explicit_atom_idxs"]

        # get the keys for uniquely identifying all pairs of planes that can be formed between
        #   a side of the tetrahedron and a plane that contains two atoms of the tetrahedral side and the chiral center
        #   (only planes with )
        plane_pair_keys = _get_plane_pair_keys_for_planes_between_chiral_center_and_tetrahedral_side(
            chiral_center, bonded_atoms, take_first_chiral_subordering=take_first_chiral_subordering
        )
        # append the plane pair keys to the list
        dihedral_plane_pair_idxs.extend(plane_pair_keys)

    if len(dihedral_plane_pair_idxs) == 0:
        # ... if no stereocenters are found, we return an empty tensor of the correct feature shape
        return torch.zeros((0, 5))

    chiral_feats = torch.zeros((len(dihedral_plane_pair_idxs), 5))
    # ... fill in the plane pair indices & ideal dihedral angle
    # We sort the tuples (but not within the tuples) to match the legacy code behavior; it has no material impact
    chiral_idxs = torch.tensor(sorted(dihedral_plane_pair_idxs), dtype=torch.long)

    chiral_feats[:, :-1] = chiral_idxs
    chiral_feats[:, -1] = _IDEAL_DIHEDRAL_ANGLE
    # ... calculate whether the dihedral angle is positive or negative, determining the chirality of the center uniquely
    chiral_feats[:, -1] *= torch.sign(
        get_dih(
            a=coords[chiral_idxs[:, 0]],
            b=coords[chiral_idxs[:, 1]],
            c=coords[chiral_idxs[:, 2]],
            d=coords[chiral_idxs[:, 3]],
        )
    )

    return chiral_feats  # [n_chirals, 5] (float)


class AddRF2AAChiralFeatures(Transform):
    """AddRF2AAChiralFeatures adds chiral features to the atom array data under the "chiral_feats" key.

    Chiral centers are taken from data["chiral_centers"], which is a list of dictionaries, of the form:
    {"chiral_center_atom_id": int, "bonded_explicit_atom_ids": list[int]}

    This metadata can be added by running e.g. the AddOpenBabelMoleculesForAtomizedMolecules and
    GetChiralCentersFromOpenBabel transforms. This transform also requires the AtomizeByCCDName transform
    to be applied previously to ensure the atom array is properly atomized.

    Args:
        data: A dictionary containing the input data, including the atom array and chiral centers.

    Returns:
        The updated data dictionary with the added chiral features under the "chiral_feats" key.

    Example:
        .. code-block:: python

            data = {
                "atom_array": atom_array,
                "chiral_centers": [
                    {"chiral_center_atom_id": 5, "bonded_explicit_atom_ids": [1, 2, 3, 4]},
                    {"chiral_center_atom_id": 10, "bonded_explicit_atom_ids": [6, 7, 8, 9]},
                ],
            }

            transform = AddRF2AAChiralFeatures()
            result = transform.forward(data)

            print(result["chiral_feats"])
            # Output might look like:
            #  (assuming the atom_id s above also correspond to the indices in the atom array,
            #   otherwise the first 4 columns look different as they are the indices in the atom array)
            # tensor([[ 5.,  1.,  2.,  3.,  0.61546...],
            #         [ 5.,  2.,  3.,  4., -0.61546...],
            #         ...
            #         [10.,  7.,  8.,  9., -0.61546...]])
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "chiral_centers"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_annotation(data, ["atomize", "atom_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        token_idxs = spread_token_wise(atom_array, np.arange(get_token_count(atom_array)))
        _id_to_token_idx = dict(zip(atom_array.atom_id, token_idxs, strict=False))
        id_to_idx = np.vectorize(_id_to_token_idx.get)

        chiral_centers_by_idx = []
        for chiral_center in data["chiral_centers"]:
            chiral_center_atom_id = chiral_center["chiral_center_atom_id"]

            if chiral_center_atom_id not in atom_array.atom_id:
                # ... skip chiral centers around atoms that are no longer in the array
                continue

            bonded_atom_ids = chiral_center["bonded_explicit_atom_ids"]
            if not np.all(np.isin(bonded_atom_ids, atom_array.atom_id)):
                # ... skip chiral centers with atoms that are no longer in the array
                continue

            # Convert from `id` to token `idx`
            chiral_center_idx = id_to_idx([chiral_center_atom_id])[0]
            bonded_atom_idxs = id_to_idx(bonded_atom_ids)
            chiral_centers_by_idx.append(
                {"chiral_center_idx": chiral_center_idx, "bonded_explicit_atom_idxs": bonded_atom_idxs}
            )

        chiral_feats = get_rf2aa_chiral_features(
            chiral_centers_by_idx,
            torch.tensor(get_af3_token_center_coords(atom_array)),
            take_first_chiral_subordering=False,
        )
        data["chiral_feats"] = chiral_feats  # [n_chirals, 5]

        return data


def _get_reference_conformer_to_residue_mapping(atom_names: np.ndarray, conformer: AtomArray) -> tuple[np.ndarray]:
    """
    Maps atom indices from a reference conformer (as an AtomArray) to the specified residue, dropping all atoms that are not in the residue.
    Args:
        - atom_names (np.ndarray): Array of atom names in the residue to map to
        - conformer (AtomArray): The reference conformer, as an AtomArray (containing the atom_name annotation)
    Returns:
        - ref_map (np.ndarray): Index in atom of reference positions (-1 if masked)
    """

    # ... mark the atoms that are in the residue (keep) and where
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer.")
            continue
        matching_atom_idx = matching_atom_idx.item()
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    return to_within_res_idx  # [n_atoms_in_conf]


def add_af3_chiral_features(
    atom_array: AtomArray, chiral_centers: dict, rdkit_mols: dict[str, Mol], take_first_chiral_subordering: bool = True
) -> torch.Tensor:
    """Computes chiral features from atom array, chiral centers, and RDKit molecules.

    See `AddAF3ChiralFeatures` for more details.
    """
    # We're going to use the same logic we do in GetAF3ReferenceMoleculeFeatures
    # ... get residue-level stochiometry
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]

    all_chirals = []
    for res_start, res_end in zip(_res_starts, _res_ends, strict=False):
        res_name = atom_array.res_name[res_start]

        chirals = chiral_centers[res_name]
        if len(chirals) == 0:
            continue

        # get rdkit->atomarray mapping
        conformer = atom_array_from_rdkit(
            rdkit_mols[res_name],
            conformer_id=0,
            remove_hydrogens=True,
        )
        _ref_to_conf_map = _get_reference_conformer_to_residue_mapping(
            atom_names=atom_array.atom_name[res_start:res_end], conformer=conformer
        )

        # calculate chirals from reference conformer
        chirals = get_rf2aa_chiral_features(
            chirals, torch.tensor(conformer.coord), take_first_chiral_subordering=take_first_chiral_subordering
        )

        # remap reference conformer to native index
        chirals[:, :4] = torch.tensor(_ref_to_conf_map)[chirals[:, :4].long()]

        # remove unmasked chirals
        mask = (chirals[:, :4] >= 0).all(dim=1)
        chirals = chirals[mask]

        # add atom offset
        chirals[:, :4] = chirals[:, :4] + res_start

        all_chirals.append(chirals)

    if all_chirals:
        return torch.cat(all_chirals, dim=0)
    else:
        # Return empty tensor with correct shape if no chiral centers found
        return torch.zeros((0, 5), dtype=torch.float32)


class AddAF3ChiralFeatures(Transform):
    """Adds chiral features into the feats dictionary.

    Adds the following features to the data dictionary under the 'feats' key:

        chiral_feats
            [N_chiral_centers, 5] A listing of chiral centers of the format:
            tensor([[ 5.,  1.,  2.,  3.,  0.61546...],...])
            Here, the first 4 columns define atom indices of chiral center; the 5th is target dihedral

    Metadata from GetRDKitChiralCenters, held in the "chiral_centers" key, is needed for this transform.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["GetRDKitChiralCenters"]

    def __init__(self, take_first_chiral_subordering: bool = True):
        super().__init__()
        self.take_first_chiral_subordering = take_first_chiral_subordering

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "chiral_centers", "rdkit"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        chiral_feats = add_af3_chiral_features(
            atom_array=data["atom_array"],
            chiral_centers=data["chiral_centers"],
            rdkit_mols=data["rdkit"],
            take_first_chiral_subordering=self.take_first_chiral_subordering,
        )

        # Split into chiral_centers and chiral_center_dihedral_angles
        chiral_centers = chiral_feats[:, :4].long()
        chiral_center_dihedral_angles = chiral_feats[:, -1].float()

        feats = data.setdefault("feats", {})
        feats.update(
            {
                "chiral_feats": chiral_feats,  # [n_chirals, 5]
                "chiral_centers": chiral_centers,  # (long) [n_chirals, 4]
                "chiral_center_dihedral_angles": chiral_center_dihedral_angles,  # (float) [n_chirals]
            }
        )

        return data
