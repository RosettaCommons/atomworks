"""
This module provides utility functions for working with OpenBabel and Biotite structures.

The functions in this module facilitate the conversion between OpenBabel and Biotite
representations of molecular structures, including atoms, bonds, and annotations. This
allows using OpenBabel for identifying e.g. stereochemistry, automorphisms, etc.

References:
    `OpenBabel documentation <https://open-babel.readthedocs.io/>`_
    `Biotite documentation <https://www.biotite-python.org/>`_
"""

import logging
from collections import Counter
from typing import Any, ClassVar

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from openbabel import openbabel, pybel

from atomworks.constants import ATOMIC_NUMBER_TO_ELEMENT, ELEMENT_NAME_TO_ATOMIC_NUMBER, UNKNOWN_LIGAND
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_does_not_contain_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger(__name__)

_BIOTITE_BOND_TYPE_TO_OPENBABEL = {
    # biotite bond type: (order, is_aromatic)
    struc.bonds.BondType.ANY: (1, False),
    struc.bonds.BondType.SINGLE: (1, False),
    struc.bonds.BondType.DOUBLE: (2, False),
    struc.bonds.BondType.TRIPLE: (3, False),
    struc.bonds.BondType.QUADRUPLE: (4, False),
    struc.bonds.BondType.AROMATIC_SINGLE: (1, True),
    struc.bonds.BondType.AROMATIC_DOUBLE: (2, True),
    struc.bonds.BondType.AROMATIC_TRIPLE: (3, True),
}
"""
Mapping from biotite bond type to openbabel bond order and aromaticity.
Unspecified bonds are interpreted as single bonds.

The mapping takes the form `biotite_bond_type -> (order, is_aromatic)`.
"""

_OPENBABEL_BOND_TYPE_TO_BIOTITE = {
    # (order, is_aromatic): biotite bond type
    (1, False): struc.bonds.BondType.SINGLE,
    (2, False): struc.bonds.BondType.DOUBLE,
    (3, False): struc.bonds.BondType.TRIPLE,
    (4, False): struc.bonds.BondType.QUADRUPLE,
    (1, True): struc.bonds.BondType.AROMATIC_SINGLE,
    (2, True): struc.bonds.BondType.AROMATIC_DOUBLE,
    (3, True): struc.bonds.BondType.AROMATIC_TRIPLE,
}
"""
Mapping from openbabel bond order and aromaticity to biotite bond type.

The mapping takes the form `(order, is_aromatic) -> biotite_bond_type`.
"""

_BIOTITE_DEFAULT_ANNOTATIONS = ["chain_id", "res_id", "res_name", "atom_name"]


_OBABEL_IMPLICIT_HYDROGEN_REF = 4294967294
"""
Openbabel's special atom ID that is used for implicit hydrogens
or lone pairs.

Reference:
    `OpenBabel Stereochemistry Documentation <https://open-babel.readthedocs.io/en/latest/Stereochemistry/stereo.html#accessing-stereochemistry-information>`_
"""


def atom_array_to_openbabel(
    atom_array: AtomArray,
    set_coords: bool = True,
    infer_aromaticity: bool = False,
    infer_hydrogens: bool = False,
    annotations_to_keep: list[str] = _BIOTITE_DEFAULT_ANNOTATIONS,
    ph_for_inferred_hydrogens: float = 7.4,
) -> openbabel.OBMol:
    """Convert a Biotite AtomArray to an OpenBabel OBMol with the option of keeping custom AtomArray annotations.

    For easier interfacing with the OBMol object, you can wrap it into a pybel.Molecule object.

    - https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html
    - https://github.com/openbabel/documentation/blob/master/pybel.py

    Args:
        atom_array: The Biotite AtomArray to convert.
        set_coords: If True, set the atomic coordinates from the AtomArray in the OBMol. Defaults to True.
        infer_aromaticity: If True, infer aromaticity in the OBMol or take the aromaticity annotations from the AtomArray. Defaults to False.
        infer_hydrogens: If True, infer hydrogens in the OBMol or take the hydrogens annotations from the AtomArray. Defaults to False.
        annotations_to_keep: List of annotation categories to keep from the AtomArray. Defaults to _BIOTITE_DEFAULT_ANNOTATIONS.
        ph_for_inferred_hydrogens: The pH value to use for inferred hydrogens. Defaults to pH 7.4 which is the openbabel default.
            The pH value is exposed here explicitly, but we recommend using the default value and only changing it if you have a good reason, as this will
            likely make it out of sync with other parts of the codebase which use the default pH value.

    Returns:
        The converted OpenBabel OBMol. The custom annotations are stored in the _annotations attribute.

    Example:
        .. code-block:: python

            from biotite.structure import AtomArray, BondType
            import numpy as np
            from atomworks.ml.transforms.openbabel_utils import atom_array_to_openbabel

            # Create AtomArray
            atom_array = AtomArray(5)
            atom_array.element = np.array(["C", "C", "O", "N", "H"])
            atom_array.coord = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [1.5, 1.5, 0.0],
                    [0.0, 1.5, 0.0],
                    [0.0, 0.0, 1.5],
                ]
            )
            # Add bonds
            atom_array.bonds = struc.BondList(len(atom_array))
            atom_array.bonds.add_bond(0, 1, BondType.SINGLE)
            atom_array.bonds.add_bond(1, 2, BondType.DOUBLE)
            atom_array.bonds.add_bond(1, 3, BondType.SINGLE)
            atom_array.bonds.add_bond(0, 4, BondType.SINGLE)
            # Convert to OpenBabel molecule
            obmol = atom_array_to_openbabel(atom_array)
            # Print number of atoms
            print(f"Number of atoms: {obmol.NumAtoms()}")
            # Number of atoms: 5
            # Print atom information
            print("\nAtom information:")
            for atom in openbabel.OBMolAtomIter(obmol):
                print(
                    f"Atomic number: {atom.GetAtomicNum()}, Coordinates: ({atom.GetX():.1f}, {atom.GetY():.1f}, {atom.GetZ():.1f})"
                )

            # Atom information:
            # Atomic number: 6, Coordinates: (0.0, 0.0, 0.0)
            # Atomic number: 6, Coordinates: (1.5, 0.0, 0.0)
            # Atomic number: 8, Coordinates: (1.5, 1.5, 0.0)
            # Atomic number: 7, Coordinates: (0.0, 1.5, 0.0)
            # Atomic number: 1, Coordinates: (0.0, 0.0, 1.5)
    """
    # Initialize empty OpenBabel molecule
    obmol = openbabel.OBMol()

    # Set atoms
    # ... keep track of openbabel internally assigned atom_ids to differentiate
    #     atoms that were originally in the atom array from atoms that might be
    #     added implicitly later, for example implicit hydrogens
    obmol_atom_ids = []
    _has_explicit_hydrogen = False
    for atom in atom_array:
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(ELEMENT_NAME_TO_ATOMIC_NUMBER[atom.element.upper()])
        if obatom.GetAtomicNum() == 1:
            _has_explicit_hydrogen = True

        if set_coords:
            # ... set coordinates
            obatom.SetVector(float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))

        obmol_atom_ids.append(obatom.GetId())

    # Attach custom atom-level annotations from the atom array
    obmol._annotations = {"openbabel_atom_id": np.array(obmol_atom_ids)}
    for annotation in annotations_to_keep:
        if annotation in atom_array.get_annotation_categories():
            obmol._annotations[annotation] = atom_array._annot[annotation]

    # Set bonds
    for start_atom_idx, end_atom_idx, bond_type in atom_array.bonds.as_array():
        if bond_type == struc.bonds.BondType.ANY:
            # ... warn if underspecified bonds are encountered
            logger.warning("Encountered BondType.ANY. Interpreting as single bond.")

        obatom_begin = obmol.GetAtom(int(start_atom_idx + 1))  # openbabel uses 1-based indexing
        obatom_end = obmol.GetAtom(int(end_atom_idx + 1))
        order, is_aromatic = _BIOTITE_BOND_TYPE_TO_OPENBABEL[bond_type]

        obbond = openbabel.OBBond()
        obbond.SetBegin(obatom_begin)
        obbond.SetEnd(obatom_end)
        obbond.SetBondOrder(order)
        if is_aromatic and not infer_aromaticity:
            obbond.SetAromatic()
            obatom_begin.SetAromatic()
            obatom_end.SetAromatic()
        obmol.AddBond(obbond)

    # ... set aromatic perception
    #  (if `SetAromaticPerceived()` is True, the aromatic annotations that are given are used)
    #  (if `SetAromaticPerceived()` is False, the aromatic annotation will try to infer when IsAromatic() is called
    obmol.SetAromaticPerceived(not infer_aromaticity)

    if infer_hydrogens:
        if _has_explicit_hydrogen:
            logger.warning(
                "Found explicit hydrogens. Correcting for PH will remove these in favor of implicit hydrogens!"
            )
        # this will remove explicit hydrogens and add implicit hydrogens inferred based on the valence model
        #  of openbabel
        obmol.CorrectForPH(float(ph_for_inferred_hydrogens))

    return obmol


def atom_array_from_openbabel(obmol: openbabel.OBMol) -> AtomArray:
    """Convert an OpenBabel OBMol object to a Biotite AtomArray object.

    This function takes an OpenBabel OBMol object and converts it into a Biotite AtomArray object,
    matching and preserving the (optional) atom-level annotations in `obmol._annotations` and bond information.

    Args:
        obmol (openbabel.OBMol): The OpenBabel OBMol object to be converted.

    Returns:
        AtomArray: A Biotite AtomArray object containing the atoms and bonds from the input OBMol object.

    Raises:
        ValueError: If the input OBMol object is invalid or cannot be processed.

    Example:
        >>> obmol = pybel.readstring("smi", "CCO").OBMol
        >>> atom_array = atom_array_from_openbabel(obmol)
        >>> print(atom_array)
        AtomArray([Atom(element='6', coord=array([0.0, 0.0, 0.0]), ...)])
    """
    # Set atoms
    atoms = []
    element_counter = Counter()
    for obatom in openbabel.OBMolAtomIter(obmol):
        element_occurence = element_counter[obatom.GetAtomicNum()]
        element_counter[obatom.GetAtomicNum()] += 1
        atoms.append(
            struc.Atom(
                element=ATOMIC_NUMBER_TO_ELEMENT[obatom.GetAtomicNum()],
                atomic_number=obatom.GetAtomicNum(),
                coord=np.array([obatom.GetX(), obatom.GetY(), obatom.GetZ()])
                if obatom.GetVector() is not None
                else np.full(3, np.nan),
                res_name=UNKNOWN_LIGAND,  # default to UNL (unknown ligand) residue, unless overridden by custom annotation later
                hetero=True,  # default to hetero atom, unless overridden by custom annotation later
                atom_name=f"{ATOMIC_NUMBER_TO_ELEMENT[obatom.GetAtomicNum()].upper()}{element_occurence}",
                charge=obatom.GetFormalCharge(),  # formal charge
                hyb=obatom.GetHyb(),  # hybridization state
                is_metal=obatom.IsMetal(),  # whether the atom is a metal
                nhyd=obatom.GetTotalDegree()
                - obatom.GetHvyDegree(),  # number of bonded hydrogens (implicit or explicit)
                total_deg=obatom.GetTotalDegree(),  # total bond count including multiplicities
                hvydeg=obatom.GetHvyDegree(),  # bond count of heavy atoms only, including multiplicities
                n_implicit_hyd=obatom.GetImplicitHCount(),  # number of implicit hydrogens
                openbabel_atom_id=obatom.GetId(),
            )
        )
    atom_array = struc.array(atoms)

    # Set bonds
    bonds = []
    _explicit_hydrogen_counts = np.zeros(len(atoms), dtype=np.int8)
    for obbond in openbabel.OBMolBondIter(obmol):
        obatom_begin = obbond.GetBeginAtom()
        obatom_end = obbond.GetEndAtom()
        start_atom_idx = obatom_begin.GetIndex()
        end_atom_idx = obatom_end.GetIndex()
        order = obbond.GetBondOrder()
        is_aromatic = obbond.IsAromatic()

        # ... count explicit hydrogens
        if obatom_begin.GetAtomicNum() == 1:
            _explicit_hydrogen_counts[end_atom_idx] += order
        if obatom_end.GetAtomicNum() == 1:
            _explicit_hydrogen_counts[start_atom_idx] += order

        bonds.append((start_atom_idx, end_atom_idx, _OPENBABEL_BOND_TYPE_TO_BIOTITE[(order, is_aromatic)]))
    # ... transform bonds into a biotite BondList
    atom_array.bonds = struc.BondList(len(atoms), np.array(bonds))

    # Set the `nhyd` annotation for the case of explicit hydrogens
    atom_array.set_annotation("n_explicit_hyd", _explicit_hydrogen_counts)

    # Set extra annotations
    annotations = obmol._annotations if hasattr(obmol, "_annotations") else {}
    if len(annotations) > 0:
        # Create mapping of array idx <> annotation idx via the openbabel atom id:
        _openbabel_id_to_annotation_idx = {
            openbabel_atom_id: idx for idx, openbabel_atom_id in enumerate(annotations["openbabel_atom_id"])
        }

        array_idx_to_annotation_idx = []
        for idx, atom in enumerate(atoms):
            if atom.openbabel_atom_id in _openbabel_id_to_annotation_idx:
                array_idx_to_annotation_idx.append((idx, _openbabel_id_to_annotation_idx[atom.openbabel_atom_id]))
        array_idx_to_annotation_idx = np.array(array_idx_to_annotation_idx)

        for key, val in annotations.items():
            if key in ["coord", "charge", "hyb", "is_metal", "nhyd", "hvydeg"]:
                logger.warning(f"Found built-in annotation: {key} as custom annotation. Skipping.")
                continue

            if np.issubdtype(val.dtype, np.integer):
                if np.issubdtype(val.dtype, np.unsignedinteger):
                    # Use max value for unsigned integers since they can't hold -1
                    default = np.iinfo(val.dtype).max
                else:
                    default = -1
            elif np.issubdtype(val.dtype, np.floating):
                default = np.nan
            elif np.issubdtype(val.dtype, np.str_):
                default = ""
            else:
                logger.warning(f"Unsupported annotation dtype: {val.dtype}. Skipping.")
                continue

            vals = np.full(len(atoms), default, dtype=val.dtype)
            vals[array_idx_to_annotation_idx[:, 0]] = annotations[key][array_idx_to_annotation_idx[:, 1]]
            atom_array.set_annotation(key, vals)

    return atom_array


def get_chiral_centers(obmol: openbabel.OBMol) -> list[int]:
    """Identify and return the indices of chiral centers in an OpenBabel OBMol object.

    This function iterates over all atoms in the given OBMol object and identifies those
    that are tetrahedral stereo centers (chiral centers). For each chiral center, it records
    the index of the chiral center atom and the indices of the atoms bonded to it, excluding
    implicit hydrogens.

    Args:
        obmol (openbabel.OBMol): The OpenBabel OBMol object to analyze.

    Returns:
        list[int]: A list of dictionaries, where each dictionary contains:
            - "chiral_center_idx" (int): The index of the chiral center atom.
            - "bonded_explicit_atom_idxs" (list[int]): A list of indices of the atoms bonded to the chiral center,
              excluding implicit hydrogens.
    """
    stereo_facade = openbabel.OBStereoFacade(obmol)

    # iterate over all tetrahedral stereo centers and record the plane pairs that define the tetrahedral side
    chiral_centers = []
    for atom_idx in range(obmol.NumAtoms()):
        if not stereo_facade.HasTetrahedralStereo(atom_idx):
            # skip if the atom is not a tetrahedral stereo center
            continue

        # get chiral information
        stereo_data = stereo_facade.GetTetrahedralStereo(atom_idx).GetConfig()
        # get the chiral center
        chiral_center_idx = stereo_data.center  # (int)
        # get the 4 bonded atoms to the chiral center
        bonded_atom_idx = [stereo_data.from_or_towards, *stereo_data.refs]  # [4] (int)
        # reduce the bonded atoms to those that are not implicit hydrogens (at most 1 implicit hydrogen is possible)
        bonded_explicit_atom_idxs = [atom for atom in bonded_atom_idx if atom != _OBABEL_IMPLICIT_HYDROGEN_REF]

        chiral_centers.append(
            {"chiral_center_idx": chiral_center_idx, "bonded_explicit_atom_idxs": bonded_explicit_atom_idxs}
        )

    return chiral_centers


def smiles_to_openbabel(smiles: str) -> openbabel.OBMol:
    """
    Convert a SMILES string to an OpenBabel OBMol object.

    Example:
        >>> smiles = "CCO"
        >>> obmol = smiles_to_openbabel(smiles)
        >>> print(obmol.NumAtoms())
        3

    Note:
        This function uses the Pybel module to read the SMILES string and convert it to an OBMol object.
    """
    mol = pybel.readstring("smi", smiles)
    return mol.OBMol


def find_automorphisms(obmol: openbabel.OBMol, max_automorphs: int = 1000, max_mem: int = 300 * (2**20)) -> np.ndarray:
    """
    Find automorphisms of a given Open Babel molecule.

    This function identifies the automorphisms (symmetry-related atom swaps) of the input molecule
    and returns them as a numpy array. If the search for automorphisms fails, it returns a single
    automorphism representing the identity (no swaps).

    Args:
        obmol (openbabel.OBMol): The Open Babel molecule for which to find automorphisms.
        max_automorphs (int): The maximum number of automorphisms to return. These are deterministically
            set to be the first `max_automorphs` automorphisms found by OpenBabel.
            For model training it is recommended to deterministically select the automorphisms
            to be used (as done in this transform) as a model might otherwise be nudged towards a specific
            automorph in one training step, but that automorph then does not show up in the next training
            step, leading to a moving target problem.
        max_mem (int): The maximum memory to use for the automorphism search, in bytes.
            Default is 300 MB, which is also the default value used by OpenBabel.

    Returns:
        automorphs (np.ndarray): A numpy array of shape [n_automorphs, n_atoms, 2], where each element
            represents an automorphism as list of paired atom indices (from_idx, to_idx).
            If the search fails (e.g. due to running out of memory), returns an array with
            a single automorphism representing the identity (no swaps).

    References:
        `OpenBabel Substructure API <https://openbabel.org/api/3.0/group__substructure.shtml#ga16841a730cf92c8e51a804ad8d746307>`_
        `Automorphisms and Symmetry Blog <https://baoilleach.blogspot.com/2010/11/automorphisms-isomorphisms-symmetry.html>`_

    Example:
        >>> from openbabel import pybel
        >>> mol = pybel.readstring("smi", "c1c(O)cccc1(O)").OBMol
        >>> automorphisms = find_automorphisms(mol)
        >>> print(automorphisms)
            [[[0 0]
              [1 1]
              [2 2]
              [3 3]
              [4 4]
              [5 5]
              [6 6]
              [7 7]]

             [[0 0]
              [1 6]
              [2 7]
              [3 5]
              [4 4]
              [5 3]
              [6 1]
              [7 2]]]
    """
    n_atoms = obmol.NumAtoms()
    assert (
        n_atoms == obmol.NumHvyAtoms()
    ), f"Found {n_atoms - obmol.NumHvyAtoms()} explicit hydrogens. This function assumes that the input molecule has no explicit hydrogens. Please remove."

    # ... initialize a vector container to store automorphs
    automorphs = openbabel.vvpairUIntUInt()  # vector<vector<pair<unsigned int, unsigned int>>>

    # ... create mask to ignore certain atoms -- this could be used to ignore explicit hydrogens, however we choose not to
    #  allow those in the first place since they mess up the indexing of the automorphs and are not of interest for modelling.
    #  The only reason we keep the `is_included` vector is to allow for the dynamic dispatch function to the signature that
    #  exposes the `max_mem` arugment, which is not available in the other signature (c.f. openbabel reference in docstring.)
    is_atom_included = openbabel.OBBitVec(n_atoms)  # vector<bool> , values are initialized to 0
    # (! WARNING: is_atom_included.Negate() does not work properly to set all bits to 1 in openbabel, for example
    #  `1I6` in the CCD does not work with that method.)
    is_atom_included.SetRangeOn(0, n_atoms)  # set all bits to 1

    # ... populate the automorphs
    _successfully_searched_automorphs = openbabel.FindAutomorphisms(obmol, automorphs, is_atom_included, max_mem)

    # ... extract the automorphs as numpy array
    if not _successfully_searched_automorphs or automorphs.size == 0:
        # ... return only a single automorphism (the identity) if the search failed (e.g. OOM)
        logger.warning("Automorphism search failed. Returning identity automorphism.")
        return np.arange(n_atoms).reshape(1, -1, 1).repeat(2, axis=2)  # [1, n_atoms, 2]

    # ... or return all the found automorphisms if the search was successful
    automorphs = np.array(automorphs)  # [n_automorphs, n_atoms, 2]
    if len(automorphs) > max_automorphs:
        logger.info(f"Found {len(automorphs)} automorphisms, truncating to first {max_automorphs}.")
        return automorphs[:max_automorphs]  # [n_automorphs, n_atoms, 2]
    return automorphs  # [n_automorphs, n_atoms, 2]


class AddOpenBabelMoleculesForAtomizedMolecules(Transform):
    """
    Add OpenBabel molecules for atomized molecules in the atom array.

    This transform converts atomized molecules in the atom array to OpenBabel OBMol objects and stores them in the
    `data` dictionary under the "openbabel" key. Each molecule is identified by the first global `atom_id` contained
    in this molecule.

    Note:
        This transform requires the `AtomizeByCCDName` transform to be applied previously.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including the atom array.

    Returns:
        dict[str, Any]: The updated `data` dictionary with the added OpenBabel molecules under the
            `"openbabel"` key.

    Example:
        >>> data = {
        >>>     "atom_array": AtomArray(...),  # Your atom array here
        >>> }
        >>> transform = AddOpenBabelMoleculesForAtomizedMolecules()
        >>> data = transform(data)
        >>> print(data["openbabel"])
        {
            0: <openbabel.OBMol object at 0x...>,
            1: <openbabel.OBMol object at 0x...>,
            ...
        }
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "AddRF2AAChiralFeatures",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
    ]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_does_not_contain_keys(data, ["openbabel"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_annotation(data, ["atomize", "atom_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        # Subset to atomized molecules
        #  NOTE: This will subset the atom array to all parts of the array that will be
        #   atomized and the bonds within these parts. Bonds to the non-atomized parts of the
        #   atom array are removed when we subset in this way. Concretely, for covalent modifications
        #   in proteins, the atomized residues would lose the N-C and C-N bonds to the rest of the polymer
        #   at the N-termnal and C-terminal ends of the atomized residues for a protein.
        _atom_array = atom_array[atom_array.atomize]

        # Iterate over the molecules (covalently bonded components in the atom_array)
        data["openbabel"] = {}
        for molecule in struc.molecule_iter(_atom_array):
            # Use the first global `atom_id` as a unique identifier for the molecule
            molecule_id = molecule.atom_id[0]

            # Convert to openbabel
            obmol = atom_array_to_openbabel(
                molecule,
                infer_hydrogens=False,
                infer_aromaticity=False,
                annotations_to_keep=["chain_id", "res_id", "res_name", "atom_name", "atom_id"],
            )
            data["openbabel"][molecule_id] = obmol

        return data


class GetChiralCentersFromOpenBabel(Transform):
    """Identify chiral centers in the OpenBabel molecules stored in the data["openbabel"] dictionary.

    These molecules typically correspond to the atomized molecules in the data["atom_array"] (c.f.
    AddOpenBabelMoleculesForAtomizedMolecules).

    Chiral centers are mapped to the global atom IDs in the atom array to enable tracking chiral
    centers regardless of cropping or reshuffling operations that may modify the atom_array.

    Args:
        data: A dictionary containing the input data, including the atom array and
            OpenBabel molecules under the data["openbabel"] key.

    Returns:
        The updated data dictionary with the identified chiral centers under the
            "chiral_centers" key. The chiral centers are stored as a list of dictionaries, where each
            dictionary contains the chiral center global atom ID and the atom IDs of the (3 to 4) atoms
            bonded to it.

    Example:
        .. code-block:: python

            data = {
                "atom_array": atom_array,
                "openbabel": {
                    1: obmol1,
                    2: obmol2,
                },
            }

            transform = GetChiralCentersFromOpenBabel()
            result = transform.forward(data)

            print(result["chiral_centers"])
            # Output might look like:
            # [
            #     {
            #         "chiral_center_atom_id": 5,
            #         "bonded_explicit_atom_ids": [1, 2, 3, 4]
            #     },
            #     {
            #         "chiral_center_atom_id": 10,
            #         "bonded_explicit_atom_ids": [6, 7, 8, 9]
            #     }
            # ]
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AddOpenBabelMoleculesForAtomizedMolecules",
        "AtomizeByCCDName",
    ]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AddRF2AAChiralFeatures"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "openbabel"])
        check_does_not_contain_keys(data, ["chiral_centers"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_annotation(data, ["atomize", "atom_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # Iterate over the molecules (covalently bonded components in the atom_array)
        data["chiral_centers"] = []
        for obmol in data["openbabel"].values():
            # Get the chiral centers (returned are the indices of the chiral center atoms
            #  within the `obmol` object)
            _obmol_chirals = get_chiral_centers(obmol)

            # Map the chiral center indices within the `obmol` to the atom_id
            molecule = atom_array_from_openbabel(obmol)

            for _chiral_center in _obmol_chirals:
                data["chiral_centers"].append(
                    {
                        # We convert from the `obmol`-internal indices to the global `atom_id` for
                        #  robustness to corretly map the chiral center indices in the global atom
                        #  array later in case of arbitrary cropping or reshuffling operations
                        "chiral_center_atom_id": molecule.atom_id[_chiral_center["chiral_center_idx"]],
                        "bonded_explicit_atom_ids": molecule.atom_id[_chiral_center["bonded_explicit_atom_idxs"]],
                    }
                )

        return data
