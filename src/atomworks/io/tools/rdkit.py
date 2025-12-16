"""Tools for using RDKit with AtomArray objects."""

import copy
import io
import logging
from collections import Counter
from collections.abc import Callable
from functools import cache, wraps
from os import PathLike
from pathlib import Path
from typing import Final, Literal

import biotite.structure as struc
import numpy as np
import toolz
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import ExplicitBitVect

import atomworks.io.transforms.atom_array as ta
from atomworks.common import exists, immutable_lru_cache, not_isin
from atomworks.constants import (
    BIOTITE_DEFAULT_ANNOTATIONS,
    CCD_MIRROR_PATH,
    HYDROGEN_LIKE_SYMBOLS,
    METAL_ELEMENTS,
    PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL,
    UNKNOWN_LIGAND,
)
from atomworks.io.utils.ccd import atom_array_from_ccd_code

logger = logging.getLogger(__name__)


# Set default pickle properties to all properties, otherwise
#  annotations get lost when pickling/unpickling molecules
# https://github.com/rdkit/rdkit/issues/1320
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

RDKIT_HYBRIDIZATION_TO_INT: Final[dict[Chem.rdchem.HybridizationType, int]] = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP2D: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
    Chem.rdchem.HybridizationType.OTHER: 7,
    Chem.rdchem.HybridizationType.UNSPECIFIED: -1,
}
"""
Mapping from RDKit hybridization types to integers.

Reference:
    `RDKit Atom Documentation <https://www.rdkit.org/docs/cppapi/classRDKit_1_1Atom.html#a58e40e30db6b42826243163175cac976>`_
"""

RDKIT_BOND_TYPE_TO_BIOTITE: Final[dict[tuple[Chem.BondType, bool], struc.bonds.BondType]] = {
    # (rdkit bond type, is_aromatic) -> biotite bond type
    (Chem.BondType.UNSPECIFIED, False): struc.bonds.BondType.ANY,
    (Chem.BondType.SINGLE, False): struc.bonds.BondType.SINGLE,
    (Chem.BondType.DOUBLE, False): struc.bonds.BondType.DOUBLE,
    (Chem.BondType.TRIPLE, False): struc.bonds.BondType.TRIPLE,
    (Chem.BondType.QUADRUPLE, False): struc.bonds.BondType.QUADRUPLE,
    (Chem.BondType.DATIVE, False): struc.bonds.BondType.COORDINATION,
    (Chem.BondType.SINGLE, True): struc.bonds.BondType.AROMATIC_SINGLE,
    (Chem.BondType.DOUBLE, True): struc.bonds.BondType.AROMATIC_DOUBLE,
    (Chem.BondType.TRIPLE, True): struc.bonds.BondType.AROMATIC_TRIPLE,
}
"""
Mapping from RDKit bond types to Biotite bond types.
Maps (rdkit bond type, is_aromatic) -> biotite bond type

Unspecified bonds are mapped to `ANY` bond type.
"""

BIOTITE_BOND_TYPE_TO_RDKIT: Final[dict[struc.bonds.BondType, tuple[Chem.BondType, bool]]] = {
    # biotite bond type -> (rdkit bond type, is_aromatic)
    struc.bonds.BondType.ANY: (Chem.BondType.UNSPECIFIED, False),
    struc.bonds.BondType.SINGLE: (Chem.BondType.SINGLE, False),
    struc.bonds.BondType.DOUBLE: (Chem.BondType.DOUBLE, False),
    struc.bonds.BondType.TRIPLE: (Chem.BondType.TRIPLE, False),
    struc.bonds.BondType.QUADRUPLE: (Chem.BondType.QUADRUPLE, False),
    struc.bonds.BondType.COORDINATION: (Chem.BondType.DATIVE, False),
    # NOTE: We map aromatics to single/double/triple instead of Chem.BondType.AROMATIC
    #       because the PDB specified bond-order (from a kekulized form of the molecule)
    #       is lost when we map to aromatic, which can lead to incorrect bond-order
    #       perception in RDKit.
    struc.bonds.BondType.AROMATIC_SINGLE: (Chem.BondType.SINGLE, True),
    struc.bonds.BondType.AROMATIC_DOUBLE: (Chem.BondType.DOUBLE, True),
    struc.bonds.BondType.AROMATIC_TRIPLE: (Chem.BondType.TRIPLE, True),
}
"""
Mapping from Biotite bond types to RDKit bond types.

Maps (biotite bond type) -> (rdkit bond type, is_aromatic)
"""

CONVERTIBLE_RDKIT_BOND_TYPES: Final[frozenset[Chem.BondType]] = frozenset(
    toolz.keymap(lambda x: x[0], RDKIT_BOND_TYPE_TO_BIOTITE)
)
"""Set of RDKit Bond types that can be converted to Biotite bond types."""


class ChEMBLNormalizer:
    """
    Normalize an RDKit molecule like the ChEMBL structure pipeline does.
    This is useful for `rescuing` molecules that failed to be sanitized by RDKit
    alone.

    Reference:
        `ChEMBL Structure Pipeline <https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py#L33C1-L73C15>`_
    """

    def __init__(self):
        with open(Path(__file__).parent / "chembl_transformations.smirks") as f:
            self._normalization_transforms = f.read()
        self._normalizer_params = rdMolStandardize.CleanupParameters()
        self._normalizer = rdMolStandardize.NormalizerFromData(
            paramData=self._normalization_transforms, params=self._normalizer_params
        )

    def normalize_in_place(self, mol: Mol) -> Mol:
        self._normalizer.normalizeInPlace(mol)
        return mol


@cache
def get_valence_checker() -> rdMolStandardize.RDKitValidation:
    """Cached RDKit valence checker."""
    return rdMolStandardize.RDKitValidation()


@cache
def get_chembl_normalizer() -> rdMolStandardize.Normalizer:
    """Cached ChEMBL normalizer."""
    return ChEMBLNormalizer()


@cache
def element_to_atomic_number(element: str) -> int:
    """
    Convert an element string or atomic number to an atomic number.

    Args:
        - element (str): The element symbol (e.g., 'C') or atomic number.

    Returns:
        - int: The atomic number of the element.

    Examples:
        >>> element_to_atomic_number("C")
        6
        >>> element_to_atomic_number("8")
        8
        >>> element_to_atomic_number(1)
        1
    """
    element = PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL.get(element, element)
    return Chem.GetPeriodicTable().GetAtomicNumber(element.capitalize())


def preserve_annotations(func: Callable[[Mol, ...], Mol]) -> Callable[[Mol, ...], Mol]:
    """
    Decorator to copy annotations from an RDKit molecule to a new molecule.

    This decorator ensures that any custom annotations stored in the `_annotations`
    attribute of an RDKit molecule are preserved when the molecule is modified or
    converted.

    Args:
        - func (callable): The function to be decorated. Must accept an RDKit molecule as
          positional argument or keyword argument with keyword 'mol'.

    Returns:
        callable: The decorated function that preserves annotations.
    """

    @wraps(func)
    def wrapped(*args, **kwargs) -> Mol:
        # Find the first RDKit molecule in the arguments or keyword arguments
        if "mol" in kwargs:
            mol = kwargs["mol"]
        else:
            mol = next(arg for arg in args if isinstance(arg, Mol))

        if hasattr(mol, "_annotations"):
            annotations = mol._annotations
            new_mol = func(*args, **kwargs)
            new_mol._annotations = annotations
        else:
            new_mol = func(*args, **kwargs)

        return new_mol

    return wrapped


def _has_correct_valence(mol: Mol) -> bool:
    """
    Check if an RDKit molecule has correct valences.
    """
    mol.UpdatePropertyCache(strict=False)
    return len(get_valence_checker().validate(mol)) == 0


def _calc_formal_charge_from_valence(rdatom: Chem.Atom) -> int:
    """
    Calculate the formal charge of an atom from its valence.
    """
    num_valence_electrons = Chem.GetPeriodicTable().GetDefaultValence(
        rdatom.GetSymbol()
    )  # ... how many electrons are missing to full outer shell
    num_electrons_in_bonds = rdatom.GetTotalValence()  # Total valence (explicit + implicit)
    num_radicals = rdatom.GetNumRadicalElectrons()  # ... how many unpaired, radical electrons
    return (num_electrons_in_bonds + num_radicals) - num_valence_electrons


def fix_charge_based_on_valence(mol: Mol) -> Mol:
    """
    Attempt to fix the formal charge of an RDKit molecule by making it compatible with its valence state.
    """
    # ... record previous mol to revert if changing charges does not fix the valence
    previous_mol = copy.deepcopy(mol)

    if not _has_correct_valence(mol):
        for rdatom in mol.GetAtoms():
            rdatom.SetFormalCharge(_calc_formal_charge_from_valence(rdatom))

    return mol if _has_correct_valence(mol) else previous_mol


def change_metal_bonds_to_dative(
    mol: Mol, *, qualifying_bond_types: set[Chem.BondType] = {Chem.BondType.SINGLE}
) -> Mol:
    """
    Change all qualifying bonds to a metal to be dative bonds (coordination bonds).
    This is useful since most bonds between metals and organic atoms are dative.

    Args:
        - mol (Mol): The input RDKit molecule
        - qualifying_bond_types (set[Chem.BondType]): The bond types to qualify as dative bonds.
            By default, only single bonds qualify for conversion.

    Returns:
        Mol: The molecule with metal bonds converted to dative bonds
    """
    if not isinstance(mol, Chem.RWMol):
        # ... create a writable copy of the molecule if not writable in-place
        mol = Chem.RWMol(mol)

    metal_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in METAL_ELEMENTS]

    for metal_idx in metal_indices:
        for bond in mol.GetAtomWithIdx(metal_idx).GetBonds():
            if bond.GetBondType() in qualifying_bond_types:
                other_idx = bond.GetOtherAtomIdx(metal_idx)
                mol.RemoveBond(metal_idx, other_idx)
                mol.AddBond(other_idx, metal_idx, Chem.BondType.DATIVE)

    return mol


@preserve_annotations
def fix_mol(
    mol: Mol,
    *,
    attempt_fix_by_calculating_charge_from_valence: bool = True,
    attempt_fix_by_normalizing_like_chembl: bool = True,
    attempt_fix_by_normalizing_like_rdkit: bool = True,
    in_place: bool = True,
    raise_on_failure: bool = True,
) -> Mol:
    """Fix an RDKit molecule (in-place).

    This function attempts to infer aromaticity, valences, implicit hydrogens, and
    formal charges to result in a molecule that can be successfully sanitized. It
    does **not** change the heavy atoms or bonds in the molecule.

    # TODO:
    #  - Add sanitization for aromatic systems with incorrect formal charges (that cannot be kekulized) (https://github.com/datamol-io/datamol/issues/231)
    #    - This may be done via Hueckel's rule (https://en.wikipedia.org/wiki/H%C3%BCckel%27s_rule): Find all aromatic systems, compute Hueckel's rule,
    #      if the number of pi electrons is not equal to 4*n+2, where n is an integer, then the aromatic system is not valid with the given formal charges.
    #      Try to adjust the formal charges on non-carbon atoms to make the aromatic system valid.
    #  - Add sanifix4 style sanitization (https://github.com/datamol-io/datamol/blob/0312388b956e2b4eeb72d791167cfdb873c7beab/datamol/_sanifix4.py#L114)
    #  - Add attempt to fix valences by changing the bond orders (c.f. https://github.com/datamol-io/datamol)
    #  - Add further ChEMBL style sanitization: https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py#L33C1-L73C15


    References:
        `RDKit Molecular Sanitization <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_
        `ChEMBL Structure Pipeline <https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py>`_
        `datamol mol.py <https://github.com/datamol-io/datamol/blob/0312388b956e2b4eeb72d791167cfdb873c7beab/datamol/mol.py>`_

    """
    if not in_place:
        mol = copy.deepcopy(mol)

    # ... try to fixe the molecule by automatically performing some standardization
    #  steps to infer formal charges and perceive aromaticity
    sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)

    if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
        # ... do not fix molecules that are not broken
        return mol

    logger.warning(
        f"Molecule failed sanitization: {sanitize_result}. Attempting to fix by inferring valences and aromaticity."
    )

    # ... recompute current valences
    mol.UpdatePropertyCache(strict=False)

    if attempt_fix_by_normalizing_like_chembl:
        # ... perform normalization steps recommended by the ChEMBL team to "rescue"
        get_chembl_normalizer().normalize_in_place(mol)
        # ... recompute valences after attempted fixing
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_by_normalizing_like_rdkit:
        # ... perform normalization steps recommended by the RDKit team.
        rdMolStandardize.NormalizeInPlace(mol)
        # ... recompute valences after attempted fixing
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_by_calculating_charge_from_valence:
        fix_charge_based_on_valence(mol)
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
        logger.warning(f"Could not fix molecule, final sanitization result: {sanitize_result}")
        if raise_on_failure:
            raise Chem.MolSanitizeException(f"Molecule failed sanitization: {sanitize_result}")

    return mol


@preserve_annotations
def add_hydrogens(mol: Mol, add_coords: bool = True) -> Mol:
    """
    Add explicit hydrogens to an RDKit molecule.
    """
    return Chem.AddHs(mol, addCoords=add_coords)


@preserve_annotations
def remove_hydrogens(mol: Mol) -> Mol:
    """
    Remove explicit hydrogens from an RDKit molecule.
    """
    return Chem.RemoveHs(mol)


def get_morgan_fingerprint_from_rdkit_mol(mol: Chem.Mol, *, radius: int = 2, n_bits: int = 2048) -> ExplicitBitVect:
    """
    Generates the Morgan fingerprint for an RDKit molecule. Useful for calculating Tanimoto
    similarity between molecules, e.g. for similarity searches.

    Default parameters are based on the AF-3 supplementary material:
        > We measure ligand Tanimoto similarity using RDKit v.2023_03_3 Morgan fingerprints (radius 2, 2048 bits)

    Args:
        - mol (Chem.Mol): The RDKit molecule to generate a fingerprint for.
        - radius (int): The radius of the fingerprint. Default is 2.
        - n_bits (int): The number of bits in the fingerprint. Default is 2048.

    Returns:
        - ExplicitBitVect: The Morgan fingerprint for the input molecule.

    References:
        AF-3 Supplement
        `RDKit Fingerprint Generator Tutorial <https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html>`_
    """
    morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = morgan_fingerprint_generator.GetFingerprint(mol)
    return fingerprint


def smiles_to_rdkit(smiles: str, *, sanitize: bool = True, timeout: int = 5, generate_conformers: bool = True) -> Mol:
    """
    Generate an RDKit molecule from a SMILES string.

    This function creates an RDKit molecule object from a SMILES string,
    performs sanitization, and perceives aromaticity.

    Args:
        - smiles (str): The SMILES string representing the molecule.
        - sanitize (bool): Whether to sanitize the molecule.
        - timeout (int): The timeout for the conformer generation.
        - generate_conformers (bool): Whether to generate and minimize conformers.
            If False, returns the molecule immediately after SMILES parsing.

    Returns:
        - rdkit.Chem.Mol: The RDKit molecule generated from the SMILES string.

    Note:
        The returned molecule is sanitized and has aromaticity perceived.
        If generate_conformers=True, the molecule will also have (implicit) hydrogens added
        and conformers generated with UFF minimization.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise Chem.MolSanitizeException(
            f"Failed to create molecule from SMILES string: {smiles}. Try setting `sanitize=False`."
        )

    if not generate_conformers:
        return mol

    # Conformer generation parameters
    _optimizer_force_tol = 1e-3
    _max_its = 500
    _energy_tol = 1e-7

    # ... add hydrogens (needed for accurate conformer generation)
    mol = Chem.AddHs(mol)

    # ... generate a conformer to keep the stereochemistry encoded in the SMILES
    # (We later re-generate a conformer; however, we need coordinates to ensure we preserve the stereochemistry)
    etkdg = AllChem.ETKDGv3()
    etkdg.useRandomCoords = True
    etkdg.optimizerForceTol = float(_optimizer_force_tol)
    etkdg.timeout = timeout
    AllChem.EmbedMolecule(mol, params=etkdg)

    if mol.GetNumConformers() > 0:
        # (Extra step to ensure we get the best conformer)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        ff.Initialize()
        ff.Minimize(energyTol=_energy_tol, maxIts=_max_its)

    # ... remove hydrogens again, since we no longer need them
    mol = remove_hydrogens(mol)

    return mol


def sdf_to_rdkit(sdf_path_or_buffer: io.StringIO | PathLike, *, sanitize: bool = True) -> Mol:
    """
    Generate an RDKit molecule from an SDF file or buffer.

    Args:
        - sdf_path_or_buffer (io.StringIO | PathLike): Either a path to an SDF file or a StringIO buffer containing
            SDF-formatted molecule data
        - sanitize (bool): Whether to sanitize the molecule during parsing. Default is True.

    Returns:
        - Mol: The RDKit molecule generated from the SDF data

    Raises:
        - ValueError: If no valid molecule could be parsed from the input
        - TypeError: If the input is neither a StringIO buffer nor a valid path
    """
    if isinstance(sdf_path_or_buffer, str | PathLike):
        supplier = Chem.SDMolSupplier(str(sdf_path_or_buffer), sanitize=sanitize)
    elif isinstance(sdf_path_or_buffer, io.StringIO):
        supplier = Chem.SDMolSupplier(sdf_path_or_buffer, sanitize=sanitize)
    else:
        raise TypeError("Input must be either a path or a StringIO buffer")

    try:
        mol = next(supplier)
    except StopIteration:
        raise ValueError("No valid molecule found in SDF input") from None

    if mol is None:
        raise ValueError("Failed to parse molecule from SDF input") from None

    return mol


def atom_array_from_rdkit(
    mol: Mol,
    *,
    set_coord_if_available: bool = True,
    conformer_id: int | None = None,
    remove_hydrogens: bool = True,
    remove_inferred_atoms: bool = False,
) -> AtomArray:
    """Convert an RDKit molecule to a Biotite AtomArray object.

    This function takes an RDKit Mol object and converts it into a Biotite AtomArray object,
    matching and preserving the (optional) atom-level annotations in `mol._annotations` and
    bond information.

    Args:
        - mol: The RDKit molecule to convert.
        - set_coord_if_available: Whether to set the coordinates from the RDKit molecule if
            a conformer is available.
        - conformer_id: The conformer ID to use for coordinates. If None, the first
          conformer is used.
        - remove_hydrogens: Whether to remove any explicit hydrogen atoms.
        - remove_inferred_atoms: Whether to remove any atoms that do not carry the `rdkit_atom_id` annotation.

    Returns:
        - An AtomArray containing the atoms and bonds from the input Mol object.

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> atom_array = atom_array_from_rdkit(mol)
        >>> print(atom_array)
        AtomArray([Atom(element='6', coord=array([0.0, 0.0, 0.0]), ...)])
    """
    mol = copy.deepcopy(mol)

    n_atoms = mol.GetNumAtoms()

    # Get coordinates (if available)
    coords = np.full((mol.GetNumAtoms(), 3), np.nan)
    n_conformers = mol.GetNumConformers()
    if set_coord_if_available and n_conformers > 0:
        conformer_id = conformer_id if conformer_id is not None else -1
        if conformer_id >= n_conformers:
            raise ValueError(f"Conformer ID {conformer_id} out of range for molecule with {n_conformers} conformers")
        coords = mol.GetConformer(conformer_id).GetPositions()

    # Set atoms
    atoms = []
    element_counter = Counter()
    for idx, rdatom in enumerate(mol.GetAtoms()):
        element_occurence = element_counter[rdatom.GetAtomicNum()]
        element_counter[rdatom.GetAtomicNum()] += 1
        atoms.append(
            struc.Atom(
                element=rdatom.GetSymbol(),
                atomic_number=rdatom.GetAtomicNum(),
                coord=coords[idx],
                charge=rdatom.GetFormalCharge(),
                hyb=RDKIT_HYBRIDIZATION_TO_INT[rdatom.GetHybridization()],
                nhyd=rdatom.GetTotalNumHs(),
                hvydeg=rdatom.GetDegree() - rdatom.GetTotalNumHs(),
                rdkit_atom_id=rdatom.GetIntProp("rdkit_atom_id") if rdatom.HasProp("rdkit_atom_id") else -1,
                hetero=True,  # per default, set all atoms to be hetero atoms
                atom_name=f"{rdatom.GetSymbol().upper()}{element_occurence}",  # per default, set atom name to be element symbol + index
                res_name=UNKNOWN_LIGAND,  # per default, set residue name to UNL (unknown ligand)
                chiral_tag=int(rdatom.GetChiralTag()),
                is_aromatic=rdatom.GetIsAromatic(),
            )
        )
    atom_array = struc.array(atoms)

    # Set bonds
    # ... kekulize to ensure aromaticity is perceived correctly and assign integer bond orders (aromatic flag is not cleared!)
    Chem.Kekulize(mol)
    # ... create bond list with integer bond orders
    bond_list = []
    for bond in mol.GetBonds():
        rdkit_bond_type = bond.GetBondType()

        if rdkit_bond_type not in CONVERTIBLE_RDKIT_BOND_TYPES:
            # ... skip undesired bonds, e.g. dative bonds (=metal coordination bonds)
            logger.warning(f"Skipping {rdkit_bond_type=}. Not in convertible bond types.")
            continue

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        is_bond_aromatic = bond.GetIsAromatic()
        # ... after kekulize, order is guaranteed to be integer
        bond_type = RDKIT_BOND_TYPE_TO_BIOTITE[(bond.GetBondType(), is_bond_aromatic)]
        bond_list.append((begin_atom_idx, end_atom_idx, bond_type))
    atom_array.bonds = struc.bonds.BondList(n_atoms, np.array(bond_list))

    # Set extra annotations
    annotations = mol._annotations if hasattr(mol, "_annotations") else {}
    if len(annotations) > 0:
        # Create mapping of array idx <> annotation idx via the openbabel atom id:
        _rdkit_id_to_annotation_idx = {
            rdkit_atom_id: idx for idx, rdkit_atom_id in enumerate(annotations["rdkit_atom_id"])
        }

        array_idx_to_annotation_idx = []
        for idx, atom in enumerate(atoms):
            if atom.rdkit_atom_id in _rdkit_id_to_annotation_idx:
                array_idx_to_annotation_idx.append((idx, _rdkit_id_to_annotation_idx[atom.rdkit_atom_id]))
        array_idx_to_annotation_idx = np.array(array_idx_to_annotation_idx)

        # Exit if there are no annotations to set
        if len(array_idx_to_annotation_idx) == 0:
            logger.warning(
                "No rdkit atoms match any annotation. You may want to check that you are "
                "using the @preserve_annotations decorator correctly. And set the "
                "rdkit pickle options to preserve properties. Returning."
            )
            return atom_array

        for key, val in annotations.items():
            if key in ["coord", "charge"]:
                logger.warning(
                    f"Found built-in annotation: {key} as custom annotation. "
                    "Using re-calculated values from RDKit instead."
                )
                continue

            if np.issubdtype(val.dtype, np.integer):
                if np.issubdtype(val.dtype, np.unsignedinteger):
                    # Use max value for unsigned integers since they can't hold -1
                    default = np.iinfo(val.dtype).max
                else:
                    default = -1
            elif np.issubdtype(val.dtype, np.floating):
                default = np.nan
            elif np.issubdtype(val.dtype, np.str_ or str):
                default = ""
            elif np.issubdtype(val.dtype, bool):
                default = False
            else:
                logger.warning(f"Unsupported annotation dtype: {val.dtype} for annotation: {key}. Skipping.")
                continue

            vals = np.full(len(atom_array), default, dtype=val.dtype)
            vals[array_idx_to_annotation_idx[:, 0]] = annotations[key][array_idx_to_annotation_idx[:, 1]]
            atom_array.set_annotation(key, vals)

    if remove_hydrogens:
        atom_array = atom_array[not_isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)]

    if remove_inferred_atoms:
        atom_array = atom_array[atom_array.rdkit_atom_id != -1]

    return atom_array


def atom_array_to_rdkit(
    atom_array: AtomArray,
    *,
    set_coord: bool | None = None,
    hydrogen_policy: Literal["infer", "remove", "keep"] = "keep",
    annotations_to_keep: list[str] = BIOTITE_DEFAULT_ANNOTATIONS,
    sanitize: bool = True,
    attempt_fixing_corrupted_molecules: bool = True,
    assume_metal_bonds_are_dative: bool = False,
) -> Mol:
    """Generate an RDKit molecule from a Biotite AtomArray object.

    Args:
        - atom_array (biotite.structure.AtomArray): The Biotite AtomArray to convert.
        - set_coord (bool | None): Whether to set atomic coordinates in the RDKit molecule.
            If None, coordinates are only set if they are not NaN.
        - hydrogen_policy (Literal["infer", "remove", "keep"]): Whether to infer hydrogens in the RDKit molecule.
        - annotations_to_keep (list[str]): List of atom annotations to preserve from the AtomArray.
        - sanitize (bool): Whether to sanitize the molecule during conversion. Default is True.
        - attempt_fixing_corrupted_molecules (bool): Whether to attempt fixing corrupted molecules during conversion. Default is True.
        - assume_metal_bonds_are_dative (bool): Whether to assume that all bonds with metals are dative bonds. Default is False.
            WARNING: This messes up RDKit conformer generation.

    Returns:
        - rdkit.Chem.Mol: RDKit Molecule generated from the AtomArray.

    Note:
        Aromaticity, hybridization states, and other properties are automatically
        perceived by RDKit's SanitizeMol during the conversion process.
    """
    # Initialize the RDKit molecule; copy AtomArray to avoid modifying the original
    atom_array = atom_array.copy()
    mol = Chem.RWMol()

    # Set atoms
    # ... we use an internal `rdkit_atom_id` property to keep track of atoms that were originally
    #     in the AtomArray from atoms that might have been added by RDKit implicitly later
    #     (implicit hydrogens)
    rdkit_atom_ids = []

    if hydrogen_policy in ("infer", "remove"):
        atom_array = ta.remove_hydrogens(atom_array)
    elif hydrogen_policy == "keep":
        pass
    else:
        raise ValueError(f"Invalid hydrogen policy: {hydrogen_policy}. Must be 'infer', 'remove', or 'keep'.")

    for atom_id, atom in enumerate(atom_array):
        atomic_number = element_to_atomic_number(atom.element)

        rdatom = Chem.Atom(atomic_number)
        if hasattr(atom, "charge"):
            # ... set formal charge if available (otherwise RDKit will assume it is 0
            #  and assign a charge state in SanitizeMol if it is required to satisfy
            #  valence constraints)
            rdatom.SetFormalCharge(int(atom.charge))

        rdatom.SetIntProp("rdkit_atom_id", atom_id)
        rdatom.SetProp("atom_name", atom.atom_name)
        rdkit_atom_ids.append(atom_id)
        mol.AddAtom(rdatom)

    # Set coordinates first
    set_coord = set_coord or not np.any(np.isnan(atom_array.coord))
    if set_coord:
        # ... add conformer (at id 0)
        conf_id = mol.AddConformer(Chem.Conformer(len(atom_array)), assignId=True)

        # ... fill in coordinates
        for atom_id, atom_coord in enumerate(atom_array.coord):
            mol.GetConformer(conf_id).SetAtomPosition(atom_id, atom_coord.tolist())

    # Set bonds from existing bonds in atom_array
    _should_be_aromatic = set()
    if exists(atom_array.bonds):
        # Use existing bonds from atom_array
        for bond in atom_array.bonds.as_array():
            atom1, atom2, bond_type = list(map(int, bond))

            if bond_type == struc.bonds.BondType.ANY:
                logger.warning("Encountered BondType.ANY. Interpreting as single bond.")

            bond_order, bond_is_aromatic = BIOTITE_BOND_TYPE_TO_RDKIT[bond_type]
            mol.AddBond(atom1, atom2, order=bond_order)

            if bond_is_aromatic and not attempt_fixing_corrupted_molecules:
                mol.GetAtomWithIdx(atom1).SetIsAromatic(True)
                mol.GetAtomWithIdx(atom2).SetIsAromatic(True)

            _should_be_aromatic.union({atom1, atom2})

    # Assign stereochemistry (requires 3D coordinates and bonds)
    if mol.GetNumConformers() > 0:
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except ValueError:
            logger.warning("Failed to assign stereochemistry to molecule.")
            pass

    # Clean up organometallics:
    # TODO: The CCD unfortunately only supplies all metal bonds as single bonds. For now we assume
    #  all bonds with metals are coordination bonds in the PDB. This will
    #  likely be true for most ligands but not all. Revisit this later.
    # Change all bonds to a metal to be dative bonds (= coordination bonds)
    if assume_metal_bonds_are_dative:
        mol = change_metal_bonds_to_dative(mol)

    if attempt_fixing_corrupted_molecules:
        # ... fix_mol has no effect if the molecule is already sanitized
        mol = fix_mol(
            mol,
            attempt_fix_by_normalizing_like_chembl=True,
            attempt_fix_by_normalizing_like_rdkit=True,
            attempt_fix_by_calculating_charge_from_valence=True,
            in_place=True,
            raise_on_failure=False,
        )

    # Clean up the molecule and infer various properties
    # (We always sanitize when attempting to fix corrupted molecules)
    if sanitize or attempt_fixing_corrupted_molecules:
        # ... verify validity of the molecule (according to Lewis octet rule)
        try:
            Chem.SanitizeMol(mol)
        except Chem.MolSanitizeException as e:
            logger.warning(
                f"Failed final sanitzation of molecule with error: {e}! It may not satisfy the octet rule, for example. Catching error and ignoring..."
            )

        # ... verify that atoms that are labelled as `_should_be_aromatic` are aromatic
        for atom_idx in _should_be_aromatic:
            assert mol.GetAtomWithIdx(
                atom_idx
            ).GetIsAromatic(), f"Atom {atom_idx} is not aromatic but was labelled as aromatic."

    # Turn into a non-editable molecule
    mol = mol.GetMol() if isinstance(mol, Chem.RWMol) else mol

    # Attach custom atom-level annotations from the atom array
    mol._annotations = {"rdkit_atom_id": np.array(rdkit_atom_ids)}
    for annotation in annotations_to_keep:
        if annotation in atom_array.get_annotation_categories():
            mol._annotations[annotation] = atom_array._annot[annotation]

    if hydrogen_policy == "infer":
        mol = add_hydrogens(mol, add_coords=set_coord)

    return mol


@immutable_lru_cache(maxsize=1000)
def ccd_code_to_rdkit(
    ccd_code: str,
    *,
    ccd_mirror_path: PathLike | None = CCD_MIRROR_PATH,
    hydrogen_policy: Literal["infer", "remove", "keep"] = "keep",
    **atom_array_to_rdkit_kwargs,
) -> Mol:
    """
    Convert a CCD residue name to an RDKit molecule.

    This function retrieves an RDKit molecule corresponding to a given CCD residue name.
    If `ccd_dir` is not provided, Biotite's internal CCD is used. Otherwise, the specified local CCD directory is used.

    By default, the function returns the 'ideal' conformer from the CCD entry.

    Args:
        ccd_code (str): The CCD code to convert. I.e, 'ALA', 'GLY', '9RH', etc.
        ccd_mirror_path (PathLike): Path to the local CCD directory. If None, Biotite's internal CCD is used.
        hydrogen_policy (Literal["infer", "remove", "keep"]): Whether to infer hydrogens in the RDKit molecule.
        **atom_array_to_rdkit_kwargs: Additional keyword arguments passed to the `atom_array_to_rdkit` function.

    Returns:
        Mol: The RDKit molecule corresponding to the given residue name.
    """
    atom_array = atom_array_from_ccd_code(ccd_code, ccd_mirror_path)
    mol = atom_array_to_rdkit(
        atom_array,
        set_coord=True,  # ... coordinate needed for stereochemistry assignment
        hydrogen_policy=hydrogen_policy,  # ... hydrogens needed for stereochemistry assignment
        **atom_array_to_rdkit_kwargs,
    )

    # ... assign stereochemistry
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except ValueError:
        logger.warning(f"Failed to assign stereochemistry to {ccd_code}. Returning unstereochem molecule.")
        pass

    return mol
