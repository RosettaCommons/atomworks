import logging
import time
from typing import Any, ClassVar, Literal

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol, rdDistGeom

from atomworks.common import default
from atomworks.io.tools.rdkit import (
    add_hydrogens,
    atom_array_from_rdkit,
    atom_array_to_rdkit,
    ccd_code_to_rdkit,
    preserve_annotations,
    remove_hydrogens,
)
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_does_not_contain_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils import timer

logger = logging.getLogger(__name__)
# ... disable RDKit logging
RDLogger.DisableLog("rdApp.*")

# Set default pickle properties to all properties, otherwise
#  annotations get lost when pickling/unpickling molecules
# https://github.com/rdkit/rdkit/issues/1320
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def _has_explicit_hydrogen(mol: Chem.Mol) -> bool:
    """Check if the molecule has explicit hydrogens."""
    return mol.GetNumAtoms() > mol.GetNumHeavyAtoms()


def _get_random_seed() -> int:
    """Get a random seed for RDKit. This needs to be a python 'int' to play well with RDKit."""
    return int(np.random.randint(0, 2**31 - 1))


@preserve_annotations
def generate_conformers(
    mol: Mol,
    *,
    seed: int | None = None,
    n_conformers: int = 1,
    method: str = "ETKDGv3",
    num_threads: int = 1,
    hydrogen_policy: Literal["infer", "remove", "keep", "auto"] = "remove",
    optimize: bool = True,
    attempts_with_distance_geometry: int = 10,
    attempts_with_random_coordinates: int = 10_000,
    **uff_optimize_kwargs: dict,
) -> Mol:
    """Generate conformations for the given molecule.

    Args:
        mol: The RDKit molecule to generate conformations for.
        seed: Random seed for reproducibility. If None, a random seed is used.
        n_conformers: Number of conformations to generate.
        method: The method to use for conformer generation. Default is "ETKDGv3".
            Allowed methods are: "ETDG", "ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3"
            See https://rdkit.org/docs/RDKit_Book.html#conformer-generation for details.
        num_threads: Number of threads to use for parallel computation. Default is 1.
        hydrogen_policy: Whether to add explicit
            hydrogens to the molecule. If "remove", hydrogens are temporarily added for conformer
            generation, but removed again before returning the molecule. If "keep" the molecule is
            used as-is (without adding or removing hydrogens). If "auto", the policy is set to "keep"
            if the molecule already has explicit hydrogens, otherwise it is set to "remove".
            If "infer", we follow the same behavior as "remove," but do not remove added hydrogens
            prior to returning the molecule.
        optimize: Whether to optimize the generated conformers using UFF.
            Default is True.
        **uff_optimize_kwargs: Additional keyword arguments for UFF optimization:
            - maxIters: Maximum number of iterations (default 200).
            - vdwThresh: Used to exclude long-range van der Waals interactions
              (default 10.0).
            - ignoreInterfragInteractions: If True, nonbonded terms between
              fragments will not be added to the forcefield (default True).

    Returns:
        The molecule with generated conformations.

    Note:
        - Optimizing conformers (optimize_conformers=True) is recommended for obtaining
          more realistic and lower-energy conformations. However, it may increase
          computation time.
        - The ETKDGv3 method is used for conformer generation, which incorporates
          torsion angle preferences and basic knowledge (e.g. aromatic rings are planar)
          for improved accuracy.
        - For macrocycles or complex ring systems, you may need to increase the number
          of conformers generated to ensure good sampling of the conformational space
          (if a representative ensemble of conformers is what you are after).

    Best Practices:
        1. Always add hydrogens before generating conformers unless you have a specific
           reason not to (e.g., you're working with a protein structure where hydrogens
           are already correctly placed).
        2. Use a non-zero seed for reproducibility in research or production environments.
        3. Generate multiple conformers (e.g., 50-100) for flexible molecules to sample
           the conformational space more thoroughly.
        4. Optimize conformers using UFF or MMFF94 for more realistic geometries, especially
           if the conformers will be used for further calculations or analysis.
        5. For very large or complex molecules, you may need to adjust parameters such as
           maxIterations or use more advanced sampling techniques.

    References:
        `Conformer tutorial <https://rdkit.org/docs/RDKit_Book.html#conformer-generation>`_
        `RDKit Cookbook <https://www.rdkit.org/docs/Cookbook.html>`_
        Riniker and Landrum, "Better Informed Distance Geometry: Using What We Know To
        Improve Conformation Generation", JCIM, 2015.
    """
    # Ensure that all properties are being pickled (needed when we use timeout)
    assert (
        Chem.GetDefaultPickleProperties() == Chem.PropertyPickleOptions.AllProps
    ), "Default pickle properties are not set to all properties. Annotation loss will occur."
    assert attempts_with_distance_geometry > 0, "Attempts with distance geometry must be greater than 0."
    assert attempts_with_random_coordinates > 0, "Attempts with random coordinates must be greater than 0."

    if hydrogen_policy == "auto":
        hydrogen_policy = "keep" if _has_explicit_hydrogen(mol) else "remove"
    if hydrogen_policy in ("infer", "remove"):
        mol = remove_hydrogens(mol)
        # ... temporarily add hydrogens for more realistic conformer generation
        mol = add_hydrogens(mol)
    elif hydrogen_policy == "keep":
        pass
    else:
        raise ValueError(f"Invalid hydrogen policy: {hydrogen_policy}. Must be one of 'infer', 'remove', or 'keep'.")

    # Setup the parameters for the coordinate embedding
    params = getattr(rdDistGeom, method)()
    params.clearConfs = True
    params.randomSeed = seed or _get_random_seed()
    params.enforceChirality = True
    params.useRandomCoords = False
    params.numThreads = num_threads

    # Newer RDKit version have renamed this parameter
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = attempts_with_distance_geometry
    else:
        params.maxIterations = attempts_with_distance_geometry

    try:
        successful_cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params)
        if len(successful_cids) < n_conformers:
            logger.warning(
                f"Initial conformer generation based on distance geometry failed. Successful: {len(successful_cids)}. "
                "Falling back to generating a conformer starting from random coordinates."
            )
            raise RuntimeError("Failed to generate enough conformers.")
    except RuntimeError:
        # Addresses issues with bad conformers, which happens when distance embeddings fail due to
        #  too many constraints or rotatable bonds, see for example:
        # https://github.com/rdkit/rdkit/issues/1433#issuecomment-305097888
        params.useRandomCoords = True
        params.enforceChirality = False
        if hasattr(params, "maxAttempts"):
            params.maxAttempts = attempts_with_random_coordinates
        else:
            params.maxIterations = attempts_with_random_coordinates
        successful_cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params)
        if len(successful_cids) < n_conformers:
            raise RuntimeError(  # noqa: B904
                f"Requested {n_conformers} conformers, but only {mol.GetNumConformers()} were generated."
            )

    if optimize:
        mol = optimize_conformers(mol, **uff_optimize_kwargs)

    mol = remove_hydrogens(mol) if hydrogen_policy == "remove" else mol
    return mol


@preserve_annotations
def optimize_conformers(
    mol: Mol,
    numThreads: int = 1,  # noqa: N803
    maxIters: int = 200,  # noqa: N803
    vdwThresh: float = 10.0,  # noqa: N803
    ignoreInterfragInteractions: bool = True,  # noqa: N803
) -> Mol:
    """
    Optimize the conformers of an RDKit molecule.

    Args:
        - mol (Mol): The RDKit molecule to optimize.
        - numThreads (int): Number of threads to use for parallel computation. Default is 1.
        - maxIters (int): Maximum number of iterations for UFF optimization. Defaults to 200.
        - vdwThresh (float): Used to exclude long-range van der Waals interactions. Defaults to 10.0.
        - ignoreInterfragInteractions (bool): If True, nonbonded terms between fragments will not be added to the
            forcefield. Defaults to True.

    Returns:
        Mol: The optimized RDKit molecule.
    """
    success = AllChem.UFFOptimizeMoleculeConfs(
        mol,
        numThreads=numThreads,
        maxIters=maxIters,
        vdwThresh=vdwThresh,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
    if not success:
        logger.warning("Conformer optimization did not converge.")
    return mol


def get_chiral_centers(mol: Mol) -> list[int]:
    """Identify and return the tetrahedral chiral centers in an RDKit molecule.

    Finds all tetrahedral chiral centers in the given molecule
    and returns their information, including the chiral center atom index and
    the indices of the atoms bonded to it.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule to analyze.

    Returns:
        - list[dict]: A list of dictionaries, where each dictionary contains:
            - "chiral_center_idx" (int): The index of the chiral center atom.
            - "bonded_explicit_atom_idxs" (list[int]): A list of indices of the atoms
              bonded to the chiral center.
            - "chirality" (str): The chirality of the center ('R' or 'S').

    Note:
        This function will generate a 3D conformation if one is not present, as
        chirality assignment requires 3D coordinates in RDKit to break the conditional
        tie between multiple possible chirality centers.
    """
    # Infer 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        generate_conformers(mol, n_conformers=1)

    # Assign chiral tags based on the 3D structure
    Chem.AssignAtomChiralTagsFromStructure(mol)

    # Identify chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    def _should_exclude_chiral_center(atom: Chem.Atom) -> bool:
        """Exclude edge cases for chiral centers."""
        # EDGE CASE: The chiral center is a Phosphorous (P, 15) or Sulfur (S, 16) atom bonded to 2 Oxygens (O, 8)
        # For an example of where this case occurs, see the CCD ligand `NAP` (e.g., in `5OCM`)
        if atom.GetAtomicNum() == 15 or atom.GetAtomicNum() == 16:
            # ... get the atomic numbers of the bonded atoms
            atomic_nums_of_bonded_atoms = [bond.GetOtherAtom(atom).GetAtomicNum() for bond in atom.GetBonds()]
            # ... check if there are 2 Oxygens (O, 8) bonded to the chiral center
            if atomic_nums_of_bonded_atoms.count(8) >= 2:
                return True

        return False

    # Filter chiral centers with tetrahedral geometry
    tetrahedral_chiral_centers = []
    for center in chiral_centers:
        idx, chirality = center
        atom = mol.GetAtomWithIdx(idx)
        if (
            atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            or atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        ):
            chiral_center_atom_name = atom.GetProp("atom_name") if atom.HasProp("atom_name") else None
            if not _should_exclude_chiral_center(atom):
                tetrahedral_chiral_centers.append(
                    {
                        "chiral_center_idx": idx,
                        "bonded_explicit_atom_idxs": [bond.GetOtherAtomIdx(idx) for bond in atom.GetBonds()],
                        "chirality": chirality,
                        # For debugging - currently unused by the pipeline
                        "chiral_center_atom_name": chiral_center_atom_name,
                        "bonded_explicit_atom_names": [
                            mol.GetAtomWithIdx(bond.GetOtherAtomIdx(idx)).GetProp("atom_name")
                            for bond in atom.GetBonds()
                            if chiral_center_atom_name
                        ]
                        if chiral_center_atom_name
                        else None,
                    }
                )

    # Remove chiral centers that are R or S bonded to 2 O

    return tetrahedral_chiral_centers


def find_automorphisms_with_rdkit(
    mol: Chem.Mol,
    max_automorphs: int = 1000,
    timeout: float | None = None,
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
) -> np.ndarray:
    """
    Find automorphisms of a given RDKit molecule.

    This function identifies the automorphisms (symmetry-related atom swaps) of the input molecule
    and returns them as a numpy array. If the search for automorphisms times out, it returns a single
    automorphism representing the identity (no swaps).

    Args:
        mol (Chem.Mol): The RDKit molecule for which to find automorphisms.
        max_automorphs (int): The maximum number of automorphisms to return. These are deterministically
            set to be the first `max_automorphs` automorphisms found by RDKit.
            For model training it is recommended to deterministically select the automorphisms
            to be used (as done in this transform) as a model might otherwise be nudged towards a specific
            automorph in one training step, but that automorph then does not show up in the next training
            step, leading to a moving target problem.
        timeout (float | None): The timeout for the automorphism search. If None, no timeout is applied and
            the timeout strategy is ignored (no subprocesses will be spawned).
        timeout_strategy (Literal["signal", "subprocess"]): The strategy to use for the timeout.
            Defaults to "subprocess".
    Returns:
        automorphs (np.ndarray): A numpy array of shape [n_automorphs, n_atoms, 2], where each element
            represents an automorphism as list of paired atom indices (from_idx, to_idx).
            If the search fails (e.g. due to running out of memory), returns an array with
            a single automorphism representing the identity (no swaps).

    Reference:
        `RDKit Mailman Discussion <https://sourceforge.net/p/rdkit/mailman/message/27897393/>`_

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

    # NOTE: We compute the automorphisms via a substructure match. This may not be the computationally most
    #  efficient way, but still works well even for highly symmetric molecules (e.g. 60C).
    #  (c.f. https://sourceforge.net/p/rdkit/mailman/message/27897393/)
    #  The probably optimal way to do this would be to access internal symmetry labels for models
    #  (c.f. https://sourceforge.net/p/rdkit/mailman/message/27902778/)
    #  but this would require using an underlying graph librarly like nauty to determine the automorphisms of
    #  the coloured graph. Until we run into performance issues, we will stick with the current approach.
    @timer.timeout(timeout=timeout, strategy=timeout_strategy)
    def _find_automorphisms() -> tuple:
        return mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_automorphs, useChirality=False)

    _start = time.time()
    try:
        automorphs_tuple = _find_automorphisms()
    except TimeoutError:
        logger.warning(
            f"Automorphism search timed out after {time.time() - _start:.2f}s. Returning identity automorphism."
        )
        automorphs_tuple = (tuple(range(mol.GetNumAtoms())),)

    # Turn the tuple of automorphisms into a numpy array of shape [n_automorphs, n_atoms, 2]
    automorphs = np.array(automorphs_tuple)
    n_automorphs, n_atoms = automorphs.shape
    identity = np.tile(np.arange(n_atoms), (n_automorphs, 1))

    return np.stack([identity, automorphs], axis=-1)


def _auto_timeout_policy(n_conformers: int, offset: float = 3.0, slope: float = 0.15) -> float:
    return offset + slope * (n_conformers - 1)


def sample_rdkit_conformer_for_atom_array(
    atom_array: AtomArray,
    n_conformers: int = 1,
    seed: int | None = None,
    timeout: float | None | tuple[float, float] = (3.0, 0.15),
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    return_mol: bool = False,
    **generate_conformers_kwargs,
) -> AtomArray:
    """Sample a conformer for a Biotite AtomArray using RDKit.

    Args:
        atom_array: The Biotite AtomArray to sample a conformer for.
        n_conformers: The number of conformers to sample.
        timeout: The timeout for conformer generation. If None,
            no timeout is applied. If a tuple, the first element is the offset and the
            second element is the slope.
        seed: The seed for conformer generation. If None, a random seed
            is generated using the global numpy RNG.
        timeout_strategy: The strategy to use for the timeout.
            Defaults to "subprocess".
        **generate_conformers_kwargs: Additional keyword arguments to pass to the
            generate_conformers function.

    Returns:
        The AtomArray with updated coordinates from the sampled conformer.
        The RDKit molecule with the generated conformer.

    Note:
        This function preserves the original atom order and properties of the input AtomArray.
    """
    seed = seed or _get_random_seed()
    atom_array = atom_array.copy()
    mol = atom_array_to_rdkit(atom_array, hydrogen_policy="remove")
    # ... remove RDKit's conformer if there is one
    if mol.GetNumConformers() > 0:
        mol.RemoveAllConformers()

    timeout = _auto_timeout_policy(n_conformers, *timeout) if isinstance(timeout, tuple) else timeout
    generate_conformers_with_timeout = timer.timeout(timeout=timeout, strategy=timeout_strategy)(generate_conformers)

    set_coord_if_available = True
    try:
        mol = generate_conformers_with_timeout(mol, n_conformers=n_conformers, seed=seed, **generate_conformers_kwargs)
    except (TimeoutError, RuntimeError):
        set_coord_if_available = False
        logger.warning(
            f"Failed to generate {n_conformers} conformers for {atom_array.res_name[0]}. Falling back to zeros."
        )

    new_atom_array = atom_array_from_rdkit(
        mol,
        set_coord_if_available=set_coord_if_available,
        remove_inferred_atoms=True,
        remove_hydrogens=False,
    )

    assert new_atom_array.array_length() == atom_array.array_length()
    assert np.all(new_atom_array.atom_name == atom_array.atom_name)
    assert np.all(new_atom_array.res_id == atom_array.res_id)
    assert np.all(new_atom_array.res_name == atom_array.res_name)
    atom_array.coord = new_atom_array.coord

    if return_mol:
        return atom_array, mol

    return atom_array


def ccd_code_to_rdkit_with_conformers(
    ccd_code: str,
    n_conformers: int,
    *,
    seed: int | None = None,
    timeout: float | None | tuple[float, float] = (3.0, 0.15),
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    skip_rdkit_conformer_generation: bool = False,
    **generate_conformers_kwargs,
) -> Chem.Mol:
    """Generate an RDKit molecule with conformers for a given residue name.

    This function attempts to generate the specified number of conformers for the given CCD code
    using RDKit's conformer generation (based on ETKDGv3 per default).
    If conformer generation fails or times out, it falls back to using the idealized conformer
    from the CCD entry if one is available.

    Args:
        ccd_code: The CCD code to generate conformers for. E.g. 'ALA' or 'GLY', '9RH' etc.
        n_conformers: The number of conformers to generate for the given CCD code.
        seed: The seed for conformer generation. If None, a random seed
            is generated using the global numpy RNG.
        timeout: The timeout for the automorphism search. If None, no timeout is applied and
            the timeout strategy is ignored (no subprocesses will be spawned). If a tuple,
            the first element is the offset and the second element is the slope.
        timeout_strategy: The strategy to use for the timeout.
            Defaults to "subprocess".
        **generate_conformers_kwargs: Additional keyword arguments to pass to the
            generate_conformers function.

    Returns:
        An RDKit molecule with the specified number of conformers.
    """
    # ... get molecule from CCD with its idealized conformer (default conformer 0)
    mol = ccd_code_to_rdkit(ccd_code, hydrogen_policy="remove")

    # ... get idealized conformer from CCD entry
    idealized_conformer = Chem.Conformer(mol.GetConformer(0))  # creates a copy

    # ... try generating `count` conformers within a given time limit
    if not skip_rdkit_conformer_generation:
        timeout = _auto_timeout_policy(n_conformers, *timeout) if isinstance(timeout, tuple) else timeout
        generate_conformers_with_timeout = timer.timeout(timeout=timeout, strategy=timeout_strategy)(
            generate_conformers
        )
        try:
            seed = seed or _get_random_seed()
            mol = generate_conformers_with_timeout(
                mol, n_conformers=n_conformers, seed=seed, **generate_conformers_kwargs
            )
        except (TimeoutError, RuntimeError, Chem.MolSanitizeException) as e:
            logger.warning(
                f"Failed to generate {n_conformers} conformers for {ccd_code=}. Falling back to idealized conformer from the CCD. Error message: {e}"
            )

    # ... if conformer generation fails or is incomplete, return the idealized conformer (set `count` conformers)
    missing_conformers = n_conformers - mol.GetNumConformers()
    if missing_conformers > 0:
        for _ in range(missing_conformers):
            mol.AddConformer(Chem.Conformer(idealized_conformer), assignId=True)

    return mol


# -------------------------------------------------------------------------------------------------
# ---------------------  RDKit related transforms  ------------------------------------------------
# -------------------------------------------------------------------------------------------------


class AddRDKitMoleculesForAtomizedMolecules(Transform):
    """
    Add RDKit molecules for atomized molecules in the atom array.

    This transform converts atomized molecules in the atom array to RDKit Mol objects and stores them in the
    `data` dictionary under the "rdkit" key. Each molecule is identified by its `pn_unit_iid`.

    Note:
        This transform requires the `AtomizeByCCDName` transform to be applied previously.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including the atom array.

    Returns:
        dict[str, Any]: The updated `data` dictionary with the added RDKit molecules under the
            `"rdkit"` key.

    Example:
        >>> data = {
        >>>     "atom_array": AtomArray(...),  # Your atom array here
        >>> }
        >>> transform = AddRDKitMoleculesForAtomizedMolecules()
        >>> data = transform(data)
        >>> print(data["rdkit"])
        {
            'A_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
            'B_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
            ...
        }
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AtomizeByCCDName"]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["CropContiguousLikeAF3", "CropSpatialLikeAF3"]

    def __init__(self, hydrogen_policy: Literal["infer", "remove", "keep"] = "keep"):
        self.hydrogen_policy = hydrogen_policy

    def check_input(self, data: dict[str, Any]) -> None:
        check_does_not_contain_keys(data, ["rdkit"])
        check_atom_array_annotation(data, ["atomize", "pn_unit_iid"])

    def _convert_atom_array_to_rdkit_robust(self, atom_array: AtomArray) -> Mol:
        """Convert an AtomArray to RDKit molecule with error sanitization failure handling"""

        conversion_kwargs = {
            "hydrogen_policy": self.hydrogen_policy,
            "annotations_to_keep": ["chain_id", "res_id", "res_name", "atom_name", "atom_id", "pn_unit_iid"],
            "sanitize": True,
        }

        try:
            # First attempt: try without fixing for better performance
            rdmol = atom_array_to_rdkit(atom_array, attempt_fixing_corrupted_molecules=False, **conversion_kwargs)

            # Check if sanitization actually succeeded
            sanitization_result = Chem.SanitizeMol(rdmol, catchErrors=True)
            if sanitization_result != Chem.SanitizeFlags.SANITIZE_NONE:
                # Molecule failed sanitization, retry with fixing enabled
                logger.warning(
                    f"Molecule {atom_array.res_name[0]} failed sanitization ({sanitization_result}). "
                    "Retrying with attempt_fixing_corrupted_molecules=True."
                )
                rdmol = atom_array_to_rdkit(atom_array, attempt_fixing_corrupted_molecules=True, **conversion_kwargs)

            return rdmol

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(
                f"Failed to convert molecule {atom_array.res_name[0]} to RDKit: {e!s}. "
                "Trying again and attempting to fix the corrupted molecule."
            )
            return atom_array_to_rdkit(atom_array, attempt_fixing_corrupted_molecules=True, **conversion_kwargs)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        # Subset to atomized molecules
        _atom_array = atom_array[atom_array.atomize]

        # Iterate over unique pn_unit_iids
        data["rdkit"] = {}
        for pn_unit_iid in np.unique(_atom_array.pn_unit_iid):
            pn_unit_mask = _atom_array.pn_unit_iid == pn_unit_iid
            molecule = _atom_array[pn_unit_mask]
            rdmol = self._convert_atom_array_to_rdkit_robust(molecule)
            data["rdkit"][pn_unit_iid] = rdmol

        return data


class GenerateRDKitConformers(Transform):
    """
    Generate conformers for RDKit molecules stored in the `data["rdkit"]` dictionary.

    This transform generates conformers for each RDKit molecule in the data dictionary and updates
    the molecules with the new conformers. The random seed for conformer generation is derived from
    the global numpy RNG state.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including RDKit molecules
            under the `"rdkit"` key.
        n_conformers (int): Number of conformations to generate for each molecule. Default is 1.

    Returns:
        dict[str, Any]: The updated `data` dictionary with RDKit molecules containing generated conformers.

    Example:
        >>> data = {
        >>>     "rdkit": {
        >>>         'A_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
        >>>         'B_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
        >>>     }
        >>> }
        >>> transform = GenerateRDKitConformers(n_conformers=3)
        >>> data = transform(data)
        >>> print(data["rdkit"]["A_1"].GetNumConformers())
        3
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["AddRDKitMoleculesForAtomizedMolecules"]

    def __init__(
        self, n_conformers: int = 1, optimize_conformers: bool = True, optimize_kwargs: dict[str, Any] | None = None
    ):
        self.n_conformers = n_conformers
        self.optimize_conformers = optimize_conformers
        self.optimize_kwargs = default(optimize_kwargs, {})

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        for pn_unit_iid, rdmol in data["rdkit"].items():
            try:
                # Generate a random seed using numpy's global RNG
                seed = np.random.randint(0, 2**16 - 1)

                rdmol_with_conformers = generate_conformers(
                    rdmol,
                    seed=seed,
                    n_conformers=self.n_conformers,
                    hydrogen_policy="auto",
                    optimize=self.optimize_conformers,
                    optimize_kwargs=self.optimize_kwargs,
                )
                data["rdkit"][pn_unit_iid] = rdmol_with_conformers
            except Exception as e:
                logger.warning(f"Failed to generate conformers for molecule {pn_unit_iid}: {e}")

        return data


def get_rdkit_chiral_centers(rdkit_mols: dict[str, Mol]) -> dict:
    """Computes the chiral centers for a dictionary of RDKit molecules.

    See the `GetRDKitChiralCenters` transform for more details.
    """
    chiral_centers = {}

    # Get chiral centers for all rdkit mols
    for resname, rdmol in rdkit_mols.items():
        try:
            # Get the chiral centers (returned are the indices of the chiral center atoms within the `obmol` object)
            chiral_centers[resname] = get_chiral_centers(rdmol)

        except Exception as e:
            logger.warning(f"Failed to find chiral centers for molecule {resname}: {e}")

    return chiral_centers


class GetRDKitChiralCenters(Transform):
    """Identify chiral centers in the RDKit molecules stored in the data["rdkit"] dictionary.

    Returns a dictionary mapping each residue name to a list of chiral centers, e.g:

    .. code-block:: python

        data["chiral_centers"] = {
            ...
            "ILE": [
                {'chiral_center_idx': 1, 'bonded_explicit_atom_idxs': [0, 2, 4], 'chirality': 'S'},
                {'chiral_center_idx': 4, 'bonded_explicit_atom_idxs': [1, 5, 6], 'chirality': 'S'}
            ],
            ...
        }

    Each chiral center is a dict with a center atom index, 3 or 4 bonded atom indices, and the
    RDKit-determined chirality.

    Uses RDKit molecules first computed in GetAF3ReferenceMoleculeFeatures.

    Args:
        data: A dictionary containing the input data, including RDKit molecules
            under the "rdkit" key.

    Returns:
        The updated data dictionary with chiral_centers containing chiral
            centers for each molecule.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["GetAF3ReferenceMoleculeFeatures"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["chiral_centers"] = get_rdkit_chiral_centers(data["rdkit"])

        return data
