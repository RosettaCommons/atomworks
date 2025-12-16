import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, ClassVar, Literal

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit import Chem

from atomworks.common import exists
from atomworks.constants import CCD_MIRROR_PATH, ELEMENT_NAME_TO_ATOMIC_NUMBER, UNKNOWN_LIGAND
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io.tools.rdkit import atom_array_from_rdkit, remove_hydrogens
from atomworks.io.utils.ccd import get_available_ccd_codes
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.rdkit_utils import (
    ccd_code_to_rdkit_with_conformers,
    sample_rdkit_conformer_for_atom_array,
)
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation

logger = logging.getLogger("atomworks.ml")


# (Lazy-load this expensive computation to avoid slow imports)
@lru_cache(maxsize=1)
def get_known_ccd_codes() -> frozenset[str]:
    """Get the set of known CCD codes, computing it lazily on first access."""
    # UNL is a special CCD code for unknown ligands; we do not consider it "known" as it has no structure
    return get_available_ccd_codes(CCD_MIRROR_PATH) - {UNKNOWN_LIGAND}


def _extract_cached_conformers(
    res_stochiometry: dict[str, int],
    max_conformers_per_residue: int | None,
    cached_residue_level_data: dict | None,
) -> tuple[dict[str, Chem.Mol], dict[str, int]]:
    """Extract cached conformers and return remaining stochiometry."""
    cached_mols = {}
    remaining_stochiometry = res_stochiometry.copy()

    if cached_residue_level_data is None:
        return cached_mols, remaining_stochiometry

    for res_name, count in res_stochiometry.items():
        needed_conformers = min(count, max_conformers_per_residue) if max_conformers_per_residue is not None else count

        if res_name in cached_residue_level_data:
            # (We remove hydrogens to be consistent with on-the-fly conformer generation)
            cached_mol = remove_hydrogens(cached_residue_level_data[res_name].get("mol"))
            if cached_mol is not None and cached_mol.GetNumConformers() >= needed_conformers:
                # We have enough cached conformers - use the cached mol
                cached_mols[res_name] = cached_mol
                del remaining_stochiometry[res_name]

    return cached_mols, remaining_stochiometry


def _get_rdkit_mols_with_conformers(
    res_stochiometry: dict[str, int],
    max_conformers_per_residue: int | None = None,
    timeout: float | None | tuple[float, float] = (3.0, 0.15),
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    **generate_conformers_kwargs,
) -> dict[str, Chem.Mol]:
    """Generate RDKit molecules with conformers for each residue in bulk (given the counts in `res_stochiometry`).
    Args:
        res_stochiometry: A dictionary mapping residue names to their count.
        max_conformers_per_residue: Maximum number of conformers to generate per residue type.
            If None, generates conformers equal to the count. If set, generates min(count, max_conformers_per_residue).
        timeout: The timeout for conformer generation. If None, no timeout is applied and
            the timeout strategy is ignored (no subprocesses will be spawned). Defaults to (3.0, 0.15), which
            gives a timeout of 3.0 + 0.15 * (n_conformers - 1) seconds per unique CCD code.
        timeout_strategy: The strategy to use for the timeout. Defaults to "subprocess".
        **generate_conformers_kwargs: Additional keyword arguments to pass to the
            generate_conformers function.

    Returns:
        A dictionary mapping residue names to RDKit molecules with generated conformers.

    Note:
        This function uses the res_name_to_rdkit_with_conformers function to generate conformers
        for each residue. If conformer generation fails or times out for a residue, it falls back
        to using the idealized conformer from the CCD entry if available.

    Reference:
        `AF3 Supplementary Information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """
    ref_mols = {}
    for res_name, count in res_stochiometry.items():
        if res_name not in get_known_ccd_codes():
            ref_mols[res_name] = None  # placeholder so that the unknown CCD codes are still counted later on
            continue

        n_conformers_to_generate = (
            min(count, max_conformers_per_residue) if max_conformers_per_residue is not None else count
        )
        mol = ccd_code_to_rdkit_with_conformers(
            ccd_code=res_name,
            n_conformers=n_conformers_to_generate,
            timeout=timeout,
            timeout_strategy=timeout_strategy,
            **generate_conformers_kwargs,
        )
        ref_mols[res_name] = mol

    return ref_mols


def _encode_atom_names_like_af3(atom_names: np.ndarray) -> np.ndarray:
    """Encodes atom names like AF3.

    This generates the `ref_atom_name_chars` feature used in AF3.
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) - 32, and names are padded to
        length 4.

    Reference:
        `AF3 Supplementary Information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """
    # Handle empty array case
    if len(atom_names) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    # Ensure uppercase
    atom_names = np.char.upper(atom_names)
    # Turn into 4 character ASCII string (this truncates longer atom names)
    atom_names = atom_names.astype("|S4")
    # Pad to 4 char string with " " (ord(" ") = 32)
    atom_names = np.char.ljust(atom_names, width=4, fillchar=" ")
    # Interpret ASCII bytes to uint8
    atom_names = atom_names.view(np.uint8)
    # Reshape to (N, 4) and subtract 32 to get back to range [0, 64]
    return atom_names.reshape(-1, 4) - 32


def _map_reference_conformer_to_residue(
    res_name: str, atom_names: np.ndarray, conformer: AtomArray
) -> tuple[np.ndarray, np.ndarray]:
    """Maps the coordinate information from a reference conformer to a
    given residue, dropping all atoms that are not in the residue.

    Args:
        - res_name (str): The name of the residue to map to.
        - atom_names (np.ndarray): Array of atom names in the residue to map to.
        - conformer (AtomArray): The reference conformer.

    Returns:
        - ref_pos (np.ndarray): Reference positions for atoms in the residue.
        - ref_mask (np.ndarray): Mask indicating valid reference positions.
    """

    # ... mark the atoms that are in the residue (keep) and where they are in the residue (to_within_res_idx)
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    # Mapping from conformer atom indices to residue atom indices
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer for residue {res_name} with {atom_names=}.")
            continue
        matching_atom_idx = matching_atom_idx[0]
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    # ... fill the reference positions
    # (We must handle the case where to_within_res_idx[keep] contains indices out of bounds for the filtered conformer)
    kept_atoms = np.where(keep)[0]
    ordering = np.array([to_within_res_idx[idx] for idx in kept_atoms])
    coord = conformer.coord[kept_atoms][np.argsort(ordering)]  # [n_atoms_in_res, 3]

    ref_pos = coord
    ref_mask = np.isfinite(coord).all(axis=-1)  # [n_atoms_in_res]

    return ref_pos, ref_mask  # [n_atoms_in_res, 3], [n_atoms_in_res]


def get_af3_reference_molecule_features(
    atom_array: AtomArray,
    conformer_generation_timeout: float | tuple[float, float] = (3.0, 0.15),
    apply_random_rotation_and_translation: bool = True,
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    max_conformers_per_residue: int | None = None,
    cached_residue_level_data: dict | None = None,
    residue_conformer_indices: dict[int, np.ndarray] | None = None,
    **generate_conformers_kwargs,
) -> tuple[dict[str, Any], dict[str, Chem.Mol]]:
    """Get AF3 reference features for each residue in the atom array.

    Args:
        atom_array: The input atom array.
        conformer_generation_timeout: Maximum time allowed for conformer generation per residue.
            Defaults to (3.0, 0.15), which gives a timeout of 3.0 + 0.15 * (n_conformers - 1) seconds.
            If None, no timeout is applied and the timeout strategy is ignored (no subprocesses will be spawned).
        apply_random_rotation_and_translation: Whether to apply a random rotation and translation to each conformer (AF-3-style)
        timeout_strategy: The strategy to use for the timeout.
            Defaults to "subprocess" (which is the most reliable choice).
        max_conformers_per_residue: Maximum number of conformers to generate per residue type.
            If None, generates conformers equal to residue count. If set, generates min(count, max_conformers_per_residue)
            and randomly samples from those conformers for each residue instance.
        cached_residue_level_data: Optional cached conformer data by residue name. If provided,
            cached conformers will be preferred over generated ones when they contain sufficient conformers.
            Expected structure (dictionary):

            .. code-block:: python

                {
                    "ALA": {"mol": rdkit.Chem.Mol},  # Mol with >=1 conformers
                    "HEM": {"mol": rdkit.Chem.Mol},  # Mol with >=1 conformers
                    ...
                }

            Requirements:
                - Keys must be CCD codes (e.g., ``"ALA"``, ``"HEM"``)
                - Each value must be a dict with a ``"mol"`` key containing an RDKit Mol object
                - Each Mol must have at least as many conformers as ``min(count, max_conformers_per_residue)``
                  to be used (otherwise conformers will be generated on-the-fly)
                - Hydrogens will be automatically removed from cached Mols for consistency

        residue_conformer_indices: Optional mapping of global residue IDs to specific conformer indices.
            If provided, these specific conformers will be used for the corresponding residues.
            Structure: ``{global_res_id: conformer_index}`` or ``{global_res_id: np.array([conformer_index])}``
        **generate_conformers_kwargs: Additional keyword arguments to pass to the generate_conformers function.

    Returns:
        ref_conformer: A dictionary containing the generated reference features.
        ref_mols: A dictionary containing all generated RDKit molecules, including those with unknown CCD codes.

    This function generates the following reference features, following AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
            translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms, 128] One-hot encoding of the element atomic number for each atom in the
            reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
            Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
            this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    (Optionally) The following custom features, helpful for extra conditioning:
        - ref_pos_is_ground_truth (optional): [N_atoms] Whether the reference conformer is the ground-truth conformer.
            Determined by the `ground_truth_conformer_policy` annotation.
        - ref_pos_ground_truth (optional): [N_atoms, 3] The ground-truth conformer positions.
            Determined by the `ground_truth_conformer_policy` annotation.
        - is_atomized_atom_level: [N_atoms] Whether the atom is atomized (atom-level version of "is_ligand")

    Reference:
        `Section 2.8 of the AF3 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """
    _has_ground_truth_conformer_policy = "ground_truth_conformer_policy" in atom_array.get_annotation_categories()
    _has_global_res_id = "res_id_global" in atom_array.get_annotation_categories()

    # Generate reference conformers for each residue (if cropped, each residue that has tokens in the crop)
    # ... get residue-level stochiometry
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    _res_names = atom_array.res_name[_res_starts]
    unique_names, counts = np.unique(_res_names, return_counts=True)
    res_stochiometry = {str(name): int(count) for name, count in zip(unique_names, counts, strict=True)}

    # Extract cached conformers and get remaining stochiometry
    if cached_residue_level_data is not None:
        cached_mols, remaining_stochiometry = _extract_cached_conformers(
            res_stochiometry=res_stochiometry,
            max_conformers_per_residue=max_conformers_per_residue,
            cached_residue_level_data=cached_residue_level_data,
        )
    else:
        cached_mols, remaining_stochiometry = {}, res_stochiometry

    # ... get reference molecules with conformers for remaining residues
    # (We do not generate conformers for unknown CCD codes here, as we will do that later)
    generated_mols = _get_rdkit_mols_with_conformers(
        res_stochiometry=remaining_stochiometry,
        max_conformers_per_residue=max_conformers_per_residue,
        hydrogen_policy="remove",
        timeout=conformer_generation_timeout,
        timeout_strategy=timeout_strategy,
        **generate_conformers_kwargs,
    )

    # Merge cached and generated molecules
    ref_mols = {**cached_mols, **generated_mols}

    # ... generate conformers for CCD codes that are unknown (including UNL)
    unknown_ccd_conformers = defaultdict(list)
    if not all(res_name in get_known_ccd_codes() for res_name in res_stochiometry):
        res_indices_with_unknown = np.where(~np.isin(_res_names, list(get_known_ccd_codes())))[0]
        for res_index in res_indices_with_unknown:
            res_name = _res_names[res_index]

            conf_i, mol_i = sample_rdkit_conformer_for_atom_array(
                atom_array[_res_starts[res_index] : _res_ends[res_index]],
                timeout=conformer_generation_timeout,
                timeout_strategy=timeout_strategy,
                return_mol=True,
                **generate_conformers_kwargs,
            )
            unknown_ccd_conformers[res_name].append(conf_i)
            ref_mols[res_name] = mol_i

    # ... initialize reference features
    ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
    ref_mask = np.zeros(len(atom_array), dtype=bool)

    if _has_ground_truth_conformer_policy:
        ref_pos_is_ground_truth = np.zeros(len(atom_array), dtype=bool)
        ref_pos_ground_truth = np.zeros((len(atom_array), 3), dtype=np.float32)

    # Fill `ref_pos` and `ref_mask` arrays
    # ... helper variable to keep track of the next conformer to use for each residue type
    _next_conf_idx = {res_name: 0 for res_name in ref_mols}

    # ... iterate over all residues in the atom array and fill the `ref_pos` and `ref_mask` arrays using the next reference conformer for each residue type
    # We also check the `ground_truth_conformer_policy` annotation to see if we should use the ground-truth conformer
    for res_start, res_end in zip(_res_starts, _res_ends, strict=False):
        res_name = atom_array.res_name[res_start]

        if _has_global_res_id and residue_conformer_indices is not None:
            res_global_id = int(atom_array.res_id_global[res_start])  # Convert to Python int
            if res_global_id in residue_conformer_indices:
                conformer_indices = residue_conformer_indices[res_global_id]
                # (We don't yet support multiple conformers per residue, so we just use the first one, which is random anyhow)
                conf_idx = int(conformer_indices[0] if isinstance(conformer_indices, np.ndarray) else conformer_indices)
            else:
                conf_idx = _next_conf_idx[res_name]
        else:
            conf_idx = _next_conf_idx[res_name]

        # ... turn conformer into an atom array
        if res_name not in get_known_ccd_codes():
            # (conformers for unknown CCD codes are already atom arrays, since we generated them directly)
            conformer = unknown_ccd_conformers[res_name][conf_idx % len(unknown_ccd_conformers[res_name])]
        else:
            # Ensure conf_idx is within bounds for generated conformers
            n_conformers = ref_mols[res_name].GetNumConformers()
            conformer = atom_array_from_rdkit(
                ref_mols[res_name],
                conformer_id=conf_idx % n_conformers,
                remove_hydrogens=True,
            )

        if _has_ground_truth_conformer_policy:
            _has_valid_ground_truth = ~np.isnan(atom_array.coord[res_start:res_end]).any()
            ground_truth_conformer = None
            if not _has_valid_ground_truth:
                logger.debug(
                    "Ground-truth conformer policy set, but NaNs found in the atom array. Conformer policy will be treated as IGNORE."
                )
            else:
                # We REPLACE the generated conformer with the ground-truth conformer if either:
                # (a) the ground-truth conformer policy is set to "REPLACE" for all atoms in the residue
                # (b) the current conformer is all 0's/NaN's (i.e., the conformer generation failed), and the policy is set to "FALLBACK" for all atoms in the residue
                if np.all(
                    atom_array.ground_truth_conformer_policy[res_start:res_end] == GroundTruthConformerPolicy.REPLACE
                ) or (
                    np.all(np.nan_to_num(conformer.coord) == 0)
                    and np.all(
                        atom_array.ground_truth_conformer_policy[res_start:res_end]
                        == GroundTruthConformerPolicy.FALLBACK
                    )
                ):
                    # NOTE: Inefficient since we generate with RDKit, and then discard, the conformer; however, this replacement-based approach is more interpretable and thus preferred
                    # ... use the ground-truth AtomArray (e.g., during inference if we provide a SDF, or if we want to leak ligand geometry)
                    conformer = atom_array[res_start:res_end]
                    # (Center around the origin to avoid leaking 1D information)
                    conformer.coord = masked_center(conformer.coord)
                    ref_pos_is_ground_truth[res_start:res_end] = True

                # We ADD another feature, `ref_pos_ground_truth`, if the policy is set to "ADD" for all atoms in the residue
                if np.all(
                    atom_array.ground_truth_conformer_policy[res_start:res_end] == GroundTruthConformerPolicy.ADD
                ):
                    if np.isnan(atom_array.coord[res_start:res_end]).any():
                        logger.warning(
                            "Ground-truth conformer requested, but NaNs found in the atom array. Conformer will not be replaced with ground truth."
                        )
                    else:
                        ground_truth_conformer = atom_array[res_start:res_end]
                        ground_truth_conformer.coord = masked_center(ground_truth_conformer.coord)

        # ... map the reference conformer information to the given residue
        _ref_pos, _ref_mask = _map_reference_conformer_to_residue(
            res_name=res_name,
            atom_names=atom_array.atom_name[res_start:res_end],
            conformer=conformer,
        )

        # ... apply a random rotation and translation to the reference conformer, if requested
        if apply_random_rotation_and_translation:
            # TODO: Implement more elegantly directly in numpy
            _ref_pos = random_rigid_augmentation(torch.from_numpy(_ref_pos[np.newaxis, :]), batch_size=1).numpy()

        # ... fill the reference features for this residue
        ref_pos[res_start:res_end] = _ref_pos
        ref_mask[res_start:res_end] = _ref_mask

        # (Repeat for the ground truth conformer, if adding through an additional feature)
        if _has_ground_truth_conformer_policy and exists(ground_truth_conformer):
            _ref_pos_ground_truth, _ = _map_reference_conformer_to_residue(
                res_name=res_name,
                atom_names=atom_array.atom_name[res_start:res_end],
                conformer=ground_truth_conformer,
            )
            if apply_random_rotation_and_translation:
                _ref_pos_ground_truth = random_rigid_augmentation(
                    torch.from_numpy(_ref_pos_ground_truth[np.newaxis, :]), batch_size=1
                ).numpy()
            ref_pos_ground_truth[res_start:res_end] = _ref_pos_ground_truth

        # ... update to the next conformer index
        _next_conf_idx[res_name] += 1

    # Generate remaining reference features
    # ... element
    ref_element = (
        atom_array.atomic_number
        if "atomic_number" in atom_array.get_annotation_categories()
        else np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(atom_array.element)
    )
    # ... charge
    ref_charge = atom_array.charge

    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)

    if use_element_for_atom_names_of_atomized_tokens:
        assert (
            "atomize" in atom_array.get_annotation_categories()
        ), "Atomize annotation is required when using element for atom names of atomized tokens."
        ref_atom_name_chars[atom_array.atomize] = _encode_atom_names_like_af3(atom_array.element[atom_array.atomize])

    # ... space uid (type conversion needed for some older torch versions)
    #     we assign a unique integer for each residue instance:
    ref_space_uid = struc.segments.spread_segment_wise(_res_start_ends, np.arange(len(_res_starts), dtype=np.int64))

    is_atomized_atom_level = atom_array.atomize if "atomize" in atom_array.get_annotation_categories() else None

    ref_conformer = {
        "ref_pos": ref_pos,  # (n_atoms, 3)
        "ref_mask": ref_mask,  # (n_atoms)
        "ref_element": ref_element,  # (n_atoms)
        "ref_charge": ref_charge,  # (n_atoms)
        "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
        "ref_space_uid": ref_space_uid,  # (n_atoms)
        "is_atomized_atom_level": is_atomized_atom_level,  # (n_atoms)
    }

    if _has_ground_truth_conformer_policy:
        ref_conformer["ref_pos_ground_truth"] = ref_pos_ground_truth  # (n_atoms, 3)
        ref_conformer["ref_pos_is_ground_truth"] = ref_pos_is_ground_truth  # (n_atoms)

    return ref_conformer, ref_mols


class GetAF3ReferenceMoleculeFeatures(Transform):
    """Generate AF3 reference molecule features for each residue in the atom array.

    This transform adds the following features to the data dictionary under the 'feats' key, following AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
          translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms] One-hot encoding of the element atomic number for each atom in the
          reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
          Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
          this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    And the following custom features, helpful for extra conditioning/downstream use:
        - ref_pos_is_ground_truth: [N_atoms] Whether the reference conformer is the ground-truth conformer.
          Determined by the `ground_truth_conformer_policy` annotation.
        - ref_pos_ground_truth: [N_atoms, 3] The ground-truth conformer positions.
          Determined by the `ground_truth_conformer_policy` annotation.
        - is_atomized_atom_level: [N_atoms] Whether the atom is atomized (atom-level version of "is_ligand")

    Note:
        This transform should be applied after cropping.

    Reference:
        `Section 2.8 of the AF3 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """

    def __init__(
        self,
        conformer_generation_timeout: float = 10.0,
        save_rdkit_mols: bool = True,
        use_element_for_atom_names_of_atomized_tokens: bool = False,
        apply_random_rotation_and_translation: bool = True,
        max_conformers_per_residue: int | None = None,
        use_cached_conformers: bool = True,
        **generate_conformers_kwargs,
    ):
        self.conformer_generation_timeout = conformer_generation_timeout
        self.generate_conformers_kwargs = generate_conformers_kwargs
        self.save_rdkit_mols = save_rdkit_mols
        self.use_element_for_atom_names_of_atomized_tokens = use_element_for_atom_names_of_atomized_tokens
        self.apply_random_rotation_and_translation = apply_random_rotation_and_translation
        self.max_conformers_per_residue = max_conformers_per_residue
        self.use_cached_conformers = use_cached_conformers

        if self.use_element_for_atom_names_of_atomized_tokens:
            logger.warning("Using element type for atom names of atomized tokens.")

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "element", "charge", "atom_name"])

        if self.use_element_for_atom_names_of_atomized_tokens:
            check_atom_array_annotation(data, ["atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Extract cached data and conformer indices, if enabled
        cached_residue_level_data = None
        if self.use_cached_conformers and "cached_residue_level_data" in data:
            cached_residue_level_data = data["cached_residue_level_data"]["residues"]
        residue_conformer_indices = data.get("residue_conformer_indices") if self.use_cached_conformers else None

        # Generate reference features
        reference_features, rdkit_mols = get_af3_reference_molecule_features(
            atom_array,
            conformer_generation_timeout=self.conformer_generation_timeout,
            use_element_for_atom_names_of_atomized_tokens=self.use_element_for_atom_names_of_atomized_tokens,
            apply_random_rotation_and_translation=self.apply_random_rotation_and_translation,
            max_conformers_per_residue=self.max_conformers_per_residue,
            cached_residue_level_data=cached_residue_level_data,
            residue_conformer_indices=residue_conformer_indices,
            **self.generate_conformers_kwargs,
        )

        # Add reference features to the 'feats' dictionary
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        if self.save_rdkit_mols:
            if "rdkit" not in data:
                data["rdkit"] = {}
            data["rdkit"].update(rdkit_mols)

        return data


def random_apply_ground_truth_conformer_by_chain_type(
    atom_array: AtomArray,
    chain_type_probabilities: dict | None = None,
    default_probability: float = 0.0,
    policy: GroundTruthConformerPolicy = GroundTruthConformerPolicy.REPLACE,
    is_unconditional: bool = False,
) -> AtomArray:
    """Apply ground truth conformer policy with configurable probabilities per chain type.

    Adds the `ground_truth_conformer_policy` annotation to the AtomArray if it does not already exist.
    This annotation indicates if/how residues should use the ground-truth coordinates (i.e., the coordinates from the original structure) as the reference conformer.

    Possible values are (as defined in the GroundTruthConformerPolicy enum):
        -  REPLACE: Use the ground-truth coordinates as the reference conformer (replacing the RDKit-generated conformer in-place)
        -  ADD: Use the ground-truth coordinates as an additional feature (rather than replacing the RDKit-generated conformer)
        -  FALLBACK: Use the ground-truth coordinates only if our standard conformer generation pipeline fails (e.g., we cannot generate a conformer with RDKit,
            and the molecule is either not in the CCD or the CCD entry is invalid)
        -  IGNORE: Do not use the ground-truth coordinates as the reference conformer, under any circumstances

    Args:
        atom_array (AtomArray): The input atom array.
        chain_type_probabilities (dict, optional): Dictionary mapping chain types to their probability
            of using ground truth conformer. Defaults to None.
        default_probability (float, optional): Default probability for any chain type not explicitly specified.
            Defaults to 0.0.
        policy (GroundTruthConformerPolicy, optional): Which ground truth conformer policy to apply when selected.
            Defaults to GroundTruthConformerPolicy.REPLACE.
        is_unconditional (bool, optional): Whether we are sampling unconditionally (and thus should not apply the policy).

    Returns:
        AtomArray: The input atom array with the `ground_truth_conformer_policy` annotation updated.
    """
    # ... add the annotation if it does not already exist, defaulting to IGNORE
    if "ground_truth_conformer_policy" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "ground_truth_conformer_policy", np.full(len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8)
        )

    if is_unconditional:
        # (If we are sampling unconditionally, we should not use the ground truth conformer at all)
        return atom_array

    # ... loop through all ChainTypes in the AtomArray and set the appropriate probability
    probabilities = np.full(len(atom_array), default_probability, dtype=np.float32)
    for chain_type in np.unique(atom_array.chain_type):
        if chain_type in chain_type_probabilities:
            # (Probability for this chain type)
            probabilities[atom_array.chain_type == chain_type] = chain_type_probabilities[chain_type]

    # ... sample Bernoulli random variables for each atom (1 = apply policy, 0 = don't apply))
    # (We will only consider the first atom in each residue for the policy)
    apply_policy = np.random.random(len(atom_array)) < probabilities  # [n_atoms]
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts = _res_start_ends[:-1]

    _should_apply_policy = struc.segments.spread_segment_wise(_res_start_ends, apply_policy[_res_starts])  # [n_atoms]

    atom_array.ground_truth_conformer_policy = np.where(
        _should_apply_policy == 1,
        policy,
        atom_array.ground_truth_conformer_policy,
    )

    return atom_array


class RandomApplyGroundTruthConformerByChainType(Transform):
    """Apply ground truth conformer policy with configurable probabilities per chain type."""

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["GetAF3ReferenceMoleculeFeatures"]

    def __init__(
        self,
        chain_type_probabilities: dict | None = None,
        default_probability: float = 0.0,
        policy: GroundTruthConformerPolicy = GroundTruthConformerPolicy.REPLACE,
    ):
        """
        Args:
            chain_type_probabilities: Dictionary mapping chain types or groups of chain types
                to their probability of using ground truth conformer. For example:
                {
                    ChainType.NON_POLYMER: 0.8,
                    (ChainType.POLYPEPTIDE_L, ChainType.POLYPEPTIDE_D): 0.2,
                }
            default_probability: Default probability for any chain type not explicitly specified
            policy: Which ground truth conformer policy to apply when selected
        """
        self.chain_type_probabilities = chain_type_probabilities or {}
        self.default_probability = default_probability
        self.policy = policy

        self._expanded_probabilities = {}
        for chain_type_key, prob in self.chain_type_probabilities.items():
            if isinstance(chain_type_key, tuple):
                # If it's a tuple of chain types, apply the same probability to each
                for ct in chain_type_key:
                    self._expanded_probabilities[ct] = prob
            else:
                # Single chain type
                self._expanded_probabilities[chain_type_key] = prob

    def forward(self, data: dict) -> dict:
        is_unconditional = data.get("is_unconditional", False)
        data["atom_array"] = random_apply_ground_truth_conformer_by_chain_type(
            data["atom_array"],
            chain_type_probabilities=self._expanded_probabilities,
            default_probability=self.default_probability,
            policy=self.policy,
            is_unconditional=is_unconditional,
        )
        return data
