"""Transforms that filter an AtomArray, removing chains, residues, or atoms based on some criteria"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, ClassVar

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.common import exists, not_isin
from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.utils.query import QueryExpression
from atomworks.io.utils.selection import get_annotation
from atomworks.io.utils.sequence import get_1_from_3_letter_code, get_3_from_1_letter_code
from atomworks.ml.preprocessing.constants import TRAINING_SUPPORTED_CHAIN_TYPES
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import ApplyFunctionToAtomArray, logger
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts


def remove_unresolved_pn_units(atom_array: AtomArray) -> AtomArray:
    """
    Filters PN units that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray.

    Can be applied before or after croppping, since cropping may lead to PN units with all unresolved atoms that were previously not entirely unresolved.
    At training time, these unresolved PN units provide minimal value and can lead to errors in the model.
    """
    # Get the PN units with resolved atoms
    pn_units_with_resolved_atoms = np.unique(atom_array.pn_unit_iid[atom_array.occupancy != 0])

    # Restrict the AtomArray to only include PN units with at least one resolved atom
    atom_array = atom_array[np.isin(atom_array.pn_unit_iid, pn_units_with_resolved_atoms)]

    return atom_array


def remove_unresolved_tokens(atom_array: AtomArray) -> AtomArray:
    """
    Filters tokens that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray.

    A token is defined by the token utilities and can be either:
        - A residue (when atomize=False)
        - An individual atom (when atomize=True)
    """
    if len(atom_array) == 0:
        return atom_array

    # Get token boundaries
    token_start_stop_idx = get_token_starts(atom_array, add_exclusive_stop=True)
    token_starts = token_start_stop_idx[:-1]
    token_stops = token_start_stop_idx[1:]

    # ... build atom mask
    atom_mask = np.zeros(len(atom_array), dtype=bool)

    for start, stop in zip(token_starts, token_stops, strict=False):
        # Keep token if ANY atom has occupancy > 0
        if np.any(atom_array.occupancy[start:stop] > 0):
            atom_mask[start:stop] = True

    return atom_array[atom_mask]


class RemoveUnresolvedPNUnits(Transform):
    """
    Filters PN units that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray.

    Can be applied before or after croppping, since cropping may lead to PN units with all unresolved atoms that were previously not entirely unresolved.
    At training time, these unresolved PN units provide minimal value and can lead to errors in the model.
    """

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "occupancy"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = remove_unresolved_pn_units(data["atom_array"])
        return data


class RemoveUnresolvedTokens(Transform):
    """Filters tokens that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray."""

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = remove_unresolved_tokens(data["atom_array"])
        return data


def remove_unsupported_chain_types(
    atom_array: AtomArray,
    query_pn_unit_iids: Sequence[str] | None = None,
    supported_chain_types: Sequence[ChainType] = TRAINING_SUPPORTED_CHAIN_TYPES,
) -> AtomArray:
    """Filter out chains with unsupported chain types from the AtomArray.

    Additionally, asserts that none of the query pn_units are of an unsupported chain type if given.
    (in which case they should have been filtered out upstream, otherwise our example is not valid).

    Args:
        query_pn_unit_iids (Sequence[str] | None): The PN unit IDs to check for unsupported chain types.
        supported_chain_types (Sequence[ChainType]): The chain types to filter out.

    Returns:
        AtomArray: The filtered AtomArray.
    """
    # We first assert that none of the query pn_units are of an unsupported chain type, which means the example should have been filtered out upstream
    if exists(query_pn_unit_iids):
        query_pn_unit_chain_types = np.unique(
            atom_array.chain_type[np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)]
        )
        assert np.all(
            np.isin(query_pn_unit_chain_types, supported_chain_types)
        ), f"Query PN unit has an unsupported chain type: {query_pn_unit_chain_types}"

    # Then, we filter out chains with unsupported chain types
    is_supported_chain_type = np.isin(atom_array.chain_type, supported_chain_types)
    return atom_array[is_supported_chain_type]


class RemoveUnsupportedChainTypes(Transform):
    """Filter out chains with unsupported chain types from the AtomArray.

    Additionally, asserts that none of the query pn_units are of an unsupported chain type if given.
    (in which case they should have been filtered out upstream, otherwise our example is not valid).
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = []

    def __init__(self, supported_chain_types: Sequence[ChainType] = TRAINING_SUPPORTED_CHAIN_TYPES):
        """
        Initialize the RemoveUnsupportedChainTypes transform.

        Args:
            supported_chain_types (Sequence[ChainType]): The chain types to keep in the AtomArray.
        """
        self.supported_chain_types = supported_chain_types

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["chain_type", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        query_pn_unit_iids = data.get("query_pn_unit_iids")

        # Apply transform
        atom_array = remove_unsupported_chain_types(atom_array, query_pn_unit_iids, self.supported_chain_types)

        # Update data
        data["atom_array"] = atom_array

        return data


def remove_hydrogens(atom_array: AtomArray, hydrogen_names: tuple | list = HYDROGEN_LIKE_SYMBOLS) -> AtomArray:
    """
    Remove hydrogens from the atom array.
    """
    is_heavy = not_isin(atom_array.element, hydrogen_names)
    return atom_array[is_heavy]


class RemoveHydrogens(Transform):
    """
    Remove hydrogens from the atom array.
    """

    def __init__(self, hydrogen_names: tuple | list = ("1", "H", "D", "T")):
        self.hydrogen_names = hydrogen_names

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        data["atom_array"] = remove_hydrogens(data["atom_array"], self.hydrogen_names)
        return data


class FilterToProteins(ApplyFunctionToAtomArray):
    """
    Filter atom array to only include protein residues.
    """

    def __init__(self, min_size: int = 5):
        super().__init__(func=lambda arr: arr[struc.filter_polymer(arr, pol_type="peptide", min_size=min_size)])


def filter_to_specified_pn_units(atom_array: AtomArray, pn_unit_iids: list | set | np.ndarray) -> AtomArray:
    """
    Filter atom array to only include specific PN units.
    """
    return atom_array[np.isin(atom_array.pn_unit_iid, pn_unit_iids)]


class FilterToSpecifiedPNUnits(Transform):
    """
    Filter atom array to only include specific PN units, denoted via the row metadata (held in `extra_info`).
    Such a filter is useful, for example, when during pre-processing we have identified clashing PN Units that we may want to exclude from the AtomArray.

    Args:
        - extra_info_key_with_pn_unit_iids_to_keep (str): The key in the "extra_info" dictionary that contains the PN unit IDs to filter to. If the key does not exist, the AtomArray is not filtered.
    """

    def __init__(self, extra_info_key_with_pn_unit_iids_to_keep: str = "all_pn_unit_iids_after_processing"):
        self.pn_unit_iid_key = extra_info_key_with_pn_unit_iids_to_keep

    def forward(self, data: dict) -> dict:
        if ("extra_info" not in data) or (self.pn_unit_iid_key not in data["extra_info"]):
            # ... short-circuit if the key does not exist in the `extra_info` dictionary
            return data
        else:
            # ... otherwise, filter the atom array
            data["atom_array"] = filter_to_specified_pn_units(
                data["atom_array"], eval(data["extra_info"][self.pn_unit_iid_key])
            )
            return data


def random_remove_pn_units_by_annotation_query(
    atom_array: AtomArray, query: str, delete_probability: float = 1.0, rng: np.random.Generator | None = None
) -> AtomArray:
    """Randomly remove pn_units from atom_array based on a query string with configurable probability.

    A pn_unit is considered to match the query if ALL atoms in the pn_unit satisfy the query condition.

    Args:
        atom_array: The AtomArray to filter
        query: Query string in atomworks.io Query syntax to identify pn_units to potentially delete
        delete_probability: Probability of deleting matched pn_units (0.0 = never delete, 1.0 = always delete)
        rng: Random number generator for probabilistic deletion
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply query to get mask of atoms matching the criteria
    query_expr = QueryExpression(query)
    matching_atoms_mask = query_expr.mask(atom_array)

    # Find pn_units where ALL atoms match the query
    pn_units_to_potentially_delete = []
    unique_pn_unit_iids = np.unique(atom_array.pn_unit_iid)

    for pn_unit_iid in unique_pn_unit_iids:
        pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
        pn_unit_atoms_matching_query = matching_atoms_mask[pn_unit_mask]

        # If ALL atoms in this pn_unit match the query, mark it for potential deletion
        if np.all(pn_unit_atoms_matching_query):
            pn_units_to_potentially_delete.append(pn_unit_iid)

    # Probabilistically decide which pn_units to delete
    if len(pn_units_to_potentially_delete) > 0:
        delete_mask = rng.random(len(pn_units_to_potentially_delete)) < delete_probability
        pn_units_to_delete = np.array(pn_units_to_potentially_delete)[delete_mask]

        # Filter out the selected pn_units
        if len(pn_units_to_delete) > 0:
            atoms_to_keep = ~np.isin(atom_array.pn_unit_iid, pn_units_to_delete)
            atom_array = atom_array[atoms_to_keep]

    return atom_array


class RandomlyRemovePNUnitsByAnnotationQuery(Transform):
    """Randomly remove pn_units from atom_array based on a query string with configurable probability.

    Args:
        query: Query string in atomworks.io query syntax to identify pn_units to potentially delete
        delete_probability: Probability of deleting matched pn_units (0.0 = never delete, 1.0 = always delete)
        rng_seed: Random seed for reproducibility (default: 42)
    """

    def __init__(self, query: str, delete_probability: float = 1.0, rng_seed: int = 42):
        self.query = query
        self.delete_probability = delete_probability
        self.rng_seed = rng_seed

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        rng = data.get("rng", np.random.default_rng(self.rng_seed))

        data["atom_array"] = random_remove_pn_units_by_annotation_query(
            data["atom_array"], self.query, self.delete_probability, rng
        )
        return data


class RandomlyRemoveLigands(RandomlyRemovePNUnitsByAnnotationQuery):
    """Randomly remove free-floating ligands (non-polymer ligands that are not covalent modifications)"""

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["is_polymer", "is_covalent_modification"])

    def __init__(self, delete_probability: float = 1.0, rng_seed: int = 42):
        query = "~is_polymer & ~is_covalent_modification"
        super().__init__(query=query, delete_probability=delete_probability, rng_seed=rng_seed)


class RemoveUnresolvedLigandAtomsIfTooMany(Transform):
    """
    If the number of unresolved (zero-occupancy) ligand atoms exceeds a specified threshold, remove all masked (zero-occupancy) ligand atoms from the atom array.

    This Transform is needed to avoid a significant proportion of the crop window from being filled with unresolved ligand atoms. Most commonly, this occurs with poorly resolved liposomes.

    Parameters:
        - unresolved_ligand_atom_limit(int): The maximum number of unresolved ligand atoms allowed in the atom array.

    Example: See PDB ID `6CLZ`, which contains a liposome with many unresolved atoms.
    """

    def __init__(self, unresolved_ligand_atom_limit: int):
        self.unresolved_ligand_atom_limit = unresolved_ligand_atom_limit

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Create a mask for unresolved ligand atoms
        is_ligand_atom = ~atom_array.is_polymer
        is_unresolved_atom = atom_array.occupancy == 0
        is_unresolved_ligand_atom = is_ligand_atom & is_unresolved_atom

        # If the number of unresolved ligand atoms exceeds the limit, remove all unresolved ligand atoms in the example
        num_unresolved_ligand_atoms = np.sum(is_unresolved_ligand_atom)
        if num_unresolved_ligand_atoms > self.unresolved_ligand_atom_limit:
            logger.info(
                f"Removing {num_unresolved_ligand_atoms} unresolved ligand atoms from the atom array, exceeding the limit of {self.unresolved_ligand_atom_limit}"
            )
            data["atom_array"] = atom_array[~is_unresolved_ligand_atom]

        return data


def remove_polymers_with_too_few_resolved_residues(atom_array: AtomArray, min_residues: int = 4) -> AtomArray:
    # ...loop over all polymer chains
    polymer_pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.is_polymer])
    for pn_unit_iid in polymer_pn_unit_iids:
        # ...count the number of resolved residues
        n_resolved_residues = len(
            np.unique(atom_array.res_id[(atom_array.pn_unit_iid == pn_unit_iid) & (atom_array.occupancy != 0)])
        )

        # ...if the number of resolved residues is less than the minimum, remove the polymer chain
        if n_resolved_residues < min_residues:
            atom_array = atom_array[atom_array.pn_unit_iid != pn_unit_iid]

    return atom_array


class RemovePolymersWithTooFewResolvedResidues(Transform):
    """
    From the AF-3 supplement, Section 2.5.4:
        > "Any polymer chain containing fewer than 4 resolved residues is filtered out."

    We implement this filter as a Transform that removes polymer chains with fewer than `min_residues` resolved residues.
    Note that upstream, we must ensure that the chosen query PN units are not polymer chains with too few resolved residues themselves.
    """

    def __init__(self, min_residues: int = 4):
        self.min_residues = min_residues

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy", "is_polymer"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = remove_polymers_with_too_few_resolved_residues(data["atom_array"], self.min_residues)
        return data


def remove_unresolved_atoms(atom_array: AtomArray, min_occupancy: float = 0.5) -> AtomArray:
    """
    Remove atoms with occupancy less than `min_occupancy` from the atom array.
    """
    return atom_array[atom_array.occupancy >= min_occupancy]


class RemoveUnresolvedAtoms(ApplyFunctionToAtomArray):
    def __init__(self, min_occupancy: float = 0.5):
        super().__init__(func=lambda arr: remove_unresolved_atoms(arr, min_occupancy))


class HandleUndesiredResTokens(Transform):
    """
    Remove, or otherwise handle, undesired residue tokens from the AtomArray.

    For undesired residue names `res_name`, the following actions are taken:
        - For undesired residues in non-polymer residues:
            - Remove the entire non-polymer (pn_unit_iid)
        - For undesired residues in polymer residues:
            - Map to the closest canonical residue name (if possible)
            - Else, map to an unknown residue name (if possible, i.e if backbone atoms are present)
            - Else, atomize
    """

    def __init__(self, undesired_res_tokens: list | tuple):
        """
        HandleUndesiredResTokens is a Transform that removes undesired residue tokens from an AtomArray.

        This class processes an AtomArray to identify and handle undesired residue names. The actions taken
        depend on whether the residues are part of a polymer or non-polymer. For non-polymer residues, the
        entire non-polymer unit is removed. For polymer residues, the undesired residue is mapped to the
        closest canonical residue name.

        Args:
            - undesired_res_tokens (list | tuple): A list or tuple of undesired residue names to be removed
              or mapped.

        Example:
            >>> transform = HandleUndesiredResTokens(undesired_res_tokens=["PTR", "SO4"])
        """
        self.undesired_res_tokens = undesired_res_tokens

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "is_polymer", "pn_unit_iid", "chain_type"])

    def _get_closest_canonical_residue(self, res_name: str, chain_type: int, force_unknown: bool = False) -> str:
        """Map a residue name to the closest canonical residue name."""
        one_letter_canonical = get_1_from_3_letter_code(
            res_name, chain_type=ChainType(chain_type), use_closest_canonical=not force_unknown
        )
        three_letter_canonical = get_3_from_1_letter_code(one_letter_canonical, chain_type=ChainType(chain_type))
        return three_letter_canonical

    @lru_cache(maxsize=1000)  # noqa: B019
    def _map_to_closest_canonical_residue(
        self, res_name: str, chain_type: int, has_hydrogens: bool, atom_name: tuple[str]
    ) -> tuple[np.ndarray, str]:
        """Map a residue name to the closest canonical residue name."""

        for force_unknown in (False, True):
            canonical_res_name = self._get_closest_canonical_residue(res_name, chain_type, force_unknown)
            canonical_res = struc.info.residue(canonical_res_name)

            # Remove hydrogens if non-canonical residue doesn't have hydrogens
            if not has_hydrogens:
                canonical_res = canonical_res[not_isin(canonical_res.element, HYDROGEN_LIKE_SYMBOLS)]

            # If canonical residue is a strict subset of the original residue,
            #  keep all matching atom names and delete the rest
            if np.all(np.isin(canonical_res.atom_name, atom_name)):
                to_keep = np.isin(atom_name, canonical_res.atom_name)
                # ... if we match without `force_unknown` break loop early
                return to_keep, canonical_res_name

        # If we could not find a canonical residue, or map to unknown, atomize the residue
        return np.ones(len(atom_name), dtype=bool), res_name

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if "atomize" not in atom_array.get_annotation_categories():
            atom_array.set_annotation("atomize", np.zeros(len(atom_array), dtype=bool))

        # Mark undesired residues
        to_remove = np.isin(atom_array.res_name, self.undesired_res_tokens)

        # Case 1: Undesired residue is part of non-polymer:
        #  - Remove the entire non-polymer (pn_unit_iid)
        is_undesired_non_poly = to_remove & (~atom_array.is_polymer)
        if np.any(is_undesired_non_poly):
            pn_unit_iids_to_remove = np.unique(atom_array.pn_unit_iid[is_undesired_non_poly])
            to_remove |= np.isin(atom_array.pn_unit_iid, pn_unit_iids_to_remove)

        # Case 2: Undesired residue is part of polymer:
        #  - Map to closest canonical residue
        is_undesired_poly = to_remove & atom_array.is_polymer
        if np.any(is_undesired_poly):
            # Iterate over all undesired residues
            _token_start_stop_idx = get_token_starts(atom_array, add_exclusive_stop=True)
            _token_starts = _token_start_stop_idx[:-1]
            _token_stops = _token_start_stop_idx[1:]
            undesired_poly_token_idxs = np.where(is_undesired_poly[_token_starts])[0]

            for token_idx in undesired_poly_token_idxs:
                token_start, token_stop = _token_starts[token_idx], _token_stops[token_idx]

                old_res_name = atom_array.res_name[token_start]
                to_keep, new_res_name = self._map_to_closest_canonical_residue(
                    res_name=old_res_name,
                    chain_type=atom_array.chain_type[token_start],
                    has_hydrogens=np.isin(atom_array.element[token_start:token_stop], HYDROGEN_LIKE_SYMBOLS).any(),
                    atom_name=tuple(atom_array.atom_name[token_start:token_stop]),  # tuple for hashability
                )

                # if new_res_name is the same as the original res_name (i.e. we didn't map to a canonical residue),
                # we atomize the residue
                if new_res_name == old_res_name:
                    atom_array.atomize[token_start:token_stop] = True

                # ... override the `to_remove` flag as `False` for the atoms that we want to keep
                to_remove[token_start:token_stop] = ~to_keep

                # ... override the old res name
                atom_array.res_name[token_start:token_stop] = new_res_name

        # Drop undesired residues
        atom_array = atom_array[~to_remove]
        data["atom_array"] = atom_array
        return data


def remove_protein_terminal_oxygen(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Remove terminal oxygen atoms (`OXT`) from protein chains.

    Terminal oxygen atoms are only removed from protein residues that are not atomized.
    """
    is_atomized = get_annotation(atom_array, "atomize", default=np.zeros(atom_array.array_length(), dtype=bool))
    is_terminal = atom_array.atom_name == "OXT"
    is_protein = np.isin(atom_array.chain_type, ChainTypeInfo.PROTEINS)
    remove = is_terminal & is_protein & ~is_atomized

    if isinstance(atom_array, AtomArray):
        return atom_array[~remove]
    elif isinstance(atom_array, AtomArrayStack):
        return atom_array[:, ~remove]


class RemoveTerminalOxygen(Transform):
    """Remove terminal oxygen atoms (`OXT`) from the atom array."""

    # TODO: Rename to RemoveProteinTerminalOxygen

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["atom_name", "chain_type"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = remove_protein_terminal_oxygen(data["atom_array"])
        return data


def remove_nucleic_acid_terminal_oxygen(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Remove terminal oxygen atoms (`OP3`) in nucleic acids from the atom array."""
    # Built mask non-atomized terminal oxygen atoms on nucleic acids
    is_atomized = get_annotation(
        atom_array, "atomize", n_body=1, default=np.zeros(atom_array.array_length(), dtype=bool)
    )
    is_terminal = atom_array.atom_name == "OP3"
    is_nucleic_acid = np.isin(atom_array.chain_type, ChainTypeInfo.NUCLEIC_ACIDS)

    remove = is_terminal & is_nucleic_acid & ~is_atomized

    if isinstance(atom_array, AtomArray):
        return atom_array[~remove]
    elif isinstance(atom_array, AtomArrayStack):
        return atom_array[:, ~remove]


class RemoveNucleicAcidTerminalOxygen(Transform):
    """
    Remove terminal oxygen atoms (`OP3`) in nucleic acids from the atom array.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["atom_name", "chain_type"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = remove_nucleic_acid_terminal_oxygen(data["atom_array"])
        data["atom_array"] = atom_array
        return data
