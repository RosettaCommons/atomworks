"""Utilities to create the dataset to be loaded by RF2AA. See the main script for term Glossary."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Final

import networkx as nx
import numpy as np
import requests
from biotite.structure import AtomArray, CellList
from scipy.spatial.distance import cdist

from atomworks.common import default, not_isin
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import ClashSeverity

logger = logging.getLogger("preprocess")


def get_pn_units_with_non_biological_bonds(atom_array: AtomArray, bond_mask: np.ndarray) -> np.ndarray:
    """
    Checks for non-biological bonds between PN units within an assembly.
    Note that "inter-PN-unit bonds" in this instance do not include bonds between non-polymers,
    which are treated as the same PN unit (by definition).

    Specifically, this function looks for inter-molecular:
    - Oxygen-oxygen bonds
    - Fluorine-fluorine bonds
    - Bonds involving free oxygen or hydroxyl groups (e.g. HOH, OH, O)

    Args:
        atom_array: AtomArray containing the relevant structure
        bond_mask: Mask of inter-molecular bonds

    Returns:
        numpy.ndarray: Array of polymer/non-polymer unit instance IDs that contain non-biological bonds
    """
    # Get atoms and residues involved in the inter-molecular bonds
    bonds_to_check = atom_array.bonds.as_array()[bond_mask]
    atom_a_elements = atom_array.atomic_number[bonds_to_check[:, 0]]
    atom_b_elements = atom_array.atomic_number[bonds_to_check[:, 1]]
    atom_a_res_names = atom_array.res_name[bonds_to_check[:, 0]]
    atom_b_res_names = atom_array.res_name[bonds_to_check[:, 1]]

    # Check for non-biological bonds
    non_biological_bonds = (
        (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])
        & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])  # Oxygen-oxygen bonds
        | (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])
        & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])  # Fluorine-fluorine bonds
        | np.isin(atom_a_res_names, ["HOH", "OH", "O"])  # Bonds involving free oxygen or hydroxyl groups
        | np.isin(atom_b_res_names, ["HOH", "OH", "O"])
    )

    # Get unique polymer/non-polymer unit IDs with non-biological bonds
    return np.unique(
        np.concatenate(
            [
                atom_array.pn_unit_iid[bonds_to_check[:, 0]][non_biological_bonds],
                atom_array.pn_unit_iid[bonds_to_check[:, 1]][non_biological_bonds],
            ]
        )
    )


def get_clashing_pn_units(
    pn_unit_iids_to_consider: np.Array, atom_array: AtomArray, cell_list: CellList, clash_distance: float
) -> tuple[set[str], dict[str, set[str]]]:
    """Wrapper function to find clashing PN units within an atom array."""
    # Process PN unit-by-PN unit to avoid memory issues
    clashing_pn_units_dict = defaultdict(set)
    clashing_pn_units_set = set()
    for query_pn_unit_iid in pn_unit_iids_to_consider:
        query_pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == query_pn_unit_iid]
        clashing_pn_units = get_pn_units_clashing_with_pn_unit(
            query_pn_unit_atom_array, atom_array, cell_list, clash_distance
        )
        if clashing_pn_units:
            clashing_pn_units_set.update(clashing_pn_units | {query_pn_unit_iid})
            clashing_pn_units_dict[query_pn_unit_iid] = set(clashing_pn_units)
    return clashing_pn_units_set, clashing_pn_units_dict


def get_atom_mask_from_cell_list(
    coord: np.array, cell_list: CellList, cell_list_size: int, cutoff: float, chunk_size: int = int(2e9)
) -> np.ndarray:
    """
    Builds a mask indicating which atoms clash with the query PN unit. If the number of comparisons is too large,
    the computation is split into manageable chunks along the rows of `coord`.

    TODO: Update documentation since this is not specific to PN units or clashes.

    Args:
        coord (ndarray): The coordinates of the query PN unit. Shape is (n, 3).
        cell_list (CellList): A CellList object that allows efficient vicinity searches.
        cell_list_size (int): The number of atoms in the cell list.
        clash_distance (float): The distance threshold below which atoms are considered to be clashing.
        chunk_size (int): The maximum number of comparisons allowed in a single chunk.

    Returns:
        ndarray: Mask indicating which atoms in `cell_list` clash with the  atoms in `coord`. Shape is (n, cell_list_size), dtype is bool.
    """
    num_coords = coord.shape[0]
    clashing_atom_mask = np.zeros((num_coords, cell_list_size), dtype=bool)

    max_rows_per_chunk = max(1, chunk_size // cell_list_size)  # Ensure at least 1 row per chunk
    if num_coords * cell_list_size > chunk_size:
        logger.info(
            f"{num_coords * cell_list_size:,} comparisons needed; distance computation split into {math.ceil(num_coords / max_rows_per_chunk)} chunks."
        )
        for i in range(0, num_coords, max_rows_per_chunk):
            end = min(i + max_rows_per_chunk, num_coords)
            clashing_atom_mask[i:end, :] = cell_list.get_atoms(coord[i:end], cutoff, as_mask=True)
    else:
        clashing_atom_mask = cell_list.get_atoms(coord, cutoff, as_mask=True)

    return clashing_atom_mask


def get_pn_units_clashing_with_pn_unit(
    query_pn_unit: AtomArray, filtered_atom_array: AtomArray, cell_list: CellList, clash_distance: float
) -> set[str]:
    """
    Finds clashes between a query PN unit and the rest of the structure.
    A clash is defined as any pair of atoms from the query PN unit and the rest of the structure that are closer than `clash_distance`.

    Args:
        query_pn_unit: AtomArray containing the query PN unit that we want to check for clashes
        filtered_atom_array: AtomArray containing atoms with non-zero occupancy
        cell_list: CellList of structure for rapid distance computations
        clash_distance: Distance threshold for clashing atoms

    Returns:
        Set of polymer/non-polymer unit instance IDs that are clashing with the query PN unit
    """
    clashing_atom_mask = get_atom_mask_from_cell_list(
        query_pn_unit.coord, cell_list, len(filtered_atom_array), clash_distance
    )
    collapsed_mask = np.any(clashing_atom_mask, axis=0)
    clashing_atoms = filtered_atom_array[collapsed_mask]

    # Filter out query PN unit atoms
    query_pn_unit_iids = np.unique(query_pn_unit.pn_unit_iid)
    clashing_atoms = clashing_atoms[not_isin(clashing_atoms.pn_unit_iid, query_pn_unit_iids)]
    return set(np.unique(clashing_atoms.pn_unit_iid))


def handle_clashing_pn_units(
    clashing_pn_units_set: set, clashing_pn_units_dict: dict, atom_array: AtomArray, pn_unit_iid_map: dict
) -> AtomArray:
    """
    Resolves clashing PN units according to the following process:
    1. Sort clashing PN units by the number of atoms within the PN unit
    2. Iterate through the sorted list keeping (1) the larger PN unit (2) the lower transformation number until all clashes are resolved

    Args:
        clashing_pn_units_set: Set of polymer/non-polymer unit instance IDs that contain clashing atoms
        clashing_pn_units_dict: Dictionary mapping polymer/non-polymer unit instance IDs to a list of clashing polymer/non-polymer unit instance IDs
        atom_array: AtomArray containing atoms with non-zero occupancy
        pn_unit_iid_map: Dictionary mapping integer representations of polymer/non-polymer unit instance IDs to the polymer/non-polymer unit instance IDs themselves

    Returns:
        AtomArray: AtomArray with clashing atoms removed
        ClashSeverity: Enum representing the severity of the clash
    """
    pn_units_to_remove = set()
    pn_units_to_keep = set()

    # Build a dictionary of clashing PN unit details
    clashing_pn_unit_details = {}
    for pn_unit in clashing_pn_units_set:
        pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == pn_unit]
        clashing_pn_unit_details[pn_unit] = {
            "num_atoms": len(pn_unit_atom_array),
            "is_metal": pn_unit_atom_array[0].element.upper in METAL_ELEMENTS,
            "is_polymer": pn_unit_atom_array[0].is_polymer,
        }

    # Sort clashing PN units by the number of atoms in the reference PN unit (high to low) and then the transformation ID (low to high)
    # The last character in the polymer/non-polymer unit instance ID is the transformation ID, even in cases where we have covalently bound chains (e.g., A_1,B_1)
    def sort_key(x: tuple[str, dict[str, Any]]) -> tuple[int, int]:
        try:
            return (x[1]["num_atoms"], -int(pn_unit_iid_map[x[0]][-1]))
        except ValueError:
            return (x[1]["num_atoms"], -ord(pn_unit_iid_map[x[0]][-1]))

    sorted_clashing_pn_units = [x[0] for x in sorted(clashing_pn_unit_details.items(), key=sort_key, reverse=True)]

    # Define the ClashSeverity
    num_polymers = len(
        np.unique(atom_array.pn_unit_iid[atom_array.is_polymer])
    )  # Number of polymer PN units in the structure
    num_clashing_polymers = len(
        np.unique([pn_unit for pn_unit in clashing_pn_units_set if clashing_pn_unit_details[pn_unit]["is_polymer"]])
    )  # Number of clashing polymer PN units

    if num_clashing_polymers / num_polymers > 0.5:
        clash_severity = ClashSeverity.SEVERE
    elif num_clashing_polymers > 0:
        clash_severity = ClashSeverity.MODERATE
    else:
        clash_severity = ClashSeverity.MILD

    # Keep the larger PN unit
    for pn_unit in sorted_clashing_pn_units:
        if pn_unit not in pn_units_to_remove:
            pn_units_to_keep.add(pn_unit)
            pn_units_to_remove.update(clashing_pn_units_dict[pn_unit] - pn_units_to_keep)
    logger.warning(
        f"Removing clashing PN units: {[pn_unit_iid_map[pn_unit] for pn_unit in pn_units_to_remove]} from the structure."
    )

    # Remove clashing PN units
    atom_array = atom_array[not_isin(atom_array.pn_unit_iid, list(pn_units_to_remove))]
    return atom_array, clash_severity


def get_contacting_pn_units(
    query_pn_unit: AtomArray,
    filtered_atom_array: AtomArray,
    cell_list: CellList,
    contact_distance: float = 4.5,
    min_contacts_required: float = 1,
    mask: np.ndarray = None,
    calculate_min_distance: bool = False,
) -> list[dict[str, str]]:
    """
    Finds PN units (proteins, nucleic acids, or small molecules) with a minimum number of atoms within a given distance of the query PN unit.

    Args:
        query_pn_unit: AtomArray containing the query PN unit that we want to check for contacts (could be a single chain or multiple covalently bonded chains)
        filtered_atom_array: AtomArray containing the set of atoms to consider when looking for contacts (must correspond to the `cell_list`)
        cell_list: CellList of `atom_array` for rapid distance computations; must correspond to the `filtered_atom_array`
        contact_distance: Distance threshold for contacting atoms
        min_contacts_required: Minimum number of atoms within the cutoff distance to consider a PN unit as a potential partner
        mask: Mask of PN units to consider as potential partners within the atom_array. If None, all PN units (except the query) are considered.
        calculate_min_distance: Whether to calculate the minimum distance between the query PN unit and each PN unit within the cutoff distance

    Returns:
        list: A list of dictionaries, each containing:
            - 'pn_unit_iid' (str): A string representing a unique partner PN unit in contact with the query PN unit.
            - 'num_atoms' (int): Number of non-zero occupancy atoms in the partner PN unit.
            - 'num_contacts' (int): Number of atoms in the partner PN unit within the contact_distance of the query PN unit.
            - 'min_distance' (float): Minimum distance between the query PN unit and the partner PN unit.
    """
    contacting_pn_unit_summary = []

    # ---------- Step 1: Find and count all atoms within contact_distance ---------- #
    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        query_pn_unit.coord, cell_list, len(filtered_atom_array), contact_distance
    )  # (n_query, n_cell_list)
    collapsed_contacting_atom_mask = np.any(full_contacting_atom_mask, axis=0)  # (n_cell_list,)

    # Filter out unwanted atoms (either those in the query PN unit, or those not in the potential partner mask)
    contacting_atoms_mask = (
        mask & collapsed_contacting_atom_mask if mask is not None else collapsed_contacting_atom_mask
    )
    # Create a mask for atoms that are not part of the query PN unit
    non_query_atoms_mask = not_isin(filtered_atom_array.pn_unit_iid, np.unique(query_pn_unit.pn_unit_iid))
    contacting_atoms_mask = contacting_atoms_mask & non_query_atoms_mask
    # Using the final mask to get contacting atoms
    contacting_atoms = filtered_atom_array[contacting_atoms_mask]

    # ---------- Step 2: Calculate the minimum distance to each contacting PN unit ---------- #
    min_distances = {}
    if calculate_min_distance and len(contacting_atoms) > 0:
        # Get a list of PN units within `contact_distance`
        close_atom_pn_unit_ids = np.unique(contacting_atoms.pn_unit_iid)
        atoms_to_check = contacting_atoms

        # Perform distance calculation and store the minimum distance for each PN unit
        distances = cdist(query_pn_unit.coord, atoms_to_check.coord)
        for pn_unit_iid in close_atom_pn_unit_ids:
            mask = atoms_to_check.pn_unit_iid == pn_unit_iid
            min_distances[pn_unit_iid] = np.min(distances[:, mask]) if np.sum(mask) > 0 else None

    # ---------- Step 3: Apply criteria and add to list ---------- #
    for pn_unit_iid in np.unique(contacting_atoms.pn_unit_iid):
        contacting_pn_unit_atoms = contacting_atoms_mask & (filtered_atom_array.pn_unit_iid == pn_unit_iid)
        # Count number of pairwise contacts (which is lower bounded by the number of contacting atoms)
        pairwise_contacts = int(np.sum(full_contacting_atom_mask[:, contacting_pn_unit_atoms]))

        if pairwise_contacts >= min_contacts_required:
            contacting_pn_unit_summary.append(
                {
                    "pn_unit_iid": pn_unit_iid,
                    "num_atoms": len(filtered_atom_array[filtered_atom_array.pn_unit_iid == pn_unit_iid]),
                    "num_contacts": pairwise_contacts,
                    "min_distance": min_distances[pn_unit_iid] if min_distances else None,
                }
            )

    return contacting_pn_unit_summary


def get_intra_pn_unit_bonds(pn_unit_iid: str, full_atom_array: AtomArray) -> np.ndarray:
    # NOTE: Not currently used; kept for potential future use
    """
    Retrieve all bonds within the PN unit.
    Includes inter-chain bonds if the PN unit is composed of multiple chains.
    Does NOT include bonds between the PN unit and any other PN units (e.g., protein-ligand bonds).

    Args:
        pn_unit_iid (str): The polymer/non-polymer unit instance ID (e.g., 'A1', 'C2,B3', etc.)
        full_atom_array (AtomArray): The full structure AtomArray. Must include all atoms, including those with zero occupancy.

    Returns:
        numpy.ndarray: Array of Biotite bond objects (atom_a_index, atom_b_index, bond_type) within the specified PN unit.
    """
    all_bonds = full_atom_array.bonds.as_array()
    atom_a_indices, atom_b_indices = all_bonds[:, 0], all_bonds[:, 1]
    mask = (full_atom_array.pn_unit_iid[atom_a_indices] == pn_unit_iid) & (
        full_atom_array.pn_unit_iid[atom_b_indices] == pn_unit_iid
    )
    return all_bonds[mask]


def calculate_molecule_diameter(full_molecule_atoms: AtomArray) -> float:
    # NOTE: Not currently used; kept for potential future use
    """
    Calculates the molecular diameter, defined as the maximum number of bonds between any two vertices
    in the molecule, using a maximum spanning tree.

    Args:
        full_molecule_atoms (AtomArray): The array of atoms representing the molecule. Must include all atoms, including those with zero occupancy.

    Returns:
        float: The molecular diameter. If the diameter cannot be computed, returns 0.0.
    """
    if len(full_molecule_atoms.bonds.as_array() > 0):
        try:
            nx_graph = full_molecule_atoms.bonds.as_graph()  # as a NetworkX graph
            tree = nx.maximum_spanning_tree(nx_graph)
            return nx.diameter(tree)
        except nx.exception.NetworkXError:
            logger.warning("Could not compute diameter.")
            return 0.0
    else:
        return 0.0


def get_soi_ligands_from_pdb_id(pdb_id: str) -> set[str]:
    """
    This function takes a PDB ID and returns a set of ligand names that are annotated as subject of
    investigation (SOI) in the PDB. Such ligands are often considered
    biologically meaningful.

    NOTE: This function is kept from the old pipeline for convenience and testing,
    but not used in the current processing pipeline, where the LOI is extracted directly
    from the cif file instead of via an FTP query.

    Args:
        pdb_id (str): The PDB ID to query.

    Returns:
        Set[str]: A set of ligand names that are annotated as SOI in the PDB.
    """
    try:
        pdb_id = pdb_id.lower()
        core_response = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
        core_json_dict = core_response.json()
        nonpolymer_ids = core_json_dict["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids", [])
        soi_ligand_names = []

        for nonpolymer_id in nonpolymer_ids:
            ligand_response = requests.get(
                f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{nonpolymer_id}"
            )
            ligand_json_dict = ligand_response.json()
            ligand_name = ligand_json_dict["pdbx_entity_nonpoly"]["comp_id"]

            if "rcsb_nonpolymer_entity_annotation" in ligand_json_dict:
                for annotation_dict in ligand_json_dict["rcsb_nonpolymer_entity_annotation"]:
                    if annotation_dict["type"] == "SUBJECT_OF_INVESTIGATION" and annotation_dict["name"] == ligand_name:
                        soi_ligand_names.append(ligand_name)
                        break
    except Exception:
        return set()
    return set(soi_ligand_names)


def get_ligand_validity_scores_from_pdb_id(pdb_id: str) -> list[dict[str, str | int | float | None]]:
    """
    Query the RCSB PDB for ligand validity scores for a given PDB ID.

    Args:
        pdb_id (str): The PDB ID to query.

    Returns:
        records: (list[dict[str, str | int | float | None]]): A list of dictionaries, each containing
            the ligand validity scores for a ligand (e.g. RSCC, RSR) as well identifiers such as the
            residue name, chain ID, and entity ID. Can easily be converted to a pandas DataFrame for
            easier handling via `pd.DataFrame(records)`.

    Reference:
        `RCSB Ligand Structure Quality Guide <https://www.rcsb.org/docs/general-help/ligand-structure-quality-in-pdb-structures>`_
    """
    pdb_graphql_url: Final[str] = "https://data.rcsb.org/graphql"

    # Query string in graphql language to get ligand validity scores from a PDB entry
    ligand_validity_query: Final[str] = """
    query ($id: String!) {
        entry(entry_id:$id){
            nonpolymer_entities {
                rcsb_nonpolymer_entity_container_identifiers {
                    nonpolymer_comp_id
                    rcsb_id
                }
                rcsb_nonpolymer_entity_annotation {
                    type
                }
                nonpolymer_entity_instances {
                    rcsb_nonpolymer_entity_instance_container_identifiers {
                        auth_seq_id
                        auth_asym_id
                        asym_id
                        entity_id
                        entry_id
                    }
                    rcsb_nonpolymer_instance_validation_score {
                        RSCC
                        RSR
                        alt_id
                        completeness
                        intermolecular_clashes
                        is_best_instance
                        mogul_angle_outliers
                        mogul_angles_RMSZ
                        mogul_bond_outliers
                        mogul_bonds_RMSZ
                        ranking_model_fit
                        ranking_model_geometry
                        score_model_fit
                        score_model_geometry
                        stereo_outliers
                        average_occupancy
                        type
                        is_subject_of_investigation
                        is_subject_of_investigation_provenance
                    }
                }
            }
        }
    }
    """

    # Perform the actual query for the target PDB ID
    response = requests.post(pdb_graphql_url, json={"query": ligand_validity_query, "variables": {"id": pdb_id}})

    # Extract the records from the response
    records = []
    if response.status_code == 200:
        data = response.json()
        try:
            nonpolymer_entities = default(data["data"]["entry"]["nonpolymer_entities"], [])
            for entity in nonpolymer_entities:
                res_name = entity["rcsb_nonpolymer_entity_container_identifiers"]["nonpolymer_comp_id"]
                for instance in entity.get("nonpolymer_entity_instances", []):
                    record_template = {"res_name": res_name}
                    record_template.update(instance.get("rcsb_nonpolymer_entity_instance_container_identifiers", {}))
                    validation_scores = default(instance["rcsb_nonpolymer_instance_validation_score"], [])
                    for score in validation_scores:
                        record = record_template.copy()
                        record.update(score)
                        records.append(record)
        except KeyError:
            logger.debug(f"No validation scores found for PDB ID: {pdb_id}")
        except TypeError:
            logger.debug(f"No validation scores found for PDB ID: {pdb_id}")
    else:
        logger.debug(f"Query failed with status code {response.status_code} and response: {response.text}")
    return records


def get_inter_pn_unit_bond_mask(atom_array: AtomArray) -> np.ndarray:
    """
    Returns a mask indicating which bonds in `atom_array.bonds` are between two distinct PN units.
    Because we are operating at the PN unit-level, such bonds cannot be bonds between non-polymers.

    WARNING: Must be applied before reassigning PN unit IIDs (e.g., as is done for covalent modifications).

    Args:
        atom_array (AtomArray): The full atom array. Must have PN unit-level annotations.

    Returns:
        numpy.ndarray: A boolean mask indicating which bonds are between two PN units.
    """
    bond_pn_unit_a = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 0]]
    bond_pn_unit_b = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 1]]
    return bond_pn_unit_a != bond_pn_unit_b


def get_bonded_polymer_pn_units(query_pn_unit_iid: str, filtered_atom_array: AtomArray) -> set[str]:
    """
    Returns a set of polymer PN units that are covalently bonded to a given PN unit.
    For example, useful to detect oligosaccharides that are covalently bonded to a protein.

    Args:
        query_pn_unit_iid (str): The full ID of the non-polymer PN unit to check for bonds.
        filtered_atom_array (AtomArray): AtomArray with non-zero occupancy

    Returns:
        set[str]: A set of full IDs of polymer PN units that are covalently bonded to the query PN unit.
    """
    # Check if the non polymer is covalently bonded to a polymer
    inter_pn_unit_bonds = filtered_atom_array.bonds.as_array()[get_inter_pn_unit_bond_mask(filtered_atom_array)]
    bond_atom_a_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 0]]
    bond_atom_b_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 1]]

    # Find bonded PN units
    bonded_pn_units = set(bond_atom_b_pn_unit_iids[np.where(bond_atom_a_pn_unit_iids == query_pn_unit_iid)[0]]) | set(
        bond_atom_a_pn_unit_iids[np.where(bond_atom_b_pn_unit_iids == query_pn_unit_iid)[0]]
    )

    # Get the set of polymers
    polymer_pn_unit_iids = set(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer])

    # Set intersection between bonded PN units and polymer PN units
    return bonded_pn_units & polymer_pn_unit_iids
