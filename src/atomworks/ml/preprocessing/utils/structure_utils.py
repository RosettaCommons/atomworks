"""Utilities for preprocessing PDB structures."""

import logging
import math
import warnings
from collections import defaultdict
from typing import Any, Final, Literal

import numpy as np
import requests
from biotite.structure import AtomArray, CellList
from scipy.spatial import cKDTree

from atomworks.common import default, not_isin
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import ClashSeverity

logger = logging.getLogger("preprocess")


def get_non_biological_bond_atom_mask(atom_array: AtomArray) -> np.ndarray:
    """Return mask of atoms involved in non-biological inter-PN-unit bonds."""
    inter_bond_mask = get_inter_pn_unit_bond_mask(atom_array)

    if not np.any(inter_bond_mask):
        return np.zeros(len(atom_array), dtype=bool)

    bonds = atom_array.bonds.as_array()[inter_bond_mask]

    atom_a_elements = atom_array.atomic_number[bonds[:, 0]]
    atom_b_elements = atom_array.atomic_number[bonds[:, 1]]
    atom_a_res_names = atom_array.res_name[bonds[:, 0]]
    atom_b_res_names = atom_array.res_name[bonds[:, 1]]

    # Identify non-biological bonds
    non_bio = (
        (
            (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])
            & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])
        )
        | (
            (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])
            & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])
        )
        | np.isin(atom_a_res_names, ["HOH", "OH", "O"])
        | np.isin(atom_b_res_names, ["HOH", "OH", "O"])
    )

    # Build atom mask - mark both atoms in each non-biological bond
    result = np.zeros(len(atom_array), dtype=bool)
    result[bonds[non_bio, 0]] = True
    result[bonds[non_bio, 1]] = True
    return result


def get_clashing_pn_units(
    pn_unit_iids_to_consider: np.ndarray,
    atom_array: AtomArray,
    clash_distance: float,
) -> dict[str, set[str]]:
    """Find clashing PN units within an atom array.

    Uses cKDTree.query_pairs for efficient O(n log n) pair detection.
    """
    tree = cKDTree(atom_array.coord)
    pairs = tree.query_pairs(r=clash_distance, output_type="ndarray")

    if len(pairs) == 0:
        return {}

    # Filter to inter-PN-unit pairs
    pn_a = atom_array.pn_unit_iid[pairs[:, 0]]
    pn_b = atom_array.pn_unit_iid[pairs[:, 1]]
    clashing_pairs = pairs[pn_a != pn_b]

    if len(clashing_pairs) == 0:
        return {}

    # Build result dict for requested PN units only
    iids_set = set(pn_unit_iids_to_consider)
    clashing_dict: dict[str, set[str]] = {}

    for i, j in clashing_pairs:
        iid_i = atom_array.pn_unit_iid[i]
        iid_j = atom_array.pn_unit_iid[j]
        if iid_i in iids_set:
            clashing_dict.setdefault(iid_i, set()).add(iid_j)
        if iid_j in iids_set:
            clashing_dict.setdefault(iid_j, set()).add(iid_i)

    return clashing_dict


def get_atom_mask_from_cell_list(
    coord: np.ndarray,
    cell_list: CellList,
    cell_list_size: int,
    cutoff: float,
    *,
    output_mode: Literal["pairwise", "any_target"] = "pairwise",
    chunk_size: int = int(2e9),
) -> np.ndarray:
    """Return distance-based mask between query coords and cell_list atoms.

    .. deprecated::
        Use :py:class:`scipy.spatial.cKDTree` instead. See warning message for details.

    Large queries are split into chunks for memory efficiency.

    Args:
        coord: Query coordinates, shape (n, 3).
        cell_list: CellList for efficient vicinity searches.
        cell_list_size: Number of atoms in cell_list.
        cutoff: Distance threshold.
        output_mode: Output format:
            - "pairwise": Full 2D mask (n_query, cell_list_size)
            - "any_target": 1D mask (cell_list_size,) collapsed via np.any(axis=0)
        chunk_size: Max comparisons per chunk for memory efficiency.

    Returns:
        Boolean mask. Shape is (n_query, cell_list_size) for "pairwise" mode,
        or (cell_list_size,) for "any_target" mode.
    """
    warnings.warn(
        "get_atom_mask_from_cell_list is deprecated. "
        "Use scipy.spatial.cKDTree instead:\n"
        "  tree = cKDTree(target_coords)\n"
        "  nearby = tree.query_ball_point(query_coords, r=cutoff)\n"
        "For pair detection, use tree.query_pairs(r=cutoff).",
        DeprecationWarning,
        stacklevel=2,
    )
    num_coords = coord.shape[0]
    max_rows_per_chunk = max(1, chunk_size // cell_list_size)

    if output_mode == "any_target":
        # Memory-efficient: collapse during chunking
        result = np.zeros(cell_list_size, dtype=bool)
        for i in range(0, num_coords, max_rows_per_chunk):
            end = min(i + max_rows_per_chunk, num_coords)
            chunk_mask = cell_list.get_atoms(coord[i:end], cutoff, as_mask=True)
            result |= np.any(chunk_mask, axis=0)
        return result

    # output_mode == "pairwise": full 2D mask
    pairwise_mask = np.zeros((num_coords, cell_list_size), dtype=bool)

    if num_coords * cell_list_size > chunk_size:
        logger.info(
            f"{num_coords * cell_list_size:,} comparisons needed; "
            f"distance computation split into {math.ceil(num_coords / max_rows_per_chunk)} chunks."
        )
        for i in range(0, num_coords, max_rows_per_chunk):
            end = min(i + max_rows_per_chunk, num_coords)
            pairwise_mask[i:end, :] = cell_list.get_atoms(coord[i:end], cutoff, as_mask=True)
    else:
        pairwise_mask = cell_list.get_atoms(coord, cutoff, as_mask=True)

    return pairwise_mask


def handle_clashing_pn_units(
    clashing_pn_units_dict: dict[str, set[str]], atom_array: AtomArray
) -> tuple[AtomArray, ClashSeverity]:
    """Resolve clashing PN units by keeping larger units."""
    clashing_pn_units_set = set(clashing_pn_units_dict.keys())
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

    def sort_key(x: tuple[str, dict[str, Any]]) -> tuple[int, int]:
        try:
            return (x[1]["num_atoms"], -int(x[0][-1]))
        except ValueError:
            return (x[1]["num_atoms"], -ord(x[0][-1]))

    sorted_clashing_pn_units = [x[0] for x in sorted(clashing_pn_unit_details.items(), key=sort_key, reverse=True)]

    # Define the ClashSeverity
    num_polymers = len(np.unique(atom_array.pn_unit_iid[atom_array.is_polymer]))
    num_clashing_polymers = len(
        np.unique([pn_unit for pn_unit in clashing_pn_units_set if clashing_pn_unit_details[pn_unit]["is_polymer"]])
    )

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
    logger.warning(f"Removing clashing PN units: {list(pn_units_to_remove)} from the structure.")

    # Remove clashing PN units
    atom_array = atom_array[not_isin(atom_array.pn_unit_iid, list(pn_units_to_remove))]
    return atom_array, clash_severity


def get_contacting_pn_units(
    atom_array: AtomArray,
    query_mask: np.ndarray,
    target_mask: np.ndarray,
    contact_distance: float,
    min_contacts_required: int = 1,
    calculate_min_distance: bool = False,
    *,
    tree: cKDTree,
) -> list[dict[str, str | int | float | None]]:
    """Find PN units with atoms within contact distance of query atoms.

    Uses cKDTree.query_ball_point for efficient sparse neighbor lookup.
    """
    query_coords = atom_array.coord[query_mask]
    if len(query_coords) == 0 or not np.any(target_mask):
        return []

    # Query returns global indices into atom_array
    nearby_lists_raw = tree.query_ball_point(query_coords, r=contact_distance)

    # Filter to target indices using vectorized boolean lookup (O(n) vs O(n*m) set lookup)
    nearby_lists = [[idx for idx in nearby if target_mask[idx]] for nearby in nearby_lists_raw]

    # Track unique contact pairs per target PN unit and min distances
    contact_pairs: dict[str, set[tuple[int, int]]] = defaultdict(set)
    min_dists: dict[str, float] = {}

    for q_idx, nearby in enumerate(nearby_lists):
        if not nearby:
            continue
        q_coord = query_coords[q_idx]
        for idx in nearby:
            iid = atom_array.pn_unit_iid[idx]
            contact_pairs[iid].add((q_idx, idx))
            if calculate_min_distance:
                t_coord = atom_array.coord[idx]
                dist = float(np.linalg.norm(q_coord - t_coord))
                if iid not in min_dists or dist < min_dists[iid]:
                    min_dists[iid] = dist

    if not contact_pairs:
        return []

    # Precompute PN unit sizes for contacting units only
    pn_unit_sizes = {iid: int(np.sum(atom_array.pn_unit_iid == iid)) for iid in contact_pairs}

    return [
        {
            "pn_unit_iid": iid,
            "num_atoms": pn_unit_sizes[iid],
            "num_contacts": len(pairs),
            "min_distance": min_dists.get(iid) if calculate_min_distance else None,
        }
        for iid, pairs in contact_pairs.items()
        if len(pairs) >= min_contacts_required
    ]


def get_ligand_validity_scores_from_pdb_id(pdb_id: str) -> list[dict[str, str | int | float | None]]:
    """Query the RCSB PDB for ligand validity scores for a given PDB ID."""
    pdb_graphql_url: Final[str] = "https://data.rcsb.org/graphql"

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

    response = requests.post(pdb_graphql_url, json={"query": ligand_validity_query, "variables": {"id": pdb_id}})

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
    """Returns a mask indicating which bonds are between two distinct PN units."""
    bond_pn_unit_a = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 0]]
    bond_pn_unit_b = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 1]]
    return bond_pn_unit_a != bond_pn_unit_b


def get_bonded_polymer_pn_units(query_pn_unit_iid: str, filtered_atom_array: AtomArray) -> set[str]:
    """Returns a set of polymer PN units that are covalently bonded to a given PN unit."""
    inter_pn_unit_bonds = filtered_atom_array.bonds.as_array()[get_inter_pn_unit_bond_mask(filtered_atom_array)]
    bond_atom_a_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 0]]
    bond_atom_b_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 1]]

    bonded_pn_units = set(bond_atom_b_pn_unit_iids[np.where(bond_atom_a_pn_unit_iids == query_pn_unit_iid)[0]]) | set(
        bond_atom_a_pn_unit_iids[np.where(bond_atom_b_pn_unit_iids == query_pn_unit_iid)[0]]
    )

    polymer_pn_unit_iids = set(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer])
    return bonded_pn_units & polymer_pn_unit_iids
