"""Preprocessing pipeline for PDB structures.

Provides a pipeline for preprocessing PDB structures:
1. parse_entry() - Load structure and fetch entry-level metadata like ligand fit-to-density
2. process_assemblies() - Filter atoms, resolve clashes, create Assembly records
3. process_pn_units() - Extract PN units and contacts
4. process_interfaces() - Create Interface records from contacts
5. _add_closest_chain_info() - Add closest_20/50_pn_unit_iids to assemblies (uses outputs from steps 1-4)
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

import atomworks.ml.preprocessing.utils.structure_utils as structure_utils
from atomworks.common import exists, not_isin
from atomworks.constants import (
    CRYSTALLIZATION_AIDS,
    METAL_ELEMENTS,
    PEPTIDE_MAX_RESIDUES,
    STANDARD_AND_UNKNOWN_POLYMER_RESIDUES,
    STANDARD_POLYMER_RESIDUES,
)
from atomworks.enums import ChainType
from atomworks.io import parse
from atomworks.ml.preprocessing.constants import (
    PDB_IDS_WITH_UNPHYSICAL_BONDS,
    ClashSeverity,
    DistanceThresholds,
)
from atomworks.ml.preprocessing.records import (
    Assembly,
    ContactInfo,
    Interface,
    PNUnit,
    serialize_record,
)
from atomworks.ml.preprocessing.utils.large_assembly import sample_large_assembly_interface
from atomworks.ml.preprocessing.utils.token import count_af3_style_tokens
from atomworks.ml.transforms.atom_array import add_global_atom_id_annotation
from atomworks.ml.transforms.atomize import atomize_by_ccd_name
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger("preprocess")


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing pipeline.

    All parameters have sensible defaults. Create custom configs by passing
    only the parameters you want to change.
    """

    # Distance thresholds (Angstroms)
    contact_distance: float = DistanceThresholds.CONTACT
    clash_distance: float = DistanceThresholds.CLASH

    # Filtering options
    ignore_residues: tuple[str, ...] = ()
    polymer_pn_unit_limit: int = 1000

    # Parse options
    build_assembly: str = "all"
    add_missing_atoms: bool = True  # Required for accurate token counts, bonds, etc.
    remove_waters: bool = True
    remove_ccds: tuple[str, ...] = tuple(CRYSTALLIZATION_AIDS)
    fix_ligands_at_symmetry_centers: bool = True
    fix_arginines: bool = True
    convert_mse_to_met: bool = True
    hydrogen_policy: Literal["remove", "infer", "keep"] = "remove"

    # Ligand scores to fetch from RCSB
    ligand_scores: tuple[str, ...] = (
        "RSCC",
        "RSR",
        "completeness",
        "intermolecular_clashes",
        "is_best_instance",
        "ranking_model_fit",
        "ranking_model_geometry",
    )

    # Token counting
    count_af3_tokens: bool = True

    # Large assembly sampling
    sample_large_assembly_interfaces: bool = True
    large_assembly_threshold: int = 20
    interface_cutoff_distance: float = DistanceThresholds.INTERFACE


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _load_structure(path: PathLike, config: PreprocessConfig) -> dict[str, Any]:
    """Load and parse structure file using PreprocessConfig."""
    return parse(
        filename=path,
        build_assembly=config.build_assembly,
        add_missing_atoms=config.add_missing_atoms,
        remove_waters=config.remove_waters,
        remove_ccds=list(config.remove_ccds),
        fix_ligands_at_symmetry_centers=config.fix_ligands_at_symmetry_centers,
        fix_arginines=config.fix_arginines,
        convert_mse_to_met=config.convert_mse_to_met,
        hydrogen_policy=config.hydrogen_policy,
    )


def _apply_filters(atom_array: AtomArray, ignore_residues: tuple[str, ...], log_id: str | None = None) -> AtomArray:
    """Apply filters: remove non-biological bonds, ignored residues, zero-occupancy atoms."""
    filtered = copy.deepcopy(atom_array)

    # Remove non-polymers with non-biological bonds to polymers
    non_bio_mask = structure_utils.get_non_biological_bond_atom_mask(atom_array)
    if np.any(non_bio_mask):
        # Get non-polymer PN units involved in non-biological bonds
        non_polymer_non_bio = non_bio_mask & ~atom_array.is_polymer
        pn_units_to_remove = np.unique(atom_array.pn_unit_iid[non_polymer_non_bio])
        if len(pn_units_to_remove) > 0:
            # Subset to atoms not in these PN units
            filtered = atom_array[~np.isin(atom_array.pn_unit_iid, pn_units_to_remove)]
            if log_id:
                logger.warning(f"{log_id}: Non-biological bonds detected between non-polymer and polymer PN units.")

    # Remove ignored residues
    if ignore_residues:
        filtered = filtered[not_isin(filtered.res_name, [c.strip() for c in ignore_residues])]

    # Remove zero-occupancy atoms
    return filtered[filtered.occupancy > 0.0]


def _detect_and_resolve_clashes(
    atom_array: AtomArray,
    pn_unit_iids: np.ndarray,
    clash_distance: float,
    log_id: str | None = None,
) -> tuple[AtomArray, ClashSeverity]:
    """Detect clashes and resolve them by removing clashing PN units."""
    clashing_dict = structure_utils.get_clashing_pn_units(pn_unit_iids, atom_array, clash_distance)

    if not clashing_dict:
        return atom_array, ClashSeverity.NO_CLASH

    if log_id:
        logger.warning(f"(Example {log_id}): Clash detected between PN units: {list(clashing_dict.keys())}")

    atom_array, severity = structure_utils.handle_clashing_pn_units(clashing_dict, atom_array)
    return atom_array, severity


def _fetch_ligand_validity_scores(pdb_id: str, ligand_scores: tuple[str, ...]) -> pd.DataFrame | None:
    """Fetch ligand validity scores from RCSB."""
    if not ligand_scores:
        return None

    scores = structure_utils.get_ligand_validity_scores_from_pdb_id(pdb_id)
    if not exists(scores) or len(scores) == 0:
        logger.debug(f"Failed to fetch ligand validity scores for ID {pdb_id}")
        return None

    df = pd.DataFrame(scores).set_index(["asym_id", "res_name"])[list(ligand_scores)]
    df.sort_index(inplace=True)
    return df


# =============================================================================
# Stage 1: Parse Entry
# =============================================================================


def parse_entry(path: Path, config: PreprocessConfig) -> dict[str, Any]:
    """Load structure and fetch ligand validity scores.

    Args:
        path: Path to structure file (.cif, .cif.gz, .pdb, .pdb.gz).
        config: Preprocessing configuration.

    Returns:
        Parser dict with 'ligand_validity_scores' key added.
    """
    result = _load_structure(path, config)
    pdb_id = result["metadata"]["id"]

    scores = _fetch_ligand_validity_scores(pdb_id, config.ligand_scores) if config.ligand_scores else None
    result["ligand_validity_scores"] = scores
    result["_path"] = str(path)

    return result


# =============================================================================
# Stage 2: Process Assemblies
# =============================================================================


def process_assemblies(
    parser_dict: dict[str, Any],
    config: PreprocessConfig,
) -> tuple[list[Assembly], dict[str, AtomArray]]:
    """Filter atoms, resolve clashes, and create Assembly records.

    Args:
        parser_dict: Output from parse_entry().
        config: Preprocessing configuration.

    Returns:
        Tuple of (Assembly list, dict mapping assembly_id to filtered AtomArray).
    """
    assemblies: list[Assembly] = []
    assembly_atoms: dict[str, AtomArray] = {}

    pdb_id = parser_dict["metadata"]["id"]
    path = parser_dict.get("_path", "")
    metadata = parser_dict["metadata"]

    for assembly_id, assembly_tuple in parser_dict["assemblies"].items():
        raw_array = assembly_tuple[0]
        num_raw_atoms = len(raw_array)

        # Calculate % zero-occupancy atoms from the raw assembly
        num_zero_occ = int(np.sum(raw_array.occupancy == 0.0))
        fraction_zero_occ = (num_zero_occ / num_raw_atoms) if num_raw_atoms > 0 else 0.0

        # Token counts (upper bound, before filtering removes unresolved atoms)
        n_atomized, n_non_atomized, n_total = 0, 0, 0
        if config.count_af3_tokens:
            tc = count_af3_style_tokens(raw_array)
            n_atomized, n_non_atomized, n_total = (
                tc.get("n_atomized_tokens", 0),
                tc.get("n_non_atomized_tokens", 0),
                tc.get("n_tokens_total", 0),
            )

        # Filter atoms
        filtered = _apply_filters(raw_array, config.ignore_residues, log_id=path)

        # Check limits
        num_polymer_pn_units = len(np.unique(filtered.pn_unit_iid[filtered.is_polymer]))
        if num_polymer_pn_units > config.polymer_pn_unit_limit:
            logger.warning(f"(Example {pdb_id}): {num_polymer_pn_units} polymer PN units; skipping.")
            continue
        if len(filtered) == 0:
            logger.warning(f"(Example {pdb_id}): No atoms remaining after filtering.")
            continue

        # Resolve clashes
        filtered, clash_severity = _detect_and_resolve_clashes(
            filtered, np.unique(filtered.pn_unit_iid), config.clash_distance, log_id=pdb_id
        )

        # Build Assembly record
        cryst = metadata.get("crystallization_details", {})
        all_pn_unit_iids = tuple(sorted(np.unique(filtered.pn_unit_iid)))

        assembly = Assembly(
            pdb_id=pdb_id,
            assembly_id=assembly_id,
            path=path,
            resolution=metadata.get("resolution"),
            deposition_date=metadata.get("deposition_date"),
            release_date=metadata.get("release_date"),
            method=metadata.get("method"),
            ph=cryst.get("pH") if cryst else None,
            clash_severity=clash_severity.value,
            num_polymer_pn_units=num_polymer_pn_units,
            num_resolved_atoms_in_processed_assembly=len(filtered),
            total_num_atoms_in_unprocessed_assembly=num_raw_atoms,
            fraction_zero_occupancy_atoms=fraction_zero_occ,
            has_unphysical_bonds=pdb_id.lower() in PDB_IDS_WITH_UNPHYSICAL_BONDS if pdb_id else False,
            n_atomized_tokens=n_atomized,
            n_non_atomized_tokens=n_non_atomized,
            n_tokens_total=n_total,
            all_pn_unit_iids_after_processing=all_pn_unit_iids,
        )

        assemblies.append(assembly)
        assembly_atoms[assembly_id] = filtered

    return assemblies, assembly_atoms


# =============================================================================
# Stage 3: Process PN Units
# =============================================================================


def process_pn_units(
    assemblies: list[Assembly],
    assembly_atoms: dict[str, AtomArray],
    parser_dict: dict[str, Any],
    config: PreprocessConfig,
) -> tuple[list[PNUnit], list[ContactInfo]]:
    """Extract PN units and compute contacts.

    Args:
        assemblies: Assembly records from process_assemblies().
        assembly_atoms: Filtered AtomArrays from process_assemblies().
        parser_dict: Original parser dict (for chain_info, ligand_info, ligand_validity_scores).
        config: Preprocessing configuration.

    Returns:
        Tuple of (PNUnit list, ContactInfo list).
    """
    pn_units: list[PNUnit] = []
    contacts: list[ContactInfo] = []

    chain_info = parser_dict.get("chain_info", {})
    loi_set = set(parser_dict.get("ligand_info", {}).get("ligand_of_interest", []))
    ligand_scores = parser_dict.get("ligand_validity_scores")

    for assembly in assemblies:
        aid = assembly.assembly_id
        if aid not in assembly_atoms:
            continue

        atoms = copy.deepcopy(assembly_atoms[aid])
        pn_iids = [iid for iid in np.unique(atoms.pn_unit_iid) if len(atoms[atoms.pn_unit_iid == iid]) > 0]

        if not pn_iids:
            continue

        # Build KDTree once for all contact queries (major speedup for large assemblies)
        kdtree = cKDTree(atoms.coord)
        seen_pairs: set[tuple[str, str]] = set()

        for pn_iid in pn_iids:
            query_mask = atoms.pn_unit_iid == pn_iid
            query = atoms[query_mask]
            if len(query) == 0:
                continue

            target_mask = ~query_mask
            query_type = ChainType(query.chain_type[0])
            pn_unit_id = query.pn_unit_id[0]

            # Get contacts (reuse kdtree for efficiency)
            contacting = structure_utils.get_contacting_pn_units(
                atoms,
                query_mask,
                target_mask,
                config.contact_distance,
                min_contacts_required=1,
                calculate_min_distance=True,
                tree=kdtree,
            )

            contacting = sorted(contacting, key=lambda x: (x["num_contacts"], -(x["min_distance"] or 0)), reverse=True)
            # Detailed contact info with num_atoms, num_contacts, min_distance
            contacting_iids = tuple(
                {
                    "pn_unit_iid": c["pn_unit_iid"],
                    "num_atoms": c.get("num_atoms", 0),
                    "num_contacts": c.get("num_contacts", 0),
                    "min_distance": c.get("min_distance"),
                }
                for c in contacting
            )

            # Extract type-specific data (inline)
            bonded: set = set()
            is_metal, is_loi, ranking_fit = False, False, None
            canonical_seq, non_canonical_seq = "", ""
            seq_len, has_non_canonical_residue, ec_nums = None, None, None
            non_polymer_names = ""

            if query_type.is_non_polymer():
                bonded = structure_utils.get_bonded_polymer_pn_units(pn_iid, atoms)
                res_names = np.unique(query.res_name)
                is_loi = bool(loi_set & set(res_names))
                is_metal = len(query) == 1 and query[0].element.upper() in METAL_ELEMENTS
                non_polymer_names = ",".join(struc.get_residues(query)[1])

                if ligand_scores is not None:
                    ligand_ids = sorted(set(zip(query.chain_id, query.res_name, strict=False)))
                    matching = [lid for lid in ligand_ids if lid in ligand_scores.index]
                    if matching:
                        ranking_fit = ligand_scores.loc[matching].to_dict().get("ranking_model_fit")
            else:  # polymer
                cinfo = chain_info.get(query.chain_id[0], {})
                canonical_seq = cinfo.get("processed_entity_canonical_sequence", "")
                non_canonical_seq = cinfo.get("processed_entity_non_canonical_sequence", "")
                seq_len = len(canonical_seq) if canonical_seq else 0
                ec_nums = cinfo.get("ec_numbers")
                has_non_canonical_residue = bool(
                    set(np.unique(query.res_name)) - set(STANDARD_AND_UNKNOWN_POLYMER_RESIDUES)
                )

            # Primary polymer partner
            if not query_type.is_non_polymer():
                primary_partner = pn_iid
            else:
                polymer_iids = np.unique(atoms.pn_unit_iid[atoms.is_polymer])
                partners = [c for c in contacting if c["pn_unit_iid"] in polymer_iids]
                primary_partner = partners[0]["pn_unit_iid"] if partners else None

            # Sampling weights
            n_prot = 1 if query_type.is_protein() and (seq_len or 0) > PEPTIDE_MAX_RESIDUES else 0
            n_peptide = 1 if query_type.is_protein() and (seq_len or 0) <= PEPTIDE_MAX_RESIDUES else 0
            n_nuc = 1 if query_type.is_nucleic_acid() else 0
            n_ligand = 1 if query_type.is_non_polymer() else 0

            bonded_iids = tuple(bonded)

            mol_id = query.molecule_id[0]
            mol_iid = query.molecule_iid[0]
            trans_id = query.transformation_id[0]

            pn_units.append(
                PNUnit(
                    pdb_id=assembly.pdb_id,
                    assembly_id=aid,
                    pn_unit_id=pn_unit_id,
                    pn_unit_iid=pn_iid,
                    molecule_id=mol_id,
                    molecule_iid=mol_iid,
                    transformation_id=trans_id,
                    pn_unit_type=query_type.value,
                    is_polymer=query_type.is_polymer(),
                    is_metal=is_metal,
                    is_loi=is_loi,
                    num_resolved_atoms=len(query),
                    num_resolved_residues=struc.get_residue_count(query),
                    is_multichain=len(np.unique(query.chain_id)) > 1,
                    is_multiresidue=len(np.unique(query.res_id)) > 1,
                    sequence_length=seq_len,
                    has_non_canonical_residue=has_non_canonical_residue,
                    processed_entity_canonical_sequence=canonical_seq or None,
                    processed_entity_non_canonical_sequence=non_canonical_seq or None,
                    processed_entity_canonical_sequence_hash=hash_sequence(canonical_seq) if canonical_seq else None,
                    processed_entity_non_canonical_sequence_hash=hash_sequence(non_canonical_seq)
                    if non_canonical_seq
                    else None,
                    ec_numbers=tuple(ec_nums) if ec_nums else None,
                    non_polymer_res_names=non_polymer_names or None,
                    bonded_polymer_pn_units=bonded_iids or None,
                    ranking_model_fit=ranking_fit,
                    primary_polymer_partner_iid=primary_partner,
                    contacting_pn_unit_iids=contacting_iids,
                    n_prot=n_prot,
                    n_peptide=n_peptide,
                    n_nuc=n_nuc,
                    n_ligand=n_ligand,
                )
            )

            # Create ContactInfo for new pairs
            for c in contacting:
                partner_iid = c["pn_unit_iid"]
                pair = tuple(sorted([pn_iid, partner_iid]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                contacts.append(
                    ContactInfo(
                        pn_unit_1_iid=pair[0],
                        pn_unit_2_iid=pair[1],
                        num_contacts=c.get("num_contacts", 0),
                        min_distance=c.get("min_distance", 0.0) or 0.0,
                        involves_covalent_modification=partner_iid in bonded,
                    )
                )

    return pn_units, contacts


# =============================================================================
# Stage 4: Process Interfaces
# =============================================================================


def process_interfaces(pn_units: list[PNUnit], contact_info: list[ContactInfo]) -> list[Interface]:
    """Create Interface records from contacts.

    Args:
        pn_units: PNUnit records from process_pn_units().
        contact_info: ContactInfo records from process_pn_units().

    Returns:
        List of Interface records.
    """
    lookup = {p.pn_unit_iid: p for p in pn_units}
    interfaces = []

    for c in contact_info:
        p1, p2 = lookup.get(c.pn_unit_1_iid), lookup.get(c.pn_unit_2_iid)
        if not (p1 and p2) or p1.assembly_id != p2.assembly_id:
            continue

        # Check covalent modification in both directions
        bonded_1 = p1.bonded_polymer_pn_units or ()
        bonded_2 = p2.bonded_polymer_pn_units or ()
        involves_covalent = c.pn_unit_2_iid in bonded_1 or c.pn_unit_1_iid in bonded_2

        interfaces.append(
            Interface(
                pdb_id=p1.pdb_id,
                assembly_id=p1.assembly_id,
                pn_unit_1_iid=c.pn_unit_1_iid,
                pn_unit_2_iid=c.pn_unit_2_iid,
                num_contacts=c.num_contacts,
                min_distance=c.min_distance,
                involves_covalent_modification=involves_covalent,
                is_inter_molecule=p1.molecule_id != p2.molecule_id,
                involves_loi=p1.is_loi or p2.is_loi,
                involves_metal=p1.is_metal or p2.is_metal,
            )
        )

    return interfaces


# =============================================================================
# Stage 5: Add Closest Chain Info (Final Step)
# =============================================================================


def _add_closest_chain_info(
    assemblies: list[Assembly],
    assembly_atoms: dict[str, AtomArray],
    pn_units: list[PNUnit],
    interfaces: list[Interface],
    config: PreprocessConfig,
) -> tuple[list[Assembly], list[PNUnit], list[Interface]]:
    """Add closest_20/50_pn_unit_iids to assemblies.

    Uses existing Interface objects to identify polymer-polymer contacts.
    For small assemblies (≤threshold), all chains are considered "closest".
    For large assemblies, samples an interface and finds closest chains.
    """
    # Build lookup: pn_unit_iid -> is_polymer
    is_polymer = {p.pn_unit_iid: p.is_polymer for p in pn_units}

    # Group polymer-polymer interfaces by assembly
    assembly_interfaces: dict[str, list[Interface]] = defaultdict(list)
    for iface in interfaces:
        if is_polymer.get(iface.pn_unit_1_iid) and is_polymer.get(iface.pn_unit_2_iid):
            assembly_interfaces[iface.assembly_id].append(iface)

    updated = []
    for assembly in assemblies:
        aid = assembly.assembly_id
        atoms = assembly_atoms.get(aid)
        all_iids = set(assembly.all_pn_unit_iids_after_processing)

        # Default: all chains are closest (for small assemblies)
        closest_20: set[str] = all_iids
        closest_50: set[str] = all_iids

        # Sample for large assemblies
        if (
            config.sample_large_assembly_interfaces
            and assembly.num_polymer_pn_units > config.large_assembly_threshold
            and atoms is not None
        ):
            polymer_ifaces = assembly_interfaces.get(aid, [])

            if polymer_ifaces:
                # We need the atomize annotation to label tokens when sampling chains near interfaces
                atoms_with_id = add_global_atom_id_annotation(atoms)
                atomized_atoms = atomize_by_ccd_name(
                    atoms_with_id,
                    atomize_by_default=True,
                    res_names_to_ignore=[*STANDARD_POLYMER_RESIDUES],
                )

                # Get polymer-polymer contact pairs
                pairs = [(i.pn_unit_1_iid, i.pn_unit_2_iid) for i in polymer_ifaces]
                result = sample_large_assembly_interface(atomized_atoms, pairs, config.interface_cutoff_distance)

                if result:
                    closest_20 = result.get(20, set())
                    closest_50 = result.get(50, set())

        updated.append(
            replace(
                assembly,
                closest_20_pn_unit_iids=tuple(sorted(closest_20 & all_iids)),
                closest_50_pn_unit_iids=tuple(sorted(closest_50 & all_iids)),
            )
        )

    # Build assembly_id -> (closest_20, closest_50) lookup
    assembly_closest = {
        a.assembly_id: (set(a.closest_20_pn_unit_iids), set(a.closest_50_pn_unit_iids)) for a in updated
    }

    # Update pn_units with closest chain flags
    updated_pn_units = []
    for p in pn_units:
        c20, c50 = assembly_closest.get(p.assembly_id, (set(), set()))
        updated_pn_units.append(
            replace(
                p,
                within_20_closest_chains=(p.pn_unit_iid in c20) if c20 else True,
                within_50_closest_chains=(p.pn_unit_iid in c50) if c50 else True,
            )
        )

    # Update interfaces with closest chain flags
    updated_interfaces = []
    for i in interfaces:
        c20, c50 = assembly_closest.get(i.assembly_id, (set(), set()))
        updated_interfaces.append(
            replace(
                i,
                within_20_closest_chains=(i.pn_unit_1_iid in c20 and i.pn_unit_2_iid in c20) if c20 else True,
                within_50_closest_chains=(i.pn_unit_1_iid in c50 and i.pn_unit_2_iid in c50) if c50 else True,
            )
        )

    return updated, updated_pn_units, updated_interfaces


# =============================================================================
# Main Entry Point
# =============================================================================


def preprocess(
    path: Path,
    config: PreprocessConfig | None = None,
) -> tuple[list[Assembly], list[PNUnit], list[Interface]]:
    """Preprocess a structure file into normalized records.

    This is the main entry point that runs all 5 stages:
    1. parse_entry() - Load and prepare structure
    2. process_assemblies() - Filter and create Assembly records
    3. process_pn_units() - Extract PN units and contacts
    4. process_interfaces() - Create Interface records
    5. _add_closest_chain_info() - Add closest chain info for large assemblies

    Args:
        path: Path to structure file (.cif, .cif.gz, .pdb, .pdb.gz).
        config: Preprocessing configuration. Uses defaults if None.

    Returns:
        Tuple of (assemblies, pn_units, interfaces).
    """
    config = config or PreprocessConfig()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    parser_dict = parse_entry(path, config)
    assemblies, assembly_atoms = process_assemblies(parser_dict, config)
    pn_units, contact_info = process_pn_units(assemblies, assembly_atoms, parser_dict, config)
    interfaces = process_interfaces(pn_units, contact_info)

    # Update assemblies, pn_units, interfaces with closest chain info
    assemblies, pn_units, interfaces = _add_closest_chain_info(assemblies, assembly_atoms, pn_units, interfaces, config)

    return assemblies, pn_units, interfaces


# =============================================================================
# Save to Parquet
# =============================================================================


def save_records(
    out_dir: Path,
    pdb_id: str,
    assemblies: list[Assembly],
    pn_units: list[PNUnit],
    interfaces: list[Interface],
    exclude: set[str] | None = None,
) -> None:
    """Save preprocessing result to parquet files.

    Creates:
        out_dir/assemblies/{pdb_id}.parquet
        out_dir/pn_units/{pdb_id}.parquet
        out_dir/interfaces/{pdb_id}.parquet

    Args:
        out_dir: Base output directory.
        pdb_id: PDB identifier for file naming.
        assemblies: List of Assembly records.
        pn_units: List of PNUnit records.
        interfaces: List of Interface records.
        exclude: Optional set of field names to exclude from output.
    """
    exclude = exclude or set()

    for name, records in [("assemblies", assemblies), ("pn_units", pn_units), ("interfaces", interfaces)]:
        if not records:
            continue
        subdir = out_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([serialize_record(r.to_dict(exclude)) for r in records])
        df.to_parquet(subdir / f"{pdb_id}.parquet", index=False)
