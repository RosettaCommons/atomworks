"""Internal processing pipeline for structure parsing with AtomWorks.

Not intended for direct use by external callers; use :py:func:`~atomworks.io.parser.parse`,
:py:func:`~atomworks.io.parser.parse_atom_array`, or :py:func:`~atomworks.io.parser.prepare_atom_array` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.info import standardize_order
from biotite.structure.io import pdbx

import atomworks.io.transforms.atom_array as ta
from atomworks.common import exists
from atomworks.constants import METAL_ELEMENTS
from atomworks.io import template
from atomworks.io.config import PrepareConfig
from atomworks.io.transforms.categories import (
    category_to_dict,
    extract_crystallization_details,
    get_ligand_of_interest_info,
)
from atomworks.io.utils.annotator import ensure_annotations, remove_annotations
from atomworks.io.utils.assembly import (
    build_assemblies_from_asym_unit,
    get_identity_assembly_gen_category,
    get_identity_op_expr_category,
)
from atomworks.io.utils.atom_array import annotate_and_remove_hydrogens, remove_components, remove_hydrogens
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, as_atom_array_plus_stack, stack_any
from atomworks.io.utils.bonds import add_bonds_from_struct_conn, filter_bonds_by_distance
from atomworks.io.utils.ccd import add_annotations_from_ccd
from atomworks.io.utils.chain_info import build_chain_info, update_sequences_from_res_names
from atomworks.io.utils.extra_fields import ExtraFieldsType, normalize_extra_fields
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standardize import standardize_atom_names
from atomworks.io.utils.testing import verify_atom_array_chain_info_consistency

logger = logging.getLogger(__name__)


def _build_assemblies(
    atoms: AtomArrayStack,
    cif_block: pdbx.CIFBlock | None,
    chain_info: dict,
    *,
    build_assembly: str | list[str] | None,
    fix_ligands_at_symmetry_centers: bool,
) -> tuple[AtomArrayStack, dict, dict]:
    """Build biological assemblies from the asymmetric unit and extract struct_oper categories."""
    extra_info: dict[str, Any] = {
        "assembly_gen_category": None,
        "struct_oper_category": None,
    }

    # Pre-built assembly: generate annotations, add to assemblies dictionary, and store identity operation info
    if "transformation_id" in atoms.get_annotation_categories():
        if exists(build_assembly):
            assert not (
                cif_block is not None
                and "pdbx_struct_oper_list" in cif_block
                and "pdbx_struct_assembly_gen" in cif_block
            ), (
                "Both transformation_id annotations and struct_oper categories detected. "
                "These are mutually exclusive ways to define assemblies. Please remove one or the other."
            )
        atoms = ta.add_iid_annotations(atoms, overwrite=True)
        assemblies = {"1": atoms}
        return atoms, assemblies, extra_info

    # Validate build_assembly if provided (not None)
    if exists(build_assembly):
        assert build_assembly in ["first", "all"] or isinstance(
            build_assembly, list | tuple
        ), "Invalid `build_assembly` option. Must be 'first', 'all', or a list/tuple of assembly IDs as strings."

    # Always read struct_oper categories from file (even if not building assemblies)
    # These may be needed later for saving struct_oper info to CIF files
    if cif_block is not None and "pdbx_struct_oper_list" in cif_block and "pdbx_struct_assembly_gen" in cif_block:
        assembly_gen_category = cif_block["pdbx_struct_assembly_gen"]
        struct_oper_category = cif_block["pdbx_struct_oper_list"]
    else:
        assembly_gen_category = get_identity_assembly_gen_category(list(chain_info.keys()))
        struct_oper_category = get_identity_op_expr_category()

    # Store assembly info (always, even if not building)
    extra_info["assembly_gen_category"] = assembly_gen_category
    extra_info["struct_oper_category"] = struct_oper_category

    # Build assemblies if requested (build_assembly is not None)
    if build_assembly is None:
        # None means: don't build assemblies, return only asymmetric unit
        assemblies = {}
    else:
        assemblies = build_assemblies_from_asym_unit(
            assembly_gen_category=assembly_gen_category,
            struct_oper_category=struct_oper_category,
            asym_unit_atom_array_stack=atoms,
            build_assembly=build_assembly,
            fix_symmetry_centers=fix_ligands_at_symmetry_centers,
        )
    return atoms, assemblies, extra_info


def _standardize_and_complete(
    model: AtomArray,
    chain_info: dict,
    *,
    extra_fields: ExtraFieldsType | None,
    add_missing_atoms: bool,
    ccd_mirror_path: str | None,
    hydrogen_policy: str,
) -> AtomArray:
    """Standardize atom names and add missing atoms (if configured)."""
    if not add_missing_atoms:
        return model

    # Normalize and validate extra_fields - require defaults when adding missing atoms
    extra_field_specs = None
    if extra_fields is not None and extra_fields != "all":
        extra_field_specs = normalize_extra_fields(extra_fields, require_defaults=True)
    # Preserve selected conformers through rebuilding; links must not cross reaction states (6DP5).
    if "label_alt_id" in model.get_annotation_categories():
        extra_field_specs = dict(extra_field_specs or {})
        extra_field_specs["label_alt_id"] = {"default": ".", "dtype": model.label_alt_id.dtype}

    # Step 1: Standardize atom names (convert alt->std, filter non-matching)
    # Non-standard atoms will be filtered out (and re-added by add_missing_atoms)
    model = standardize_atom_names(
        model,
        on_mismatch="raise",  # Raise if we have a heavy atom mismatch
        on_mismatch_non_heavy="filter",  # Remove non-standard non-heavy atoms (e.g., deuterium, tritium, etc.)
        ccd_mirror_path=ccd_mirror_path,
        non_canonical_only=False,  # Standardize ALL residues to fix alternate atom names (H5''1->H5'', HB4->HB3, etc.)
    )

    # If removing hydrogens while still adding missing atoms, do so upfront (otherwise we may double-count)
    if hydrogen_policy == "remove":
        model = remove_hydrogens(model)

    # Step 2: Add missing atoms (includes standardize_order and atomic_number)
    model = template.add_missing_atoms(
        model,
        chain_info_dict=chain_info,
        ccd_mirror_path=ccd_mirror_path,
        hydrogen_policy=hydrogen_policy,
        extra_field_specs=extra_field_specs,
    )

    # Standardize order so that subsequent operations can assume a consistent ordering
    model = model[standardize_order(model)]

    # Step 3: Add CCD annotations
    # NOTE: We override the nhyd annotation when we build from CCD templates (but preserve if set for non-CCD residues)
    assert "charge" in model.get_annotation_categories(), (
        "Charge annotation missing on model before add_annotations_from_ccd(). "
        "Structures loaded via parse()/get_structure() always set charge — this assert guards against silent "
        "zero-fills that would mask upstream charge-loss bugs."
    )
    model = add_annotations_from_ccd(
        model,
        ccd_mirror_path=ccd_mirror_path,
        hydrogen_policy=hydrogen_policy,
        overwrite={"nhyd": True},
    )
    return model


def _add_bonds(
    model: AtomArray,
    cif_block: pdbx.CIFBlock | None,
    *,
    add_bond_types_from_struct_conn: tuple[str, ...],
    struct_conn_distance_policy: str,
    sanitize: bool,
    ccd_mirror_path: str | None,
) -> AtomArray:
    """Parse custom bonds, struct_conn bonds, and infer residue bonds."""
    # Parse CIF custom bonds (if CIF file available)
    cif_custom_bonds = None
    if cif_block is not None and "chem_comp_bond" in cif_block:
        cif_custom_bonds = pdbx.convert._parse_intra_residue_bonds(cif_block["chem_comp_bond"])

    # Add struct_conn bonds (disulfides, metal coordination, etc.)
    if cif_block is not None:
        model = add_bonds_from_struct_conn(
            model,
            cif_block,
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            struct_conn_distance_policy=struct_conn_distance_policy,
            allow_missing_templates=not sanitize,
        )

    # Chemical preparation requires concrete orders; minimal parsing preserves unknown orders.
    if sanitize and model.bonds is not None:
        model.bonds.convert_bond_type(struc.BondType.ANY, struc.BondType.SINGLE)

    # Add intra- and inter-residue bonds (based on CCD identity and polymer sequence)
    model = template.infer_bonds_from_residue_names(
        model,
        custom_bond_dict=cif_custom_bonds,  # Use CIF bonds if available!
        sanitize=sanitize,
        ccd_mirror_path=ccd_mirror_path,
    )

    # Retype metal-incident bonds to COORDINATION so metals are typed consistently and (via the
    # COORDINATION exclusion in add_id_and_entity_annotations) become their own molecule/entity.
    model = _retype_metal_bonds_to_coordination(model)
    return model


def _retype_metal_bonds_to_coordination(model: AtomArray) -> AtomArray:
    """Set every bond incident to a metal atom to :py:attr:`BondType.COORDINATION`."""
    if model.bonds is None:
        return model

    metal_mask = np.isin(np.char.upper(model.element.astype(str)), list(METAL_ELEMENTS))
    if not metal_mask.any():
        return model

    # A bond qualifies if either of its two endpoint atoms is a metal.
    bonds = model.bonds.as_array()  # (n_bonds, 3): atom_i, atom_j, bond_type
    touches_metal = metal_mask[bonds[:, 0]] | metal_mask[bonds[:, 1]]

    if touches_metal.any():
        bonds[touches_metal, 2] = int(struc.BondType.COORDINATION)
        model.bonds = struc.BondList(model.array_length(), bonds)

    return model


def _get_crystallization_details(cif_block: pdbx.CIFBlock | None) -> dict:
    """Extract crystallization details from CIF."""
    if cif_block is not None and "exptl_crystal_grow" in cif_block:
        return extract_crystallization_details(category_to_dict(cif_block, "exptl_crystal_grow"))
    return {"pH": None}


def _get_ligand_info(cif_block: pdbx.CIFBlock | None) -> dict:
    """Extract ligand info from CIF, or return defaults if cif_block is None."""
    if cif_block is not None:
        return get_ligand_of_interest_info(cif_block)
    return {"has_ligand_of_interest": False, "ligand_of_interest": []}


def _get_msa_paths_from_cif(cif_block: pdbx.CIFBlock | None) -> dict[str, Path]:
    """Extract per-chain MSA paths from CIF block."""
    if cif_block is None or "msa_paths_by_chain_id" not in cif_block:
        return {}
    logger.info("MSA paths detected in CIF file. Adding to chain information...")
    msa_paths = category_to_dict(cif_block, "msa_paths_by_chain_id")
    return {chain_id: Path(msa_path.item()) for chain_id, msa_path in msa_paths.items()}


_TMP_ANNOTATIONS = ("leaving_atom_flag", "is_leaving_atom", "is_n_terminal_atom", "is_c_terminal_atom", "index")


def _maybe_promote_to_plus(atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Promote to the ``Plus`` variant (or return unchanged if already Plus)."""
    if isinstance(atoms, AtomArrayStack):
        return as_atom_array_plus_stack(atoms)
    if isinstance(atoms, AtomArray):
        return as_atom_array_plus(atoms)
    return atoms


def _assemble_parse_result(
    *,
    atoms: AtomArrayStack,
    chain_info: dict,
    cif_block: pdbx.CIFBlock | None,
    config: PrepareConfig,
    build_assembly: str | list[str] | None,
    metadata: dict,
    keep_cif_block: bool = False,
) -> dict[str, Any]:
    """Build assemblies, gather metadata, clean annotations, and return the result dict.

    This is the post-:py:func:`_prepare_atom_array_or_stack` step shared by
    :py:func:`~atomworks.io.parser.parse` and
    :py:func:`~atomworks.io.parser.parse_atom_array`.
    """
    atoms, assemblies, extra_info = _build_assemblies(
        atoms,
        cif_block,
        chain_info,
        build_assembly=build_assembly,
        fix_ligands_at_symmetry_centers=config.fix_ligands_at_symmetry_centers,
    )

    metadata["crystallization_details"] = _get_crystallization_details(cif_block)
    ligand_info = _get_ligand_info(cif_block)
    for chain_id, msa_path in _get_msa_paths_from_cif(cif_block).items():
        chain_info[chain_id]["msa_path"] = msa_path

    if config.add_missing_atoms:
        verify_atom_array_chain_info_consistency(
            chain_info=chain_info,
            atom_array=atoms[0] if isinstance(atoms, AtomArrayStack) else atoms,
            verify_sequences=True,
        )

    # Clean temporary annotations
    remove_annotations(atoms, *_TMP_ANNOTATIONS)
    for assembly in assemblies.values():
        remove_annotations(assembly, *_TMP_ANNOTATIONS)

    if config.return_atom_array_plus:
        atoms = _maybe_promote_to_plus(atoms)
        assemblies = {k: _maybe_promote_to_plus(v) for k, v in assemblies.items()}

    result: dict[str, Any] = {
        "chain_info": chain_info,
        "ligand_info": ligand_info,
        "asym_unit": atoms,
        "assemblies": assemblies,
        "metadata": metadata,
        "extra_info": extra_info,
    }
    if keep_cif_block:
        result["cif_block"] = cif_block
    return result


def _check_resid_ordering(atoms: AtomArray | AtomArrayStack) -> None:
    """Raise ``ValueError`` if ``res_id`` is not non-decreasing within any chain."""
    arr = atoms[0] if isinstance(atoms, AtomArrayStack) else atoms

    # For assemblies with transformation_id, only check the first transformation
    # since res_ids naturally repeat across transformation copies of the same chain.
    if "transformation_id" in arr.get_annotation_categories():
        first_tid = arr.transformation_id[0]
        arr = arr[arr.transformation_id == first_tid]

    for chain_id in np.unique(arr.chain_id):
        res_ids = arr.res_id[arr.chain_id == chain_id]
        if np.any(np.diff(res_ids) < 0):
            raise ValueError(f"res_id values are not in non-decreasing order for chain '{chain_id}'")


def _prepare_atom_array_or_stack(
    atoms: AtomArray | AtomArrayStack,
    *,
    cif_block: pdbx.CIFBlock | None,
    config: PrepareConfig,
    extra_fields: ExtraFieldsType | None = None,
) -> tuple[AtomArrayStack, dict]:
    """Prepare an AtomArray or AtomArrayStack with standardized annotations, bonds, and chain info.

    This is the shared core of :py:func:`~atomworks.io.parser.parse`,
    :py:func:`~atomworks.io.parser.parse_atom_array`, and
    :py:func:`~atomworks.io.parser.prepare_atom_array`.
    """
    has_cif = cif_block is not None

    if get_annotation_categories(atoms, n_body=2) and has_cif:
        raise ValueError(
            "Providing a CIF file is not supported when parsing an AtomArrayPlus or AtomArrayPlusStack "
            "that contains 2-body annotations. Consider using parse() instead, which accepts CIF files directly."
        )

    if exists(extra_fields) and not has_cif:
        logger.warning("The `extra_fields` argument will be ignored if there is no CIF file input.")

    # +------ Initialization ------+

    if atoms.bonds is None:
        atoms.bonds = struc.BondList(atoms.array_length())

    if "label_entity_id" not in atoms.get_annotation_categories():
        # NOTE: int16 (not int8)
        if "chain_entity" in atoms.get_annotation_categories():
            atoms.set_annotation("label_entity_id", atoms.chain_entity.astype(np.int16))
        else:
            atoms.set_annotation(
                "label_entity_id",
                pdbx.convert._determine_entity_id(atoms.chain_id).astype(np.int16),
            )

    if "occupancy" not in atoms.get_annotation_categories():
        atoms.set_annotation("occupancy", np.ones(atoms.array_length()))

    atoms = ta.ensure_atom_array_stack(atoms)

    if "transformation_id" in atoms.get_annotation_categories():
        atoms.transformation_id = np.array(atoms.transformation_id, dtype=str)

    atoms = remove_components(atoms, remove_waters=config.remove_waters)

    _check_resid_ordering(atoms)

    # +------ Chain info and structural annotations ------+

    cif_cats = {}
    if cif_block is not None:
        for name in ("entity", "entity_poly", "entity_poly_seq"):
            if name in cif_block:
                cif_cats[name] = category_to_dict(cif_block, name)

    chain_info = build_chain_info(
        atoms,
        entity=cif_cats.get("entity"),
        entity_poly=cif_cats.get("entity_poly"),
        entity_poly_seq=cif_cats.get("entity_poly_seq"),
    )

    if "auth_seq_id" in atoms.get_annotation_categories():
        # Replace non-polymeric chain sequence ids with author sequence ids (since the non-polymer sequence ID's are not informative)
        # TODO: Regenerate regression tests and remove this function altogether (we don't need it anymore given Biotite upgrade)
        atoms = ta.update_nonpoly_seq_ids(atoms, chain_info)

    atoms = ta.add_polymer_annotation(atoms, chain_info)
    atoms = ta.add_chain_type_annotation(atoms, chain_info)

    if config.convert_mse_to_met:
        atoms = ta.mse_to_met(atoms, chain_info=chain_info)

    # +------ Per-model processing ------+

    models = []
    for model_idx in range(atoms.stack_depth()):
        model = atoms[model_idx]
        model = _standardize_and_complete(
            model,
            chain_info,
            extra_fields=extra_fields,
            add_missing_atoms=config.add_missing_atoms,
            ccd_mirror_path=config.ccd_mirror_path,
            hydrogen_policy=config.hydrogen_policy,
        )

        # NOTE: Relies on the custom CCD registry for inferring chemical component types
        model = _add_bonds(
            model,
            cif_block,
            add_bond_types_from_struct_conn=config.add_bond_types_from_struct_conn,
            struct_conn_distance_policy=config.struct_conn_distance_policy,
            sanitize=config.add_missing_atoms,
            ccd_mirror_path=config.ccd_mirror_path,
        )

        # Coordinate-dependent: must stay per-model
        if config.fix_arginines:
            model = ta.resolve_arginine_naming_ambiguity(model, raise_on_error=False)

        if config.long_bond_policy != "keep":
            model = filter_bonds_by_distance(model, policy=config.long_bond_policy)

        models.append(model)

    atoms = stack_any(models)

    # Filter candidates only after covalent bonds and distance policies are resolved.
    if config.remove_ccds:
        n_atoms = atoms.array_length()
        atoms = remove_components(atoms, remove_waters=False, remove_ccds=config.remove_ccds)
        if atoms.array_length() != n_atoms:
            remaining_chains = set(atoms.chain_id)
            chain_info = {chain_id: info for chain_id, info in chain_info.items() if chain_id in remaining_chains}
            for chain_id, info in chain_info.items():
                # Preserve unresolved sequence entries unless they are explicitly excluded.
                keep = ~np.isin(info["res_name"], [ccd.upper() for ccd in config.remove_ccds]) | np.isin(
                    info["res_id"], atoms.res_id[atoms.chain_id == chain_id]
                )
                if not keep.all():
                    for key in ("res_id", "res_name"):
                        info[key] = np.asarray(info[key])[keep].tolist()
                    update_sequences_from_res_names(info)

    # +------ Stack-wide annotations ------+

    if config.hydrogen_policy == "remove" and not config.add_missing_atoms:
        atoms = annotate_and_remove_hydrogens(atoms, increment=True)

    if config.add_id_and_entity_annotations:
        atoms = ta.add_id_and_entity_annotations(atoms)

    ensure_annotations(atoms, "atomic_number", "chem_comp_type")

    # Clean temp annotations so callers get a tidy result
    remove_annotations(atoms, *_TMP_ANNOTATIONS)

    return atoms, chain_info
