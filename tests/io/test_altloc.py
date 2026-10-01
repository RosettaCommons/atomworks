"""Tests for altloc edge cases on difficult PDB structures."""

import dataclasses
import io

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack
from scipy.spatial import cKDTree

from atomworks.constants import ALTLOC_DEFAULT_IDS
from atomworks.io.parser import ParseConfig, parse
from atomworks.io.utils.altloc import (
    _build_constraints,
    _identify_altloc_groups,
    select_altlocs_clash_aware,
)
from atomworks.io.utils.io_utils import read_any
from atomworks.io.utils.selection import get_annotation
from atomworks.io.utils.testing import get_pdb_path_or_buffer

_ALTLOC_EDGE_CASE_IDS = [
    "1cbn",  # 14 altloc residues with intra-chain clashes between combinations
    "1xvk",  # struct_conn altloc-dependent disulfide bond (NYC-N2C); cyclic peptide with many close contacts
    "1rcq",  # PLP in two partially occupied active-site states; struct_conn in one conformation only
    "145d",  # DNA structure with chain-level altlocs
    "1aym",  # 11 altloc residues, 29 incompatibilities — heavy constraint stress test
]

_CLASH_AWARE_CONFIG = ParseConfig(
    altloc="random_clash_aware",
    build_assembly=None,
    add_missing_atoms=False,
)


def assert_no_inter_residue_altloc_clashes(
    arr: AtomArray | AtomArrayStack,
    clash_distance: float = 1.5,
    label: str = "",
) -> None:
    """Assert no unbonded inter-residue steric clashes among altloc atoms."""
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    if not hasattr(arr, "altloc_id"):
        return

    # Only check atoms with non-default altloc IDs
    altloc_mask = ~np.isin(arr.altloc_id, ALTLOC_DEFAULT_IDS)
    altloc_idx = np.where(altloc_mask & ~np.any(np.isnan(arr.coord), axis=1))[0]
    if len(altloc_idx) < 2:
        return

    tree = cKDTree(arr.coord[altloc_idx])
    close_pairs = tree.query_pairs(r=clash_distance, output_type="ndarray")
    if len(close_pairs) == 0:
        return

    # Map back to original indices
    pairs_orig = altloc_idx[close_pairs]

    # Remove intra-residue pairs
    same_res = (arr.chain_id[pairs_orig[:, 0]] == arr.chain_id[pairs_orig[:, 1]]) & (
        arr.res_id[pairs_orig[:, 0]] == arr.res_id[pairs_orig[:, 1]]
    )
    pairs_orig = pairs_orig[~same_res]
    if len(pairs_orig) == 0:
        return

    # Remove bonded pairs
    if arr.bonds is not None:
        bond_arr = arr.bonds.as_array()[:, :2]
        bonded = set(map(tuple, bond_arr)) | set(map(tuple, bond_arr[:, ::-1]))
        not_bonded = np.array([tuple(p) not in bonded for p in pairs_orig])
        pairs_orig = pairs_orig[not_bonded]

    assert len(pairs_orig) == 0, (
        f"Found {len(pairs_orig)} inter-residue altloc clashes{f' ({label})' if label else ''}:\n"
        + "\n".join(
            f"  {arr.chain_id[i]}:{arr.res_name[i]}{arr.res_id[i]}:{arr.atom_name[i]} <-> "
            f"{arr.chain_id[j]}:{arr.res_name[j]}{arr.res_id[j]}:{arr.atom_name[j]} "
            f"dist={np.linalg.norm(arr.coord[i] - arr.coord[j]):.2f}A"
            for i, j in pairs_orig[:5]
        )
    )


@pytest.mark.parametrize("pdb_id", _ALTLOC_EDGE_CASE_IDS)
@pytest.mark.parametrize("add_missing_atoms", [True, False])
def test_no_clashes_and_coordinate_diversity(pdb_id: str, add_missing_atoms: bool):
    """Clash-aware altloc selection produces no clashes and diverse coordinates across seeds."""
    path = get_pdb_path_or_buffer(pdb_id)
    coord_hashes = set()
    for seed in range(3):
        config = dataclasses.replace(
            _CLASH_AWARE_CONFIG,
            altloc_seed=seed,
            add_missing_atoms=add_missing_atoms,
        )
        asym = parse(path, config=config)["asym_unit"]
        assert_no_inter_residue_altloc_clashes(
            asym,
            label=f"{pdb_id} seed={seed} add_missing={add_missing_atoms}",
        )
        coord_hashes.add(asym.coord.tobytes())

    assert (
        len(coord_hashes) > 1
    ), f"{pdb_id} add_missing={add_missing_atoms}: all 3 seeds produced identical coordinates"


def test_random_per_chain():
    """random_per_chain: one altloc letter per chain, different seeds produce diversity."""
    path = get_pdb_path_or_buffer("1cbn")
    coord_hashes = set()
    for seed in range(5):
        config = ParseConfig(altloc="random_per_chain", altloc_seed=seed, build_assembly=None)
        asym = parse(path, config=config)["asym_unit"]

        # Only one altloc letter per chain (plus defaults)
        altloc_ids = get_annotation(asym, "label_alt_id")
        if altloc_ids is not None:
            for chain_id in np.unique(asym.chain_id):
                chain_alts = altloc_ids[asym.chain_id == chain_id]
                letters = {a for a in chain_alts if a not in ALTLOC_DEFAULT_IDS}
                assert len(letters) <= 1, f"seed={seed} chain={chain_id}: multiple altlocs {letters}"

        coord_hashes.add(asym.coord.tobytes())

    assert len(coord_hashes) > 1, "All 5 seeds produced identical coordinates"


def test_6lzb_cross_chain_altlocs_linked():
    """6LZB: cross-chain proximal residues with shared altloc letters are consistently linked."""
    path = get_pdb_path_or_buffer("6lzb")
    cif = read_any(path)
    raw = pdbx.get_structure(cif, extra_fields=["label_alt_id"], altloc="all", use_author_fields=False)
    if hasattr(raw, "stack_depth"):
        raw = raw[0]

    altloc_ids = get_annotation(raw, "altloc_id")
    defaults = np.isin(altloc_ids, ALTLOC_DEFAULT_IDS)
    groups = _identify_altloc_groups(raw, altloc_ids, defaults)
    _, preferences = _build_constraints(raw, groups, clash_distance=1.5, proximity_distance=5.0)
    assert len(preferences) > 0, "6LZB: expected cross-chain proximity preferences"

    for seed in range(5):
        result = select_altlocs_clash_aware(raw, seed=seed)
        result_altlocs = get_annotation(result, "altloc_id")
        assert_no_inter_residue_altloc_clashes(result, label=f"6lzb seed={seed}")

        for res_i, res_j in preferences:
            mask_i = (result.chain_id == res_i[0]) & (result.res_id == res_i[1])
            mask_j = (result.chain_id == res_j[0]) & (result.res_id == res_j[1])
            letters_i = set(result_altlocs[mask_i]) - set(ALTLOC_DEFAULT_IDS)
            letters_j = set(result_altlocs[mask_j]) - set(ALTLOC_DEFAULT_IDS)
            if letters_i and letters_j:
                assert letters_i == letters_j, f"seed={seed}: {res_i}<->{res_j} got {letters_i} vs {letters_j}"


def test_6qhp_mutually_exclusive_catalytic_states():
    """6QHP: ASB/FAH are mutually exclusive catalytic states; ASB falls back to parent ASP."""
    path = get_pdb_path_or_buffer("6qhp")
    asp_template_atoms = {"C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"}

    states_seen: set[tuple[bool, bool]] = set()
    fallback_verified = False
    for seed in range(3):
        config = dataclasses.replace(_CLASH_AWARE_CONFIG, altloc_seed=seed)
        asym = parse(path, config=config)["asym_unit"]
        assert_no_inter_residue_altloc_clashes(asym, label=f"6qhp seed={seed}")

        has_asb = bool(np.any(asym.res_name == "ASB"))
        has_fah_c = bool(np.any((asym.res_name == "FAH") & (asym.chain_id == "C")))
        assert not (has_asb and has_fah_c), f"seed={seed}: ASB and FAH(C) both present"

        res112_mask = (asym.chain_id == "A") & (asym.res_id == 112)
        if has_fah_c:
            assert np.any(res112_mask), f"seed={seed}: res 112 absent — parent fallback failed"
            assert (
                asym.res_name[res112_mask][0] == "ASP"
            ), f"seed={seed}: expected ASP, got {asym.res_name[res112_mask][0]}"
            actual_atoms = set(asym.atom_name[res112_mask])
            assert (
                actual_atoms <= asp_template_atoms
            ), f"seed={seed}: unexpected atoms: {actual_atoms - asp_template_atoms}"
            assert len(actual_atoms) >= 7, f"seed={seed}: too few atoms: {actual_atoms}"
            arr = asym[0] if hasattr(asym, "stack_depth") else asym
            assert not np.any(np.isnan(arr.coord[res112_mask])), f"seed={seed}: NaN in fallback coords"
            fallback_verified = True

        states_seen.add((has_asb, has_fah_c))

    assert (True, False) in states_seen, "Never sampled post-reaction state (ASB present)"
    assert (False, True) in states_seen, "Never sampled pre-reaction state (FAH present)"
    assert fallback_verified, "No seed triggered parent fallback"


def test_4hbt_independent_altloc_sampling():
    """4HBT: 9 unconstrained altloc residues — 4 target residues sampled independently."""
    path = get_pdb_path_or_buffer("4hbt")
    target_res_ids = [14, 68, 73, 246]
    combos_seen: set[tuple[str, ...]] = set()

    for seed in range(3):
        if isinstance(path, io.StringIO):
            path.seek(0)
        config = dataclasses.replace(_CLASH_AWARE_CONFIG, altloc_seed=seed)
        asym = parse(path, config=config)["asym_unit"]
        assert_no_inter_residue_altloc_clashes(asym, label=f"4hbt seed={seed}")

        altloc_ids = get_annotation(asym, "altloc_id")
        chain_mask = asym.chain_id == "A"

        letters = []
        for rid in target_res_ids:
            res_mask = chain_mask & (asym.res_id == rid)
            non_default = ~np.isin(altloc_ids[res_mask], ALTLOC_DEFAULT_IDS)
            res_letters = set(altloc_ids[res_mask][non_default])
            assert len(res_letters) == 1, f"seed={seed}, res_id={rid}: expected 1 altloc, got {res_letters}"
            letters.append(res_letters.pop())

        combos_seen.add(tuple(letters))

    # With 5 seeds and 2^4=16 combos, we won't see all 16 — just verify diversity
    assert len(combos_seen) > 1, f"All 5 seeds produced identical altloc combinations: {combos_seen}"
