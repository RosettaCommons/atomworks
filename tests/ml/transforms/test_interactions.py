"""Tests for non-covalent interaction annotation (PLIP-aligned)."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from atomworks.io import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.protonation import ensure_hydrogens
from atomworks.ml.transforms.interactions import (
    AnnotateInteractions,
    HBondRole,
    MetalRole,
    PiCationRole,
    annotate_interactions,
)
from atomworks.ml.utils.testing import cached_parse, get_pdb_mirror_path

_INTERACTION_DATA = Path(__file__).parent.parent.parent / "data" / "io" / "interactions"


def _parse_pdb(pdb_id: str) -> dict:
    """Parse a PDB structure by ID, returning the standard parse dict."""
    data = parse(get_pdb_mirror_path(pdb_id))
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    return data


def _parse_with_h(pdb_id: str) -> AtomArray:
    """Parse structure and add explicit H for interaction detection."""
    aa = _parse_pdb(pdb_id)["atom_array"]
    return ensure_hydrogens(aa)


def test_inter_chain_only_protein_metal():
    """1fu2: protein + ZN. Inter-chain only should find fewer H-bonds than all-chain."""
    atom_array = _parse_with_h("1fu2")

    results_inter = annotate_interactions(atom_array, inter_chain_only=True)
    results_all = annotate_interactions(atom_array, inter_chain_only=False)

    assert len(results_all["hbond"]) > len(
        results_inter["hbond"]
    ), "Inter-chain filtering should remove intra-chain H-bonds"

    chain_ids = atom_array.chain_id
    for d, a in results_inter["hbond"]:
        assert chain_ids[d] != chain_ids[a], "Inter-chain H-bonds should be between different chains"

    for ci, ai in results_inter["saltbridge"]:
        assert chain_ids[ci] != chain_ids[ai], "Inter-chain salt bridges should be between different chains"


def test_inter_chain_metal_coordination():
    """1fu2: ZN metal should have inter-chain coordination with protein residues."""
    atom_array = _parse_with_h("1fu2")

    results = annotate_interactions(atom_array, inter_chain_only=True)

    zn_mask = atom_array.element == "ZN"
    if zn_mask.any():
        assert len(results["metal"]) > 0, "Should find ZN metal coordination"
        metal_atoms = {p[0] for p in results["metal"]} | {p[1] for p in results["metal"]}
        assert metal_atoms & set(np.where(zn_mask)[0]), "ZN should participate in metal coordination"


def test_annotate_interactions_protein_dna():
    """6w13: protein + DNA + ligand + MG. All interaction types should work."""
    atom_array = _parse_with_h("6w13")

    results = annotate_interactions(atom_array, hbond_model="rosetta", inter_chain_only=True)

    chain_ids = atom_array.chain_id
    for d, a in results["hbond"]:
        assert chain_ids[d] != chain_ids[a]

    mg_mask = atom_array.element == "MG"
    if mg_mask.any():
        assert len(results["metal"]) > 0, "Should find MG metal coordination"


def test_hbond_rosetta_vs_plip():
    """Rosetta mode should find fewer but higher-quality H-bonds than PLIP mode."""
    atom_array = _parse_with_h("1fu2")

    rosetta = annotate_interactions(
        atom_array, hbond_model="rosetta", inter_chain_only=False, interaction_types=("hbond",)
    )
    plip = annotate_interactions(atom_array, hbond_model="plip", inter_chain_only=False, interaction_types=("hbond",))

    assert len(rosetta["hbond"]) > 0
    assert len(plip["hbond"]) > 0
    assert len(rosetta["hbond"]) <= len(
        plip["hbond"]
    ), f"Rosetta ({len(rosetta['hbond'])}) should find <= PLIP ({len(plip['hbond'])}) H-bonds"


def test_1d_role_annotations():
    """1D per-atom H-bond roles should be consistent with detected 2D pairs."""
    atom_array = _parse_with_h("1fu2")
    results = annotate_interactions(atom_array, hbond_model="rosetta", inter_chain_only=False)

    hbonds = results["hbond"]
    assert len(hbonds) > 0, "Should detect H-bonds in 1fu2"

    hbond_role = np.full(len(atom_array), HBondRole.NONE, dtype=np.int32)
    for d, a in hbonds:
        hbond_role[d] |= HBondRole.DONOR
        hbond_role[a] |= HBondRole.ACCEPTOR

    assert set(np.unique(hbond_role)) <= {HBondRole.NONE, HBondRole.DONOR, HBondRole.ACCEPTOR, HBondRole.BOTH}
    assert (hbond_role != HBondRole.NONE).any(), "Should have some H-bond roles"


def test_hbond_both_role():
    """Atoms that are both donor and acceptor should get BOTH=3."""
    atom_array = _parse_with_h("1fu2")
    results = annotate_interactions(atom_array, hbond_model="plip", inter_chain_only=False)

    hbonds = results["hbond"]
    hbond_role = np.full(len(atom_array), HBondRole.NONE, dtype=np.int32)
    for d, a in hbonds:
        hbond_role[d] |= HBondRole.DONOR
        hbond_role[a] |= HBondRole.ACCEPTOR

    both_mask = hbond_role == HBondRole.BOTH
    assert both_mask.any(), "Expected some atoms with BOTH role in 1fu2 (plip mode, intra+inter chain)"

    for idx in np.where(both_mask)[0]:
        assert hbond_role[idx] == 3
        assert hbond_role[idx] & HBondRole.DONOR
        assert hbond_role[idx] & HBondRole.ACCEPTOR


def test_transform_produces_2d_and_1d_annotations():
    """AnnotateInteractions should produce both 2D and 1D annotations."""
    data = _parse_pdb("1fu2")

    transform = AnnotateInteractions(hbond_model="rosetta", inter_chain_only=True)
    result = transform(data)
    aap = result["atom_array"]

    expected_2d = [
        "interaction_hbond",
        "interaction_hydrophobic",
        "interaction_pistacking",
        "interaction_pication",
        "interaction_saltbridge",
        "interaction_halogen",
        "interaction_metal",
    ]
    for name in expected_2d:
        assert name in aap.get_annotation_2d_categories(), f"Missing 2D annotation: {name}"

    expected_1d = [
        "interaction_hbond_role",
        "interaction_hydrophobic",
        "interaction_aromatic",
        "interaction_pistacking",
        "interaction_pication_role",
        "interaction_charged_role",
        "interaction_metal_role",
        "interaction_halogen",
    ]
    for name in expected_1d:
        assert name in aap.get_annotation_categories(), f"Missing 1D annotation: {name}"


def test_no_same_residue_hydrophobic():
    """Hydrophobic contacts should not include atoms within the same residue."""
    atom_array = _parse_with_h("1fu2")

    results = annotate_interactions(atom_array, interaction_types=("hydrophobic",), inter_chain_only=False)
    for ai, aj in results["hydrophobic"]:
        same_res = atom_array.res_id[ai] == atom_array.res_id[aj] and atom_array.chain_id[ai] == atom_array.chain_id[aj]
        assert not same_res, f"Hydrophobic pair ({ai}, {aj}) in same residue"


@pytest.mark.parametrize("with_charge", [False, True])
def test_synthetic_saltbridge(with_charge):
    """Synthetic test: ARG-ASP pair across chains should be detected."""
    atom_array = AtomArray(4)
    atom_array.chain_id = np.array(["A", "A", "B", "B"])
    atom_array.res_id = np.array([1, 1, 2, 2])
    atom_array.res_name = np.array(["ARG", "ARG", "ASP", "ASP"])
    atom_array.atom_name = np.array(["CZ", "NH1", "CG", "OD1"])
    atom_array.element = np.array(["C", "N", "C", "O"])
    atom_array.coord = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [4.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
        dtype=np.float32,
    )

    atom_array.set_annotation("chain_iid", np.array(["A_1", "A_1", "B_1", "B_1"]))
    if with_charge:
        atom_array.set_annotation("charge", np.zeros(4, dtype=int))
    atom_array.bonds = struc.connect_via_residue_names(atom_array)
    results = annotate_interactions(atom_array, interaction_types=("saltbridge",), inter_chain_only=True)

    assert len(results["saltbridge"]) > 0, "Should detect inter-chain salt bridge"
    for ci, ai in results["saltbridge"]:
        assert atom_array.chain_id[ci] != atom_array.chain_id[ai]


def test_saltbridge_with_charged_ligand():
    """A negatively charged ligand oxygen forms a salt bridge with a nearby lysine."""
    aa = AtomArray(2)
    aa.chain_id = ["A", "B"]
    aa.res_id = [1, 2]
    aa.res_name = ["LYS", "LIG"]
    aa.atom_name = ["NZ", "O1"]
    aa.element = ["N", "O"]
    aa.hetero = [False, True]
    aa.coord = np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float32)
    aa.bonds = struc.BondList(2)
    aa.set_annotation("charge", np.array([1, -1]))

    result = annotate_interactions(aa, interaction_types=("saltbridge",))
    assert result["saltbridge"] == [(0, 1)]

    aa.charge[1] = 0
    assert annotate_interactions(aa, interaction_types=("saltbridge",))["saltbridge"] == []

    aa.charge[1] = -1
    aa.coord[1] = [6, 0, 0]
    assert annotate_interactions(aa, interaction_types=("saltbridge",))["saltbridge"] == []


def test_metal_role_annotations():
    """1fu2: metal role annotations should mark ZN as METAL and coordinating atoms."""
    data = _parse_pdb("1fu2")

    transform = AnnotateInteractions(inter_chain_only=True)
    result = transform(data)
    aap = result["atom_array"]

    metal_role = aap.interaction_metal_role
    zn_mask = aap.element == "ZN"

    if zn_mask.any() and (metal_role == MetalRole.METAL).any():
        assert np.all(metal_role[zn_mask & (metal_role != MetalRole.NONE)] == MetalRole.METAL)
        assert (metal_role == MetalRole.COORDINATING).any(), "Should have coordinating atoms"


def test_ensure_hydrogens_updates_charge():
    """ensure_hydrogens should assign pH-aware charges via Dimorphite-DL.

    Verifies that after protonation at pH 7.4:
    - ASP carboxylate (pKa ~3.7) is deprotonated → negative charge
    - LYS NZ (pKa ~10.5) is protonated → positive charge
    - H atoms were added to the structure
    """
    aa = _load_fixture_raw("PTE_enhanced_KCX.cif")
    if "chain_iid" not in aa.get_annotation_categories():
        aa.set_annotation("chain_iid", np.char.add(aa.chain_id.astype(str), "_1"))
    aa.charge = np.zeros(len(aa), dtype=int)

    original_annots = set(aa.get_annotation_categories())

    result = ensure_hydrogens(aa, ph=7.4)

    assert len(result) > len(aa), "Should have added H atoms"
    assert set(result.get_annotation_categories()).issuperset(original_annots), "Should preserve original annotations"

    atom_names = np.char.strip(result.atom_name.astype(str))

    ensure_annotations(result, "is_standard_aa")
    is_backbone_n = (result.atom_name == "N") & result.is_standard_aa
    for idx in np.where(is_backbone_n)[0]:
        bonded_atoms = list(result.bonds.get_bonds(idx)[0])
        bonded_h = [b for b in bonded_atoms if result.element[b] in HYDROGEN_LIKE_SYMBOLS]

        # Amide-bonded backbone N should have neutral charge and one bonded H at pH 7.4
        if any(
            (result.atom_name[target_atom] == "C") and result.is_standard_aa[target_atom]
            for target_atom in bonded_atoms
        ):
            assert result.charge[idx] == 0, f"Backbone N should be neutral, got {result.charge[idx]}"
            assert len(bonded_h) == 1, f"Backbone N should have 1 bonded H, got {len(bonded_h)}"

        # Primary amine backbone N should be protonated and positively charged at pH 7.4
        elif len(bonded_h) == len(bonded_atoms) - 1:
            assert (
                result.charge[idx] == 1
            ), f"Non-amide N should be protonated and positively charged, got {result.charge[idx]}"
            assert len(bonded_h) == 3, f"Non-amide N should have 3 bonded H, got {len(bonded_h)}"

    # ASP carboxylate should be deprotonated at pH 7.4 (pKa ~3.7)
    asp_mask = result.res_name == "ASP"
    asp_od_mask = asp_mask & np.isin(atom_names, ["OD1", "OD2"])
    if asp_od_mask.any():
        asp_charge = result.charge[asp_od_mask].sum()
        assert asp_charge < 0, f"ASP OD1/OD2 should carry negative charge at pH 7.4, got {asp_charge}"

    # LYS ammonium should be protonated at pH 7.4 (pKa ~10.5)
    lys_nz_mask = (result.res_name == "LYS") & (atom_names == "NZ")
    if lys_nz_mask.any():
        lys_charge = result.charge[lys_nz_mask].sum()
        assert lys_charge > 0, f"LYS NZ should carry positive charge at pH 7.4, got {lys_charge}"


# ---------------------------------------------------------------------------
# Ground-truth interaction tests (Avery's conditioning examples)
# ---------------------------------------------------------------------------


def _load_fixture_raw(filename: str, remove_waters: bool = True) -> struc.AtomArray:
    """Load a CIF fixture via parse with bonds from file and chain_iid."""
    aa = parse(str(_INTERACTION_DATA / filename), config="rcsb")["asym_unit"][0]
    if remove_waters:
        aa = aa[aa.res_name != "HOH"]
    if "chain_iid" not in aa.get_annotation_categories():
        aa.set_annotation("chain_iid", np.char.add(aa.chain_id.astype(str), "_1"))
    return aa


def _load_fixture(filename: str, inter_chain_only: bool = True, remove_waters: bool = True) -> dict:
    """Load a CIF fixture and run AnnotateInteractions."""
    aa = _load_fixture_raw(filename, remove_waters=remove_waters)
    transform = AnnotateInteractions(inter_chain_only=inter_chain_only)
    return transform({"atom_array": aa})


def _find_atom(aap, chain: str, res_id: int, atom_name: str) -> int | None:
    """Return the index of a specific atom, or None if not found."""
    mask = (aap.chain_id == chain) & (aap.res_id == res_id) & (np.char.strip(aap.atom_name) == atom_name)
    indices = np.where(mask)[0]
    return int(indices[0]) if len(indices) > 0 else None


def _has_2d_pair(aap, annotation_name: str, chain_a: str, res_id_a: int, chain_b: str, res_id_b: int) -> bool:
    """Check if a 2D annotation contains a pair between two residues (either direction)."""
    ann_2d = aap.get_annotation_2d(annotation_name)
    pairs = ann_2d.as_array()
    for row in pairs:
        a, b = int(row[0]), int(row[1])
        if (
            aap.chain_id[a] == chain_a
            and aap.res_id[a] == res_id_a
            and aap.chain_id[b] == chain_b
            and aap.res_id[b] == res_id_b
        ) or (
            aap.chain_id[a] == chain_b
            and aap.res_id[a] == res_id_b
            and aap.chain_id[b] == chain_a
            and aap.res_id[b] == res_id_a
        ):
            return True
    return False


def test_iron_coordination_9oer():
    """9OER: Fe octahedral coordination with His, Cl, and substrate oxygens.

    Ground truth (from Avery):
      Chain B Fe: fully octahedral — CL 304, A1CA1 302 O07/O05/O01,
                  His 209 NE2, His 142 NE2 (6 partners)
      Chain R Fe: 5/6 octahedral — His 209 NE2, His 142 NE2,
                  CL 304, AKG 302 O2/O5
    Fe and its coordinating residues share the same chain, so
    inter_chain_only=False is needed.
    """
    aa = cached_parse("9oer")["atom_array"]
    transform = AnnotateInteractions(inter_chain_only=False)
    result = transform({"atom_array": aa})
    aap = result["atom_array"]
    metal_role = aap.interaction_metal_role

    # Fe atoms should be marked as METAL
    fe_mask = aap.element == "FE"
    assert fe_mask.any(), "Structure should contain Fe atoms"
    assert (metal_role[fe_mask] == MetalRole.METAL).all(), "All Fe atoms should have METAL role"

    # 2D: interaction_metal pairs should exist
    metal_2d = aap.get_annotation_2d("interaction_metal")
    pairs = metal_2d.as_array()
    assert len(pairs) > 0, "Should detect metal coordination pairs"

    fe_indices = set(np.where(fe_mask)[0])
    pair_fe = [(int(a), int(b)) for a, b in pairs[:, :2] if int(a) in fe_indices or int(b) in fe_indices]
    assert len(pair_fe) >= 5, f"Fe should have >= 5 coordination pairs total, got {len(pair_fe)}"

    # 2D: verify specific Fe-His coordination pairs
    assert _has_2d_pair(
        aap, "interaction_metal", "K", 303, "A", 142
    ), "2D metal annotation should contain Fe(AA303)-His(E142) pair"
    assert _has_2d_pair(
        aap, "interaction_metal", "K", 303, "A", 209
    ), "2D metal annotation should contain Fe(303)-His(209) pair"

    # 1D: His 142 NE2 and His 209 NE2 should be COORDINATING in chain A
    for res_id in (142, 209):
        idx = _find_atom(aap, "A", res_id, "NE2")
        assert idx is not None, f"Expected chain A His {res_id} NE2 in fixture"
        assert (
            metal_role[idx] == MetalRole.COORDINATING
        ), f"Chain A His {res_id} NE2 should be COORDINATING, got {metal_role[idx]}"

    # Overall: multiple atoms should be marked COORDINATING
    n_coordinating = (metal_role == MetalRole.COORDINATING).sum()
    assert n_coordinating >= 5, f"Expected >= 5 coordinating atoms, got {n_coordinating}"


def test_magnesium_coordination_1izc():
    """1IZC: Mg coordination by Glu, Asp, pyruvate oxygens, and water.

    Ground truth (from Avery):
      Mg coordinated by: Chain A Glu 185 OE2, Chain A Asp 211 OD2,
      Chain C PYR 2001 O/O3, Chain I HOH 2036 O, Chain H HOH 2245 O
    Mg is in chain G (label_asym_id); coordinating atoms span multiple
    chains, so inter_chain_only=False captures all coordination.
    """
    result = _load_fixture("1IZC_macrophomate_synthase_dimer.cif", inter_chain_only=False, remove_waters=False)
    aap = result["atom_array"]
    metal_role = aap.interaction_metal_role

    # Mg atoms should be marked as METAL
    mg_mask = aap.element == "MG"
    assert mg_mask.any(), "Structure should contain Mg atoms"
    assert (metal_role[mg_mask] == MetalRole.METAL).all(), "All Mg atoms should have METAL role"

    # 2D: interaction_metal pairs should exist
    metal_2d = aap.get_annotation_2d("interaction_metal")
    pairs = metal_2d.as_array()
    assert len(pairs) > 0, "Should detect Mg coordination pairs"

    mg_indices = set(np.where(mg_mask)[0])
    pair_mg = [(int(a), int(b)) for a, b in pairs[:, :2] if int(a) in mg_indices or int(b) in mg_indices]
    assert len(pair_mg) >= 4, f"Mg should have >= 4 coordination pairs, got {len(pair_mg)}"

    # 1D: Glu 185 OE1 should be COORDINATING (OE1 is the carboxylate O
    # closest to Mg at 2.08 A; Avery's notes say OE2 but the 3D geometry
    # places OE1 in the coordination shell)
    idx = _find_atom(aap, "A", 185, "OE1")
    assert idx is not None, "Expected chain A Glu 185 OE1 in fixture"
    assert (
        metal_role[idx] == MetalRole.COORDINATING
    ), f"Chain A Glu 185 OE1 should be COORDINATING, got {metal_role[idx]}"

    # 1D: Asp 211 OD2 should be COORDINATING (2.12 A from Mg)
    idx = _find_atom(aap, "A", 211, "OD2")
    assert idx is not None, "Expected chain A Asp 211 OD2 in fixture"
    assert (
        metal_role[idx] == MetalRole.COORDINATING
    ), f"Chain A Asp 211 OD2 should be COORDINATING, got {metal_role[idx]}"

    # Overall coordinating count
    n_coordinating = (metal_role == MetalRole.COORDINATING).sum()
    assert n_coordinating >= 4, f"Expected >= 4 coordinating atoms, got {n_coordinating}"


def test_pistacking_7mkv():
    """7MKV: His 148 pi-stacks with LLP 258 (covalently bound PLP).

    Ground truth (from Avery):
      Chain B His 148 pi-stacks Chain B LLP 258. Same in Chain C.
      Both are in the same chain, so inter_chain_only=False is needed.
    """
    result = _load_fixture("7MKV_UstD_2_dimer.cif", inter_chain_only=False)
    aap = result["atom_array"]

    # 2D: pi-stacking pairs should exist
    pistack_2d = aap.get_annotation_2d("interaction_pistacking")
    pairs = pistack_2d.as_array()
    assert len(pairs) > 0, "Should detect pi-stacking interactions"

    # 2D: His 148 - LLP 258 pair should be explicitly detected in both chains
    assert _has_2d_pair(
        aap, "interaction_pistacking", "B", 148, "B", 258
    ), "2D pistacking should contain His(148)-LLP(258) pair in chain B"
    assert _has_2d_pair(
        aap, "interaction_pistacking", "C", 148, "C", 258
    ), "2D pistacking should contain His(148)-LLP(258) pair in chain C"

    # 1D: His 148 ring atoms in chain B should be marked aromatic
    is_aromatic = aap.interaction_aromatic
    his148_ring_names = {"CG", "ND1", "CD2", "CE1", "NE2"}
    chain_b_his148 = (aap.chain_id == "B") & (aap.res_id == 148)
    his148_ring_mask = chain_b_his148 & np.isin(np.char.strip(aap.atom_name), list(his148_ring_names))

    assert his148_ring_mask.any(), "Expected His 148 ring atoms in chain B"
    assert is_aromatic[his148_ring_mask].any(), "Chain B His 148 ring atoms should be marked aromatic"

    # 1D: LLP 258 should also have aromatic atoms (detected via RDKit SSSR)
    chain_b_llp258 = (aap.chain_id == "B") & (aap.res_id == 258) & (aap.res_name == "LLP")
    assert chain_b_llp258.any(), "Expected LLP 258 in chain B"
    assert is_aromatic[chain_b_llp258].any(), "Chain B LLP 258 should have aromatic atoms from RDKit ring detection"

    # Same pattern should occur in chain C
    chain_c_his148 = (aap.chain_id == "C") & (aap.res_id == 148)
    his148_c_ring = chain_c_his148 & np.isin(np.char.strip(aap.atom_name), list(his148_ring_names))
    assert his148_c_ring.any(), "Expected His 148 ring atoms in chain C"
    assert is_aromatic[his148_c_ring].any(), "Chain C His 148 ring atoms should also be marked aromatic"

    # interaction_pistacking: ring atoms in detected pairs should be AROMATIC
    is_pistacking = aap.interaction_pistacking
    assert is_pistacking[his148_ring_mask].all(), "Chain B His 148 ring atoms should be marked in pistacking_role"
    # Non-ring atoms of His 148 in chain B should not be marked
    assert not is_pistacking[~his148_ring_mask & (aap.chain_id == "B") & (aap.res_id == 148)].any()


# ---------------------------------------------------------------------------
# Theozyme ground-truth tests (Seth's enzyme design examples)
# ---------------------------------------------------------------------------


def test_pte_theozyme_metal_coordination():
    """PTE group1 theozyme: binuclear Zn site with His/Glu coordination.

    Golden values (exact pairs detected):
      ZN1 (YYE 9): His 1 NE2, His 2 NE2, Glu 6 OE1, YYE O1, YYE O3
      ZN2 (YYE 9): His 4 NE2, His 5 NE2, YYE O2, YYE O3, YYE O5
      No pi-stacking in this minimal theozyme.
    """
    result = _load_fixture("PTE_group1_theozyme.cif", inter_chain_only=False)
    aap = result["atom_array"]
    metal_role = aap.interaction_metal_role

    # Both Zn atoms should be METAL
    zn_mask = aap.element == "ZN"
    assert zn_mask.sum() == 2, f"Expected 2 Zn atoms, got {zn_mask.sum()}"
    assert (metal_role[zn_mask] == MetalRole.METAL).all(), "Both Zn should have METAL role"

    # Exactly 10 metal coordination pairs (5 per Zn)
    metal_2d = aap.get_annotation_2d("interaction_metal")
    pairs = metal_2d.as_array()
    assert len(pairs) == 10, f"Expected 10 metal pairs, got {len(pairs)}"

    # ZN1 coordination: His 1 NE2, His 2 NE2, Glu 6 OE1, YYE O1, YYE O3
    for chain, res_id, atom_name in [("A", 1, "NE2"), ("B", 2, "NE2"), ("F", 6, "OE1")]:
        idx = _find_atom(aap, chain, res_id, atom_name)
        assert idx is not None, f"Expected chain {chain} res {res_id} {atom_name}"
        assert (
            metal_role[idx] == MetalRole.COORDINATING
        ), f"Chain {chain} res {res_id} {atom_name} should be COORDINATING"

    # ZN2 coordination: His 4 NE2, His 5 NE2
    for chain, res_id in [("D", 4), ("E", 5)]:
        idx = _find_atom(aap, chain, res_id, "NE2")
        assert idx is not None, f"Expected chain {chain} His {res_id} NE2"
        assert metal_role[idx] == MetalRole.COORDINATING, f"Chain {chain} His {res_id} NE2 should be COORDINATING"

    # No pi-stacking in this minimal theozyme
    pistack_2d = aap.get_annotation_2d("interaction_pistacking")
    assert len(pistack_2d.as_array()) == 0, "No pi-stacking expected in minimal PTE theozyme"


def test_enhanced_pte_multivalent_pistacking():
    """Enhanced PTE (KCX): Phe 132 makes 2 pi-stack interactions (multivalent).

    Golden values (exact pairs detected):
      Trp 131 CG -- Phe 132 CG (parallel, Trp 5-ring)
      Trp 131 CD2 -- Phe 132 CG (parallel, Trp 6-ring)
      Phe 132 CG -- His 201 CG (T-stack)

    Metal coordination (10 pairs):
      ZN1: His 55 NE2, His 57 NE2, ASP 301 OD2, XDW O2, XDW O3
      ZN2: His 201 ND1, His 230 NE2, XDW O1, XDW O3, XDW O5
    """
    result = _load_fixture("PTE_enhanced_KCX.cif", inter_chain_only=False)
    aap = result["atom_array"]
    atom_names = np.char.strip(aap.atom_name)

    # --- Pi-stacking: exactly 3 pairs ---
    pistack_2d = aap.get_annotation_2d("interaction_pistacking")
    pistack_pairs = pistack_2d.as_array()
    assert len(pistack_pairs) == 96, f"Expected 96 pi-stacking atom pairs, got {len(pistack_pairs)}"

    # 1D: Phe 132, Trp 131, His 201 should all be aromatic
    is_aromatic = aap.interaction_aromatic
    for res_id, res_name in [(131, "TRP"), (132, "PHE"), (201, "HIS")]:
        res_mask = (aap.chain_id == "A") & (aap.res_id == res_id)
        assert is_aromatic[res_mask].any(), f"{res_name} {res_id} should have aromatic atoms"

    # Phe 132 participates in stacking with Trp 131 and His 201 (multivalent)
    phe132_ring_names = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    phe132_mask = (aap.chain_id == "A") & (aap.res_id == 132) & np.isin(atom_names, list(phe132_ring_names))
    phe132_indices = set(np.where(phe132_mask)[0])

    phe132_partners = set()
    for row in pistack_pairs:
        a, b = int(row[0]), int(row[1])
        if a in phe132_indices:
            phe132_partners.add(aap.res_id[b])
        elif b in phe132_indices:
            phe132_partners.add(aap.res_id[a])
    assert phe132_partners == {131, 201}, f"Phe 132 should pi-stack with Trp 131 and His 201, got: {phe132_partners}"

    # --- Metal coordination: exactly 10 pairs ---
    metal_2d = aap.get_annotation_2d("interaction_metal")
    metal_pairs = metal_2d.as_array()
    assert len(metal_pairs) == 10, f"Expected 10 metal pairs, got {len(metal_pairs)}"

    # His 55, 57, 201, 230 should be COORDINATING
    metal_role = aap.interaction_metal_role
    for res_id, atom_name in [(55, "NE2"), (57, "NE2"), (201, "ND1"), (230, "NE2")]:
        idx = _find_atom(aap, "A", res_id, atom_name)
        assert idx is not None, f"Expected His {res_id} {atom_name}"
        assert metal_role[idx] == MetalRole.COORDINATING, f"His {res_id} {atom_name} should be COORDINATING"


def test_pication_roles_1gai():
    """1GAI: LYS 107 NZ makes pi-cation interactions with TYR 115 and TRP 119.

    Ground truth: LYS 107 NZ (cation) interacts with the aromatic rings of both
    TYR 115 and TRP 119 in the same chain.  All three residues are intra-chain,
    so inter_chain_only=False is required.

    Checks both 2D pair detection and 1D role assignments for each interaction.
    """
    aa = cached_parse("1gai")["atom_array"]
    transform = AnnotateInteractions(inter_chain_only=False)
    result = transform({"atom_array": aa})
    aap = result["atom_array"]

    # --- 2D pair checks ---
    assert _has_2d_pair(
        aap, "interaction_pication", "A", 107, "A", 115
    ), "interaction_pication should contain LYS(107)-TYR(115) pair in chain A"
    assert _has_2d_pair(
        aap, "interaction_pication", "A", 107, "A", 119
    ), "interaction_pication should contain LYS(107)-TRP(119) pair in chain A"

    pication_role = aap.interaction_pication_role

    # --- 1D role checks: TYR 115 ring atoms should be AROMATIC ---
    tyr_ring_names = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    tyr115_ring = (
        (aap.chain_id == "A") & (aap.res_id == 115) & np.isin(np.char.strip(aap.atom_name), list(tyr_ring_names))
    )
    assert tyr115_ring.sum() == len(tyr_ring_names), "Expected TYR 115 ring atoms in chain A"
    assert (
        pication_role[tyr115_ring] & PiCationRole.AROMATIC
    ).all(), "All TYR 115 ring atoms should have AROMATIC role in pication"

    # --- 1D role checks: TRP 119 ring atoms should be AROMATIC ---
    trp_ring_names = {"CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"}
    trp119_ring = (
        (aap.chain_id == "A") & (aap.res_id == 119) & np.isin(np.char.strip(aap.atom_name), list(trp_ring_names))
    )
    assert trp119_ring.sum() == len(trp_ring_names), "Expected TRP 119 benzene ring atoms in chain A"
    assert (
        pication_role[trp119_ring] & PiCationRole.AROMATIC
    ).all(), "TRP 119 benzene ring atoms should have AROMATIC role in pication"

    # --- 1D role checks: LYS 107 NZ should be CATION ---
    lys107_nz = _find_atom(aap, "A", 107, "NZ")
    assert lys107_nz is not None, "Expected LYS 107 NZ in chain A"
    assert pication_role[lys107_nz] & PiCationRole.CATION, "LYS 107 NZ should have CATION role in pication"


def test_ensure_hydrogens_nan_robustness():
    """ensure_hydrogens places H on atoms whose neighbors have NaN coords.

    7MKV_UstD_2_dimer.cif has several residues  with unresolved side chains
    that are given NaN coordinates.  Their CB atoms become boundary atoms at the edge of the
    protonation component, bonded to NaN-coord neighbors. Before the fix, those
    neighbors were included in the local RDKit fragment, corrupting bond-vector geometry
    and triggering the "zero perpendicular" assertion inside AddHs(addCoords=True).
    """
    aa = _load_fixture_raw("7MKV_UstD_2_dimer.cif")
    result = ensure_hydrogens(aa, ph=7.4)

    # (chain, res_id, res_name, expected_CB_nhyd)
    # THR CB is CH1 → 1 H; GLU CB is CH2 → 2 H
    cases = [
        ("B", 31, "THR", 1),
        ("B", 52, "GLU", 2),
    ]

    for chain, res_id, res_name, expected_h in cases:
        cb_mask = (result.chain_id == chain) & (result.res_id == res_id) & (result.atom_name == "CB")
        cb_indices = np.where(cb_mask)[0]
        assert len(cb_indices) == 1, f"Expected exactly one CB for {chain}:{res_id} {res_name}"

        bonded, _ = result.bonds.get_bonds(int(cb_indices[0]))
        n_h = sum(1 for b in bonded if result.element[b] in HYDROGEN_LIKE_SYMBOLS)
        assert n_h == expected_h, f"{chain}:{res_id} {res_name} CB: expected {expected_h} bonded H, got {n_h}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
