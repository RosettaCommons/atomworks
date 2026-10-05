"""Interactions AnnotateInteractions finds on real structures, run as callers run it."""

from pathlib import Path

import numpy as np
import pytest

from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.ml.transforms.interactions import AnnotateInteractions, MetalRole, PiCationRole

DATA = Path(__file__).parents[2] / "data" / "io" / "interactions"


def _annotate(entry, inter_chain_only=False, remove_waters=True):
    atoms = parse(str(DATA / entry), config="rcsb")["asym_unit"][0]
    if remove_waters:
        atoms = atoms[atoms.res_name != "HOH"]
    if "chain_iid" not in atoms.get_annotation_categories():
        atoms.set_annotation("chain_iid", np.char.add(atoms.chain_id.astype(str), "_1"))
    return AnnotateInteractions(inter_chain_only=inter_chain_only)({"atom_array": atoms})["atom_array"]


def _residue_pairs(atoms, interaction):
    pairs = atoms.get_annotation_2d(f"interaction_{interaction}").as_array()[:, :2].astype(int)
    keys = list(zip(atoms.chain_id, atoms.res_id.tolist(), strict=True))
    return {frozenset((keys[a], keys[b])) for a, b in pairs}


def _atom(atoms, chain, res_id, atom_name):
    (index,) = np.flatnonzero((atoms.chain_id == chain) & (atoms.res_id == res_id) & (atoms.atom_name == atom_name))
    return index


# (entry, interaction, residue, residue): a pair of residues the entry is known for
PAIRS = [
    pytest.param("iron_site_9oer.cif.zst", "metal", ("K", 303), ("A", 142), id="9OER Fe-HIS142"),
    pytest.param("iron_site_9oer.cif.zst", "metal", ("K", 303), ("A", 209), id="9OER Fe-HIS209"),
    pytest.param("pication_1gai.cif.zst", "pication", ("A", 107), ("A", 115), id="1GAI LYS107-TYR115"),
    pytest.param("pication_1gai.cif.zst", "pication", ("A", 107), ("A", 119), id="1GAI LYS107-TRP119"),
    pytest.param("7MKV_UstD_2_dimer.cif", "pistacking", ("B", 148), ("B", 258), id="7MKV HIS148-LLP258 B"),
    pytest.param("7MKV_UstD_2_dimer.cif", "pistacking", ("C", 148), ("C", 258), id="7MKV HIS148-LLP258 C"),
    pytest.param("PTE_enhanced_KCX.cif", "pistacking", ("A", 132), ("A", 131), id="PTE PHE132-TRP131"),
    pytest.param("PTE_enhanced_KCX.cif", "pistacking", ("A", 132), ("A", 201), id="PTE PHE132-HIS201"),
]


@pytest.mark.parametrize(("entry", "interaction", "first", "second"), PAIRS)
def test_an_entry_shows_the_interaction_it_is_known_for(entry, interaction, first, second):
    assert frozenset((first, second)) in _residue_pairs(_annotate(entry), interaction)


# (entry, keep waters, chain, residue, atom): a donor the entry's metal is known to hold
COORDINATING = [
    pytest.param("1IZC_macrophomate_synthase_dimer.cif", False, "A", 185, "OE1", id="1IZC GLU185 OE1"),
    pytest.param("1IZC_macrophomate_synthase_dimer.cif", False, "A", 211, "OD2", id="1IZC ASP211 OD2"),
    pytest.param("PTE_group1_theozyme.cif", True, "A", 1, "NE2", id="PTE HIS1 NE2"),
    pytest.param("PTE_group1_theozyme.cif", True, "F", 6, "OE1", id="PTE GLU6 OE1"),
    pytest.param("PTE_enhanced_KCX.cif", True, "A", 201, "ND1", id="PTE HIS201 ND1"),
    pytest.param("PTE_enhanced_KCX.cif", True, "A", 230, "NE2", id="PTE HIS230 NE2"),
]


@pytest.mark.parametrize(("entry", "remove_waters", "chain", "res_id", "atom_name"), COORDINATING)
def test_a_metal_holds_the_donors_the_entry_is_known_for(entry, remove_waters, chain, res_id, atom_name):
    atoms = _annotate(entry, remove_waters=remove_waters)

    assert atoms.interaction_metal_role[_atom(atoms, chain, res_id, atom_name)] == MetalRole.COORDINATING


def test_a_lysine_over_two_rings_is_the_cation_of_both():
    """1GAI LYS 107 NZ is the cation of its pi-cation pairs."""
    atoms = _annotate("pication_1gai.cif.zst")

    assert atoms.interaction_pication_role[_atom(atoms, "A", 107, "NZ")] & PiCationRole.CATION


@pytest.mark.parametrize("model", ["plip", "rosetta"])
@pytest.mark.parametrize("inter_chain_only", [False, True])
@pytest.mark.parametrize(
    "entry, expected",
    [
        (
            "1bna",
            {(1, "N4", 12, "O6"), (12, "N1", 1, "N3"), (12, "N2", 1, "O2"), (5, "N6", 8, "O4"), (8, "N3", 5, "N1")},
        ),
        ("1ehz", {(1, "N1", 72, "N3"), (1, "N2", 72, "O2"), (72, "N4", 1, "O6")}),
    ],
)
def test_canonical_base_pair_hydrogen_bonds(entry, expected, inter_chain_only, model):
    atoms = parse(
        DATA / "nucleic_hbonds" / f"{entry}.cif.zst",
        config=ParseConfig.from_preset("minimal", add_id_and_entity_annotations=True),
    )["asym_unit"][0]
    annotated = AnnotateInteractions(
        interaction_types=("hbond",), hbond_model=model, inter_chain_only=inter_chain_only
    )({"atom_array": atoms})["atom_array"]
    pairs = annotated.get_annotation("interaction_hbond", n_body=2).pairs
    observed = {(atoms.res_id[i], atoms.atom_name[i], atoms.res_id[j], atoms.atom_name[j]) for i, j in pairs}
    assert observed == (set() if inter_chain_only and entry == "1ehz" else expected)
