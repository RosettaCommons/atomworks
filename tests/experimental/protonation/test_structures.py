"""Protonation of deposited structures, parsed and protonated end to end.

The fixtures are the metal sites, histidines, ligands, covalent attachments and
non-canonical residues TMol checks its own protonation against, cropped to the
residues around each site. Residue numbers are ``label_seq_id``.
"""

import functools
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import METAL_ELEMENTS
from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens
from atomworks.experimental.protonation._assign import AQUA_PKA, DEFAULT_AQUA_PKA, WATER_PKA
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from tests.experimental.protonation.test_regressions import DATA, hydrogens_on, is_h, protonate_array, residue_key


def atoms_named(atom_array, res_name, atom_name):
    return np.flatnonzero((atom_array.res_name == res_name) & (atom_array.atom_name == atom_name))


def partners_outside(atom_array, index):
    """Heavy atoms bonded to *index* in another residue."""
    neighbours, _ = atom_array.bonds.get_bonds(int(index))
    own = residue_key(atom_array, index)
    return [int(n) for n in neighbours if not is_h(atom_array)[n] and residue_key(atom_array, n) != own]


_TESTS = Path(__file__).parents[2]
# Fixtures other tests already keep, read from their files.
SHARED = {
    "acetylated_peptide_1j8z": _TESTS / "io" / "test_outputs" / "1j8z_new.cif",
    "af3_cyclic_peptide_7ubd": _TESTS / "data" / "io" / "7ubd_from_af3.cif",
    "conditional_generation": _TESTS / "data" / "io" / "example_conditional_generation_output.cif",
    "plp_enzyme_7mkv": _TESTS / "data" / "io" / "interactions" / "7MKV_UstD_2_dimer.cif",
    "repeated_glycans_6mub": _TESTS / "io" / "test_outputs" / "6mub_new.cif",
    "unknown_heavy_atom_1a8o": _TESTS / "data" / "io" / "1a8o_modified.cif",
    "unresolved_unl": _TESTS / "data" / "io" / "test_unl_ligand_with_bonds.cif",
}
# 1QD7 resolves only its phosphorus trace.
UNPROTONATABLE = {"atomized_nucleotide_chain_1qd7", "ccd_model_coords_3uq", "unknown_heavy_atom_1a8o", "unresolved_unl"}
FIXTURES = sorted(({p.name.split(".cif")[0] for p in DATA.glob("*.cif.zst")} | set(SHARED)) - UNPROTONATABLE)


def _path(name: str) -> Path:
    return SHARED.get(name, DATA / f"{name}.cif.zst")


def parse_fixture(name: str, **overrides):
    """A fixture parsed without hydrogens, completed, with covalent and metal bonds from struct_conn."""
    config = {
        "hydrogen_policy": "remove",
        "add_missing_atoms": True,
        "add_bond_types_from_struct_conn": ("covale", "metalc"),
        **overrides,
    }
    return parse(_path(name), config=ParseConfig(**config))["asym_unit"][0]


@functools.cache
def _protonated(name: str, overrides: tuple):
    return protonate_array(parse_fixture(name, **dict(overrides)))


def protonate(name: str, **overrides):
    """A fixture parsed and protonated at pH 7.4 (cached; a copy is returned)."""
    return _protonated(name, tuple(sorted(overrides.items()))).copy()


# Parsing drops crystallization aids such as PO4 and GOL by default.
_KEEP_AIDS = {"remove_ccds": ()}
OVERRIDES = {"phosphate_charge_4js1": _KEEP_AIDS}
# A metal takes the proton's place on these donors.
PROTON_DONORS = {"HIS": {"ND1", "NE2"}, "CYS": {"SG"}, "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
# The histidine and cysteine atoms each metal holds, as TMol's metal fixtures list them.
METAL_DONORS = {
    "zn_tetrahedral_3ks3": {"A:HIS94.NE2", "A:HIS96.NE2", "A:HIS119.ND1"},
    "cu_blue_copper_2ov0": {"A:HIS53.ND1", "A:HIS95.ND1", "A:CYS92.SG"},
    "cu_zn_sod_3f7l": {"A:HIS44.ND1", "A:HIS46.NE2", "A:HIS118.NE2", "A:HIS61.ND1", "A:HIS69.ND1", "A:HIS78.ND1"},
    "feo_myohemerythrin_2mhr": {f"A:HIS{n}.NE2" for n in (25, 54, 73, 77, 106)},
    "cua_ba3_2cua": {
        f"{chain}:{atom}"
        for chain in "AB"
        for atom in ("HIS81.ND1", "HIS124.ND1", "HIS84.NE2", "CYS116.SG", "CYS120.SG")
    },
    "heme_myoglobin_5yce": {"A:HIS94.NE2"},
    "fe_rubredoxin_30oh": {f"A:CYS{n}.SG" for n in (10, 13, 43, 46)},
    "fes_ferredoxin_6e6r": {f"A:CYS{n}.SG" for n in (4, 6, 38, 41)},
    "sf4_ferredoxin_1fdn": {f"A:CYS{n}.SG" for n in (8, 11, 14, 18, 37, 40, 43, 47)},
    "sf4_ferredoxin_2fdn": {f"A:CYS{n}.SG" for n in (8, 11, 14, 18, 37, 40, 43, 47)},
    "clf_nitrogenase_7adr": {"A:CYS49.SG", "A:CYS75.SG", "A:CYS138.SG", "B:CYS31.SG", "B:CYS56.SG", "B:CYS115.SG"},
}
# (fixture, residue, atoms or None for all, bonded to another residue or None for either,
#  hydrogens, formal charge or None for any[, parse overrides])
ATOM_STATES = [
    ("sf4_ferredoxin_1fdn", "SF4", None, None, 0, 0),
    ("sf4_ferredoxin_2fdn", "SF4", None, None, 0, 0),
    ("clf_nitrogenase_7adr", "CLF", None, None, 0, 0),
    ("fes_ferredoxin_6e6r", "FES", None, None, 0, 0),
    ("feo_myohemerythrin_2mhr", "FEO", None, None, 0, 0),
    ("cua_ba3_2cua", "CUA", None, None, 0, 0),
    ("mgf_phosphoglucomutase_6h8z", "MGF", ("F1", "F2", "F3"), None, 0, 0),
    ("mgf_phosphoglucomutase_6h8z", "MGF", ("MG",), None, 0, -1),
    ("nco_zdna_1dn8", "NCO", ("N1", "N2", "N3", "N4", "N5", "N6"), None, 3, 0),
    ("nco_zdna_1dn8", "NCO", ("CO",), None, 0, 3),
    ("heme_myoglobin_5yce", "HEM", ("NA", "NB", "NC", "ND", "FE"), None, 0, 0),
    ("chloride_complex_4hbt", "CL", ("CL",), None, 0, -1),
    # An acylated lysine is a neutral amide.
    ("lys_biotin_1bdo", "LYS", ("NZ",), True, 1, 0),
    ("lys_biotin_1bdo", "LYS", ("NZ",), False, 3, 1),
    # An attachment takes the place of one hydrogen.
    ("nglycan_tree_1ax2", "ASN", ("ND2",), True, 1, 0),
    ("oglycan_sia_1g1s", "THR", ("OG1",), True, 0, 0),
    ("terminal_asj_glycans_1iau", "ASJ", ("C",), True, 1, 0),
    ("hydrolase_intermediate_1tqh", "SER", ("OG",), True, 0, 0),
    # 1TQH's tetrahedral intermediate, read as an oxyanion, is prepared as an alcohol, as TMol does.
    ("hydrolase_intermediate_1tqh", "4PA", ("OAD",), None, 1, 0),
    ("palmitoyl_ester_thioester_8trb", "SER", ("OG",), True, 0, 0),
    ("palmitoyl_ester_thioester_8trb", "CYS", ("SG",), True, 0, 0),
    ("palmitoyl_ester_thioester_8trb", "PLM", ("C1", "O2"), None, 0, 0),
    ("sulfur_attachments_3t14", "H2S", ("S",), True, 1, 0),
    ("sulfur_attachments_3t14", "S2H", ("S1",), True, 0, 0),
    ("sulfur_attachments_3t14", "S2H", ("S2",), False, 1, 0),
    ("sulfur_attachments_3t14", "CYS", ("SG",), True, 0, 0),
    ("collagen_hyp_1bkv", "HYP", ("N",), True, 0, 0),
    # 1J8Z states its ACE C at +1, which its CH3, =O and peptide N leave no room for.
    ("acetylated_peptide_1j8z", "ACE", ("C",), None, 0, 0, {"add_missing_atoms": False}),
    ("phosphopeptide_5ema", "SEP", ("O1P", "O2P", "O3P"), None, 0, None),
    # 8CH1's VDF OP3 esterifies FRU in altloc B (altloc A holds LAO on the other FRU).
    ("phosphate_attachment_8ch1", "VDF", ("OP3",), None, 0, None, {"altloc": "B"}),
    ("phosphate_charge_4js1", "PO4", ("O2",), None, 1, 0, _KEEP_AIDS),
    ("phosphate_charge_4js1", "PO4", ("O1", "O3", "O4"), None, 0, None, _KEEP_AIDS),
    # 183D's 8OG flags OP2 and OP3 as leaving; the O3' bond takes OP3, and OP2 is the phosphodiester's anion.
    ("na_dna_8og_183d", "8OG", ("P",), None, 0, None),
    ("na_dna_8og_183d", "8OG", ("OP2",), None, 0, -1),
    # 5XAG's IMD keeps N1's H whether or not N3 is bonded to a GOL.
    ("free_and_attached_solutes_5xag", "IMD", ("N1",), None, 1, None, _KEEP_AIDS),
    ("free_and_attached_solutes_5xag", "IMD", ("N3",), None, 0, None, _KEEP_AIDS),
]
# (fixture, residue, net formal charge, hydrogens or None[, parse overrides]) of every resolved copy
RESIDUE_STATES = [
    ("heme_myoglobin_5yce", "HEM", -2, 30),
    ("phosphate_charge_4js1", "PO4", -2, 1, _KEEP_AIDS),
    ("phosphopeptide_5ema", "SEP", -2, 4),
    ("free_and_attached_solutes_5xag", "GOL", 0, None, _KEEP_AIDS),
]
# S-H 1.34 A, P-H 1.42 A (5ME4 hypophosphite), other X-H 0.97-1.09 A
BOND_LENGTHS = {"S": (1.2, 1.5), "P": (1.3, 1.5)}


def _holds_protons(atom_array):
    elements = np.char.upper(atom_array.element.astype(str))
    return np.isin(elements, sorted(m for m in METAL_ELEMENTS if AQUA_PKA.get(m, DEFAULT_AQUA_PKA) < WATER_PKA))


def _state(atom_array, index):
    """Hydrogen count, formal charge, covalent bond to another residue and bond to a metal of one atom."""
    neighbours, _ = atom_array.bonds.get_bonds(int(index))
    heavy = neighbours[~is_h(atom_array)[neighbours]]
    elsewhere = np.array([residue_key(atom_array, n) != residue_key(atom_array, index) for n in heavy], dtype=bool)
    on_metal = _holds_protons(atom_array)[heavy]
    n_h = len(neighbours) - len(heavy)
    return n_h, int(atom_array.charge[index]), bool((elsewhere & ~on_metal).any()), bool(on_metal.any())


def _residues(atom_array, res_name):
    """Heavy-atom indices of each resolved residue called *res_name*."""
    mask = (atom_array.res_name == res_name) & ~is_h(atom_array) & np.isfinite(atom_array.coord).all(-1)
    keys = sorted({residue_key(atom_array, i)[:2] for i in np.flatnonzero(mask)})
    return [np.flatnonzero(mask & (atom_array.chain_id == c) & (atom_array.res_id == r)) for c, r in keys]


def _label(atom_array, i):
    return f"{atom_array.chain_id[i]}:{atom_array.res_name[i]}{atom_array.res_id[i]}.{atom_array.atom_name[i]}"


def _padded(rows, width):
    return [(*row, {}) if len(row) < width else row for row in rows]


@pytest.mark.parametrize("name", FIXTURES)
def test_every_hydrogen_is_placed_once_at_a_bond_length_on_a_sane_charge(name):
    protonated = protonate(name, **OVERRIDES.get(name, {}))
    h = is_h(protonated)
    bonds = protonated.bonds.as_array()
    degree = np.bincount(bonds[:, :2].ravel(), minlength=protonated.array_length())
    metal = np.isin(np.char.upper(protonated.element.astype(str)), sorted(METAL_ELEMENTS))
    to_h = bonds[h[bonds[:, 0]] != h[bonds[:, 1]], :2]
    heavy = np.where(h[to_h[:, 0]], to_h[:, 1], to_h[:, 0])
    lengths = np.linalg.norm(protonated.coord[to_h[:, 0]] - protonated.coord[to_h[:, 1]], axis=-1)
    low, high = np.array([BOND_LENGTHS.get(str(e).upper(), (0.9, 1.15)) for e in protonated.element[heavy]]).T

    assert h.any()
    assert np.isfinite(protonated.coord[h]).all()
    assert (degree[h] == 1).all()
    assert ((lengths >= low) & (lengths <= high)).all(), [_label(protonated, i) for i in heavy[lengths > high]]
    assert (np.abs(protonated.charge[~metal]) <= 1).all()


@pytest.mark.parametrize(
    "name", ["zn_tetrahedral_3ks3", "phosphopeptide_5ema", "nglycan_tree_1ax2", "heme_myoglobin_5yce"]
)
def test_the_state_counts_the_hydrogens_add_hydrogens_places(name):
    heavy = parse_fixture(name)
    assigned = assign_hydrogens(heavy)
    protonated = place_hydrogens(assigned)
    h = is_h(protonated)
    to_h = protonated.bonds.as_array()[:, :2]
    to_h = to_h[h[to_h[:, 0]] != h[to_h[:, 1]]]
    placed = np.bincount(np.where(h[to_h[:, 0]], to_h[:, 1], to_h[:, 0]), minlength=len(h))

    assert not is_h(assigned).any()
    np.testing.assert_array_equal(assigned.atom_name, heavy.atom_name)
    np.testing.assert_array_equal(assigned.coord, heavy.coord)
    np.testing.assert_array_equal(assigned.charge, protonated.charge[~h])
    np.testing.assert_array_equal(assigned.nhyd, placed[~h] + protonated.nhyd[~h])


@pytest.mark.parametrize("name", sorted(METAL_DONORS))
def test_each_metal_holds_the_donors_tmol_lists_and_takes_their_protons(name):
    """A donor on the metal has no H; a cysteine there is a thiolate, and a histidine protonates its other N."""
    protonated = protonate(name)
    holds = _holds_protons(protonated)
    donors = [
        donor
        for left, right, _ in protonated.bonds.as_array()
        for donor, metal in ((left, right), (right, left))
        if holds[metal]
        and not holds[donor]
        and str(protonated.atom_name[donor]) in PROTON_DONORS.get(str(protonated.res_name[donor]), ())
    ]
    candidates = np.isin(protonated.res_name, ["HIS", "CYS"]) & np.isin(protonated.atom_name, ["ND1", "NE2", "SG"])

    assert {_label(protonated, i) for i in np.flatnonzero(candidates) if _state(protonated, i)[3]} == METAL_DONORS[name]
    assert donors
    assert all(len(hydrogens_on(protonated, donor)) == 0 for donor in donors)


def test_a_cysteine_bridging_two_irons_of_the_p_cluster_binds_both():
    protonated = protonate("clf_nitrogenase_7adr")
    for chain, res_id in (("A", 75), ("B", 56)):
        (sg,) = np.flatnonzero(
            (protonated.chain_id == chain) & (protonated.res_id == res_id) & (protonated.atom_name == "SG")
        )
        neighbours, _ = protonated.bonds.get_bonds(int(sg))
        assert np.count_nonzero(protonated.element[neighbours] == "FE") == 2


@pytest.mark.parametrize("name", FIXTURES)
def test_standard_side_chains_take_their_state_at_ph_7_4(name):
    """Free lysines and arginines are cations, free carboxylates anions, free thiols, phenols and imidazoles neutral.

    A donor on a metal takes the form that donates: a thiolate, a carboxylate, a
    phenolate, or a neutral imidazole with its proton on the free nitrogen. A
    histidine is an imidazolate only where it bridges two metals (1SPD HIS 63).
    """
    protonated = protonate(name, **OVERRIDES.get(name, {}))
    wrong = []
    for res in _residues(protonated, "LYS"):
        for i in res[protonated.atom_name[res] == "NZ"]:
            n_h, charge, linked, on_metal = _state(protonated, i)
            if not linked and not on_metal and (n_h, charge) != (3, 1):
                wrong.append(i)
    for res_name, atoms, expected in (
        ("ARG", ("NE", "NH1", "NH2"), (5, 1)),
        ("ASP", ("OD1", "OD2"), (0, -1)),
        ("GLU", ("OE1", "OE2"), (0, -1)),
    ):
        for res in _residues(protonated, res_name):
            states = [_state(protonated, i) for i in res[np.isin(protonated.atom_name[res], atoms)]]
            total = (sum(s[0] for s in states), sum(s[1] for s in states))
            if len(states) == len(atoms) and not any(s[2] for s in states) and total != expected:
                wrong.append(res[0])
    for res_name, atom in (("CYS", "SG"), ("TYR", "OH")):
        for res in _residues(protonated, res_name):
            for i in res[protonated.atom_name[res] == atom]:
                n_h, charge, linked, on_metal = _state(protonated, i)
                if (n_h, charge) != ((0, -1) if on_metal else (0, 0) if linked else (1, 0)):
                    wrong.append(i)
    for res in _residues(protonated, "HIS"):
        ring = res[np.isin(protonated.atom_name[res], ("ND1", "NE2"))]
        if len(ring) != 2:
            continue
        states = [_state(protonated, i) for i in ring]
        n_h, charge = sum(s[0] for s in states), sum(s[1] for s in states)
        bound = sum(s[2] or s[3] for s in states)
        on_one_metal = any(s[3] for s in states) and bound == 1
        bridging = all(s[3] for s in states) and (n_h, charge) == (0, -1)
        if not bridging and (
            n_h + bound == 0 or charge < 0 or ((on_one_metal or not bound) and (n_h, charge) != (1, 0))
        ):
            wrong.append(ring[0])

    assert not [_label(protonated, i) for i in wrong]


@pytest.mark.parametrize(
    ("name", "res_name", "atoms", "linked", "n_h", "charge", "overrides"),
    _padded(ATOM_STATES, 7),
    ids=[
        f"{r[0]}-{r[1]}-{'+'.join(r[2] or ('all',))}{ {None: '', True: '-linked', False: '-free'}[r[3]]}"
        for r in ATOM_STATES
    ],
)
def test_an_atom_takes_the_state_its_chemistry_gives_it(name, res_name, atoms, linked, n_h, charge, overrides):
    protonated = protonate(name, **overrides)
    selected = [
        (str(protonated.atom_name[i]), _state(protonated, i)[:2])
        for res in _residues(protonated, res_name)
        for i in res
        if (atoms is None or str(protonated.atom_name[i]) in atoms) and linked in (None, _state(protonated, i)[2])
    ]

    assert selected
    assert all(state[0] == n_h and charge in (None, state[1]) for _, state in selected), selected


@pytest.mark.parametrize(
    ("name", "res_name", "charge", "n_h", "overrides"),
    _padded(RESIDUE_STATES, 5),
    ids=[f"{r[0]}-{r[1]}" for r in RESIDUE_STATES],
)
def test_a_residue_carries_its_net_charge_at_ph_7_4(name, res_name, charge, n_h, overrides):
    protonated = protonate(name, **overrides)
    residues = _residues(protonated, res_name)

    assert residues
    for res in residues:
        assert int(protonated.charge[res].sum()) == charge
        assert n_h in (None, sum(len(hydrogens_on(protonated, i)) for i in res))


def test_the_parse_reads_a_tetrahedral_intermediate_as_a_serine_bonded_oxyanion():
    heavy = parse_fixture("hydrolase_intermediate_1tqh")
    (oad,) = atoms_named(heavy, "4PA", "OAD")
    linked = [p for i in np.flatnonzero(heavy.res_name == "4PA") for p in partners_outside(heavy, i)]

    assert heavy.charge[oad] == -1
    assert any(heavy.res_name[p] == "SER" and heavy.atom_name[p] == "OG" for p in linked)


def test_a_phosphate_bridging_to_a_sugar_leaves_the_sugar_saturated():
    protonated = protonate("phosphate_attachment_8ch1")
    degree = np.bincount(protonated.bonds.as_array()[:, :2].ravel(), minlength=protonated.array_length())
    carbons = np.flatnonzero(np.isin(protonated.res_name, ["GLC", "FRU"]) & (protonated.element == "C"))

    assert len(carbons)
    assert (protonated.charge[carbons] == 0).all()
    assert (degree[carbons] == 4).all()


@pytest.mark.parametrize("name", ["terminal_and_linked_glycans_1en2", "terminal_and_linked_glycans_4ndz"])
def test_a_glycosidic_bond_displaces_o1_and_a_reducing_end_keeps_it(name):
    protonated = protonate(name)
    sugars = {"BGC", "BMA", "FRU", "FUC", "GAL", "GLC", "MAN", "NAG", "NGA"} & set(protonated.res_name)
    residues = [res for sugar in sorted(sugars) for res in _residues(protonated, sugar)]

    assert residues
    for res in residues:
        chain, res_id, _ = residue_key(protonated, res[0])
        names = protonated.atom_name[(protonated.chain_id == chain) & (protonated.res_id == res_id)]
        c1 = res[protonated.atom_name[res] == "C1"]
        assert protonated.charge[res].sum() == 0
        if len(c1):
            assert ("O1" in names) != bool(partners_outside(protonated, c1[0]))


def test_an_n_terminal_proline_missing_ring_atoms_still_takes_its_hydrogen():
    protonated = protonate("missing_proline_ring_7no8")
    terminal = [i for i in atoms_named(protonated, "PRO", "N") if not partners_outside(protonated, i)]

    assert terminal
    for n in terminal:
        assert len(hydrogens_on(protonated, n))


def test_a_nucleotide_listing_two_leaving_oxygens_keeps_the_one_it_resolves():
    assert not len(atoms_named(protonate("na_dna_8og_183d"), "8OG", "OP3"))


def test_what_protonation_cannot_read_is_left_or_rejected():
    heavy = parse_fixture("unresolved_unl")
    protonated = protonate("unresolved_unl")

    assert (heavy.res_name == "UNL").any()
    assert not np.isfinite(heavy.coord).all(axis=-1).any()
    np.testing.assert_array_equal(protonated.atom_name, heavy.atom_name)
    np.testing.assert_array_equal(protonated.charge, heavy.charge)
    with pytest.raises(ValueError, match="XYZ"):
        parse_fixture("unknown_heavy_atom_1a8o")


@pytest.mark.parametrize(
    ("name", "res_ids"), [("nmr_his_cation_1r21", [27, 56, 62, 69, 92]), ("beta_peptide_3c3g", None)]
)
def test_authored_hydrogens_on_both_ring_nitrogens_make_a_histidine_cation(name, res_ids):
    drawn = parse_fixture(name, hydrogen_policy="keep")
    authored = {residue_key(drawn, i) for i in atoms_named(drawn, "HIS", "HD1")}
    authored &= {residue_key(drawn, i) for i in atoms_named(drawn, "HIS", "HE2")}
    protonated = protonate(name, hydrogen_policy="keep")

    assert authored
    assert res_ids is None or sorted(key[1] for key in authored) == res_ids
    for res in _residues(protonated, "HIS"):
        if residue_key(protonated, res[0]) in authored:
            ring = [_state(protonated, i)[:2] for i in res[np.isin(protonated.atom_name[res], ("ND1", "NE2"))]]
            assert [n_h for n_h, _ in ring] == [1, 1]
            assert sum(charge for _, charge in ring) == 1


def test_an_authored_hydrogen_bonded_to_nothing_is_rebuilt_on_its_atom():
    authored = protonate("orphan_hydrogen_9ewf", hydrogen_policy="keep")
    degree = np.bincount(authored.bonds.as_array()[:, :2].ravel(), minlength=authored.array_length())

    assert (degree[is_h(authored)] == 1).all()
    assert np.count_nonzero(is_h(authored)) == np.count_nonzero(is_h(protonate("orphan_hydrogen_9ewf")))


def test_a_water_on_a_metal_is_trigonal_and_one_bridging_two_zinc_is_a_hydroxide():
    """1HZY's phosphotriesterase site: the water bridging both Zn is the hydroxide, its H on the bisector, and
    the water on one Zn lies in a plane with it, the metal on the bisector of H-O-H (M-O-H 125.3 deg)."""
    protonated = protonate("zn_hydroxide_bridge_1hzy", remove_waters=False)
    metal = np.isin(protonated.element, list(METAL_ELEMENTS))
    waters = {}
    for o in np.flatnonzero((protonated.res_name == "HOH") & (protonated.element == "O")):
        neighbours, _ = protonated.bonds.get_bonds(int(o))
        if metal[neighbours].any():
            waters[o] = neighbours[metal[neighbours]]
    angles = {
        o: [
            np.degrees(struc.angle(protonated.coord[h], protonated.coord[o], protonated.coord[m]))
            for h in hydrogens_on(protonated, o)
            for m in metals
        ]
        for o, metals in waters.items()
    }

    assert sorted(
        (len(metals), protonated.charge[o], len(hydrogens_on(protonated, o))) for o, metals in waters.items()
    ) == [
        (1, 0, 2),
        (2, -1, 1),
    ]
    for o, metals in waters.items():
        h = hydrogens_on(protonated, o)
        if len(metals) == 1:
            assert angles[o] == pytest.approx([125.3, 125.3], abs=0.1)
            assert np.degrees(struc.angle(*protonated.coord[[h[0], o, h[1]]])) == pytest.approx(109.5, abs=0.1)
            dihedral = struc.dihedral(*protonated.coord[[h[0], o, metals[0], h[1]]])
            assert abs(np.degrees(dihedral)) == pytest.approx(180.0, abs=0.1)
        else:
            assert min(angles[o]) > 115.0
