"""End-to-end protonation of trimmed PDB entries, each test or row named by the entry whose rule it guards.

Each parses an entry as documented, runs ``assign_hydrogens`` then ``add_hydrogens``, and checks the named atoms.
"""

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pytest

from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens
from atomworks.experimental.protonation._assign import template_hydrogen_names
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from tests.experimental.protonation.test_assign import DATA, ROWS, parse_entry


def protonate_array(atom_array, **kwargs):
    return place_hydrogens(assign_hydrogens(atom_array, **kwargs))


def protonate(name, **overrides):
    return protonate_array(parse_entry(name, **overrides))


def is_h(atom_array):
    return np.isin(atom_array.element, ("H", "D"))


def hydrogens_on(atom_array, index):
    neighbours, _ = atom_array.bonds.get_bonds(int(index))
    return neighbours[is_h(atom_array)[neighbours]]


def residue_key(atom_array, index):
    return str(atom_array.chain_id[index]), int(atom_array.res_id[index]), str(atom_array.res_name[index])


def hydrogen_positions(atom_array):
    """Each hydrogen's position keyed by chain, residue number and name."""
    return {
        residue_key(atom_array, i)[:2] + (str(atom_array.atom_name[i]),): atom_array.coord[i]
        for i in np.flatnonzero(is_h(atom_array))
    }


@pytest.mark.parametrize(("name", "atoms_too"), [("6dmz_mod_l", False), ("missing_ligand_carbon_5hs6", True)])
def test_hydrogens_do_not_depend_on_atom_order(name, atoms_too):
    """Bug 3: an amide H was built against whichever bonded residue came earlier in the array, and
    5HS6's J3Z HO3 against whichever of two carbons did."""
    heavy = parse_entry(name)
    starts = struc.get_residue_starts(heavy, add_exclusive_stop=True)
    if atoms_too:
        reverse = np.arange(len(heavy))[::-1]
    else:
        reverse = np.concatenate([np.arange(a, b) for a, b in zip(starts[-2::-1], starts[:0:-1], strict=True)])
    reordered = heavy[reverse]
    reordered.bonds = struc.BondList(len(reordered), reordered.bonds.as_array()[::-1])
    forward, backward = hydrogen_positions(protonate_array(heavy)), hydrogen_positions(protonate_array(reordered))

    assert forward.keys() == backward.keys()
    assert max(float(np.linalg.norm(forward[k] - backward[k])) for k in forward) < 1e-4


@pytest.mark.parametrize("name", ["6dmz_mod_l", "6dmz_mod_d"])
def test_an_amide_hydrogen_lies_in_the_peptide_plane_whatever_the_handedness(name):
    """Bug 7: a D residue after GLY took the tetrahedral position, 54.7 degrees out of the plane."""
    protonated = protonate(name)
    angles = []
    for n in np.flatnonzero(protonated.atom_name == "N"):
        neighbours, _ = protonated.bonds.get_bonds(int(n))
        heavy = {str(protonated.atom_name[j]): j for j in neighbours if not is_h(protonated)[j]}
        hydrogens = hydrogens_on(protonated, n)
        if set(heavy) != {"C", "CA"} or len(hydrogens) != 1:
            continue
        normal = np.cross(*(protonated.coord[heavy[a]] - protonated.coord[n] for a in ("CA", "C")))
        bond = protonated.coord[hydrogens[0]] - protonated.coord[n]
        angles.append(np.degrees(np.arcsin(abs(normal @ bond) / np.linalg.norm(normal) / np.linalg.norm(bond))))

    assert len(angles) > 40
    assert max(angles) < 0.5


def test_a_hydroxyl_hydrogen_is_not_placed_on_another_residue_s_hydrogen():
    """Bug 8: 4NDZ's GLC HO4, 2.4 A from ARG B66 NH2, took its template torsion 0.84 A from HH21."""
    protonated = protonate("terminal_and_linked_glycans_4ndz")
    o4 = np.flatnonzero((protonated.res_name == "GLC") & (protonated.atom_name == "O4"))
    nh2 = np.flatnonzero((protonated.res_name == "ARG") & (protonated.atom_name == "NH2"))
    distances = np.linalg.norm(protonated.coord[o4, None] - protonated.coord[None, nh2], axis=-1)
    contacts = np.argwhere(distances < 2.6)

    assert len(contacts)
    for g, a in contacts:
        (ho4,) = hydrogens_on(protonated, o4[g])
        arg = (protonated.chain_id == protonated.chain_id[nh2[a]]) & (protonated.res_id == protonated.res_id[nh2[a]])
        assert np.linalg.norm(protonated.coord[arg & is_h(protonated)] - protonated.coord[ho4], axis=-1).min() > 1.5


def test_a_three_prime_terminal_nucleotide_names_its_prochiral_hydrogens_as_the_internal_ones():
    """Bug 9: 1D17's 3'-terminal DG swapped H2'/H2'' and H21/H22 against every internal residue."""
    protonated = protonate("na_dna_5mc_1d17")
    senses, amines = {}, {}
    for res_id in np.unique(protonated.res_id):
        in_residue = protonated.res_id == res_id
        xyz = dict(zip(protonated.atom_name[in_residue], protonated.coord[in_residue], strict=True))
        code = f"{protonated.res_name[in_residue][0]}{res_id}"
        if "H2''" in xyz and code[:2] in ("DG", "DA", "DT", "DC"):
            senses[code] = np.linalg.det(np.stack([xyz[a] - xyz["C2'"] for a in ("C1'", "C3'", "H2'")])) > 0
        if "H21" in xyz:
            axis = (xyz["C2"] - xyz["N2"]) / np.linalg.norm(xyz["C2"] - xyz["N2"])

            def across(v, axis=axis):
                return v - (v @ axis) * axis

            amines[code] = across(xyz["H21"] - xyz["N2"]) @ across(xyz["N1"] - xyz["C2"]) > 0

    assert {"DG2", "DG6"} <= set(senses) & set(amines)
    assert all(senses.values()) and all(amines.values()), (senses, amines)


@pytest.mark.parametrize(("name", "res_name", "atom_name", "expected", "overrides"), ROWS)
def test_a_sweep_entry_takes_the_state_its_chemistry_gives_and_keeps_it(name, res_name, atom_name, expected, overrides):
    """Entries the whole-PDB and TMol sweeps failed on: the atom takes *expected* as ``(charge, hydrogens)``, and
    protonating the output again keeps it. The output is what TMol reads: each added hydrogen bonded once, to
    its heavy atom; the input atoms' atom_id carried through; a hydrogen without a position kept as an atom."""
    heavy = parse_entry(name, **overrides)
    heavy.set_annotation("atom_id", np.arange(len(heavy)))
    assigned = assign_hydrogens(heavy)
    protonated = place_hydrogens(assigned)

    h = is_h(protonated)
    bonds = protonated.bonds.as_array()[:, :2]
    assert (np.bincount(bonds.ravel(), minlength=len(protonated))[h] == 1).all()
    assert not h[bonds].all(axis=1).any()
    np.testing.assert_array_equal(np.sort(protonated.atom_id[~h]), np.sort(assigned.atom_id[~is_h(assigned)]))
    built = assigned.nhyd[~assigned.skip_hydrogen_placement & ~is_h(assigned)].sum()
    assert h.sum() == is_h(assigned).sum() + built
    assert {"charge", "tautomer_free"} <= set(assigned.get_annotation_categories())

    for structure in (protonated, protonate_array(protonated)):
        placed = np.isfinite(structure.coord).all(axis=-1)
        atoms = np.flatnonzero((structure.res_name == res_name) & (structure.atom_name == atom_name) & placed)
        assert {(int(structure.charge[i]), len(hydrogens_on(structure, i))) for i in atoms} == {expected}


def test_a_nucleotide_missing_a_sugar_carbon_keeps_the_dictionary_s_chirality():
    """7N5V DT H1': with C2' unresolved, C1' keeps the hand the dictionary gives it."""

    def hand(xyz):
        return np.sign(np.linalg.det(np.stack([xyz[a] - xyz["C1'"] for a in ("O4'", "N1", "H1'")])))

    protonated = protonate("missing_sugar_carbon_7n5v")
    template = info.residue("DT")
    expected = hand(dict(zip(template.atom_name, template.coord, strict=True)))
    hands = []
    for atoms in struc.residue_iter(protonated):
        if atoms.res_name[0] == "DT" and {"C1'", "O4'", "N1", "H1'"} <= set(atoms.atom_name):
            hands.append(hand(dict(zip(atoms.atom_name, atoms.coord, strict=True))))

    assert len(hands) > 1
    assert set(hands) == {expected}


def test_a_generated_hydrogen_name_avoids_the_names_the_dictionary_gives_other_atoms():
    """6Q9T QUK: a hydrogen the dictionary does not name takes no name it gives another atom's hydrogen."""
    protonated = protonate("modified_components_6q9t")
    declared = template_hydrogen_names("QUK")
    quk = np.flatnonzero((protonated.res_name == "QUK") & is_h(protonated))

    assert len(quk)
    for h in quk:
        (parent,) = [j for j in protonated.bonds.get_bonds(int(h))[0] if not is_h(protonated)[j]]
        others = {n for p, names in declared.items() if p != protonated.atom_name[parent] for n in names}
        assert protonated.atom_name[h] not in others, protonated.atom_name[h]


def test_assembly_copies_of_a_chain_each_keep_their_n_terminus():
    """1AWD's assembly repeats chain A: the second copy's TYR 1 does not follow the first copy's last residue."""
    config = ParseConfig(hydrogen_policy="remove", add_missing_atoms=True, build_assembly="first")
    assembly = next(iter(parse(DATA / "assembly_copies_1awd.cif.zst", config=config)["assemblies"].values()))[0]
    protonated = protonate_array(assembly)

    n = np.flatnonzero((protonated.res_name == "TYR") & (protonated.res_id == 1) & (protonated.atom_name == "N"))
    assert len(n) == 2
    assert {(int(protonated.charge[i]), len(hydrogens_on(protonated, i))) for i in n} == {(1, 3)}
    aromatic = assembly.copy()
    bonds = aromatic.bonds.as_array()
    bonds[bonds[:, 2] == struc.BondType.DOUBLE, 2] = struc.BondType.AROMATIC
    aromatic.bonds = struc.BondList(len(aromatic), bonds)
    with pytest.raises(ValueError, match="unstated order"):
        assign_hydrogens(aromatic)


def test_a_histidine_bridging_copper_and_zinc_is_the_imidazolate():
    """1SPD HIS 63 bridges superoxide dismutase's Cu and Zn through both ring nitrogens."""
    protonated = protonate("bridging_imidazolate_1spd", remove_ccds=())

    ring = np.flatnonzero((protonated.res_name == "HIS") & np.isin(protonated.atom_name, ["ND1", "NE2"]))
    assert (-1, 0) in {(int(protonated.charge[i]), len(hydrogens_on(protonated, i))) for i in ring}


@pytest.mark.parametrize(("element", "length"), [("NA", 2.4), ("K", 2.8)])
def test_a_histidine_bridging_zinc_and_an_alkali_ion_keeps_its_proton_off_the_zinc(element, length):
    """3KS3 HIS 94 NE2 is on Zn; an alkali ion on ND1 lowers no water's pKa, so ND1 keeps the proton."""
    heavy = parse_entry("zn_tetrahedral_3ks3")
    his = (heavy.chain_id == "A") & (heavy.res_id == 94) & (heavy.res_name == "HIS")
    (nd1,) = np.flatnonzero(his & (heavy.atom_name == "ND1"))
    ring = his & np.isin(heavy.atom_name, ["CG", "ND1", "CE1", "NE2", "CD2"])
    outward = heavy.coord[nd1] - heavy.coord[ring].mean(axis=0)
    ion = heavy[[nd1]]
    ion.res_name[:], ion.atom_name[:], ion.element[:] = element, element, element
    ion.chain_id[:], ion.res_id[:], ion.charge[:], ion.hetero[:] = "Z", 1, 1, True
    for name in ("pn_unit_iid", "chain_iid", "pn_unit_id"):
        if name in ion.get_annotation_categories():
            ion.get_annotation(name)[:] = "Z_1"
    ion.coord += outward / np.linalg.norm(outward) * length
    site = heavy + ion
    site.bonds.add_bond(int(nd1), len(heavy), struc.BondType.COORDINATION)

    protonated = protonate_array(site)

    names = protonated.atom_name
    ring_n = [
        np.flatnonzero((protonated.res_id == 94) & (protonated.chain_id == "A") & (names == n))[0]
        for n in ("ND1", "NE2")
    ]
    assert [len(hydrogens_on(protonated, i)) for i in ring_n] == [1, 0]


def test_counts_declared_from_a_polar_hydrogen_input_outrank_the_ph():
    """PDBbind 10GS (the pocket around its ligand) draws polar hydrogens only (backbone N-H, ARG HE, LYS HZ1-3).
    TMol declares the counts drawn in each residue that draws any, and assign_hydrogens keeps every one of them."""
    atoms = parse(
        DATA / "polar_hydrogens_10gs.pdb.zst", config=ParseConfig(hydrogen_policy="keep", add_missing_atoms=False)
    )["asym_unit"][0]
    h = is_h(atoms)
    bonds = atoms.bonds.as_array()[:, :2]
    drawn = np.bincount(bonds[h[bonds[:, ::-1]]], minlength=len(atoms))
    residue = struc.get_all_residue_positions(atoms)
    draws_h = (np.bincount(residue, weights=h) > 0)[residue]
    declared = np.where(draws_h & np.isin(atoms.element, ["N", "O"]), drawn, -1)

    protonated = place_hydrogens(assign_hydrogens(atoms, hydrogens=declared))

    heavy = np.flatnonzero(~h)
    kept = {
        (atoms.chain_id[i], atoms.res_id[i], atoms.atom_name[i]): int(declared[i]) for i in heavy if declared[i] >= 0
    }
    found = {
        (protonated.chain_id[i], protonated.res_id[i], protonated.atom_name[i]): len(hydrogens_on(protonated, i))
        for i in np.flatnonzero(~is_h(protonated))
    }
    assert any(count == 0 for count in kept.values())
    assert {key: found[key] for key in kept} == kept
    (nz,) = np.flatnonzero((protonated.res_name == "LYS") & (protonated.res_id == 44) & (protonated.atom_name == "NZ"))
    assert (len(hydrogens_on(protonated, nz)), int(protonated.charge[nz])) == (3, 1)


def test_unsupported_placement_geometry_preserves_the_assigned_state():
    from tests.experimental.protonation.test_assign import molecule_from_smiles

    atoms = molecule_from_smiles("[PH](C)(C)(C)C")
    assigned = assign_hydrogens(atoms, hydrogens=atoms.nhyd)
    expected_counts = assigned.nhyd.copy()
    with pytest.raises(ValueError, match="more hydrogens than free directions"):
        place_hydrogens(assigned)
    np.testing.assert_array_equal(assigned.nhyd, expected_counts)


@pytest.mark.parametrize("smiles", ["CC[NH3+]", "CC[NH2+]C", "CC[NH+](C)C", "CC"])
def test_metal_contacts_do_not_displace_a_saturated_atoms_hydrogens(smiles, caplog):
    from tests.experimental.protonation.test_assign import molecule_from_smiles

    atoms = molecule_from_smiles(f"{smiles}.[Mg+2]")
    atoms.res_id[-1], atoms.res_name[-1], atoms.atom_name[-1] = 2, "MG", "MG"
    atoms.pn_unit_iid[-1] = 1
    parent = 2 if "N" in smiles else 1
    atoms.bonds.add_bond(parent, len(atoms) - 1, struc.BondType.COORDINATION)
    assigned = assign_hydrogens(atoms, hydrogens=atoms.nhyd)
    placed = place_hydrogens(assigned)

    kept = placed.atom_id < len(assigned)
    np.testing.assert_array_equal(placed.coord[kept], assigned.coord)
    np.testing.assert_array_equal(placed.charge[kept], assigned.charge)
    np.testing.assert_array_equal(placed[kept].bonds.as_array(), assigned.bonds.as_array())
    (placed_parent,) = np.flatnonzero(placed.atom_id == parent)
    assert len(hydrogens_on(placed, placed_parent)) == int(assigned.nhyd[parent])
    assert np.isfinite(placed.coord).all()
    assert "Ignored metal contacts as hydrogen-placement constraints" in caplog.text


@pytest.mark.parametrize("keep_hydrogens", [False, True])
def test_3bpc_lysine_keeps_three_hydrogens_despite_its_recorded_magnesium_contact(keep_hydrogens, caplog):
    atoms = parse(
        DATA / "protonated_lysine_mg_3bpc.pdb",
        config=ParseConfig(hydrogen_policy="keep", add_missing_atoms=False),
    )["asym_unit"][0]
    atoms = atoms[~is_h(atoms) | (keep_hydrogens & (atoms.atom_name == "HZ1"))]
    (nz,) = np.flatnonzero(atoms.atom_name == "NZ")
    (mg,) = np.flatnonzero(atoms.element == "MG")
    neighbours, bond_types = atoms.bonds.get_bonds(nz)
    assert bond_types[neighbours == mg].tolist() == [struc.BondType.COORDINATION]
    declared = np.full(len(atoms), -1)
    declared[nz] = 3
    assigned = assign_hydrogens(atoms, hydrogens=declared)
    placed = place_hydrogens(assigned)

    kept = np.array([np.flatnonzero(placed.atom_id == atom_id)[0] for atom_id in assigned.atom_id])
    np.testing.assert_array_equal(placed.coord[kept], assigned.coord)
    np.testing.assert_array_equal(placed.charge[kept], assigned.charge)
    np.testing.assert_array_equal(placed[kept].bonds.as_array(), assigned.bonds.as_array())
    (nz,) = np.flatnonzero(placed.atom_name == "NZ")
    assert placed.charge[nz] == 1
    assert len(hydrogens_on(placed, nz)) == 3
    assert np.isfinite(placed.coord).all()
    assert "four covalent neighbours" in caplog.text


def test_metal_hydride_counts_remain_explicitly_unplaced(caplog):
    import logging

    atoms = parse_entry("nife_hydride_6fpi", remove_ccds=())
    with caplog.at_level(logging.WARNING):
        assigned = assign_hydrogens(atoms)
    hydrides = np.isin(assigned.element, ["FE", "NI"]) & (assigned.nhyd > 0)
    assert hydrides.sum() == 4
    assert assigned.skip_hydrogen_placement[hydrides].all()
    assert "hydride placement is unsupported" in caplog.text

    placed = place_hydrogens(assigned)
    np.testing.assert_array_equal(placed.nhyd[~is_h(placed)][hydrides], assigned.nhyd[hydrides])


def test_hydrogen_placement_preserves_pair_annotations_across_residue_reordering():
    from atomworks.io.utils.atom_array_plus import AnnotationList2D, as_atom_array_plus
    from tests.experimental.protonation.test_assign import molecule_from_smiles

    atoms = as_atom_array_plus(molecule_from_smiles("CC.O"))
    atoms.res_id[-1], atoms.res_name[-1], atoms.atom_name[-1] = 2, "HOH", "O"
    atoms.pn_unit_iid[-1] = 1
    atoms.set_annotation("atom_id", np.arange(len(atoms)))
    atoms.set_annotation("restraint", AnnotationList2D(len(atoms), [[0, 2]], np.array([1.7])), n_body=2)

    placed = place_hydrogens(assign_hydrogens(atoms))
    annotation = placed.get_annotation("restraint", n_body=2)
    np.testing.assert_array_equal(placed.atom_id[annotation.pairs], [[0, 2]])
    np.testing.assert_array_equal(annotation.values, [1.7])
    assert annotation.n_atoms == len(placed)
    np.testing.assert_array_equal(atoms.get_annotation("restraint", n_body=2).pairs, [[0, 2]])
