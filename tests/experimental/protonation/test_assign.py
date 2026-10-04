"""assign_hydrogens on trimmed PDB entries: each row is a rule, named by the entry that needs it."""

import logging
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from atomworks.experimental.protonation import assign_hydrogens
from atomworks.experimental.protonation.external.dimorphite_dl import dimorphite_dl
from atomworks.io import parse
from atomworks.io.config import ParseConfig

DATA = Path(__file__).parents[2] / "data" / "experimental" / "protonation"
# The documented parse: covalent, metal and disulfide bonds from struct_conn, missing atoms added.
PARSE = {
    "hydrogen_policy": "remove",
    "add_missing_atoms": True,
    "add_bond_types_from_struct_conn": ("covale", "metalc", "disulf"),
}


def parse_entry(name, **overrides):
    return parse(DATA / f"{name}.cif.zst", config=ParseConfig(**{**PARSE, **overrides}))["asym_unit"][0]


# (entry, residue, atom, (charge, hydrogens), parse overrides)
ROWS = [
    # metalc to element X, which has no covalent radius
    pytest.param("unknown_atom_on_nickel_1jqk", "UNX", "UNK", (0, 0), {}, id="1JQK UNX UNK"),
    pytest.param("sodium_histidine_1vmj", "HIS", "NE2", (0, 0), {}, id="1VMJ HIS NE2"),  # on a sodium
    # an amide after STA, drawn with its H when read back
    pytest.param("pepstatin_after_gap_4ejk", "ALA", "N", (0, 1), {}, id="4EJK ALA N"),
    # nitric oxide and peroxide on their metals, which the dictionary draws without hydrogens
    pytest.param("nitric_oxide_on_heme_8hbf", "NO", "N", (0, 0), {}, id="8HBF NO N"),
    pytest.param("peroxide_on_heme_copper_7ypy", "PER", "O1", (-1, 0), {}, id="7YPY PER O1"),
    # the carboxylate's -1 deposited on the O the dictionary draws double-bonded
    pytest.param("full_conjugate_9ewf", "SIA", "O1A", (0, 0), {}, id="9EWF SIA O1A"),
    # a proline missing its CD: N takes a hydrogen in its place
    pytest.param("missing_proline_ring_7no8", "PRO", "N", (0, 2), {"add_missing_atoms": False}, id="7NO8 PRO N"),
    # a primary carboxamide N on Fe(III) stays a neutral amide
    pytest.param("carboxamide_on_iron_7urh", "ASN", "ND2", (0, 2), {}, id="7URH ASN ND2"),
    # a chloride deposited at charge 0 is a chloride
    pytest.param("chloride_complex_4hbt", "CL", "CL", (-1, 0), {"add_missing_atoms": False}, id="4HBT CL CL"),
    # an unresolved OXT is missing density: the C terminus takes no hydrogen in its place
    pytest.param("c_terminus_without_oxt_5ark", "LYS", "C", (0, 0), {"add_missing_atoms": False}, id="5ARK LYS C"),
    # a nitro N deposited at charge 0 takes the dictionary's +1, so the peptide's free N terminus is titrated
    pytest.param("nitrophenylalanine_peptide_1ytj", "PPN", "N", (1, 3), {"add_missing_atoms": False}, id="1YTJ PPN N"),
    # a phosphoryl P is P(V): hypophosphite keeps its two P-H
    pytest.param("hypophosphite_5me4", "HP4", "P1", (0, 2), {"remove_ccds": ()}, id="5ME4 HP4 P1"),
    # only protons carry over from Dimorphite-DL's redrawn azide
    pytest.param("azide_ligand_3wd2", "A1L", "N7", (-1, 0), {"remove_ccds": ()}, id="3WD2 A1L N7"),
    # Dimorphite-DL draws perchlorate charge-separated; the Cl keeps the charge its bonds draw
    pytest.param("perchlorate_1jfv", "LCP", "CL", (0, 0), {"remove_ccds": ()}, id="1JFV LCP CL"),
    # perchlorate keeps the charges its bonds draw; its O takes no proton
    pytest.param("perchlorate_1jfv", "LCP", "O1", (0, 0), {"remove_ccds": ()}, id="1JFV LCP O1"),
    # free sulfate is a dianion
    pytest.param("sulfate_1al1", "SO4", "O3", (-1, 0), {"remove_ccds": ()}, id="1AL1 SO4 O3"),
    # bicarbonate keeps its proton (pKa2 10.3)
    pytest.param("bicarbonate_5zas", "BCT", "O3", (0, 1), {"remove_ccds": ()}, id="5ZAS BCT O3"),
    # a selenol (pKa 5.2) is a selenolate
    pytest.param("selenocysteine_5azz", "SEC", "SE", (-1, 0), {"remove_ccds": ()}, id="5AZZ SEC SE"),
    # cacodylic acid (pKa 6.3) is cacodylate
    pytest.param("cacodylate_8his", "CAD", "O1", (-1, 0), {"remove_ccds": ()}, id="8HIS CAD O1"),
    # arsenate is HAsO4 2-
    pytest.param("arsenate_3we3", "ART", "O2", (-1, 0), {"remove_ccds": ()}, id="3WE3 ART O2"),
    # an amine N-oxide O is no base
    pytest.param("amine_oxide_3t2a", "TMO", "OAE", (-1, 0), {"remove_ccds": ()}, id="3T2A TMO OAE"),
    # ammonia (pKa 9.25) is ammonium
    pytest.param("ammonium_5k8h", "NH4", "N", (1, 4), {"remove_ccds": ()}, id="5K8H NH4 N"),
    # free phosphate is HPO4 2-
    pytest.param("phosphate_5txd", "PO4", "O2", (0, 1), {"remove_ccds": ()}, id="5TXD PO4 O2"),
    # a tertiary amide N is no base
    pytest.param("tertiary_amide_4elb", "34S", "N17", (0, 0), {"remove_ccds": ()}, id="4ELB 34S N17"),
    # a tertiary sulfonamide N is no base
    pytest.param("sulfonamide_4e4l", "0NH", "N5", (0, 0), {"remove_ccds": ()}, id="4E4L 0NH N5"),
    # a phosphoramide N is no base
    pytest.param("phosphoramide_4p45", "2F9", "N", (0, 1), {"remove_ccds": ()}, id="4P45 2F9 N"),
    # a vinamidine takes its proton on the imine N
    pytest.param("vinamidine_5oxd", "B2W", "NAE", (1, 1), {"remove_ccds": ()}, id="5OXD B2W NAE"),
    # an enamine N is no base
    pytest.param("enamine_5cbb", "4ZF", "N08", (0, 1), {"remove_ccds": ()}, id="5CBB 4ZF N08"),
    # a neutral histidine nothing decides takes its proton on the tele N (NE2)
    pytest.param("6dmz_mod_l", "HIS", "NE2", (0, 1), {}, id="6DMZ HIS NE2"),
    pytest.param("6dmz_mod_l", "HIS", "ND1", (0, 0), {}, id="6DMZ HIS ND1"),
    # a nitrosamine N is no base
    pytest.param("nitrosamine_4eji", "0QA", "N2", (0, 0), {"remove_ccds": ()}, id="4EJI 0QA N2"),
    # an OG1 2.98 A from Hg is in contact, not coordinated, and stays an alcohol
    pytest.param("metal_contact_5cqo", "THR", "OG1", (0, 1), {"remove_ccds": ()}, id="5CQO THR OG1"),
    # a guanidinium (pKa 13.8) on Zn keeps its protons
    pytest.param("arginine_on_zinc_7pyk", "ARG", "NH2", (1, 2), {"remove_ccds": ()}, id="7PYK ARG NH2"),
    # a ligand without stated bonds keeps its charges and counts
    pytest.param("unbonded_ligand_2aj6", "UNL", "O1", (0, 0), {"remove_ccds": ()}, id="2AJ6 UNL O1"),
    # a sulfur the dictionary draws at a higher valence keeps its hydrogen
    pytest.param("dictionary_valence_2yak", "OSV", "S16", (0, 1), {"remove_ccds": ()}, id="2YAK OSV S16"),
]


@pytest.mark.parametrize(("name", "res_name", "atom_name", "expected", "overrides"), ROWS)
def test_an_entry_takes_the_state_its_chemistry_gives(name, res_name, atom_name, expected, overrides):
    assigned = assign_hydrogens(parse_entry(name, **overrides))

    placed = np.isfinite(assigned.coord).all(axis=-1)
    atoms = np.flatnonzero((assigned.res_name == res_name) & (assigned.atom_name == atom_name) & placed)
    assert len(atoms)
    assert {(int(assigned.charge[i]), int(assigned.nhyd[i])) for i in atoms} == {expected}


def test_an_internal_phosphodiester_stays_an_anion_below_a_monoester_s_pka():
    """1TTD DT OP2: titrated with its links capped, an internal phosphodiester (pKa ~1.5) stays an anion at pH 6.3;
    only the 5' monoester (pKa2 6.5) takes a proton."""
    assigned = assign_hydrogens(parse_entry("na_dna_ttd_1ttd"), ph=6.3)
    op2 = assigned.atom_name == "OP2"
    first = assigned.res_id[op2].min()

    assert set(
        zip(
            assigned.charge[op2 & (assigned.res_id > first)].tolist(),
            assigned.nhyd[op2 & (assigned.res_id > first)].tolist(),
            strict=False,
        )
    ) == {(-1, 0)}
    assert set(
        zip(
            assigned.charge[op2 & (assigned.res_id == first)].tolist(),
            assigned.nhyd[op2 & (assigned.res_id == first)].tolist(),
            strict=False,
        )
    ) == {(0, 1)}


def test_hydrogens_on_an_unrelated_chain_do_not_change_a_protein_s_counts():
    """3KS3's backbone N-H counts stay the same beside 3C3G's explicitly hydrogenated peptide."""
    protein = parse_entry("zn_tetrahedral_3ks3", add_missing_atoms=False)
    peptide = parse_entry("beta_peptide_3c3g", hydrogen_policy="keep", add_missing_atoms=False)
    peptide.set_annotation("nhyd", np.zeros(len(peptide), dtype=protein.nhyd.dtype))
    for annotation in ("chain_id", "chain_iid", "pn_unit_id", "pn_unit_iid"):
        if annotation in peptide.get_annotation_categories():
            peptide.set_annotation(annotation, np.full(len(peptide), "peptide"))

    alone = assign_hydrogens(protein)
    together = assign_hydrogens(protein + peptide)[: len(protein)]

    assert np.array_equal(alone.nhyd, together.nhyd)
    assert np.array_equal(alone.charge, together.charge)


class _OneProductReaction:
    """A reaction that fails the test when a ``RunReactants`` call builds more than one product."""

    def __init__(self, reaction):
        self._reaction = reaction

    def RunReactants(self, *args, **kwargs):  # noqa: N802
        products = self._reaction.RunReactants(*args, **kwargs)
        if len(products) > 1:
            pytest.fail(f"a neutralising reaction built {len(products)} products to keep one")
        return products


def test_an_atomized_nucleotide_chain_titrates_in_seconds(monkeypatch):
    """1QD7's chain of 271 unknown nucleotides is titrated as one atomized molecule. Each phosphate site
    resanitized all of it (50 s), and protonating the output again built every product of each neutralising
    reaction to keep the first (over 30 min)."""
    calls = []
    sanitize = Chem.SanitizeMol
    monkeypatch.setattr(Chem, "SanitizeMol", lambda *args, **kwargs: calls.append(1) or sanitize(*args, **kwargs))
    reactions = tuple((query, _OneProductReaction(r)) for query, r in dimorphite_dl._NEUTRALIZING_REACTIONS)
    monkeypatch.setattr(dimorphite_dl, "_NEUTRALIZING_REACTIONS", reactions)

    assigned = assign_hydrogens(parse_entry("atomized_nucleotide_chain_1qd7"))
    assert len(calls) < (assigned.element == "P").sum() == 271
    assign_hydrogens(assigned)


def test_only_placed_input_hydrogens_are_reported_dropped(caplog):
    """6FPI kept with its hydrogens: the two deposited EJ2 hydrides are dropped, not the unplaced template ones."""
    heavy = parse(DATA / "nife_hydride_6fpi.cif.zst", config=ParseConfig(hydrogen_policy="keep", remove_ccds=()))
    with caplog.at_level(logging.WARNING, logger="atomworks.ml"):
        assign_hydrogens(heavy["asym_unit"][0])

    assert "Dropped 2 placed input hydrogens" in caplog.text


def molecule_from_smiles(smiles):
    """A resolved non-polymer with explicit connectivity, independent of CCD entries."""
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_from_rdkit

    atoms = atom_array_from_rdkit(Chem.MolFromSmiles(smiles))
    atoms.coord = np.arange(len(atoms) * 3, dtype=np.float32).reshape(-1, 3) / 4
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.set_annotation("is_polymer", np.zeros(len(atoms), dtype=bool))
    atoms.set_annotation("chain_type", np.full(len(atoms), int(ChainType.NON_POLYMER)))
    atoms.set_annotation("pn_unit_iid", np.zeros(len(atoms), dtype=int))
    return atoms


@pytest.mark.parametrize("element", ["P", "As", "Sb"])
def test_declared_hydrogens_preserve_five_coordinate_chemistry(element):
    atoms = molecule_from_smiles(f"[{element}H](C)(C)(C)C")
    assigned = assign_hydrogens(atoms, hydrogens=atoms.nhyd)

    np.testing.assert_array_equal(assigned.nhyd, atoms.nhyd)
    np.testing.assert_array_equal(assigned.charge, atoms.charge)
    assert "skip_hydrogen_placement" in assigned.get_annotation_categories()
    assert not assigned.skip_hydrogen_placement.any()


@pytest.mark.parametrize("on_metal", [False, True])
def test_inconsistent_declared_valence_is_rejected_with_or_without_coordination(on_metal):
    import biotite.structure as struc

    atoms = molecule_from_smiles("CC=O.[Mg+2]" if on_metal else "CC=O")
    declared = atoms.nhyd.copy()
    declared[1] = 2  # The carbonyl carbon has room for one H, independent of coordination.
    if on_metal:
        atoms.res_id[-1], atoms.res_name[-1], atoms.atom_name[-1] = 2, "MG", "MG"
        atoms.pn_unit_iid[-1] = 1
        atoms.bonds.add_bond(1, len(atoms) - 1, struc.BondType.COORDINATION)

    with pytest.raises(ValueError, match="Declared hydrogen count exceeds chemical valence"):
        assign_hydrogens(atoms, hydrogens=declared)


@pytest.mark.parametrize(("code", "element"), [("F", "F"), ("CL", "Cl"), ("BR", "Br"), ("IOD", "I")])
def test_monatomic_halides_follow_ccd_chemistry_but_covalent_halogens_do_not(code, element, caplog):
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    template = atom_array_from_ccd_code(code)
    atoms = molecule_from_smiles(f"[{element}-]")
    atoms.res_name[:] = code
    atoms.atom_name[:] = template.atom_name
    atoms.charge[:] = 0  # Missing atom_site charges are read as zero.
    assigned = assign_hydrogens(atoms)
    assert assigned.charge.tolist() == [-1]
    assert assigned.nhyd.tolist() == [0]
    assert "Zero also represents missing charge" in caplog.text

    covalent = assign_hydrogens(molecule_from_smiles(f"C{element}"))
    assert covalent.charge[1] == 0
    assert covalent.nhyd[1] == 0


@pytest.mark.parametrize(("code", "element"), [("F", "F"), ("CL", "Cl"), ("BR", "Br"), ("IOD", "I")])
@pytest.mark.parametrize("hydrogens", [0, 1])
def test_declared_monatomic_states_override_the_ccd_charge(code, element, hydrogens):
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    atoms = molecule_from_smiles(f"[{element}]")
    atoms.res_name[:] = code
    atoms.atom_name[:] = atom_array_from_ccd_code(code).atom_name
    assigned = assign_hydrogens(atoms, hydrogens=np.array([hydrogens]))
    assert assigned.charge.tolist() == [0]
    assert assigned.nhyd.tolist() == [hydrogens]
    assert atoms.charge.tolist() == [0]


def test_custom_monatomic_component_defines_its_own_charge():
    from atomworks.io.utils.ccd import custom_ccd_residues

    # An explicit neutral chlorine atom is distinct from the CCD's chloride ion.
    atoms = molecule_from_smiles("[Cl]")
    atoms.res_name = np.full(len(atoms), "CUSTOM_CL")
    with custom_ccd_residues({"CUSTOM_CL": atoms}):
        assigned = assign_hydrogens(atoms)
    assert assigned.charge.tolist() == [0]
    assert assigned.nhyd.tolist() == [0]
