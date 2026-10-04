"""Focused regressions for authored-link product chemistry."""

import numpy as np
import pytest
from biotite.structure import BondType

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array import get_bond_degree_per_atom
from atomworks.io.utils.io_utils import read_any
from atomworks.io.utils.link_chemistry import _has_valid_rdkit_valence
from tests.io.conftest import get_pdb_path


def _parse(pdb_id, *, include_metals=False, hydrogen_policy="remove"):
    struct_conn_types = ("covale", "metalc") if include_metals else ("covale",)
    return parse(
        get_pdb_path(pdb_id),
        config=ParseConfig.from_preset(
            "rcsb",
            model=1,
            build_assembly=None,
            hydrogen_policy=hydrogen_policy,
            add_bond_types_from_struct_conn=struct_conn_types,
        ),
    )["asym_unit"][0]


def _atom(atoms, chain_id, res_id, res_name, atom_name):
    return np.flatnonzero(
        (atoms.chain_id == chain_id)
        & (atoms.res_id == res_id)
        & (atoms.res_name == res_name)
        & (atoms.atom_name == atom_name)
    ).item()


def _bond_type(atoms, atom1, atom2):
    neighbors, types = atoms.bonds.get_bonds(atom1)
    return types[neighbors == atom2].item()


def test_phosphoryl_addition_lowers_metal_coordinated_double_bond():
    """1CUL's fifth P substituent changes Mg-coordinated P=O to P-O(-)."""
    atoms = _parse("1cul", include_metals=True)
    phosphorus = _atom(atoms, "G", 1003, "3PO", "PA")
    oxygen = _atom(atoms, "G", 1003, "3PO", "O1A")

    assert _bond_type(atoms, phosphorus, oxygen) == BondType.SINGLE
    assert BondType.COORDINATION in atoms.bonds.get_bonds(oxygen)[1]
    assert atoms.charge[oxygen] == -1
    assert get_bond_degree_per_atom(atoms)[phosphorus] == 5


@pytest.mark.parametrize("hydrogen_policy", ["keep", "remove"])
def test_terminal_carboxylate_link_shifts_resonance(hydrogen_policy):
    """1PPV links GLU OE1 as a single bond and moves C=O to equivalent OE2."""
    atoms = _parse("1ppv", include_metals=True, hydrogen_policy=hydrogen_policy)
    carbon = _atom(atoms, "A", 116, "GLU", "CD")
    linked = _atom(atoms, "A", 116, "GLU", "OE1")
    sibling = _atom(atoms, "A", 116, "GLU", "OE2")

    assert _bond_type(atoms, carbon, linked) == BondType.SINGLE
    assert _bond_type(atoms, carbon, sibling) == BondType.DOUBLE
    assert atoms.charge[[linked, sibling]].tolist() == [0, 0]
    assert get_bond_degree_per_atom(atoms)[[linked, sibling]].tolist() == [2, 2]


@pytest.mark.parametrize(
    ("pdb_id", "donor_key", "center_key", "acceptor_key", "acceptor_bond", "acceptor_hydrogens"),
    [
        ("8qia", ("A", 62, "CYS", "SG"), ("B", 500, "FMN", "C4A"), ("B", 500, "FMN", "N5"), 1, 1),
        ("1u9x", ("A", 27, "CYS", "SG"), ("B", 300, "IHJ", "C12"), ("B", 300, "IHJ", "N13"), 2, 1),
        ("2zz6", ("A", 75, "LYS", "NZ"), ("C", 301, "6AZ", "C6"), ("C", 301, "6AZ", "C5"), 1, 2),
    ],
)
def test_pi_bond_addition_transfers_donor_hydrogen(
    pdb_id, donor_key, center_key, acceptor_key, acceptor_bond, acceptor_hydrogens
):
    """C=N, C-triple-N, and C=C additions reduce the pi bond and transfer H."""
    atoms = _parse(pdb_id)
    donor, center, acceptor = (_atom(atoms, *key) for key in (donor_key, center_key, acceptor_key))

    assert _bond_type(atoms, center, donor) == BondType.SINGLE
    assert _bond_type(atoms, center, acceptor) == BondType(acceptor_bond)
    assert atoms.nhyd[acceptor] == acceptor_hydrogens
    assert all(_has_valid_rdkit_valence(atoms, index) for index in (donor, center, acceptor))


def test_dna_link_removes_missing_op3_without_touching_observed_op2():
    """6W13 8OG consumes missing OP3 but preserves the deposited OP2 exactly."""
    path = get_pdb_path("6w13")
    atom_site = read_any(path).block["atom_site"]
    source = (
        (atom_site["label_asym_id"].as_array(str) == "B")
        & (atom_site["label_seq_id"].as_array(str) == "3")
        & (atom_site["label_comp_id"].as_array(str) == "8OG")
        & (atom_site["label_atom_id"].as_array(str) == "OP2")
        & (atom_site["pdbx_PDB_model_num"].as_array(str) == "1")
    )
    source_coord = np.array(
        [atom_site[axis].as_array(float)[source].item() for axis in ("Cartn_x", "Cartn_y", "Cartn_z")]
    )
    atoms = _parse("6w13")
    residue = (atoms.chain_id == "B") & (atoms.res_id == 3) & (atoms.res_name == "8OG")
    op2 = _atom(atoms, "B", 3, "8OG", "OP2")

    np.testing.assert_array_equal(atoms.coord[op2], source_coord.astype(atoms.coord.dtype))
    assert "OP3" not in atoms.atom_name[residue]


def test_thymine_photodimer_consumes_both_pi_bonds():
    """1N4E's two authored links convert adjacent thymines into a cyclobutane."""
    atoms = _parse("1n4e")
    degree = get_bond_degree_per_atom(atoms)

    for atom_name in ("C5", "C6"):
        first = _atom(atoms, "B", 5, "DT", atom_name)
        second = _atom(atoms, "B", 6, "DT", atom_name)
        assert _bond_type(atoms, first, second) == BondType.SINGLE
        assert degree[[first, second]].tolist() == [4, 4]
    for res_id in (5, 6):
        assert (
            _bond_type(atoms, _atom(atoms, "B", res_id, "DT", "C5"), _atom(atoms, "B", res_id, "DT", "C6"))
            == BondType.SINGLE
        )


def test_asn_lys_link_removes_only_unresolved_leaving_nitrogen():
    """6N0A replaces missing ASN ND2 while retaining its observed carbonyl O."""
    atoms = _parse("6n0a")
    residue = (atoms.chain_id == "A") & (atoms.res_id == 153) & (atoms.res_name == "ASN")
    carbon = _atom(atoms, "A", 153, "ASN", "CG")
    nitrogen = _atom(atoms, "A", 15, "LYS", "NZ")

    assert "ND2" not in atoms.atom_name[residue]
    assert "OD1" in atoms.atom_name[residue]
    assert _bond_type(atoms, carbon, nitrogen) == BondType.SINGLE
    assert get_bond_degree_per_atom(atoms)[carbon] == 4


@pytest.mark.parametrize(
    ("pdb_id", "atom_key", "charge"),
    [
        ("8r2z", ("B", 301, "XR9", "B2"), -1),
        ("5lcl", ("C", 8, "8AF", "P"), 1),
        ("1q29", ("A", 1, "G", "O5'"), 1),
        ("8q2p", ("Y", 2424, "IRY", "SE1"), 1),
    ],
)
def test_link_products_assign_only_valid_required_charges(pdb_id, atom_key, charge):
    """Representative borate, phosphonium, oxonium, and selenonium products."""
    atoms = _parse(pdb_id)
    atom = _atom(atoms, *atom_key)

    assert atoms.charge[atom] == charge
    assert _has_valid_rdkit_valence(atoms, atom)


def test_impossible_link_product_still_fails_hard():
    """1VQ7's PAE phosphorus cannot be repaired without bond-order sum seven."""
    with pytest.raises(ValueError, match=r"Unresolved link valence at C/5/PAE/P.*bond-order sum=7"):
        _parse("1vq7")
