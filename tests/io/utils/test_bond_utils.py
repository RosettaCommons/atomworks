from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, BondList, BondType

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.template import get_empty_ccd_template
from atomworks.io.utils.atom_array import get_bond_degree_per_atom
from atomworks.io.utils.bonds import (
    correct_formal_charges_for_specified_atoms,
    hash_atom_array,
)
from atomworks.io.utils.link_chemistry import (
    get_chem_comp_leaving_atom_groups,
    get_inter_residue_atom_mask,
    resolve_leaving_atoms,
)
from tests.io.conftest import TEST_DATA_IO, get_pdb_path

TEST_DATA_DIR = Path(__file__).parent

LEAVING_GROUP_TEST_CASES = {
    "ALA": {"N": (("H2",),), "C": (("OXT", "HXT"),), "OXT": (("HXT",),)},
    "TYR": {"N": (("H2",),), "C": (("OXT", "HXT"),), "OXT": (("HXT",),)},
}


def _atom_index(atoms, chain_id, res_id, res_name, atom_name):
    return np.flatnonzero(
        (atoms.chain_id == chain_id)
        & (atoms.res_id == res_id)
        & (atoms.res_name == res_name)
        & (atoms.atom_name == atom_name)
    ).item()


def test_struct_conn_uses_canonical_labels_not_author_insertions():
    """5A7X's three ASN 88B glycosylation bonds resolve by canonical label identity."""
    atoms = parse(
        TEST_DATA_IO / "5a7x.cif.zst",
        config=ParseConfig.from_preset("minimal", model=1, build_assembly=None),
    )["asym_unit"][0]
    bond_pairs = {frozenset(bond[:2]) for bond in atoms.bonds.as_array()}

    for protein_chain, glycan_chain in (("N", "IB"), ("P", "JB"), ("R", "KB")):
        nd2 = _atom_index(atoms, protein_chain, 88, "ASN", "ND2")
        c1 = _atom_index(atoms, glycan_chain, 1000, "NAG", "C1")
        assert frozenset((nd2, c1)) in bond_pairs


def test_struct_conn_keeps_explicit_long_glycosylation(caplog):
    """2ODP's authored 2.211 A N-glycosylation survives the ordinary 1.7 A filter."""
    atoms = parse(
        TEST_DATA_IO / "2odp.cif.zst",
        config=ParseConfig.from_preset("minimal", model=1, build_assembly=None),
    )["asym_unit"][0]
    bond_pairs = {frozenset(bond[:2]) for bond in atoms.bonds.as_array()}

    nd2 = _atom_index(atoms, "A", 408, "ASN", "ND2")
    c1 = _atom_index(atoms, "B", 1, "NAG", "C1")
    assert frozenset((nd2, c1)) in bond_pairs
    assert "pdbx_dist_value 2.211 A exceeds 1.7 A threshold (keeping through 2.4 A)" in caplog.text


def test_struct_conn_filters_long_competing_altloc_bond(caplog):
    """1RCQ filters the competing altloc bond and infers the kept PLP-DLY double bond."""
    atoms = parse(
        TEST_DATA_IO / "1rcq.cif.zst",
        config=ParseConfig.from_preset("minimal", model=1, build_assembly=None, altloc="B"),
    )["asym_unit"][0]
    bond_pairs = {frozenset(bond[:2]) for bond in atoms.bonds.as_array()}

    lys = _atom_index(atoms, "A", 33, "LYS", "NZ")
    plp = _atom_index(atoms, "B", 358, "PLP", "C4A")
    dly = _atom_index(atoms, "C", 359, "DLY", "NZ")
    assert frozenset((plp, dly)) in bond_pairs
    assert frozenset((plp, lys)) not in bond_pairs
    neighbors, orders = atoms.bonds.get_bonds(plp)
    assert orders[neighbors == dly].item() == 2
    assert "pdbx_dist_value 1.784 A exceeds 1.7 A threshold" in caplog.text


@pytest.mark.parametrize("ccd_code, expected_leaving_groups", LEAVING_GROUP_TEST_CASES.items())
def test_leaving_group_computation(ccd_code, expected_leaving_groups):
    assert get_chem_comp_leaving_atom_groups(ccd_code) == expected_leaving_groups


def test_fix_formal_charge_of_deprotonated_alanine():
    ala = get_empty_ccd_template("ALA", res_id=1, hydrogen_policy="keep")
    assert np.array_equal(ala.charge, np.zeros(len(ala)))

    ala_oxt_deprotonated = ala[ala.atom_name != "HXT"]
    assert (
        correct_formal_charges_for_specified_atoms(ala_oxt_deprotonated, np.ones(len(ala) - 1, dtype=bool))[
            ala_oxt_deprotonated.atom_name == "OXT"
        ].charge
        == -1
    )


def test_hash_atom_array():
    arr1 = get_empty_ccd_template("ALA", res_id=1, chain_id="A", hydrogen_policy="keep")
    arr2 = arr1.copy()
    assert hash_atom_array(arr1, annotations=None) == hash_atom_array(arr2, annotations=None)
    assert hash_atom_array(arr1, annotations=["atom_name"]) == hash_atom_array(arr2, annotations=["atom_name"])
    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )
    # ... invert the order
    invert_order = np.arange(len(arr1))[::-1]
    arr2 = arr1[invert_order]

    # DEBUG: Uncomment for manual inspection
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from atomworks.io.utils.bonds import _atom_array_to_networkx_graph
    # gs = []
    # annotations = ["element"]
    # for arr in [arr1, arr2]:
    #     g = _atom_array_to_networkx_graph(arr, annotations=annotations, bond_order=True)
    #     gs.append(g)
    # def show_graph(G, figsize=(10, 10), node_attr="node_data", edge_attr="bond_type"):
    #     fig, ax = plt.subplots(figsize=figsize)
    #     pos = nx.kamada_kawai_layout(G)
    #     node_values = [G.nodes[node].get(node_attr, "") for node in G.nodes()]
    #     edge_values = [G[u][v].get(edge_attr, "") for u, v in G.edges()]
    #     unique_node_values = list(set(node_values))
    #     unique_edge_values = list(set(edge_values))
    #     node_colors = [unique_node_values.index(val) for val in node_values]
    #     edge_colors = [unique_edge_values.index(val) for val in edge_values]
    #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20)
    #     nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.tab20)
    #     node_labels = {node: G.nodes[node].get(node_attr, "") for node in G.nodes()}
    #     nx.draw_networkx_labels(G, pos, labels=node_labels)
    #     edge_labels = {(u, v): G[u][v].get(edge_attr, "") for u, v in G.edges()}
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #     plt.axis("off")
    #     return fig, ax
    # show_graph(gs[0])
    # show_graph(gs[1])

    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )
    # ... swap first two atoms
    swap_first_two = np.arange(len(arr1))
    swap_first_two[0], swap_first_two[1] = swap_first_two[1], swap_first_two[0]
    arr2 = arr1[swap_first_two]
    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )


@pytest.mark.parametrize("pdb_id", ["1TQH"])
def test_resolve_leaving_atoms_handles_nucleophilic_additions(pdb_id: str):
    """Verify that resolve_leaving_atoms handles nucleophilic additions (bond order decrement for carbons).

    1TQH chain D has a 4PA-701 residue where SER-94 OG attacks the carbonyl carbon CAI.
    The C=O bond to OAD must be decremented to C-O so that the carbon doesn't exceed degree 4.
    """
    path = get_pdb_path(pdb_id)

    # Parse without leaving atom resolution — carbon should be over-saturated
    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
        add_missing_atoms=False,
    )
    atom_array = result["assemblies"]["1"][0]
    carbon_mask = atom_array.element == "C"
    degrees = get_bond_degree_per_atom(atom_array)
    assert not np.all(degrees[carbon_mask] <= 4), "Example does not show a nucleophilic addition!"

    # Parse with defaults — resolve_leaving_atoms should fix the bond order
    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
    )
    atom_array = result["assemblies"]["1"][0]
    carbon_mask = atom_array.element == "C"
    degrees = get_bond_degree_per_atom(atom_array)
    assert np.all(degrees[carbon_mask] <= 4), "Carbon degree > 4 after resolve_leaving_atoms!"

    # Verify that the heteroatom partner (OAD) gets charge=-1 after bond-type + charge correction.
    # OAD is an oxygen on 4PA-701 whose C=O bond was downgraded to C-O, making it an alkoxide (charge=-1).
    oad_mask = (atom_array.atom_name == "OAD") & (atom_array.res_name == "4PA")
    assert oad_mask.any(), "OAD atom not found in 4PA residue"
    assert np.all(
        atom_array.charge[oad_mask] == -1
    ), f"OAD should have charge=-1 after C=O->C-O correction, got {atom_array.charge[oad_mask]}"


def _count_bonded_hydrogens(atom_array: AtomArray, atom_idx: int) -> int:
    """Count hydrogens bonded to the atom at ``atom_idx``."""
    bonds = atom_array.bonds.as_array()
    return sum(
        1
        for b in bonds
        if (b[0] == atom_idx and atom_array.element[b[1]] == "H")
        or (b[1] == atom_idx and atom_array.element[b[0]] == "H")
    )


def test_resolve_leaving_atoms_respects_double_bond_order():
    """Verify that a double inter-residue bond removes 2 H, not 1.

    Uses a Schiff base theozyme where LYS NZ=CX (double bond via pdbx_value_order).
    CCD gives NZ 3 H (HZ1-3); the double bond should displace exactly 2, leaving 1 H.
    """
    # The CCD template for a free lysine carries 3 hydrogens on NZ
    lys_template = get_empty_ccd_template("LYS", res_id=1, hydrogen_policy="keep")
    nz_idx_template = int((lys_template.atom_name == "NZ").nonzero()[0][0])
    h_before = _count_bonded_hydrogens(lys_template, nz_idx_template)
    assert h_before == 3, f"CCD template should give NZ 3 H, got {h_before}"

    # Parsing resolves leaving atoms, which should displace exactly 2 of them for the double bond
    path = str(TEST_DATA_DIR / "schiff_base_double_bond.cif")
    result = parse(filename=path, build_assembly=None, hydrogen_policy="keep")
    aa = result["asym_unit"][0]

    nz_idx = int((aa.atom_name == "NZ").nonzero()[0][0])
    h_after = _count_bonded_hydrogens(aa, nz_idx)
    assert h_after == 1, f"Double bond should leave 1 H on NZ (removed 2 of 3), got {h_after}"


@pytest.mark.parametrize("hydrogen_policy", ["keep", "remove"])
def test_coordination_preserves_leaving_atoms_and_valence(hydrogen_policy):
    """Metal coordination preserves the donor's hydrogens, valence count, and charge."""
    atoms = get_empty_ccd_template("ALA", res_id=1, hydrogen_policy=hydrogen_policy)
    atoms += get_empty_ccd_template("MN", res_id=2, hydrogen_policy=hydrogen_policy)
    degree = get_bond_degree_per_atom(atoms)
    (oxygen,) = np.flatnonzero(atoms.atom_name == "OXT")
    atoms.bonds.add_bond(int(oxygen), len(atoms) - 1, BondType.COORDINATION)
    before = atoms.copy()

    resolved, impacted = resolve_leaving_atoms(atoms)
    resolved = correct_formal_charges_for_specified_atoms(resolved, impacted)

    assert not impacted.any()
    np.testing.assert_array_equal(resolved.atom_name, before.atom_name)
    np.testing.assert_array_equal(resolved.bonds.as_array(), before.bonds.as_array())
    np.testing.assert_array_equal(resolved.charge, before.charge)
    np.testing.assert_array_equal(get_bond_degree_per_atom(resolved), degree)
    if hydrogen_policy == "remove":
        np.testing.assert_array_equal(resolved.nhyd, before.nhyd)


def test_inter_residue_bonds_preserve_assembly_and_insertion_identity():
    atoms = AtomArray(7)
    atoms.chain_id[:] = "A"
    atoms.chain_id[4] = "B"
    atoms.res_id[:] = [1, 1, 1, 1, 1, 2, 3]
    atoms.ins_code[3] = "A"
    atoms.set_annotation("transformation_id", np.array([0, 0, 1, 0, 0, 0, 0]))
    atoms.bonds = BondList(
        7,
        np.array(
            [
                [0, 1, BondType.SINGLE],
                [0, 2, BondType.SINGLE],
                [0, 3, BondType.SINGLE],
                [0, 4, BondType.SINGLE],
                [0, 5, BondType.SINGLE],
                [0, 6, BondType.COORDINATION],
            ]
        ),
    )
    np.testing.assert_array_equal(get_inter_residue_atom_mask(atoms), [True, False, True, True, True, True, False])


if __name__ == "__main__":
    test_resolve_leaving_atoms_handles_nucleophilic_additions("1j8z")
