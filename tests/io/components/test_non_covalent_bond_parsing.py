import re

import biotite.structure as struc
import pytest

from atomworks.constants import STRUCT_CONN_BOND_TYPES
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

# (pdb_id, assembly_id, expected_num_coord_bonds)
TEST_CASE_COORD = [
    ("3u8v", "3", 6),  # Contains some coordination bonds
]

TEST_CASE_HYDROG = [
    ("2k33", "1"),  # Structure doesn't matter here, an error will be thrown regardless
]

# (pdb_id, assembly_id, expected_num_potentially_spurious_non_coord_bonds, expected_num_potentially_spurious_coordination_bonds)
# Potentially spurious bonds are those between non-polymer atoms with different auth_seq_ids
# These should only be present if specified in the struct_conn category (from which these expected values are obtained)
TEST_CASE_FIX_BIOTITE_BONDS = [
    ("2msb", "1", 6, 2),  # Contains adjacent identical non-polymer residues, causing biotite to add spurious bonds
]

# (pdb_id, assembly_id, num_coord_bonds, num_disulfide_bonds, num_covalent_bonds)
# All bond counts are those present in the `struct_conn` category of the PDB file, not overall in the structure
TEST_CASE_FILTERING = [
    ("2msb", "1", 34, 4, 6),  # Contains coordination, disulfide, and covalent bonds
    (
        "1n45",
        "1",
        1,
        0,
        0,
    ),  # Contains intra-residue coordination bonds (heme). The CCD does not distinguish these from covalent.
]


@pytest.mark.parametrize("test_case", TEST_CASE_FIX_BIOTITE_BONDS)
def test_fix_biotite_bonds(test_case: tuple):
    (
        pdb_id,
        assembly_id,
        expected_num_potentially_spurious_non_coordination_bonds,
        expected_num_potentially_spurious_coordination_bonds,
    ) = test_case
    path = get_pdb_path(pdb_id)

    # Parse AtomArray from CIF
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_missing_atoms=False,
        add_bond_types_from_struct_conn=[
            "covale",
            "metalc",
            "disulf",
        ],
    )
    atom_array = result["assemblies"][assembly_id][0]
    is_polymer = atom_array.is_polymer
    auth_seq_ids = atom_array.auth_seq_id

    bonds_array = atom_array.bonds.as_array()
    bonds_between_non_polymers = ~is_polymer[bonds_array[:, 0]] & ~is_polymer[bonds_array[:, 1]]
    bonds_between_auth_seq_ids = auth_seq_ids[bonds_array[:, 0]] != auth_seq_ids[bonds_array[:, 1]]

    coord_bonds_mask = bonds_array[:, 2] == struc.BondType.COORDINATION
    potentially_spurious_bonds_mask = bonds_between_non_polymers & bonds_between_auth_seq_ids

    # Check that we have the expected number of metal bonds
    assert (
        sum(potentially_spurious_bonds_mask & coord_bonds_mask) == expected_num_potentially_spurious_coordination_bonds
    )

    # Check that we have the expected number of non-coordination bonds
    # This language is general, but these should not be disulfides, which should involve polymer atoms
    assert (
        sum(potentially_spurious_bonds_mask & ~coord_bonds_mask)
        == expected_num_potentially_spurious_non_coordination_bonds
    )


@pytest.mark.parametrize("test_case", TEST_CASE_COORD)
def test_coordination_bond_parsing(test_case: tuple):
    pdb_id, assembly_id, expected_num_coord_bonds = test_case
    path = get_pdb_path(pdb_id)

    # Parse AtomArray from CIF
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_bond_types_from_struct_conn=[
            "covale",
            "metalc",
            "disulf",
        ],
    )
    atom_array = result["assemblies"][assembly_id][0]

    num_coord_bonds = sum(atom_array.bonds.as_array()[:, 2] == struc.BondType.COORDINATION)

    assert num_coord_bonds == expected_num_coord_bonds


@pytest.mark.parametrize("test_case", TEST_CASE_HYDROG)
def test_hydrogen_bond_parsing(test_case: tuple):
    pdb_id, assembly_id = test_case
    path = get_pdb_path(pdb_id)

    expected_error_msg = re.escape(
        f"Invalid bond type(s) provided: { {'hydrog'} }! Valid bond types are: {STRUCT_CONN_BOND_TYPES}"
    )

    with pytest.raises(ValueError, match=expected_error_msg):
        parse(
            filename=path,
            build_assembly=[assembly_id],
            add_bond_types_from_struct_conn=[
                "covale",
                "metalc",
                "disulf",
                "hydrog",
            ],
        )


@pytest.mark.parametrize("test_case", TEST_CASE_FILTERING)
def test_bond_type_filtering(test_case: tuple):
    (
        pdb_id,
        assembly_id,
        num_coord_bonds_struct_conn,
        num_disulfide_bonds_struct_conn,
        num_covalent_bonds_struct_conn,
    ) = test_case
    path = get_pdb_path(pdb_id)

    # Parse AtomArray from CIF, including all supported bond types
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_missing_atoms=False,
        add_bond_types_from_struct_conn=[
            "covale",
            "metalc",
            "disulf",
        ],
    )
    atom_array_all_bonds = result["assemblies"][assembly_id][0]

    # Record the total number of bonds present
    total_num_bonds = atom_array_all_bonds.bonds.as_array().shape[0]

    # Count coordination bonds.
    # NOTE: Parsing retypes every metal-incident bond to COORDINATION regardless of its source, so this total
    # is *not* the `struct_conn` count -- intra-residue metal bonds from `chem_comp_bond` (e.g. the heme Fe-N
    # bonds in 1n45) are coordination bonds too. What `struct_conn` filtering controls is the *difference*
    # between the two parses, which is asserted below.
    num_coord_bonds_all = sum(atom_array_all_bonds.bonds.as_array()[:, 2] == struc.BondType.COORDINATION)

    # Parse AtomArray from CIF, including only covalent bonds from `struct_conn`
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_missing_atoms=False,
        add_bond_types_from_struct_conn=[
            "covale",
        ],
    )
    atom_array_covalent_only = result["assemblies"][assembly_id][0]

    # Record the total number of bonds present
    total_num_bonds_no_struct_conn_coord_or_disulfide = atom_array_covalent_only.bonds.as_array().shape[0]

    # Dropping `metalc` removes exactly the struct_conn coordination bonds; metal bonds inferred from the
    # CCD remain and stay typed COORDINATION.
    num_coord_bonds_covalent_only = sum(atom_array_covalent_only.bonds.as_array()[:, 2] == struc.BondType.COORDINATION)
    assert num_coord_bonds_all - num_coord_bonds_covalent_only == num_coord_bonds_struct_conn

    # We cannot directly detect disulfides, but we can infer how many were filtered out
    num_filtered_disulfide_bonds = (
        total_num_bonds - total_num_bonds_no_struct_conn_coord_or_disulfide - num_coord_bonds_struct_conn
    )
    assert num_filtered_disulfide_bonds == num_disulfide_bonds_struct_conn

    # Parse AtomArray from CIF, including no bonds from `struct_conn`
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_missing_atoms=False,
        add_bond_types_from_struct_conn=[],
    )
    atom_array_no_struct_conn = result["assemblies"][assembly_id][0]

    # Record the total number of bonds present
    num_non_struct_conn_bonds = atom_array_no_struct_conn.bonds.as_array().shape[0]
    num_filtered_covalent_bonds = total_num_bonds_no_struct_conn_coord_or_disulfide - num_non_struct_conn_bonds

    assert num_filtered_covalent_bonds == num_covalent_bonds_struct_conn
