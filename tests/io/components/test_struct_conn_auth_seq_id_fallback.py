"""Regression test for struct_conn auth_seq_id fallback bug.

When ptnr_label_seq_id is "." for non-polymers, the code correctly falls back to
ptnr_auth_seq_id to determine the residue ID. However, the matching logic was using
atom_array.res_id (which contains label_seq_id values) instead of auth_seq_id,
causing inter-chain bonds to be missed when auth_asym_id differs from label_asym_id.
"""

import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

# Representative PDB entries affected by the bug
AFFECTED_PDB_IDS = ["9b5c", "8an9", "1lgc"]


@pytest.mark.parametrize("pdb_id", AFFECTED_PDB_IDS)
def test_struct_conn_auth_seq_id_fallback(pdb_id: str):
    """Verify inter-chain bonds are found when label_seq_id='.' requires auth_seq_id fallback."""
    path = get_pdb_path(pdb_id)
    result = parse(filename=path)
    # Use asym_unit (the raw parsed structure) for bond checking
    atom_array = result["asym_unit"][0]

    assert atom_array.bonds is not None
    bonds = atom_array.bonds.as_array()
    chain_ids = atom_array.chain_id

    # Check for inter-chain bonds
    inter_chain_bonds = [(chain_ids[b[0]], chain_ids[b[1]]) for b in bonds if chain_ids[b[0]] != chain_ids[b[1]]]
    assert len(inter_chain_bonds) > 0, f"No inter-chain bonds found for {pdb_id}"


if __name__ == "__main__":
    pytest.main([__file__])
