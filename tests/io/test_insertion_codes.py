import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse


def test_add_missing_atoms_keeps_residues_sharing_a_number(tmp_path):
    """PDB insertion codes repeat a number (1PPB chain L runs 1H, 1G, ..., 1) and must not collide."""
    residues = [(1, "H", "THR"), (1, "G", "PHE"), (1, "F", "GLY"), (1, "", "CYS"), (2, "", "GLY")]
    atoms = struc.array(
        [
            struc.Atom(
                [3.8 * i + x, 0.0, 0.0],
                chain_id="L",
                res_id=res_id,
                ins_code=ins_code,
                res_name=res_name,
                atom_name=atom_name,
                element=atom_name[0],
            )
            for i, (res_id, ins_code, res_name) in enumerate(residues)
            for atom_name, x in (("N", 0.0), ("CA", 1.46))
        ]
    )
    path = tmp_path / "insertion_codes.pdb"
    file = pdb.PDBFile()
    file.set_structure(atoms)
    file.write(path)

    parsed = parse(str(path))["asym_unit"][0]

    starts = struc.get_residue_starts(parsed)
    assert list(zip(parsed.res_id[starts], parsed.ins_code[starts], strict=True)) == [(r, i) for r, i, _ in residues]
    assert (~np.isnan(parsed.coord).any(axis=1)).sum() == 2 * len(residues)


def _write_pdb(tmp_path, atoms):
    path = tmp_path / "insertion_codes.pdb"
    file = pdb.PDBFile()
    file.set_structure(atoms)
    file.write(path)
    return str(path)


def test_chain_info_tells_residues_sharing_a_number_apart(tmp_path):
    residues = [(100, "", "GLY"), (100, "A", "GLY"), (101, "", "GLY")]
    atoms = struc.array(
        [
            struc.Atom([3.8 * i, 0.0, 0.0], chain_id="A", res_id=r, ins_code=c, res_name=n, atom_name="CA", element="C")
            for i, (r, c, n) in enumerate(residues)
        ]
    )

    info = parse(_write_pdb(tmp_path, atoms))["chain_info"]["A"]

    assert list(zip(info["res_id"], info["ins_code"], info["res_name"], strict=True)) == residues


def test_a_removed_component_leaves_chain_info_when_its_number_stays(tmp_path):
    """SO4 B100 is removed while ZN B100A, in the same non-polymer chain, stays."""
    atoms = struc.array(
        [
            struc.Atom([0.0, 0.0, 0.0], chain_id="B", res_id=100, res_name="SO4", atom_name="S", element="S"),
            struc.Atom(
                [20.0, 0.0, 0.0], chain_id="B", res_id=100, ins_code="A", res_name="ZN", atom_name="ZN", element="ZN"
            ),
        ]
    )
    atoms.hetero[:] = True

    info = parse(_write_pdb(tmp_path, atoms), config=ParseConfig(remove_ccds=("SO4",)))["chain_info"]["B"]

    assert list(zip(info["res_id"], info["ins_code"], info["res_name"], strict=True)) == [(100, "A", "ZN")]
