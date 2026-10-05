import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

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
