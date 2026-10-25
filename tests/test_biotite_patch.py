import biotite.structure as struc
import numpy as np

import atomworks  # noqa: F401  (importing atomworks applies the Biotite patches)


def test_array_keeps_full_strings_when_rebuilt_from_its_own_atoms():
    atom_array = struc.array(
        [
            struc.Atom([1.0, 2.0, 3.0], chain_id="A", res_name="ALA", atom_name="C"),
            struc.Atom([2.0, 3.0, 4.0], chain_id="LONGCHAIN", res_name="HIS", atom_name="CA"),
        ]
    )
    atom_array.set_annotation("note", np.array(["x", "longer"]))

    rebuilt = struc.array(list(atom_array))

    assert list(rebuilt.atom_name) == ["C", "CA"]
    assert list(rebuilt.chain_id) == ["A", "LONGCHAIN"]
    assert list(rebuilt.res_name) == ["ALA", "HIS"]
    assert list(rebuilt.note) == ["x", "longer"]


def test_array_keeps_non_string_annotation_dtypes_when_rebuilt_from_its_own_atoms():
    atom_array = struc.array(
        [
            struc.Atom([1.0, 2.0, 3.0], res_id=1, hetero=True),
            struc.Atom([2.0, 3.0, 4.0], res_id=2, hetero=False),
        ]
    )
    atom_array.set_annotation("b_factor", np.array([1.5, 2.5], dtype=np.float32))

    rebuilt = struc.array(list(atom_array))

    assert rebuilt.res_id.dtype == atom_array.res_id.dtype
    assert rebuilt.hetero.dtype == atom_array.hetero.dtype
    assert rebuilt.b_factor.dtype == np.float32
    assert list(rebuilt.res_id) == [1, 2]
    assert np.array_equal(rebuilt.b_factor, [1.5, 2.5])
