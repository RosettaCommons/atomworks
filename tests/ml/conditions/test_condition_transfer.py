"""Unit tests for the atom-level / residue-level / on_mismatch contract of :py:meth:`ConditionBase.transfer`."""

import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_SEQ
from atomworks.ml.conditions.base import ConditionConflictError


def _arr(n: int, res_names: list[str] | None = None) -> AtomArray:
    a = AtomArray(n)
    a.chain_id = np.array(["A"] * n)
    a.res_id = np.arange(1, n + 1)
    a.res_name = np.array(res_names if res_names is not None else ["ALA"] * n, dtype="<U8")
    a.atom_name = np.array(["CA"] * n)
    a.element = np.array(["C"] * n)
    a.coord = np.zeros((n, 3), dtype=np.float32)
    return as_atom_array_plus(a)


def test_transfer_atom_level_residue_level_and_raise_on_mismatch():
    src, tgt = _arr(2, ["PHE", "SER"]), _arr(2)
    src.coord[:] = [[1, 1, 1], [2, 2, 2]]
    C_CRD.set_annotation(src, src.coord.copy())
    C_SEQ.set_annotation(src, src.res_name.copy())

    # atom-level: per-atom correspondence (src atoms 0,1 -> tgt atoms 1,0)
    C_CRD.transfer(src, tgt, np.array([0, 1]), np.array([1, 0]))
    ann = C_CRD.annotation(tgt, default="generate")
    assert np.allclose(ann[0], [2, 2, 2]) and np.allclose(ann[1], [1, 1, 1])

    # residue-level values copied per position (aligned pairs)
    C_SEQ.transfer(src, tgt, np.array([0, 1]), np.array([0, 1]))
    assert C_SEQ.annotation(tgt, default="generate").tolist() == ["PHE", "SER"]

    # unequal-length indices are a misuse -> raise
    with pytest.raises(ValueError, match="equal-length"):
        C_SEQ.transfer(src, tgt, np.array([0]), np.array([0, 1]))

    # on_mismatch="raise" (default): a differing value on an already-conditioned target clashes,
    # and the target is left untouched by the raise.
    with pytest.raises(ConditionConflictError):
        C_SEQ.transfer(src, tgt, np.array([1]), np.array([0]))  # src[1]=SER vs target=PHE
    assert C_SEQ.annotation(tgt, default="generate")[0] == "PHE"


if __name__ == "__main__":
    pytest.main([__file__])
