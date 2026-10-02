from __future__ import annotations

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.enums import ChainType
from atomworks.io.utils.link_chemistry import get_chem_comp_leaving_atom_groups
from atomworks.ml.transforms.filters import HandleUndesiredResTokens

MAPPABLE_RESIDUES = {"ABA": "ALA", "SAR": "GLY", "PTR": "TYR", "SEP": "SER"}


def _make_polymer_residue(res_name: str, *, keep_leaving_group: bool) -> AtomArray:
    """Build a heavy-atom polymer residue, retaining leaving atoms only for a terminus."""
    residue = struc.info.residue(res_name)
    residue = residue[residue.element != "H"]

    if not keep_leaving_group:
        leaving = {
            str(name)
            for groups in get_chem_comp_leaving_atom_groups(res_name).values()
            for group in groups
            for name in group
        }
        residue = residue[~np.isin(residue.atom_name, list(leaving))]

    n_atoms = residue.array_length()
    residue.res_id = np.full(n_atoms, 5)
    residue.set_annotation("is_polymer", np.ones(n_atoms, dtype=bool))
    residue.set_annotation("pn_unit_iid", np.full(n_atoms, -1, dtype=int))
    residue.set_annotation("chain_type", np.full(n_atoms, int(ChainType.POLYPEPTIDE_L)))
    residue.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))
    return residue


@pytest.mark.parametrize(("res_name", "expected_canonical"), MAPPABLE_RESIDUES.items())
@pytest.mark.parametrize("keep_leaving_group", [False, True], ids=["internal", "terminal"])
def test_canonical_substitution_preserves_retained_atoms(
    res_name: str, expected_canonical: str, keep_leaving_group: bool
):
    """Map internal and terminal residues while preserving retained atoms and any terminal OXT."""
    residue = _make_polymer_residue(res_name, keep_leaving_group=keep_leaving_group)
    residue.set_annotation("source_atom_index", np.arange(len(residue)))
    canonical_names = struc.info.residue(expected_canonical).atom_name
    expected = residue[np.isin(residue.atom_name, canonical_names)]
    expected.res_name[:] = expected_canonical

    transform = HandleUndesiredResTokens(undesired_res_tokens=[res_name])
    result = transform.forward({"atom_array": residue})["atom_array"]

    assert np.all(result.res_name == expected_canonical)
    assert not result.atomize.any()
    assert ("OXT" in result.atom_name) == keep_leaving_group
    np.testing.assert_array_equal(result.coord, expected.coord)
    assert set(result.get_annotation_categories()) == set(expected.get_annotation_categories())
    for annotation in expected.get_annotation_categories():
        np.testing.assert_array_equal(result.get_annotation(annotation), expected.get_annotation(annotation))
    np.testing.assert_array_equal(result.bonds.as_array(), expected.bonds.as_array())


@pytest.mark.parametrize("res_name", MAPPABLE_RESIDUES)
def test_missing_required_backbone_atom_prevents_substitution(res_name: str):
    """A missing backbone N prevents both canonical and unknown-residue substitution."""
    residue = _make_polymer_residue(res_name, keep_leaving_group=False)
    residue = residue[residue.atom_name != "N"]
    original = residue.copy()

    transform = HandleUndesiredResTokens(undesired_res_tokens=[res_name])
    result = transform.forward({"atom_array": residue})["atom_array"]

    assert np.all(result.res_name == res_name)
    assert result.atomize.all()
    np.testing.assert_array_equal(result.atom_name, original.atom_name)
    np.testing.assert_array_equal(result.coord, original.coord)
    np.testing.assert_array_equal(result.bonds.as_array(), original.bonds.as_array())
