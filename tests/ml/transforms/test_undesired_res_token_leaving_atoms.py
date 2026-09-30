from __future__ import annotations

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.enums import ChainType
from atomworks.io.utils.link_chemistry import get_chem_comp_leaving_atom_groups
from atomworks.ml.transforms.filters import HandleUndesiredResTokens

# Non-canonical residues found inside protein chains whose canonical parent is unambiguous
# once the leaving group is disregarded.
MAPPABLE_RESIDUES = [
    pytest.param("ABA", "ALA", id="ABA-to-ALA"),
    pytest.param("SAR", "GLY", id="SAR-to-GLY"),
    pytest.param("PTR", "TYR", id="PTR-to-TYR"),
    pytest.param("SEP", "SER", id="SEP-to-SER"),
]


def _make_polymer_residue(res_name: str, *, keep_leaving_group: bool) -> AtomArray:
    """Build a single protein polymer residue with the annotations the transform requires.

    Args:
        res_name: CCD component id, e.g. ``"PTR"``.
        keep_leaving_group: If ``False``, drop the CCD-declared leaving atoms, i.e. the state
            of a residue inside a chain. If ``True``, keep them, i.e. a chain terminus.

    Returns:
        AtomArray of shape ``[n_atoms]``, annotated with `is_polymer`, `pn_unit_iid`,
        `chain_type` and `atomize`.
    """
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


def _apply(residue: AtomArray, res_name: str) -> AtomArray:
    transform = HandleUndesiredResTokens(undesired_res_tokens=[res_name])
    return transform.forward({"atom_array": residue})["atom_array"]


@pytest.mark.parametrize(("res_name", "expected_canonical"), MAPPABLE_RESIDUES)
def test_residue_inside_a_polymer_maps_to_closest_canonical(res_name: str, expected_canonical: str):
    """A residue in a chain is substituted even though it has lost its leaving group."""
    result = _apply(_make_polymer_residue(res_name, keep_leaving_group=False), res_name)

    assert set(map(str, result.res_name)) == {
        expected_canonical
    }, f"{res_name} should map to {expected_canonical}, got {sorted(set(map(str, result.res_name)))}"
    assert not result.atomize.any(), f"{res_name} should not be atomized after a successful substitution"


@pytest.mark.parametrize(("res_name", "expected_canonical"), MAPPABLE_RESIDUES)
def test_chain_terminus_retains_its_leaving_group(res_name: str, expected_canonical: str):
    """Leaving groups are not *required*, but they are still *kept* when present.

    A chain terminus has a real, observed `OXT`; dropping it would discard experimental
    density. Only the required-atom set excludes leaving groups; the kept-atom set does not.
    """
    result = _apply(_make_polymer_residue(res_name, keep_leaving_group=True), res_name)

    assert set(map(str, result.res_name)) == {expected_canonical}
    assert "OXT" in set(
        map(str, result.atom_name)
    ), f"{res_name} at a chain terminus should keep its OXT, got {sorted(set(map(str, result.atom_name)))}"


@pytest.mark.parametrize(("res_name", "expected_canonical"), MAPPABLE_RESIDUES)
def test_missing_required_backbone_atom_prevents_substitution(res_name: str, expected_canonical: str):
    """A missing backbone N prevents both canonical and unknown-residue substitution."""
    residue = _make_polymer_residue(res_name, keep_leaving_group=False)
    residue = residue[residue.atom_name != "N"]
    original = residue.copy()

    result = _apply(residue, res_name)

    assert np.all(result.res_name == res_name)
    assert result.atomize.all()
    np.testing.assert_array_equal(result.atom_name, original.atom_name)
    np.testing.assert_array_equal(result.coord, original.coord)
    np.testing.assert_array_equal(result.bonds.as_array(), original.bonds.as_array())


@pytest.mark.parametrize(("res_name", "expected_canonical"), MAPPABLE_RESIDUES)
@pytest.mark.parametrize("keep_leaving_group", [False, True], ids=["internal", "terminal"])
def test_substitution_preserves_retained_atom_annotations_and_bonds(
    res_name: str, expected_canonical: str, keep_leaving_group: bool
):
    """Filtering preserves each retained atom's annotations, coordinates, and bond indices."""
    residue = _make_polymer_residue(res_name, keep_leaving_group=keep_leaving_group)
    residue.set_annotation("source_atom_index", np.arange(len(residue)))
    canonical_names = struc.info.residue(expected_canonical).atom_name
    expected = residue[np.isin(residue.atom_name, canonical_names)]
    expected.res_name[:] = expected_canonical

    result = _apply(residue, res_name)

    np.testing.assert_array_equal(result.coord, expected.coord)
    assert set(result.get_annotation_categories()) == set(expected.get_annotation_categories())
    for annotation in expected.get_annotation_categories():
        np.testing.assert_array_equal(result.get_annotation(annotation), expected.get_annotation(annotation))
    np.testing.assert_array_equal(result.bonds.as_array(), expected.bonds.as_array())
