"""Checks for hydrogen accounting consistency between "keep" and "remove" paths."""

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure.io import pdbx

from atomworks.constants import STANDARD_AA
from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array import (
    annotate_hydrogens,
    count_bonded_hydrogens,
    get_bond_degree_per_atom,
    remove_hydrogens,
)
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.testing import assert_same_atom_array_or_stack, get_pdb_path


@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "1a1e"])
def test_nhyd_invariant(pdb_id):
    """nhyd absent ↔ explicit H mode; nhyd present ↔ implicit H (no explicit H atoms)."""
    path = get_pdb_path(pdb_id)

    # hydrogen_policy="keep": nhyd must NOT be present
    aa_keep = parse(path, config=ParseConfig(hydrogen_policy="keep", add_missing_atoms=True))["asym_unit"][0]
    assert (
        "nhyd" not in aa_keep.get_annotation_categories()
    ), f"{pdb_id} keep: 'nhyd' should not be set when explicit H atoms are present"

    # hydrogen_policy="remove": nhyd must BE present and no explicit H atoms
    aa_remove = parse(path, config=ParseConfig(hydrogen_policy="remove", add_missing_atoms=True))["asym_unit"][0]
    assert (
        "nhyd" in aa_remove.get_annotation_categories()
    ), f"{pdb_id} remove: 'nhyd' should be set when H atoms are implicit"
    has_explicit_h = np.isin(aa_remove.element, ["H", "D"]).any()
    assert not has_explicit_h, f"{pdb_id} remove: explicit H atoms found despite nhyd being set"


def test_nhyd_explicit_h_double_count_guard():
    """nhyd consumers must raise if an atom carries both nhyd>0 and an explicit bonded H."""
    # CH4: carbon with four explicit bonded H atoms.
    ch4 = struc.array(
        [
            struc.Atom([0.0, 0.0, 0.0], element="C", atom_name="C"),
            *[struc.Atom([1.0, 0.0, 0.0], element="H", atom_name=f"H{i}") for i in range(4)],
        ]
    )
    ch4.bonds = struc.BondList(5, np.array([[0, i, 1] for i in range(1, 5)]))

    # annotate_hydrogens sets nhyd=4 on C *without* removing the explicit H -> illegal both-present state.
    both = annotate_hydrogens(ch4)
    assert both.nhyd[0] == 4 and np.isin(both.element, ["H", "D"]).any()

    with pytest.raises(AssertionError):
        get_bond_degree_per_atom(both)
    with pytest.raises(AssertionError):
        count_bonded_hydrogens(both, 0, include_implicit=True)

    # An all-zero nhyd annotation alongside explicit H is legal (adding zero is a no-op).
    ch4.set_annotation("nhyd", np.zeros(ch4.array_length(), dtype=np.int8))
    np.testing.assert_array_equal(get_bond_degree_per_atom(ch4), [4, 1, 1, 1, 1])


@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "1a1e"])
def test_atom_array_equivalence_keep_vs_remove(pdb_id):
    """Heavy-atom array from 'keep'→annotate→strip should equal 'remove' path."""
    path = get_pdb_path(pdb_id)

    # Path 1: keep H atoms, annotate nhyd from bond graph, then strip H
    aa_keep = parse(path, config=ParseConfig(hydrogen_policy="keep", add_missing_atoms=True))["asym_unit"][0]
    aa_keep = annotate_hydrogens(aa_keep)
    aa_keep_heavy = remove_hydrogens(aa_keep)

    # Path 2: H atoms never added; nhyd sourced from CCD template
    aa_remove = parse(path, config=ParseConfig(hydrogen_policy="remove", add_missing_atoms=True))["asym_unit"][0]

    annotations_to_compare = sorted(
        set(aa_keep_heavy.get_annotation_categories()) & set(aa_remove.get_annotation_categories())
    )
    assert_same_atom_array_or_stack(
        aa_keep_heavy,
        aa_remove,
        compare_coords=False,
        compare_bonds=True,
        annotations_to_compare=annotations_to_compare,
    )


@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "1a1e"])
@pytest.mark.parametrize("missing_value", [None, "?", "."])
@pytest.mark.parametrize("extra_fields", [None, "all"])
def test_nhyd_roundtrip_via_cif(pdb_id, missing_value, extra_fields, tmp_path):
    """Ensure nhyd survives a save→reload cycle."""
    path = get_pdb_path(pdb_id)

    aa = parse(path, config=ParseConfig(hydrogen_policy="remove", add_missing_atoms=True))["asym_unit"][0]

    cif_path = tmp_path / f"{pdb_id}_nhyd.cif"
    to_cif_file(aa, cif_path, extra_fields=["nhyd"])
    expected = aa.nhyd.copy()
    if missing_value is not None:
        cif = pdbx.CIFFile.read(cif_path)
        counts = cif.block["atom_site"]["nhyd"].as_array(str)
        index = np.flatnonzero(aa.atom_name == "CB")[0]
        counts[index] = missing_value
        cif.block["atom_site"]["nhyd"] = pdbx.CIFColumn(counts)
        cif.write(cif_path)
        expected[index] = 0

    aa_reloaded = parse(
        cif_path,
        config=ParseConfig(hydrogen_policy="remove", add_missing_atoms=False, extra_fields=extra_fields),
    )["asym_unit"][0]

    assert "nhyd" in aa_reloaded.get_annotation_categories()
    assert not np.isin(aa_reloaded.element, ["H", "D"]).any()
    np.testing.assert_array_equal(expected, aa_reloaded.nhyd)


def _bonded_h_count(atom_array):
    """Number of *explicit* bonded H per atom."""
    return annotate_hydrogens(atom_array.copy()).nhyd


def _is_non_terminal_backbone_amide_n(atom_array):
    """Mask of backbone N that should carry exactly one amide H (standard AA, not Pro, resolved)."""
    bonds = atom_array.bonds.as_array()
    i, j = bonds[:, 0], bonds[:, 1]
    is_n, is_c = atom_array.atom_name == "N", atom_array.atom_name == "C"
    # A preceding residue's C is bonded to this N, i.e. the N is not a chain start
    inter_res = atom_array.res_id[i] != atom_array.res_id[j]
    has_prev_c = np.zeros(len(atom_array), dtype=bool)
    has_prev_c[i[is_n[i] & is_c[j] & inter_res]] = True
    has_prev_c[j[is_n[j] & is_c[i] & inter_res]] = True

    return (
        has_prev_c
        & np.isin(atom_array.res_name, STANDARD_AA)
        & (atom_array.res_name != "PRO")  # tertiary amide, no N-H
        & ~np.isnan(atom_array.coord).any(axis=-1)  # unresolved atoms are never protonated
    )


@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "1a1e"])
def test_protonation_gives_backbone_amide_nitrogens_their_hydrogen(pdb_id):
    """Backbone amide N must get its H from implicit-H input, identically to explicit-H input."""
    path = get_pdb_path(pdb_id)
    config = {"add_missing_atoms": True}

    # The production path: H are implicit (carried in nhyd), no explicit H atoms
    aa_implicit = parse(path, config=ParseConfig(hydrogen_policy="remove", **config))["asym_unit"][0]
    from_implicit = place_hydrogens(assign_hydrogens(aa_implicit))

    backbone_n = _is_non_terminal_backbone_amide_n(from_implicit)
    n_h = _bonded_h_count(from_implicit)[backbone_n]
    assert (n_h == 1).all(), f"{pdb_id}: {(n_h == 0).sum()} of {backbone_n.sum()} backbone amide N have no H"

    # Explicit-H input gives the same protonation, heavy atom for heavy atom, except where it states
    # a tautomer the implicit input leaves free (1A1E HIS62 is drawn as a cation)
    aa_explicit = parse(path, config=ParseConfig(hydrogen_policy="keep", **config))["asym_unit"][0]
    from_explicit = place_hydrogens(assign_hydrogens(aa_explicit))
    free = from_implicit.tautomer_free[~np.isin(from_implicit.element, ["H", "D"])]
    per_heavy = []
    for protonated in (from_implicit, from_explicit):
        heavy = ~np.isin(protonated.element, ["H", "D"])
        per_heavy.append(np.where(free, -1, _bonded_h_count(protonated)[heavy]))
    np.testing.assert_array_equal(*per_heavy)


@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "1a1e"])
def test_added_hydrogens_survive_cif_roundtrip(pdb_id, tmp_path):
    """Added H come back unchanged from a save->reload cycle."""
    aa = parse(get_pdb_path(pdb_id), config=ParseConfig(hydrogen_policy="remove", add_missing_atoms=True))["asym_unit"][
        0
    ]
    protonated = place_hydrogens(assign_hydrogens(aa))
    assert np.isin(protonated.element, ["H", "D"]).any(), f"{pdb_id}: nothing was protonated"

    cif_path = tmp_path / f"{pdb_id}_protonated.cif"
    to_cif_file(protonated, cif_path)
    reloaded = parse(cif_path, config=ParseConfig(hydrogen_policy="keep", add_missing_atoms=False))["asym_unit"][0]

    # Bonds are excluded: H carry generated names that no CCD template knows, so the reader
    # cannot rebuild their bonds from the file.
    assert_same_atom_array_or_stack(
        protonated,
        reloaded,
        compare_bonds=False,
        cast_to_common_dtype=True,
        annotations_to_compare=sorted(
            set(protonated.get_annotation_categories()) & set(reloaded.get_annotation_categories())
        ),
    )
