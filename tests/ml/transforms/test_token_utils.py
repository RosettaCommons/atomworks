import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.io.utils.sequence import STANDARD_PURINE_RESIDUES, STANDARD_PYRIMIDINE_RESIDUES
from atomworks.io.utils.testing import assert_same_atom_array
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalResIdAnnotation,
    AddGlobalTokenIdAnnotation,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import (
    apply_segment_wise_2d,
    get_af3_token_center_masks,
    get_af3_token_representative_masks,
    get_token_count,
    get_token_starts,
    token_iter,
)


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_tokens_are_residues_without_atomization(pdb_id: str):
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]

    assert get_token_count(atom_array) == struc.get_residue_count(atom_array)
    assert np.all(get_token_starts(atom_array) == struc.get_residue_starts(atom_array))
    assert np.all(
        get_token_starts(atom_array, add_exclusive_stop=True)
        == struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    )
    for res_1, res_2 in zip(struc.residue_iter(atom_array), token_iter(atom_array), strict=False):
        assert_same_atom_array(res_1, res_2)


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_tokens_are_atoms_with_full_atomization(pdb_id: str):
    data = cached_parse(pdb_id)
    data = AtomizeByCCDName(atomize_by_default=True)(data)
    atom_array = data["atom_array"]
    assert get_token_count(atom_array) == len(atom_array)
    assert np.all(get_token_starts(atom_array) == np.arange(len(atom_array)))
    assert np.all(get_token_starts(atom_array, add_exclusive_stop=True) == np.arange(len(atom_array) + 1))


def test_apply_segment_wise_2d():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    segment_start_end_idxs = np.array([0, 2, 3])
    assert np.all(
        apply_segment_wise_2d(array, segment_start_end_idxs, reduce_func=np.sum) == np.array([[12, 9], [15, 9]])
    )


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_fully_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True),  # atomize all residues
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert np.all(atom_array.atom_id == atom_array.token_id), (
        "atom_id and token_id should be the same for a fully atomized atom_array"
    )

    # cross check by iterating over the tokens
    # ... via token starts
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for counter, (start, end) in enumerate(zip(token_start_idxs[:-1], token_start_idxs[1:], strict=False)):
        token = atom_array[start:end]
        assert len(token) == 1, f"token should have length 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"

    # ... via token_iter
    for counter, token in enumerate(token_iter(atom_array)):
        assert len(token) == 1, f"token should have length 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"

    # ... via atom iter (since when fully atomized, tokens are atoms)
    for counter, token in enumerate(atom_array):
        assert token.token_id == counter, f"token_id should be {counter} but is {token.token_id}"


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_not_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert atom_array.atom_id[-1] > atom_array.token_id[-1], "There should be more atom_ids than token_ids."

    # cross check by iterating over the tokens
    # ... via token starts
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for counter, (start, end) in enumerate(zip(token_start_idxs[:-1], token_start_idxs[1:], strict=False)):
        token = atom_array[start:end]
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"

    # ... via token_iter
    for counter, token in enumerate(token_iter(atom_array)):
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"

    # ... via residue iter (since when not atomizing, tokens are residues)
    counter = 0
    for residue in struc.residue_iter(atom_array):
        assert len(residue) >= 1, f"residue should have length at least 1 but has length {len(residue)}"
        assert np.all(residue.token_id == counter), f"token_id should be {counter} but is {residue.token_id}"
        counter += 1


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_partially_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert atom_array.atom_id[-1] > atom_array.token_id[-1], "There should be more atom_ids than token_ids."

    # cross check by iterating over the tokens
    # ... via token starts
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for counter, (start, end) in enumerate(zip(token_start_idxs[:-1], token_start_idxs[1:], strict=False)):
        token = atom_array[start:end]
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"

    # ... via token_iter
    for counter, token in enumerate(token_iter(atom_array)):
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_get_token_center_atoms(pdb_id):
    data = cached_parse(pdb_id)

    atom_array = data["atom_array"]
    # HACK: refactor to make this not necessary
    # currently the atomize field is set even for nonprotein residues by the AtomizeByCCDName transform
    # so you must call the AtomizeByCCDName transform to get token atoms or representative atoms
    tranform = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
        move_atomized_part_to_end=False,
        validate_atomize=False,
    )
    data = tranform(data)
    token_center_masks = get_af3_token_center_masks(atom_array)
    token_center_atoms = token_center_masks.nonzero()[0]
    assert len(token_center_atoms) == get_token_count(atom_array)
    # test that nucleotides get C1' as the center atom, and proteins get CA and the rest of the nodes are atomized
    for token, mask in zip(token_iter(atom_array), token_center_atoms, strict=False):
        assert len(set(token.res_name)) == 1
        res_name = token.res_name[0]
        if res_name in STANDARD_AA:
            assert atom_array[mask].atom_name == "CA"
        elif res_name in STANDARD_RNA + STANDARD_DNA:
            assert atom_array[mask].atom_name == "C1'"
        else:
            assert atom_array[mask].atomize  # atomize should be True


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_get_token_representative_atoms(pdb_id):
    data = cached_parse(pdb_id)

    atom_array = data["atom_array"]

    # HACK: refactor to make this not necessary
    # currently the atomize field is set even for nonprotein residues by the AtomizeByCCDName transform
    # so you must call the AtomizeByCCDName transform to get token atoms or representative atoms
    tranform = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
        move_atomized_part_to_end=False,
        validate_atomize=False,
    )
    data = tranform(data)
    representative_atoms = get_af3_token_representative_masks(atom_array)
    representative_atoms = representative_atoms.nonzero()[0]

    assert len(representative_atoms) == get_token_count(atom_array)
    # test that purines get C4, pyrimdines get C2, proteins other than glycine get CB, glycine gets CB and the rest are atoms
    for token, mask in zip(token_iter(atom_array), representative_atoms, strict=False):
        assert len(set(token.res_name)) == 1
        res_name = token.res_name[0]
        if res_name in STANDARD_PURINE_RESIDUES:
            assert atom_array[mask].atom_name == "C4"
        elif res_name in STANDARD_PYRIMIDINE_RESIDUES:
            assert atom_array[mask].atom_name == "C2"
        elif res_name in STANDARD_AA and res_name != "GLY":
            assert atom_array[mask].atom_name == "CB"
        elif res_name in STANDARD_AA and res_name == "GLY":
            assert atom_array[mask].atom_name == "CA"
        else:
            assert atom_array[mask].atomize  # atomize should be True


def test_add_global_res_id_annotation():
    """Test that AddGlobalResIdAnnotation adds correct residue IDs."""
    data = cached_parse("5ocm")
    transform = AddGlobalResIdAnnotation()
    data = transform(data)

    atom_array = data["atom_array"]

    assert "res_id_global" in atom_array.get_annotation_categories()

    # Get the expected number of residues
    expected_n_residues = struc.get_residue_count(atom_array)

    # Check that res_id_global values are in the expected range
    unique_global_res_ids = np.unique(atom_array.res_id_global)
    assert len(unique_global_res_ids) == expected_n_residues, (
        f"Expected {expected_n_residues} unique residue IDs, got {len(unique_global_res_ids)}"
    )
    assert np.all(unique_global_res_ids == np.arange(expected_n_residues)), (
        "Global residue IDs should be 0-indexed and continuous"
    )

    # Check that residues have consistent res_id_global values
    for counter, residue in enumerate(struc.residue_iter(atom_array)):
        assert len(residue) >= 1, f"Residue should have length at least 1 but has length {len(residue)}"
        assert np.all(residue.res_id_global == counter), (
            f"All atoms in residue should have res_id_global {counter} but got {residue.res_id_global}"
        )


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
