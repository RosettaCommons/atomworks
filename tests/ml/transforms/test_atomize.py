import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import STANDARD_AA
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.utils.testing import cached_parse


def test_fail_on_invalid_init():
    with pytest.raises(ValueError):
        AtomizeByCCDName(
            atomize_by_default=True, res_names_to_atomize=["ALA", "ILE"], res_names_to_ignore=["MET", "ALA"]
        )


def test_fail_on_invalid_atomize_annotation():
    with pytest.raises(ValueError):
        transform = AtomizeByCCDName(atomize_by_default=False, validate_atomize=True)
        data = {"atom_array": struc.info.residue("ALA")}
        data["atom_array"].set_annotation("atomize", np.array([False] * len(data["atom_array"])))

        # Introduce invalid, mixed atomize annotation for ALA (cannot atomize only some atoms of the residue)
        data["atom_array"].atomize[0] = True

        data = transform(data)


def test_res_name_to_atomize_overrides_default():
    transform = AtomizeByCCDName(atomize_by_default=False, res_names_to_atomize=["MET"])

    data = {"atom_array": struc.info.residue("ALA")}
    data = transform(data)
    assert not np.any(data["atom_array"].atomize), "ALA should not be atomized"

    data = {"atom_array": struc.info.residue("MET")}
    data = transform(data)
    assert np.all(data["atom_array"].atomize), "MET should be atomized"


def test_res_name_to_ignore_overrides_default():
    transform = AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=["ALA"])

    data = {"atom_array": struc.info.residue("ALA")}
    data = transform(data)
    assert not np.any(data["atom_array"].atomize), "ALA should not be atomized"

    data = {"atom_array": struc.info.residue("ILE")}
    data = transform(data)
    assert np.all(data["atom_array"].atomize), "ILE should be atomized"


def test_custom_atomize_annotation_overwrites_default():
    transform = AtomizeByCCDName(atomize_by_default=False)

    # Check default works as expected
    data = {"atom_array": struc.info.residue("MET")}
    data = transform(data)
    assert not np.any(data["atom_array"].atomize), "MET should not be atomized"

    # Check override `default`:
    data = {"atom_array": struc.info.residue("MET")}
    data["atom_array"].set_annotation("atomize", np.array([True] * len(data["atom_array"])))
    data = transform(data)
    assert np.all(data["atom_array"].atomize), "MET should be atomized"  # noqa


def test_custom_atomize_annotation_overwrites_ignore():
    transform = AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=["ALA"])

    # Check default works as expected
    data = {"atom_array": struc.info.residue("MET")}
    data = transform(data)
    assert np.all(data["atom_array"].atomize), "MET should be atomized"

    # Check ignore works as expected
    data = {"atom_array": struc.info.residue("ALA")}
    data = transform(data)
    assert not np.any(data["atom_array"].atomize), "ALA should not be atomized"

    # Check override `ignore`
    data = {"atom_array": struc.info.residue("ALA")}
    data["atom_array"].set_annotation("atomize", np.array([True] * len(data["atom_array"])))
    data = transform(data)
    assert np.all(data["atom_array"].atomize), "ALA should be atomized"


@pytest.mark.parametrize("pdb_id", ["5ocm"])
def test_atomizing_non_protein_residues(pdb_id: str):
    data = cached_parse(pdb_id)

    transform = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA,
    )
    data = transform(data)

    _is_protein = struc.filter_canonical_amino_acids(data["atom_array"])
    assert not np.any(data["atom_array"][_is_protein].atomize), "Protein residues should not be atomized"
    assert np.all(data["atom_array"][~_is_protein].atomize), "Non-protein residues should be atomized"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
