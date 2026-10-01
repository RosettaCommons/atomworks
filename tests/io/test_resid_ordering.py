import pytest
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from atomworks.io._pipeline import _check_resid_ordering
from atomworks.io.utils.io_utils import _validate_label_seq_ids


def test_check_resid_ordering_raises_on_decreasing():
    atoms = AtomArray(3)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 3, 2]

    with pytest.raises(ValueError, match="chain 'A'"):
        _check_resid_ordering(atoms)


def test_cif_identity_invariants():
    block = pdbx.CIFBlock()
    block["atom_site"] = pdbx.CIFCategory(
        {"label_asym_id": ["A", "A"], "label_seq_id": ["1", "."], "label_entity_id": ["1", "1"]}
    )
    with pytest.raises(ValueError, match="mixes defined and missing label_seq_id"):
        _validate_label_seq_ids(block)

    block["atom_site"]["label_seq_id"] = [".", "."]
    block["entity"] = pdbx.CIFCategory({"id": ["1"], "type": ["polymer"]})
    with pytest.raises(ValueError, match="including noncanonical monomers"):
        _validate_label_seq_ids(block)

    block["atom_site"]["label_seq_id"] = ["1", "2"]
    block["atom_site"]["label_entity_id"] = ["1", "2"]
    with pytest.raises(ValueError, match="maps to multiple label_entity_id"):
        _validate_label_seq_ids(block)


@pytest.mark.parametrize("entity_type", ["polymer", " Polymer ", "non-polymer", "water"])
def test_cif_missing_sequence_identity_depends_on_entity_type(entity_type):
    block = pdbx.CIFBlock()
    block["atom_site"] = pdbx.CIFCategory(
        {"label_asym_id": ["A", "B", "C"], "label_seq_id": ["1", ".", "?"], "label_entity_id": ["1", "2", "3"]}
    )
    block["entity"] = pdbx.CIFCategory({"id": ["1", "2"], "type": ["polymer", entity_type]})
    if entity_type.strip().lower() == "polymer":
        with pytest.raises(ValueError, match="polymer label_asym_id 'B' has missing label_seq_id"):
            _validate_label_seq_ids(block)
    else:
        _validate_label_seq_ids(block)
