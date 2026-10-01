"""Round-trip tests for variable-length multi-model CIF writing and reading."""

import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import read_any, to_cif_file
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.utils.testing import get_pdb_mirror_path

# Re-use the SA helper from the existing SA test module
from tests.ml.standard_annotations.test_standard_annotations import add_randomized_standard_annotations

# Annotations excluded from round-trip comparison (consistent with test_io.py::test_cif_round_trip):
# - chain_type: derived from entity categories (_entity, _entity_poly); omitted in
#   multi-model CIF as they may vary among models.
# - atom_id: _atom_site.id is a globally unique counter across all models in the file
#   (standard mmCIF practice, matching PDB NMR structures). Per-model values are not preserved.
# - auth_seq_id: biotite's set_structure always writes _atom_site.auth_seq_id = res_id
#   (label_seq_id), discarding the original author numbering. Same limitation applies to
#   single-model round-trips; excluded in test_io.py for the same reason.
_ANNOTATIONS_NOT_PRESERVED = {"chain_type", "atom_id", "auth_seq_id"}


@pytest.mark.parametrize("n_models", [2, 3])
@pytest.mark.parametrize("file_type", ["cif", "cif.gz", "cif.gzip", "cif.zst"])
def test_multi_model_cif_round_trip(n_models, tmp_path, file_type):
    """Save a list of AtomArrays with different lengths to CIF and reload each model.

    Uses non-default model_ids to verify that custom IDs are written to the file
    and that models are returned in the correct order on load.
    """
    pdb_ids = ["6lyz", "1a8o", "2e2h"][:n_models]
    model_ids = [5, 12, 99][:n_models]
    arrays = [as_atom_array_plus(parse(get_pdb_mirror_path(pid), config="minimal")["asym_unit"][0]) for pid in pdb_ids]
    lengths = [arr.array_length() for arr in arrays]
    assert len(set(lengths)) > 1, "Test structures must have different atom counts"

    arrays = [add_randomized_standard_annotations(arr) for arr in arrays]

    path = tmp_path / ("multi_model." + file_type)
    to_cif_file(arrays, path, model_ids=model_ids, save_standard_annotations=True)

    # Verify the custom model_ids are written in the correct order
    cif_file = read_any(path)
    block = cif_file[list(cif_file.keys())[0]]
    written_model_nums = list(dict.fromkeys(block["atom_site"]["pdbx_PDB_model_num"].as_array(int).tolist()))
    assert written_model_nums == model_ids

    results = parse(
        path, config=ParseConfig.from_preset("minimal", load_standard_annotations=True, return_atom_array_plus=True)
    )
    assert isinstance(results, list), "parse() must return list[dict] for variable-length multi-model CIF"
    assert len(results) == n_models

    for original, result in zip(arrays, results, strict=False):
        loaded = result["asym_unit"][0]
        annotations_to_compare = [
            f for f in original.get_annotation_categories() if f not in _ANNOTATIONS_NOT_PRESERVED
        ]
        assert_same_atom_array_or_stack(
            original,
            loaded,
            annotations_to_compare=annotations_to_compare,
            compare_bonds=False,
        )
