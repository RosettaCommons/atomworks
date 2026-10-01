from atomworks.io.parser import parse
from atomworks.io.utils.catcif import to_catcif_file
from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.testing import assert_same_atom_array_or_stack, get_pdb_path
from tests.conftest import skip_if_no_catcif

# Annotations not reliably preserved in CIF round-trips (same exclusions as test_cif_round_trip):
# - chain_type: derived from entity categories, not written in minimal config
# - atom_id: globally unique counter rewritten on load
# - auth_seq_id: biotite always writes label_seq_id as auth_seq_id, discarding original
_CATCIF_ANNOTATIONS_NOT_PRESERVED = {"chain_type", "atom_id", "auth_seq_id"}


@skip_if_no_catcif
def test_catcif_round_trip(tmp_path):
    """Write two structures to the same catcif file, reload them via parse() and load_any()."""
    pdb_ids = ["6lyz", "1a8o"]
    structures = [parse(get_pdb_path(pid), config="minimal")["asym_unit"][0] for pid in pdb_ids]

    catcif_path = str(tmp_path / "test.catcif")
    catcif_tag_paths = [
        to_catcif_file(structure, catcif_path, tag) for structure, tag in zip(structures, pdb_ids, strict=False)
    ]

    for original, tag_path in zip(structures, catcif_tag_paths, strict=False):
        # parse() round-trip
        loaded_parse = parse(tag_path, config="minimal")["asym_unit"][0]
        annotations_to_compare = [
            f for f in original.get_annotation_categories() if f not in _CATCIF_ANNOTATIONS_NOT_PRESERVED
        ]
        assert_same_atom_array_or_stack(
            original,
            loaded_parse,
            annotations_to_compare=annotations_to_compare,
            compare_bonds=False,
        )

        # load_any() round-trip — compare shared annotations only
        loaded_any = load_any(tag_path, model=1)
        shared_annotations = [
            f
            for f in original.get_annotation_categories()
            if f in loaded_any.get_annotation_categories() and f not in _CATCIF_ANNOTATIONS_NOT_PRESERVED
        ]
        assert_same_atom_array_or_stack(
            original,
            loaded_any,
            annotations_to_compare=shared_annotations,
            compare_bonds=False,
        )
