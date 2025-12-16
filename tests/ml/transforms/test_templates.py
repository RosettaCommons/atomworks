import copy

import numpy as np
import pytest
import torch

from atomworks.ml.encoding_definitions import RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    get_chain_instance_starts,
)
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.encoding import AF3SequenceEncoding, EncodeAtomArray, TokenEncoding
from atomworks.ml.transforms.filters import FilterToProteins, RemoveHydrogens, RemoveTerminalOxygen
from atomworks.ml.transforms.template import (
    AddInputFileTemplate,
    AddRFTemplates,
    FeaturizeTemplatesLikeAF3,
    FeaturizeTemplatesLikeRF2AA,
    RandomSubsampleTemplates,
    RF2AATemplate,
    add_input_file_template,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from tests.ml.conftest import TEMPLATE_DIR, TEMPLATE_LOOKUP

TEST_CASES = [
    {
        "pdb_id": "5ocm",  # multi-chain template
        "n_templates": {"A": 446, "B": 446, "C": 446, "D": 446, "E": 446, "F": 446},
    },
    {
        "pdb_id": "6lyz",  # single-chain template
        "n_templates": {"A": 383},
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_rf_templates(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    data = AddRFTemplates(
        max_n_template=1000,
        pick_top=False,
        min_seq_similarity=10,
        max_seq_similarity=100,
        min_template_length=10,
        template_lookup_path=TEMPLATE_LOOKUP,
        template_base_dir=TEMPLATE_DIR,
    )(data)

    for chain, n_templates in test_case["n_templates"].items():
        assert (
            len(data["template"][chain]) == n_templates
        ), f"For {pdb_id}-{chain}: Expected {n_templates} templates, got {len(data['template'][chain])}"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_subsample_template(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    # Create a list of the number of templates for each chain
    template_counts = []

    out_before_subsampling = AddRFTemplates(
        max_n_template=20,
        pick_top=False,
        min_seq_similarity=10,
        max_seq_similarity=100,
        min_template_length=10,
        template_lookup_path=TEMPLATE_LOOKUP,
        template_base_dir=TEMPLATE_DIR,
    )(data)

    with rng_state(create_rng_state_from_seeds(12345)):
        for _ in range(100):
            # Sample 10 times
            out_after_subsampling = RandomSubsampleTemplates(n_template=10)(copy.deepcopy(out_before_subsampling))
            template_counts.extend(len(templates) for templates in out_after_subsampling["template"].values())

    # Assert that at least one template has < n_template
    assert any(count < 10 for count in template_counts), "Expected at least one template to have less than 10 templates"

    # Assert that no template has > n_template
    assert all(count <= 10 for count in template_counts), "Expected no template to have more than 10 templates"

    # Assert the mean is what we would expect (~7.5 = 0.5 * 10 + 0.5 * 5)
    assert (
        7.5 - 1 < sum(template_counts) / len(template_counts) < 7.5 + 1
    ), f"Expected mean to be around 7.5. Found {sum(template_counts) / len(template_counts)}. This is a stochastic test, so running again may fix this."


@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("min_seq_similarity, max_seq_similarity", [(30, 55), (20, 50), (10, 80)])
@pytest.mark.parametrize("min_template_length", [10, 60])
def test_add_rf_templates_filters(
    test_case: dict, min_seq_similarity: int, max_seq_similarity: int, min_template_length: int
):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    transform = AddRFTemplates(
        max_n_template=5,
        pick_top=True,
        min_seq_similarity=min_seq_similarity,
        max_seq_similarity=max_seq_similarity,
        min_template_length=min_template_length,
        template_lookup_path=TEMPLATE_LOOKUP,
        template_base_dir=TEMPLATE_DIR,
    )
    data = transform(data)

    for chain, templates in data["template"].items():
        assert len(templates) > 0, f"No templates found for {pdb_id}-{chain}"
        assert len(templates) <= 5, f"Expected 5 templates, got {len(templates)} for {pdb_id}-{chain}"
        for template in templates:
            assert (
                min_seq_similarity <= template["seq_similarity"] <= max_seq_similarity
            ), f"Expected seq similarity between {min_seq_similarity} and {max_seq_similarity}, got {template['seq_similarity']} for {pdb_id}-{chain}-{template['id']}"
            assert (
                template["n_res"] >= min_template_length
            ), f"Expected at least {min_template_length} residues, got {template['n_res']} for {pdb_id}-{chain}-{template['id']}"


@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("encoding", [RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING])
def test_featurize_rf_templates(test_case: dict, encoding: TokenEncoding, n_template: int = 3):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    pipe = Compose(
        [
            RemoveHydrogens(),
            RemoveTerminalOxygen(),
            AddWithinPolyResIdxAnnotation(),
            FilterToProteins(),
            AddGlobalTokenIdAnnotation(),
            AddRFTemplates(
                max_n_template=2,
                pick_top=False,
                min_seq_similarity=20,
                max_seq_similarity=60,
                min_template_length=10,
                template_lookup_path=TEMPLATE_LOOKUP,
                template_base_dir=TEMPLATE_DIR,
            ),
            FeaturizeTemplatesLikeRF2AA(
                n_template=n_template,
                encoding=encoding,
                mask_token_idx=21,
                init_coords=RF2AATemplate.RF2AA_INIT_TEMPLATE_COORDINATES,
            ),
            EncodeAtomArray(encoding=encoding),
        ],
        track_rng_state=False,
    )
    with rng_state(create_rng_state_from_seeds(12345)):
        data = pipe(data)

    atom_array = data["atom_array"]
    n_tokens = len(atom_array[atom_array.atom_name == "CA"])
    len_token_vocabulary = len(encoding.token_atoms)

    xyz_encoded = data["encoded"]["xyz"]
    mask_encoded = data["encoded"]["mask"]
    seq_encoded = torch.nn.functional.one_hot(torch.tensor(data["encoded"]["seq"]), encoding.n_tokens)

    xyz_template = data["template_feat"]["xyz"]
    mask_template = data["template_feat"]["mask"]
    t1d_template = data["template_feat"]["t1d"]

    # Check the template features are of the correct shape
    assert xyz_template.shape == (n_template, n_tokens, 36, 3)
    assert mask_template.shape == (n_template, n_tokens, 36)
    assert t1d_template.shape == (
        n_template,
        n_tokens,
        len_token_vocabulary - 1 + 1,
    )  # -1 for removing mask, +1 for alignment confidence

    assert xyz_template[0].shape == xyz_encoded.shape
    assert mask_template[0].shape == mask_encoded.shape
    assert t1d_template[0].shape == seq_encoded.shape

    # Check that the t1d last axis adds up to one when excluding the last dimension
    assert torch.all(
        t1d_template[..., :-1].sum(dim=-1) == 1
    ), f"Expected t1d last axis to add up to one (one-hot encoded), but got {t1d_template[..., :-1].sum(dim=-1)}"

    # Check that alignment confidences exist for all non-masked tokens
    is_masked = t1d_template[..., 21] == 1
    assert torch.all(
        t1d_template[..., -1][~is_masked] > 0
    ), "Expected t1d last axis to be greater than zero (alignment confidence)"

    # Check that all not-masked tokens have finite coordinates
    assert torch.all(torch.isfinite(xyz_template[mask_template])), "Expected non-masked template xyz to be finite"

    # Ensure that at least something was filled in
    assert torch.any(mask_template), "Expected at least one token to be filled in"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_input_file_template(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    data = AddRFTemplates(
        max_n_template=1000,
        pick_top=False,
        min_seq_similarity=10,
        max_seq_similarity=100,
        min_template_length=10,
        template_lookup_path=TEMPLATE_LOOKUP,
        template_base_dir=TEMPLATE_DIR,
    )(data)

    # first modify the atom_array to have the is_input_file_template annotation
    # get a mask that is true for all atoms in the first polymer chain
    atom_array = data["atom_array"]
    atom_array.set_annotation("is_input_file_templated", np.zeros(len(atom_array), dtype=bool))
    chain_starts = get_chain_instance_starts(atom_array)
    chain_ends = chain_starts[1:]
    chain_ends = np.append(chain_ends, len(atom_array))
    chosen_chain_to_template = None
    for start, end in zip(chain_starts, chain_ends, strict=False):
        if atom_array.is_polymer[start]:
            atom_array.is_input_file_templated[start:end] = True
            chosen_chain_to_template = atom_array.chain_id[start]
            break

    input_file_template = add_input_file_template(atom_array)
    training_pipeline_template = data["template"]

    # check that the template has the same fields as the template from AddRFTemplates
    assert (
        set(input_file_template[chosen_chain_to_template][0].keys())
        == training_pipeline_template[chosen_chain_to_template][0].keys()
    ), f"Expected input file template to have the same keys as the training pipeline template, but got {set(input_file_template[chosen_chain_to_template][0].keys())} and {set(training_pipeline_template[chosen_chain_to_template][0].keys())}"

    # check that the template has the same fields as the template from AddRFTemplates
    required_annotations = [
        "aligned_query_res_idx",
        "alignment_confidence",
    ]
    input_file_template_annotations = input_file_template[chosen_chain_to_template][0][
        "atom_array"
    ].get_annotation_categories()
    training_pipeline_template_annotations = training_pipeline_template[chosen_chain_to_template][0][
        "atom_array"
    ].get_annotation_categories()
    assert set(
        required_annotations
    ).issubset(
        input_file_template_annotations
    ), f"Input file template is missing the following annotations: {set(required_annotations) - set(input_file_template_annotations)}"
    assert set(
        required_annotations
    ).issubset(
        training_pipeline_template_annotations
    ), f"Training pipeline template is missing the following annotations: {set(required_annotations) - set(training_pipeline_template_annotations)}"

    # check that one template has added for the first polymer chain
    assert (
        len(input_file_template[chosen_chain_to_template]) == 1
    ), f"Expected input file template to have one template, but got {len(input_file_template[chosen_chain_to_template])}"

    assert np.all(
        input_file_template[chosen_chain_to_template][0]["atom_array"].alignment_confidence == 1
    ), f"Expected input file template to have template confidence of 1, but got {input_file_template[chosen_chain_to_template][0]['atom_array'].alignment_confidence}"

    assert np.all(
        input_file_template[chosen_chain_to_template][0]["atom_array"].aligned_query_res_idx
        == input_file_template[chosen_chain_to_template][0]["atom_array"].res_id
    ), f"Expected input file template to have aligned query res idx equal to res id, but got {input_file_template[chosen_chain_to_template][0]['atom_array'].aligned_query_res_idx} and {input_file_template[chosen_chain_to_template][0]['atom_array'].res_id}"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_featurize_input_templates(test_case):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    af3_sequence_encoding = AF3SequenceEncoding()
    # first modify the atom_array to have the is_input_file_template annotation
    # get a mask that is true for all atoms in the first polymer chain
    atom_array = data["atom_array"]
    atom_array.set_annotation("is_input_file_templated", np.zeros(len(atom_array), dtype=bool))
    chain_starts = get_chain_instance_starts(atom_array)
    chain_ends = chain_starts[1:]
    chain_ends = np.append(chain_ends, len(atom_array))
    for start, end in zip(chain_starts, chain_ends, strict=False):
        if atom_array.is_polymer[start]:
            atom_array.is_input_file_templated[start:end] = True
            break
    pipeline = Compose(
        [
            AddWithinChainInstanceResIdx(),
            AddGlobalTokenIdAnnotation(),
            AddInputFileTemplate(),
            FeaturizeTemplatesLikeAF3(
                sequence_encoding=af3_sequence_encoding,
            ),
        ]
    )
    data = pipeline(data)
    expected_template_features = [
        "template_restype",
        "template_pseudo_beta_mask",
        "template_backbone_frame_mask",
        "template_distogram",
        "template_unit_vector",
    ]
    assert set(expected_template_features).issubset(
        data["feats"].keys()
    ), f"Expected template features to be present, but got {set(data['feats'].keys())} instead"
