import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from atomworks.constants import (
    AF3_EXCLUDED_LIGANDS,
    GAP,
)
from atomworks.enums import ChainType
from atomworks.io import parse
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.utils.testing import assert_same_atom_array
from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

example_names = ["1wym", "6w13", "1fix"]


def build_pipelines():
    pipes = {}
    for is_inference in (True, False):
        pipes[is_inference] = build_af3_transform_pipeline(
            is_inference=is_inference,
            protein_msa_dirs=[],
            rna_msa_dirs=[],
            n_recycles=5,
            # Crop params
            crop_size=384,
            crop_center_cutoff_distance=15.0,
            crop_contiguous_probability=0.5,
            crop_spatial_probability=0.5,
            max_atoms_in_crop=None,
            undesired_res_names=AF3_EXCLUDED_LIGANDS,
            conformer_generation_timeout=5.0,  # seconds
            use_element_for_atom_names_of_atomized_tokens=False,
            max_n_template=20,
            n_template=4,
            template_max_seq_similarity=60.0,
            template_min_seq_similarity=10.0,
            template_min_length=10,
            template_allowed_chain_types=[
                ChainType.POLYPEPTIDE_L,
                ChainType.RNA,
            ],
            template_distogram_bins=torch.linspace(3.25, 50.75, 38),
            template_default_token=GAP,
            max_msa_sequences=10_000,
            n_msa=10_000,
            dense_msa=True,
            msa_cache_dir=None,
            sigma_data=16.0,
            diffusion_batch_size=48,
            run_confidence_head=False,
            return_atom_array=True,
            pad_dna_p_skip=0.0,
            b_factor_min=None,
            b_factor_max=None,
        )
    return pipes


pipelines = build_pipelines()


def instantiate_example(example_name: str):
    test_data_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__))), "test_data")
    file = test_data_dir / example_name / f"{example_name}.cif.gz"
    result_dict = parse(
        filename=file,
        build_assembly=("1",),
        **STANDARD_PARSER_ARGS,
    )

    # Only set msa_path if the MSA file exists
    msa_file = test_data_dir / example_name / f"{example_name}.a3m"
    if msa_file.exists():
        for chain_id in result_dict["chain_info"]:
            result_dict["chain_info"][chain_id]["msa_path"] = msa_file

    input = {
        "atom_array": result_dict["assemblies"]["1"][0],  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    return input


def _run_pipeline_test(example_name: str, is_inference: bool) -> dict:
    """Run a single pipeline test and return the result."""

    # Run pipeline with fixed random seed for reproducibility
    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        input_data = instantiate_example(example_name)
        input_data["example_id"] = example_name
        result = pipelines[is_inference](input_data)

    assert result is not None, "Pipeline should return a result"
    return result


def _assert_pipeline_results_equal(result: dict, expected: dict, example_name: str, is_inference: bool):
    """Assert that two pipeline results are equal."""
    # Check that both have the same keys
    # missing_keys = set(expected.keys()) - set(result.keys())
    mode = f"{'inference' if is_inference else 'training'}"
    missing_keys = set(expected.keys()) - set(result.keys()) - {"extra_info", "atom_array"}
    assert not missing_keys, f"Missing feature keys {missing_keys} for {example_name} in {mode} mode"

    # Check atom array if present
    if is_inference:
        assert "atom_array" in result, "Atom array not found in result"
        assert_same_atom_array(
            result["atom_array"],
            expected["atom_array"],
            compare_coords=True,
            compare_bonds=True,
            # (All annotation categories present in the expected atom array are compared)
            annotations_to_compare=expected["atom_array"].get_annotation_categories(),
        )

    # Check features
    assert "feats" in result, "Features not found in result"
    _assert_features_equal(result["feats"], expected["feats"], example_name, mode)


def _assert_features_equal(feats: dict, expected_feats: dict, example_name: str, mode: str):
    """Assert that feature dictionaries are equal, with new features being a superset of old features."""
    # Check that all expected feature keys are present in the new features
    missing_keys = set(expected_feats.keys()) - set(feats.keys())
    assert not missing_keys, f"Missing feature keys {missing_keys} for {example_name} in {mode} mode"

    # Only check features that were in the expected results (allows for new features)
    for key in expected_feats:
        if key == "ref_pos":
            # rdkit versions seem to be compiled differently on different operating systems, making
            #  this an operating system dependent test. Instead of the values, we just check the
            #  shapes
            assert feats[key].shape == expected_feats[key].shape, (
                f"Feature {key} shape mismatch for {example_name} in {mode} mode: {feats[key].shape} vs {expected_feats[key].shape}"
            )
            continue
        feat = feats[key]
        expected_feat = expected_feats[key]

        # Check shapes
        assert feat.shape == expected_feat.shape, (
            f"Feature {key} shape mismatch for {example_name} in {mode} mode: {feat.shape} vs {expected_feat.shape}"
        )

        # Check values with tolerance
        _assert_tensor_or_array_equal(
            feat,
            expected_feat,
            f"Feature {key} values don't match for {example_name} in {mode} mode",
        )


def _assert_tensor_or_array_equal(actual, expected, error_msg: str):
    """Assert that two tensors or arrays are equal, with appropriate tolerance for different dtypes."""
    if torch.is_tensor(actual):
        if actual.dtype == torch.bool or actual.dtype in [torch.int32, torch.int64]:
            torch.testing.assert_close(
                actual, expected, atol=0, rtol=0, equal_nan=True, msg=lambda x: error_msg + ": " + x
            )
        else:
            torch.testing.assert_close(
                actual, expected, atol=1e-4, rtol=1e-4, equal_nan=True, msg=lambda x: error_msg + ": " + x
            )
    elif isinstance(actual, np.ndarray):
        if (
            actual.dtype.kind in ["U", "S"] or actual.dtype == bool or np.issubdtype(actual.dtype, np.integer)
        ):  # String dtypes
            assert np.testing.assert_array_equal(actual, expected, err_msg=error_msg)
        else:
            assert np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4, equal_nan=True, err_msg=error_msg)
    else:
        assert actual == expected, error_msg


def _make_test_identifier(example_name: str, is_inference: bool) -> str:
    """Generate a test identifier for parametrized tests and file paths."""
    mode = "inference" if is_inference else "train"
    return f"{example_name}-{mode}"


def _get_regression_data_path(example_name: str, is_inference: bool) -> Path:
    """Get the path for regression test data based on the example name and inference mode."""
    regression_dir = Path(__file__).parent / "regression_test_data"
    regression_dir.mkdir(parents=True, exist_ok=True)

    # Use shared identifier logic for consistent naming
    identifier = _make_test_identifier(example_name, is_inference)
    # Convert to file-friendly format (replace hyphens with underscores)
    file_name = identifier.replace("-", "_")
    return regression_dir / f"{file_name}.pkl"


@pytest.mark.parametrize(
    "example_name,is_inference",
    [
        pytest.param(
            example_name,
            is_inference,
            id=_make_test_identifier(example_name, is_inference),
        )
        for example_name in example_names
        for is_inference in (True, False)
    ],
)
def test_af3_pipeline_regression(example_name: str, is_inference: bool):
    """Test the AF3 pipeline against stored regression results for various configurations."""

    # Run the pipeline test
    result = _run_pipeline_test(example_name, is_inference)

    # Get regression data path using shared logic
    regression_path = _get_regression_data_path(example_name, is_inference)

    # # Uncomment the following lines to create/update the regression data
    # with regression_path.open("wb") as f:
    #     pickle.dump(result, f)
    #     logger.info(f"Saved regression data to {regression_path}")

    # Load expected result
    with regression_path.open("rb") as f:
        expected_result = pickle.load(f)

    # Compare results
    _assert_pipeline_results_equal(result, expected_result, example_name, is_inference)
