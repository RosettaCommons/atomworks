"""Tests for the AF3 inference pipeline to ensure proper transformation of protein structures."""

import pytest
import torch

from atomworks.io import parse
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
    read_chai_fasta,
)
from atomworks.io.utils.io_utils import to_cif_buffer
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array
from atomworks.io.utils.testing import assert_same_atom_array
from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.utils.testing import cached_parse
from tests.conftest import skip_if_on_github_runner
from tests.ml.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS, TEST_DATA_ML


@skip_if_on_github_runner
def test_af3_confidence_pipeline_from_chai_fasta():
    """Test the AF3 transformation pipeline with confidence feats.

    Tests proper composition of confidence features from input fastas.
    """
    # Load chai fasta
    fasta_path = TEST_DATA_ML / "inference_like_chai_fasta.fasta"
    inference_input_components = read_chai_fasta(fasta_path)
    atom_array = components_to_atom_array(inference_input_components)
    chain_info = initialize_chain_info_from_atom_array(atom_array)

    assert atom_array is not None, "Failed to load atom array from FASTA file"

    # Build and run af3 inference pipeline
    pipeline = build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        run_confidence_head=True,
    )

    transformed_data = pipeline(
        data={
            "example_id": str(fasta_path),
            "atom_array": atom_array,
            "chain_info": chain_info,
        }
    )

    # Basic validation checks
    assert "confidence_feats" in transformed_data, "Missing feats in pipeline output."
    # Check that none of the feats is `nan`
    for feat_name, feat in transformed_data["feats"].items():
        assert (
            feat.isfinite().all() if isinstance(feat, torch.Tensor) else True
        ), f"Found NaN in feats: {feat_name=}, {feat=}"


@skip_if_on_github_runner
def test_af3_pipeline_from_chai_fasta():
    """Test the AF3 transformation pipeline with different configurations.

    Tests loading FASTA files and running them through the AF3 pipeline with different settings
    to ensure proper transformation of protein structures.
    """
    # Load chai fasta
    fasta_path = TEST_DATA_ML / "inference_like_chai_fasta.fasta"
    inference_input_components = read_chai_fasta(fasta_path)
    atom_array = components_to_atom_array(inference_input_components)
    chain_info = initialize_chain_info_from_atom_array(atom_array)

    assert atom_array is not None, "Failed to load atom array from FASTA file"

    # Build and run af3 inference pipeline
    pipeline = build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    )

    transformed_data = pipeline(
        data={
            "example_id": str(fasta_path),
            "atom_array": atom_array,
            "chain_info": chain_info,
        }
    )

    # Basic validation checks
    assert "feats" in transformed_data, "Missing feats in pipeline output."
    # Check that none of the feats is `nan`
    for feat_name, feat in transformed_data["feats"].items():
        assert (
            feat.isfinite().all() if isinstance(feat, torch.Tensor) else True
        ), f"Found NaN in feats: {feat_name=}, {feat=}"


AF3_PIPELINE_FROM_COMPONENTS_TEST_CASES = [
    [
        {
            "seq": "IIGGHEAKPHSRPYMAYLQIMDEYSGSKKCGGFLIREDFVLTAAHCSGSKIQVTLGAHNIKEQEKMQQIIPVVKIIPHPAYNSKTISNDIMLLKLKSKAKRSSAVKPLNLPRRNVKVKPGDVCYVAGWGKLGPMGKYSDTLQEVELTVQEDQKCESYLKNYFDKANEICAGDPKIKRASFRGDSGGPLVCKKVAAGIVSYGQNDGSTPRAFTKVSTFLSWIKKTMKKSIEPD",
            "chain_type": "polypeptide(l)",
            "msa_path": f"{TEST_DATA_ML}/msa_for_inference.a3m",
            "chain_id": "A",
        },
        {
            "smiles": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C",
            "chain_type": "non-polymer",
        },
    ],
    [
        {
            "seq": "MTVDEMVAEAERAEAEGDRERAAELYNEAADKALEEGDVERWTELEVRRADVLERPQVKPYIEEAGEIAKEDPEAARRAWRAMREAAEEARRRREELLAEGMPEEEAEARRLELIREGMDRVAAASDERGRRFVEAIRKAFEALHA",
            "chain_id": "A",
        },
        {"smiles": "ClC1(C(N(O)O)C=C(N(O)O)C=C1)", "chain_id": "B"},
        {"smiles": "CCOC(=O)[CH](C#N)c1ccccc1", "chain_id": "C"},
    ],
]


@skip_if_on_github_runner
@pytest.mark.parametrize("inference_components", AF3_PIPELINE_FROM_COMPONENTS_TEST_CASES)
def test_af3_pipeline_from_sequence_and_smiles(inference_components):
    atom_array, initialized_components = components_to_atom_array(inference_components, return_components=True)
    chain_info = initialize_chain_info_from_atom_array(atom_array)

    # Spoof MSA paths
    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(initialized_components)
    for chain_id, msa_path in msa_paths_by_chain_id.items():
        chain_info[chain_id]["msa_path"] = msa_path

    assert atom_array is not None, "Failed to load atom array from inference components"

    # Build and run af3 inference pipeline
    pipeline = build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    )

    transformed_data = pipeline(
        data={
            "example_id": "test_example",
            "atom_array": atom_array,
            "chain_info": chain_info,
        }
    )

    # Basic validation checks
    assert "feats" in transformed_data, "Missing feats in pipeline output."

    if any("msa_path" in d for d in inference_components):
        assert transformed_data["feats"]["msa_stack"].shape[1] > 1, "MSA stack has only one sequence"

    # Check that none of the feats is `nan`
    for feat_name, feat in transformed_data["feats"].items():
        assert (
            feat.isfinite().all() if isinstance(feat, torch.Tensor) else True
        ), f"Found NaN in feats: {feat_name=}, {feat=}"

    # Check that we successfully generated a reference conformer for the ligand
    assert not torch.any(torch.all(transformed_data["feats"]["ref_pos"] == 0, dim=1))


def test_same_pipeline_outputs_from_cif_and_inference():
    transformation_id = "1"
    data = cached_parse("7rxs", hydrogen_policy="remove")
    atom_array_from_cif = data["assemblies"][transformation_id][0]

    pipeline = build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    )

    # Run the pipeline on the CIF data
    cif_out = pipeline(
        data={
            "example_id": "test_example",
            "atom_array": atom_array_from_cif,
            "chain_info": data["chain_info"],
        }
    )

    # Run the pipeline on the inference components, derived from the CIF data
    monomer = [
        {
            "seq": data["chain_info"]["A"]["unprocessed_entity_non_canonical_sequence"],
            "chain_type": data["chain_info"]["A"]["chain_type"],
            "chain_id": "A",
        }
    ]
    ligand = [{"smiles": "Cc1cc(cc(c1)Oc2nccc(n2)c3c(ncn3[C@H]4CCN(C4)CCN)c5ccc(cc5)I)C", "chain_id": "C"}]
    buffer = to_cif_buffer(components_to_atom_array(monomer + ligand), include_entity_poly=True)
    pipeline_inputs_from_inference = parse(buffer, hydrogen_policy="remove")
    atom_array_from_inference = pipeline_inputs_from_inference["assemblies"][transformation_id][0]

    annotations_to_compare = set(atom_array_from_cif.get_annotation_categories()) - {
        "res_name",
        "atom_name",
        "res_id",
        "stereo",
        "b_factor",
        "alt_atom_id",
        "is_aromatic",
        "occupancy",
    }
    assert_same_atom_array(
        atom_array_from_cif,
        atom_array_from_inference,
        compare_coords=False,
        compare_bonds=True,
        annotations_to_compare=annotations_to_compare,
        enforce_order=False,
    )

    inference_out = pipeline(
        data={
            "example_id": "test_example",
            "atom_array": pipeline_inputs_from_inference["assemblies"][transformation_id][0],
            "chain_info": data["chain_info"],
        }
    )

    # Assert same shapes
    assert cif_out["feats"]["ref_pos"].shape == inference_out["feats"]["ref_pos"].shape

    # Assert no NaNs in ref_pos
    assert torch.isfinite(cif_out["feats"]["ref_pos"]).all()
    assert torch.isfinite(inference_out["feats"]["ref_pos"]).all()


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
