import time

import biotite.structure as struc
import numpy as np
import pytest
import torch

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.enums import ChainType, GroundTruthConformerPolicy
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.af3_reference_molecule import (
    GetAF3ReferenceMoleculeFeatures,
    RandomApplyGroundTruthConformerByChainType,
    _encode_atom_names_like_af3,
    _get_rdkit_mols_with_conformers,
    _map_reference_conformer_to_residue,
    get_af3_reference_molecule_features,
)
from atomworks.ml.transforms.atom_array import AddGlobalResIdAnnotation, add_global_token_id_annotation
from atomworks.ml.transforms.atomize import atomize_by_ccd_name
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData, RandomSubsampleCachedConformers
from atomworks.ml.transforms.chirals import AddAF3ChiralFeatures
from atomworks.ml.transforms.rdkit_utils import GetRDKitChiralCenters
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from tests.ml.conftest import TEST_DATA_ML


def test_contrived_tyr():
    """Test _map_reference_conformer_to_residue functionality with a contrived TYR residue."""
    # Create general residue
    orig = struc.info.residue("TYR")
    orig = orig[orig.atom_name != "OXT"]
    orig = orig[orig.element != "H"]
    orig[np.array([5, 6])] = orig[np.array([6, 5])]  # swap two atoms
    orig = add_global_token_id_annotation(orig)
    # Create reference molecule
    conformer = struc.info.residue("TYR")

    # Map reference molecule to residue
    ref_pos, ref_mask = _map_reference_conformer_to_residue(
        res_name="TYR",
        atom_names=orig.atom_name,
        conformer=conformer,
    )

    # Check that the reference molecule is correctly mapped to the residue
    assert ref_pos.shape == (len(orig), 3), f"{ref_pos.shape=} should be ({len(orig)}, 3)"
    assert ref_mask.shape == (len(orig),), f"{ref_mask.shape=} should be ({len(orig)},)"

    assert np.allclose(ref_pos, orig.coord), f"{ref_pos=} should be {orig.coord=}"
    assert np.allclose(ref_mask, True)


@pytest.mark.parametrize(
    "res_name",
    [
        "TYR",
        "ALA",
        "GLY",
        "PHE",
        "PRO",
        "VAL",
        "CYS",
        "LEU",
        "MET",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
        "SER",
        "THR",
        "ASN",
        "GLN",
        "HIS",
        "TRP",
        "UNK",
        "R2R",
    ],
)
def test_get_af3_reference_molecule_features_res(res_name):
    atom_array = struc.info.residue(res_name)
    atom_array = atom_array[atom_array.atom_name != "OXT"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = add_global_token_id_annotation(atom_array)

    n_atom = len(atom_array)

    # Check feature shapes
    features, _ = get_af3_reference_molecule_features(atom_array)

    assert features["ref_pos"].shape == (n_atom, 3)
    assert features["ref_mask"].shape == (n_atom,)
    assert features["ref_element"].shape == (n_atom,)
    assert features["ref_charge"].shape == (n_atom,)
    assert features["ref_atom_name_chars"].shape == (n_atom, 4)


def test_get_af3_reference_molecule_features_chain():
    atom_array = struc.info.residue("ALA") + struc.info.residue("R2R") + struc.info.residue("TYR")
    # Add the necessary annotations from `parse`
    atom_array = atom_array[atom_array.atom_name != "OXT"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = add_global_token_id_annotation(atom_array)

    # We atomize so that we can test using the element for atom names of atomized tokens
    atom_array = atomize_by_ccd_name(atom_array, res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA)

    n_atoms = len(atom_array)

    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        features, _ = get_af3_reference_molecule_features(atom_array, apply_random_rotation_and_translation=False)
        features_with_elements_for_atomized_atom_names, _ = get_af3_reference_molecule_features(
            atom_array, apply_random_rotation_and_translation=False, use_element_for_atom_names_of_atomized_tokens=True
        )

    assert "ref_pos" in features
    assert "ref_mask" in features
    assert "ref_element" in features
    assert "ref_charge" in features
    assert "ref_atom_name_chars" in features

    # ... check that the atom name features are the same for non-atomized tokens
    assert np.all(
        features["ref_atom_name_chars"][~atom_array.atomize]
        == features_with_elements_for_atomized_atom_names["ref_atom_name_chars"][~atom_array.atomize]
    )

    # ... check that the atom name features for atomized tokens are encoded correctly, when indicated
    encoded_elements = _encode_atom_names_like_af3(atom_array.element)
    assert np.all(
        features_with_elements_for_atomized_atom_names["ref_atom_name_chars"][atom_array.atomize]
        == encoded_elements[atom_array.atomize]
    )

    assert features["ref_pos"].shape == (n_atoms, 3)
    assert features["ref_mask"].shape == (n_atoms,)
    assert features["ref_element"].shape == (n_atoms,)
    assert features["ref_charge"].shape == (n_atoms,)
    assert features["ref_atom_name_chars"].shape == (n_atoms, 4)

    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        features_with_random_rototranslation, _ = get_af3_reference_molecule_features(
            atom_array, apply_random_rotation_and_translation=True
        )

        # Assert that the features are different
        assert not np.allclose(features["ref_pos"], features_with_random_rototranslation["ref_pos"])
        assert np.allclose(features["ref_mask"], features_with_random_rototranslation["ref_mask"])


def test_reference_conformer_generation_for_two_molecules_only_differing_by_transformation_id():
    # fmt: off
    atom_array = struc.array([
        struc.Atom(np.array([44.869,     8.188,    36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  atomic_number=7,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([45.024,     7.456,    34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([44.142,     6.714,    34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", atomic_number=8,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.669,     8.171,    36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.812,     8.982,    38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.152,     8.296,    39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.479,     9.3  ,    40.792 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="SD", atomic_number=16, charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.232,     8.184,    42.102 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CE", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.46 ,     8.724,    36.151 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="C",  atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.339,     9.907,    35.831 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O",  atomic_number=8,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([58.656483, 34.763695, 36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  atomic_number=7,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.212917, 35.263927, 34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([60.296505, 34.87109 , 34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", atomic_number=8,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.271206, 33.732964, 36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([58.49736 , 33.451305, 38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.42145 , 33.22273 , 39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", atomic_number=6,  charge=0,  transformation_id="2"),
    ])
    # fmt: on

    assert len(get_residue_starts(atom_array)) == 2
    atom_array.set_annotation("token_id", np.arange(len(atom_array)))
    features, _ = get_af3_reference_molecule_features(atom_array)
    assert len(features) > 0, "Expected features to be non-empty"


def _assert_ref_pos_matches_ground_truth(
    ground_truth_coord: np.ndarray,
    ref_pos: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Assert that the reference positions are correctly aligned with the ground truth."""
    assert not np.any(np.isclose(ref_pos, ground_truth_coord)), (
        "Reference positions should differ from atom array coordinates under a transformation."
    )

    # Assert similar distances in masked positions
    dist1 = np.linalg.norm(ground_truth_coord[mask][:, None] - ground_truth_coord[mask], axis=2)
    dist2 = np.linalg.norm(ref_pos[mask][:, None] - ref_pos[mask], axis=2)
    assert np.allclose(dist1, dist2), "Distances should be similar for masked positions regardless of transformation."


def test_replace_conformer_with_ground_truth():
    """Test that the ground truth conformer is used when indicated"""
    atom_array = struc.info.residue("HEM")
    atom_array = atom_array[atom_array.element != "H"]
    atom_array.set_annotation(
        "ground_truth_conformer_policy", np.full(len(atom_array), GroundTruthConformerPolicy.REPLACE)
    )

    features, _ = get_af3_reference_molecule_features(atom_array, apply_random_rotation_and_translation=True)

    _assert_ref_pos_matches_ground_truth(
        ground_truth_coord=atom_array.coord,
        ref_pos=features["ref_pos"],
        mask=features["ref_mask"],
    )


def test_random_apply_ground_truth_conformer_by_chain_type(seed: int = 42):
    """Test that we can randomly flag non-polymers to use the ground truth conformer"""
    pdb_id = "5ocm"
    data = cached_parse(pdb_id, hydrogen_policy="remove")

    # Define probabilities for different chain types
    chain_type_probabilities = {
        tuple(ChainType.get_non_polymers()): 0.8,
    }

    pipe = Compose(
        [
            RandomApplyGroundTruthConformerByChainType(
                chain_type_probabilities=chain_type_probabilities,
                default_probability=0.0,
                policy=GroundTruthConformerPolicy.REPLACE,
            ),
            GetAF3ReferenceMoleculeFeatures(apply_random_rotation_and_translation=True),
        ]
    )

    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        out = pipe(data)

    feats = out["feats"]
    atom_array = out["atom_array"]

    # Assert that all polymer atoms have the IGNORE policy
    polymer_mask = atom_array.is_polymer
    assert np.all(atom_array.ground_truth_conformer_policy[polymer_mask] == GroundTruthConformerPolicy.IGNORE), (
        f"Expected all polymer atoms to have IGNORE policy, but got {atom_array.ground_truth_conformer_policy[polymer_mask]}"
    )

    # Assert that most non-polymer atoms have the REPLACE policy...
    non_polymer_mask = ~polymer_mask
    assert np.sum(
        atom_array.ground_truth_conformer_policy[non_polymer_mask] == GroundTruthConformerPolicy.REPLACE
    ) > 0.5 * np.sum(non_polymer_mask), (
        f"Expected most non-polymer atoms to have REPLACE policy, but got {np.sum(atom_array.ground_truth_conformer_policy[non_polymer_mask] == GroundTruthConformerPolicy.REPLACE)}"
    )
    # ... but not all
    assert np.any(atom_array.ground_truth_conformer_policy[non_polymer_mask] == GroundTruthConformerPolicy.IGNORE)

    # (Get the start and stop indices of each residue)
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]

    for i, residue in enumerate(struc.residue_iter(atom_array)):
        # ... check that the GroundTruthConformerPolicy is the same for all atoms in the residue
        assert np.all(residue.ground_truth_conformer_policy == residue.ground_truth_conformer_policy[0])

        if residue.ground_truth_conformer_policy[0] == GroundTruthConformerPolicy.REPLACE:
            # ... check that the reference positions are aligned with the ground truth
            _assert_ref_pos_matches_ground_truth(
                ground_truth_coord=residue.coord,
                ref_pos=feats["ref_pos"][_res_starts[i] : _res_ends[i]],
                mask=feats["ref_mask"][_res_starts[i] : _res_ends[i]],
            )


def test_fallback_to_ground_truth_conformer_on_error():
    """Test that we fallback to the ground truth conformer when an error occurs during RDKit conformer generation and the residue is not in the CCD"""
    atom_array = components_to_atom_array([{"path": f"{TEST_DATA_ML}/example_sdf.sdf"}])
    atom_array.set_annotation(
        "ground_truth_conformer_policy", np.full(len(atom_array), GroundTruthConformerPolicy.FALLBACK)
    )

    pipe = Compose(
        [
            GetAF3ReferenceMoleculeFeatures(
                apply_random_rotation_and_translation=True,
                conformer_generation_timeout=0.0,  # Force a timeout
            ),
        ]
    )
    out = pipe({"atom_array": atom_array})

    # Ensure that the ground truth conformer is used
    assert np.all(out["feats"]["ref_pos_is_ground_truth"])
    _assert_ref_pos_matches_ground_truth(
        ground_truth_coord=atom_array.coord,
        ref_pos=out["feats"]["ref_pos"],
        mask=torch.ones(len(atom_array), dtype=torch.bool),
    )


def test_add_ground_truth_conformer_as_feature():
    """Test that we can add the ground truth conformer as an additional feature"""
    atom_array = struc.info.residue("HEM")
    atom_array = atom_array[atom_array.element != "H"]

    # Set policy to ADD for all atoms
    atom_array.set_annotation("ground_truth_conformer_policy", np.full(len(atom_array), GroundTruthConformerPolicy.ADD))

    features, _ = get_af3_reference_molecule_features(atom_array, apply_random_rotation_and_translation=False)

    # Verify that ref_pos_ground_truth exists and contains the ground truth coordinates
    assert "ref_pos_ground_truth" in features

    # The reference positions should NOT be the ground truth (since we used ADD not REPLACE)...
    assert not np.any(features["ref_pos_is_ground_truth"])

    _assert_ref_pos_matches_ground_truth(
        ground_truth_coord=atom_array.coord,
        ref_pos=features["ref_pos_ground_truth"],
        mask=features["ref_mask"],
    )


def test_ref_space_uid():
    def _prepare(res_name: str, chain_id: str, res_id: int, transformation_id: str) -> struc.AtomArray:
        res = struc.info.residue(res_name)
        res = res[res.atom_name != "OXT"]
        res = res[res.element != "H"]
        res.chain_id[:] = chain_id
        res.res_id[:] = res_id
        res.set_annotation("transformation_id", [transformation_id] * len(res))
        return res

    ala1 = _prepare("ALA", "A", 1, "1")
    ala2 = _prepare("ALA", "A", 1, "2")
    ala3 = _prepare("ALA", "A", 1, "3")
    tyr1 = _prepare("TYR", "B", 1, "1")
    eoh1 = _prepare("EOH", "C", 1, "1")
    eoh2 = _prepare("EOH", "C", 1, "2")
    atom_array = ala1 + tyr1 + ala2 + ala3 + eoh1 + eoh2

    # fmt: off
    expected_ref_space_uid = np.array(
        [0] * len(ala1) +
        [1] * len(tyr1) +
        [2] * len(ala2) +
        [3] * len(ala3) +
        [4] * len(eoh1) +
        [5] * len(eoh2)
    )
    # fmt: on

    # Run mini-pipeline
    pipe = GetAF3ReferenceMoleculeFeatures(apply_random_rotation_and_translation=True)
    out = pipe({"atom_array": atom_array})

    assert np.all(out["feats"]["ref_space_uid"] == expected_ref_space_uid), (
        f"{out['feats']['ref_space_uid']}, but expected {expected_ref_space_uid}"
    )


def test_max_conformers_per_residue_functionality():
    """Test that max_conformers_per_residue actually limits conformer generation."""
    # Test with a stoichiometry that would normally generate many conformers
    test_stoichiometry = {
        "ALA": 10,  # Would normally generate 10 conformers
        "VAL": 8,  # Would normally generate 8 conformers
    }

    # Generate with no limit
    ref_mols_no_limit = _get_rdkit_mols_with_conformers(
        res_stochiometry=test_stoichiometry, max_conformers_per_residue=None, timeout=(1.0, 0.1)
    )

    # Generate with limit of 3
    ref_mols_with_limit = _get_rdkit_mols_with_conformers(
        res_stochiometry=test_stoichiometry, max_conformers_per_residue=3, timeout=(1.0, 0.1)
    )

    # Check that limits were applied
    if ref_mols_no_limit["ALA"] and ref_mols_with_limit["ALA"]:
        ala_conformers_no_limit = ref_mols_no_limit["ALA"].GetNumConformers()
        ala_conformers_with_limit = ref_mols_with_limit["ALA"].GetNumConformers()

        assert ala_conformers_no_limit >= ala_conformers_with_limit
        assert ala_conformers_with_limit <= 3

    if ref_mols_no_limit["VAL"] and ref_mols_with_limit["VAL"]:
        val_conformers_no_limit = ref_mols_no_limit["VAL"].GetNumConformers()
        val_conformers_with_limit = ref_mols_with_limit["VAL"].GetNumConformers()

        assert val_conformers_no_limit >= val_conformers_with_limit
        assert val_conformers_with_limit <= 3


def test_af3_reference_molecule_features_with_cached_conformers(cache_dir):
    """Test AF3 reference molecule features using cached conformers."""
    data = cached_parse("1crn", hydrogen_policy="remove")
    pipe = Compose(
        [
            AddGlobalResIdAnnotation(),
            LoadCachedResidueLevelData(dir=cache_dir, sharding_depth=1),
        ]
    )
    cached_residue_data = pipe(data)

    # Create transform with cached conformers enabled and max conformers limit
    transform_with_cache = GetAF3ReferenceMoleculeFeatures(
        max_conformers_per_residue=3, use_cached_conformers=True, save_rdkit_mols=True
    )
    transform_no_cache = GetAF3ReferenceMoleculeFeatures(
        max_conformers_per_residue=3, use_cached_conformers=False, conformer_generation_timeout=5.0
    )
    data_no_cache = {"atom_array": cached_residue_data["atom_array"]}

    # ... time the cached version
    start_time = time.time()
    result_data_cached = transform_with_cache(cached_residue_data)
    cached_time = time.time() - start_time

    # ... time the non-cached version
    start_time = time.time()
    _ = transform_no_cache(data_no_cache)
    no_cache_time = time.time() - start_time

    assert cached_time < 0.8 * no_cache_time, (
        f"Cached version should be faster than no cache version, but got {cached_time} vs {no_cache_time}"
    )

    feats = result_data_cached["feats"]
    assert not np.any(np.isnan(feats["ref_pos"]))
    assert not np.any(np.all(feats["ref_pos"] == 0, axis=1))


@pytest.fixture
def data_with_subsampled_conformers(cache_dir):
    """Fixture providing AF3 reference molecule features with subsampled conformers."""
    data = cached_parse("1crn", hydrogen_policy="remove")
    pipeline = Compose(
        [
            AddGlobalResIdAnnotation(),
            LoadCachedResidueLevelData(dir=cache_dir, sharding_depth=1),
            RandomSubsampleCachedConformers(n_conformers=3, seed=42),
            GetAF3ReferenceMoleculeFeatures(
                max_conformers_per_residue=5, use_cached_conformers=True, apply_random_rotation_and_translation=False
            ),
        ]
    )
    return pipeline(data)


def test_af3_reference_molecule_features_with_subsampled_conformers(data_with_subsampled_conformers):
    """Ensure that the actual reference molecules at each res_idx match"""
    result_data = data_with_subsampled_conformers

    atom_array = result_data["atom_array"]
    feats = result_data["feats"]
    cached_residue_level_data = result_data["cached_residue_level_data"]["residues"]
    conformer_indices = result_data["residue_conformer_indices"]

    # Get residue start/end positions
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]

    # Loop through each residue and check that the reference conformer coordinates match
    # the coordinates using the appropriate index from the subsampled conformer indices
    for res_start, res_end in zip(_res_starts, _res_ends, strict=False):
        res_name = atom_array.res_name[res_start]
        res_global_id = int(atom_array.res_id_global[res_start])

        # Get the cached RDKit molecule and the conformer index that was selected
        cached_mol = cached_residue_level_data[res_name]["mol"]
        assert cached_mol is not None and cached_mol.GetNumConformers() > 0, f"No conformers found for {res_name}"
        selected_conformer_idx = int(
            conformer_indices[res_global_id][0]
        )  # Take first conformer index and convert to Python int

        # Get the expected coordinates from RDKit using the selected conformer index
        expected_conformer = atom_array_from_rdkit(
            cached_mol,
            conformer_id=selected_conformer_idx,
            remove_hydrogens=True,
        )

        # Map the expected conformer coordinates to the residue atom order
        expected_ref_pos, expected_ref_mask = _map_reference_conformer_to_residue(
            res_name=res_name,
            atom_names=atom_array.atom_name[res_start:res_end],
            conformer=expected_conformer,
        )

        # Get the actual reference positions from the AF3 features
        actual_ref_pos = feats["ref_pos"][res_start:res_end]
        actual_ref_mask = feats["ref_mask"][res_start:res_end]

        assert np.array_equal(actual_ref_mask, expected_ref_mask), "Reference masks don't match"
        assert np.allclose(expected_ref_pos[expected_ref_mask], actual_ref_pos[actual_ref_mask], atol=1e-6), (
            "Reference coordinates don't match"
        )


def test_chiral_centers_with_cached_conformers(cache_dir, data_with_subsampled_conformers):
    chiral_pipe = Compose(
        [
            GetRDKitChiralCenters(),
            AddAF3ChiralFeatures(),
        ]
    )
    # Smoke tests
    _ = chiral_pipe(data_with_subsampled_conformers)


@pytest.mark.parametrize(
    "res_name",
    [
        "G",  # 1-letter CCD code (cached at G/G/G.pt)
        "DISEP",  # 5-letter CCD code/ligand (cached at D/DISEP/DISEP.pt)
    ],
)
def test_cached_data_with_different_ccd_lengths(cache_dir, res_name):
    """Test that CCD codes of different lengths work correctly with sharding.

    Verifies that:
    1. Sharding works correctly for 1-letter and 5-letter CCD codes
    2. Data is actually loaded from cache for different length codes
    3. Loaded data contains expected keys (mol, descriptors, etc.)
    """
    # Create atom array with the specified residue name
    atom_array = struc.info.residue("ALA")  # Use ALA as template
    atom_array = atom_array[atom_array.element != "H"]
    atom_array.res_name[:] = res_name  # Set to test residue name
    atom_array = add_global_token_id_annotation(atom_array)

    # Test with sharding_depth=1 (matches actual cache structure)
    pipe = Compose(
        [
            AddGlobalResIdAnnotation(),
            LoadCachedResidueLevelData(dir=cache_dir, sharding_depth=1),
        ]
    )

    result = pipe({"atom_array": atom_array})

    # Verify basic structure
    assert "atom_array" in result
    assert "cached_residue_level_data" in result

    # Verify the cached data structure
    cached_data = result["cached_residue_level_data"]
    assert "residues" in cached_data
    assert "metadata" in cached_data

    # Verify data was actually loaded for this residue
    residues_data = cached_data["residues"]
    assert res_name in residues_data, f"Cache data not loaded for {res_name} - cache file should exist at cache_dir"

    # Verify the loaded data has expected structure
    res_data = residues_data[res_name]
    assert isinstance(res_data, dict), f"Expected dict for {res_name}, got {type(res_data)}"

    # Check for expected keys (mol is the most important one)
    assert "mol" in res_data, f"Expected 'mol' key in cached data for {res_name}"

    # Verify mol is not None and has conformers
    assert res_data["mol"] is not None, f"Expected non-None mol for {res_name}"
    assert res_data["mol"].GetNumConformers() > 0, f"Expected conformers for {res_name}, got 0"


if __name__ == "__main__":
    pytest.main([__file__])
