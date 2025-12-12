import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.datasets.parsers import PNUnitsDFParser, load_example_from_metadata_row
from atomworks.ml.preprocessing.constants import TRAINING_SUPPORTED_CHAIN_TYPES, ChainType
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import flag_and_reassign_covalent_modifications
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RemoveUnresolvedTokens,
    RemoveUnsupportedChainTypes,
    random_remove_pn_units_by_annotation_query,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_count, get_token_starts


@pytest.mark.parametrize("test_case", [{"pdb_id": "1s2k"}])
def test_remove_polymers_with_too_few_resolved_residues(test_case):
    # ...load the example from the CIF parser
    data = cached_parse(test_case["pdb_id"])
    atom_array = data["atom_array"]

    min_residues = 4

    def get_min_num_residues(atom_array: AtomArray) -> int:
        """
        Calculate the minimum number of unique residues in each polymer chain.
        """
        unique_chain_iids = np.unique(atom_array.chain_iid[atom_array.is_polymer])
        min_num_residues = np.min(
            [len(np.unique(atom_array.res_id[atom_array.chain_iid == chain_iid])) for chain_iid in unique_chain_iids]
        )
        return min_num_residues

    # ...assert that we have a polymer with too few resolved residues
    min_num_residues = get_min_num_residues(atom_array)
    assert min_num_residues < min_residues

    pipeline = Compose(
        [
            RemovePolymersWithTooFewResolvedResidues(min_residues=min_residues),
        ],
        track_rng_state=False,
    )
    output = pipeline(data)
    output_atom_array = output["atom_array"]

    # ...assert that the polymer with too few resolved residues has been removed
    min_num_residues = get_min_num_residues(output_atom_array)
    assert min_num_residues >= min_residues


def test_remove_terminal_oxygen():
    atom_array = struc.info.residue("ILE")
    assert "OXT" in atom_array.atom_name

    # Add required chain_type annotation for the RemoveTerminalOxygen transform
    atom_array.set_annotation("chain_type", np.full(atom_array.array_length(), ChainType.POLYPEPTIDE_L, dtype=int))

    data = {"atom_array": atom_array}

    transform = RemoveTerminalOxygen()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "OXT" not in atom_array_new.atom_name
    assert len(atom_array_new) == len(atom_array) - 1


@pytest.mark.parametrize("pdb_id", ["4gqa"])
def test_remove_unresolved_pn_units(pdb_id):
    # ...load the example from the CIF parser
    data = cached_parse(pdb_id)
    data["atom_array"] = data["assemblies"]["1"][0]

    # Artificially set the occupancy for all atoms in chain_iid "G_1" to 0
    data["atom_array"].occupancy[data["atom_array"].chain_iid == "G_1"] = 0

    pipeline = Compose(
        [
            RemoveUnresolvedPNUnits(),
        ],
        track_rng_state=False,
    )
    output = pipeline(data)

    # Assert that the atom array has no unresolved PN units
    pn_unit_iids = np.unique(output["atom_array"].pn_unit_iid)
    resolved_mask = output["atom_array"].occupancy != 0
    resolved_pn_unit_iids = np.unique(output["atom_array"].pn_unit_iid[resolved_mask])

    assert set(pn_unit_iids) == set(resolved_pn_unit_iids)


@pytest.mark.parametrize("pdb_id", ["3en2"])
def test_remove_unresolved_tokens(pdb_id):
    data = cached_parse(pdb_id)
    original_atom_array = data["atom_array"].copy()

    # ... count original tokens
    original_token_count = get_token_count(original_atom_array)

    # Apply RemoveUnresolvedTokens
    pipeline = Compose(
        [
            RemoveUnresolvedTokens(),
        ],
        track_rng_state=False,
    )
    output = pipeline(data)
    filtered_atom_array = output["atom_array"]

    # Verify that no token in the result has ALL atoms with occupancy 0
    token_starts = get_token_starts(filtered_atom_array, add_exclusive_stop=True)

    for i in range(len(token_starts) - 1):
        start, stop = token_starts[i], token_starts[i + 1]
        token_occupancies = filtered_atom_array.occupancy[start:stop]
        # Every remaining token should have at least one atom with occupancy > 0
        assert np.any(token_occupancies > 0), f"Token at positions {start}:{stop} has all unresolved atoms"

    # Should have removed some tokens (3en2 has unresolved tokens)
    filtered_token_count = get_token_count(filtered_atom_array)
    assert filtered_token_count < original_token_count, "Should have removed some wholly unresolved tokens"


def test_remove_hydrogens_original_pdb():
    atom_array = struc.info.residue("ILE")
    assert "H" in atom_array.element

    data = {"atom_array": atom_array}

    transform = RemoveHydrogens()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "H" not in atom_array_new.element
    assert len(atom_array_new) == len(atom_array[atom_array.element != "H"])


@pytest.mark.parametrize("pdb_id", ["5ocm", "1b4y", "1tqn"])
def test_remove_hydrogens_parsed_pdb(pdb_id: str):
    data = cached_parse(pdb_id, hydrogen_policy="keep")
    atom_array = data["atom_array"]
    assert 1 in atom_array.atomic_number

    transform = RemoveHydrogens()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert 1 not in atom_array_new.atomic_number
    assert len(atom_array_new) == len(atom_array[atom_array.element != "H"])


UNSUPPORTED_CHAIN_TYPE_TEST_CASES = [
    "104D",  # DNA/RNA Hybrid
    "5X3O",  # polypeptide(D)
]


@pytest.mark.parametrize("pdb_id", UNSUPPORTED_CHAIN_TYPE_TEST_CASES)
def test_remove_unsupported_chain_types(pdb_id: str, pn_units_df: pd.DataFrame):
    rows = pn_units_df[
        (pn_units_df["pdb_id"] == pdb_id.lower()) & (pn_units_df["assembly_id"] == "1")
    ]  # We only need the first assembly for UNSUPPORTED_CHAIN_TYPE_TEST_CASES

    assert not rows.empty

    for _, row in rows.iterrows():
        data = load_example_from_metadata_row(row, PNUnitsDFParser())
        is_unsupported_type = row["q_pn_unit_type"] not in TRAINING_SUPPORTED_CHAIN_TYPES
        original_atom_array = data["atom_array"].copy()

        # Apply transforms
        # fmt: off
        pipeline = Compose([
            RemoveUnsupportedChainTypes(),
        ], track_rng_state=False)
        # fmt: on

        output = None
        if is_unsupported_type:
            with pytest.raises(AssertionError):
                output = pipeline(data)
        else:
            output = pipeline(data)

        if output:
            atom_array = output["atom_array"]
            num_unsupported_atoms = len(original_atom_array) - len(atom_array)
            assert num_unsupported_atoms > 0, "There should be some atoms removed"
            chain_types = np.unique(atom_array.chain_type)
            assert np.all(np.isin(chain_types, TRAINING_SUPPORTED_CHAIN_TYPES)), (
                "All remaining chain types should be supported"
            )


def test_handle_undesired_res_single():
    transform = HandleUndesiredResTokens(["PTR", "SEP", "SO4", "NH2"])

    for with_hydrogens in (True, False):
        # Case 1:
        res = struc.info.residue("ALA")
        res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
        res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
        res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))

        if not with_hydrogens:
            res = res[res.element != "H"]

        res_out = transform({"atom_array": res})["atom_array"]
        assert np.all(res.coord == res_out.coord)
        assert np.all(res.atom_name == res_out.atom_name)
        assert np.all(res.is_polymer == np.ones(res.array_length(), dtype=bool))

    # Case 2:
    res = struc.info.residue("PTR")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))
    res_out_target = struc.info.residue("TYR")

    res = res[res.element != "H"]
    res_out_target = res_out_target[res_out_target.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "TYR")
    assert np.all(res_out.is_polymer == np.ones(res_out.array_length(), dtype=bool))
    assert np.all(res_out.coord.shape == res_out_target.coord.shape)
    assert np.all(res_out.coord == res.coord[np.isin(res.atom_name, res_out_target.atom_name)])

    # Case 3:
    res = struc.info.residue("SEP")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))
    res_out_target = struc.info.residue("SER")

    res = res[res.element != "H"]
    res_out_target = res_out_target[res_out_target.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "SER")
    assert np.all(res_out.is_polymer == np.ones(res_out.array_length(), dtype=bool))
    assert np.all(res_out.coord.shape == res_out_target.coord.shape)
    assert np.all(res_out.coord == res.coord[np.isin(res.atom_name, res_out_target.atom_name)])

    # Case 4 (atomize polymer bits that cannot be mapped to a canonical or unknown residue)
    res = struc.info.residue("NH2")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))

    res = res[res.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "NH2")
    assert len(res_out) == 1
    assert np.all(res_out.atomize == 1)

    # Case 5 (remove non-polymer bits)
    res = struc.info.residue("SO4")
    res.set_annotation("is_polymer", np.zeros(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 8 * np.ones(res.array_length(), dtype=int))

    res = res[res.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert len(res_out) == 0


CLASHING_PN_UNITS_TEST_CASES = [{"pdb_id": "1gt3", "assembly_id": "1"}]


@pytest.mark.parametrize("test_case", CLASHING_PN_UNITS_TEST_CASES)
def test_filter_to_specified_pn_units(test_case: str, pn_units_df: pd.DataFrame):
    pdb_id = test_case["pdb_id"]
    assembly_id = test_case["assembly_id"]

    rows = pn_units_df[(pn_units_df["pdb_id"] == pdb_id.lower()) & (pn_units_df["assembly_id"] == assembly_id)]

    assert not rows.empty

    # ... choose the first row, for speed (any row with the same PDB ID and assembly ID will do)
    rows = rows.iloc[:1]

    for _, row in rows.iterrows():
        data = load_example_from_metadata_row(row, PNUnitsDFParser())

        # Apply transforms
        # fmt: off
        pipeline = Compose([
            FilterToSpecifiedPNUnits(extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"),
        ], track_rng_state=False)
        # fmt: on
        output = pipeline(data)

        remaining_pn_unit_iids = np.unique(output["atom_array"].pn_unit_iid)
        expected_pn_unit_iids = set(eval(row["all_pn_unit_iids_after_processing"]))
        assert set(remaining_pn_unit_iids) == expected_pn_unit_iids


TERMINAL_OXYGEN_TEST_CASES = [
    {
        "pdb_id": "4z3c",
    }
]


@pytest.mark.parametrize("test_case", TERMINAL_OXYGEN_TEST_CASES)
def test_add_terminal_oxygen_indices(test_case: dict):
    data = cached_parse(test_case["pdb_id"])

    # Apply base transforms
    base_pipeline = Compose(
        [
            # Base pipeline
            RemoveHydrogens(),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
                move_atomized_part_to_end=False,
                validate_atomize=False,
            ),
        ]
    )

    prepared_data = base_pipeline(data)
    num_atoms_before_removal = len(prepared_data["atom_array"])
    assert "OP3" in prepared_data["atom_array"].atom_name

    remove_terminal_oxygen_indices_pipeline = Compose(
        [
            RemoveNucleicAcidTerminalOxygen(),
        ]
    )

    confidence_data = remove_terminal_oxygen_indices_pipeline(prepared_data)
    assert num_atoms_before_removal == len(confidence_data["atom_array"]) + 2
    assert "OP3" not in confidence_data["atom_array"].atom_name


def _categorize_pn_units(atom_array: AtomArray) -> dict:
    """Helper function to categorize pn_units by type"""
    all_pn_units = set(np.unique(atom_array.pn_unit_iid))
    polymer_pn_units = set(np.unique(atom_array.pn_unit_iid[atom_array.is_polymer]))
    covalent_mod_pn_units = set(np.unique(atom_array.pn_unit_iid[atom_array.is_covalent_modification]))
    free_floating_ligand_pn_units = set(
        np.unique(atom_array.pn_unit_iid[~atom_array.is_polymer & ~atom_array.is_covalent_modification])
    )

    return {
        "all": all_pn_units,
        "polymer": polymer_pn_units,
        "covalent_modification": covalent_mod_pn_units,
        "free_floating_ligand": free_floating_ligand_pn_units,
        "counts": {
            "all": len(all_pn_units),
            "polymer": len(polymer_pn_units),
            "covalent_modification": len(covalent_mod_pn_units),
            "free_floating_ligand": len(free_floating_ligand_pn_units),
        },
    }


def test_randomly_remove_ligands():
    """Test RandomlyRemoveLigands with different probabilities"""
    data = cached_parse("4js1")
    atom_array_with_covalent_mods = flag_and_reassign_covalent_modifications(data["atom_array"])

    # Categorize original pn_units
    original_categories = _categorize_pn_units(atom_array_with_covalent_mods)

    # Test with 0% probability - nothing should be removed
    with rng_state(create_rng_state_from_seeds(np_seed=42)):
        result_0_percent = random_remove_pn_units_by_annotation_query(
            atom_array_with_covalent_mods.copy(),
            query="~is_polymer & ~is_covalent_modification",
            delete_probability=0.0,
        )
    categories_0_percent = _categorize_pn_units(result_0_percent)
    assert categories_0_percent["all"] == original_categories["all"], (
        f"With 0% probability, all pn_units should remain. Expected {original_categories['counts']['all']}, got {categories_0_percent['counts']['all']}"
    )

    # Test with 100% probability - all free-floating ligands should be removed
    with rng_state(create_rng_state_from_seeds(np_seed=42)):
        result_100_percent = random_remove_pn_units_by_annotation_query(
            atom_array_with_covalent_mods.copy(),
            query="~is_polymer & ~is_covalent_modification",
            delete_probability=1.0,
        )
    categories_100_percent = _categorize_pn_units(result_100_percent)

    expected_remaining_100_percent = original_categories["all"] - original_categories["free_floating_ligand"]
    assert categories_100_percent["all"] == expected_remaining_100_percent, (
        f"With 100% probability, only free-floating ligands should be removed. Expected {len(expected_remaining_100_percent)}, got {categories_100_percent['counts']['all']}"
    )

    # Test with 50% probability multiple times to verify randomness works
    results_50_percent = []
    for seed in range(10):  # Run 10 times with different seeds
        with rng_state(create_rng_state_from_seeds(np_seed=seed)):
            result_50_percent = random_remove_pn_units_by_annotation_query(
                atom_array_with_covalent_mods.copy(),
                query="~is_polymer & ~is_covalent_modification",
                delete_probability=0.5,
            )
        categories_50_percent = _categorize_pn_units(result_50_percent)
        results_50_percent.append(categories_50_percent["all"])

        # Verify that the remaining pn_units are a subset of the original
        assert categories_50_percent["all"].issubset(original_categories["all"]), (
            f"Remaining pn_units should be a subset of original (seed {seed})"
        )

        # Verify that polymer pn_units are always preserved (they don't match the query)
        assert categories_50_percent["polymer"] == original_categories["polymer"], (
            f"Polymer pn_units should always be preserved (seed {seed})"
        )

    # Verify that we get different results across different seeds (randomness check)
    unique_results = {frozenset(result) for result in results_50_percent}
    assert len(unique_results) > 1, "With 50% probability and different seeds, we should get varying results"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
