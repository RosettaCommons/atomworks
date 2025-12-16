import numpy as np
import pytest

from atomworks.io.utils.testing import assert_same_atom_array
from atomworks.ml.encoding_definitions import (
    RF2AA_ATOM36_ENCODING,
)
from atomworks.ml.transforms.atom_array import CopyAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.encoding import EncodeAtomArray
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
    mask_polymer_residues_with_unresolved_frame_atoms,
)
from atomworks.ml.transforms.filters import RemoveUnresolvedPNUnits
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import (
    get_af3_token_center_idxs,
    get_af3_token_center_masks,
    get_af3_token_representative_idxs,
    get_af3_token_representative_masks,
    token_iter,
)


@pytest.mark.parametrize("pdb_id", ["6wtf"])
def test_mask_residues_with_unresolved_backbone_atoms(pdb_id):
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]

    # ... manually set the occupancy of a CA atom to zero
    resolved_ca_atoms = (atom_array.atom_name == "CA") & (atom_array.occupancy > 0)

    # ... set the first CA atom to zero occupancy
    atom_array.occupancy[resolved_ca_atoms] = np.array([0.0] + [1.0] * (np.sum(resolved_ca_atoms) - 1))
    changed_atom = atom_array[resolved_ca_atoms][0]

    # ... apply the transform
    updated_atom_array = mask_polymer_residues_with_unresolved_frame_atoms(atom_array)

    # ... assert that the manually set CA atom's residue is masked
    changed_residue_mask = (updated_atom_array.chain_id == changed_atom.chain_id) & (
        updated_atom_array.res_id == changed_atom.res_id
    )
    assert np.all(updated_atom_array.occupancy[changed_residue_mask] == 0)

    # ... assert that the rest of the residues are unchanged
    unchanged_residue_mask = ~changed_residue_mask
    assert np.all(updated_atom_array.occupancy[unchanged_residue_mask] == atom_array.occupancy[unchanged_residue_mask])


FEATURIZE_UNRESOLVED_RESIDUES_TEST_CASES = ["6wtf", "7rcu", "8e83", "7okl", "7z24"]


@pytest.mark.parametrize("pdb_id", FEATURIZE_UNRESOLVED_RESIDUES_TEST_CASES)
def test_place_unresolved_token_atoms_on_representative_atom(pdb_id):
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]

    # ... check for unresolved polymer atoms (there will be lots of unresolved hydrogens, so we leave them in  as a test case)
    unresolved_polymer_atoms = atom_array[(atom_array.is_polymer) & (atom_array.occupancy == 0)]

    # ... same thing for unresolved non-polymer atoms (hydrogens will be unresolved)
    unresolved_non_polymer_atoms = atom_array[(~atom_array.is_polymer) & (atom_array.occupancy == 0)]

    assert len(unresolved_polymer_atoms) > 0
    assert len(unresolved_non_polymer_atoms) > 0

    encoding = RF2AA_ATOM36_ENCODING
    pipe = Compose(
        [
            FlagNonPolymersForAtomization(),
            MaskPolymerResiduesWithUnresolvedFrameAtoms(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            CopyAnnotation("coord", "coord_to_be_noised"),
            EncodeAtomArray(encoding),
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord_to_be_noised"),
        ]
    )
    output = pipe(data)
    output_atom_array = output["atom_array"]

    # ... get the unresolved polymer atoms again, but applying atomization mask as well
    unresolved_polymer_atoms = output_atom_array[
        (output_atom_array.is_polymer) & (output_atom_array.occupancy == 0) & (~output_atom_array.atomize)
    ]
    unresolved_non_polymer_atoms = output_atom_array[
        (~output_atom_array.is_polymer) & (output_atom_array.occupancy == 0) & (output_atom_array.atomize)
    ]

    # ... loop through each unresolved polymer token, and ensure that the unresolved atoms have the same coordinates as the representative atom
    for chain_iid in np.unique(unresolved_polymer_atoms.chain_iid):
        chain_atom_array = output_atom_array[(output_atom_array.chain_iid == chain_iid) & (~output_atom_array.atomize)]
        for res_id in np.unique(chain_atom_array.res_id):
            residue_atom_array = chain_atom_array[chain_atom_array.res_id == res_id]
            unresolved_atom_mask = residue_atom_array.occupancy == 0

            representative_atom_idx = get_af3_token_representative_idxs(residue_atom_array)
            center_atom_idx = get_af3_token_center_idxs(residue_atom_array)

            output_atom_array_residue = output_atom_array[
                (output_atom_array.chain_iid == chain_iid) & (output_atom_array.res_id == res_id)
            ]

            if residue_atom_array.occupancy[representative_atom_idx]:
                # If the representative atom is resolved, all coordinates should be at the representative atom
                assert np.allclose(
                    residue_atom_array.coord_to_be_noised[unresolved_atom_mask],
                    output_atom_array_residue.coord_to_be_noised[representative_atom_idx],
                    atol=1e-6,
                    equal_nan=True,
                )
            else:
                # Otherwise, all coordinates should be at the center atom
                # (The NaN case is also handled by this check)
                assert np.allclose(
                    residue_atom_array.coord_to_be_noised[unresolved_atom_mask],
                    output_atom_array_residue.coord_to_be_noised[center_atom_idx],
                    atol=1e-6,
                    equal_nan=True,
                )

    # ... loop through each unresolved non-polymer token, and ensure that nothing changed
    for chain_iid in np.unique(unresolved_non_polymer_atoms.chain_iid):
        output_chain_atom_array = output_atom_array[output_atom_array.chain_iid == chain_iid]
        input_chain_atom_array = atom_array[atom_array.chain_iid == chain_iid]
        assert_same_atom_array(output_chain_atom_array, input_chain_atom_array)


@pytest.mark.parametrize("pdb_id", FEATURIZE_UNRESOLVED_RESIDUES_TEST_CASES)
def test_place_unresolved_token_on_closest_resolved_token_in_sequence(pdb_id):
    data = cached_parse(pdb_id)

    encoding = RF2AA_ATOM36_ENCODING
    pipe = Compose(
        [
            FlagNonPolymersForAtomization(),
            RemoveUnresolvedPNUnits(),
            MaskPolymerResiduesWithUnresolvedFrameAtoms(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            EncodeAtomArray(encoding),
            CopyAnnotation("coord", "coord_to_be_noised"),
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord_to_be_noised"),
        ],
        track_rng_state=False,
    )
    output = pipe(data)

    # ... apply the transform
    output = PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(
        annotation_to_update="coord_to_be_noised", annotation_to_copy="coord_to_be_noised"
    )(output)
    output_atom_array = output["atom_array"]

    assert not np.isnan(
        output_atom_array.coord_to_be_noised
    ).any(), "There should be no NaNs in the output coordinates!"

    for chain_id in np.unique(output_atom_array.chain_id):
        chain_atom_array = output_atom_array[output_atom_array.chain_id == chain_id]

        # ... ensure that resolved atoms are unchanged
        assert np.allclose(
            chain_atom_array.coord[chain_atom_array.occupancy > 0],
            chain_atom_array.coord_to_be_noised[chain_atom_array.occupancy > 0],
            atol=1e-6,
            equal_nan=True,
        )

        representative_tokens_coordinates = chain_atom_array.coord_to_be_noised[
            get_af3_token_representative_masks(chain_atom_array)
        ]

        # ... ensure that unresolved tokens have their tokens placed on the closest resolved token
        for idx, token in enumerate(token_iter(chain_atom_array)):
            # (Skip tokens with resolved center or representative atoms)
            representative_atom_mask = get_af3_token_representative_masks(token)
            center_atom_mask = get_af3_token_center_masks(token)
            if (representative_atom_mask | center_atom_mask).any():
                continue

            # ... assert all coordinates are the same within the token
            assert np.all(np.all(token.coord_to_be_noised == token.coord_to_be_noised[0]))

            # ... find the index of the closest resolved token
            # (Check below)
            lower_index = -float("inf")
            for i in range(idx - 1, -1, -1):
                if not np.isnan(representative_tokens_coordinates[i]).any():
                    lower_index = i
                    break
            # (Check above)
            upper_index = float("inf")
            for i in range(idx + 1, len(representative_tokens_coordinates)):
                if not np.isnan(representative_tokens_coordinates[i]).any():
                    upper_index = i
                    break

            # ... calculate the distance in sequence space to both the lower and upper resolved tokens
            if abs(idx - lower_index) <= abs(upper_index - idx):
                # The closest resolved token should be the lower one
                assert np.allclose(
                    token.coord_to_be_noised, representative_tokens_coordinates[lower_index], equal_nan=True
                )
            else:
                # The closest resolved token should be the upper one
                assert np.allclose(
                    token.coord_to_be_noised, representative_tokens_coordinates[upper_index], equal_nan=True
                )


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
