from itertools import permutations

import biotite.structure as struc
import numpy as np
import pytest
import torch
from openbabel import openbabel, pybel

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.chirals import (
    AddAF3ChiralFeatures,
    AddRF2AAChiralFeatures,
    get_dih,
    get_rf2aa_chiral_features,
)
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.crop import CropSpatialLikeAF3
from atomworks.ml.transforms.filters import RemoveHydrogens
from atomworks.ml.transforms.openbabel_utils import (
    AddOpenBabelMoleculesForAtomizedMolecules,
    GetChiralCentersFromOpenBabel,
    atom_array_to_openbabel,
    get_chiral_centers,
    smiles_to_openbabel,
)
from atomworks.ml.transforms.rdkit_utils import GetRDKitChiralCenters
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse


# NOTE: The following section is copied directly from rf2aa to ensure repeatability
# -----
def standardize_dihedral_retain_first(a, b, c, d):
    isomorphisms = [(a, b, c, d), (a, c, b, d)]  # clockwise & counterclockwise are the same
    return sorted(isomorphisms)[0]


def get_chirals_legacy(obmol, xyz):
    """
    get all quadruples of atoms forming chiral centers and the expected ideal pseudodihedral between them
    """
    stereo = openbabel.OBStereoFacade(obmol)
    angle = np.arcsin(1 / 3**0.5)
    chiral_idx_set = set()

    # For each tetrahedral chiral center ...
    for i in range(obmol.NumAtoms()):
        # ... skip if not a stereocenter
        if not stereo.HasTetrahedralStereo(i):
            continue
        si = stereo.GetTetrahedralStereo(i)
        config = si.GetConfig()

        o = config.center
        # i.e.: Looking from atom `config.from_or_towards` the atom IDs `config.refs` are arranged clockwise (the default config)

        # ... enumerate all sets of 3 atom neighbors in all orders
        i, j, k = list(config.refs)
        for a, b, c in permutations((config.from_or_towards, i, j, k), 3):
            chiral_idx_set.add(standardize_dihedral_retain_first(o, a, b, c))

    chiral_idx = list(chiral_idx_set)
    chiral_idx.sort()
    chiral_idx = torch.tensor(chiral_idx, dtype=torch.float32)

    # (drop out 3 atom neighbors where one atom is an implicit hydrogen)
    chiral_idx = chiral_idx[
        (chiral_idx < obmol.NumAtoms()).all(dim=-1)
    ]  # drops out the implicit hydrogens enumerations

    if chiral_idx.numel() == 0:
        return torch.zeros((0, 5))

    # for each ordering compute the pseudo-dihedral angle between the 4 points (center & 3 heavy atoms)
    dih = get_dih(*xyz[chiral_idx.long()].split(split_size=1, dim=1))[:, 0]
    chirals = torch.nn.functional.pad(chiral_idx, (0, 1), mode="constant", value=angle)
    # note whether the angle is positive or negative, determining the chirality of the center uniquely
    chirals[dih < 0.0, -1] *= -1
    return chirals


# -----


SMILES_TEST_CASES = [
    "F[C@@](Cl)(Br)I",  # 1 chiral around F
    "C[C@H](N)C(=O)O",  # 1 chiral around C (CA - this is alanine)
    "CC",  # no chirals
    "c1ccccc1",  # no chirals
    "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N",  # NAP, 10 chirals
]


@pytest.mark.parametrize("smiles", SMILES_TEST_CASES)
def test_get_chirals_from_smiles(smiles: str):
    obmol = smiles_to_openbabel(smiles)
    mol = pybel.Molecule(obmol)

    # generate conformer 3D coordinates
    builder = openbabel.OBBuilder()
    builder.Build(obmol)

    # extract coordinates
    coords = torch.tensor([atom.coords for atom in mol.atoms])

    legacy_chirals = get_chirals_legacy(obmol, coords)  # TODO: Add chirals test case that is not from smiles

    chiral_centers = get_chiral_centers(obmol)
    new_chirals = get_rf2aa_chiral_features(chiral_centers, coords, take_first_chiral_subordering=False)

    assert torch.allclose(legacy_chirals, new_chirals)


def test_chiral_count():
    # from atom array
    atom_array = cached_parse("5ocm")["atom_array"]
    nap = atom_array[(atom_array.res_name == "NAP") & (atom_array.chain_id == "G")]
    obmol = atom_array_to_openbabel(nap, infer_hydrogens=False, infer_aromaticity=False)
    assert len(get_chiral_centers(obmol)) == 10, f"Expected 10 chirals, found {len(get_chiral_centers(obmol))}"

    # from smiles
    obmol = smiles_to_openbabel(
        "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N"
    )
    assert len(get_chiral_centers(obmol)) == 10, f"Expected 10 chirals, found {len(get_chiral_centers(obmol))}"


TEST_CASES = [
    {"pdb_id": "5ocm", "expected_chiral_count": 20, "expected_chiral_feats_shape": (240, 5)},
    {"pdb_id": "6lyz", "expected_chiral_count": 0, "expected_chiral_feats_shape": (0, 5)},
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_chiral_featurization(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
            AddRF2AAChiralFeatures(),
        ],
        track_rng_state=False,
    )

    data = pipe(data)

    assert len(data["chiral_centers"]) == test_case["expected_chiral_count"]
    assert data["chiral_feats"].shape == test_case["expected_chiral_feats_shape"]


TEST_CASES_WITH_ATOMIZED_RESIDUES = [
    {"pdb_id": "5ocm", "expected_chiral_count": 20, "expected_chiral_feats_shape": (240, 5)},
    {"pdb_id": "6lyz", "expected_chiral_count": 0, "expected_chiral_feats_shape": (0, 5)},
]


@pytest.mark.parametrize("test_case", TEST_CASES_WITH_ATOMIZED_RESIDUES)
def test_chiral_featurization_with_atomized_residues(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    res_names_to_ignore = [token for token in RF2AA_ATOM36_ENCODING.tokens if token not in ["ALA"]]
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=res_names_to_ignore),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
            AddRF2AAChiralFeatures(),
        ],
        track_rng_state=False,
    )

    data = pipe(data)

    n_alanines = len(list(filter(lambda x: x == "ALA", struc.get_residues(data["atom_array"])[1])))

    assert len(data["chiral_centers"]) == test_case["expected_chiral_count"] + n_alanines
    assert data["chiral_feats"].shape == (
        test_case["expected_chiral_feats_shape"][0] + n_alanines * 12,
        test_case["expected_chiral_feats_shape"][1],
    )


TEST_CASES_WITH_COVALENT_MODIFICATION = [
    {
        # 4js1: A_1 61 (protein residue) is covalently bound to B_1 (multi-chain sugar)
        "pdb_id": "4js1",
        "residues_to_be_atomized": [
            {
                "polymer_pn_unit_iid": "A_1",
                "polymer_res_id": 61,
                "non_polymer_pn_unit_iid": "B_1",
                "non_polymer_pn_unit_id": "B",
            }
        ],
        "expected_chiral_count": 50,
        # 3 * 50 = 150 plane-pairs to compare (since all chirals here are with an implicit hydrogen there's 3 plane-pairs per chiral center)
        "expected_chiral_feats_shape": (150, 5),
        "spotcheck_openbabel_molecule": {"atom_id": 996, "pn_unit_iid": "B_1"},
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES_WITH_COVALENT_MODIFICATION)
def test_chiral_featurization_with_covalent_modification(test_case: dict):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
            AddRF2AAChiralFeatures(),
        ],
        track_rng_state=False,
    )

    data = pipe(cached_parse(test_case["pdb_id"]))

    atom_array = data["atom_array"]
    spotcheck_atom_id = test_case["spotcheck_openbabel_molecule"]["atom_id"]
    spotcheck_pn_unit_iid = test_case["spotcheck_openbabel_molecule"]["pn_unit_iid"]
    openbabel_molecules = data["openbabel"]

    assert spotcheck_atom_id in openbabel_molecules
    assert openbabel_molecules[spotcheck_atom_id].NumAtoms() == len(
        atom_array[atom_array.pn_unit_iid == spotcheck_pn_unit_iid]
    )
    assert len(data["chiral_centers"]) == test_case["expected_chiral_count"]
    assert data["chiral_feats"].shape == test_case["expected_chiral_feats_shape"]


def test_chiral_featurize_after_cropping():
    pdb_id = "5rx1"
    seed = 2

    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            CropSpatialLikeAF3(crop_size=128),
            GetChiralCentersFromOpenBabel(),
            AddRF2AAChiralFeatures(),
        ],
        track_rng_state=False,
    )

    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        data = cached_parse(pdb_id, hydrogen_policy="remove")
        data = pipe(data)

    expected = torch.tensor(
        [
            [116.0000, 115.0000, 121.0000, 126.0000, 0.6155],
            [116.0000, 115.0000, 126.0000, 121.0000, -0.6155],
            [116.0000, 121.0000, 126.0000, 115.0000, 0.6155],
        ]
    )

    assert data["chiral_feats"].shape == (3, 5)
    assert torch.allclose(data["chiral_feats"], expected, atol=1e-3)


TEST_CASES_RDKIT = [
    {"pdb_id": "5ocm", "expected_chiral_feats_shape": (1782, 5), "expected_positive_chirals": 1176},
    {"pdb_id": "6lyz", "expected_chiral_feats_shape": (390, 5), "expected_positive_chirals": 260},
]


@pytest.mark.parametrize("test_case", TEST_CASES_RDKIT)
def test_rdkit_chiral_featurization(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id, hydrogen_policy="remove")

    pipe = Compose(
        [
            GetAF3ReferenceMoleculeFeatures(),
            GetRDKitChiralCenters(),
            AddAF3ChiralFeatures(),
        ],
        track_rng_state=False,
    )

    data = pipe(data)

    assert data["feats"]["chiral_feats"].shape == test_case["expected_chiral_feats_shape"]
    assert (data["feats"]["chiral_feats"][:, -1] > 0).sum() == test_case["expected_positive_chirals"]


def test_take_first_chiral_subordering():
    # EW6 is a small molecule with a chiral center bonded to four non-hydrogen atoms
    residue = struc.info.residue("EW6", allow_missing_coord=True)
    data = {"atom_array": residue}
    pipe = Compose(
        [
            RemoveHydrogens(),
            GetAF3ReferenceMoleculeFeatures(),
            GetRDKitChiralCenters(),
            # Take only the first tetrahedral side (so all chiral centers should have the same number of features)
            AddAF3ChiralFeatures(take_first_chiral_subordering=True),
        ],
        track_rng_state=False,
    )
    data = pipe(data)

    assert data["feats"]["chiral_feats"].shape == (4, 5)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
