import biotite.structure as struc
import pytest

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.chirals import (
    AddAF3ChiralFeatures,
)
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.filters import RemoveHydrogens
from atomworks.ml.transforms.openbabel_utils import (
    AddOpenBabelMoleculesForAtomizedMolecules,
    GetChiralCentersFromOpenBabel,
    atom_array_to_openbabel,
    get_chiral_centers,
    smiles_to_openbabel,
)
from atomworks.ml.transforms.rdkit_utils import GetRDKitChiralCenters
from atomworks.ml.utils.testing import cached_parse


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
    {"pdb_id": "5ocm", "expected_chiral_count": 20},
    {"pdb_id": "6lyz", "expected_chiral_count": 0},
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_openbabel_chiral_centers(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
        ],
        track_rng_state=False,
    )

    data = pipe(data)

    assert len(data["chiral_centers"]) == test_case["expected_chiral_count"]


TEST_CASES_WITH_ATOMIZED_RESIDUES = [
    {"pdb_id": "5ocm", "expected_chiral_count": 20},
    {"pdb_id": "6lyz", "expected_chiral_count": 0},
]


@pytest.mark.parametrize("test_case", TEST_CASES_WITH_ATOMIZED_RESIDUES)
def test_openbabel_chiral_centers_with_atomized_residues(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    res_names_to_ignore = [token for token in RF2AA_ATOM36_ENCODING.tokens if token not in ["ALA"]]
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=res_names_to_ignore),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
        ],
        track_rng_state=False,
    )

    data = pipe(data)

    n_alanines = len(list(filter(lambda x: x == "ALA", struc.get_residues(data["atom_array"])[1])))

    assert len(data["chiral_centers"]) == test_case["expected_chiral_count"] + n_alanines


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
        "spotcheck_openbabel_molecule": {"atom_id": 996, "pn_unit_iid": "B_1"},
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES_WITH_COVALENT_MODIFICATION)
def test_openbabel_chiral_centers_with_covalent_modification(test_case: dict):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            GetChiralCentersFromOpenBabel(),
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
