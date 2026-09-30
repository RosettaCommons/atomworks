from typing import Any

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_single

from atomworks.ml.transforms.sasa import (
    CalculateSASA,
    _get_element_radii,
    _get_protor_element_fallback_radii,
    calculate_atomwise_rasa,
)
from atomworks.ml.utils.testing import cached_parse

SASA_TEST_CASES = [
    {
        "pdb_id": "1fu2",  # multi-chain protein with ZN
        "probe_radius": 1.4,
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H", "expect_nan": True},
            {"atom_name": "ZN", "expect_nan": True},  # ProtOr has no metal radii
            {"atom_name": "N", "expect_nan": False},
        ],
    },
    {
        "pdb_id": "3p42",  # protein with NaN coordinates
        "probe_radius": 1.4,
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H", "expect_nan": True},
            {"atom_name": "N", "expect_nan": False},
        ],
    },
    {
        "pdb_id": "1fu2",  # element mode: covers metals and ligand atoms
        "probe_radius": 1.4,
        "atom_radii": "element",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H", "expect_nan": True},
            {"atom_name": "ZN", "expect_nan": False},  # element mode has metal radii
            {"atom_name": "N", "expect_nan": False},
        ],
    },
]


@pytest.mark.parametrize("test_case", SASA_TEST_CASES)
def test_calculate_sasa(test_case: dict[str, Any]):
    """Test CalculateSASA with ProtOr radii (protein-only)."""
    data = cached_parse(test_case["pdb_id"])
    transform = CalculateSASA(
        probe_radius=test_case["probe_radius"],
        atom_radii=test_case["atom_radii"],
        point_number=test_case["point_number"],
    )
    data = transform(data)

    for spot_check in test_case["spot_checks"]:
        atom_mask = data["atom_array"].atom_name == spot_check["atom_name"]
        if spot_check["expect_nan"]:
            assert np.isnan(
                data["atom_array"][atom_mask].sasa
            ).all(), f"{spot_check['atom_name']} should have NaN SASA with {test_case['atom_radii']} radii"
        else:
            valid_mask = ~np.isnan(data["atom_array"][atom_mask].coord).any(axis=-1)
            assert np.all(
                data["atom_array"][atom_mask][valid_mask].sasa >= 0
            ), f"{spot_check['atom_name']} should have valid SASA with {test_case['atom_radii']} radii"


@pytest.mark.parametrize("test_case", SASA_TEST_CASES)
def test_calculate_rasa(test_case: dict[str, Any]):
    """Test RASA with ProtOr radii (protein-only)."""
    data = cached_parse(test_case["pdb_id"])
    atom_array = data["atom_array"]
    rasa = calculate_atomwise_rasa(
        atom_array,
        probe_radius=test_case["probe_radius"],
        atom_radii=test_case["atom_radii"],
        point_number=test_case["point_number"],
    )

    has_coords = ~np.isnan(atom_array.coord).any(axis=-1)
    is_heavy = atom_array.element != "H"
    has_sasa = ~np.isnan(rasa)
    valid_mask = has_coords & is_heavy & has_sasa

    assert np.all(rasa[valid_mask] >= 0), "RASA should be >= 0 for valid heavy atoms"
    assert np.all(rasa[valid_mask] <= 1), "RASA should be <= 1 for valid heavy atoms"

    no_coords_mask = ~has_coords
    assert np.isnan(rasa[no_coords_mask]).all(), "Atoms without coords should have NaN RASA"


def _assert_valid_rasa(rasa: np.ndarray, atom_array: AtomArray, mode: str) -> None:
    """Shared assertions for RASA across modes."""
    has_coords = ~np.isnan(atom_array.coord).any(axis=-1)
    is_heavy = atom_array.element != "H"
    has_sasa = ~np.isnan(rasa)
    valid_mask = has_coords & is_heavy & has_sasa

    valid_rasa = rasa[valid_mask]
    assert np.all(valid_rasa >= 0), f"[{mode}] RASA should be >= 0"
    assert np.all(valid_rasa <= 1), f"[{mode}] RASA should be <= 1"


def test_rasa_auto_protein_metal():
    """Test auto mode on 1fu2 (protein + ZN): ProtOr for residues, element fallback for ZN."""
    data = cached_parse("1fu2")
    atom_array = data["atom_array"]

    rasa = calculate_atomwise_rasa(atom_array, atom_radii="auto")
    _assert_valid_rasa(rasa, atom_array, "auto")

    has_coords = ~np.isnan(atom_array.coord).any(axis=-1)
    zn_mask = (atom_array.element == "ZN") & has_coords
    if zn_mask.any():
        assert not np.isnan(rasa[zn_mask]).any(), "ZN should have valid RASA with auto mode"

    radii = _get_protor_element_fallback_radii(atom_array)
    ala_ca_mask = (atom_array.res_name == "ALA") & (atom_array.atom_name == "CA")
    if ala_ca_mask.any():
        protor_ca = struc.info.radii.vdw_radius_protor("ALA", "CA")
        assert radii[ala_ca_mask][0] == pytest.approx(protor_ca), "Standard residues should use ProtOr"
    zn_only = atom_array.element == "ZN"
    if zn_only.any():
        assert radii[zn_only][0] == pytest.approx(vdw_radius_single("ZN")), "ZN should use element fallback"


def test_rasa_auto_protein_dna_metal():
    """Test auto mode on 6w13 (protein + DNA + MG + ligand): all atom types should work."""
    data = cached_parse("6w13")
    atom_array = data["atom_array"]

    rasa = calculate_atomwise_rasa(atom_array, atom_radii="auto")
    _assert_valid_rasa(rasa, atom_array, "auto")

    has_coords = ~np.isnan(atom_array.coord).any(axis=-1)
    mg_mask = (atom_array.element == "MG") & has_coords
    if mg_mask.any():
        assert not np.isnan(rasa[mg_mask]).any(), "MG should have valid RASA"

    dna_residues = {"DA", "DT", "DC", "DG"}
    dna_mask = np.isin(atom_array.res_name, dna_residues) & has_coords & (atom_array.element != "H")
    if dna_mask.any():
        assert not np.isnan(rasa[dna_mask]).any(), "DNA heavy atoms should have valid RASA"


def test_element_radii_unknown_raises():
    """Unknown elements should raise ValueError, not silently default."""
    atom_array = AtomArray(1)
    atom_array.element = np.array(["XX"])
    atom_array.atom_name = np.array(["XX"])
    atom_array.res_name = np.array(["UNK"])

    with pytest.raises(ValueError, match="No VdW radius found for element 'XX'"):
        _get_element_radii(atom_array)


def test_element_radii_ligand_atoms():
    """Element-based SASA/RASA should work for synthetic ligand-like atom arrays."""
    atom_array = AtomArray(6)
    atom_array.chain_id = np.array(["A", "A", "A", "L", "L", "L"])
    atom_array.res_id = np.array([1, 1, 1, 0, 0, 0])
    atom_array.res_name = np.array(["ALA", "ALA", "ALA", "L", "L", "L"])
    atom_array.atom_name = np.array(["CA", "C", "N", "C0", "O0", "N0"])
    atom_array.element = np.array(["C", "C", "N", "C", "O", "N"])
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [5.0, 5.0, 5.0],
            [6.5, 5.0, 5.0],
            [5.0, 6.5, 5.0],
        ]
    )

    rasa = calculate_atomwise_rasa(atom_array, atom_radii="element")
    assert not np.isnan(rasa).any(), "All atoms should have valid RASA"
    assert np.all(rasa >= 0)
    assert np.all(rasa <= 1)
