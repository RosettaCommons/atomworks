"""Tests for the inference processing tools."""

import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.enums import ChainType
from atomworks.io import parse
from atomworks.io.tools.fasta import split_generalized_fasta_sequence
from atomworks.io.tools.inference import (
    ChemicalComponent,
    Protein,
    SequenceComponent,
    SmilesComponent,
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
    one_letter_to_ccd_code,
    read_chai_fasta,
)
from atomworks.io.utils.testing import assert_same_atom_array
from tests.io.conftest import TEST_DATA_IO, get_pdb_path


@pytest.fixture
def dict_inputs():
    """Fixture providing example chemical components for testing."""
    cif_path = [
        {
            "path": f"{TEST_DATA_IO}/test_cif_loading_4q8n.cif.gz",  # Contains two symmetry transformations
            "msa_paths": {
                "A": "/example/msa/path.a3m.gz",
            },
        }
    ]

    monomer = [
        {
            "seq": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "chain_id": "C",
            "is_polymer": True,
        }
    ]

    dimer = [
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "D",
        },
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "E",
        },
    ]

    noncanonical = [
        {
            "seq": "KVFGRCE(SEP)AAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
        }
    ]

    custom_residues = [
        {
            "seq": "G(C:0)G(SEP)G",
            "chain_type": "polypeptide(l)",
        }
    ]

    ligand = [
        {
            "smiles": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "F",
        }
    ]

    glycan_1 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "G",
        }
    ]
    glycan_2 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "H",
        }
    ]

    sdf = [
        {
            "path": f"{TEST_DATA_IO}/HEM_ideal.sdf",
        }
    ]

    return {
        "monomer": monomer,
        "dimer": dimer,
        "noncanonical": noncanonical,
        "custom_residues": custom_residues,
        "ligand": ligand,
        "glycan_1": glycan_1,
        "glycan_2": glycan_2,
        "sdf": sdf,
        "cif_path": cif_path,
    }


@pytest.fixture
def bonds_glycan_glycan():
    # Bond between the two NAG residues (O4 and C1 atoms) and NAG and the protein (ND2 on ASN and C1 on NAG)
    # For details on the bond API, see `bonds.py`
    return [("G/NAG/1/O4", "H/NAG/1/C1"), ("C/ASN/19/ND2", "G/NAG/1/C1")]


@pytest.fixture
def custom_residues():
    return {
        "C:0": {
            "path": f"{TEST_DATA_IO}/example_ncaa.cif",
        }
    }


@pytest.fixture
def chai_fasta_input():
    example_fasta_content = """
>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(example_fasta_content)
    fasta_path = Path(f.name)
    yield fasta_path
    fasta_path.unlink()


def test_split_generalized_fasta_sequence():
    """Test splitting sequences with and without parentheses."""
    assert split_generalized_fasta_sequence("ABC") == ["A", "B", "C"]
    assert split_generalized_fasta_sequence("ABC(DEF)GH") == ["A", "B", "C", "(DEF)", "G", "H"]
    assert split_generalized_fasta_sequence("A(XYZ)C(DEF)") == ["A", "(XYZ)", "C", "(DEF)"]


def test_one_letter_to_ccd_code():
    """Test conversion of one-letter codes to CCD codes."""
    # Test standard amino acids
    seq = ["A", "C", "D"]
    result = one_letter_to_ccd_code(seq, ChainType.POLYPEPTIDE_L)
    assert result == ["ALA", "CYS", "ASP"]

    # Test with modified residue
    seq = ["A", "(SEP)", "G"]
    result = one_letter_to_ccd_code(seq, ChainType.POLYPEPTIDE_L)
    assert result == ["ALA", "SEP", "GLY"]

    # Test invalid modified residue
    with pytest.raises(ValueError):
        one_letter_to_ccd_code(["A", "(THISDOESNOTEXIST)"], ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_monomer(dict_inputs):
    """Test conversion of monomer components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["monomer"])

    assert isinstance(atom_array, AtomArray)
    assert len(atom_array) > 0

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "C" in chain_ids
    print(chain_ids)

    # Verify polymer annotation
    assert np.all(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_dimer(dict_inputs):
    """Test conversion of dimer components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["dimer"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 2
    assert "D" in chain_ids
    assert "E" in chain_ids

    # Verify polymer annotation
    assert np.all(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_noncanonical(dict_inputs):
    """Test conversion of components with non-canonical residues to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["noncanonical"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "A" in chain_ids

    # Verify SEP residue is present
    res_names = np.unique(atom_array.res_name)
    assert "SEP" in res_names


def test_components_to_atom_array_ligand(dict_inputs):
    """Test conversion of ligand components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["ligand"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "F" in chain_ids

    # Verify non-polymer annotation
    assert not np.any(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.NON_POLYMER)

    # Assert that all elements are letters, not string representations of numbers
    assert all(s.isupper() for s in atom_array.element)


def test_components_to_atom_array_glycan(dict_inputs, bonds_glycan_glycan):
    """Test conversion of ligand components to AtomArray."""
    components = dict_inputs["glycan_1"] + dict_inputs["glycan_2"] + dict_inputs["monomer"]
    atom_array = components_to_atom_array(components, bonds=bonds_glycan_glycan)

    assert isinstance(atom_array, AtomArray)

    # Check only one molecule ID (e.g., all atoms are connected via bonds)
    assert len(np.unique(atom_array.molecule_id)) == 1

    # Check two PN units
    assert len(np.unique(atom_array.pn_unit_id)) == 2

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 3


def test_components_to_atom_array_cif(dict_inputs):
    """Test modification of CIF file inputs and conversion of the modified CIF to AtomArray."""
    component_dicts = dict_inputs["cif_path"] + dict_inputs["dimer"]
    atom_array = components_to_atom_array(component_dicts)

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 4

    # Check chain IIDs
    chain_iids = np.unique(atom_array.chain_iid)
    assert len(chain_iids) == 6  # Two additional iids due to the symmetry transformation in the input CIF
    assert "A_1" in chain_iids
    assert "A_2" in chain_iids

    # Build MSA paths
    components = [ChemicalComponent.from_dict(component_dict) for component_dict in component_dicts]
    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(components)

    # Ensure that the MSA path was assigned correctly
    assert len(msa_paths_by_chain_id) == 1
    assert msa_paths_by_chain_id["A"] == "/example/msa/path.a3m.gz"


def test_chemical_component_from_dict():
    """Test creation of chemical components from dictionaries."""
    # Test sequence component
    seq_dict = {"seq": "ACDEF", "chain_type": "polypeptide(l)", "is_polymer": True}
    comp = ChemicalComponent.from_dict(seq_dict)
    assert isinstance(comp, SequenceComponent)

    # Test SMILES component
    smiles_dict = {"smiles": "CCCCCCCCCCCCCC(=O)O", "chain_type": "non-polymer", "is_polymer": False}
    comp = ChemicalComponent.from_dict(smiles_dict)
    assert isinstance(comp, SmilesComponent)

    # Test invalid component
    with pytest.raises(ValueError):
        ChemicalComponent.from_dict({"invalid": "component"})


def test_read_chai_fasta(chai_fasta_input):
    """Test reading components from FASTA file."""
    components = read_chai_fasta(chai_fasta_input)

    assert len(components) == 4
    assert isinstance(components[0], Protein)
    assert isinstance(components[1], Protein)
    assert isinstance(components[2], Protein)
    assert isinstance(components[3], SmilesComponent)

    # Check the ligand
    assert components[3].smiles == "CCCCCCCCCCCCCC(=O)O"
    assert components[3].chain_type == ChainType.NON_POLYMER
    assert components[1].chain_type == ChainType.POLYPEPTIDE_L
    assert components[2].is_polymer
    assert not components[3].is_polymer


def test_sequence_component_validation():
    """Test validation of sequence components."""
    # Test invalid polymer flag for SMILES
    with pytest.raises(ValueError):
        SmilesComponent(smiles="CCCC", is_polymer=True)

    # Test invalid chain type for SMILES
    with pytest.raises(ValueError):
        SmilesComponent(smiles="CCCC", chain_type=ChainType.POLYPEPTIDE_L)

    with pytest.raises(ValueError):
        SequenceComponent(seq="(MET)ACGT", chain_type=ChainType.RNA, is_polymer=True)

    with pytest.raises(ValueError):
        SequenceComponent(seq="(DA)MKL", chain_type=ChainType.POLYPEPTIDE_L, is_polymer=True)


def test_full_chai_input(chai_fasta_input):
    components = read_chai_fasta(chai_fasta_input)
    atom_array = components_to_atom_array(components)

    assert isinstance(atom_array, AtomArray)
    assert np.unique(atom_array.chain_id).shape[0] == 4


def test_full_components_input(dict_inputs, custom_residues):
    components = sum(dict_inputs.values(), start=[])
    atom_array, components = components_to_atom_array(
        components, return_components=True, custom_residues=custom_residues
    )

    # Assert that the extracted chain IDs match the values recovered from the components
    extracted_chain_ids = [entry.get("chain_id", "") for entries in dict_inputs.values() for entry in entries]

    for index, chain_id in enumerate(extracted_chain_ids):
        if chain_id:
            chain_id_from_component = components[index].chain_id
            assert (
                chain_id == chain_id_from_component
            ), f"Mismatch at index {index}: {chain_id} != {chain_id_from_component}"

    # Sanity check outputs
    assert isinstance(atom_array, AtomArray)
    assert (
        np.unique(atom_array.chain_id).shape[0] == 11
    )  # 2 from CIF, 1 monomer, 2 dimers, 1 noncanonical, 1 custom residue, 1 ligand, 2 glycans, 1 SDF (HEM)

    # fmt: off
    # Note that duplicate chain IDs have been merged
    assert set(np.unique(atom_array.chain_id)) == { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K" }
    # fmt: on

    assert set(np.unique(atom_array.chain_type)) == {ChainType.POLYPEPTIDE_L, ChainType.NON_POLYMER}


def test_sdf_input(dict_inputs):
    components = dict_inputs["monomer"] + dict_inputs["sdf"]
    atom_array = components_to_atom_array(components)

    # Assert that all coordinates are present for the non-polymers
    non_polymer_atom_array = atom_array[atom_array.chain_type == ChainType.NON_POLYMER]
    assert not np.any(np.isnan(non_polymer_atom_array.coord))


def test_custom_residues(dict_inputs, custom_residues):
    # (Name of the custom residue within the CIF file)
    custom_residue_name = "C:0"

    atom_array = components_to_atom_array(dict_inputs["custom_residues"], custom_residues=custom_residues)

    # ... all atoms should be part of the same chain (only one chain in the example)
    assert len(np.unique(atom_array.chain_id)) == 1

    # ... all atoms should be polymers
    assert np.all(atom_array.chain_type == ChainType.POLYPEPTIDE_L)
    assert np.all(atom_array.is_polymer)

    # (Bonds)
    # ... get inter-residue bonds
    bonds = atom_array.bonds.as_array()
    inter_residue_bond_mask = atom_array.res_name[bonds[:, 0]] != atom_array.res_name[bonds[:, 1]]

    # ... ensure the custom residue is present
    atoms_a = atom_array[bonds[inter_residue_bond_mask, 0]]
    atoms_b = atom_array[bonds[inter_residue_bond_mask, 1]]
    assert np.any(atoms_a.res_name == custom_residue_name)
    assert np.any(atoms_b.res_name == custom_residue_name)


def test_recover_bonds_from_cif(dict_inputs):
    data = parse(
        filename=TEST_DATA_IO / "test_unl_ligand_with_bonds.cif",
        fix_ligands_at_symmetry_centers=False,
    )
    atom_array = data["asym_unit"][0]

    assert all(atom_array.res_name == "UNL")
    assert len(atom_array) == 28
    assert atom_array.bonds.as_array().shape[0] == 32


def test_same_atom_array_from_cif_and_inference():
    """Tests if the bonds inferred from the components are the same as the bonds in the CIF file."""
    transformation_id = "1"
    data = parse(get_pdb_path("7rxs"), hydrogen_policy="remove")
    atom_array_from_cif = data["assemblies"][transformation_id][0]

    # ... extract the sequence and build inference input
    monomer = [
        {
            "seq": data["chain_info"]["A"]["unprocessed_entity_non_canonical_sequence"],
            "chain_type": data["chain_info"]["A"]["chain_type"],
            "chain_id": "A",
            "is_polymer": data["chain_info"]["A"]["is_polymer"],
        }
    ]
    ligand = [
        {
            "smiles": "Cc1cc(cc(c1)Oc2nccc(n2)c3c(ncn3[C@H]4CCN(C4)CCN)c5ccc(cc5)I)C",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "C",
        }
    ]

    atom_array_from_inference = components_to_atom_array(monomer + ligand)

    for chain_id in np.unique(atom_array_from_cif.chain_id):
        chain_atom_array_from_inference = atom_array_from_inference[(atom_array_from_inference.chain_id == chain_id)]
        chain_atom_array_from_cif = atom_array_from_cif[atom_array_from_cif.chain_id == chain_id]

        # Inference should have full occupancy and null b_factor
        assert np.all(chain_atom_array_from_inference.occupancy == 1.0)
        assert np.all(np.isnan(chain_atom_array_from_inference.b_factor))

        # Assert same atom array
        annotations_to_compare = list(
            set(chain_atom_array_from_cif.get_annotation_categories())
            - {
                "occupancy",
                "b_factor",
                "is_aromatic",
                "alt_atom_id",
                "molecule_id",  # The molecule_id and all entity annotations may differ between the two
                "molecule_iid",
                "molecule_entity",
                "pn_unit_entity",
            }
        )
        is_ligand = not chain_atom_array_from_cif.is_polymer[0]

        if is_ligand:
            # Renumber residues to be 0-indexed for non-polymers, 1-indexed for polymers
            chain_atom_array_from_cif.res_id = struc.spread_residue_wise(
                chain_atom_array_from_cif, np.arange(struc.get_residue_count(chain_atom_array_from_cif))
            )
            # ... and don't compare residue names, atom names, stereo annotations, as that won't work for UNL
            for item in ["res_name", "atom_name", "stereo"]:
                annotations_to_compare.remove(item)
        else:
            chain_atom_array_from_cif.res_id = struc.spread_residue_wise(
                chain_atom_array_from_cif, np.arange(1, struc.get_residue_count(chain_atom_array_from_cif) + 1)
            )

        assert_same_atom_array(
            chain_atom_array_from_inference,
            chain_atom_array_from_cif,
            compare_coords=False,
            compare_bonds=True,
            annotations_to_compare=annotations_to_compare,
            enforce_order=False,
            compare_bond_order=True,
        )


if __name__ == "__main__":
    pytest.main([__file__ + "::test_full_components_input"])
