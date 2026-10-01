"""PyTest function to check whether past problematic entries run through without error."""

import pytest

from atomworks.ml.preprocessing.preprocess import preprocess
from atomworks.ml.utils.testing import get_pdb_mirror_path
from tests.ml.preprocessing.conftest import TEST_CONFIG

"""
PDB IDs with unusual characteristics that we will ensure run through the data preprocessing pipeline
without error. We comment out examples that are included elsewhere, but leave them in the master list
for reference.
"""
EDGE_CASE_LIST = [
    # "1A8O",  # Single-chain protein complex with non-trivial symmetry
    # "1IVO",  # Protein with glycosylation, with covalently attached ligands
    "3K4A",  # Proteins with modified amino acids (selenomethionine)
    "3KFA",  # No `struct_conn` field, which can cause errors
    # "6WJC",  # Molecule  with "Subject of Investigation" label; missing residues up until label_seq_id 26
    "1EN2",  # Proteins with sequence heterogeneity
    "1CBN",  # Proteins with sequence heterogeneity
    "133D",  # Very simple DNA molecule (6 bases)
    "4JS1",  # Glycol (oligosaccharide) covalently bound to a protein (oligosaccharide with multiple residues)
    "1L2Y",  # Simple, single-residue protein solved through NMR (multiple models)
    "2K0A",  # Heinous multi-chain NMR structure
    "4CPA",  # Molecule with unknown or ambiguous element (marked with 'X')
    # "1ZY8",  # Incorrect in the legacy parser. An FAD ligand, (P, 4750, FAD), has two alternative locations; in the label-assigned ID's (and in PyMol) those are correctly noted, but they have different author residue ID's and thus are both present in the legacy parser.
    "6DMH",  # Incorrect in the legacy parser. Waters with multiple occupancies not resolved correctly.
    # "1FU2",  # Simple, small example with multiple chains
    # "6DMG",  # Multiconformer ligand (covered by regression tests)
    "1Y1W",  # Protein-nucleic-acid complex
    # "5XNL",  # Another ribosomal monstrous molecule to limit-test loading speeds (slow to load; commenting out)
    # "2E2H",  # Complex with protein, DNA, and RNA (slow to load; commenting out)
    # "4NDZ",  # Large assembly with two ligand chains covalently bound together; also has non-biological bonds
    # "3NE7",  # Small assembly with two ligand chains covalently bound together. Also has an "UNL" residue (unknown ligand). Further complicated by the fact that only atoms in one occupancy state are covalently bonded between the two ligands, but not in the other.
    "3NEZ",  # Sequence heterogeneity with non-standard residues (CH6/NRQ), with equal occupancy. Requires additional checks when loading covalent bonds
    # "1RXZ",  # Single protein chain in complex with a short peptide (11 residues) with non-trivial symmetry (multiple transformations)
    # "3J31",  # Enormous molecule (viral capsid - very large bioassembly) (slow to load; commenting out)
    # "7MUB",  # Has a clashing molecule, which is also a LOI - has a potassium at symmetry center
    # "1QK0",  # Has 3 LOIs declared in the cif file, but 1 LOI on the PDB summary page / FTP service
    "1DYL",  # Non-numeric transformation ID
    "7SBV",  # Oligosaccharide defined as separate chains
    "3EPC",  # Invalid index to scalar
    "6O7K",  # Empty coordinates after filtering
    # "104D",  # DNA/RNA Hybrid (covered by regression tests)
    "5X3O",  # polypeptide(D)
    "5GAM",  # Complex with proteins and RNA; used in MSA tests
    "6A5J",  # Small peptide, used in MSA tests (ensure no MSA)
    # "3NE2",  # Manageable-size example with two simple proteins (covered by regression tests)
    "1MNA",  # Simple homomer (for MSA testing)
    "1HGE",  # Simple heteromer (for MSA testing)
    "3EJJ",  # Simple heteromer (for MSA testing)
    "112M",  # Protein-ligand, no LOI, heme ligand
    # "1A3G",  # Involves covalent modification, protein-ligand (covered by regression tests)
    "1A2N",  # Protein-protein homeric interface
    "1A2Y",  # Protein-protein heteromeric interface
    "1BDV",  # Protein-nucleic acid interface
    "184D",  # DNA strands with MG ions around
    "4HF4",  # Includes zinc ion, ligand, protein
    "3LPV",  # DNA
    "2NVZ",  # RNA (large structure, overall)
    # "5OCM",  # Domain swapped homo-dimer with small molecule ligands
]


@pytest.mark.slow
@pytest.mark.parametrize("test_case", EDGE_CASE_LIST)
def test_prior_bugs_and_edge_cases(test_case):
    """Runs data loading for a list of prior problematic entries to ensure they run through without error."""
    assemblies, pn_units, interfaces = preprocess(get_pdb_mirror_path(test_case), TEST_CONFIG)
    assert assemblies is not None


def examine_specific_case(pdb_id):
    """Used for debugging"""
    assemblies, pn_units, interfaces = preprocess(get_pdb_mirror_path(pdb_id), TEST_CONFIG)
    assert assemblies is not None
