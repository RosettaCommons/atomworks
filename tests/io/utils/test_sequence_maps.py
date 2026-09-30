import re

import numpy as np
import pytest

from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.tools.fasta import infer_chain_type_from_one_letter
from atomworks.io.utils.sequence import (
    get_1_from_3_letter_code,
    get_3_from_1_letter_code,
    infer_chain_type_from_three_letter,
)
from tests.io.conftest import get_pdb_path

SEQUENCE_TEST_CASES = ["155c", "2e2h", "4cpa", "1en2", "1aqc", "1ivo", "3k4a", "1cbn", "133d", "1l2y", "3nez"]


def non_canonical_sequence_length(s):
    """Calculate the length of a non-canonical sequence string.

    Example:
        >>> custom_length("(ABC)FDS(DCS)")
        5
    """
    return len(re.sub(r"\(.*?\)", "(", s))


@pytest.mark.parametrize("pdb_id", SEQUENCE_TEST_CASES)
def test_parser_one_letter_sequence_outputs(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        build_assembly="all",
        fix_arginines=False,
        convert_mse_to_met=False,
    )

    chain_info = result["chain_info"]
    for chain_id, chain_details in chain_info.items():
        chain_type = chain_details["chain_type"]
        # Get the atom array for that specific chain and count the number of unique residues
        atom_array = result["asym_unit"][0]  # First model
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        num_residues = len(np.unique(chain_atom_array.res_id))

        if (
            chain_type == "polypeptide(D)"
            or chain_type == "polypeptide(L)"
            or chain_type == "polydeoxyribonucleotide"
            or chain_type == "polyribonucleotide"
        ):
            unprocessed_entity_canonical_sequence = chain_details["unprocessed_entity_canonical_sequence"]
            unprocessed_entity_non_canonical_sequence = chain_details["unprocessed_entity_non_canonical_sequence"]
            processed_entity_canonical_sequence = chain_details["processed_entity_canonical_sequence"]
            processed_entity_non_canonical_sequence = chain_details["processed_entity_non_canonical_sequence"]

            # Ensure that we didn't lose any residues during processing (e.g., unknown residues)
            assert non_canonical_sequence_length(unprocessed_entity_non_canonical_sequence) == num_residues

            # Ensure that the processed canonical and non-canonical sequences have the same length
            assert len(processed_entity_canonical_sequence) == len(processed_entity_non_canonical_sequence)

            # Assert that the unprocessed canonical sequence is at least as long as the processed canonical sequence (due to sequence heterogeneity, or NCAA that map to two AA)
            assert len(unprocessed_entity_canonical_sequence) >= len(processed_entity_canonical_sequence)

            # More concise regex to remove characters: B, Z, X, and also the content within parentheses
            if chain_type == "polypeptide(D)" or chain_type == "polypeptide(L)":
                unprocessed_cleaned = re.sub(r"\(.*?\)|[BZX]", "", unprocessed_entity_non_canonical_sequence)
                processed_cleaned = processed_entity_non_canonical_sequence.replace("X", "")
                assert len(unprocessed_cleaned) == len(processed_cleaned)

            # Ensure that the length of both processed sequences matches the number of residues in the chain
            assert len(processed_entity_canonical_sequence) == num_residues
            assert len(processed_entity_non_canonical_sequence) == num_residues

            # Ensure that the length of the unprocessed entity canonical sequence is >= the length of the processed entity canonical sequence
            assert len(unprocessed_entity_canonical_sequence) >= len(processed_entity_canonical_sequence)

            # If there's no sequence heterogeneity, perform additional checks
            if (
                not chain_details.get("has_sequence_heterogeneity", False)
                and (chain_type == "polypeptide(D)" or chain_type == "polypeptide(L)")
                and (unprocessed_cleaned != processed_cleaned)
            ):
                mismatches = []
                for i, (u, p) in enumerate(zip(unprocessed_cleaned, processed_cleaned, strict=False)):
                    if u != p:
                        mismatches.append(f"position {i + 1}: {u} != {p}")
                    mismatch_details = "\n".join(mismatches)
                    raise AssertionError(
                        f"Sequence mismatch found:\n"
                        f"Unprocessed: {unprocessed_cleaned}\n"
                        f"Processed:   {processed_cleaned}\n"
                        f"Mismatches at:\n{mismatch_details}"
                    )


# Define test cases for proteins
PROTEIN_TEST_CASES = [
    ("A", "polypeptide(D)", "ALA"),
    ("C", "polypeptide(D)", "CYS"),
    ("D", "polypeptide(D)", "ASP"),
    ("E", "polypeptide(D)", "GLU"),
    ("F", "polypeptide(D)", "PHE"),
    ("G", "polypeptide(D)", "GLY"),
    ("H", "polypeptide(D)", "HIS"),
    ("I", "polypeptide(D)", "ILE"),
    ("K", "polypeptide(D)", "LYS"),
    ("L", "polypeptide(D)", "LEU"),
    ("M", "polypeptide(D)", "MET"),
    ("N", "polypeptide(D)", "ASN"),
    ("P", "polypeptide(D)", "PRO"),
    ("Q", "polypeptide(D)", "GLN"),
    ("R", "polypeptide(D)", "ARG"),
    ("S", "polypeptide(D)", "SER"),
    ("T", "polypeptide(D)", "THR"),
    ("V", "polypeptide(D)", "VAL"),
    ("W", "polypeptide(D)", "TRP"),
    ("Y", "polypeptide(D)", "TYR"),
    ("-", "polypeptide(D)", "<G>"),
]

# Define test cases for DNA
DNA_TEST_CASES = [
    ("A", "polydeoxyribonucleotide", "DA"),
    ("C", "polydeoxyribonucleotide", "DC"),
    ("G", "polydeoxyribonucleotide", "DG"),
    ("T", "polydeoxyribonucleotide", "DT"),
    ("-", "polydeoxyribonucleotide", "<G>"),
]

# Define test cases for RNA
RNA_TEST_CASES = [
    ("A", "polyribonucleotide", "A"),
    ("C", "polyribonucleotide", "C"),
    ("G", "polyribonucleotide", "G"),
    ("U", "polyribonucleotide", "U"),
    ("-", "polyribonucleotide", "<G>"),
]

# Define test cases for unknown letters
UNKNOWN_TEST_CASES = [
    ("B", "polypeptide(D)", "UNK"),
    ("Z", "polypeptide(D)", "UNK"),
    ("X", "polypeptide(D)", "UNK"),
    ("B", "polydeoxyribonucleotide", "DN"),
    ("Z", "polydeoxyribonucleotide", "DN"),
    ("X", "polydeoxyribonucleotide", "DN"),
    ("B", "polyribonucleotide", "N"),
    ("Z", "polyribonucleotide", "N"),
    ("X", "polyribonucleotide", "N"),
]


@pytest.mark.parametrize(
    "letter, chain_type, expected_three_letter",
    PROTEIN_TEST_CASES + DNA_TEST_CASES + RNA_TEST_CASES + UNKNOWN_TEST_CASES,
)
def test_get_3_from_1_letter_code(letter, chain_type, expected_three_letter):
    assert get_3_from_1_letter_code(letter, ChainType.as_enum(chain_type)) == expected_three_letter


# We can't test the reverse mapping for unknown letters
@pytest.mark.parametrize(
    "expected_one_letter, chain_type, three_letter_code", PROTEIN_TEST_CASES + DNA_TEST_CASES + RNA_TEST_CASES
)
def test_get_1_from_3_letter_code(three_letter_code, chain_type, expected_one_letter):
    assert get_1_from_3_letter_code(three_letter_code, ChainType.as_enum(chain_type)) == expected_one_letter


@pytest.mark.parametrize(
    "ccd_codes,expected",
    [
        (["ALA", "CYS", "ASP"], ChainType.POLYPEPTIDE_L),
        (["GLY", "PRO", "VAL", "LEU"], ChainType.POLYPEPTIDE_L),
        (["DA", "DT", "DG", "DC"], ChainType.DNA),
        (["A", "U", "G", "C"], ChainType.RNA),
    ],
)
def test_infer_chain_type_from_three_letter(ccd_codes, expected):
    """Test the new function name."""
    result = infer_chain_type_from_three_letter(ccd_codes)
    assert result == expected


@pytest.mark.parametrize(
    "seq,expected",
    [
        # Simple one-letter protein
        ("ACDEFG", ChainType.POLYPEPTIDE_L),
        (
            "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTET",
            ChainType.POLYPEPTIDE_L,
        ),
        # Simple one-letter DNA
        ("ATGC", ChainType.DNA),
        ("CGCGAATTCGCG", ChainType.DNA),
        # Simple one-letter RNA
        ("ACGU", ChainType.RNA),
        ("CGCGAAUUCGCG", ChainType.RNA),
        # Parenthesized notation - DNA
        ("(DA)(DT)(DG)(DC)", ChainType.DNA),
        (["(DA)", "(DT)", "(DG)", "(DC)"], ChainType.DNA),
        # Parenthesized notation - RNA
        ("(A)(U)(G)(C)", ChainType.RNA),
        # Mixed protein with non-standard - unambiguous protein codes dominate
        ("ACDE(SEP)FG", ChainType.POLYPEPTIDE_L),
        # List format
        (["A", "C", "D", "E", "F"], ChainType.POLYPEPTIDE_L),
        (["A", "T", "G", "C"], ChainType.DNA),
        # Contains valid amino acid Y (tyrosine), so infers as protein
        ("XYZ", ChainType.POLYPEPTIDE_L),
    ],
)
def test_infer_chain_type_from_one_letter(seq, expected):
    """Test chain type inference from sequence notation."""
    result = infer_chain_type_from_one_letter(seq)
    assert result == expected


def test_infer_chain_type_from_one_letter_raises_on_ambiguous():
    """Test that inference raises ValueError for truly ambiguous sequences."""
    # All letters A, C, G are ambiguous (could be protein or nucleotide)
    with pytest.raises(ValueError, match="Could not infer chain type"):
        infer_chain_type_from_one_letter(["A", "C", "(SEP)", "G"])
