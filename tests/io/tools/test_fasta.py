"""Tests for FASTA utilities."""

import pytest

from atomworks.enums import ChainType
from atomworks.io.tools.fasta import infer_chain_type_from_one_letter


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
