"""Convenience utils for working with (generalized) FASTA files."""

import logging
import os
import re
from collections import Counter

from atomworks.constants import (
    CCD_MIRROR_PATH,
    STANDARD_AA_ONE_LETTER,
    STANDARD_DNA,
    STANDARD_DNA_ONE_LETTER,
    STANDARD_RNA,
)
from atomworks.enums import ChainType
from atomworks.io.utils.ccd import (
    check_ccd_codes_are_available,
)
from atomworks.io.utils.sequence import get_3_from_1_letter_code

logger = logging.getLogger("atomworks.io")


def split_generalized_fasta_sequence(sequence: str) -> list[str]:
    """
    Splits a sequence at each letter, keeping groups with parentheses intact.

    Args:
        - sequence (str): The input sequence to be split.

    Returns:
        - List[str]: A list of individual letters and/or groups with parentheses.

    Example:
        >>> split_generalized_fasta_sequence("ABC(DEF)GH(IJ)K")
        ['A', 'B', 'C', '(DEF)', 'G', 'H', '(IJ)', 'K']
    """
    pattern = r"\([^)]*\)|\w"
    return re.findall(pattern, sequence)


def one_letter_to_ccd_code(
    seq: list[str], chain_type: ChainType, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, check_ccd_codes: bool = True
) -> list[str]:
    """
    Convert a sequence of one-letter codes or parenthesized full CCD IDs to full CCD IDs.

    This function takes a list of either one-letter amino acid codes or parenthesized CCD IDs and
    converts them to their corresponding full CCD (Chemical Component Dictionary) IDs. It handles
    both standard amino acids and non-standard chemical components.

    Args:
        seq (list[str]): A list of one-letter codes or parenthesized CCD IDs.
        chain_type (ChainType): The type of chain (e.g., POLYPEPTIDE_L, DNA, RNA) to determine the correct
            conversion for one-letter codes.
        check_ccd_codes (bool): If True, check if the CCD IDs are available in the CCD mirror.

    Returns:
        - list[str]: A list of full CCD IDs corresponding to the input sequence.

    Raises:
        - ValueError: If a non-standard chemical component ID is not found in the processed CCD.

    Example:
        >>> seq = ["A", "C", "(SEP)", "G", "H"]
        >>> chain_type = ChainType.POLYPEPTIDE_L
        >>> one_letter_to_ccd_code(seq, chain_type)
        ['ALA', 'CYS', 'SEP', 'GLY', 'HIS']
    """
    seq_with_ccd_ids = []
    for chem_comp_id in seq:
        if "(" in chem_comp_id:
            # ... this is a non-standard chemical component that only has a unique
            #     >1 letter code

            # ... remove the parentheses and yield the 3-letter code
            chem_comp_id = chem_comp_id.strip("()")

            # ... ensure it is contained in the CCD mirror
            if check_ccd_codes:
                check_ccd_codes_are_available([chem_comp_id], ccd_mirror_path=ccd_mirror_path, mode="raise")

        else:
            chem_comp_id = get_3_from_1_letter_code(chem_comp_id, chain_type=chain_type)

        seq_with_ccd_ids.append(chem_comp_id)

    return seq_with_ccd_ids


def infer_chain_type_from_one_letter(seq: str | list[str]) -> ChainType:
    """Infer chain type from one-letter sequence notation.

    Supports all common sequence input formats:

    - Simple one-letter: ``"ACDEFG"``
    - Parenthesized notation: ``"(DA)(DT)(DG)"`` (PDB format for DNA/RNA disambiguation)
    - Mixed with non-canonical amino acids: ``"ACDE(SEP)FG"``

    Args:
      seq: Sequence as string or list (supports all notation types).

    Returns:
      Inferred chain type (POLYPEPTIDE_L, DNA, or RNA).

    Raises:
      ValueError: If chain type cannot be inferred from the sequence.

    See Also:
      :py:func:`~atomworks.io.utils.non_rcsb.infer_chain_type_from_three_letter` -
      For CCD code arrays from parsed structures.
    """
    # Convert string to list if necessary
    if isinstance(seq, str):
        seq = split_generalized_fasta_sequence(seq)

    # Define one-letter code sets
    protein_codes = set(STANDARD_AA_ONE_LETTER)
    dna_codes = set(STANDARD_DNA_ONE_LETTER)
    rna_codes = set(STANDARD_RNA)

    hits = Counter()
    for letter in seq:
        if letter.startswith("("):
            # Parenthesized notation - strip and check against CCD codes
            ccd_code = letter.strip("()")
            if ccd_code in STANDARD_DNA:
                # DNA is commonly provided like (DA), (DT), etc.
                hits["dna"] += 1
            elif ccd_code in STANDARD_RNA:
                # We also support RNA codes like (A), (U), etc; though these are less common
                hits["rna"] += 1
            else:
                hits["unknown"] += 1
        else:
            # Single-letter code checks
            if letter in protein_codes:
                hits["protein"] += 1
            if letter in dna_codes:
                hits["dna"] += 1
            if letter in rna_codes:
                hits["rna"] += 1

    # Heuristics:
    # If the sequence contains more protein hits than DNA or RNA hits, it's probably a protein
    if hits["protein"] > hits["dna"] and hits["protein"] > hits["rna"]:
        return ChainType.POLYPEPTIDE_L

    # Else, if the sequence is all RNA hits, it's probably RNA
    elif hits["rna"] == len(seq):
        return ChainType.RNA

    # Else, if the sequence is all DNA hits, it's probably DNA
    elif hits["dna"] == len(seq):
        return ChainType.DNA

    raise ValueError(f"Could not infer chain type from sequence: {seq=}")
