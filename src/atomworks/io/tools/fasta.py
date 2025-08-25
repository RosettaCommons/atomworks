"""
Convenience utils for working with (generalized) FASTA files.
"""

import logging
import os
import re

from atomworks.constants import CCD_MIRROR_PATH
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
