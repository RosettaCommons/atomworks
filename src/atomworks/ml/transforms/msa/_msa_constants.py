"""Constants used internally by the MSA transforms"""

import numpy as np

from atomworks.enums import ChainType
from atomworks.io.utils.sequence import get_3_from_1_letter_code

# Special codes for MSA sequences
GAP_THREE_LETTER = "<G>"
GAP_ONE_LETTER = "-"


def create_lookup_table(one_letter_to_int: dict, fallback_letter: str) -> np.ndarray:
    """Create a numpy lookup table from int8 one-letter codes to integer indices.

    Args:
        one_letter_to_int (dict): Dictionary mapping one-letter codes to integer indices.
        fallback_letter (str): The one-letter code to use as a fallback for undefined inputs.

    Returns:
        numpy.ndarray: A lookup table of length 256 for advanced indexing.
    """
    # ...ensure the fallback letter is in the dictionary
    if fallback_letter not in one_letter_to_int:
        raise ValueError(f"Fallback letter '{fallback_letter}' must be in the input dictionary.")

    # ...create a lookup table initialized with the fallback value
    lookup_table = np.full(256, one_letter_to_int[fallback_letter], dtype=np.int8)

    # ...fill in the lookup table with the provided mappings
    for letter, index in one_letter_to_int.items():
        # ...convert the letter to its ASCII value
        ascii_value = ord(letter)
        # ...ensure we're only dealing with 8-bit ASCII within the lookup table
        if ascii_value < 256:
            lookup_table[ascii_value] = index
        else:
            raise ValueError(f"Invalid ASCII value {ascii_value} for letter '{letter}")

    return lookup_table


AMINO_ACID_ONE_LETTER_TO_INT = {
    # Canonical amino acids
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "E": 5,
    "Q": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    # Gap
    "-": 20,
    # Unknown
    "X": 21,
    # Ambiguous
    "B": 22,  # Asparagine or aspartic acid
    "Z": 23,  # Glutamine or glutamic acid
    "J": 24,  # Leucine or isoleucine
    # Rare amino acids
    "U": 25,  # Selenocysteine, the 21st amino acid; analogue of the more common cystein with selenhium in lieu of sulfer (encoded by UGA codon)
    "O": 26,  # Pyrrolysine, the 22nd amino acid; found in some archaea and bacteria (encoded by UAG codon)
}
"""
Ordered list of protein amino acid one-letter codes, including gaps, ambiguous, and rare amino acids.

References:
    `IUPAC Amino Acid Codes <https://iupac.qmul.ac.uk/AminoAcid/A2021.html#AA21>`_
    `Pyrollisine <https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html>`_
"""

RNA_NUCLEOTIDE_ONE_LETTER_TO_INT = {
    # Canonical RNA residues (starting from 27 to avoid overlap with amino acids)
    "A": 27,
    "C": 28,
    "G": 29,
    "U": 30,
    "T": 30,  # Thymine is a DNA residue but is often used in RNA sequences; it should be treated as a synonym for U, so we map it to the same index
    # Gap
    "-": 20,  # Map gap to the same index as the amino acid gap
    # Unknown
    "N": 31,  # Any RNA nucleotide (A, C, G, or U)
    # Ambiguity codes
    # Reference:
    "R": 32,  # Purine (A or G)
    "K": 33,  # Keto (G or T)
    "S": 34,  # Strong (G or C)
    "Y": 35,  # Pyrimidine (C or T)
    "M": 36,  # Amino (A or C)
    "W": 37,  # Weak (A or T)
    "B": 38,  # Not A (C, G, or T)
    "H": 39,  # Not G (A, C, or T)
    "D": 40,  # Not C (A, G, or T)
    "V": 41,  # Not T (A, C, or G)
}
"""
Ordered list of RNA nucleotide one-letter codes, including gaps, ambiguous, and rare residues.

Reference:
    `IUPAC Ambiguity Codes for Nucleotide Degeneracy <https://www.promega.com/resources/guides/nucleic-acid-analysis/restriction-enzyme-resource/restriction-enzyme-resource-tables/iupac-ambiguity-codes-for-nucleotide-degeneracy/>`_
"""

# Create lookup tables from MSA one letter codes to integers, based on the above mappings
AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE = create_lookup_table(AMINO_ACID_ONE_LETTER_TO_INT, fallback_letter="X")
RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE = create_lookup_table(
    RNA_NUCLEOTIDE_ONE_LETTER_TO_INT, fallback_letter="N"
)


# Create lookup tables from MSA integers to three-letter codes
def create_msa_integer_to_three_letter() -> dict[int, str]:
    msa_integer_to_three_letter = {}

    # Amino Acids
    for letter, integer in AMINO_ACID_ONE_LETTER_TO_INT.items():
        three_letter = get_3_from_1_letter_code(
            letter=letter,
            chain_type=ChainType.POLYPEPTIDE_L,  # Any protein chain type will do
            gap_one_letter=GAP_ONE_LETTER,
            gap_three_letter=GAP_THREE_LETTER,
        )
        msa_integer_to_three_letter[integer] = three_letter

    # RNA Nucleotides
    for letter, integer in RNA_NUCLEOTIDE_ONE_LETTER_TO_INT.items():
        if letter == "-":
            continue  # Skip the gap, as it's already handled in the amino acid section
        three_letter = get_3_from_1_letter_code(
            letter=letter,
            chain_type=ChainType.RNA,
            gap_one_letter=GAP_ONE_LETTER,
            gap_three_letter=GAP_THREE_LETTER,
        )
        msa_integer_to_three_letter[integer] = three_letter

    return msa_integer_to_three_letter


MSA_INTEGER_TO_THREE_LETTER = create_msa_integer_to_three_letter()
THREE_LETTER_TO_MSA_INTEGER = {v: k for k, v in MSA_INTEGER_TO_THREE_LETTER.items()}
