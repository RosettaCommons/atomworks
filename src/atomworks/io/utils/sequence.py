"""Utility functions for working with monomer sequences."""

__all__ = [
    "get_1_from_3_letter_code",
    "get_3_from_1_letter_code",
]

import functools
import logging

import numpy as np
import toolz

from atomworks.constants import (
    GAP,
    GAP_ONE_LETTER,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_NA,
    STANDARD_PURINE_RESIDUES,
    STANDARD_PYRIMIDINE_RESIDUES,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType
from atomworks.io.utils.ccd import (
    aa_chem_comps,
    chem_comp_to_one_letter,
    na_chem_comps,
)

logger = logging.getLogger("atomworks.io")


@functools.cache
def aa_chem_comp_3to1(standard_only: bool = False) -> dict[str, str]:
    """Returns a dictionary mapping 3-letter amino acid codes to 1-letter codes.

    Args:
        standard_only: If True, only include standard amino acids.

    Returns:
        Dictionary mapping 3-letter to 1-letter amino acid codes.
    """
    aa_3to1 = toolz.keyfilter(lambda x: x in aa_chem_comps(), chem_comp_to_one_letter())
    if standard_only:
        return toolz.keyfilter(lambda x: x in STANDARD_AA, aa_3to1)
    return aa_3to1


@functools.cache
def na_chem_comp_3to1(standard_only: bool = False) -> dict[str, str]:
    """Returns a dictionary mapping 3-letter DNA codes to 1-letter codes.

    Args:
        standard_only: If True, only include standard nucleic acids.

    Returns:
        Dictionary mapping 3-letter to 1-letter nucleic acid codes.
    """
    na_3to1 = toolz.keyfilter(lambda x: x in na_chem_comps(), chem_comp_to_one_letter())
    if standard_only:
        return toolz.keyfilter(lambda x: x in STANDARD_NA, na_3to1)
    return na_3to1


@functools.cache
def aa_chem_comp_1to3() -> dict[str, str]:
    return {val: key for key, val in aa_chem_comp_3to1(standard_only=True).items()}


@functools.cache
def rna_chem_comp_1to3() -> dict[str, str]:
    """
    Returns a dictionary mapping 1-letter RNA codes to 3-letter codes.
    """
    return {val: key for key, val in na_chem_comp_3to1().items() if key in STANDARD_RNA}


@functools.cache
def dna_chem_comp_1to3() -> dict[str, str]:
    """
    Returns a dictionary mapping 1-letter DNA codes to 3-letter codes.
    """
    return {val: key for key, val in na_chem_comp_3to1().items() if key in STANDARD_DNA}


def get_1_from_3_letter_code(
    res_name: str,
    chain_type: ChainType,
    use_closest_canonical: bool = False,
    gap_three_letter: str = GAP,
    gap_one_letter: str = GAP_ONE_LETTER,
) -> str:
    """
    Converts a 3-letter residue name to its 1-letter code based on the chain type.

    Optionally, the closest canonical mapping can be used.

    Args:
        res_name (str): The 3-letter residue name.
        chain_type (ChainType): The type of chain, using the ChainType enum.
        use_closest_canonical (bool): Whether to use the closest canonical mapping (from BioPython). Defaults to False.
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).

    Returns:
        str: The corresponding 1-letter code. Returns "X" if the residue name or chain type is not supported.
    """
    # ...convert gaps ("<G>") to "-", or whatever is specified
    if res_name == gap_three_letter:
        return gap_one_letter

    if chain_type.is_protein():
        return aa_chem_comp_3to1(standard_only=not use_closest_canonical).get(res_name, "X")
    elif chain_type.is_nucleic_acid():
        return na_chem_comp_3to1(standard_only=not use_closest_canonical).get(res_name, "N")
    else:
        logger.info(f"Unsupported chain type: {chain_type}")
        return "X"


def get_3_from_1_letter_code(
    letter: str,
    chain_type: ChainType,
    gap_one_letter: str = GAP_ONE_LETTER,
    gap_three_letter: str = GAP,
) -> str:
    """
    Converts a 1-letter residue name to its 3-letter code based on the chain type.

    Note:
        Converting from a three-letter, to a one-letter, back to a three-letter
        code is not invertible (i.e., 1:1) and may result in a different three-letter sequence.

    Args:
        letter (str): The 1-letter residue name.
        chain_type (ChainType): The type of chain, using the ChainType enum.
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".

    Returns:
        str: The corresponding 3-letter code.
    """
    assert len(letter) == 1, "The 1-letter code must be a single character."

    # Convert gaps (-) to "<G>", or whatever is specified
    if letter == gap_one_letter:
        return gap_three_letter

    if chain_type.is_protein():
        # Proteins
        return aa_chem_comp_1to3().get(letter, UNKNOWN_AA)
    elif chain_type == ChainType.DNA:
        # DNA
        return dna_chem_comp_1to3().get(letter, UNKNOWN_DNA)
    elif chain_type == ChainType.RNA:
        # RNA
        return rna_chem_comp_1to3().get(letter, UNKNOWN_RNA)
    else:
        logger.error(f"Unsupported {chain_type=}, returning unknown protein residue {UNKNOWN_AA=}.")
        return UNKNOWN_AA


def is_pyrimidine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_PYRIMIDINE_RESIDUES)


def is_purine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_PURINE_RESIDUES)


def is_unknown_nucleotide(ccd_code_array: np.ndarray) -> np.ndarray:
    ccd_code_array = np.asarray(ccd_code_array)
    return (ccd_code_array == UNKNOWN_DNA) | (ccd_code_array == UNKNOWN_RNA)


def is_standard_aa(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_AA)


def is_glycine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.asarray(ccd_code_array) == "GLY"


def is_standard_aa_not_glycine(ccd_code_array: np.ndarray) -> np.ndarray:
    _aa_not_gly = [res for res in STANDARD_AA if res != "GLY"]
    return np.isin(ccd_code_array, _aa_not_gly)


def is_protein_unknown(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.asarray(ccd_code_array) == UNKNOWN_AA
