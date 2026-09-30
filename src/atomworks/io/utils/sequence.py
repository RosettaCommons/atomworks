"""Utility functions for working with monomer sequences."""

__all__ = [
    "get_1_from_3_letter_code",
    "get_3_from_1_letter_code",
]

import functools
import logging
from collections.abc import Sequence

import numpy as np
import toolz

from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    GAP,
    GAP_ONE_LETTER,
    POLYPEPTIDE_D_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
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
    get_chem_comp_type,
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


@functools.cache
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


@functools.cache
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


def infer_chain_type_from_three_letter(ccd_code_seq: Sequence[str]) -> ChainType:
    """Infer chain type from three-letter CCD code arrays.

    Used for parsed structure data where residues are represented as CCD codes.
    Assigns chain type based on the residue chem types of the provided CCD codes.

    Args:
      ccd_code_seq: List of three-letter CCD codes (e.g., ``["ALA", "CYS", "ASP"]``).

    Returns:
      Inferred chain type enum value.

    Examples:
      >>> infer_chain_type_from_three_letter(["ALA", "CYS", "ASP"])
      <ChainType.POLYPEPTIDE_L: ...>
      >>> infer_chain_type_from_three_letter(["DA", "DT", "DG", "DC"])
      <ChainType.DNA: ...>

    See Also:
      :py:func:`~atomworks.io.tools.fasta.infer_chain_type_from_one_letter` -
      For sequence notation (one-letter format).
    """
    chain_type_counts = dict.fromkeys(
        [
            "aa_like",
            ChainType.POLYPEPTIDE_D,
            ChainType.POLYPEPTIDE_L,
            ChainType.DNA,
            ChainType.RNA,
            ChainType.NON_POLYMER,
        ],
        0,
    )

    for res_name in ccd_code_seq:
        chem_comp = get_chem_comp_type(res_name, mode="warn")

        # Increment the count for the appropriate chain type category
        # (All amino acid-like chem types are considered "aa_like")
        if chem_comp in AA_LIKE_CHEM_TYPES:
            chain_type_counts["aa_like"] += 1
            # (We further differentiate between L- and D-polypeptides)
            if chem_comp in POLYPEPTIDE_D_CHEM_TYPES:
                chain_type_counts[ChainType.POLYPEPTIDE_D] += 1
            elif chem_comp in POLYPEPTIDE_L_CHEM_TYPES:
                chain_type_counts[ChainType.POLYPEPTIDE_L] += 1

        # (We differentiate between RNA and DNA)
        elif chem_comp in RNA_LIKE_CHEM_TYPES:
            chain_type_counts[ChainType.RNA] += 1
        elif chem_comp in DNA_LIKE_CHEM_TYPES:
            chain_type_counts[ChainType.DNA] += 1

        # (All other chem types are considered non-polymer)
        else:
            chain_type_counts[ChainType.NON_POLYMER] += 1

    # WARNING: The following logic is heuristic, and may fail in cases of multiple residues types within a chain.

    # If we have both RNA and DNA, set the chain type to RNA/DNA hybrid
    if chain_type_counts[ChainType.RNA] > 0 and chain_type_counts[ChainType.DNA] > 0:
        chain_type = ChainType.DNA_RNA_HYBRID

    #  If we have proteins, set to either L- or D-polypeptide, depending on the counts
    elif chain_type_counts[ChainType.POLYPEPTIDE_L] > 0 or chain_type_counts[ChainType.POLYPEPTIDE_D] > 0:
        # ... if we have equal or more L-polypeptides than D-polypeptides in the chain, set to L-polypeptide
        if chain_type_counts[ChainType.POLYPEPTIDE_L] >= chain_type_counts[ChainType.POLYPEPTIDE_D]:
            chain_type = ChainType.POLYPEPTIDE_L

        # ... if we have more D-polypeptides than L-polypeptides, set to D-polypeptide
        elif chain_type_counts[ChainType.POLYPEPTIDE_L] < chain_type_counts[ChainType.POLYPEPTIDE_D]:
            chain_type = ChainType.POLYPEPTIDE_D

    # If we only have "aa_like", default to "polypeptide(L)"
    elif (
        chain_type_counts["aa_like"] > 0
        and chain_type_counts[ChainType.POLYPEPTIDE_L] == 0
        and chain_type_counts[ChainType.POLYPEPTIDE_D] == 0
    ):
        chain_type = ChainType.POLYPEPTIDE_L

    # ... if we have RNA, set to polyribonucleotide
    elif chain_type_counts[ChainType.RNA] > 0:
        chain_type = ChainType.RNA
    # ... if we have DNA, set to polydeoxyribonucleotide
    elif chain_type_counts[ChainType.DNA] > 0:
        chain_type = ChainType.DNA
    # Otherwise, set to non-polymer (if we have non-polymer residues)
    elif chain_type_counts[ChainType.NON_POLYMER] > 0:
        chain_type = ChainType.NON_POLYMER
    else:
        raise ValueError(f"Could not infer chain type from residue names: {ccd_code_seq}")

    return chain_type


def convert_to_one_letter_sequences(
    res_names: list[str],
    chain_type: ChainType,
) -> tuple[str, str]:
    """Convert 3-letter residue codes to 1-letter (non_canonical, canonical) sequences.

    Args:
      res_names: List of 3-letter residue codes.
      chain_type: Chain type (e.g., protein, RNA, DNA).

    Returns:
      Tuple of (non_canonical_sequence, canonical_sequence) where:
        - non_canonical keeps modified residues as-is (e.g., MSE → X)
        - canonical maps to closest standard residue (e.g., MSE → M)
    """
    non_canonical = "".join(get_1_from_3_letter_code(rn, chain_type, use_closest_canonical=False) for rn in res_names)
    canonical = "".join(get_1_from_3_letter_code(rn, chain_type, use_closest_canonical=True) for rn in res_names)
    return non_canonical, canonical
