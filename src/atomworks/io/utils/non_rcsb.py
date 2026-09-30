"""Utility functions for handling non-RCSB CIF files.

Such files do not follow the standard CIF format and thus may require special handling.
"""

__all__ = [
    "get_identity_assembly_gen_category",
    "get_identity_op_expr_category",
]

import logging
from collections.abc import Sequence

from biotite.structure.io.pdbx import CIFCategory

from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
)
from atomworks.enums import ChainType
from atomworks.io.utils.ccd import get_chem_comp_type

logger = logging.getLogger("atomworks.io")


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


def get_identity_op_expr_category() -> CIFCategory:
    return CIFCategory.deserialize(
        """_pdbx_struct_oper_list.id                   1 
        _pdbx_struct_oper_list.type                 'identity operation' 
        _pdbx_struct_oper_list.name                 1_555 
        _pdbx_struct_oper_list.symmetry_operation   x,y,z 
        _pdbx_struct_oper_list.matrix[1][1]         1.0000000000 
        _pdbx_struct_oper_list.matrix[1][2]         0.0000000000 
        _pdbx_struct_oper_list.matrix[1][3]         0.0000000000 
        _pdbx_struct_oper_list.vector[1]            0.0000000000 
        _pdbx_struct_oper_list.matrix[2][1]         0.0000000000 
        _pdbx_struct_oper_list.matrix[2][2]         1.0000000000 
        _pdbx_struct_oper_list.matrix[2][3]         0.0000000000 
        _pdbx_struct_oper_list.vector[2]            0.0000000000 
        _pdbx_struct_oper_list.matrix[3][1]         0.0000000000 
        _pdbx_struct_oper_list.matrix[3][2]         0.0000000000 
        _pdbx_struct_oper_list.matrix[3][3]         1.0000000000 
        _pdbx_struct_oper_list.vector[3]            0.0000000000 """  # noqa: W291
    )


def get_identity_assembly_gen_category(chain_ids: list[str]) -> CIFCategory:
    return CIFCategory.deserialize(
        f"""_pdbx_struct_assembly_gen.assembly_id 1
        _pdbx_struct_assembly_gen.oper_expression 1
        _pdbx_struct_assembly_gen.asym_id_list {",".join(chain_ids)}
        """
    )
