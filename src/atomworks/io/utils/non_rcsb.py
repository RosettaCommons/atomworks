"""
Utility functions for handling non-RCSB CIF files.
Such files do not follow the standard CIF format and thus may require special handling.
"""

__all__ = [
    "get_identity_assembly_gen_category",
    "get_identity_op_expr_category",
    "initialize_chain_info_from_atom_array",
]

import logging
from collections.abc import Sequence

import numpy as np
from biotite.structure import AtomArray
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
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code

logger = logging.getLogger("atomworks.io")


def infer_chain_type_from_ccd_codes(ccd_code_seq: Sequence[str]) -> ChainType:
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

    # ... infer the chain type based on each residue
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

    # ... if we have both RNA and DNA, set the chain type to RNA/DNA hybrid
    if chain_type_counts[ChainType.RNA] > 0 and chain_type_counts[ChainType.DNA] > 0:
        chain_type = ChainType.DNA_RNA_HYBRID

    # ... if we have proteins, set to either L- or D-polypeptide, depending on the counts
    elif chain_type_counts[ChainType.POLYPEPTIDE_L] > 0 or chain_type_counts[ChainType.POLYPEPTIDE_D] > 0:
        # ... if we have equal or more L-polypeptides than D-polypeptides in the chain, set to L-polypeptide
        if chain_type_counts[ChainType.POLYPEPTIDE_L] >= chain_type_counts[ChainType.POLYPEPTIDE_D]:
            chain_type = ChainType.POLYPEPTIDE_L

        # ... if we have more D-polypeptides than L-polypeptides, set to D-polypeptide
        elif chain_type_counts[ChainType.POLYPEPTIDE_L] < chain_type_counts[ChainType.POLYPEPTIDE_D]:
            chain_type = ChainType.POLYPEPTIDE_D

    # ... if we only have "aa_like", default to "polypeptide(L)"
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
    # ... otherwise set to non-polymer, if we have non-polymer residues
    elif chain_type_counts[ChainType.NON_POLYMER] > 0:
        chain_type = ChainType.NON_POLYMER
    else:
        raise ValueError(f"Could not infer chain type from residue names: {ccd_code_seq}")

    return chain_type


def initialize_chain_info_from_atom_array(
    atom_array: AtomArray,
    infer_chain_type: bool = True,
    infer_chain_sequences: bool = True,
    use_chain_iids: bool = False,
) -> dict:
    """Infer specified information (chain type and sequence information) from an AtomArray.

    Required when the chain type or polymer sequence information is not explicitly provided in the CIF file.
    Such situations may arise when the CIF file is not from the RCSB PDB database (e.g., distillation).

    WARNING: Use this function for computationally predicted structures or for inference; do not use for files from the RCSB PDB database.

    In particular, this function adds the following information to the chain_info_dict:
        - The RCSB entity ID for each chain (e.g., 1, 2, 3, etc.), if present in the AtomArray (under the entity_id atom site label)
        - The unprocessed one-letter entity canonical and non-canonical sequences.
        - (Optionally) A boolean flag indicating whether the chain is a polymer.
        - (Optionally) The chain type as an IntEnum (e.g., polypeptide(L), non-polymer, etc.)
        - (Optionally) The residue IDs and residue names, inferred from the AtomArray.

    Args:
        atom_array (AtomArray): The AtomArray object to infer chain information from.
        infer_chain_type (bool, optional): Whether to infer the chain type from the AtomArray. Defaults to True.
        infer_chain_sequences (bool, optional): Whether to infer the chain sequences from the AtomArray. Defaults to True.
        use_chain_iids (bool, optional): Whether to use the chain_iid annotation rather than the chain_id. Defaults to False.

    Returns:
        dict: A dictionary containing the specified chain information as keys
    """
    logger.info(
        "Could not read ChainType from CIF file, inferring from AtomArray (ensure this is the correct behavior)!"
    )
    chain_info_dict = {}

    _res_starts = get_residue_starts(atom_array)
    if use_chain_iids:
        chain_identifiers = atom_array.chain_iid[_res_starts]
    else:
        chain_identifiers = atom_array.chain_id[_res_starts]
    res_ids = atom_array.res_id[_res_starts]
    res_names = atom_array.res_name[_res_starts]
    hetero = atom_array.hetero[_res_starts]

    for chain_identifier in np.unique(chain_identifiers):
        if not chain_identifier:
            raise ValueError(
                'Chain identifier is empty! Please ensure that in your input file, each chain has a unique identifier (e.g., `label_asym_id` in a CIF file cannot be "").'
            )

        is_in_chain = chain_identifiers == chain_identifier
        seq = res_names[is_in_chain]

        # ... EDGE CASE: If all atoms are "HETATM", override the chain type to non-polymer
        if np.all(hetero[is_in_chain]):
            chain_type = ChainType.NON_POLYMER
        else:
            chain_type = infer_chain_type_from_ccd_codes(seq)

        processed_entity_non_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=False) for ccd_code in seq
        )
        processed_entity_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=True) for ccd_code in seq
        )

        chain_info_dict[chain_identifier] = {}

        if "label_entity_id" in atom_array.get_annotation_categories():
            # (Entity ID is the same for all atoms in the chain)
            atom_level_chain_identifiers = atom_array.chain_iid if use_chain_iids else atom_array.chain_id
            chain_info_dict[chain_identifier]["rcsb_entity"] = int(
                atom_array.label_entity_id[atom_level_chain_identifiers == chain_identifier][0]
            )

        if infer_chain_type:
            chain_info_dict[chain_identifier]["chain_type"] = chain_type
            chain_info_dict[chain_identifier]["is_polymer"] = chain_type.is_polymer()

        if infer_chain_sequences:
            chain_info_dict[chain_identifier]["res_id"] = list(res_ids[is_in_chain])
            chain_info_dict[chain_identifier]["res_name"] = list(res_names[is_in_chain])
            chain_info_dict[chain_identifier]["processed_entity_non_canonical_sequence"] = (
                processed_entity_non_canonical_sequence
            )
            chain_info_dict[chain_identifier]["processed_entity_canonical_sequence"] = (
                processed_entity_canonical_sequence
            )

    return chain_info_dict


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
