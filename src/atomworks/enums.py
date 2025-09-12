"""Enums used across atomworks."""

from enum import IntEnum, StrEnum, auto
from types import MappingProxyType
from typing import Final, Union

import numpy as np
from toolz import keymap

from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
)


class ChainType(IntEnum):
    """IntEnum representing the type of chain in a RCSB mmCIF file from the Protein Data Bank (PDB).

    Useful constants relating to ChainType are defined in :class:`ChainTypeInfo`.

    Note:
        The chain type fields in the PDB are not stable; note the specific versions
        of the dictionaries used (updated November, 2024)

    References:
        `RCSB mmCIF Dictionary - entity.type <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity.type.html>`_
        `RCSB mmCIF Dictionary - entity_poly.type <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html>`_
    """

    # Polymers
    CYCLIC_PSEUDO_PEPTIDE = 0  # cyclic-pseudo-peptide, from `entity_poly.type`
    OTHER_POLYMER = 1  # other, from `entity_poly.type`
    PEPTIDE_NUCLEIC_ACID = 2  # peptide-nucleic-acid, from `entity_poly.type`
    DNA = 3  # polydeoxyribonucleotide, from `entity_poly.type`
    DNA_RNA_HYBRID = 4  # polydeoxyribonucleotide/polyribonucleotide hybrid, from `entity_poly.type`
    POLYPEPTIDE_D = 5  # polypeptide(D), from `entity_poly.type`
    POLYPEPTIDE_L = 6  # polypeptide(L), from `entity_poly.type`
    RNA = 7  # polyribonucleotide, from `entity_poly.type`

    # Non-polymers
    BRANCHED = 10  # branched, from `entity.type`
    MACROLIDE = 11  # macrolide, from `entity.type`
    NON_POLYMER = 8  # non-polymer, from `entity.type`
    WATER = 9  # water, from `entity.type`

    @classmethod
    def from_string(cls, str_value: str) -> "ChainType":
        """Convert a string to a ChainType enum.

        Args:
            str_value: The string value to convert.

        Returns:
            The corresponding ChainType enum.

        Raises:
            ValueError: If the string value is not a valid chain type.
        """
        try:
            return ChainTypeInfo.STRING_TO_ENUM[str_value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid chain type: {str_value=}. Allowed values: {set(ChainTypeInfo.STRING_TO_ENUM.keys())}"
            ) from None

    @staticmethod
    def get_chain_type_strings() -> list[str]:
        """Get a list of all chain type strings.

        Returns:
            List of all valid chain type strings.
        """
        return list(ChainTypeInfo.STRING_TO_ENUM.keys())

    @staticmethod
    def get_polymers() -> list["ChainType"]:
        """Get a list of all polymer chain types.

        Returns:
            List of polymer chain types.
        """
        return ChainTypeInfo.POLYMERS

    @staticmethod
    def get_non_polymers() -> list["ChainType"]:
        """Get a list of all non-polymer chain types.

        Returns:
            List of non-polymer chain types.
        """
        return ChainTypeInfo.NON_POLYMERS

    @staticmethod
    def get_proteins() -> list["ChainType"]:
        """Get a list of all protein chain types.

        Returns:
            List of protein chain types.
        """
        return ChainTypeInfo.PROTEINS

    @staticmethod
    def get_nucleic_acids() -> list["ChainType"]:
        """Get a list of all nucleic acid chain types.

        Returns:
            List of nucleic acid chain types.
        """
        return ChainTypeInfo.NUCLEIC_ACIDS

    @staticmethod
    def get_all_types() -> list["ChainType"]:
        """Get a list of all chain types.

        Returns:
            List of all chain types.
        """
        return list(ChainType)

    def __eq__(self, other: Union["ChainType", int, str]) -> bool:
        """Check if two ChainType enums are equal.

        Args:
            other: Another ChainType, int, or string to compare with.

        Returns:
            True if the chain types are equal, False otherwise.
        """
        if isinstance(other, ChainType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            try:
                # Attempt to convert the string to a ChainType
                other_chain_type = ChainType.from_string(other)
                return self.value == other_chain_type.value
            except ValueError:
                # Could not convert the string to a ChainType
                return False
        return NotImplemented

    def __hash__(self):
        """Hash a ChainType enum.

        Returns:
            Hash value of the enum.
        """
        return hash(self.value)

    def __str__(self) -> str:
        """Convert a ChainType enum to a string.

        Returns:
            String representation of the chain type.
        """
        return self.to_string()

    def get_valid_chem_comp_types(self) -> set[str]:
        """Get the set of valid chemical component types for a ChainType.

        Returns:
            Set of valid chemical component types for this chain type.
        """
        return ChainTypeInfo.VALID_CHEM_COMP_TYPES[self]

    def is_protein(self) -> bool:
        """Check if a ChainType is a protein.

        Returns:
            True if this chain type represents a protein, False otherwise.
        """
        return self in ChainTypeInfo.PROTEINS

    def is_nucleic_acid(self) -> bool:
        """Check if a ChainType is a nucleic acid.

        Returns:
            True if this chain type represents a nucleic acid, False otherwise.
        """
        return self in ChainTypeInfo.NUCLEIC_ACIDS

    def is_polymer(self) -> bool:
        """Check if a ChainType is a polymer.

        Returns:
            True if this chain type represents a polymer, False otherwise.
        """
        return self in ChainTypeInfo.POLYMERS

    def is_non_polymer(self) -> bool:
        """Check if a ChainType is a non-polymer.

        Returns:
            True if this chain type represents a non-polymer, False otherwise.
        """
        return self in ChainTypeInfo.NON_POLYMERS

    def to_string(self) -> str:
        """Convert a ChainType enum to a string.

        Note:
            Returns UPPERCASE string (e.g., "POLYPEPTIDE(D)" instead of "polypeptide(D)")

        Returns:
            Uppercase string representation of the chain type.
        """
        return ChainTypeInfo.ENUM_TO_STRING[self]

    @staticmethod
    def as_enum(value: Union[str, int, "ChainType"]) -> "ChainType":
        """Convert a string, int, or ChainType to a ChainType enum.

        Args:
            value: The value to convert to a ChainType enum.

        Returns:
            The corresponding ChainType enum.

        Raises:
            ValueError: If the value cannot be converted to a ChainType.
        """
        if isinstance(value, ChainType):
            return value
        elif isinstance(value, str):
            return ChainType.from_string(value)
        elif isinstance(value, int | np.integer):
            return ChainType(value)
        else:
            raise ValueError(f"Invalid value: {value}")


class ChainTypeInfo:
    """Companion class containing metadata and helper methods for ChainType enum.

    This class should not be instantiated - it serves as a namespace for
    ChainType-related constants and utilities.
    """

    POLYMERS: Final[tuple[ChainType, ...]] = (
        ChainType.CYCLIC_PSEUDO_PEPTIDE,
        ChainType.OTHER_POLYMER,
        ChainType.PEPTIDE_NUCLEIC_ACID,
        ChainType.DNA,
        ChainType.DNA_RNA_HYBRID,
        ChainType.POLYPEPTIDE_D,
        ChainType.POLYPEPTIDE_L,
        ChainType.RNA,
    )

    NON_POLYMERS: Final[tuple[ChainType, ...]] = (
        ChainType.BRANCHED,
        ChainType.MACROLIDE,
        ChainType.NON_POLYMER,
        ChainType.WATER,
    )

    PROTEINS: Final[tuple[ChainType, ...]] = (
        ChainType.POLYPEPTIDE_D,
        ChainType.POLYPEPTIDE_L,
        ChainType.CYCLIC_PSEUDO_PEPTIDE,
    )

    NUCLEIC_ACIDS: Final[tuple[ChainType, ...]] = (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)

    STRING_TO_ENUM: Final[MappingProxyType[str, ChainType]] = MappingProxyType(
        keymap(
            str.upper,
            {
                # Polymers
                "CYCLIC-PSEUDO-PEPTIDE": ChainType.CYCLIC_PSEUDO_PEPTIDE,
                "OTHER": ChainType.OTHER_POLYMER,  # WARNING! Paradoxically, "other" is a polymer type.
                "PEPTIDE NUCLEIC ACID": ChainType.PEPTIDE_NUCLEIC_ACID,
                "POLYDEOXYRIBONUCLEOTIDE": ChainType.DNA,
                "POLYDEOXYRIBONUCLEOTIDE/POLYRIBONUCLEOTIDE HYBRID": ChainType.DNA_RNA_HYBRID,
                "POLYPEPTIDE(D)": ChainType.POLYPEPTIDE_D,
                "POLYPEPTIDE(L)": ChainType.POLYPEPTIDE_L,
                "POLYRIBONUCLEOTIDE": ChainType.RNA,
                # Non-polymers
                "BRANCHED": ChainType.BRANCHED,
                "MACROLIDE": ChainType.MACROLIDE,
                "NON-POLYMER": ChainType.NON_POLYMER,
                "WATER": ChainType.WATER,
            },
        )
    )
    """Mapping from chain_type strings to ChainType enums."""

    ENUM_TO_STRING: Final[MappingProxyType[ChainType, str]] = MappingProxyType(
        {v: k for k, v in STRING_TO_ENUM.items()}
    )
    """Mapping from ChainType enums to chain_type strings."""

    VALID_CHEM_COMP_TYPES: Final[MappingProxyType[ChainType, set[str]]] = MappingProxyType(
        {
            ChainType.CYCLIC_PSEUDO_PEPTIDE: AA_LIKE_CHEM_TYPES,
            ChainType.PEPTIDE_NUCLEIC_ACID: AA_LIKE_CHEM_TYPES | DNA_LIKE_CHEM_TYPES | RNA_LIKE_CHEM_TYPES,
            ChainType.DNA: DNA_LIKE_CHEM_TYPES,
            ChainType.DNA_RNA_HYBRID: DNA_LIKE_CHEM_TYPES | RNA_LIKE_CHEM_TYPES,
            ChainType.POLYPEPTIDE_D: POLYPEPTIDE_D_CHEM_TYPES
            | {"PEPTIDE LINKING"},  # GLY counts as a peptide linking without L/D
            ChainType.POLYPEPTIDE_L: POLYPEPTIDE_L_CHEM_TYPES
            | {"PEPTIDE LINKING"},  # GLY counts as a peptide linking without L/D
            ChainType.RNA: RNA_LIKE_CHEM_TYPES,
        }
    )
    """Mapping from ChainType enums to valid chemical component types."""

    CHEM_COMP_TYPE_TO_ENUM: Final[MappingProxyType[str, ChainType]] = MappingProxyType(
        {
            chem_comp_type: chain_type
            for chain_type, chem_comp_types in VALID_CHEM_COMP_TYPES.items()
            for chem_comp_type in chem_comp_types
        }
    )
    """Mapping from chemical component types to ChainType enums."""

    ATOMS_AT_POLYMER_BOND: Final[MappingProxyType[ChainType, tuple[str, str]]] = MappingProxyType(
        {
            # peptide bonds
            ChainType.POLYPEPTIDE_D: ("C", "N"),
            ChainType.POLYPEPTIDE_L: ("C", "N"),
            ChainType.CYCLIC_PSEUDO_PEPTIDE: ("C", "N"),
            # phosphodiester bonds
            ChainType.RNA: ("O3'", "P"),
            ChainType.DNA: ("O3'", "P"),
            ChainType.DNA_RNA_HYBRID: ("O3'", "P"),
        }
    )
    """Mapping of chain types to the atoms that they link when part of a polymer."""


class GroundTruthConformerPolicy(IntEnum):
    """Enum for ground truth conformer policy.

    Possible values are:
        - REPLACE: Use the ground-truth coordinates as the reference conformer,
          replacing the coordinates generated by RDKit in-place (and add a flag
          to indicate that the coordinates were replaced)
        - ADD: Return an additional feature (with the same shape as ref_pos)
          containing the ground-truth coordinates
        - FALLBACK: Use the ground-truth coordinates only if our standard
          conformer generation pipeline fails (e.g., we cannot generate a
          conformer with RDKit, and the molecule is either not in the CCD or
          the CCD entry is invalid)
        - IGNORE: Do not use the ground-truth coordinates as the reference
          conformer under any circumstances
    """

    REPLACE = 1
    ADD = 2
    FALLBACK = 3
    IGNORE = 4


class HydrogenPolicy(StrEnum):
    """Enum for hydrogen policy.

    Possible values are:
        - KEEP: Keep the hydrogens as they are
        - REMOVE: Remove the hydrogens
        - INFER: Infer the hydrogens from the atom array
    """

    KEEP = auto()
    REMOVE = auto()
    INFER = auto()
