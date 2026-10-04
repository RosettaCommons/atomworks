"""Constants used in the AtomWorks library."""

import logging
import os
import sys
from types import MappingProxyType
from typing import Final

import numpy as np
from biotite.structure.bonds import BondType
from toolz import keymap

logger = logging.getLogger(__name__)

DEFAULT_MSA_SERVER_URL: Final[str] = "https://api.colabfold.com"
"""Public ColabFold MMseqs2 API endpoint."""


def _load_env_var(var_name: str, default: str | bool | None = None) -> str | bool | None:
    """Load an environment variable.

    Args:
        var_name: The name of the environment variable to load.
        default: The default value if not set. If default is a bool, will parse as bool.

    Returns:
        - bool if the value is a boolean string (1/0, true/false, yes/no, on/off) or default is bool
        - str if the value is a non-boolean string and default is not bool
        - None if not set and no default provided
    """
    value = os.environ.get(var_name)

    # Not set - return default or warn
    if value is None:
        if default is None:
            logger.warning(
                f"Environment variable {var_name} not set. "
                "Will not be able to use function requiring this variable. "
                "To set it you may:\n"
                "  (1) add the line 'export VAR_NAME=path/to/variable' to your .bashrc or .zshrc file\n"
                "  (2) set it in your current shell with 'export VAR_NAME=path/to/variable'\n"
                "  (3) write it to a .env file in the root of the atomworks.io repository"
            )
        return default

    # Auto-detect boolean or parse based on default type
    value_lower = value.lower().strip()
    if isinstance(default, bool) or value_lower in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        if value_lower in ("1", "true", "yes", "on"):
            return True
        elif value_lower in ("0", "false", "no", "off"):
            return False
        else:
            logger.warning(
                f"Invalid boolean value for {var_name}: {value!r}. "
                f"Expected one of: 1, true, yes, on, 0, false, no, off. "
                f"Using default: {default}"
            )
            return default if default is not None else False

    # Return as string
    return value


CCD_MIRROR_PATH: Final[str] = _load_env_var("CCD_MIRROR_PATH")
"""A path to a carbon-copy mirror of the CCD ligands in the RCSB CCD.

Reference:
    `RCSB Chemical Component Dictionary <https://www.rcsb.org/ligand>`_
"""

PDB_MIRROR_PATH: Final[str] = _load_env_var("PDB_MIRROR_PATH")
"""A path to a mirror of the PDB.

Reference:
    `Protein Data Bank <https://www.rcsb.org/>`_
"""

ALLOW_BIOTITE_CCD: Final[bool] = _load_env_var("ALLOW_BIOTITE_CCD", default=True)
"""Whether to allow falling back to Biotite's built-in CCD when no local mirror is available."""

# Defualt / unknown values for atoms

UNKNOWN_ELEMENT: Final[str] = "X"
"""The element name for an unknown element."""

UNKNOWN_ATOMIC_NUMBER: Final[int] = 0
"""The atomic number for an unknown element."""

UNKNOWN_ATOM_NAME: Final[str] = "UNK"
"""The atom name for an unknown atom."""

UNKNOWN_COORD_VALUE: Final[float] = np.nan
"""The coordinate value for an unknown atom."""

DEFAULT_B_FACTOR: Final[float] = np.nan
"""The default B-factor value for an atom."""

DEFAULT_OCCUPANCY: Final[float] = 0.0
"""The default occupancy value for an atom."""

DEFAULT_CHARGE: Final[int] = 0
"""The default charge value for an atom."""

DEFAULT_INS_CODE: Final[str] = ""
"""The default insertion code value for an atom."""

DEFAULT_ALTLOC_ID: Final[str] = "."
"""The default altloc ID value for an atom."""

# fmt: off
ELEMENT_NAME_TO_ATOMIC_NUMBER: Final[MappingProxyType[str, int]] = MappingProxyType(keymap(str.upper, {
    "H": 1,    # Hydrogen (isotopes D, T handled via PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL)
    "He": 2,   "Li": 3,   "Be": 4,   "B": 5,   "C": 6,   "N": 7,    "O": 8,    "F": 9,   "Ne": 10,
    "Na": 11,  "Mg": 12,  "Al": 13,  "Si": 14,  "P": 15,  "S": 16,  "Cl": 17,  "Ar": 18,  "K": 19,  "Ca": 20,
    "Sc": 21,  "Ti": 22,  "V": 23,   "Cr": 24,  "Mn": 25, "Fe": 26, "Co": 27,  "Ni": 28,  "Cu": 29, "Zn": 30,
    "Ga": 31,  "Ge": 32,  "As": 33,  "Se": 34,  "Br": 35, "Kr": 36, "Rb": 37,  "Sr": 38,  "Y": 39,  "Zr": 40,
    "Nb": 41,  "Mo": 42,  "Tc": 43,  "Ru": 44,  "Rh": 45, "Pd": 46, "Ag": 47,  "Cd": 48,  "In": 49, "Sn": 50,
    "Sb": 51,  "Te": 52,  "I": 53,   "Xe": 54,  "Cs": 55, "Ba": 56, "La": 57,  "Ce": 58,  "Pr": 59, "Nd": 60,
    "Pm": 61,  "Sm": 62,  "Eu": 63,  "Gd": 64,  "Tb": 65, "Dy": 66, "Ho": 67,  "Er": 68,  "Tm": 69, "Yb": 70,
    "Lu": 71,  "Hf": 72,  "Ta": 73,  "W": 74,   "Re": 75, "Os": 76, "Ir": 77,  "Pt": 78,  "Au": 79, "Hg": 80,
    "Tl": 81,  "Pb": 82,  "Bi": 83,  "Po": 84,  "At": 85, "Rn": 86, "Fr": 87,  "Ra": 88,  "Ac": 89, "Th": 90,
    "Pa": 91,  "U": 92,   "Np": 93,  "Pu": 94,  "Am": 95, "Cm": 96, "Bk": 97,  "Cf": 98,  "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,"Sg": 106, "Bh": 107,"Hs": 108, "Mt": 109,"Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
    UNKNOWN_ELEMENT: UNKNOWN_ATOMIC_NUMBER
}))
"""Map canonical *UPPERCASE* 2 letter element names to their atomic numbers.

Warning:
    Case-sensitive.

Reference:
    `IUPAC Periodic Table <https://iupac.org/what-we-do/periodic-table-of-elements/>`_
"""

ATOMIC_NUMBER_TO_ELEMENT: Final[MappingProxyType[int | str, str]] = MappingProxyType(
    {v: k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()} |
    {str(v): k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()}
)
"""Map atomic numbers (int/str) to their canonical *UPPERCASE* 2 letter element names.

Warning:
    Case-sensitive.

Reference:
    `IUPAC Periodic Table <https://iupac.org/what-we-do/periodic-table-of-elements/>`_
"""

METAL_ELEMENTS: Final[frozenset[str]] = frozenset(map(str.upper, [
    "Li", "Na", "K", "Rb", "Cs", "Fr",                          # alkali metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",                         # alkaline earth metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", # 3d transition metals
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", # 4d transition metals
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",        # 5d transition metals
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",                   # post-transition metals
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # lanthanides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",   # actinides
]))
"""A set of all metal elements, all *UPPERCASE*.

Warning:
    Case-sensitive.

Reference:
    `IUPAC Periodic Table - Metals <https://iupac.org/what-we-do/periodic-table-of-elements/>`_
"""
# fmt: on

# Common element groups
CHNO_ELEMENTS: Final[frozenset[str]] = frozenset({"C", "H", "N", "O"})
"""Carbon, Hydrogen, Nitrogen, Oxygen - the four core elements of organic chemistry."""

CHNOPS_ELEMENTS: Final[frozenset[str]] = frozenset({"C", "H", "N", "O", "P", "S"})
"""Carbon, Hydrogen, Nitrogen, Oxygen, Phosphorus, Sulfur - the six most abundant elements in living organisms."""

# Bond distance thresholds (in Angstroms)
BOND_DISTANCE_THRESHOLD_CHNO: Final[float] = 1.7
"""Maximum bond distance for bonds involving only C, H, N, O atoms."""

BOND_DISTANCE_THRESHOLD_CHNOPS: Final[float] = 2.4
"""Maximum bond distance for bonds involving only C, H, N, O, P, S atoms."""

BOND_DISTANCE_THRESHOLD_OTHER: Final[float] = 3.6
"""Maximum bond distance for bonds involving metals or other elements."""

# fmt: off
CHEM_COMP_TYPES: Final[tuple[str, ...]] = tuple([
    chemtype.upper() for chemtype in (
        "D-beta-peptide, C-gamma linking", "D-gamma-peptide, C-delta linking", "D-peptide COOH carboxy terminus", "D-peptide NH3 amino terminus", "D-peptide linking",
        "D-saccharide", "D-saccharide, alpha linking", "D-saccharide, beta linking", "DNA OH 3 prime terminus", "DNA OH 5 prime terminus", "DNA linking", "L-DNA linking",
        "L-RNA linking", "L-beta-peptide, C-gamma linking", "L-gamma-peptide, C-delta linking", "L-peptide COOH carboxy terminus", "L-peptide NH3 amino terminus",
        "L-peptide linking", "L-saccharide", "L-saccharide, alpha linking", "L-saccharide, beta linking", "RNA OH 3 prime terminus", "RNA OH 5 prime terminus",
        "RNA linking", "non-polymer", "other", "peptide linking", "peptide-like", "saccharide",
    )
])
# fmt: on
"""Allowed Chemical Component Types for residues in the PDB + `mask`.

All uppercase.

Reference:
    `RCSB mmCIF Dictionary - chem_comp.type <http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html>`_
"""

# fmt: off
AA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in (
        "D-beta-peptide, C-gamma linking", "D-gamma-peptide, C-delta linking", "D-peptide COOH carboxy terminus", "D-peptide NH3 amino terminus", "D-peptide linking",
        "L-beta-peptide, C-gamma linking", "L-gamma-peptide, C-delta linking", "L-peptide COOH carboxy terminus", "L-peptide NH3 amino terminus", "L-peptide linking",
        "peptide linking", "peptide-like",
    )
])
# fmt: on
"""Set of amino acid-like chemical component types. All uppercase."""

# fmt: off
POLYPEPTIDE_L_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in (
        "L-beta-peptide, C-gamma linking", "L-gamma-peptide, C-delta linking", "L-peptide COOH carboxy terminus", "L-peptide NH3 amino terminus", "L-peptide linking",
    )
])
# fmt: on
"""Set of polypeptide-L (left-handed amino acids) chemical component types. All uppercase."""

# fmt: off
POLYPEPTIDE_D_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in (
        "D-beta-peptide, C-gamma linking", "D-gamma-peptide, C-delta linking", "D-peptide COOH carboxy terminus", "D-peptide NH3 amino terminus", "D-peptide linking",
    )
])
# fmt: on
"""Set of polypeptide-D (right-handed amino acids) chemical component types. All uppercase."""

# fmt: off
RNA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in ("L-RNA linking", "RNA OH 3 prime terminus", "RNA OH 5 prime terminus", "RNA linking")
])
# fmt: on
"""Set of RNA-like chemical component types. All uppercase."""

# fmt: off
DNA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in ("DNA OH 3 prime terminus", "DNA OH 5 prime terminus", "DNA linking", "L-DNA linking")
])
# fmt: on
"""Set of DNA-like chemical component types. All uppercase."""

NA_LIKE_CHEM_TYPES: Final[frozenset[str]] = RNA_LIKE_CHEM_TYPES | DNA_LIKE_CHEM_TYPES
"""DNA or RNA-like chemical component types."""

# fmt: off
CARBOHYDRATE_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in (
        "D-saccharide", "D-saccharide, alpha linking", "D-saccharide, beta linking", "L-saccharide", "L-saccharide, alpha linking", "L-saccharide, beta linking", "saccharide",
    )
])
# fmt: on
"""Set of carbohydrate-like chemical component types. All uppercase."""

# fmt: off
CARBOHYDRATE_L_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in ("L-saccharide", "L-saccharide, alpha linking", "L-saccharide, beta linking")
])
# fmt: on
"""Set of carbohydrate-L (left-handed saccharides) chemical component types. All uppercase."""

# fmt: off
CARBOHYDRATE_D_CHEM_TYPES: Final[frozenset[str]] = frozenset([
    chemtype.upper() for chemtype in ("D-saccharide", "D-saccharide, alpha linking", "D-saccharide, beta linking")
])
# fmt: on
"""Set of carbohydrate-D (right-handed saccharides) chemical component types. All uppercase."""

LIGAND_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([chemtype.upper() for chemtype in ("non-polymer", "other")])
"""Set of ligand-like chemical component types. All uppercase."""

MASK_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([chemtype.upper() for chemtype in ("mask",)])
"""Set of mask-like chemical component types. All uppercase."""

AA_OR_NA_CHEM_COMP_TYPES: Final[frozenset[str]] = AA_LIKE_CHEM_TYPES | NA_LIKE_CHEM_TYPES
"""Amino acid or DNA/RNA-like chemical component types."""

CHEM_TYPE_POLYMERIZATION_ATOMS: Final[MappingProxyType[str, tuple[str, str]]] = MappingProxyType(
    keymap(
        str.upper,
        {
            # peptide bonds
            "peptide linking": ("C", "N"),
            "L-peptide linking": ("C", "N"),
            "D-peptide linking": ("C", "N"),
            "L-beta-peptide, C-gamma linking": ("CG", "N"),
            "D-beta-peptide, C-gamma linking": ("CG", "N"),
            "L-gamma-peptide, C-delta linking": ("CD", "N"),
            "D-gamma-peptide, C-delta linking": ("CD", "N"),
            # phosphodiester bonds
            "DNA linking": ("O3'", "P"),
            "L-DNA linking": ("O3'", "P"),
            "RNA linking": ("O3'", "P"),
            "L-RNA linking": ("O3'", "P"),
        },
    )
)
"""A mapping of chemical component types to the atoms that they link when part of a polymer."""

CCD_POLYMERIZATION_ATOM_OVERRIDES: Final[MappingProxyType[str, tuple[str | None, str | None]]] = MappingProxyType(
    {
        # Nucleotide analogues where an element other than oxygen sits at the 3' position, so the
        # linkage cannot be recovered from the chemical component type's canonical ``O3'``.
        "TSP": ("S3'", "P"),  # 3'-thio-thymidine
        "US1": ("S", "P"),  # 2'-deoxy-3'-thiouridine
        "C42": ("N", "P"),  # 3'-amino-2'-deoxy-cytidine (phosphoramidate)
        "TCP": ("O3'", "CP"),  # 5'-methylthymidine; a methylene replaces the 5' phosphorus
        "TFT": ("O2T", "P"),  # threofuranosyl (TNA); the tetrose sugar's 3'-equivalent is O2T
        "3DA": ("O2'", "P"),  # 3'-deoxyadenosine (cordycepin); no O3', so it links 2'->5'
        "8RJ": ("O2G", "P"),  # glycol nucleic acid; the free hydroxyl is O2G (5V1K)
        "4DG": (None, "P"),  # acyclic guanine phosphonate; a 3' terminus, so nothing exits (3KD1/3KD5)
        # Backbones that simply use non-canonical atom names, where the CCD entry flags a leaving
        # group for only one side and the other side therefore cannot be derived.
        "AEA": ("C5", "N1"),
        "WCM": ("C14", "N5"),
        # 2'-amino-2'-deoxyadenosine. Its sole leaving-atom flag is ``H2``
        # on the purine ``C2``, which would otherwise be derived as a spurious exiting carbon.
        "2AD": ("O3'", "N"),
    }
)
"""Polymerization atoms for CCD entries whose own annotations cannot yield the right answer.

Each entry is ``(exiting_atom, entering_atom)``, matching :py:func:`get_polymerization_atoms`, and
every one is confirmed against ``struct_conn`` records in the PDB. Consulted only once a component
is known not to carry the canonical atom names for its chemical component type, so a custom
component registered under one of these codes is unaffected.
"""

MAX_CHEM_TYPE_LENGTH: Final[int] = max(len(chem_type) for chem_type in CHEM_COMP_TYPES)
"""Maximum length of a valid chem_comp_type identifier. Determines the dtype of the `chem_comp_type` annotation."""

STRUCT_CONN_BOND_TYPES: Final[frozenset[str]] = frozenset({"covale", "disulf", "metalc"})
"""A set of bond types that are considered when adding bonds to the atom array.

Reference:
    `struct_conn.conn_type_id <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.conn_type_id.html>`_
"""

STRUCT_CONN_BOND_ORDER_TO_INT: Final[MappingProxyType[str, int]] = MappingProxyType(
    {
        "sing": 1,
        "doub": 2,
        "trip": 3,
        "quad": 4,
    }
)
"""
Mapping from `struct_conn.pdbx_value_order` to integer bond orders.

Reference:
    `struct_conn.pdbx_value_order <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.pdbx_value_order.html>`_
"""

BIOTITE_BOND_TYPE_TO_BOND_ORDER: Final[MappingProxyType[BondType, int]] = MappingProxyType(
    {
        # biotite bond type -> bond order
        BondType.ANY: 1,  # 0
        BondType.SINGLE: 1,  # 1
        BondType.DOUBLE: 2,  # 2
        BondType.TRIPLE: 3,  # 3
        BondType.QUADRUPLE: 4,  # 4
        BondType.AROMATIC_SINGLE: 1,  # 5
        BondType.AROMATIC_DOUBLE: 2,  # 6
        BondType.AROMATIC_TRIPLE: 3,  # 7
        BondType.COORDINATION: 0,  # 8, excluded from covalent valence bookkeeping
    }
)
"""Mapping from Biotite bond types to bond orders.

NOTE: BondType.AROMATIC (9) has no well-defined bond order and must be handled separately.

Reference:
    `biotite.structure.BondType <https://www.biotite-python.org/latest/apidoc/biotite.structure.BondType.html>`_
"""

DEFAULT_VALENCE = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "B": 3,
}
"""Default valences of common elements in organic compounds.
Only elements that have unambiguous valences are included.

Reference:
    `RDKit Book - Valence Calculation <https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences>`_
"""

# fmt: off
CRYSTALLIZATION_AIDS: Final[list[str]] = ["SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS", "PGE", "PG4", "FMT", "EPE", "MPD", "MES", "CD", "IOD"]
# fmt: on
"""A list of CCD codes of common crystallization aids used in the crystallization of proteins.

Reference:
    `AF3 (Supp. Table 9) <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
"""

# fmt: off
AF3_EXCLUDED_LIGANDS: Final[list[str]] = [
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE", "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT",
    "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ", "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
    "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL", "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH", "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE",
    "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA", "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP",
    "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP", "PVO", "SAR", "SCN", "SEO",
    # "SEP",  # Phosphoserine; a commonly occuring PTM in proteins, useful in cellular signaling pathways
    "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU", "TAR", "TBU", "TME",
    # "TPO",  # Phosphothreonine; a commonly occuring PTM in proteins, useful in cellular signaling pathways
    "TRS", "UNK", "UNL", "UNX", "UPL", "URE",
]
# fmt: on
"""A list of CCD codes of ligands that were excluded in AF3.

Reference:
    `AF3 (Supp. Table 10) <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
"""

AF3_EXCLUDED_LIGANDS_REGEX: Final[str] = r"(?:^|,)\s*(?:" + "|".join(AF3_EXCLUDED_LIGANDS) + r")\s*(?:,|$)"
"""A regex pattern that matches any of the ligands in `AF3_EXCLUDED_LIGANDS`. Used for filtering out ligands from the assembled dataframes."""

# fmt: off
# TODO: Replace this by general mapping of CCD codes to one-letter codes.
DICT_THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "ASX": "B", "GLX": "Z", "UNK": "X", " * ": "*",
}
# fmt: on
"""A dictionary that maps three-letter amino acid codes to one-letter codes.

Reference:
    `Biotite seqtypes.py <https://github.com/biotite-dev/biotite/blob/v0.41.0/src/biotite/sequence/seqtypes.py#L348-L556>`_
"""

UNKNOWN_LIGAND: Final[str] = sys.intern("UNL")
"""The CCD code for unknown ligands (`UNL`).

Reference:
    `wwPDB Documentation <https://www.wwpdb.org/documentation/procedure>`_
"""

UNKNOWN_AA: Final[str] = sys.intern("UNK")
"""The CCD code for unknown amino acids (`UNK`).

Reference:
    `wwPDB Documentation <https://www.wwpdb.org/documentation/procedure>`_
"""

# TODO: Change these to something unique.
UNKNOWN_RNA: Final[str] = sys.intern("N")
"""The CCD code for unknown RNA nucleotides (`N`).

Reference:
    `wwPDB Documentation <https://www.wwpdb.org/documentation/procedure>`_
"""

UNKNOWN_DNA: Final[str] = sys.intern("DN")
"""The CCD code for unknown DNA nucleotides (`DN`).

Reference:
    `wwPDB Documentation <https://www.wwpdb.org/documentation/procedure>`_
"""

UNKNOWN_ATOM: Final[str] = sys.intern("UNX")
"""The CCD code for unknown atoms (`UNX`).

Reference:
    `wwPDB Documentation <https://www.wwpdb.org/documentation/procedure>`_
"""

GAP: Final[str] = sys.intern("<G>")
"""The (non-standard) code for a gap token."""

GAP_ONE_LETTER: Final[str] = sys.intern("-")
"""The one-letter code for a gap token."""

MASKED: Final[str] = sys.intern("<M>")
"""The (non-standard) code for a masked token."""

# fmt: off
STANDARD_AA: Final[tuple[str, ...]] = tuple(sorted([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]))
# fmt: on
"""Tuple of the CCD codes for the standard 20 amino acids, alphabetically sorted by their three-letter CCD codes."""

STANDARD_AA_ONE_LETTER: Final[tuple[str, ...]] = tuple(map(DICT_THREE_TO_ONE.get, STANDARD_AA))
"""Tuple of the one-letter symbols for the standard 20 amino acids, alphabetically sorted by their three-letter CCD codes."""

STANDARD_RNA: Final[tuple[str, ...]] = tuple(sorted(["A", "C", "G", "U"]))
"""Tuple of the CCD codes for the standard 4 RNA nucleotides. These happen to be the same as the one-letter symbols."""

STANDARD_DNA: Final[tuple[str, ...]] = tuple(sorted(["DA", "DC", "DG", "DT"]))
"""Tuple of the CCD codes for the standard 4 DNA nucleotides."""

STANDARD_NA: Final[tuple[str, ...]] = STANDARD_RNA + STANDARD_DNA
"""Tuple of the CCD codes for the standard 8 nucleotides (4 RNA + 4 DNA)."""

STANDARD_DNA_ONE_LETTER: Final[tuple[str, ...]] = tuple(sorted(["A", "C", "G", "T"]))
"""Tuple of the one-letter symbols for the standard 4 DNA nucleotides."""

STANDARD_POLYMER_RESIDUES: Final[tuple[str, ...]] = (*STANDARD_AA, *STANDARD_RNA, *STANDARD_DNA)
"""Tuple of all standard polymer residues (20 AA, 4 RNA, 4 DNA)."""

UNKNOWN_POLYMER_RESIDUES: Final[tuple[str, ...]] = (UNKNOWN_AA, UNKNOWN_RNA, UNKNOWN_DNA)
"""Tuple of unknown polymer residues (UNK, N, DN)."""


STANDARD_AND_UNKNOWN_AA: Final[tuple[str, ...]] = (*STANDARD_AA, UNKNOWN_AA)
"""Tuple of the CCD codes for the standard 20 amino acids + unknown (UNK)."""

STANDARD_AND_UNKNOWN_RNA: Final[tuple[str, ...]] = (*STANDARD_RNA, UNKNOWN_RNA)
"""Tuple of the CCD codes for the standard 4 RNA nucleotides + unknown (N)."""

STANDARD_AND_UNKNOWN_DNA: Final[tuple[str, ...]] = (*STANDARD_DNA, UNKNOWN_DNA)
"""Tuple of the CCD codes for the standard 4 DNA nucleotides + unknown (DN)."""

STANDARD_AND_UNKNOWN_POLYMER_RESIDUES: Final[tuple[str, ...]] = (
    *STANDARD_POLYMER_RESIDUES,
    *UNKNOWN_POLYMER_RESIDUES,
)
"""Tuple of all standard and unknown polymer residues (AA, RNA, DNA)."""

BIOTITE_DEFAULT_ANNOTATIONS: Final[tuple[str, ...]] = (
    "chain_id",
    "res_id",
    "res_name",
    "atom_name",
    "hetero",
    "element",
)
"""The default mandatory annotations for Biotite AtomArrays."""

BIOTITE_COMMON_OPTIONAL_ANNOTATIONS: Final[tuple[str, ...]] = (
    "atom_id",
    "charge",
    "occupancy",
    "b_factor",
)
"""Optional annotations commonly used in Biotite.

They are saved by biotite `set_structure` by default, and hence by atomworks `to_cif_file`.
They are not loaded by biotite `get_structure` by default, but they ARE loaded by default in atomworks `load_any`
"""

ALTLOC_DEFAULT_IDS: Final[tuple[str, ...]] = (".", "?", " ", "")
"""Altloc ID values that indicate a default (non-alternate) atom position."""

STANDARD_CIF_ANNOTATIONS: Final[frozenset[str]] = frozenset(
    {
        "chain_id",
        "res_id",
        "res_name",
        "atom_name",
        "atom_id",
        "element",
        "ins_code",
        "hetero",
        "altloc_id",
        "charge",
        "occupancy",
        "b_factor",
    }
)
"""All standard CIF annotations handled by Biotite, mandatory or optional.

These are reserved names that are excluded from `extra_annotations`.
"""

ATOM_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    # +---- Standard mmCIF/Biotite atom-level annotations ----+
    "atom_name",  # CCD atom name (e.g., "CA", "N", "O")
    "element",  # Element symbol (e.g., "C", "N", "O")
    "atom_id",  # Unique atom identifier
    "altloc_id",  # Alternate location identifier (mutliple-occupancy models)
    "charge",  # Atomic charge
    "occupancy",  # Occupancy value
    "b_factor",  # B-factor (temperature factor)
    # +---- AtomWorks extensions ----+
    "atomic_number",  # Atomic number (derived from element)
)
"""Standard atom-level annotations"""

RESIDUE_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    # +---- Standard mmCIF/Biotite residue-level annotations ----+
    "res_id",  # Residue ID (sequence number)
    "res_name",  # Residue name (CCD 3-letter code)
    "hetero",  # Hetero flag (non-standard residue)
    "ins_code",  # Insertion code (legacy from PDB format, rarely used)
    "label_alt_id",  # Alternate conformer ID — per-atom in mmCIF, but residue-level after altloc selection
)
"""Residue-level annotations - shared by all atoms in a residue."""

CHAIN_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    # +---- Standard mmCIF/Biotite chain-level annotations ----+
    "chain_id",  # Chain identifier (author chain ID)
    # +---- AtomWorks extensions ----+
    "chain_iid",  # Chain instance identifier (disambiguates across transforms)
    "chain_entity",  # Chain entity identifier
    "chain_type",  # Chain type as int enum (polypeptide, DNA, RNA, etc.)
    "is_polymer",  # Convenience chain-level polymer flag (derived from chain_type)
)
"""Chain-level annotations."""

PN_UNIT_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    "pn_unit_id",  # PN Unit identifier (Polymer XOR Non-polymer unit)
    "pn_unit_iid",  # PN Unit instance identifier (disambiguates across transforms)
    "pn_unit_entity",  # PN Unit entity identifier
)
"""PN Unit-level annotations - Polymer XOR Non-polymer unit annotations."""

MOLECULE_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    "molecule_id",  # Molecule identifier
    "molecule_iid",  # Molecule instance identifier (disambiguates across transforms)
    "molecule_entity",  # Molecule entity identifier
)
"""Molecule-level annotations."""

TRANSFORMATION_LEVEL_ANNOTATIONS: Final[tuple[str, ...]] = (
    "transformation_id",  # Transformation identity in multi-transformation assemblies
)
"""Transformation-level annotations - for bio-assemblies."""

ATOMWORKS_COMMON_ANNOTATIONS: Final[tuple[str, ...]] = (
    ATOM_LEVEL_ANNOTATIONS
    + RESIDUE_LEVEL_ANNOTATIONS
    + CHAIN_LEVEL_ANNOTATIONS
    + PN_UNIT_LEVEL_ANNOTATIONS
    + MOLECULE_LEVEL_ANNOTATIONS
    + TRANSFORMATION_LEVEL_ANNOTATIONS
)
"""All standard annotations used in AtomWorks.

Includes both standard mmCIF/Biotite annotations and AtomWorks extensions.
"""

STANDARD_PYRIMIDINE_RESIDUES: Final[tuple[str, ...]] = ("C", "U", "DC", "DT")
"""Tuple of the CCD codes for the 4 standard pyrimidine nucleotides."""

STANDARD_PURINE_RESIDUES: Final[tuple[str, ...]] = ("A", "G", "DA", "DG")
"""Tuple of the CCD codes for the 4 standard purine nucleotides."""

HYDROGEN_LIKE_SYMBOLS: Final[tuple[str, ...]] = ("H", "H2", "D", "T")
"""
A tuple of symbols for (isotopes of) hydrogen.

WARNING: It is important that this remains a tuple, as it is used by `np.isin`
 downstream, which does not play well with sets.
"""

WATER_LIKE_CCDS: Final[tuple[str, ...]] = ("HOH", "DOD")
"""A tuple of CCD codes for water-like molecules.

WARNING: It is important that this remains a tuple, as it is used by `np.isin`
 downstream, which does not play well with sets.
"""

DO_NOT_MATCH_CCD: Final[frozenset[str]] = frozenset((*WATER_LIKE_CCDS, UNKNOWN_LIGAND))
"""CCDs that should not be matched to a template for the purpose of adding missing atoms."""

DEFAULT_CCD_ANNOTATIONS: Final[tuple[str, ...]] = (
    "charge",
    "stereo",
    "is_aromatic",
    "is_leaving_atom",
    "is_backbone_atom",
    "is_n_terminal_atom",
    "is_c_terminal_atom",
    "nhyd",
)
"""Common annotations derived from Chemical Component Dictionary (CCD) entries."""

PEPTIDE_MAX_RESIDUES: Final[int] = 16
"""The maximum number of residues until which we consider a protein-like sequence to be a peptide.

NOTE: AF3 uses either 10 or 16 for this number in different places.
"""

PDB_ISOTOPE_SYMBOL_TO_ELEMENT_SYMBOL: Final[dict[str, str]] = {
    "D": "H",
    "T": "H",
}
"""Map isotopes symbols used in the PDB to the element symbols.

NOTE: Other isotopes like 14C do not have a special symbol in the PDB.
"""

# fmt: off
STANDARD_AA_TIP_ATOM_NAMES: Final[dict[str, list[str]]] = {
    "ALA": ["CB"], "ARG": ["NH1", "NH2"], "ASN": ["OD1", "ND2"], "ASP": ["OD1", "OD2"], "CYS": ["SG"], "GLN": ["OE1", "NE2"], "GLU": ["OE1", "OE2"], "GLY": ["CA"],
    "HIS": ["CE1", "NE2"], "ILE": ["CD1"], "LEU": ["CD1", "CD2"], "LYS": ["NZ"], "MET": ["CE"], "PHE": ["CZ"], "PRO": ["CD", "CG"], "SER": ["OG"],
    "THR": ["OG1", "CG2"], "TRP": ["CH2"], "TYR": ["OH"], "VAL": ["CG1", "CG2"],
}
# fmt: on
"""A dictionary that maps the standard 20 amino acids to their tip atoms.

Tip atoms are defined as the side-chain heavy atoms that are furthest away
from the backbone oxygen atom in the residue's bond graph. With the exception of GLY,
which has no backbone oxygen atom and we therefore use the CA atom as the tip atom.
"""

PROTEIN_FRAME_ATOM_NAMES: Final[tuple[str, ...]] = ("N", "CA", "C")
"""A tuple of the names of the frame atoms (backbone) proteins."""

NUCLEIC_ACID_FRAME_ATOM_NAMES: Final[tuple[str, ...]] = ("C1'", "C3'", "C4'")
"""A tuple of the names of the frame atoms (backbone) for nucleic acids."""

PROTEIN_BACKBONE_ATOM_NAMES: Final[tuple[str, ...]] = ("N", "CA", "C", "O", "OXT")
"""A tuple of the names of all protein backbone atoms (N-CA-C backbone + carbonyl oxygen + terminal OXT)."""

# fmt: off
RNA_BACKBONE_ATOM_NAMES: Final[tuple[str, ...]] = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "C2'", "C1'", "O4'", "O3'", "O2'")
# fmt: on
"""A tuple of the names of RNA backbone atoms (sugar-phosphate backbone including 2' hydroxyl)."""

# fmt: off
DNA_BACKBONE_ATOM_NAMES: Final[tuple[str, ...]] = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "C2'", "C1'", "O4'", "O3'")
# fmt: on
"""A tuple of the names of DNA backbone atoms (sugar-phosphate backbone, no 2' hydroxyl)."""

NUCLEIC_ACID_BACKBONE_ATOM_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(set(RNA_BACKBONE_ATOM_NAMES) | set(DNA_BACKBONE_ATOM_NAMES))
)
"""A tuple of the names of all nucleic acid backbone atoms (union of RNA and DNA backbones)."""

# fmt: off
NA_VALUES = [
    "", " ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None", "n/a", "nan", "null",
]
# fmt: on
"""A list of strings that are considered as NA/NaN ("missing" values) values in dataframes.

NOTE: By default, "NA" is considered as a missing value by Pandas, which can lead to subtle bugs.
"""
