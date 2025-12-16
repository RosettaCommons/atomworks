"""Definitions of the various standard encodings."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import cycle
from logging import getLogger

import biotite.structure as struc
import numpy as np

from atomworks.common import exists
from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    CHEM_COMP_TYPES,
    DNA_LIKE_CHEM_TYPES,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    GAP,
    RNA_LIKE_CHEM_TYPES,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.io.utils.ccd import get_chem_comp_type

logger = getLogger(__name__)

UNKNOWN_ELEMENT_TOKEN = 0
"""The token to use for an unknown element."""


@dataclass
class TokenEncoding:
    """A class to represent a fixed length token encoding.

    Args:
        token_atoms: A dictionary mapping token names to atom names.
            The order of the tokens in the sequence determines the integer encoding of the token.
            The order of the atom names in the tuple determines the integer encoding of the atom name
            within the token.
        chemcomp_type_to_unknown: A dictionary mapping chemical component types
            to unknown token names. This is used to map unknown residues to the respective unknown
            token. Different chemical component types may map to different unknown token names.
            Defaults to ``{}``, meaning that no unknown tokens are defined, leading to a ``KeyError``
            if an unknown residue is encountered.

    Note:
        We follow these conventions for tokens to make them compatible with the CCD for
        robust and easy tokenization. If you want to use the Transforms written for automatically
        tokenizing and encoding, you need to follow these conventions:

        - When encoding a residue, we use the standardized (up to) 3-letter residue name from the CCD,
            e.g. ``'ALA'`` for Alanine, or ``'DA'`` for Deoxyadenosine, or ``'U'`` for Uracil.
        - When encoding unknown tokens, we may define different unknown tokens for different
            chemical components (e.g. a different unknown for proteins, vs. dna, ...). The
            unknown tokens can take on any arbitrary 3-letter code that we want to map to, but
            they should not clash with existing residue names in the CCD.
        - When encoding an atom, we use the atomic number of the element as a string as the
            token name. E.g. ``'1'`` for Hydrogen, ``'6'`` for Carbon, ``'9'`` for Fluorine, ...
            For unknown atoms, we use ``'0'`` as the token name.
            # TODO: Deal with ligand names such as ``'100'`` which is also an atomic number
        - To denote masked tokens, we use a ``'<...>'`` syntax. E.g. ``'<M>'`` for a generic mask token,
            or ``'<MP>'`` for a mask token for proteins. The ... can be any arbitrary string. We
            use the angle brackets to avoid clashes with existing residue names in the CCD.
    """

    token_atoms: dict[str | int, np.ndarray]
    chemcomp_type_to_unknown: dict[str, str] = None

    def __post_init__(self):
        _none_to_empty_str = np.vectorize(lambda x: x if x is not None else "")
        _strip_str = np.vectorize(lambda x: x.strip())
        _process = lambda x: _strip_str(_none_to_empty_str(x))  # noqa
        self.token_atoms = {
            token.strip() if isinstance(token, str) else token: _process(np.asarray(atoms))
            for token, atoms in self.token_atoms.items()
        }

        # Ensure all values are of type `np.ndarray` and have the same 1-dimensional shape
        _target_len = len(next(iter(self.token_atoms.values())))
        for token, atoms in self.token_atoms.items():
            assert isinstance(
                atoms, np.ndarray
            ), f"Expected `atoms` to be a `np.ndarray`, but got {type(atoms)} for token {token}."
            assert (
                atoms.ndim == 1
            ), f"Expected `atoms` to be a 1-dimensional array, but got {atoms.ndim} dimensions for token {token}."
            assert (
                len(atoms) == _target_len
            ), f"Expected all atoms to have length {_target_len}, but got {len(atoms)} for token {token}."

        # Define mapping of unknown `chemcomp_type` to unknown token names
        if not exists(self.chemcomp_type_to_unknown):
            self.chemcomp_type_to_unknown = {}
        else:
            # ... ensure chemcomp_types are uppercase
            self.chemcomp_type_to_unknown = {
                chemcomp_type.upper(): unknown_token
                for chemcomp_type, unknown_token in self.chemcomp_type_to_unknown.items()
            }

        # Validate unknown tokens
        for chemcomp_type, unknown_token in self.chemcomp_type_to_unknown.items():
            assert unknown_token in self.token_atoms, f"Unknown token {unknown_token} not defined in `token_atoms`."
            assert chemcomp_type in CHEM_COMP_TYPES, f"Unknown chemcomp type {chemcomp_type}."

        # Set function to resolve unknown tokens.
        # NOTE: This is set here to use caching.
        @lru_cache(maxsize=10000)
        def resolve_unknown_token_name(token_name: str | int, token_is_atom: bool) -> str:
            assert isinstance(
                token_name, str | int | np.integer
            ), f"Expected `token_name` to be a string or int, but got {type(token_name)}: token_name={token_name}, token_is_atom={token_is_atom}."

            # Case 1: Token is known & valid
            if token_name in self.token_atoms:
                # ... escape
                return token_name

            # Case 2: Token is unknown atom
            if token_is_atom:
                # ... for unknown atoms
                if UNKNOWN_ELEMENT_TOKEN not in self.token_atoms:
                    # ... ensure that the `UNKNOWN_ELEMENT_TOKEN` is in the encoding
                    raise KeyError(
                        f"Encountered unknown atom token `{token_name}` which is not in the encoding, "
                        f"but the `UNKNOWN_ELEMENT_TOKEN` (`{UNKNOWN_ELEMENT_TOKEN}`) is also not in the encoding."
                    )
                return UNKNOWN_ELEMENT_TOKEN

            # Case 3: Token is unknown residue
            if exists(self.chemcomp_type_to_unknown):
                # ... try to resolve which unknown residue token to use based on the chemical component type
                chem_type = get_chem_comp_type(token_name)
                if chem_type not in self.chemcomp_type_to_unknown:
                    raise KeyError(
                        f"Could not resolve unknown residue token name: `{token_name}`, "
                        f"chemcomp_type: `{chem_type}` not in `encoding.chemcomp_type_to_unknown`."
                        "You will either have to:\n"
                        "(1) filter out this token before encoding,\n"
                        "(2) use an encoding that contains a `chemcomp_type_to_unknown` mapping "
                        "for this chemcomp type,\n"
                        "(3) use an encoding that contains this token, or\n"
                        "(4) atomize this token (provided your specified encoding contains atom-level "
                        "tokens)."
                    )
                return self.chemcomp_type_to_unknown[chem_type]
            else:
                raise KeyError(
                    f"Encountered unknown residue token name: `{token_name}` which is not in the encoding, "
                    f"and no `chemcomp_type_to_unknown` mapping is defined."
                )

        self._resolve_unknown_token_name = resolve_unknown_token_name

    @cached_property
    def tokens(self) -> np.ndarray:
        dtypes = {type(token) for token in self.token_atoms}
        if all(issubclass(dtype, str) for dtype in dtypes):
            return np.array(list(self.token_atoms.keys()), dtype=str)
        elif all(issubclass(dtype, int | np.integer) for dtype in dtypes):
            return np.array(list(self.token_atoms.keys()), dtype=int)
        else:
            return np.array(list(self.token_atoms.keys()), dtype=object)

    @cached_property
    def unknown_tokens(self) -> np.ndarray:
        dtypes = {type(token) for token in self.chemcomp_type_to_unknown.values()}
        if all(issubclass(dtype, str) for dtype in dtypes):
            return np.array(list(self.chemcomp_type_to_unknown.values()), dtype=str)
        elif all(issubclass(dtype, int | np.integer) for dtype in dtypes):
            return np.array(list(self.chemcomp_type_to_unknown.values()), dtype=int)
        else:
            return np.array(list(self.chemcomp_type_to_unknown.values()), dtype=object)

    @cached_property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @cached_property
    def n_atoms_per_token(self) -> int:
        return len(self.token_atoms[self.tokens[0]])

    @cached_property
    def idx_to_token(self) -> np.ndarray:
        """For rapid decoding of token indices to token names via numpy indexing."""
        return self.tokens  # [n_tokens] (str)

    @cached_property
    def idx_to_atom(self) -> np.ndarray:
        """For rapid decoding of token & atom indices to atom names via numpy indexing."""
        return np.vstack(list(self.token_atoms.values()))  # [n_res, n_atoms_per_token] (str)

    @cached_property
    def idx_to_element(self) -> np.ndarray:
        """For rapid decoding of token & atom indices to atom names via numpy indexing."""
        atomic_number_to_pdb_element_name = {}
        for elt, atomic_number in ELEMENT_NAME_TO_ATOMIC_NUMBER.items():
            atomic_number_to_pdb_element_name[str(atomic_number)] = elt.upper()
            atomic_number_to_pdb_element_name[atomic_number] = elt.upper()

        elements = np.full((self.n_tokens, self.n_atoms_per_token), "", dtype="<U3")
        for idx, (_token, atom_names) in enumerate(self.token_atoms.items()):
            # ... case 1: atom names - try to infer elements from atom names
            inferred_elements = struc.infer_elements(atom_names)
            if np.all(inferred_elements == ""):
                # ... case 2: atomic numbers - try to infer elements from atomic numbers
                inferred_elements = np.array(
                    [atomic_number_to_pdb_element_name.get(elt, elt) for elt in atom_names], dtype="<U3"
                )
            # set elements
            elements[idx] = inferred_elements

        return elements  # [n_res, n_atoms_per_token] (str)

    @cached_property
    def token_to_idx(self) -> dict[str | str, int]:
        """For encoding token names to token indices. (token) -> token_idx"""
        return {token: i for i, token in enumerate(self.tokens)}

    @cached_property
    def atom_to_idx(self) -> dict[tuple[str | int, str], int]:
        """For encoding atoms (token, atom) to atom indices. (token, atom) -> atom_idx"""
        token_and_atom_to_idx = {}
        for token in self.tokens:
            for atom_idx, atom_name in enumerate(self.token_atoms[token]):
                if atom_name != "":
                    # Atom name exists in this token (otherwise it will be `''`)
                    token_and_atom_to_idx[token, atom_name] = atom_idx
        return token_and_atom_to_idx

    def resolve_unknown_token_name(self, token_name: str, token_is_atom: bool) -> str:
        return self._resolve_unknown_token_name(token_name, token_is_atom)

    def to_str(self) -> str:
        """Convenience function for printing the encoding."""
        max_token_length = max(len(str(token)) for token in self.tokens)
        max_atom_length = max(len(atom) for atoms in self.token_atoms.values() for atom in atoms)
        max_atoms_per_token = max(len(atoms) for atoms in self.token_atoms.values())

        # Create header
        header = f" Token{'':<{max_token_length}} | " + " | ".join(
            f"{i:<{max_atom_length}}" for i in range(max_atoms_per_token)
        )
        result = [header, "-" * len(header)]

        # Create rows
        for idx, token in enumerate(self.tokens):
            atoms = self.token_atoms[token]
            atom_str = " | ".join(f"{atom:<{max_atom_length}}" for atom in atoms)
            # Fill the remaining columns with spaces if the number of atoms is less than max_atoms_per_token
            atom_str += " | " * (max_atoms_per_token - len(atoms))
            result.append(f"{idx:>3} : {token:<{max_token_length}} | {atom_str}")

        return "\n".join(result)

    def __repr__(self):
        _str = f"Encoding(n_tokens={self.n_tokens}, n_atoms_per_token={self.n_atoms_per_token})" + "\n"
        _str += f"{self.to_str()}"
        return _str


# fmt: off
AF2_ATOM14_ENCODING = TokenEncoding(
    #            0    1     2    3    4     5     6      7      8      9      10     11     12     13
    token_atoms= {
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],    # 0
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],    # 1
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],    # 2
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],    # 3
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],    # 4
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],    # 5
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],    # 6
        'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],    # 7
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],    # 8
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],    # 9
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],    # 10
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],    # 11
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],    # 12
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],    # 13
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],    # 14
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],    # 15
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],    # 16
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], # 17
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],    # 18
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],    # 19
        'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],    # 20
    },
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

"""AF2's atom14 encoding.

Reference:
    `AlphaFold residue_constants.py <https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/common/residue_constants.py#L505>`_
"""

AF2_ATOM37_ENCODING = TokenEncoding(
    token_atoms= {
        #        0       1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31      32      33      34      35      36
        'ALA': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 0
        'ARG': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'NE ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'NH1' , 'NH2' , '   ' , 'CZ ' , '   ' , '   ' , '   ' , 'OXT'],  # 1
        'ASN': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'ND2' , 'OD1' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 2
        'ASP': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OD1' , 'OD2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 3
        'CYS': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'SG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 4
        'GLN': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'NE2' , 'OE1' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 5
        'GLU': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OE1' , 'OE2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 6
        'GLY': ['N  ' , 'CA ' , 'C  ' , '   ' , 'O  ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 7
        'HIS': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD2' , 'ND1' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CE1' , '   ' , '   ' , '   ' , '   ' , 'NE2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 8
        'ILE': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , 'CG1' , 'CG2' , '   ' , '   ' , '   ' , '   ' , 'CD1' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 9
        'LEU': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD1' , 'CD2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 10
        'LYS': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CE ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'NZ ' , 'OXT'],  # 11
        'MET': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'SD ' , 'CE ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 12
        'PHE': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD1' , 'CD2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CE1' , 'CE2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CZ ' , '   ' , '   ' , '   ' , 'OXT'],  # 13
        'PRO': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 14
        'SER': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , '   ' , '   ' , 'OG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 15
        'THR': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , '   ' , 'CG2' , '   ' , 'OG1' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 16
        'TRP': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD1' , 'CD2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CE2' , 'CE3' , '   ' , 'NE1' , '   ' , '   ' , '   ' , 'CH2' , '   ' , '   ' , '   ' , '   ' , 'CZ2' , 'CZ3' , '   ' , 'OXT'],  # 17
        'TYR': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , 'CG ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CD1' , 'CD2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'CE1' , 'CE2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OH ' , 'CZ ' , '   ' , '   ' , '   ' , 'OXT'],  # 18
        'VAL': ['N  ' , 'CA ' , 'C  ' , 'CB ' , 'O  ' , '   ' , 'CG1' , 'CG2' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , 'OXT'],  # 19
        'UNK': ['   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   ' , '   '],  # 20
    },
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

AF2_ATOM37_WITH_ATOMIZATION = TokenEncoding(
    token_atoms={**AF2_ATOM37_ENCODING.token_atoms, 0: ['0' if i == 1 else '' for i in range(37)]},
    chemcomp_type_to_unknown=AF2_ATOM37_ENCODING.chemcomp_type_to_unknown,
)
"""AF2's atom37 encoding with atomization support.

Reference:
    `AlphaFold residue_constants.py <https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/common/residue_constants.py#L492-L544>`_
"""

# fmt: off
NA_ATOM37_ENCODING = TokenEncoding(
    token_atoms={
        #        0     1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36
        #        P     C1'    C2'    O2'    C3'    O3'    C4'    O4'    C5'    O5'    OP1    OP2    N9     C8     N7     C5     C4     N3     C2     N1     C6     N6     N2     O6     N1     C2     O2     N3     C4     C5     C6     N4     O4     C7     -      -      -
        #        ^     ^MUST BE SLOT 1                                                               |<------- Purine base atoms ------>|      |<------ Pyrimidine base atoms ----->|
        'DA': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DC': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'DG': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DT': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  'C7',  '',    '',    ''],
        'DN': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'A':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'C':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'G':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'U':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  '',    '',    '',    ''],
        'N':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
    },
    chemcomp_type_to_unknown=(
        {chem_type: UNKNOWN_DNA for chem_type in DNA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_RNA for chem_type in RNA_LIKE_CHEM_TYPES}
    ),
)
"""Nucleic acid atom37-like encoding for DNA and RNA.

Provides a unified 37-slot encoding for both DNA and RNA nucleotides, analogous to the
protein atom37 encoding. Key features:

- Slot 0: P (phosphate backbone)
- Slot 1: C1' (prime) (anomeric carbon - analogous to CA in proteins)
- Slot 3: O2' (prime) (present in RNA, empty in DNA)
- Slots 12-23: Purine base atoms (A, G, DA, DG)
- Slots 24-33: Pyrimidine base atoms (C, U, T, DC, DT)
- No hydrogens included (heavy atoms only)

This encoding ensures that structurally equivalent atoms across different nucleotides
occupy the same slot, while maintaining unique positions for purine vs pyrimidine atoms
that have different structural roles despite sharing atom names.
"""
# fmt: on

# fmt: off
UNIFIED_ATOM37_ENCODING = TokenEncoding(
    token_atoms={
        # Mask token (class 0)
        '<M>': ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],

        # Standard amino acids (classes 1-20)
        #        0       1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31      32      33      34      35      36
        'ALA': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ARG': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NE ', '   ', '   ', '   ', '   ', '   ', 'NH1', 'NH2', '   ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'ASN': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'ND2', 'OD1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ASP': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OD1', 'OD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'CYS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', '   ', '   ', 'SG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLN': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NE2', 'OE1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLU': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OE1', 'OE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLY': ['N  ', 'CA ', 'C  ', '   ', 'O  ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'HIS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD2', 'ND1', '   ', '   ', '   ', '   ', '   ', 'CE1', '   ', '   ', '   ', '   ', 'NE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ILE': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', 'CG1', 'CG2', '   ', '   ', '   ', '   ', 'CD1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'LEU': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'LYS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CE ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NZ ', 'OXT'],
        'MET': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'SD ', 'CE ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'PHE': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', 'CE1', 'CE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'PRO': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'SER': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', 'OG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'THR': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', 'CG2', '   ', 'OG1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'TRP': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CE2', 'CE3', '   ', 'NE1', '   ', '   ', '   ', 'CH2', '   ', '   ', '   ', '   ', 'CZ2', 'CZ3', '   ', 'OXT'],
        'TYR': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', 'CE1', 'CE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OH ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'VAL': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', 'CG1', 'CG2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],

        # Unknown amino acid (class 21)
        'UNK': ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],

        # RNA nucleotides (classes 22-25): A, C, G, U
        #       0     1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36
        'A':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'C':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'G':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'U':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  '',    '',    '',    ''],

        # Unknown RNA (class 26)
        'N':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],

        # DNA nucleotides (classes 27-30): DA, DC, DG, DT
        'DA': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DC': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'DG': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DT': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  'C7',  '',    '',    ''],

        # Unknown DNA (class 31)
        'DN': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],

        # Atomised token (class 32) - placeholder for atomised small molecules, always put atom in the second position
        '<A>': ['   ', 'X', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],
    },
    chemcomp_type_to_unknown=(
        {chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES}
        | {chem_type: "DN" for chem_type in DNA_LIKE_CHEM_TYPES}
        | {chem_type: "N" for chem_type in RNA_LIKE_CHEM_TYPES}
    ),
)
"""Unified atom37 encoding for all token types in ConditionalResidueTypeSeqFeat.

Provides a comprehensive 37-slot encoding that encompasses:
- Class 0: MASK token (special masking token)
- Classes 1-20: Standard amino acids (ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
                LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL)
- Class 21: UNK (unknown amino acid)
- Classes 22-25: RNA nucleotides (A, C, G, U)
- Class 26: N (unknown RNA)
- Classes 27-30: DNA nucleotides (DA, DC, DG, DT)
- Class 31: DN (unknown DNA)
- Class 32: ATOMIZED (atomized small molecule token)

This encoding is compatible with the conditional residue type feature used in protein
foundation models, enabling unified handling of proteins, RNA, DNA, and small molecules
in a single representation space.

Usage:
    UNIFIED_ATOM37_ENCODING serves as the single source of truth for:
    - Atom37 layout operations (coordinate processing):
        * atom_array_to_encoding() / atom_array_from_encoding()
        * Converting between AtomArray and atom37 coordinate tensors
    - Sequence encoding operations (residue type indices):
        * Use UNIFIED_ATOM37_ENCODING.token_to_idx to encode residue names
        * Use UNIFIED_ATOM37_ENCODING.idx_to_token to decode indices
"""
# fmt: on

# fmt: off
RF2AA_TOKEN_TO_STANDARD_TOKEN = {
    'ALA': 'ALA',
    'ARG': 'ARG',
    'ASN': 'ASN',
    'ASP': 'ASP',
    'CYS': 'CYS',
    'GLN': 'GLN',
    'GLU': 'GLU',
    'GLY': 'GLY',
    'HIS': 'HIS',
    'ILE': 'ILE',
    'LEU': 'LEU',
    'LYS': 'LYS',
    'MET': 'MET',
    'PHE': 'PHE',
    'PRO': 'PRO',
    'SER': 'SER',
    'THR': 'THR',
    'TRP': 'TRP',
    'TYR': 'TYR',
    'VAL': 'VAL',
    'UNK': 'UNK',
    'MAS': '<M>',
    ' DA': 'DA',
    ' DC': 'DC',
    ' DG': 'DG',
    ' DT': 'DT',
    ' DX': 'DN',
    ' RA': 'A',
    ' RC': 'C',
    ' RG': 'G',
    ' RU': 'U',
    ' RX': 'N',
    'HIS_D': 'HIS_D',
    'Al': 13,
    'As': 33,
    'Au': 79,
    'B': 5,
    'Be': 4,
    'Br': 35,
    'C': 6,
    'Ca': 20,
    'Cl': 17,
    'Co': 27,
    'Cr': 24,
    'Cu': 29,
    'F': 9,
    'Fe': 26,
    'Hg': 80,
    'I': 53,
    'Ir': 77,
    'K': 19,
    'Li': 3,
    'Mg': 12,
    'Mn': 25,
    'Mo': 42,
    'N': 7,
    'Ni': 28,
    'O': 8,
    'Os': 76,
    'P': 15,
    'Pb': 82,
    'Pd': 46,
    'Pr': 59,
    'Pt': 78,
    'Re': 75,
    'Rh': 45,
    'Ru': 44,
    'S': 16,
    'Sb': 51,
    'Se': 34,
    'Si': 14,
    'Sn': 50,
    'Tb': 65,
    'Te': 52,
    'U': 92,
    'W': 74,
    'V': 23,
    'Y': 39,
    'Zn': 30,
    'ATM': 0
}
"""Dictionary to interconvert between RF2AA token names and standardized token names."""

RF2AA_STANDARDIZED_TOKENS = list(RF2AA_TOKEN_TO_STANDARD_TOKEN.values())
"""List of standardized tokens in RF2AA."""

RF2_ATOM14_ENCODING = TokenEncoding(
    token_atoms={
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
        'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
        'UNK': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
        '<M>': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '']
    },
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)
"""RF2 atom14 encoding for proteins.

- Encodes only the heavy atoms (max 14, for ``TRP``)
- Includes 1 unknown tokens: ``UNK``

Print it out to see a visual representation of the encoding.
"""

RF2_ATOM23_ENCODING = TokenEncoding(
    token_atoms={
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', '', '', '', '', '', '', '', '', '', ''],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', ''],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2', '', '', '', '', '', '', '', '', ''],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', '', '', '', '', '', '', '', '', '', ''],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'UNK': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        '<M>': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        'DA': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N6', '', ''],
        'DC': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', ''],
        'DG': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N2', 'O6', ''],
        'DT': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6', '', '', ''],
        'DN': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", '', '', '', '', '', '', '', '', '', '', '', ''],
        'A': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9', ''],
        'C': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', ''],
        'G': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9'],
        'U': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', '', '', ''],
        'N': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", '', '', '', '', '', '', '', '', '', '', '']
    },
    chemcomp_type_to_unknown=(
        {chem_type: UNKNOWN_AA for chem_type in AA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_DNA for chem_type in DNA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_RNA for chem_type in RNA_LIKE_CHEM_TYPES}
    ),
)
"""RF2 atom23 encoding for proteins and nucleic acids.

- Encodes only the heavy atoms (max 22, for ``RG``)
- Includes 3 unknown tokens: ``UNK`` for proteins, ``DN`` for dna, ``N`` for RNA

Print it out to see a visual representation of the encoding.
"""

RF2_ATOM36_ENCODING = TokenEncoding(
    token_atoms={
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', 'HE', '1HH1', '2HH1', '1HH2', '2HH2'],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD2', '2HD2', '', '', '', '', '', '', ''],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '', '', '', '', '', '', '', '', ''],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', 'HG', '', '', '', '', '', '', '', ''],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE2', '2HE2', '', '', '', '', ''],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '', '', '', '', '', '', ''],
        'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', '1HA', '2HA', '', '', '', '', '', '', '', '', '', ''],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '2HD', '1HE', '2HE', '', '', '', '', '', ''],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', 'HB', '1HG2', '2HG2', '3HG2', '1HG1', '2HG1', '1HD1', '2HD1', '3HD1', '', ''],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', 'HG', '1HD1', '2HD1', '3HD1', '1HD2', '2HD2', '3HD2', '', ''],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '1HE', '2HE', '1HZ', '2HZ', '3HZ'],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE', '2HE', '3HE', '', '', '', ''],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '2HD', '1HE', '2HE', 'HZ', '', '', '', ''],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '', '', '', '', '', ''],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HG', 'HA', '1HB', '2HB', '', '', '', '', '', '', '', ''],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HG1', 'HA', 'HB', '1HG2', '2HG2', '3HG2', '', '', '', '', '', ''],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '1HE', 'HZ2', 'HH2', 'HZ3', 'HE3', '', '', ''],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '1HE', '2HE', '2HD', 'HH', '', '', '', ''],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', 'HB', '1HG1', '2HG1', '3HG1', '1HG2', '2HG2', '3HG2', '', '', '', ''],
        'UNK': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        '<M>': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        'DA': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N6', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', ''],
        'DC': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', ''],
        'DG': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N2', 'O6', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', ''],
        'DT': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H3', 'H71', 'H72', 'H73', 'H6', ''],
        'DN': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", '', '', '', '', '', '', '', '', '', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", '', '', '', '', '', ''],
        'A': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', ''],
        'C': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', ''],
        'G': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', ''],
        'U': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H3', 'H5', 'H6', '', '', ''],
        'N': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", '', '', '', '', '', '', '', '', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", '', '', '', '', '', '']
    },
    chemcomp_type_to_unknown=(
        {chem_type: UNKNOWN_AA for chem_type in AA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_DNA for chem_type in DNA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_RNA for chem_type in RNA_LIKE_CHEM_TYPES}
    ),
)


RF2AA_ATOM36_ENCODING = TokenEncoding(
    token_atoms={
        'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', 'HE', '1HH1', '2HH1', '1HH2', '2HH2'],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD2', '2HD2', '', '', '', '', '', '', ''],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '', '', '', '', '', '', '', '', ''],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', 'HG', '', '', '', '', '', '', '', ''],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE2', '2HE2', '', '', '', '', ''],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '', '', '', '', '', '', ''],
        'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', '1HA', '2HA', '', '', '', '', '', '', '', '', '', ''],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '2HD', '1HE', '2HE', '', '', '', '', '', ''],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', 'HB', '1HG2', '2HG2', '3HG2', '1HG1', '2HG1', '1HD1', '2HD1', '3HD1', '', ''],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', 'HG', '1HD1', '2HD1', '3HD1', '1HD2', '2HD2', '3HD2', '', ''],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '1HE', '2HE', '1HZ', '2HZ', '3HZ'],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE', '2HE', '3HE', '', '', '', ''],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '2HD', '1HE', '2HE', 'HZ', '', '', '', ''],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '', '', '', '', '', ''],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HG', 'HA', '1HB', '2HB', '', '', '', '', '', '', '', ''],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HG1', 'HA', 'HB', '1HG2', '2HG2', '3HG2', '', '', '', '', '', ''],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '1HE', 'HZ2', 'HH2', 'HZ3', 'HE3', '', '', ''],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '1HD', '1HE', '2HE', '2HD', 'HH', '', '', '', ''],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', 'HB', '1HG1', '2HG1', '3HG1', '1HG2', '2HG2', '3HG2', '', '', '', ''],
        'UNK': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        '<M>': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '3HB', '', '', '', '', '', '', '', ''],
        'DA': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N6', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', ''],
        'DC': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', ''],
        'DG': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N2', 'O6', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', ''],
        'DT': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H3', 'H71', 'H72', 'H73', 'H6', ''],
        'DN': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", '', '', '', '', '', '', '', '', '', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", '', '', '', '', '', ''],
        'A': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', ''],
        'C': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', ''],
        'G': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', ''],
        'U': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H3', 'H5', 'H6', '', '', ''],
        'N': ['OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C1'", "C2'", "O2'", '', '', '', '', '', '', '', '', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", '', '', '', '', '', ''],
        'HIS_D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'NE2', 'CD2', 'CE1', 'ND1', '', '', '', '', '', '', '', '', '', '', '', '', '', 'H', 'HA', '1HB', '2HB', '2HD', '1HE', '1HD', '', '', '', '', '', ''],
        13: ['', '13', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        33: ['', '33', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        79: ['', '79', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        5: ['', '5', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        4: ['', '4', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        35: ['', '35', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        6: ['', '6', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        20: ['', '20', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        17: ['', '17', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        27: ['', '27', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        24: ['', '24', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        29: ['', '29', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        9: ['', '9', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        26: ['', '26', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        80: ['', '80', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        53: ['', '53', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        77: ['', '77', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        19: ['', '19', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        3: ['', '3', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        12: ['', '12', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        25: ['', '25', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        42: ['', '42', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        7: ['', '7', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        28: ['', '28', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        8: ['', '8', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        76: ['', '76', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        15: ['', '15', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        82: ['', '82', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        46: ['', '46', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        59: ['', '59', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        78: ['', '78', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        75: ['', '75', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        45: ['', '45', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        44: ['', '44', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        16: ['', '16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        51: ['', '51', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        34: ['', '34', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        14: ['', '14', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        50: ['', '50', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        65: ['', '65', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        52: ['', '52', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        92: ['', '92', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        74: ['', '74', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        23: ['', '23', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        39: ['', '39', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        30: ['', '30', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        0: ['', '0', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    },
    chemcomp_type_to_unknown=(
        {chem_type: UNKNOWN_AA for chem_type in AA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_DNA for chem_type in DNA_LIKE_CHEM_TYPES}
        | {chem_type: UNKNOWN_RNA for chem_type in RNA_LIKE_CHEM_TYPES}
    ),
)
"""RF2AA all atom encoding for proteins, nucleic acids and various other elements
    - Encodes heavy atoms and hydrogens (max 36 in total)
    - Includes 3 unknown tokens: `UNK` for proteins, `DN` for dna, `N` for RNA
    - Covers:
        - 20 amino acids (+ unknown, + mask),
        - 4  DNA bases (+ unknown),
        - 4  RNA bases (+ unknown),
        - 1  outdated histindine token `HIS_D`
        - 45 atom tokens (+ unknown)
"""
# fmt: on

# NOTE: There was a bug in the original code that saved the RF2 templates: Tryptophan (AA17) was using
#  a wrong atom name ordering. This was fixed in the public version of the code:
#  https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L2068C1-L2070C285
#  but we include the legacy (=broken) encoding here to, to be able to correctly decode the legacy templates
_legacy_rf2_atom14_token_atoms = copy.deepcopy(RF2_ATOM14_ENCODING.token_atoms)
_legacy_rf2_atom14_token_atoms["TRP"] = np.array(
    [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ]
)
LEGACY_RF2_ATOM14_ENCODING = TokenEncoding(
    token_atoms=_legacy_rf2_atom14_token_atoms,
    chemcomp_type_to_unknown=RF2_ATOM14_ENCODING.chemcomp_type_to_unknown,
)

# fmt: off
AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    *STANDARD_AA, UNKNOWN_AA,
    # 4 RNA + 1 unknown RNA
    *STANDARD_RNA, UNKNOWN_RNA,
    # 4 DNA + 1 unknown DNA
    *STANDARD_DNA, UNKNOWN_DNA,
    # 1 gap
    GAP,
)
"""Sequence tokens in AF3"""
# fmt: on


class AF3SequenceEncoding:
    """Encodes and decodes sequence tokens for AlphaFold 3.

    This class provides functionality to convert between residue names and their
    corresponding integer encodings as used in AlphaFold 3. It handles standard
    amino acids, RNA, DNA, and unknown residues.
    """

    def __init__(self):
        # Load CCD from biotite
        ccd = struc.info.ccd.get_ccd()

        # Get all residue names and their corresponding chemtypes
        self.all_res_names = ccd["chem_comp"]["id"].as_array()
        self.all_res_chemtypes = np.char.upper(ccd["chem_comp"]["type"].as_array())

        # Get boolean arrays for each chemtype

        self.is_rna_like = np.isin(self.all_res_chemtypes, list(RNA_LIKE_CHEM_TYPES))
        self.is_dna_like = np.isin(self.all_res_chemtypes, list(DNA_LIKE_CHEM_TYPES))
        self.is_aa_like = np.isin(self.all_res_chemtypes, list(AA_LIKE_CHEM_TYPES))

        # Build mappings for all CCD residue names to AF3 tokens
        res_name_to_token = dict(zip(self.all_res_names[self.is_rna_like], cycle([UNKNOWN_RNA])))
        res_name_to_token |= dict(zip(self.all_res_names[self.is_dna_like], cycle([UNKNOWN_DNA])))
        res_name_to_token |= dict(zip(AF3_TOKENS, AF3_TOKENS, strict=False))
        self.res_name_to_token = res_name_to_token

        # Build mappings for AF3 tokens to indices
        self.af3_token_to_int = {token: i for i, token in enumerate(AF3_TOKENS)}

    @property
    def tokens(self) -> list[str]:
        return AF3_TOKENS

    def res_name_to_af3_token(self, res_name: str) -> str:
        return np.vectorize(lambda res_name: self.res_name_to_token.get(res_name, UNKNOWN_AA))(res_name)

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self.af3_token_to_int

    @cached_property
    def idx_to_token(self) -> np.ndarray:
        return np.array(AF3_TOKENS)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def encode(self, res_names: Sequence[str]) -> np.ndarray:
        # NOTE: Defined here rather than as attribute to allow pickling for multiprocessing
        encode_func = np.vectorize(lambda x: self.af3_token_to_int.get(x, self.af3_token_to_int[UNKNOWN_AA]))
        return encode_func(res_names)

    def decode(self, token_idxs: int | Sequence[int]) -> np.ndarray:
        return self.idx_to_token[token_idxs]
