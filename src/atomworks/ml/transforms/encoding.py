"""Transforms and helper functions to convert from `AtomArray` objects to various encoding schemes.

During encoding, sequences of tokens are converted to sequences of integers, and the
AtomArray of coordinates is converted to a (N_token, N_atoms_per_token, 3) tensor.

The token type (residue-level or atom-level) is encoded as a boolean in the `atomize` flag.
"""

from logging import getLogger
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray
from torch.nn import functional as F  # noqa: N812

from atomworks.common import KeyToIntMapper, exists
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.io.utils.ccd import get_std_to_alt_atom_name_map
from atomworks.ml.encoding_definitions import AF3SequenceEncoding, TokenEncoding
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.atom_array import get_within_entity_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_count, get_token_starts, token_iter

logger = getLogger(__name__)


def atom_array_to_encoding(
    atom_array: AtomArray,
    encoding: TokenEncoding,
    default_coord: np.ndarray | float = float("nan"),
    occupancy_threshold: float = 0.0,
    extra_annotations: list[str] = [
        "chain_id",
        "chain_entity",
        "molecule_iid",
        "chain_iid",
        "transformation_id",
    ],
) -> dict:
    """
    Encode an atom array using a specified `TokenEncoding`.

    This function processes an `AtomArray` to generate encoded representations, including coordinates, masks,
    sequences, and additional annotations. The encoded data comes in numpy arrays which can readily be converted
    to tensors and used in machine learning tasks

    NOTE:
        - `n_token` refers to the number of tokens in the atom array.
        - `n_atoms_per_token` indicates the number of atoms associated with each token in the `encoding`.
          The number of atoms in a token corresponds to the number of residues in the atom array, unless
          the atom array has the `atomize` annotation, in which case the number of tokens may exceed the
          number of residues.

    Args:
        - atom_array (AtomArray): The atom array containing polymer information. If the atom array has the
          `atomize` annotation (True for atoms that should be atomized), the number of tokens will differ
          from the number of residues.
        - encoding (TokenEncoding): The encoding scheme to apply to the atom array.
        - default_coord (np.ndarray | float, optional): Default coordinate value to use for uninitialized
          coordinates. Defaults to float("nan").
        - occupancy_threshold (float, optional): Minimum occupancy for atoms to be considered resolved
          in the mask. Defaults to 0.0 (only completely unresolved atoms are masked).
        - extra_annotations (list[str], optional): A list of additional annotations to encode. These must
          be `id` style annotations (e.g., `chain_id`, `molecule_iid`). The encoding will be generated as
          integers, where the first occurrence of a given ID is encoded as `0`, and subsequent occurrences
          are encoded as `1`, `2`, etc. Defaults to
          ["chain_id", "chain_entity", "molecule_iid", "chain_iid", "transformation_id"].

    Returns:
        - dict: A dictionary containing the following keys:
            - `xyz` (np.ndarray): Encoded coordinates of shape [n_token, n_atoms_per_token, 3].
            - `mask` (np.ndarray): Encoded mask of shape [n_token, n_atoms_per_token], indicating which
              atoms are resolved in the encoded sequence.
            - `seq` (np.ndarray): Encoded sequence of shape [n_token].
            - `token_is_atom` (np.ndarray): Boolean array of shape [n_token] indicating whether each token
              corresponds to an atom.
            - Various additional annotations encoded as extra keys in the dictionary. Each extra annotation
                that gets exposed is results in 2 keys in the dictionary. One for the encoded annotation itself
                and one mapping the annotation to integers if e.g. the original annotation was strings.
                For example, the defaults above result in:
                - `chain_id` (np.ndarray): Encoded chain IDs of shape [n_token].
                - `chain_id_to_int` (dict): Mapping of chain IDs to integers in the `chain_id` array.
                - `chain_entity` (np.ndarray): Encoded entity IDs of shape [n_token].
                - `chain_entity_to_int` (dict): Mapping of entity IDs to integers in the `chain_entity` array.
    """
    # Extract atom array information
    n_token = get_token_count(atom_array)

    # Init encoded arrays
    encoded_coord = np.full(
        (n_token, encoding.n_atoms_per_token, 3), fill_value=default_coord, dtype=np.float32
    )  # [n_token, n_atoms_per_token, 3] (float)

    encoded_mask = np.zeros((n_token, encoding.n_atoms_per_token), dtype=bool)  # [n_token, n_atoms_per_token] (bool)
    encoded_seq = np.empty(n_token, dtype=int)  # [n_token] (int)
    encoded_token_is_atom = np.empty(n_token, dtype=bool)  # [n_token] (bool)

    # init additional annotation
    extra_annot_counters = {}
    extra_annot_encoded = {}
    for key in extra_annotations:
        if key in atom_array.get_annotation_categories():
            extra_annot_counters[key] = KeyToIntMapper()
            extra_annot_encoded[key] = []

    # Iterate over residues and encode (# TODO: Speed up by vectorizing if necessary)
    # ... record whether the atom array has the `atomize` annotation to deal with atomized residues
    has_atomize = "atomize" in atom_array.get_annotation_categories()
    for i, token in enumerate(token_iter(atom_array)):
        # ... extract token name
        # ... case 1: atom tokens (e.g. 6 - for carbon)
        if (has_atomize and token.atomize[0]) or len(token) == 1:
            token_name = (
                token.atomic_number[0]
                if "atomic_number" in token.get_annotation_categories()
                else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
            )
            token_is_atom = True
        # ... case 2: residue tokens (e.g. "ALA")
        else:
            token_name = token.res_name[0]
            token_is_atom = False

        if token_name not in encoding.token_to_idx:
            token_name = encoding.resolve_unknown_token_name(token_name, token_is_atom)
            assert token_name in encoding.token_to_idx, f"Unknown token name: {token_name}"

        # Encode sequence
        encoded_seq[i] = encoding.token_to_idx[token_name]

        # Encode if token is an `atom-level` token or a `residue-level` token
        encoded_token_is_atom[i] = token_is_atom

        # Encode coords
        for atom in token:
            atom_name = str(token_name) if token_is_atom else atom.atom_name
            # (token_name, atom_name) is e.g.
            #  ... ('ALA', 'CA') if  token_is_atom=False
            #  ... ('UNK', whatever) if token_is_atom=False but we had to resolve an unknown token
            #  ... (6, '6') if token_is_atom=True

            # ... case 1: atom name is in the encoding
            if (token_name, atom_name) in encoding.atom_to_idx:
                to_idx = encoding.atom_to_idx[(token_name, atom_name)]
                encoded_coord[i, to_idx, :] = atom.coord
                encoded_mask[i, to_idx] = atom.occupancy > occupancy_threshold

            # ... case 2: atom name does not exist for token, but token is an `unknown` token,
            #  so it's `ok` to not match
            elif token_name in encoding.unknown_tokens:
                continue

            # ... case 3: atom name is not in encoding, but token is, and try_matching_alt_atom_name_if_fails is True
            elif not token_is_atom:
                alt_to_std = get_std_to_alt_atom_name_map(token_name)
                alt_atom_name = alt_to_std.get(atom_name, None)
                if exists(alt_atom_name) and (token_name, alt_atom_name) in encoding.atom_to_idx:
                    to_idx = encoding.atom_to_idx[(token_name, alt_atom_name)]
                    encoded_coord[i, to_idx, :] = atom.coord

            # ... case 4: failed to find the relevant atom_name for this token when we should, so we raise an error
            else:
                msg = f"Atom ({token_name}, {atom_name}) not in encoding for token `{token_name}`"
                msg += "\nProblematic atom:\n"
                msg += f"{atom}"
                raise ValueError(msg)

        # Encode additional annotation
        for key in extra_annot_counters:
            annot = token.get_annotation(key)[0]
            extra_annot_encoded[key].append(extra_annot_counters[key](annot))

    return {
        "xyz": encoded_coord,  # [n_token_in_atom_array, n_atoms_per_token, 3] (float)
        "mask": encoded_mask,  # [n_token_in_atom_array, n_atoms_per_token] (bool)
        "seq": encoded_seq,  # [n_token_in_atom_array] (int)
        "token_is_atom": encoded_token_is_atom,  # [n_token_in_atom_array] (bool)
        **{annot: np.array(extra_annot_encoded[annot], dtype=np.int16) for annot in extra_annot_encoded},
        **{annot + "_to_int": extra_annot_counters[annot].key_to_id for annot in extra_annot_counters},
    }


def atom_array_from_encoding(
    encoded_coord: torch.Tensor | np.ndarray,
    encoded_mask: torch.Tensor | np.ndarray,
    encoded_seq: torch.Tensor | np.ndarray,
    encoding: TokenEncoding,
    chain_id: str = "A",  # TODO: Allow passing a numpy array of chain ids
    token_is_atom: torch.Tensor | np.ndarray | None = None,
    **other_annotations: np.ndarray | None,
    # TODO: Allow passing a res_id
) -> AtomArray:
    """Create an AtomArray from encoded coordinates, mask, and sequence.

    This function takes encoded data and reconstructs an AtomArray, which is a
    structured representation of atomic information. The encoded coordinates,
    mask, and sequence are used to populate the AtomArray, ensuring that all
    relevant annotations are included.

    Args:
        encoded_coord: Encoded coordinates tensor.
        encoded_mask: Encoded mask tensor.
        encoded_seq: Encoded sequence tensor.
        encoding: The encoding to use for encoding the atom array.
        chain_id: Chain ID. Can be a single string (e.g., "A")
          or a numpy array of shape (n_res,) corresponding to each residue. Defaults to "A".
        token_is_atom: Boolean mask indicating
          whether each token corresponds to an atom.
        **other_annotations: Additional annotations to include in the
          AtomArray. The shape must match one of the following:

            - scalar, for global annotations
            - (n_atom,) for per-atom annotations,
            - (n_res,) for per-residue annotations,
            - (n_chain,) for per-chain annotations.

    Returns:
        The created AtomArray containing the encoded atomic information.
    """
    # Turn tensors into numpy arrays if necessary
    _from_tensor = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x  # noqa E731
    encoded_coord = _from_tensor(encoded_coord)
    encoded_mask = _from_tensor(encoded_mask)
    encoded_seq = _from_tensor(encoded_seq)
    token_is_atom = _from_tensor(token_is_atom)
    other_annotations = {annot: _from_tensor(annot_arr) for annot, annot_arr in other_annotations.items()}

    # Extract token, element and atom name information via the encoding
    seq = encoding.idx_to_token[encoded_seq]  # [n_res] (str)
    element = encoding.idx_to_element[encoded_seq]  # [n_res, n_atoms_per_token] (str)
    atom_name = encoding.idx_to_atom[encoded_seq]  # [n_res, n_atoms_per_token] (str)

    # Determine which atoms should exist in each token, and how many atoms are in each token
    atom_should_exist = atom_name != ""  # [n_res, n_atoms_per_token] (bool)
    atoms_per_res = np.sum(atom_should_exist, axis=1)  # [n_res] (int)

    # Set up atom array
    n_res = len(seq)
    n_atom = np.sum(atoms_per_res)
    atom_array = AtomArray(length=n_atom)

    # ... flatten occupancy & validate that masking did not miss any existing atoms
    atom_array.set_annotation("occupancy", np.asarray(encoded_mask[atom_should_exist], dtype=np.float32))
    assert np.sum(encoded_mask) == np.sum(atom_array.occupancy)

    # ... set atomize annotation if `token_is_atom` is provided
    if token_is_atom is not None:
        atom_array.set_annotation("atomize", np.asarray(token_is_atom[atom_should_exist], dtype=np.bool_))

    # ... flatten and annotate coordinates
    atom_array.coord = encoded_coord[atom_should_exist]

    # ... flatten atom names and strip whitespace in atom names
    _strip_whitespace = np.vectorize(lambda x: x.strip())
    atom_array.atom_name = _strip_whitespace(atom_name[atom_should_exist])

    # ... flatten element info
    atom_array.element = element[atom_should_exist]
    atom_array.atomic_number = np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(atom_array.element)

    # ... repeat residue name and id for each atom in the residue
    atom_array.res_name = np.repeat(seq, atoms_per_res)
    atom_array.res_id = np.repeat(np.arange(1, n_res + 1), atoms_per_res)
    atom_array.atom_id = np.arange(n_atom)

    if np.isscalar(chain_id):
        # ... assign same, global chain id to all atoms
        atom_array.chain_id = np.repeat(np.array(chain_id), n_atom)
    else:
        # ... repeat chain id for each atom in the residue
        atom_array.chain_id = np.repeat(chain_id, atoms_per_res)
    unique_chains, atoms_per_chain = np.unique(atom_array.chain_id, return_counts=True)

    # Add additional atom/residue/chain/global annotations
    for annot, annot_arr in other_annotations.items():
        if np.isscalar(annot_arr):
            annot_arr = np.repeat(annot_arr, n_atom)
        elif annot_arr.shape[0] == n_atom:
            atom_array.set_annotation(annot, annot_arr)
        elif annot_arr.shape[0] == n_res:
            atom_array.set_annotation(annot, np.repeat(annot_arr, atoms_per_res))
        elif annot_arr.shape[0] == len(unique_chains):
            atom_array.set_annotation(annot, np.repeat(annot_arr, atoms_per_chain))
        else:
            raise ValueError(
                f"Annotation `{annot}` has incorrect shape: {annot_arr.shape}. Expected [n_atom] ({n_atom}) or [n_res] ({n_res})."
            )

    return atom_array


class EncodeAtomArray(Transform):
    """Encode an atom array to an arbitrary `TokenEncoding`.

    This will add the following information to the data dict:
        - `encoding` (dict)
            - `xyz`: Atom coordinates (`xyz`)
            - `mask`: Atom mask giving information about which atoms are resolved in the encoded sequence (`mask`)
            - `seq`: Token sequence (`seq`)
            - `token_is_atom`: Token type (atom or residue) (`token_is_atom`)
            - Various other optional annotations such as `chain_id`, `chain_entity`, etc. See `atom_array_to_encoding`
              for more details.
    """

    def __init__(
        self,
        encoding: TokenEncoding,
        default_coord: float | np.ndarray = float("nan"),
        occupancy_threshold: float = 0.0,
        extra_annotations: list[str] = [
            "chain_id",
            "chain_entity",
            "molecule_iid",
            "chain_iid",
            "transformation_id",
        ],
    ):
        """
        Convert an atom array to an encoding.

        Args:
            - `encoding` (TokenEncoding): The encoding to use for encoding the atom array.
            - `default_coord` (float | np.ndarray, optional): Default coordinate value. Defaults to float("nan").
            - `occupancy_threshold` (float, optional): Minimum occupancy for atoms to be considered resolved
                in the mask. Defaults to 0.0 (only completely unresolved atoms are masked).
            - `extra_annotations` (list[str], optional): Extra annotations to encode. These must be `id` style annotations
                like `chain_id` or `molecule_iid`, as the encoding will be generated as `int`s. Each first occurrence
                of a given `id` will be encoded as `0`, and each subsequent occurrence will be encoded as `1`, `2`, etc.
                Defaults to ["chain_id", "chain_entity", "molecule_iid", "chain_iid", "transformation_id"].
        """
        if not isinstance(encoding, TokenEncoding):
            raise ValueError(f"Encoding must be a `TokenEncoding`, but got: {type(encoding)}.")
        self.encoding = encoding
        self.default_coord = default_coord
        self.occupancy_threshold = occupancy_threshold
        self.extra_annotations = extra_annotations

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        encoded = atom_array_to_encoding(
            atom_array,
            encoding=self.encoding,
            default_coord=self.default_coord,
            occupancy_threshold=self.occupancy_threshold,
            extra_annotations=self.extra_annotations,
        )

        data["encoded"] = encoded
        return data


class AddTokenAnnotation(Transform):
    """
    Add a token annotation to the atom array. This is mostly meant as a debug transform and not expected to be used in production.

    Sets the `token` annotation to the token name for each atom in the atom array.
    """

    def __init__(self, encoding: TokenEncoding):
        self.encoding = encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Iterate over residues and tokenize
        # ... record whether the atom array has the `atomize` annotation
        has_atomize = "atomize" in atom_array.get_annotation_categories()
        tokens = []
        for _i, token in enumerate(token_iter(atom_array)):
            # ... extract token name
            if (has_atomize and token.atomize[0]) or len(token) == 1:
                assert len(token.atomic_number) == 1, "Atomize annotation is only allowed for single atoms."
                token_name = token.atomic_number[0]
                token_is_atom = True
            else:
                token_name = token.res_name[0]
                token_is_atom = False

            if token_name not in self.encoding.token_to_idx:
                token_name = self.encoding.resolve_unknown_token_name(token_name, token_is_atom)
            tokens.extend([token_name] * len(token))

        atom_array.set_annotation("token", np.asarray(tokens, dtype=object))
        return data


class EncodeAF3TokenLevelFeatures(Transform):
    """A transform that encodes token-level features like AF3. The token-level features are returned as:

    feats:
        (Standard AF3 token-level features)

        residue_index
            Residue number in the token's original input chain (pre-crop)
        token_index
            Token number. Increases monotonically; does not restart at 1 for new
            chains. (Runs from 0 to N_tokens)
        asym_id
            Unique integer for each distinct chain (pn_unit_iid)
            NOTE: We use pn_unit_iid rather than chain_iid to be more consistent
            with handling of multi-residue/multi-chain ligands (especially sugars)
        entity_id
            Unique integer for each distinct sequence (pn_unit entity)
        sym_id
            Unique integer within chains of this sequence. E.g. if pn_units A, B and C
            share a sequence but D does not, their sym_ids would be [0, 1, 2, 0].
        restype
            Integer encoding of the sequence. 32 possible values: 20 AA + unknown,
            4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap. Ligands are
            represented as unknown amino acid (UNK)
        is_protein
            whether a token is of protein type
        is_rna
            whether a token is of RNA type
        is_dna
            whether a token is of DNA type
        is_ligand
            whether a token is a ligand residue

        (Custom token-level features)

        is_atomized
            whether a token is an atomized token

    feat_metadata:
        asym_name
            The asymmetric unit name for each id in asym_id. Acts as a legend.
        entity_name
            The entity name for each id in entity_id. Acts as a legend.
        sym_name
            The symmetric unit name for each id in sym_id. Acts as a legend.

    Reference:
        `Section 2.8 of the AF3 supplementary (Table 5) <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """

    def __init__(self, sequence_encoding: AF3SequenceEncoding):
        self.sequence_encoding = sequence_encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            [
                "atomize",
                "pn_unit_iid",
                "chain_entity",
                "res_name",
                "within_chain_res_idx",
            ],
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        # ... get token-level array
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]

        # ... identifier tokens
        # ... (residue)
        residue_index = token_level_array.within_chain_res_idx
        # ... (token)
        token_index = np.arange(len(token_starts))
        # ... (chain instance)
        asym_name, asym_id = np.unique(token_level_array.pn_unit_iid, return_inverse=True)
        # ... (chain entity)
        entity_name, entity_id = np.unique(token_level_array.pn_unit_entity, return_inverse=True)
        # ... (within chain entity)
        sym_name, sym_id = get_within_entity_idx(token_level_array, level="pn_unit")

        # ... sequence tokens
        restype = self.sequence_encoding.encode(token_level_array.res_name)

        # HACK: MSA transformations rely on the encoded query sequence being stored in "encoded/seq"
        # We could consider finding a consistent place to store the encoded query sequence across RF2AA and AF3 (e.g., "encoded" vs. "feats/restype")
        data["encoded"] = {"seq": restype}

        # ...one-hot encode the restype (NOTE: We one-hot encode here, since we have access to the sequence encoding object)
        restype = F.one_hot(torch.tensor(restype), num_classes=self.sequence_encoding.n_tokens).numpy()

        # ... molecule type
        _aa_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_aa_like]
        is_protein = np.isin(token_level_array.res_name, _aa_like_res_names)

        _rna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_rna_like]
        is_rna = np.isin(token_level_array.res_name, _rna_like_res_names)

        _dna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_dna_like]
        is_dna = np.isin(token_level_array.res_name, _dna_like_res_names)

        is_ligand = ~(is_protein | is_rna | is_dna)

        # ... add to data dict
        if "feats" not in data:
            data["feats"] = {}
        if "feat_metadata" not in data:
            data["feat_metadata"] = {}

        # ... add to data dict
        data["feats"] |= {
            "residue_index": residue_index,  # (N_tokens) (int)
            "token_index": token_index,  # (N_tokens) (int)
            "asym_id": asym_id,  # (N_tokens) (int)
            "entity_id": entity_id,  # (N_tokens) (int)
            "sym_id": sym_id,  # (N_tokens) (int)
            "restype": restype,  # (N_tokens, 32) (float, one-hot)
            "is_protein": is_protein,  # (N_tokens) (bool)
            "is_rna": is_rna,  # (N_tokens) (bool)
            "is_dna": is_dna,  # (N_tokens) (bool)
            "is_ligand": is_ligand,  # (N_tokens) (bool)
            "is_atomized": token_level_array.atomize,  # (N_tokens) (bool)
        }
        data["feat_metadata"] |= {
            "asym_name": asym_name,  # (N_asyms)
            "entity_name": entity_name,  # (N_entities)
            "sym_name": sym_name,  # (N_entities)
        }

        return data
