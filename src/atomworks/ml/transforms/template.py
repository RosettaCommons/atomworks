"""Transforms for adding and featurizing templates."""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from os import PathLike
from typing import Any, ClassVar

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from torch.nn.functional import normalize

from atomworks.common import exists
from atomworks.constants import NA_VALUES
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import (
    LEGACY_RF2_ATOM14_ENCODING,
    RF2AA_ATOM36_ENCODING,
    AF3SequenceEncoding,
    TokenEncoding,
)
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
    chain_instance_iter,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import atom_array_from_encoding, atom_array_to_encoding
from atomworks.ml.transforms.featurize_unresolved_residues import mask_polymer_residues_with_unresolved_frame_atoms
from atomworks.ml.utils.geometry import apply_inverse_rigid, rigid_from_3_points
from atomworks.ml.utils.numpy import select_data_by_id
from atomworks.ml.utils.token import get_token_count, get_token_starts

logger = logging.getLogger(__name__)


@dataclass
class RF2AATemplate:
    """Data class for holding template information in the RF, RF2 & RF2AA format.

    NOTE:
        - RF templates only exist for proteins
        - This is a helper class to cast the templates into a more readable format and also
          to provide an interface layer that allows us to deal with templates as atom_arrays, if
          we ever re-create templates or add templates for non-proteins
        - RF-style templates already come encoded in atom14 representation (RFAtom14, not AF2Atom14)

    Keys:
        - xyz: Tensor([1, n_templates x n_atoms_per_template, 14, 3]), raw coordinates of all templates
        - mask: Tensor([1, n_templates x n_atom_per_template, 14]), mask of all templates
        - qmap: Tensor([1, n_templates x n_atom_per_template, 2]), alignment mapping of all templates
            - index 0: which index in the query protein this template index matches to
            - index 1: which template index this matches to
        - f0d: Tensor([1, n_templates, 8?]), [0,:,4] holds sequence identity info
        - f1d: Tensor([1, n_templates x n_atoms_per_template, 3]), something in there may be related to template confidence, gaps?
        - seq: Tensor([1, 100677]) (tensor, encoded with Chemdata.aa2num encoding)
        - ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
        - label: list[str]  # holds the lookup_id for this template
    """

    xyz: torch.Tensor  # [1, n_templates x n_atoms_per_template, 14, 3]
    mask: torch.Tensor  # [1, n_templates x n_atom_per_template, 14]
    qmap: torch.Tensor  # [1, n_templates x n_atom_per_template, 2]
    f0d: torch.Tensor  # [1, n_templates, 8?]
    f1d: torch.Tensor  # [1, n_templates x n_atoms_per_template, 3]
    seq: torch.Tensor  # [1, n_templates x n_atoms_per_template]
    ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
    label: list[str]  # holds the lookup_id for this template

    # RF2AA ideal N, CA, C initial coordinates (protein), copied from `chemdata` in RF2AA to decouple from `atomworks.ml`
    _INIT_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
    _INIT_CA = torch.zeros_like(_INIT_N)
    _INIT_C = torch.tensor([1.5233, 0.000, 0.000]).float()
    RF2AA_INIT_TEMPLATE_COORDINATES = torch.full((36, 3), np.nan)
    RF2AA_INIT_TEMPLATE_COORDINATES[:3] = torch.stack((_INIT_N, _INIT_CA, _INIT_C), dim=0)  # (3,3)

    def __post_init__(self):
        self.ids = np.array(self.ids).flatten().squeeze()  # Flatten the list of tuples into an array
        # Convert all tensors to numpy
        self.xyz = self.xyz.numpy()
        self.mask = self.mask.numpy()
        self.qmap = self.qmap.numpy()
        self.f0d = self.f0d.numpy()
        self.f1d = self.f1d.numpy()
        self.seq = self.seq.numpy()
        self.label = np.array(self.label)

    @property
    def lookup_id(self) -> str:
        return self.label[0]

    @property
    def n_templates(self) -> int:
        return self.f0d.shape[1]

    @property
    def seq_similarity_to_query(self) -> np.ndarray:
        return self.f0d[0, :, 4]

    @property
    def alignment_confidence(self) -> np.ndarray:
        return self.f1d[0, :, 2]

    @property
    def pdb_ids(self) -> np.ndarray:
        return np.array([i.split("_")[0] for i in self.ids])

    @property
    def chain_ids(self) -> np.ndarray:
        return np.array([i.split("_")[1] for i in self.ids])

    @property
    def n_res_per_template(self) -> np.ndarray:
        return np.unique(self.qmap[:, :, 1], return_counts=True)[1]

    @property
    def max_aligned_query_res_idx(self) -> np.ndarray:
        aligned_query_res_idxs = self.qmap[0, :, 0]
        new_template_start_idxs = np.cumsum(self.n_res_per_template)[:-1]
        groups = np.split(aligned_query_res_idxs, new_template_start_idxs)
        # get max in each group (= template)
        return np.array([np.max(g) for g in groups])

    @property
    def template_ids(self) -> list[str]:
        return np.array(self.ids)

    def subset(self, template_idxs: list[int]) -> "RF2AATemplate":
        """
        Subset the template to only include the template indices specified in `template_idxs`.
        """
        assert np.unique(template_idxs).size == len(template_idxs), "`template_idxs` must be unique"

        # Subset the data
        template_atom_idxs = np.where(np.isin(self.qmap[0, :, 1], template_idxs))[0]
        self.xyz = self.xyz[:, template_atom_idxs]
        self.mask = self.mask[:, template_atom_idxs]
        self.qmap = self.qmap[:, template_atom_idxs]

        # Update internal template index to be from 0 to n_templates
        n_res_per_template = np.unique(self.qmap[:, :, 1], return_counts=True)[1]
        self.qmap[0, :, 1] = np.repeat(np.arange(len(template_idxs)), n_res_per_template)

        self.f0d = self.f0d[:, template_idxs]
        self.f1d = self.f1d[:, template_atom_idxs]
        self.seq = self.seq[:, template_atom_idxs]
        self.ids = self.ids[template_idxs]
        return self

    def to_atom_array(self, template_idx: int) -> AtomArray:
        assert (
            isinstance(template_idx, int) and 0 <= template_idx <= self.n_templates - 1
        ), f"template_idx must be an int between 0 and {self.n_templates - 1}, got {template_idx}"

        # Get pdb_id and chain_id
        template_id = self.ids[template_idx]
        pdb_id, chain_id = template_id.split("_")

        # Get indices to select the residues for the template
        template_res_idxs = np.where(self.qmap[0, :, 1] == template_idx)[0]

        # Select the template data
        # ... coordinate info
        atom14_coords = self.xyz[0, template_res_idxs, :, :]
        # ... occupancy info
        atom14_mask = self.mask[0, template_res_idxs, :]
        # ... sequence info
        seq_tokenized = self.seq[0, template_res_idxs]

        # NOTE: There was a bug in the original code that saved the RF2 templates: Tryptophan (AA17) was using
        #  a wrong atom name ordering. This was fixed in the public version of the code:
        #  https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L2068C1-L2070C285
        #  and we include this fix here:
        # Create atom array
        atom_array = atom_array_from_encoding(
            atom14_coords,
            seq_tokenized,
            LEGACY_RF2_ATOM14_ENCODING,
            encoded_mask=atom14_mask,
        )
        n_atom = len(atom_array)

        # ... repeat chain id for each atom in the residue
        atom_array.chain_id = np.repeat(np.array(chain_id), n_atom)

        # ... set the `is_polymer` annotation to True (all templates are polymers)
        atom_array.set_annotation("is_polymer", np.full(n_atom, True))

        # ... append custom annotation for which residue in the query protein this template
        #  residue aligns to (indexing starts with 0 at query sequence start)
        aligned_query_res_idx = self.qmap[0, template_res_idxs, 0]
        atom_array.set_annotation("aligned_query_res_idx", struc.spread_residue_wise(atom_array, aligned_query_res_idx))

        # ... append custom annotation for alignment confidence
        alignment_confidence = self.f1d[0, template_res_idxs, 2]
        # NOTE: Some templates have the rare bug that the alignment confidence is `inf`. In this case
        #  we set it to 0.5 (since this was a presumably a parsing bug of the HHR file) and warn the user
        if np.isinf(alignment_confidence).any():
            logger.warning(f"Template {template_id} has `inf` alignment confidence. Setting to 0.5.")
            alignment_confidence = np.where(np.isinf(alignment_confidence), 0.5, alignment_confidence)
        atom_array.set_annotation("alignment_confidence", struc.spread_residue_wise(atom_array, alignment_confidence))

        # ...mask residues with unresolved backbone atoms
        atom_array = mask_polymer_residues_with_unresolved_frame_atoms(atom_array)

        return atom_array


def blank_rf2aa_template_features(
    n_template: int,
    n_token: int,
    encoding: TokenEncoding,
    mask_token_idx: int,
    init_coords: torch.Tensor | float,
) -> torch.Tensor:
    """
    Generates blank template features for RF2AA.

    Args:
        n_template (int): Number of templates.
        n_token (int): Number of tokens in the structure.
        encoding (TokenEncoding): Encoding object containing token and atom information.
        mask_token_idx (int, optional): Index of the mask token. Defaults to 20.
        init_coords (torch.Tensor | float, optional): Initial coordinates for the atoms.

    Returns:
        tuple: A tuple containing the following elements:
            - xyz (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_atoms_per_token, 3) containing the coordinates of the atoms.
            - t1d (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_tokens) containing the 1D template features.
            - mask (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_atoms_per_token) containing the mask information.
            - template_origin (np.ndarray): Array of shape (n_template,) containing the origin of the templates.
    """
    # TODO: Fix fill value
    # Initialize blank template features
    xyz = torch.full((n_template, n_token, encoding.n_atoms_per_token, 3), fill_value=float("nan"))
    mask = torch.zeros((n_template, n_token, encoding.n_atoms_per_token), dtype=torch.bool)
    t1d = torch.zeros((n_template, n_token, encoding.n_tokens))
    template_origin = np.full(n_template, "")

    # Fill in the initial coordinates and mask values
    xyz[:, :] = init_coords

    t1d[..., mask_token_idx] = 1.0  # Set the mask token to 1.0
    # NOTE: In RF2AA the last dim of t1d is the `confidence`. We set it here just
    #  for code clarity.
    _confidence = torch.zeros((n_template, n_token))
    t1d[..., -1] = _confidence

    return xyz, t1d, mask, template_origin


@cache
def _lazy_load_template_lookup_dict(template_lookup_path: PathLike) -> dict[str, int]:
    template_lookup_df = pd.read_csv(template_lookup_path, keep_default_na=False, na_values=NA_VALUES)
    template_lookup_df["HASH"] = template_lookup_df["HASH"].apply(lambda x: f"{x:06d}")
    pdb_chain_id_to_hash_dict = dict(
        zip(template_lookup_df["CHAINID"].tolist(), template_lookup_df["HASH"].tolist(), strict=False)
    )
    return pdb_chain_id_to_hash_dict


def _get_rf_template_id(pdb_id: str, chain_id: str, chain_type: ChainType, template_lookup_path: PathLike) -> str:
    """
    Retrieves the template lookup ID for a given PDB and chain ID combination.
    (NOTE: This is the `chid_to_hash` ID used for MSAs & Templates used in the original RF2AA)

    Parameters:
    - pdb_id (str): The PDB ID of the protein structure. E.g., "1A2K".
    - chain_id (str): The chain ID within the PDB structure. E.g., "A". Notably, no transformation ID.
    - chain_type (ChainType): The type of the chain, as defined in the ChainType enum.
    - template_lookup_path (PathLike): Path to the template MSA lookup file, typically on the DIGS.

    Returns:
    - str: The template lookup ID corresponding to the combined PDB and chain ID.
    """
    combined_id = f"{pdb_id}_{chain_id}"
    if chain_type == ChainType.POLYPEPTIDE_L:
        # For polypeptide(L) chains, we lookup the identified based on the mapping stored on disk
        # If we don't find a match, we append "_single_sequence" to the combined ID to ensure we won't find any MSAs
        return _lazy_load_template_lookup_dict(template_lookup_path=template_lookup_path).get(combined_id)
    elif chain_type == ChainType.RNA or chain_type == ChainType.DNA:
        # For nucleic acids, we use `{pdb_id}_{chain_id}` as the identifier
        return combined_id


def _load_rf_template(rf_template_id: str | None, template_base_dir: PathLike) -> torch.Tensor | None:
    if rf_template_id is None:
        # ... skip if no template ID (e.g. no matching template ID found in the lookup dict)
        return None

    path_to_template = f"{template_base_dir}/{rf_template_id[:3]}/{rf_template_id}.pt"
    if not os.path.exists(path_to_template):
        # ... skip if template file does not exist
        return None

    return torch.load(path_to_template, map_location="cpu", weights_only=True)


class AddRFTemplates(Transform):
    """
    Adds RF templates to the data.

    The templates are added to the data under the key `template`.

    Output features:
        - template (dict): A dictionary with chain IDs as keys and a list of templates for that chain as values.
            Each template is a dictionary with the following keys:
                - id (str): The template ID.
                - pdb_id (str): The PDB ID of the template.
                - chain_id (str): The chain ID of the template.
                - template_lookup_id (str): The lookup ID for the template - this is the `chid_to_hash` ID
                    used for MSAs & Templates used in the original RF2AA which is used to retrieve the template
                    from disk.
                - seq_similarity (float): The sequence similarity of the template to the query.
                - atom_array (AtomArray): The atom array of the template.
                - n_res (int): The number of residues in the template.
    """

    def __init__(
        self,
        max_n_template: int = 1,
        pick_top: bool = True,
        min_seq_similarity: float = 0.0,
        max_seq_similarity: float = 100.0,
        min_template_length: int = 0,
        filter_by_query_length: bool = False,
        template_lookup_path: PathLike | None = None,
        template_base_dir: PathLike | None = None,
    ):
        """
        Initialize the AddRFTemplates transform.

        Args:
            max_n_template (int): Maximum number of templates to add. If more `max_n_template` is larger than the
                number of available templates for a chain, all templates are added. Default is 1.
            pick_top (bool): Whether to pick the top templates based on sequence similarity if there are more than
                `max_n_template` templates available. Default is True.
            min_seq_similarity (float): Minimum sequence similarity for templates to be included. Default is 0.0.
            max_seq_similarity (float): Maximum sequence similarity for templates to be included. Default is 100.0.
            min_template_length (int): Minimum length of the template to be included. Default is 0.
            filter_by_query_length (bool): Whether to filter templates by query length. Default is False.
            template_lookup_path (PathLike): Path to the template lookup table. We attempt to load from the environment variable, and
                fall back to the default path on the DIGS if unset
            template_base_dir (PathLike): Base directory for the template files. We attempt to load from the environment variable, and fall back to the
                default path on the DIGS if unset

        Raises:
            AssertionError: If `min_seq_similarity` or `max_seq_similarity` are not between 0.0 and 100.0.
            AssertionError: If `n_template` is not a positive integer.
            AssertionError: If `min_template_length` is not a non-negative integer.
        """
        assert (
            0.0 <= min_seq_similarity <= 100.0
        ), f"min_seq_similarity must be between 0.0 and 100.0, got {min_seq_similarity}"
        assert (
            0.0 <= max_seq_similarity <= 100.0
        ), f"max_seq_similarity must be between 0.0 and 100.0, got {max_seq_similarity}"
        assert (
            isinstance(max_n_template, int) and max_n_template > 0
        ), f"max_n_template must be a positive integer, got {max_n_template}"
        assert (
            isinstance(min_template_length, int) and min_template_length >= 0
        ), f"min_template_length must be a non-negative integer, got {min_template_length}"

        self.n_template = max_n_template
        self.pick_top = pick_top
        self.min_seq_similarity = min_seq_similarity
        self.max_seq_similarity = max_seq_similarity
        self.min_template_length = min_template_length
        self.filter_by_query_length = filter_by_query_length
        self.template_lookup_path = template_lookup_path or os.environ.get("TEMPLATE_LOOKUP_PATH")
        self.template_base_dir = template_base_dir or os.environ.get("TEMPLATE_BASE_DIR")

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if "pdb_id" not in data:
            logger.warning("No PDB ID found in data. Skipping template addition.")
            data["template"] = {}
            return data

        pdb_id = data["pdb_id"]
        chain_info = data["chain_info"]

        # Load template information
        # NOTE: Currently templates only exist for proteins
        templates = {}
        for chain_id in chain_info:
            # get chain_type and convert to Enum
            chain_type = chain_info[chain_id]["chain_type"]
            chain_type = ChainType.as_enum(chain_type)

            rf_template = None
            if exists(self.template_lookup_path) and exists(self.template_base_dir):
                rf_template_id = _get_rf_template_id(pdb_id, chain_id, chain_type, self.template_lookup_path)
                rf_template = _load_rf_template(rf_template_id, self.template_base_dir)

            if rf_template is None:
                logger.debug(f"No RF template found for {pdb_id}_{chain_id}.")
                # early exit if no templates
                continue

            # NOTE: Could be made a lazy-load for each template only if it is selected
            #  if worker memory or speed becomes a bottleneck
            chain_templates = RF2AATemplate(**rf_template)
            is_valid = np.ones(chain_templates.n_templates, dtype=bool)

            # TODO: Revisit filtering logic once `cropping` is implemented to enable crop
            #  dependent filtering below (currently the below operates on the full query seq)
            if self.max_seq_similarity <= 100.0:
                # filter out templates with sequence similarity higher than cutoff
                is_valid &= chain_templates.seq_similarity_to_query <= self.max_seq_similarity

            if self.min_seq_similarity > 0.0:
                # filter out templates with sequence similarity lower than cutoff
                is_valid &= chain_templates.seq_similarity_to_query >= self.min_seq_similarity

            if self.min_template_length > 0:
                # filter out templates with fewer residues than cutoff
                is_valid &= chain_templates.n_res_per_template >= self.min_template_length

            # TODO: Possibly filter by deposition date. This will require a query to the PDB
            #  to get the deposition date of each template

            if not np.any(is_valid):
                # early exit if no valid templates after filter criteria
                continue

            # pick `n_template` (or fewer if fewer exist) valid templates
            valid_template_idxs = np.where(is_valid)[0]
            if not self.pick_top:
                valid_template_idxs = np.random.permutation(valid_template_idxs)

            # Add templates to template dict
            chain_templates = chain_templates.subset(valid_template_idxs[: self.n_template])
            templates[chain_id] = [
                {
                    "id": chain_templates.ids[i],
                    "pdb_id": chain_templates.pdb_ids[i],
                    "chain_id": chain_templates.chain_ids[i],
                    "template_lookup_id": chain_templates.lookup_id,
                    "seq_similarity": chain_templates.seq_similarity_to_query[i],
                    "atom_array": chain_templates.to_atom_array(i),
                    "n_res": chain_templates.n_res_per_template[i],
                }
                for i in range(chain_templates.n_templates)
            ]
            logger.debug(f"Added {len(templates[chain_id])} templates for chain {chain_id}: {chain_templates.ids}.")

        data["template"] = templates
        return data


class FeaturizeTemplatesLikeRF2AA(Transform):
    """
    A transform that featurizes RFTemplates templates for RF2AA.

    This class takes the templates added by the `AddRFTemplates` transform and featurizes them
    for use in the RF2AA model. The templates are added to the data under the key `template`.

    Attributes:
        - n_template (int): The number of templates to use.
        - mask_token_idx (int): The index of the mask token. Defaults to 21.
        - init_coords (torch.Tensor | float): The initial coordinates for the templates.
        - encoding (TokenEncoding): The encoding to use for the templates. Defaults to `RF2AA_ATOM36_ENCODING`.

    Methods:
        check_input(data: dict[str, Any]) -> None:
            Checks the input data for the required keys and types.

        forward(data: dict[str, Any]) -> dict[str, Any]:
            Featurizes the templates and adds them to the data.

    Raises:
        AssertionError: If `n_template` is not a positive integer.
        AssertionError: If `encoding` is not an instance of `TokenEncoding`.
        AssertionError: If `init_coords` is a tensor and its dimensions do not match the expected shape.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AddRFTemplates, AddWithinPolyResIdxAnnotation]

    def __init__(
        self,
        n_template: int,
        init_coords: torch.Tensor | float,
        mask_token_idx: int = 21,  # NOTE: This is the mask token `MSK` index in the original RF2AA code
        encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
        allowed_chain_types: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
    ):
        """
        Initializes the FeaturizeRFTemplatesForRF2AA transform.

        Args:
            - n_template (int): The number of templates to use. Must be a positive integer.
            - mask_token_idx (int, optional): The index of the mask token. Defaults to 21.
            - init_coords (torch.Tensor or float, optional): The initial coordinates for the templates.
                If a tensor, its dimensions must match the expected shape.
            - encoding (TokenEncoding, optional): The encoding to use for the templates.
                Must be an instance of `TokenEncoding`. Defaults to `RF2AA_ATOM36_ENCODING`.

        Raises:
            AssertionError: If `n_template` is not a positive integer.
            AssertionError: If `encoding` is not an instance of `TokenEncoding`.
            AssertionError: If `init_coords` is a tensor and its dimensions do not match the expected shape.
            AssertionError: If `allowed_chain_types` is not a list or contains any elements that are not instances of `ChainType`.
        """
        assert (
            isinstance(n_template, int) and n_template > 0
        ), f"n_template must be a positive integer, got {n_template}"
        assert isinstance(
            encoding, TokenEncoding
        ), f"encoding must be an instance of TokenEncoding, got {type(encoding)}"
        assert (
            isinstance(allowed_chain_types, list) and len(allowed_chain_types) > 0
        ), f"allowed_chain_types must be a non-empty list, got {allowed_chain_types}"
        assert np.isin(
            allowed_chain_types, ChainType
        ).all(), f"Allowed chain types must be a list of ChainType enums. Got {allowed_chain_types=}."
        self.n_template = n_template
        self.mask_token_idx = mask_token_idx
        self.init_coords = init_coords
        self.encoding = encoding
        self.allowed_chain_types = allowed_chain_types

        if isinstance(init_coords, torch.Tensor):
            n_dim = init_coords.shape[-1]
            assert n_dim == 3, f"init_coords must have 3 dimensions, got {n_dim}"

            if init_coords.ndim >= 2:
                n_token = init_coords.shape[-2]
                assert (
                    n_token == encoding.n_atoms_per_token
                ), f"init_coords must have {encoding.n_atoms_per_token} tokens, got {n_token}"

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["template", "atom_array"])
        check_is_instance(data, "template", dict)
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "within_poly_res_idx"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # Extract data
        atom_array = data["atom_array"]
        templates_by_chain = data["template"]

        # Initialize empty template features (= all padded) to fill later
        xyz, t1d, mask, _ = blank_rf2aa_template_features(
            n_template=self.n_template,
            n_token=get_token_count(atom_array),
            encoding=self.encoding,
            mask_token_idx=self.mask_token_idx,
            init_coords=self.init_coords,
        )

        # Get full atom array token starts (useful for going from atom-level > token-level annotations)
        _a_token_starts = get_token_starts(atom_array)  # [n_token] (int)

        # Fill the template features chain by chain and template by template ...
        for chain in chain_instance_iter(atom_array):
            # Check for allowable chain types
            if chain.chain_type[0] not in self.allowed_chain_types:
                # Only fill templates for proteins
                continue

            # Check for chains where templates exist
            chain_id = chain.chain_id[0]
            if chain_id not in templates_by_chain:
                # Early exit if there are no templates for this chain
                continue

            # Get chain token starts (useful for going from atom-level > token-level annotations)
            _c_token_starts = get_token_starts(chain)  # [n_token_in_chain] (int)
            # ... atomized tokens cannot be matched to templates
            if "atomize" in chain.get_annotation_categories():
                is_token_atomized = chain.atomize[_c_token_starts]  # [n_token_in_chain] (bool)
            else:
                is_token_atomized = np.zeros_like(_c_token_starts, dtype=bool)
            matchable_query_chain_tokens = _c_token_starts[~is_token_atomized]  # [n_matchable_token_in_chain] (int)

            # Featurize the templates and insert into the template features
            for tmpl_idx, tmpl_data in enumerate(templates_by_chain[chain_id]):
                template = tmpl_data["atom_array"]

                # Filter the template to only include tokens that are aligned to the query chain and that are not atomized
                # ... we use -1 as a placeholder query_res_idx for template tokens without alignment
                has_aligned_res_annotation = template.aligned_query_res_idx >= 0
                # ... find all template tokens that are aligned to the query chain
                has_match_in_query_chain = np.isin(
                    template.aligned_query_res_idx, chain.within_poly_res_idx[matchable_query_chain_tokens]
                )
                # ... check there is at least one template token that is aligned to the query chain
                if not np.any(has_match_in_query_chain & has_aligned_res_annotation):
                    # skip templates that do not have any aligned residues in the query
                    # (e.g. because query chain was cropped and crop does not overlap with template)
                    continue
                # ... subset the template to only the relevant tokens
                template = template[has_match_in_query_chain & has_aligned_res_annotation]

                # Get template token starts (useful for going from atom-level > token-level annotations)
                _t_token_starts = get_token_starts(template)

                # Annotate the global `token_id` for the template tokens which will be used to match
                #  the template tokens to the query chain to fill the template features
                template_token_id = select_data_by_id(
                    select_ids=template.aligned_query_res_idx[_t_token_starts],
                    data_ids=chain.within_poly_res_idx[matchable_query_chain_tokens],
                    data=chain.token_id[matchable_query_chain_tokens],
                    axis=0,
                )  # [n_token_in_template] (int)

                # Encode template
                template_encoded = atom_array_to_encoding(
                    template, self.encoding
                )  # [n_token_in_template, ...] (float/bool/int)

                # Match based on global token ids
                _is_matched_token = np.isin(atom_array.token_id[_a_token_starts], template_token_id)  # [n_token] (bool)
                token_ids_to_fill = atom_array.token_id[_a_token_starts][
                    _is_matched_token
                ]  # [n_matchable_token_in_template] (int)
                token_idxs_to_fill = np.where(_is_matched_token)[0]  # [n_matchable_token_in_template] (int)

                # Fill coordinates
                _tmpl_xyz = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["xyz"],
                    axis=0,
                )
                xyz[tmpl_idx, token_idxs_to_fill] = torch.tensor(_tmpl_xyz)

                # Fill mask
                _tmpl_mask = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["mask"],
                    axis=0,
                )
                mask[tmpl_idx, token_idxs_to_fill] = torch.tensor(_tmpl_mask)

                # Fill 1D template features
                _tmpl_seq = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["seq"],
                    axis=0,
                )
                _tmpl_confidence = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template.alignment_confidence[_t_token_starts],
                    axis=0,
                )
                # ... set one-hot encoded sequence for tokens where template features can be filled
                t1d[tmpl_idx, token_idxs_to_fill, :-1] = torch.nn.functional.one_hot(
                    torch.tensor(_tmpl_seq), self.encoding.n_tokens - 1
                ).float()
                # ... set confidence for tokens where template features can be filled
                #     for this we extract the residue-wise alignment confidence
                t1d[tmpl_idx, token_idxs_to_fill, -1] = torch.tensor(_tmpl_confidence)

        # Save the template features
        data["template_feat"] = {
            "xyz": xyz,  # [n_template, n_res, n_atoms_per_token, 3] (float)
            "mask": mask,  # [n_template, n_res, n_atoms_per_token] (bool)
            "t1d": t1d,  # [n_tepmlate, n_res, n_tokens],  [0:n_tokens-1] = one-hot encoded sequence, [-1] = confidence
        }
        return data


def blank_af3_template_features(n_templates: int, n_tokens: int, gap_token_index: int) -> dict[str, torch.Tensor]:
    """
    Generates blank template features for AF3.

    Args:
        - n_templates (int): Number of templates.
        - n_tokens (int): Number of tokens.
        - gap_token_index (int): Index of the gap token in the sequence encoding.

    Returns:
        dict: A dictionary containing initialized template features.
    """
    return {
        "template_restype": torch.full((n_templates, n_tokens), gap_token_index, dtype=int),
        "template_pseudo_beta_mask": torch.zeros((n_templates, n_tokens), dtype=bool),
        "template_backbone_frame_mask": torch.zeros((n_templates, n_tokens), dtype=bool),
        "template_distogram": torch.full((n_templates, n_tokens, n_tokens), fill_value=float("nan")),
        "template_unit_vector": torch.zeros((n_templates, n_tokens, n_tokens, 3)),
    }


def featurize_templates_like_af3(
    atom_array: AtomArray,
    templates_by_chain: dict[str, list[dict[str, Any]]],
    sequence_encoding: AF3SequenceEncoding,
    gap_token: str = "<G>",
    allowed_chain_type: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
    distogram_bins: torch.Tensor = torch.linspace(3.25, 50.75, 38),  # in Angstrom # noqa: B008
) -> dict[str, torch.Tensor]:
    """
    Generate AF3 template features for a given (cropped) atom array and the corresponding templates.

    NOTE: Number of templates (n_template) is determined by the number of templates in the templates_by_chain dict.

    This function adds the following features to the returned dictionary:
        - template_restype: [N_templ, N_token] One-hot encoding of the template sequence.
        - template_pseudo_beta_mask: [N_templ, N_token] Mask indicating if the CB (CA for glycine)
            has coordinates for the template at this residue.
        - template_backbone_frame_mask: [N_templ, N_token] Mask indicating if coordinates exist for
            all atoms required to compute the backbone frame (used in the template_unit_vector feature).
        - template_distogram: [N_templ, N_token, N_token, n_bins] A pairwise feature indicating the distance
            between Cβ atoms (CA for glycine). AF3 uses 38 bins between 3.25 Å and 50.75 Å with one extra
            bin for distances beyond 50.75 Å.
        - template_unit_vector: [N_templ, N_token, N_token, 3] The unit vector of the displacement
            of the CA atom of all residues within the local frame of each residue.

    Args:
        - atom_array (AtomArray): The input atom array.
        - templates_by_chain (dict): Dictionary of templates for each chain.
        - sequence_encoding (AF3SequenceEncoding): Encoding for the sequence.
        - gap_token (str): Token used for gaps in the sequence and as default to pad empty template tokens.
            NOTE: For templates a token is always a residue
        - allowed_chain_type (list): List of allowed chain types.
        - distogram_bins (torch.Tensor): Bins for discretizing distances in the distogram.

    Returns:
        dict: A dictionary containing the template features.

    References:
        `Section 2.8 of the AF3 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
        `AF2 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_

    NOTE: For templates a token is always a residue since we never align ligands, non-canonicals, PTMs, etc.
    """

    # Get the maximum number of templates for any chain, which will be the number of templates to fill
    n_templates = (
        max(len(templates_by_chain.get(chain_id, [])) for chain_id in templates_by_chain) if templates_by_chain else 0
    )
    n_templates = max(n_templates, 1)  # Ensure at least one template is filled (use a blank template if no templates)

    # Get full atom array token starts (useful for going from atom-level > token-level annotations)
    _a_token_starts = get_token_starts(atom_array)  # [n_token] (int)

    # Initialize features to fill
    _n_token = len(_a_token_starts)

    blank_af3_template = blank_af3_template_features(n_templates, _n_token, sequence_encoding.token_to_idx[gap_token])
    res_type = blank_af3_template["template_restype"]  # [n_templates, n_token] (int)
    template_pseudo_beta_mask = blank_af3_template["template_pseudo_beta_mask"]  # [n_templates, n_token] (bool)
    template_backbone_frame_mask = blank_af3_template["template_backbone_frame_mask"]  # [n_templates, n_token] (bool)
    template_distogram = blank_af3_template["template_distogram"]  # [n_templates, n_token, n_token] (float)
    template_unit_vector = blank_af3_template["template_unit_vector"]  # [n_templates, n_token, n_token, 3] (float)

    # Fill the template features chain by chain and template by template ...
    for chain in chain_instance_iter(atom_array):
        # Check for allowable chain types
        if chain.chain_type[0] not in allowed_chain_type:
            # Only fill templates for proteins
            continue

        # Check for chains where templates exist
        chain_id = chain.chain_id[0]
        if chain_id not in templates_by_chain:
            # Early exit if there are no templates for this chain
            continue

        # Get chain token starts (useful for going from atom-level > token-level annotations)
        _c_token_starts = get_token_starts(chain)  # [n_token_in_chain] (int)
        # ... atomized tokens cannot be matched to templates
        if "atomize" in chain.get_annotation_categories():
            is_token_atomized = chain.atomize[_c_token_starts]  # [n_token_in_chain] (bool)
        else:
            is_token_atomized = np.zeros_like(_c_token_starts, dtype=bool)
        matchable_query_chain_tokens = _c_token_starts[~is_token_atomized]  # [n_matchable_token_in_chain] (int)

        # Featurize the templates and insert into the template features
        for tmpl_idx, tmpl_data in enumerate(templates_by_chain[chain_id]):
            template = tmpl_data["atom_array"]

            # Filter the template to only include tokens that are aligned to the query chain and that are not atomized
            # ... we use -1 as a placeholder query_res_idx for template tokens without alignment
            has_aligned_res_annotation = template.aligned_query_res_idx >= 0
            # ... find all template tokens that are aligned to the query chain
            has_match_in_query_chain = np.isin(
                template.aligned_query_res_idx, chain.within_chain_res_idx[matchable_query_chain_tokens]
            )
            # ... check there is at least one template token that is aligned to the query chain
            if not np.any(has_match_in_query_chain & has_aligned_res_annotation):
                # skip templates that do not have any aligned residues in the query
                # (e.g. because query chain was cropped and crop does not overlap with template)
                continue
            # ... subset the template to only the relevant tokens
            template = template[has_match_in_query_chain & has_aligned_res_annotation]

            # Get template token starts (useful for going from atom-level > token-level annotations)
            _t_token_starts = get_token_starts(template)

            # Annotate the global `token_id` for the template tokens which will be used to match
            #  the template tokens to the query chain to fill the template features
            template_token_id = select_data_by_id(
                select_ids=template.aligned_query_res_idx[_t_token_starts],
                data_ids=chain.within_chain_res_idx[matchable_query_chain_tokens],
                data=chain.token_id[matchable_query_chain_tokens],
                axis=0,
            )  # [n_token_in_template] (int)
            # ... match based on global token ids
            _is_matched_token = np.isin(atom_array.token_id[_a_token_starts], template_token_id)  # [n_token] (bool)
            token_ids_to_fill = atom_array.token_id[_a_token_starts][
                _is_matched_token
            ]  # [n_matchable_token_in_template] (int)
            token_idxs_to_fill = np.where(_is_matched_token)[0]  # [n_matchable_token_in_template] (int)

            # ... fill the res_type
            res_type[tmpl_idx, token_idxs_to_fill] = torch.as_tensor(
                sequence_encoding.encode(struc.get_residues(template)[1])
            )

            # ...fill the template_pseudo_beta_mask
            #   get information on whether the (pseudo) CB is resolved
            _is_cb = template.atom_name == "CB"
            _is_glycine_ca = (template.atom_name == "CA") & (template.res_name == "GLY")
            _is_pseudo_cb_resolved = (_is_cb | _is_glycine_ca) & (template.occupancy > 0)
            # ... spread it accross the token axis
            _has_pseudo_cb = struc.apply_residue_wise(template, data=_is_pseudo_cb_resolved, function=np.any)
            template_pseudo_beta_mask[tmpl_idx, token_idxs_to_fill] = torch.as_tensor(_has_pseudo_cb)

            # ... fill the template_backbone_frame_mask
            _is_n_ca_c_resolved = (
                (template.atom_name == "CA")
                | (template.atom_name == "N")
                | (template.atom_name == "C") & (template.occupancy > 0)
            )
            _has_n_ca_c_resolved = struc.apply_residue_wise(template, data=(_is_n_ca_c_resolved), function=np.sum) == 3
            template_backbone_frame_mask[tmpl_idx, token_idxs_to_fill] = torch.as_tensor(_has_n_ca_c_resolved)

            # ... fill the template_distogram
            template_coords = torch.tensor(template.coord)
            ix1, ix2 = np.ix_(token_ids_to_fill[_has_pseudo_cb], token_ids_to_fill[_has_pseudo_cb])
            template_distogram[tmpl_idx, ix1.astype(int), ix2.astype(int)] = torch.cdist(
                template_coords[_is_pseudo_cb_resolved],
                template_coords[_is_pseudo_cb_resolved],
                compute_mode="donot_use_mm_for_euclid_dist",
            )

            # ... fill the template_unit_vector

            residues_with_resolved_n_ca_c = struc.spread_residue_wise(template, _has_n_ca_c_resolved)
            template_frames = rigid_from_3_points(
                x1=template_coords[(template.atom_name == "N") & (residues_with_resolved_n_ca_c)],
                x2=template_coords[(template.atom_name == "CA") & (residues_with_resolved_n_ca_c)],
                x3=template_coords[(template.atom_name == "C") & (residues_with_resolved_n_ca_c)],
            )  # (n_template_res, 3, 3), (n_template_res, 3)
            # ... get CA coords in the respective frames
            ca_coords_in_frames = apply_inverse_rigid(
                rigid=(template_frames[0][:, None, :, :], template_frames[1][:, None, :]),
                points=template_coords[(template.atom_name == "CA") & (residues_with_resolved_n_ca_c)],
            )  # (n_template_res, n_template_res, 3)
            ca_direction_in_frames = normalize(ca_coords_in_frames, dim=-1, eps=1e-3)
            # ... reset diagonal to 0 (can be non-zero due to normalization & numerical error)
            ca_direction_in_frames[0, 0] = 0.0

            ix1, ix2 = np.ix_(token_ids_to_fill[_has_n_ca_c_resolved], token_ids_to_fill[_has_n_ca_c_resolved])
            template_unit_vector[tmpl_idx, ix1.astype(int), ix2.astype(int)] = ca_direction_in_frames

    # ... bucketize the distogram
    template_distogram = torch.bucketize(
        template_distogram,
        boundaries=torch.as_tensor(distogram_bins, dtype=template_distogram.dtype, device=template_distogram.device),
    )
    n_bins = len(distogram_bins) + 1
    template_distogram = torch.nn.functional.one_hot(template_distogram, num_classes=n_bins).to(
        torch.float32
    )  # We don't need int64 precision

    return {
        "template_restype": res_type,
        "template_pseudo_beta_mask": template_pseudo_beta_mask,
        "template_backbone_frame_mask": template_backbone_frame_mask,
        "template_distogram": template_distogram,
        "template_unit_vector": template_unit_vector,
    }


class FeaturizeTemplatesLikeAF3(Transform):
    """
    A transform that featurizes templates for AlphaFold 3.

    This transform generates the following template features (as torch.Tensors):
        - template_restype: [N_templ, N_token] Residue type for each template token.
        - template_pseudo_beta_mask: [N_templ, N_token] Mask indicating if pseudo-beta atom exists.
        - template_backbone_frame_mask: [N_templ, N_token] Mask indicating if coordinates exist for
            all atoms required to compute the backbone frame.
        - template_distogram: [N_templ, N_token, N_token] A pairwise feature indicating the distance
            between Cβ atoms (CA for glycine), discretized into bins.
        - template_unit_vector: [N_templ, N_token, N_token, 3] The unit vector of the displacement
            of the CA atom of all residues within the local frame of each residue.

    References:
        `Section 2.8 of the AF3 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
        `AF2 supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AddRFTemplates|AddInputFileTemplate",
        "AddWithinChainInstanceResIdx",
        "AddGlobalTokenIdAnnotation",
    ]

    def __init__(
        self,
        sequence_encoding: AF3SequenceEncoding,
        gap_token: str = "<G>",
        allowed_chain_type: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
        distogram_bins: torch.Tensor = torch.linspace(3.25, 50.75, 38),  # noqa: B008
    ):
        self.gap_token = gap_token
        self.allowed_chain_type = allowed_chain_type
        self.distogram_bins = distogram_bins
        self.sequence_encoding = sequence_encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "template"])
        check_is_instance(data, "atom_array", AtomArray)
        check_is_instance(data, "template", dict)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        templates_by_chain = data["template"]

        template_features = featurize_templates_like_af3(
            atom_array=atom_array,
            templates_by_chain=templates_by_chain,
            sequence_encoding=self.sequence_encoding,
            gap_token=self.gap_token,
            allowed_chain_type=self.allowed_chain_type,
            distogram_bins=self.distogram_bins,
        )

        # Add the template features to the `feats` dict
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(template_features)

        return data


def random_subsample_templates(
    template_dictionary: dict[str, list[dict[str, Any]]], n_template: int = 4
) -> dict[str, list[dict[str, Any]]]:
    """
    Subsample the templates for each chain in the template dictionary. We support the "training" implementation with this function;
    for inference, do not use this function (and instead e.g. set `max_n_template=4` to directly take the first 4 templates).

    From the AF-3 supplement:
        > "Templates are sorted by e-value. At most 20 templates can be returned by our search, and the model uses up to 4
            (Ntempl ≤ 4). At inference time we take the first 4. At training time we choose k random templates out of the available
            n, where k ~ min(Uniform[0, n], 4). This reduces the efficacy of simply copying the template.
    """
    for chain_id, templates in template_dictionary.items():
        # ...at training time we choose k random templates out of the available n, where k ~ min(Uniform[0, n], 4)
        n_available_templates = len(templates)
        n_templates_to_sample = min(np.random.randint(0, n_available_templates + 1), n_template)

        # ...choose k random templates, if k < n
        if n_templates_to_sample < n_available_templates:
            sampled_templates = np.random.choice(templates, n_templates_to_sample, replace=False).tolist()
            template_dictionary[chain_id] = sampled_templates

    return template_dictionary


class RandomSubsampleTemplates(Transform):
    """Subsample the templates for each chain in the template dictionary.

    Args:
        n_template (int): The maximum possible number of templates to use. Default is 4.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        FeaturizeTemplatesLikeAF3,
        FeaturizeTemplatesLikeRF2AA,
        "OneHotTemplateRestype",
    ]

    def __init__(self, n_template: int = 4):
        self.n_template = n_template

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["template"])
        check_is_instance(data, "template", dict)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["template"] = random_subsample_templates(template_dictionary=data["template"], n_template=self.n_template)

        return data


class OneHotTemplateRestype(Transform):
    """
    One-hot encode residue types within templates.
    NOTE: We keep as a separate transform since the AF-3 supplement did not
    explicitly mention the one-hot encoding of the residue types for templates.
    """

    def __init__(self, encoding: AF3SequenceEncoding):
        self.encoding = encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["feats"])
        check_is_instance(data, "feats", dict)

        check_contains_keys(data["feats"], ["template_restype"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        template_restype = data["feats"]["template_restype"]

        # One-hot encode the template restype
        template_restype_onehot = torch.nn.functional.one_hot(
            template_restype, num_classes=self.encoding.n_tokens
        ).float()

        # Add the one-hot encoded template restype to the `feats` dict
        data["feats"]["template_restype"] = template_restype_onehot

        return data


def add_input_file_template(
    atom_array: AtomArray,
) -> dict[str, list[dict[str, Any]]]:
    template = defaultdict(list)
    for chain in chain_instance_iter(atom_array):
        # Check for allowable chain types
        if chain.chain_type[0] not in [ChainType.POLYPEPTIDE_L, ChainType.RNA]:
            # Only fill templates for proteins
            continue

        # Check for chains where templates exist
        if np.sum(chain.is_input_file_templated) == 0:
            # Early exit if there are no templates for this chain
            logger.debug(f"No templates for chain {chain.chain_id[0]}.")
            continue

        chain_id = chain.chain_id[0]
        template_chain = atom_array[atom_array.is_input_file_templated]
        # add extra template annotations to the template
        # aligned_query_res_idx, alignment_confidence
        template_chain.set_annotation("aligned_query_res_idx", template_chain.res_id)
        template_chain.set_annotation("alignment_confidence", np.ones(len(template_chain), dtype=float))
        template[chain_id].append(
            {
                "id": None,
                "pdb_id": None,
                "chain_id": None,
                "template_lookup_id": None,
                "seq_similarity": 100.0,
                "atom_array": template_chain,
                "n_res": len(np.unique(template_chain.res_id)),
            }
        )
    return template


class AddInputFileTemplate(Transform):
    """
    If atoms from the input file have been marked as templates, add them to the template dictionary.
    This is useful for when users want to use a part of their design as a template using
    the template_selection_syntax argument in the inference script.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["is_input_file_templated"])
        return super().check_input(data)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        template = add_input_file_template(atom_array)
        # Add the templates to the data
        if "template" in data:
            raise ValueError("Template already exists in data. Cannot add input file template.")
        data["template"] = template
        logger.info("Templating from input file.")
        return data
