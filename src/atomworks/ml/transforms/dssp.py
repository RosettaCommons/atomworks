"""Secondary structure annotation using DSSP."""

import logging
from enum import IntEnum, StrEnum

import biotite.application.dssp as dssp
import numpy as np
from biotite.structure import AtomArray

from atomworks.enums import ChainType
from atomworks.ml.executables.dssp import DSSPExecutable
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts, spread_token_wise

logger = logging.getLogger("atomworks.ml")


class SSEnum(IntEnum):
    """Secondary structure enum for protein residues.

    Groups DSSP codes into five categories for efficient storage and manipulation.
    Used for both DSSP output annotations and secondary structure conditioning.

    Values:
      NONE: Not set/not conditioned (value: 0)
      HELIX: Alpha helix (H, G, I in DSSP) (value: 1)
      SHEET: Beta sheet (E, B in DSSP) (value: 2)
      LOOP: Coil/loop/turn (T, S, C, P in DSSP) (value: 3)
      NON_PROTEIN: Non-protein chain or DSSP failed (value: 4)
    """

    NONE = 0
    HELIX = 1
    SHEET = 2
    LOOP = 3
    NON_PROTEIN = 4

    @classmethod
    def names(cls) -> list[str]:
        """Return human-readable names for each group."""
        return ["none", "helix", "sheet", "loop", "non_protein"]

    @classmethod
    def to_string(cls, value: int) -> str:
        """Convert integer value to human-readable string."""
        # Handle NONE specially since it's 0
        if value == 0:
            return "none"
        return cls.names()[value]


class DSSPCode(StrEnum):
    """DSSP secondary structure codes as defined by the DSSP program."""

    ALPHA_HELIX = "H"  # alpha-helix
    ISOLATED_BETA_BRIDGE = "B"  # residue in isolated beta-bridge
    EXTENDED_STRAND = "E"  # extended strand, participates in beta ladder
    THREE_TEN_HELIX = "G"  # 3-10 helix
    PI_HELIX = "I"  # pi-helix
    POLYPROLINE_HELIX = "P"  # kappa-helix (poly-proline II helix)
    HYDROGEN_BONDED_TURN = "T"  # hydrogen-bonded turn
    BEND = "S"  # bend
    OTHER = "C"  # loop, coil, or irregular
    NON_PROTEIN = "!"  # non-protein

    @classmethod
    def valid_codes(cls) -> set[str]:
        """Return set of valid DSSP codes."""
        return {e.value for e in cls}

    @classmethod
    def to_group_index(cls, code: str) -> int:
        """Map DSSP code to SSEnum index."""
        if code in {cls.ALPHA_HELIX.value, cls.THREE_TEN_HELIX.value, cls.PI_HELIX.value}:
            return SSEnum.HELIX
        if code in {cls.EXTENDED_STRAND.value, cls.ISOLATED_BETA_BRIDGE.value}:
            return SSEnum.SHEET
        if code == cls.NON_PROTEIN.value:
            return SSEnum.NON_PROTEIN
        return SSEnum.LOOP


def _get_chain_sse_and_valid(chain_atom_array: AtomArray, bin_path: str) -> tuple[np.ndarray, bool]:
    """Run DSSP on a chain's protein atoms, return group indices and whether DSSP ran successfully.

    Args:
      chain_atom_array: AtomArray containing atoms from a single chain.
      bin_path: Path to DSSP executable.

    Returns:
      Tuple of (group_indices, is_valid) where group_indices are integers from
      SSEnum and is_valid indicates if DSSP ran successfully.
    """
    try:
        dssp_codes = dssp.DsspApp.annotate_sse(chain_atom_array, bin_path=bin_path)
        # Convert DSSP codes to group indices
        group_indices = np.array([DSSPCode.to_group_index(code) for code in dssp_codes], dtype=np.int8)
        return group_indices, True
    except Exception as e:
        chain_id = getattr(chain_atom_array, "chain_id", ["?"])[0]
        logger.error(
            f"Error running DSSP for entity {chain_id}: {e}; "
            f"using NON_PROTEIN code for this entity's residues, and setting is_valid annotation to False"
        )
        return (
            np.full(len(chain_atom_array), SSEnum.NON_PROTEIN, dtype=np.int8),
            False,
        )


def annotate_secondary_structure(
    atom_array: AtomArray,
    bin_path: str | None = None,
    annotation_name: str = "dssp_sse",
    is_valid_annotation_name: str | None = None,
) -> AtomArray:
    """Annotate secondary structure for each residue using DSSP.

    Only protein tokens are assigned secondary structure groups; all others are
    set to NON_PROTEIN.

    Also adds a boolean annotation indicating whether the SSE is valid (not default
    NON_PROTEIN due to error).

    Args:
      atom_array: AtomArray to annotate.
      bin_path: Path to DSSP executable. If ``None``, uses executable from ``DSSPExecutable``.
      annotation_name: Name for the SSE annotation. Defaults to ``"dssp_sse"``.
      is_valid_annotation_name: Name for the validity annotation. If ``None``,
        uses ``"{annotation_name}_is_valid"``. Defaults to ``None``.

    Returns:
      AtomArray with secondary structure annotations added.
    """
    # Get bin_path from executable manager if not provided
    if bin_path is None:
        dssp_exec = DSSPExecutable.get_or_initialize()
        bin_path = dssp_exec.get_bin_path()

    # Atom-level masks
    is_protein_atom_lvl = np.isin(atom_array.chain_type, ChainType.get_proteins())
    is_atomized_atom_lvl = (
        atom_array.atomize
        if "atomize" in atom_array.get_annotation_categories()
        else np.zeros(atom_array.array_length(), dtype=bool)
    )

    # Token-level masks
    token_starts = get_token_starts(atom_array)
    atom_array_token_lvl = atom_array[token_starts]

    # Default all tokens to NON_PROTEIN and all is_valid to False
    sse = np.full(len(atom_array_token_lvl), SSEnum.NON_PROTEIN, dtype=np.int8)
    is_valid = np.zeros(len(atom_array_token_lvl), dtype=bool)

    if np.any(is_protein_atom_lvl):
        # Loop over chain instances
        for chain_iid in np.unique(atom_array.chain_iid):
            chain_iid_mask = atom_array.chain_iid == chain_iid
            chain_iid_protein_mask = chain_iid_mask & is_protein_atom_lvl & ~is_atomized_atom_lvl

            if not np.any(chain_iid_protein_mask):
                # Early exit if this chain has no protein atoms
                continue

            # Get chain atoms and compute SSE
            chain_atom_array = atom_array[chain_iid_protein_mask]
            sse_chain, is_valid_chain = _get_chain_sse_and_valid(chain_atom_array, bin_path)

            # Assign to all tokens in this chain instance
            token_mask = chain_iid_protein_mask[token_starts]
            if len(sse_chain) == token_mask.sum():
                sse[token_mask] = sse_chain
                is_valid[token_mask] = is_valid_chain
            else:
                # Catch-all for situations that arise (usually due to cropping)
                logger.warning(
                    f"Mismatch in SSE length for chain {chain_iid}: {len(sse_chain)} != {token_mask.sum()}. "
                    f"We will use NON_PROTEIN for this chain, and set is_valid to False."
                )

    # Spread token-wise to all atoms and set annotations
    sse_spread = spread_token_wise(atom_array, sse)
    is_valid_spread = spread_token_wise(atom_array, is_valid)

    # Use provided is_valid annotation name or default
    if is_valid_annotation_name is None:
        is_valid_annotation_name = f"{annotation_name}_is_valid"

    atom_array.set_annotation(annotation_name, sse_spread)
    atom_array.set_annotation(is_valid_annotation_name, is_valid_spread)

    return atom_array


class AnnotateSecondaryStructure(Transform):
    """Annotate secondary structure for each residue using DSSP.

    Adds integer annotations from :py:class:`SSEnum` indicating the secondary structure type.

    Args:
      bin_path: Path to DSSP executable. If ``None``, uses ``DSSP`` environment variable. Defaults to ``None``.
      annotation_name: Name for the SSE annotation. Defaults to ``"dssp_sse"``.
      is_valid_annotation_name: Name for the validity annotation. If ``None``,
        uses ``"{annotation_name}_is_valid"``. Defaults to ``None``.
      max_n_tokens: Maximum number of tokens to run DSSP on. If structure exceeds this,
        DSSP is skipped and no annotations are added. Defaults to 800, which encompasses most proteins.
    """

    def __init__(
        self,
        bin_path: str | None = None,
        annotation_name: str = "dssp_sse",
        is_valid_annotation_name: str | None = None,
        max_n_tokens: int | None = 800,
    ):
        # Initialize executable if not already done
        if bin_path is None:
            DSSPExecutable.get_or_initialize()
        else:
            DSSPExecutable.get_or_initialize(bin_path)
        self.annotation_name = annotation_name
        self.is_valid_annotation_name = is_valid_annotation_name
        self.max_n_tokens = max_n_tokens

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["chain_type", "chain_iid"])

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        # Check if structure exceeds max_n_tokens
        if self.max_n_tokens is not None:
            token_starts = get_token_starts(atom_array)
            n_tokens = len(token_starts)

            if n_tokens > self.max_n_tokens:
                # Skip DSSP and return data without annotations
                logger.info(f"Skipping DSSP: structure has {n_tokens} tokens, exceeds max_n_tokens={self.max_n_tokens}")
                return data

        # Proceed with normal DSSP annotation
        data["atom_array"] = annotate_secondary_structure(
            atom_array,
            bin_path=None,  # Use executable manager
            annotation_name=self.annotation_name,
            is_valid_annotation_name=self.is_valid_annotation_name,
        )
        return data
