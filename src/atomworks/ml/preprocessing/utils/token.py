"""AF3-style token counting utilities.

NOTE: Due to changing token definitions and/or stochasticity like random atomization of sidechains,
the actual number of sampled tokens may differ and this number should be viewed as approximate.
"""

from biotite.structure import AtomArray

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.filters import (
    RemoveHydrogens,
)
from atomworks.ml.utils.token import get_token_starts


def count_af3_style_tokens(atom_array: AtomArray) -> dict[str, int]:
    """Count AF3-style tokens in an AtomArray.

    Tokens are defined as:
        - Polymer tokens (residue-level): Standard amino acids and nucleotides
        - Non-polymer tokens (atom-level): Ligands and non-canonical amino acids

    Args:
        atom_array: The AtomArray to process and count tokens in.

    Returns:
        Dictionary with n_atomized_tokens, n_non_atomized_tokens, n_tokens_total.
    """
    # Build preprocessing pipeline to match AF3 tokenization (as an upper-bound estimate)
    transforms = [
        RemoveHydrogens(),
        AnnotateCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=list(STANDARD_AA) + list(STANDARD_RNA) + list(STANDARD_DNA),
        ),
    ]

    pipeline = Compose(transforms)
    atom_array = pipeline({"atom_array": atom_array})["atom_array"]

    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]

    n_atomized = len(token_level_array[token_level_array.atomize])
    n_non_atomized = len(token_level_array[~token_level_array.atomize])

    return {
        "n_atomized_tokens": n_atomized,
        "n_non_atomized_tokens": n_non_atomized,
        "n_tokens_total": n_atomized + n_non_atomized,
    }
