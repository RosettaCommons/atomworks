"""Transforms for augmentation of nucleic acids"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Iterable
from functools import partial
from os import PathLike
from typing import Any, ClassVar

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.basepairs import _check_dssr_criteria, _get_proximate_residues
from biotite.structure.filter import filter_nucleotides
from biotite.structure.residues import get_residue_masks, get_residue_starts_for

from atomworks.constants import STANDARD_DNA
from atomworks.io.transforms.atom_array import remove_nan_coords
from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.selection import ResIdxSlice
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from atomworks.ml.executables.x3dna import X3DNAFiber
from atomworks.ml.preprocessing.constants import ChainType
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import _renumber_res_ids_around_reference
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.geometry import align_atom_arrays
from atomworks.ml.utils.misc import randomly_select_items_with_weights
from atomworks.ml.utils.testing import is_clash

logger = logging.getLogger("atomworks.ml")

dna_transform_dir = os.path.abspath(os.path.dirname(__file__))

_WATSON_CRICK_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
"""Watson-Crick complement of standard nucleotides."""

_WATSON_CRICK_COMPLEMENT_TRANSLATION_TABLE = str.maketrans(_WATSON_CRICK_COMPLEMENT)
"""Translation table for the Watson-Crick complement of nucleotides."""


def to_reverse_complement(seq: str) -> str:
    """
    Get a Watson-Crick complement of a nucleic acid sequence (assuming one-letter codes).
    """
    return seq.upper()[::-1].translate(_WATSON_CRICK_COMPLEMENT_TRANSLATION_TABLE)


def generate_bform_dna(seq: str) -> AtomArray:
    """
    Uses x3dna's 'fiber' executable to generate ideal bform DNA
    with the given sequence, then returns the structure parsed into
    an AtomArray.
    """
    seq = seq.upper()
    assert len(seq) > 0, "Sequence must be non-empty"
    assert set(seq).issubset({"A", "T", "C", "G"}), f"Sequence must contain only valid DNA nucleotides: {seq=}"

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp_file:
        tmp_pdb_path = pathlib.Path(tmp_file.name)

        x3dna_cmd = [str(X3DNAFiber.get_bin_path()), "-b", f"-seq={seq}", str(tmp_pdb_path)]
        result = subprocess.run(x3dna_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0 or not tmp_pdb_path.exists() or tmp_pdb_path.stat().st_size == 0:
            raise RuntimeError(
                "Failed to generate B-form DNA using X3DNA 'fiber'. Ensure X3DNA is installed and PATH set."
            )

        # load the generated structure and remove any potential nans
        atom_array = load_any(tmp_pdb_path, file_type="pdb", model=1)
        atom_array = remove_nan_coords(atom_array)

        assert len(atom_array) > 0, "Generated structure must contain atoms"

    return atom_array


def _get_residue_info_for_dna_chains(atom_array: AtomArray, annotations: Iterable[str]) -> np.ndarray:
    """
    Get residue information for DNA chains in an AtomArray.
    """
    assert np.all(atom_array.chain_type == ChainType.DNA), "AtomArray must contain DNA chains"
    to_canonical_dna_1_letter = np.vectorize(
        partial(get_1_from_3_letter_code, use_closest_canonical=True, chain_type=ChainType.DNA)
    )
    _res_start_stop_idxs = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts = _res_start_stop_idxs[:-1]
    dtypes = [("is_nan", "bool"), ("canonical_seq", "U1")]
    for annotation in annotations:
        dtypes.append((annotation, atom_array.get_annotation(annotation).dtype))

    seq_array = np.zeros(len(_res_start_stop_idxs) - 1, dtype=np.dtype(dtypes))
    seq_array["is_nan"] = struc.segments.apply_segment_wise(
        _res_start_stop_idxs, np.isnan(atom_array.coord).any(axis=-1), function=np.all
    )
    seq_array["canonical_seq"] = to_canonical_dna_1_letter(atom_array.res_name[_res_starts])
    for annotation in annotations:
        seq_array[annotation] = atom_array.get_annotation(annotation)[_res_starts]

    return seq_array


def _is_true_contiguous(arr: np.ndarray) -> bool:
    """Returns True if the True values in arr form a single contiguous block."""
    true_idx = np.flatnonzero(arr)

    if len(true_idx) == 0:
        return False  # No islands if no True values

    # True island is contiguous if the difference between the last and first index
    # is exactly the number of True values minus one
    return (true_idx[-1] - true_idx[0] + 1) == len(true_idx)


def _get_overhang_lengths(is_overhang: np.ndarray) -> np.ndarray:
    """Given a boolean array indicating overhangs, returns the lengths of the overhangs on the left and right."""
    left = 0 if not is_overhang[0] else np.argmax(~is_overhang)
    right = 0 if not is_overhang[-1] else np.argmax(~is_overhang[::-1])
    return np.array([left, right])


# reimplementation of biotite base_pairs
def base_pairs(
    atom_array: AtomArray, min_atoms_per_base: int = 3, unique: bool = True, no_hbond_dist_cut: float = 4.0
) -> np.ndarray:
    # Get the nucleotides for the given atom_array
    nucleotides_boolean = filter_nucleotides(atom_array)

    # Disregard the phosphate-backbone
    non_phosphate_boolean = ~np.isin(atom_array.atom_name, ["O5'", "P", "OP1", "OP2", "OP3", "HOP2", "HOP3"])

    # Combine the two boolean masks
    boolean_mask = nucleotides_boolean & non_phosphate_boolean

    # Get only nucleosides
    nucleosides = atom_array[boolean_mask]

    # Get the base pair candidates according to a N/O cutoff distance,
    # where each base is identified as the first index of its respective
    # residue
    n_o_mask = np.isin(nucleosides.element, ["N", "O"])
    basepair_candidates, n_o_matches = _get_proximate_residues(nucleosides, n_o_mask, no_hbond_dist_cut)

    # Contains the plausible base pairs
    basepairs = []
    # Contains the number of hydrogens for each plausible base pair
    basepairs_hbonds = []

    # Get the residue masks for each residue
    base_masks = get_residue_masks(nucleosides, basepair_candidates.flatten())

    # Group every two masks together for easy iteration (each 'row' is
    # respective to a row in ``basepair_candidates``)
    base_masks = base_masks.reshape((basepair_candidates.shape[0], 2, nucleosides.shape[0]))

    for (base1_index, base2_index), (base1_mask, base2_mask), n_o_pairs in zip(
        basepair_candidates, base_masks, n_o_matches, strict=False
    ):
        base1 = nucleosides[base1_mask]
        base2 = nucleosides[base2_mask]

        hbonds = _check_dssr_criteria((base1, base2), min_atoms_per_base, unique)

        # If no hydrogens are present use the number N/O pairs to
        # decide between multiple pairing possibilities.

        if hbonds is None:
            # Each N/O-pair is detected twice. Thus, the number of
            # matches must be divided by two.
            hbonds = n_o_pairs / 2
        if hbonds != -1:
            basepairs.append((base1_index, base2_index))
            if unique:
                basepairs_hbonds.append(hbonds)

    basepair_array = np.array(basepairs)

    if unique:
        # Contains all non-unique base pairs that are flagged to be
        # removed
        to_remove = []

        # Get all bases that have non-unique pairing interactions
        base_indices, occurrences = np.unique(basepairs, return_counts=True)
        for base_index, occurrence in zip(base_indices, occurrences, strict=False):
            if occurrence > 1:
                # Write the non-unique base pairs to a dictionary as
                # 'index: number of hydrogen bonds'
                remove_candidates = {}
                for i, row in enumerate(np.asarray(basepair_array == base_index)):
                    if np.any(row):
                        remove_candidates[i] = basepairs_hbonds[i]
                # Flag all non-unique base pairs for removal except the
                # one that has the most hydrogen bonds
                del remove_candidates[max(remove_candidates, key=remove_candidates.get)]
                to_remove += list(remove_candidates.keys())
        # Remove all flagged base pairs from the output `ndarray`
        basepair_array = np.delete(basepair_array, to_remove, axis=0)

    # Remap values to original atom array
    if len(basepair_array) > 0:
        basepair_array = np.where(boolean_mask)[0][basepair_array]
        for i, row in enumerate(basepair_array):
            basepair_array[i] = get_residue_starts_for(atom_array, row)
    return basepair_array


class PadDNA(Transform):
    """
    Structurally pads DNA duplexes by extending them with randomly sampled DNA in B-form conformation.

    This transform identifies DNA duplexes in the structure, completes any overhanging single-stranded regions with
    complementary bases, and optionally extends the duplex with additional base pairs. The padding is done both at the
    sequence level and structural level, ensuring proper base pairing and B-form DNA geometry. The original sequence
    is not modified and placed at a random position in the padded sequence.

    Args:
        - x3dna_path (PathLike | None): Path to the X3DNA installation directory or executable. If None, this
            assumes the 'X3DNA' environment variable is set to infer the x3dna executable path.
        - p_skip (float): Probability of skipping the transform. Must be between 0 and 1. Defaults to 0.
        - max_overhang (int): Maximum allowed length of single-stranded overhangs. Defaults to 2.
            If the overhang is longer than this, the transform will skip the DNA chain.
        - max_pad (int): Maximum number of base pairs to add in a single padding event. Defaults to 100.
            If the total length of the padded sequence is longer than this, the transform will skip the DNA chain.
        - max_pad_tot (int): Maximum total length of padded DNA duplex. Defaults to 100.
        - min_pad (int): Minimum number of base pairs to add when padding. Defaults to 20.
        - pad_type_weights (dict): Weights for different padding strategies. Keys are 'none', 'pdb', 'uniform'.
            Defaults to {"none": 0, "pdb": 0, "uniform": 1}.
        - pad_nt_weights (dict): Weights for nucleotide selection during padding. Keys are 'A', 'T', 'C', 'G'.
            Defaults to {"A": 1, "C": 1, "G": 1, "T": 1}.
        - align_len_weights (dict): Weights for selecting alignment lengths. Keys are integers. Defaults to {1: 1}.

    Raises:
        - AssertionError: If p_skip is not between 0 and 1.
        - X3DNAExecutableError: If X3DNA executable validation fails.
    """

    # fmt: off
    pdb_dna_lengths: ClassVar[dict[int, int]] = dict(enumerate([
        #1    #2    #3    #4    #5    #6    #7    #8    #9    #10
        0,    5,    53,   60,   342,  616,  764,  817,  797,  675, # 10
        1258, 971,  1713, 985,  795,  669,  1224, 464,  697,  363, # 20
        435,  858,  336,  218,  313,  280,  317,  351,  318,  161, # 30
        217,  154,  218,  99,   109,  231,  155,  84,   157,  98,  # 40
        320,  46,   285,  45,   84,   93,   61,   69,   267,  124, # 50
        276,  40,   52,   46,   94,   41,   58,   35,   29,   31,  # 60
        90,   39,   17,   38,   30,   18,   15,   13,   14,   8,   # 70
        61,   22,   18,   4,    10,   14,   3,    9,    13,   18,  # 80
        26,   14,   0,    5,    32,   45,   3,    1,    4,    1,   # 90
        28,   2,    3,    3,    6,    3,    10,   0,    2,    35,  # 100
        15,   0,    0,    0,    1,    11,   42,   0,    4,    2,   # 110
        0,    0,    1,    0,    1,    2,    4,    0,    5,    2,   # 120
        8,    2,    4,    5,    2,    4,    0,    3,    1,    0,   # 130
        0,    0,    0,    4,    0,    0,    4,    2,    2,    12,  # 140
        0,    2,    0,    4,    9,    203,  154,  317,  3,    670  # 150
    ]))
    """Distribution of DNA lengths in PDB as tuples of (length, count)."""
    # fmt: on

    def __init__(
        self,
        x3dna_path: PathLike | None = None,
        p_skip: float = 0,
        max_overhang: int = 2,
        max_pad: int = 100,
        max_pad_tot: int = 100,
        min_pad: int = 0,
        pad_type_weights: dict = {"none": 0, "pdb": 0, "uniform": 1},
        pad_nt_weights: dict = {"A": 1, "C": 1, "G": 1, "T": 1},
        align_len_weights: dict = {1: 1},
        no_hbond_dist_cut: float = 4.0,
    ):
        """
        Args:
            x3dna_path (str) : Path to the x3dna directory. For example,
                 "path/to/prot_dna/x3dna-v2.4/"
            p_skip (float, from 0 to 1): probability that this transform is skipped and does nothing.
            max_overhang (int, positive): maximum number of overhanging bases on either end of the DNA duplex
                allowed. If the input's DNA has a larger overhang, this transform will do nothing.
                Example: if max_overhang=2 and the DNA is ATT / AATCGCGCGCGC, the transform will do nothing.
            max_pad (int, >= 0): maximum number of additional basepairs to add to the DNA duplex.
                Must be larger than min_pad. If set to 0, this transform will change the structure of
                the duplex DNA's termini but not add any new basepairs.
            max_pad_tot (int, > 0): maximum total basepairs the DNA duplex can have after padding.
                If the input structure already exceeds this value, this transform will do nothing.
            min_pad (int, >= 0): minimum number of additional basepairs to add to the DNA duplex.
            pad_type_weights (dict[Literal("none","pdb","uniform"):int]):
                Weights for different padding distributions. Probability for each will be weight / sum(weights)
                "none" is no padding, same as max_pad=0.
                "pdb" follows the distribution of DNA sequence lengths in the RCSB PDB.
                "uniform" is uniformly distributed between min_pad and max_pad.
            pad_nt_weights (dict[Literal("A","C","G","T"):int]): Relative weights for nucleotides
                to be used in random sequence selection for padding. Default is uniform.
                Example: {"A":1, "C":5, "G":5, "T":1} would generate G/C-rich padding.
            align_len_weights (dict[int:int]): Weights for align_len.
                align_len is the number of basepairs used for aligning generated DNA to the original DNA.
                Higher values will make the overall DNA chain smoother, with worse local accuracy.
                Low values (1 or 2 especially) will favor local accuracy, with any bends in the
                original DNA very obvious in the overall chain. A value of 1 matches AF3 behavior.

        """
        assert 0 <= p_skip <= 1, "p_skip must be between 0 and 1"

        # Configure X3DNA executable
        if p_skip < 1:
            # ... set up x3dna executable manager
            X3DNAFiber.get_or_initialize(x3dna_path)
        else:
            logger.warning("p_skip is 1, skipping x3dna executable validation. This transform will have no effect.")

        self.p_skip = p_skip
        self.max_overhang = max_overhang
        self.max_pad = max_pad
        self.max_pad_tot = max_pad_tot
        self.min_pad = min_pad
        self.pad_type_weights = pad_type_weights
        self.pad_nt_weights = pad_nt_weights
        self.align_len_weights = align_len_weights
        self.no_hbond_dist_cut = no_hbond_dist_cut

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "chain_iid"])

    def _copy_annotations(
        self, target_array: AtomArray, source_array: AtomArray, keys: [list, str] = "all"
    ) -> AtomArray:
        """
        Copies chain-level annotations from a source array to a target one.
        """
        if keys == "all":
            # default behavior: copy all keys that are the same across the whole structure in the source
            keys = source_array.get_annotation_categories()
            keys = [x for x in keys if len(set(source_array.get_annotation(x))) == 1]

        l_target = len(target_array)

        for key in keys:
            value = source_array.get_annotation(key)[0]
            annot_array = [value for _ in range(l_target)]
            target_array.set_annotation(key, annot_array)

        return target_array

    def _get_random_padded_dna_sequence(self, n: int, seq: str) -> tuple[str, int]:
        """
        Adds random sequence padding to a DNA sequence string.
        """
        new_seq = randomly_select_items_with_weights(self.pad_nt_weights, n=n)

        # place original sequence at a random position in the new sequence
        seq_idx = np.random.randint(0, len(new_seq) - len(seq) + 1)
        new_seq = np.concatenate((new_seq[:seq_idx], np.array(list(seq)), new_seq[seq_idx + len(seq) :]))
        new_seq = "".join(new_seq)
        return new_seq, seq_idx

    def _pad_dna_seq(self, completed_duplex_seq: str, original_seq: str) -> tuple[str, int]:
        """
        Extends a DNA sequence by adding random nucleotides whilst ensuring the original
        sequence is contained exactly once as a contiguous substring.

        Args:
            - completed_duplex_seq (str): DNA sequence of one of the duplex strands
                after overhang completion.
            - original_seq (str): Original DNA sequence of the same strand before
                overhang completion.

        Returns:
            - tuple[str, int]: Tuple containing:
                - Padded DNA sequence.
                - Index where original sequence starts in padded sequence, or -1 if padding fails.

        Notes:
            The padding length is chosen based on pad_type_weights:
            - 'pdb': Samples from distribution of DNA lengths in PDB
            - 'uniform': Samples uniformly between current length + min_pad and current length + max_pad
            - 'none': No padding
        """
        # fail if duplex sequence is too short to reasonably pad without introducing copies
        if len(completed_duplex_seq) < 4:
            logger.warning(
                "PadDNA failed. Sequences shorter than 4 residues cannot be padded without introducing copies."
            )
            return completed_duplex_seq, -1

        # randomly choose style of padding
        pad_options = list(self.pad_type_weights.keys())
        pad_weights = [self.pad_type_weights[x] for x in pad_options]
        pad_weights = np.array(pad_weights) / sum(pad_weights)
        pad_choice = np.random.choice(pad_options, 1, p=pad_weights)[0]

        if pad_choice == "pdb":
            # final length is sampled from distribution of DNA chains in pdb, excluding those smaller than the input
            pad_dict = self.pdb_dna_lengths
        elif pad_choice == "uniform":
            # final length is sampled uniformly between starting length and configurable maximum (default: 100)
            if len(completed_duplex_seq) > self.max_pad_tot:
                # no padding if sequence is already too long
                return completed_duplex_seq, -1
            min_len = len(completed_duplex_seq) + self.min_pad
            max_len = len(completed_duplex_seq) + self.max_pad
            max_len = min(max_len, self.max_pad_tot)
            pad_dict = {i: 1 for i in range(min_len, max_len + 1)}
        else:
            # otherwise, no padding
            return completed_duplex_seq, -1

        # sample what the final length will be
        new_len = randomly_select_items_with_weights(pad_dict)

        # generate random DNA sequence of chosen final length
        # can use a configurable distribution for sequence generation (default is uniform ACTG)
        for _ in range(100):
            padded_seq, seq_insertion_idx = self._get_random_padded_dna_sequence(new_len, completed_duplex_seq)

            # retry if the original sequence appears more than once
            if padded_seq.count(original_seq) != 1:
                continue

            # retry if the completed duplex sequence does not appear exactly once
            if padded_seq.count(completed_duplex_seq) != 1:
                continue

            return padded_seq, seq_insertion_idx

        logger.warning("PadDNA failed. PadDNA failed to generate an acceptable padded sequence in _pad_dna_seq()")
        return completed_duplex_seq, -1

    def _pad_dna_structure(
        self, atom_array: AtomArray, dna_chain_ids: Iterable, overhangs: list, new_seq: str, new_seq_idx: int
    ) -> AtomArray | None:
        """
        Generates a new structure with padded DNA duplex by combining original and ideal B-form DNA coordinates.

        Args:
            - atom_array (AtomArray): Input structure containing DNA duplex.
            - dna_chain_ids (Iterable): Chain identifiers for the DNA duplex.
            - overhangs (list): List of overhang sequences.
            - new_seq (str): Target padded sequence.
            - new_seq_idx (int): Index where original sequence starts in padded sequence.

        Returns:
            - AtomArray | None: New structure with padded DNA duplex, or None if padding fails due to:
                - Non-canonical nucleotides at padding junction
                - Poor alignment between original and ideal DNA (RMSD > 5Å)
                - Clashes between new DNA coordinates and original non-DNA atoms

        Notes:
            The padding process:
            1. Removes overhanging nucleotides plus one base pair
            2. Generates ideal B-form DNA for complete sequence
            3. Aligns ideal DNA to original structure at junction points
            4. Combines aligned ideal DNA with original structure
        """
        # first, split atom_array into DNA and non-DNA parts
        is_dna_to_pad = (atom_array.chain_iid == dna_chain_ids[0]) | (atom_array.chain_iid == dna_chain_ids[1])
        atom_array_non_dna = atom_array[~is_dna_to_pad].copy()
        dna_array = atom_array[is_dna_to_pad].copy()

        # remove the overhanging DNA NTs, plus one additional NT on each terminus
        rm_a_begin, rm_b_end, rm_a_end, rm_b_begin = [len(x) + 1 for x in overhangs]

        array_a = dna_array[dna_array.chain_iid == dna_chain_ids[0]]
        array_a = array_a[ResIdxSlice(rm_a_begin, -1 * rm_a_end)]
        array_a_nan_free = remove_nan_coords(array_a)

        array_b = dna_array[dna_array.chain_iid == dna_chain_ids[1]]
        array_b = array_b[ResIdxSlice(rm_b_begin, -1 * rm_b_end)]
        array_b_nan_free = remove_nan_coords(array_b)

        try:
            assert struc.get_residue_count(array_a) == struc.get_residue_count(array_b)
        except AssertionError:
            logger.warning("PadDNA failed. PadDNA found mismatch between first and second DNA chains.")
            return None

        # next, generate ideal B-form DNA with the desired final sequence using x3dna
        try:
            array_ideal = generate_bform_dna(new_seq)
        except AssertionError:
            logger.warning("PadDNA failed. Attempted to generate invalid DNA sequence with x3dna.")
            return None

        # correct or add in annotations like chain_id, pn_unit_id, etc. for the generated AtomArray
        array_ideal_a = array_ideal[ResIdxSlice(None, len(new_seq))]
        array_ideal_b = array_ideal[ResIdxSlice(-1 * len(new_seq), None)]
        array_ideal_a = self._copy_annotations(target_array=array_ideal_a, source_array=array_a)
        array_ideal_b = self._copy_annotations(target_array=array_ideal_b, source_array=array_b)
        array_ideal = array_ideal_a + array_ideal_b

        # prepare for alignment
        # Collect short-named variables needed for indexing sections to align
        n = struc.get_residue_count(array_a_nan_free)
        a = randomly_select_items_with_weights(self.align_len_weights, 1)
        a = min(a, n)  # can't align more residues than there are in the structure
        o = len(new_seq)
        r, u = rm_a_begin, rm_b_end
        i = new_seq_idx + (r + u - 1)
        j = o - (i + n)

        # for alignment, target is array_trim and mobile is array_ideal
        # left-side alignment: (target) first `a` NTs of chain A and last `a` of chain B
        #                      (mobile) corresponding section of ideal, padded structure
        # right-side alignment:(target) last `a` NTs of chain A and first `a` of chain B
        #                      (mobile) corresponding section of ideal, padded structure

        tgt_array_left = array_a_nan_free[ResIdxSlice(None, a)] + array_b_nan_free[ResIdxSlice(-1 * a, None)]

        tgt_array_right = array_a_nan_free[ResIdxSlice(-1 * a, None)] + array_b_nan_free[ResIdxSlice(None, a)]

        mbl_array_left = array_ideal[ResIdxSlice(i, i + a)] + array_ideal[ResIdxSlice(-1 * (i + a), -1 * i)]

        mbl_array_right = array_ideal[ResIdxSlice(o - (j + a), o - j)] + array_ideal[ResIdxSlice(o + j, o + (j + a))]

        res_names = set()
        for array in (tgt_array_left, tgt_array_right, mbl_array_left, mbl_array_right):
            res_names.update(array.res_name)
        noncanonical_res_names = res_names - set(STANDARD_DNA)
        if noncanonical_res_names:
            logger.warning("PadDNA failed. PadDNA found a noncanonical nucleotide at the padding junction.")
            return None

        # apply alignments to generated ideal DNA
        left_aligned_ideal, left_rmsd = align_atom_arrays(mbl_array_left, tgt_array_left, array_ideal)
        right_aligned_ideal, right_rmsd = align_atom_arrays(mbl_array_right, tgt_array_right, array_ideal)

        if left_rmsd > 5 or right_rmsd > 5:
            logger.warning("PadDNA found that the junction basepairs do not align well with idealized DNA.")
            return None

        # select components for the final hybrid structure
        component_first = left_aligned_ideal[ResIdxSlice(None, i)]
        component_second = array_a
        component_third = right_aligned_ideal[ResIdxSlice(o - j, o)]
        component_fourth = right_aligned_ideal[ResIdxSlice(o, o + j)]
        component_fifth = array_b
        component_sixth = left_aligned_ideal[ResIdxSlice(-1 * i, None)]

        # check for clash between newly generated DNA coords and the original non-DNA coords
        all_new_array = component_first + component_third + component_fourth + component_sixth
        if is_clash(all_new_array, atom_array_non_dna):
            logger.warning(
                "PadDNA failed. PadDNA found a clash between newly generated DNA coords and the original non-DNA coords."
            )
            return None

        # renumber residues in idealized components to fit around the original numbering
        component_first = _renumber_res_ids_around_reference(component_first, ref=component_second, where="before")
        component_third = _renumber_res_ids_around_reference(component_third, ref=component_second, where="after")
        component_fourth = _renumber_res_ids_around_reference(component_fourth, ref=component_fifth, where="before")
        component_sixth = _renumber_res_ids_around_reference(component_sixth, ref=component_fifth, where="after")

        dna_chain_first = component_first + component_second + component_third
        dna_chain_second = component_fourth + component_fifth + component_sixth

        # finally, if chains were symmetric, padding breaks symmetry so we need to relabel
        if dna_chain_first.molecule_entity[0] == dna_chain_second.molecule_entity[0]:
            next_id = np.max((atom_array_non_dna + dna_chain_first).molecule_entity) + 1
            dna_chain_second.molecule_entity = np.full_like(dna_chain_second.molecule_entity, next_id)

        return atom_array_non_dna + dna_chain_first + dna_chain_second

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if np.random.rand() < self.p_skip:
            return data

        atom_array = data["atom_array"]
        is_dna = atom_array.chain_type == ChainType.DNA

        if not is_dna.any():
            # ... early stop if no DNA is present
            return data

        # Find base-pairs & unpaired overhangs / bases
        # ... filter to only DNA chains
        dna_array = atom_array[is_dna]

        # ... annotate each base with a unique ID for tracking (i.e. idx in the 'dna_array')
        _res_start_stop_idxs = struc.get_residue_starts(dna_array, add_exclusive_stop=True)
        base_ids = np.arange(len(_res_start_stop_idxs) - 1)
        dna_array.set_annotation("base_id", struc.segments.spread_segment_wise(_res_start_stop_idxs, base_ids))

        # ... compute all base-pairs based on the structure
        dna_array_no_nan = remove_nan_coords(dna_array)
        _base_pair_idxs = base_pairs(dna_array_no_nan, no_hbond_dist_cut=self.no_hbond_dist_cut)

        if not len(_base_pair_idxs):
            # ... early stop if no base-pairs are found
            logger.warning("PadDNA found no base pairs in the input structure.")
            return data

        base_pair_ids = dna_array_no_nan.get_annotation("base_id")[_base_pair_idxs]  # (n_base_pairs, 2)
        chain_pair_iids = np.unique(dna_array_no_nan.chain_iid[_base_pair_idxs], axis=0)  # (n_chain_pairs, 2)

        # ... for each base & chain, annotate whether it is paired. Also annotate overhanging bases.
        dna_array.set_annotation("is_base_paired", np.isin(dna_array.base_id, base_pair_ids))
        dna_array.set_annotation("is_chain_paired", np.isin(dna_array.chain_iid, chain_pair_iids))
        dna_array.set_annotation("is_overhang", dna_array.is_chain_paired & ~dna_array.is_base_paired)

        # ... get number of partner-chains for each chain
        n_partner_chains = dict(zip(*np.unique(chain_pair_iids, return_counts=True), strict=False))

        # ... filter to 'extensible' chains, i.e. chains which are uniquely paired with only one other chain
        chain_pair_iids_in_duplex = [
            pair for pair in chain_pair_iids if n_partner_chains[pair[0]] == 1 and n_partner_chains[pair[1]] == 1
        ]

        # early stop if no extensible duplexes are present
        if len(chain_pair_iids_in_duplex) == 0:
            return data

        new_atom_array = atom_array.copy()
        annotations = ("base_id", "is_base_paired", "is_chain_paired", "is_overhang")
        for chain_iids in chain_pair_iids_in_duplex:
            chain1_iid, chain2_iid = chain_iids
            try:
                chain1 = _get_residue_info_for_dna_chains(dna_array[dna_array.chain_iid == chain1_iid], annotations)
                chain2 = _get_residue_info_for_dna_chains(dna_array[dna_array.chain_iid == chain2_iid], annotations)
            except AssertionError:
                logger.warning(
                    f"In PadDNA, _get_residue_info_for_dna_chains() failed for duplex {chain1_iid}:{chain2_iid}. Skipping."
                )
                continue

            ### TODO: Move into function
            # ... skip if desired duplex is already of the desired length
            if chain1["is_base_paired"].sum() > self.max_pad_tot:
                continue

            # ... skip if there is an unpaired stretch in the middle of the duplex
            if not _is_true_contiguous(chain1["is_base_paired"]) or not _is_true_contiguous(chain2["is_base_paired"]):
                # ... unpaired stretches in the middle of the duplex
                logger.warning(
                    f"DNA duplex {chain1_iid}:{chain2_iid} has unpaired stretches in the middle. Skipping in PadDNA."
                )
                continue

            # ... skip if overhang is too long
            chain1_overhang = _get_overhang_lengths(chain1["is_overhang"])
            chain2_overhang = _get_overhang_lengths(chain2["is_overhang"])
            if np.any(chain1_overhang > self.max_overhang) or np.any(chain2_overhang > self.max_overhang):
                logger.warning(
                    f"DNA duplex {chain1_iid}:{chain2_iid} has overhangs longer than {self.max_overhang}. Skipping in PadDNA."
                )
                continue

            # ... build the completed duplex sequence (i.e. insert reverse-complements for each overhang)
            # The following are paired:
            #  chain 1 (fwd)      seq1_lhs - paired - seq1_rhs
            #  chain 2 (fwd)      seq2_lhs - paired - seq2_rhs
            # e.g.
            #  chain 1 (fwd):              - CAGGT  - CT
            #  chain 2 (fwd):              - ACCTG  - G
            #  chain 2 (rev):            G - GTCCA  -
            #
            # then the completed sequence for chain1 is:
            #  completed chain 1 (fwd):  C - CAGGT  - CT
            #
            # NOTE: seq1_lhs & seq2_rhs as well as seq1_rhs & seq2_lhs are mutually exclusive.
            # skip if either pair is present

            seq1_lhs = "".join(chain1["canonical_seq"][: chain1_overhang[0]])
            seq1_paired = "".join(chain1["canonical_seq"][chain1["is_base_paired"]])
            if chain1_overhang[1] == 0:  # -0 index does not behave as expected
                seq1_rhs = ""
            else:
                seq1_rhs = "".join(chain1["canonical_seq"][-chain1_overhang[1] :])
            seq2_lhs = "".join(chain2["canonical_seq"][: chain2_overhang[0]])
            seq2_paired = "".join(chain2["canonical_seq"][chain2["is_base_paired"]])
            if chain2_overhang[1] == 0:  # -0 index does not behave as expected
                seq2_rhs = ""
            else:
                seq2_rhs = "".join(chain2["canonical_seq"][-chain2_overhang[1] :])

            try:
                assert seq1_paired == to_reverse_complement(
                    seq2_paired
                ), "sequences to be joined must be reverse-complements of each other"
                assert seq1_lhs == "" or seq2_rhs == "", "overhang1_lhs and overhang1_rhs are mutually exclusive"
                assert seq1_rhs == "" or seq2_lhs == "", "overhang2_lhs and overhang2_rhs are mutually exclusive"
            except AssertionError:
                logger.warning(
                    "DNA duplex {chain1_iid}:{chain2_iid} has mutually exclusive overhangs. Skipping in PadDNA."
                )
                continue

            # NOTE: at least one of the first two elements & at least one of the last two elements are
            #  guaranteed to be empty strings due to the above assertions, so we can safely sum the lists
            overhangs = [seq1_lhs, to_reverse_complement(seq2_rhs), seq1_rhs, to_reverse_complement(seq2_lhs)]
            completed_seq1 = "".join(overhangs[:2] + [seq1_paired] + overhangs[2:])
            full_seq1 = "".join(chain1["canonical_seq"])

            padded_seq, seq_insertion_idx = self._pad_dna_seq(completed_seq1, full_seq1)
            if seq_insertion_idx == -1:
                continue

            tmp_atom_array = self._pad_dna_structure(
                new_atom_array.copy(), chain_iids, overhangs, padded_seq, seq_insertion_idx
            )
            if tmp_atom_array is None:  # _pad_dna_structure failed
                continue
            new_atom_array = tmp_atom_array

        data["atom_array"] = new_atom_array
        return data
