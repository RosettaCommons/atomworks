"""Transforms to handle the assignment of RF2AA's atom frames"""

from typing import Any, ClassVar

import networkx as nx
import numpy as np
import torch
from biotite.structure import AtomArray

from atomworks.ml.encoding_definitions import TokenEncoding
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import ComputeAtomToTokenMap
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import EncodeAtomArray
from atomworks.ml.transforms.filters import RemoveNucleicAcidTerminalOxygen, RemoveTerminalOxygen
from atomworks.ml.utils.token import get_token_starts

# Constants copied from `chemdata` to decouple the RF2AA repository from the atomworks.ml pipeline
NUM2AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "MAS",
    " DA",
    " DC",
    " DG",
    " DT",
    " DX",
    " RA",
    " RC",
    " RG",
    " RU",
    " RX",
    "HIS_D",  # only used for cart_bonded
    "Al",
    "As",
    "Au",
    "B",
    "Be",
    "Br",
    "C",
    "Ca",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe",
    "Hg",
    "I",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Ni",
    "O",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Sn",
    "Tb",
    "Te",
    "U",
    "W",
    "V",
    "Y",
    "Zn",
    "ATM",
]

FRAME_PRIORITY_TO_ATOM = [
    "F",
    "Cl",
    "Br",
    "I",
    "O",
    "S",
    "Se",
    "Te",
    "N",
    "P",
    "As",
    "Sb",
    "C",
    "Si",
    "Sn",
    "Pb",
    "B",
    "Al",
    "Zn",
    "Hg",
    "Cu",
    "Au",
    "Ni",
    "Pd",
    "Pt",
    "Co",
    "Rh",
    "Ir",
    "Pr",
    "Fe",
    "Ru",
    "Os",
    "Mn",
    "Re",
    "Cr",
    "Mo",
    "W",
    "V",
    "U",
    "Tb",
    "Y",
    "Be",
    "Mg",
    "Ca",
    "Li",
    "K",
    "ATM",
]
ATOM_TO_FRAME_PRIORITY = {x: i for i, x in enumerate(FRAME_PRIORITY_TO_ATOM)}


def find_all_paths_of_length_n(
    graph: nx.Graph, n: int, order_independent_atom_frame_prioritization: bool = True
) -> list:
    """Find all paths of a given length n in a NetworkX graph.

    Args:
        graph: The input graph.
        n: The length of the paths to find.
        order_independent_atom_frame_prioritization: If True, considers paths with the same nodes but in different orders as equivalent.
            Defaults to True.

    Returns:
        A tensor containing all unique paths of length n.

    Reference:
        `StackOverflow: Finding all paths of given length <https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph>`_
    """

    def find_paths(graph: nx.Graph, u: Any, n: int) -> list[list[Any]]:
        """Find all paths of length n starting from node u in graph G."""
        if n == 0:
            return [[u]]
        paths = [
            [u, *path]
            for neighbor in graph.neighbors(u)
            for path in find_paths(graph, neighbor, n - 1)
            if u not in path
        ]
        return paths

    # All paths of length n
    if order_independent_atom_frame_prioritization:
        # Reverse paths if the first node is greater than the last node (which we later deduplicate with a set)
        allpaths = [
            tuple(p) if p[0] < p[-1] else tuple(reversed(p)) for node in graph for p in find_paths(graph, node, n)
        ]
    else:
        # If order_independent_frame_prioritization is False, do not reverse paths
        allpaths = [tuple(p) for node in graph for p in find_paths(graph, node, n)]

    # Ensure paths are unique
    allpaths = list(set(allpaths))

    return allpaths


def get_rf2aa_atom_frames(
    encoded_query_pn_unit: np.ndarray, graph: nx.Graph, order_independent_atom_frame_prioritization: bool = True
) -> torch.Tensor:
    """
    Choose a frame of 3 bonded atoms for each atom in the molecule,
    using a rule-based system that prioritizes frames based on atom types.

    Parameters:
        encoded_query_pn_unit (torch.Tensor): Sequence of the pn_unit that we want to build frames for,
            encoded using the RF2AA TokenEncoding.
        G (nx.Graph): The input graph representing the non-polymer molecule.
        order_independent_frame_prioritization (bool, optional):
            If True, sorts atom types within frames to consider them order-independent.
            Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the selected frames for each atom.
    """

    frames = find_all_paths_of_length_n(graph, 2, order_independent_atom_frame_prioritization)
    selected_frames = []

    for n in range(encoded_query_pn_unit.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]

        # Some chemical groups don't have two bonded heavy atoms; so, choose a frame with an atom two bonds away
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]

        # If the atom isn't in a three-atom frame, it should be ignored in loss calculation; set all the atoms to n
        if not frames_with_n:
            selected_frames.append([(0, 1), (0, 1), (0, 1)])
            continue

        frame_priorities = []
        for frame in frames_with_n:
            # HACK: Uses the "query_seq" to convert index of the atom into an "atom type", and converts that into a priority
            indices = [index for index in frame if index != n]
            aas = [NUM2AA[int(encoded_query_pn_unit[index])] for index in indices]

            #
            if order_independent_atom_frame_prioritization:
                frame_priorities.append(sorted([ATOM_TO_FRAME_PRIORITY[aa] for aa in aas]))
            else:
                frame_priorities.append([ATOM_TO_FRAME_PRIORITY[aa] for aa in aas])

        # NOTE: np.argsort doesn't sort tuples correctly so just sort a list of indices using a key
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])

        # Calculate residue offset for frame
        frame = [(frame - n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)

    assert encoded_query_pn_unit.shape[0] == len(selected_frames)
    return torch.tensor(selected_frames).long()


class AddAtomFrames(Transform):
    """
    Add atom frames to the data dictionary. See the RF2AA supplement for more details.

    NOTE: We do not assume that all atomized residues are at the end of the AtomArray to allow for more flexibility in the future.

    Parameters:
        order_independent_atom_frame_prioritization (bool, optional):
            If True, sorts atom types within frames to consider them order-independent.
            Defaults to True.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, EncodeAtomArray]

    def __init__(self, order_independent_atom_frame_prioritization: bool = True):
        self.order_independent_atom_frame_prioritization = order_independent_atom_frame_prioritization

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["encoded", "atom_array"])
        check_is_instance(data, "atom_array", AtomArray)  # TODO: Add other checks
        check_atom_array_annotation(data, ["pn_unit_iid"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        token_wise_atom_array = atom_array[token_starts]

        # Initialize the atom frames
        seq = data["encoded"]["seq"]
        atom_frames = torch.zeros((seq.shape[0], 3, 2), dtype=torch.int64)  # [n_tokens_across_chains, 3, 2] (int)

        # Loop through all atomized pn_units_iids
        pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.atomize])
        for pn_unit_iid in pn_unit_iids:
            token_level_pn_unit_mask = (token_wise_atom_array.pn_unit_iid == pn_unit_iid) & (
                token_wise_atom_array.atomize
            )

            # Generate the networkx graph for the pn_unit
            pn_unit_instance_bonds = token_wise_atom_array.bonds[token_level_pn_unit_mask]
            graph = pn_unit_instance_bonds.as_graph()

            # Get the frames
            pn_unit_instance_atom_frames = get_rf2aa_atom_frames(
                seq[token_level_pn_unit_mask], graph, self.order_independent_atom_frame_prioritization
            )

            # Fill in the atom frames
            atom_frames[token_level_pn_unit_mask] = pn_unit_instance_atom_frames

        data["rf2aa_atom_frames"] = atom_frames

        return data

    # TODO: Tests for `AddAtomFrames`


class AddIsRealAtom(Transform):
    """
    Makes a faux version of is_real_atom that we previously derived from the ChemData heavy atom mask. Determines how many atoms are in each residue based on the atom array to
    accomodate terminal oxygens etc...
    This mask is used in the pLDDT calculation, where it is used to mask pLDDT logits in the [B,I,Max_N_Atoms] representation.
    Uses the atom_to_token_map to determine the number of atoms in each residue, outputting a boolean mask in [I,36] format. This can accomodate
    up to 36 atoms per residue, as the RF2aa is_real_atom object has 36 atoms per residue. In AF3, the maximum number of atoms per residue
    is 23, and this tensor is truncated in the pLDDT calculation.

    Adds:
        - 'is_real_atom': torch.Tensor of shape [I, 36] (bool)
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        ComputeAtomToTokenMap,
        RemoveTerminalOxygen,
        RemoveNucleicAcidTerminalOxygen,
    ]

    def __init__(self, token_encoding: TokenEncoding):
        self.max_n_atoms = token_encoding.n_atoms_per_token

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        tok_idx = data["feats"]["atom_to_token_map"]

        is_real_atom = torch.zeros(tok_idx.max() + 1, self.max_n_atoms, dtype=torch.bool)
        for i in range(is_real_atom.shape[0]):
            is_real_atom[i, : torch.sum(tok_idx == i)] = True

        data["is_real_atom"] = is_real_atom

        return data


class AddPolymerFrameIndices(Transform):
    """
    Adds indices for the atoms that will constitute the backbone frames for non-ligands.
    Adds an I,3 tensor, where the first index is the atom index of the N, the second is
    the atom index of the CA, and the third is the atom index of the C for protein.
    Follows the AF3 pattern for nucleic acids. For ligands and noncanonicals (ie anything
    atomized), this functions adds the index of each atom to the CA position.

    Adds:
        - 'frame_idxs': torch.Tensor of shape [I, 3] (long)
    """

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["feats", "atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # construct the masks
        atom_array = data["atom_array"]
        is_protein = data["feats"]["is_protein"]
        is_nucleic_acid = data["feats"]["is_rna"] | data["feats"]["is_dna"]

        # problem is that noncanonical proteins and nas are marked as protein/nucleic acid, not as ligand, so instead we use the atomize mask
        is_ligand = atom_array.atomize[get_token_starts(atom_array)]

        token_len = max(atom_array.token_id) + 1

        frame_idxs = np.zeros((token_len, 3), dtype=np.int64)

        nitrogen_atoms = (
            is_protein[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "N")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        c_alpha_atoms = (
            is_protein[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "CA")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        c_atoms = (
            is_protein[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "C")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        o_four_prime_atoms = (
            is_nucleic_acid[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "O4'")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        c_one_prime_atoms = (
            is_nucleic_acid[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "C1'")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        c_two_prime_atoms = (
            is_nucleic_acid[atom_array.token_id.astype(np.int32)]
            & (atom_array.atom_name == "C2'")
            & ~is_ligand[atom_array.token_id.astype(np.int32)]
        )
        ligand_atoms = is_ligand[atom_array.token_id.astype(np.int32)]

        frame_idxs[~is_ligand, 0] = np.where(nitrogen_atoms | o_four_prime_atoms)[0]
        frame_idxs[:, 1] = np.where(c_alpha_atoms | c_one_prime_atoms | ligand_atoms)[0]
        frame_idxs[~is_ligand, 2] = np.where(c_atoms | c_two_prime_atoms)[0]

        frame_idxs = torch.from_numpy(frame_idxs)

        data["pae_frame_idx_token_lvl_from_atom_lvl"] = frame_idxs

        return data
