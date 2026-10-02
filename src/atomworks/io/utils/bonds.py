"""
Utility functions for the detection, and creation, of bonds in a structure.
"""

import hashlib
import logging
from typing import Any, Literal

import biotite.structure as struc
import networkx as nx
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, BondType
from biotite.structure.io import pdbx

from atomworks.common import sum_string_arrays
from atomworks.constants import (
    BOND_DISTANCE_THRESHOLD_CHNO,
    BOND_DISTANCE_THRESHOLD_CHNOPS,
    BOND_DISTANCE_THRESHOLD_OTHER,
    CCD_MIRROR_PATH,
    STANDARD_AND_UNKNOWN_POLYMER_RESIDUES,
    STRUCT_CONN_BOND_ORDER_TO_INT,
    STRUCT_CONN_BOND_TYPES,
)
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.atom_array import (
    _bonds_to_dict,
)
from atomworks.io.utils.ccd import (
    atom_array_from_ccd_code,
)
from atomworks.io.utils.link_chemistry import (
    _element_distance_thresholds,
    filter_link_distances,
    infer_link_order,
)
from atomworks.io.utils.link_chemistry import (
    correct_charged_amide_nitrogens as correct_charged_amide_nitrogens,
)
from atomworks.io.utils.link_chemistry import (
    correct_formal_charges_for_specified_atoms as correct_formal_charges_for_specified_atoms,
)
from atomworks.io.utils.selection import get_annotation, get_residue_starts

logger = logging.getLogger("atomworks.io")

LongBondPolicyType = Literal[
    "keep",
    "filter",
    "warn",
    "raise",
    "filter_nonstandard_only",
    "warn_nonstandard_only",
    "raise_nonstandard_only",
]
"""Type alias for long bond filtering policy options."""


def remap_intra_residue_coordination_bonds(structure: AtomArray) -> AtomArray:
    """Remap intra-residue COORDINATION bonds to SINGLE.

    Biotite's CIF writer routes intra-residue bonds to ``chem_comp_bond``, which
    cannot represent COORDINATION; inter-residue bonds go to ``struct_conn`` (which can).
    """
    if structure.bonds is None:
        return structure

    bond_array = structure.bonds.as_array()
    is_coord = bond_array[:, 2] == BondType.COORDINATION
    if not np.any(is_coord):
        return structure

    res_starts = get_residue_starts(structure)
    atom1_res = np.searchsorted(res_starts, bond_array[:, 0], side="right") - 1
    atom2_res = np.searchsorted(res_starts, bond_array[:, 1], side="right") - 1
    remap_mask = is_coord & (atom1_res == atom2_res)
    if not np.any(remap_mask):
        return structure

    structure = structure.copy()
    new_bonds = structure.bonds.as_array()
    new_bonds[remap_mask, 2] = BondType.SINGLE
    structure.bonds = struc.BondList(structure.array_length(), new_bonds)

    return structure


def build_bond_dict_for_atom_array(
    atom_array: AtomArray,
    custom_bond_dict: dict[str, dict[tuple[str, str], int]] | None = None,
    ccd_mirror_path: str | None = CCD_MIRROR_PATH,
) -> dict[str, dict[tuple[str, str], int]] | None:
    """Build a complete bond dictionary for an AtomArray by merging CCD bonds with custom overrides.

    Fetches CCD bonds for **all** residues as the baseline, then overlays ``custom_bond_dict``
    on top. This ensures that atoms added later (e.g., H atoms via ``add_missing_atoms``) always
    have bonds available from CCD, while custom connectivity still takes precedence for the pairs
    it explicitly defines.

    Args:
        atom_array: Structure containing residues to get bonds for.
        custom_bond_dict: Optional custom bonds (e.g., from CIF ``chem_comp_bond``).
            Maps residue names to ``{(atom1_name, atom2_name): bond_type_int}``.
            These override the CCD entries for matching atom pairs.
        ccd_mirror_path: Path to local CCD mirror. Defaults to ``CCD_MIRROR_PATH``.

    Returns:
        Complete bond dictionary with CCD bonds as baseline and custom bonds as overrides,
        or ``None`` if no bonds could be found for any residue.

    Example:
        >>> # Custom bonds for VER override CCD heavy-atom pairs; CCD still provides H bonds
        >>> custom_bonds = {"VER": {("C1", "C2"): 1, ("C2", "O1"): 2}}
        >>> bond_dict = build_bond_dict_for_atom_array(atom_array, custom_bond_dict=custom_bonds, ccd_mirror_path="/path/to/ccd")
        >>> # bond_dict contains VER bonds (CCD H-bonds + custom heavy-atom overrides) + ALA, GLY, etc. (CCD)
    """
    # Get unique residue names from atom_array
    unique_res_names = np.unique(atom_array.res_name)

    # Fetch CCD bonds for ALL residues...
    result_dict: dict[str, dict[tuple[str, str], int]] = {}
    for res_name in unique_res_names:
        try:
            template = atom_array_from_ccd_code(res_name, ccd_mirror_path, coords=None)
            bond_dict = _bonds_to_dict(template)
            if bond_dict is not None:
                result_dict[res_name] = bond_dict
        except (ValueError, KeyError, AttributeError):
            # Residue not found in CCD or malformed CCD entry (e.g., ions without atom data), skip
            continue

    # ... and override with custom bonds where provided
    if custom_bond_dict is not None:
        for res_name, bonds in custom_bond_dict.items():
            # Normalize custom bond keys to sorted tuples to match _bonds_to_dict's convention.
            normalized = {tuple(sorted(k)): v for k, v in bonds.items()}
            if res_name in result_dict:
                result_dict[res_name].update(normalized)
            else:
                result_dict[res_name] = normalized

    return result_dict if result_dict else None


# +---- Adapted from biotite.structure.io.pdbx.convert ----+
_FIND_MATCHES_SWITCH_THRESHOLD = 4_000_000

# atom_site field names used to match struct_conn bond partners against atoms
_STRUCT_CONN_MATCH_FIELDS = (
    "label_asym_id",
    "label_comp_id",
    "label_seq_id",  # special: "." → auth_seq_id (non-polymer residues)
    "label_atom_id",  # atom name
    "pdbx_label_alt_id",  # special: "."/"?" → "" (optional, skipped if not on atom array)
)


def _get_struct_conn_col_name(col_name: str, partner: int) -> str:
    """Translate an atom_site column name to the struct_conn partner column name."""
    if col_name.startswith("pdbx_"):
        return f"pdbx_ptnr{partner}_{col_name[5:]}"
    return f"ptnr{partner}_{col_name}"


def _find_matches(query_arrays: list[np.ndarray], reference_arrays: list[np.ndarray]) -> np.ndarray:
    """Return the reference index matching each query row, or -1 if not found."""
    if query_arrays[0].shape[0] * reference_arrays[0].shape[0] <= _FIND_MATCHES_SWITCH_THRESHOLD:
        return _find_matches_by_dense_array(query_arrays, reference_arrays)
    return _find_matches_by_dict(query_arrays, reference_arrays)


def _find_matches_by_dense_array(query_arrays: list[np.ndarray], reference_arrays: list[np.ndarray]) -> np.ndarray:
    """Pure-numpy matching via broadcasting; last match wins (handles alt-locs)."""
    masks = np.stack(
        [q[:, np.newaxis] == r[np.newaxis, :] for q, r in zip(query_arrays, reference_arrays, strict=False)],
        axis=-1,
    )
    result = np.full(len(query_arrays[0]), -1, dtype=int)
    query_matches, ref_matches = np.where(np.all(masks, axis=-1))
    result[query_matches] = ref_matches  # last match wins for duplicates (alt-locs)
    return result


def _find_matches_by_dict(query_arrays: list[np.ndarray], reference_arrays: list[np.ndarray]) -> np.ndarray:
    """Dict-based matching for large structures; last match wins (handles alt-locs)."""
    ref_dict = {row: idx for idx, row in enumerate(zip(*reference_arrays, strict=False))}
    return np.array([ref_dict.get(tuple(q), -1) for q in zip(*query_arrays, strict=False)])


def get_struct_conn_bonds(
    atom_array: AtomArray,
    struct_conn_dict: dict[str, np.ndarray],
    add_bond_types: tuple[str, ...] = ("covale",),
    raise_on_failure: bool = False,
    distance_policy: LongBondPolicyType = "filter",
) -> struc.BondList:
    """Find inter-residue bonds from the CIF ``struct_conn`` category.

    Modified from biotite's internal ``_get_struct_conn_bonds``.

    Args:
        atom_array: The atom array used to look up atom indices.
        struct_conn_dict: The ``struct_conn`` category of a CIF block as a dict of numpy arrays.
            Required keys: ``conn_type_id``, ``ptnr{1,2}_label_asym_id``,
            ``ptnr{1,2}_label_comp_id``, ``ptnr{1,2}_label_seq_id``,
            ``ptnr{1,2}_label_atom_id``.
        add_bond_types: Bond type IDs to include. Valid values are ``"covale"``, ``"disulf"``,
            and ``"metalc"``. Defaults to ``["covale"]``.
        raise_on_failure: If ``True``, raise on missing atoms or residues. Defaults to ``False``.
        distance_policy: How to handle implausibly long bonds. Explicit glycosylation links and
            covalent links between CCD-recognized carbohydrates are warned above 1.7 A but kept through 2.4 A.

    Returns:
        A :py:class:`biotite.structure.BondList` ready to merge into the atom array's bond list.

    Reference:
        `struct_conn.conn_type_id <https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.conn_type_id.html>`_
    """
    invalid_bond_types = set(add_bond_types) - STRUCT_CONN_BOND_TYPES
    if invalid_bond_types:
        raise ValueError(
            f"Invalid bond type(s) provided: {invalid_bond_types}! Valid bond types are: {STRUCT_CONN_BOND_TYPES}"
        )

    n_atoms = atom_array.array_length()

    if not struct_conn_dict:
        return struc.BondList(n_atoms)

    # Filter rows by requested bond types
    conn_types = struct_conn_dict["conn_type_id"]
    row_mask = np.isin(conn_types, add_bond_types)

    # Filter out bonds involving crystal-symmetry mates (identity symmetry = "1_555")
    _IDENTITY = "1_555"  # noqa: N806
    if "ptnr1_symmetry" in struct_conn_dict:
        row_mask &= struct_conn_dict["ptnr1_symmetry"].astype(str) == _IDENTITY
    if "ptnr2_symmetry" in struct_conn_dict:
        row_mask &= struct_conn_dict["ptnr2_symmetry"].astype(str) == _IDENTITY

    if not row_mask.any():
        # No matches
        return struc.BondList(n_atoms)

    filtered = {k: v[row_mask] for k, v in struct_conn_dict.items()}
    n_rows = int(row_mask.sum())
    logger.debug(f"Attempting to add {n_rows} bonds from `struct_conn`")

    # --- Extract per-row struct_conn fields ---
    def _get_field(name: str, default: str = "") -> np.ndarray:
        return filtered[name] if name in filtered else np.full(n_rows, default)

    # --- Build reference arrays (one entry per atom) ---
    atom_names = atom_array.atom_name
    alt_atom_ids = get_annotation(atom_array, "alt_atom_id", default=atom_names)
    uses_alt_atom_id = get_annotation(atom_array, "uses_alt_atom_id", default=np.zeros(n_atoms, dtype=bool))
    eff_atom_names = np.where(uses_alt_atom_id, alt_atom_ids, atom_names)

    p1_transformation_id = filtered.get("ptnr1_transformation_id")
    p2_transformation_id = filtered.get("ptnr2_transformation_id")
    has_transformation = "transformation_id" in atom_array.get_annotation_categories()
    use_transformation_in_key = (
        has_transformation and p1_transformation_id is not None and p2_transformation_id is not None
    )

    # --- Build reference values keyed by field name ---
    # (built once, reused for both partners)
    ref_vals: dict[str, np.ndarray] = {
        "label_asym_id": atom_array.chain_id,
        "label_comp_id": atom_array.res_name,
        "label_seq_id": atom_array.res_id.astype(str),
        "label_atom_id": eff_atom_names,
    }

    # Optional: alt-conf matching (only if label_alt_id is on the atom array AND
    # struct_conn actually contains the pdbx_ptnr{N}_label_alt_id columns
    if "label_alt_id" in atom_array.get_annotation_categories() and (
        "pdbx_ptnr1_label_alt_id" in filtered or "pdbx_ptnr2_label_alt_id" in filtered
    ):
        alt_ids_norm = atom_array.label_alt_id.copy()
        alt_ids_norm[np.isin(alt_ids_norm, (".", "?", " "))] = ""
        ref_vals["pdbx_label_alt_id"] = alt_ids_norm

    # --- Build per-partner query arrays and match ---
    atom_indices: list[np.ndarray] = []
    for partner in (1, 2):
        q_arrays: list[np.ndarray] = []
        r_arrays: list[np.ndarray] = []

        for field in _STRUCT_CONN_MATCH_FIELDS:
            if field not in ref_vals:
                continue  # optional field (e.g., pdbx_label_alt_id) not on atom array
            sc_col = _get_struct_conn_col_name(field, partner)
            q = _get_field(sc_col)
            if field == "label_seq_id":
                auth_col = _get_struct_conn_col_name("auth_seq_id", partner)
                q = np.where(q == ".", _get_field(auth_col), q)
            elif field == "pdbx_label_alt_id":
                q = np.where(np.isin(q, (".", "?", " ")), "", q)
            q_arrays.append(q)
            r_arrays.append(ref_vals[field])

        # NOTE: We differ from Biotite by adding custom handling of pre-built transformations
        if use_transformation_in_key:
            tfm_col = f"ptnr{partner}_transformation_id"
            q_arrays.append(filtered[tfm_col].astype(str))
            r_arrays.append(atom_array.transformation_id.astype(str))

        atom_indices.append(_find_matches(q_arrays, r_arrays))
    idx1, idx2 = atom_indices

    # --- Filter missing bonds, log, and optionally raise ---
    # kept only for error logging
    chains = [filtered["ptnr1_label_asym_id"], filtered["ptnr2_label_asym_id"]]
    label_seqs = [filtered["ptnr1_label_seq_id"], filtered["ptnr2_label_seq_id"]]
    res_names_q = [filtered["ptnr1_label_comp_id"], filtered["ptnr2_label_comp_id"]]
    atom_ids = [filtered["ptnr1_label_atom_id"], filtered["ptnr2_label_atom_id"]]

    valid = (idx1 != -1) & (idx2 != -1)
    if not valid.all():
        for i in np.where(~valid)[0]:
            logger.info(
                f"Covalent bond involving atoms {chains[0][i]}/{label_seqs[0][i]}/{res_names_q[0][i]}/{atom_ids[0][i]}"
                f" or {chains[1][i]}/{label_seqs[1][i]}/{res_names_q[1][i]}/{atom_ids[1][i]}"
                " not found in the atom array!"
            )
        if raise_on_failure:
            i = int(np.where(~valid)[0][0])
            raise ValueError(
                f"Atom {atom_ids[0][i]} in residue {chains[0][i]}/{label_seqs[0][i]}/{res_names_q[0][i]}"
                f" or atom {atom_ids[1][i]} in residue {chains[1][i]}/{label_seqs[1][i]}/{res_names_q[1][i]}"
                " not found in the atom array!"
            )

    valid = filter_link_distances(atom_array, filtered, idx1, idx2, valid, distance_policy)

    # Assign bond types after atom matching and distance filtering; metal coordination has no covalent order.
    # Preserve recognized explicit orders, infer missing/unknown orders from CCD chemistry, and reject other values.
    pdbx_value_order = filtered.get("pdbx_value_order")
    bond_types = np.empty(n_rows, dtype=int)
    for i in np.flatnonzero(valid):
        if filtered["conn_type_id"][i] == "metalc":
            bond_types[i] = int(struc.BondType.COORDINATION)
            continue
        order = str(pdbx_value_order[i]) if pdbx_value_order is not None else "?"
        if order in STRUCT_CONN_BOND_ORDER_TO_INT:
            bond_types[i] = STRUCT_CONN_BOND_ORDER_TO_INT[order]
        elif order in ("?", ".", ""):
            bond_types[i] = infer_link_order(atom_array, int(idx1[i]), int(idx2[i]))
        else:
            raise ValueError(f"Unsupported struct_conn bond order: {order!r}")

    bonds_array = (
        np.stack([idx1[valid], idx2[valid], bond_types[valid]], axis=1) if valid.any() else np.empty((0, 3), dtype=int)
    )
    return struc.BondList(n_atoms, bonds_array)


def get_coarse_graph_as_nodes_and_edges(
    atom_array: AtomArray,
    annotations: str | tuple[str],
    exclude_bond_types: set[BondType] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the coarse-grained nodes and edges at the given annotation level based on the atom array's bond connectivity.

    Args:
        - atom_array (AtomArray): The atom array containing atomic information and bonds.
        - annotations (str | tuple[str]): A single annotation or a tuple of annotations to be used for node
            identification.
        - exclude_bond_types (set | None): Bond types to exclude from connectivity. For example,
            pass ``{BondType.COORDINATION}`` to prevent metal-ligand bonds from merging components.

    Returns:
        - nodes (np.ndarray): An array of unique nodes, each represented by a combination of annotations.
        - edges (np.ndarray): An array of edges, where each edge is a tuple of node indices representing a bond
            between two nodes.

    Example:
        >>> atom_array = cached_parse("5ocm")["atom_array"]
        >>> nodes, edges = get_coarse_graph(atom_array, ["chain_id", "transformation_id"])
        >>> print(nodes)
        array([('A', '1'), ('F', '1'), ('G', '1'), ('H', '1'), ('I', '1'),
               ('W', '1'), ('X', '1'), ('Y', '1')],
              dtype=[('chain_id', '<U4'), ('transformation_id', '<U1')])
        >>> print(edges)
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [5, 5],
               [6, 6]])
    """
    annotations = [annotations] if isinstance(annotations, str) else annotations

    bond_array = atom_array.bonds.as_array()
    if exclude_bond_types:
        mask = ~np.isin(bond_array[:, 2], [int(bt) for bt in exclude_bond_types])
        bond_array = bond_array[mask]
    atom1, atom2 = bond_array[:, 0], bond_array[:, 1]

    if len(annotations) > 1:
        _annots = np.zeros(
            len(atom_array), dtype=[(annot, atom_array.get_annotation(annot).dtype) for annot in annotations]
        )
        for annot in annotations:
            _annots[annot] = atom_array.get_annotation(annot)  # [n_atoms, n_annotations]
    else:
        _annots = atom_array.get_annotation(annotations[0])  # [n_atoms]

    # Group atoms by annotation and assign sorted node indices.
    nodes, atom_to_node = np.unique(_annots, axis=0, return_inverse=True)
    if len(nodes) == 0:
        return nodes, np.empty((0, 2), dtype=nodes.dtype)

    # Include a self-edge for every node, including isolated nodes.
    node_indices = np.arange(len(nodes))
    self_edges = np.column_stack((node_indices, node_indices))

    # Map bonds between distinct nodes, then sort and deduplicate edges.
    node1, node2 = atom_to_node[atom1], atom_to_node[atom2]
    between_nodes = node1 != node2
    bond_edges = np.column_stack((node1[between_nodes], node2[between_nodes]))
    edges = np.unique(np.concatenate((self_edges, bond_edges)), axis=0)

    return nodes, edges


def get_connected_nodes(nodes: np.ndarray, edges: np.ndarray) -> list[list[Any]]:
    """Returns connected nodes as a mapped list given corresponding arrays of nodes and edges.

    Example:
        >>> nodes = np.array([("A", "1"), ("B", "1"), ("C", "1"), ("D", "1")])
        >>> edges = np.array([[0, 1], [0, 2], [1, 2]])
        >>> connected_nodes = get_connected_nodes(nodes, edges)
        >>> print(connected_nodes)
        [[("A", "1"), ("B", "1"), ("C", "1")], [("D", "1")]]
    """
    # ...make the graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # ...return lists of connected chains
    return [[nodes[x] for x in component] for component in nx.connected_components(graph)]


def hash_graph(
    graph: nx.Graph,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    iterations: int = 3,
    digest_size: int = 16,
) -> str:
    """
    Computes a hash for a given graph using the Weisfeiler-Lehman (WL) graph hashing algorithm and additionally
    adds a node and edge attribute hash, if specified, to deal with common edge cases where WL fails (e.g.
    disconnected graphs).

    Args:
        - graph (networkx.Graph): The input graph to be hashed.
        - node_attr (str | None): The node attribute to be used for hashing. If None, node attributes are ignored.
        - edge_attr (str | None): The edge attribute to be used for hashing. If None, edge attributes are ignored.
        - iterations (int): The number of iterations for the WL algorithm. Default is 3.
        - digest_size (int): The size of the hash digest for WL. Default is 16.

    Returns:
        - str: The computed hash of the graph.

    Example:
        >>> import networkx as nx
        >>> G = nx.gnm_random_graph(10, 15)
        >>> hash_graph(G)
        '504894f49dd84b17c391b163af69624b'
    """
    # ... compute WL-hash
    hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        graph, node_attr=node_attr, edge_attr=edge_attr, iterations=iterations, digest_size=digest_size
    )

    if node_attr is not None:
        # ... add number of unique nodes to hash
        hash += f"_{len(graph.nodes)}"
        # ... add number of unique node attributes with counts to hash
        node_attr_dict = nx.get_node_attributes(graph, node_attr)
        hash += "_" + ",".join(
            [
                f"{elt}:{count}"
                for elt, count in zip(*np.unique(list(node_attr_dict.values()), return_counts=True), strict=False)
            ]
        )
    if edge_attr is not None:
        # ... add number of unique edges to hash
        hash += f"_{len(graph.edges)}"
    return hash


def _atom_array_to_networkx_graph(
    atom_array: AtomArray,
    annotations: tuple[str] = ("element", "atom_name"),
    bond_order: bool = True,
    cast_aromatic_bonds_to_same_type: bool = True,
) -> nx.Graph:
    """Convert an AtomArray to a NetworkX graph."""
    # ... create the bond graph
    bonds = atom_array.bonds.as_array()

    # ... create the bond graph for the atom array, adding all nodes first to ensure correct indexing
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(range(len(atom_array)))
    bond_list = []

    # ... add edges from bond list
    if len(bonds) > 0:
        bond_list = [tuple(bond) for bond in bonds[:, :2]]
        bond_graph.add_edges_from(bond_list)

    # ... annotate the bond graph with bond order
    if bond_order:
        bond_type = bonds[:, -1]
        if cast_aromatic_bonds_to_same_type:
            bond_type[bond_type == struc.BondType.AROMATIC_SINGLE] = 0
            bond_type[bond_type == struc.BondType.AROMATIC_DOUBLE] = 0
            bond_type[bond_type == struc.BondType.AROMATIC_TRIPLE] = 0

        nx.set_edge_attributes(
            bond_graph, {tuple(bond): type for bond, type in zip(bond_list, bond_type, strict=False)}, "bond_type"
        )

    # ... annotate the bond graph with the desired node annotations
    if annotations:
        node_data = sum_string_arrays(*[atom_array.get_annotation(annot).astype(str) for annot in annotations])
        # ... map the node annotations to the bond graph
        nx.set_node_attributes(bond_graph, {n: node_data[n] for n in bond_graph.nodes()}, "node_data")

    return bond_graph


def hash_atom_array(
    atom_array: AtomArray,
    annotations: tuple[str] = ("element", "atom_name"),
    bond_order: bool = True,
    cast_aromatic_bonds_to_same_type: bool = False,
    use_md5: bool = False,
    md5_length: int | None = None,
) -> str:
    """
    Computes a hash for an AtomArray based on the bond connectivity and the selected node annotations.

    Args:
        atom_array (AtomArray): The array of atoms to hash
        annotations (tuple[str]): The node annotations to include in the hash
        bond_order (bool): Whether to include bond order in the hash
        cast_aromatic_bonds_to_same_type (bool): Whether to treat all aromatic bonds as the same type
        use_md5 (bool): Whether to use MD5 hashing on the output
        md5_length (int | None): If using MD5, the number of characters to keep from the hash. If None, returns full hash.

    Returns:
        str: The computed hash
    """
    # ... create the bond graph
    bond_graph = _atom_array_to_networkx_graph(
        atom_array,
        annotations=annotations,
        bond_order=bond_order,
        cast_aromatic_bonds_to_same_type=cast_aromatic_bonds_to_same_type,
    )

    hash_str = hash_graph(
        bond_graph, node_attr="node_data" if annotations else None, edge_attr="bond_type" if bond_order else None
    )

    if use_md5:
        hash_str = hashlib.md5(hash_str.encode()).hexdigest()
        if md5_length is not None:
            hash_str = hash_str[:md5_length]

    return hash_str


def generate_inter_level_bond_hash(
    atom_array: AtomArray,
    lower_level_id: str,
    lower_level_entity: str | None = None,
    exclude_bond_types: set[BondType] | None = None,
) -> str:
    """Generates a hash string representing the inter-level bonds within an AtomArray.

    When computing entities IDs, we must consider inter-level bonds at the atom- and residue-level to avoid ambiguity.

    Args:
        atom_array (AtomArray): The array of atoms containing bond and annotation information.
        lower_level_id (str): The level which to find, and hash, the inter-level bonds. For example, when computing molecule entities, we'd consider the inter-PN Unit bonds.
        lower_level_entity (str | None): An additional entity annotation to use when computing the hash. Optional; if None, then only residue ID, residue name, and atom name are used.

    Returns:
        str: A hash string representing the inter-level bonds.
    """
    # ... find the inter-level bonds
    bond_a = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 0]]
    bond_b = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 1]]
    inter_level_bonds = atom_array.bonds.as_array()[bond_a != bond_b]

    # Filter excluded bond types
    if exclude_bond_types and inter_level_bonds.size:
        mask = ~np.isin(inter_level_bonds[:, 2], [int(bt) for bt in exclude_bond_types])
        inter_level_bonds = inter_level_bonds[mask]

    if inter_level_bonds.size:
        # Extract annotations once (outside loop) for efficient indexing
        res_ids = atom_array.res_id
        res_names = atom_array.res_name
        atom_names = atom_array.atom_name

        # Handle optional lower_level_entity annotation
        if lower_level_entity is not None:
            lower_level_entity_values = atom_array.get_annotation(lower_level_entity)
        else:
            lower_level_entity_values = None

        # ... loop over the bonds and create a (sorted) list of tuples with the relevant information
        bond_tuples = []
        for atom_idx in range(inter_level_bonds.shape[0]):
            # Use direct indexing instead of AtomArray slicing
            idx_a = inter_level_bonds[atom_idx, 0]
            idx_b = inter_level_bonds[atom_idx, 1]

            bond_tuples.append(
                tuple(
                    sorted(
                        [
                            (
                                lower_level_entity_values[idx_a] if lower_level_entity_values is not None else None,
                                res_ids[idx_a],
                                res_names[idx_a],
                                atom_names[idx_a],
                            ),
                            (
                                lower_level_entity_values[idx_b] if lower_level_entity_values is not None else None,
                                res_ids[idx_b],
                                res_names[idx_b],
                                atom_names[idx_b],
                            ),
                        ]
                    )
                )
            )

        # ...sort the list of tuples, and hash
        return str(hash(tuple(sorted(bond_tuples))))
    else:
        return ""


def filter_bonds_by_distance(
    atom_array: AtomArray,
    policy: LongBondPolicyType = "filter",
    chno_threshold: float = BOND_DISTANCE_THRESHOLD_CHNO,
    chnops_threshold: float = BOND_DISTANCE_THRESHOLD_CHNOPS,
    other_threshold: float = BOND_DISTANCE_THRESHOLD_OTHER,
) -> AtomArray:
    """Handle unphysical bonds based on element-dependent distance thresholds.

    The thresholds are:
    1. Bonds involving ONLY C, H, N, O: flagged if distance > chno_threshold
    2. Bonds involving ONLY C, H, N, O, P, S: flagged if distance > chnops_threshold
    3. Any other bonds (metals, etc.): flagged if distance > other_threshold

    Args:
        atom_array: The AtomArray containing atoms and bonds to check.
        policy: How to handle long bonds. Options:

            - ``"keep"``: No checking, return atom_array unchanged.
            - ``"filter"``: Remove all bonds exceeding thresholds.
            - ``"warn"``: Log warning about all long bonds but keep them.
            - ``"raise"``: Raise ``ValueError`` if any long bonds detected.
            - ``"filter_nonstandard_only"``: Remove long bonds only in non-standard residues.
              Bonds where both atoms are in standard AA (20 canonical + UNK), RNA (A, C, G, U + N),
              or DNA (DA, DC, DG, DT + DN) are preserved.
            - ``"warn_nonstandard_only"``: Warn about long bonds in non-standard residues only.
            - ``"raise_nonstandard_only"``: Raise error if long bonds in non-standard residues.

            Defaults to ``"filter"``.
        chno_threshold: Maximum distance for bonds between only C, H, N, O atoms.
            Defaults to 1.8 Angstroms.
        chnops_threshold: Maximum distance for bonds between only C, H, N, O, P, S atoms.
            Defaults to 2.4 Angstroms.
        other_threshold: Maximum distance for bonds involving any other elements.
            Defaults to 3.6 Angstroms.

    Returns:
        The input AtomArray, potentially with long bonds removed (if ``policy="filter"``).

    Raises:
        ValueError: If ``policy="raise"`` and long bonds are detected.

    Note:
        Bonds involving atoms with NaN coordinates are preserved (not flagged).
    """
    if policy == "keep":
        return atom_array

    if atom_array.bonds is None or len(atom_array.bonds.as_array()) == 0:
        return atom_array

    bonds_arr = atom_array.bonds.as_array()
    atom1_idxs = bonds_arr[:, 0]
    atom2_idxs = bonds_arr[:, 1]

    # Get coordinates and elements
    coords = atom_array.coord
    elements = atom_array.element

    # Compute distances
    distances = np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)

    # Check for NaN coordinates (unresolved atoms) - preserve these bonds
    has_nan_coords = np.any(np.isnan(coords[atom1_idxs]), axis=1) | np.any(np.isnan(coords[atom2_idxs]), axis=1)

    # Determine threshold for each bond based on elements
    thresholds = _element_distance_thresholds(
        elements[atom1_idxs], elements[atom2_idxs], chno_threshold, chnops_threshold, other_threshold
    )

    # Identify bonds exceeding thresholds (excluding NaN coords)
    exceeds_threshold = (distances > thresholds) & ~has_nan_coords

    # Apply standard residue exemption for *_nonstandard_only policies
    if policy.endswith("_nonstandard_only"):
        # Check if both atoms in bond are in standard/unknown residues
        res_names = atom_array.res_name
        is_standard_1 = np.isin(res_names[atom1_idxs], STANDARD_AND_UNKNOWN_POLYMER_RESIDUES)
        is_standard_2 = np.isin(res_names[atom2_idxs], STANDARD_AND_UNKNOWN_POLYMER_RESIDUES)
        both_standard = is_standard_1 & is_standard_2

        # Exclude standard residue bonds from being flagged
        exceeds_threshold = exceeds_threshold & ~both_standard

    n_long_bonds = np.sum(exceeds_threshold)

    if n_long_bonds == 0:
        return atom_array

    msg = (
        f"Found {n_long_bonds} bonds exceeding distance thresholds "
        f"(CHNO: {chno_threshold}A, CHNOPS: {chnops_threshold}A, other: {other_threshold}A)"
    )

    # Extract base policy action (filter, warn, or raise)
    base_policy = policy.split("_")[0] if "_" in policy else policy

    if base_policy == "raise":
        raise ValueError(msg)
    elif base_policy == "warn":
        logger.warning(msg)
        return atom_array
    elif base_policy == "filter":
        logger.warning(f"Filtering {n_long_bonds} long bonds. {msg}")
        # Create new BondList with only valid bonds
        valid_bonds = bonds_arr[~exceeds_threshold]
        atom_array.bonds = struc.BondList(atom_array.array_length(), valid_bonds)
        return atom_array
    else:
        raise ValueError(
            f"Invalid policy: {policy}. Must be 'keep', 'filter', 'warn', 'raise', "
            f"or one of their *_nonstandard_only variants."
        )


def add_bonds_from_struct_conn(
    atom_array: AtomArray | AtomArrayStack,
    cif_block: pdbx.CIFBlock,
    add_bond_types_from_struct_conn: tuple[str, ...] = ("covale",),
    struct_conn_distance_policy: LongBondPolicyType = "filter",
) -> AtomArray | AtomArrayStack:
    """Convenience wrapper to add bonds to an AtomArray from a struct_conn CIF category.

    See Also:
      :py:func:`~atomworks.io.utils.bonds.get_struct_conn_bonds` -
      For arguments and return types.
    """
    struct_conn_dict = category_to_dict(cif_block, "struct_conn")

    # For AtomArrayStack, bonds are shared across all frames, so we only need to
    # extract bonds once using the first frame's annotations
    reference_array = atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array

    # Get bonds from struct_conn and merge with existing bonds
    new_bonds = get_struct_conn_bonds(
        reference_array,
        struct_conn_dict=struct_conn_dict,
        add_bond_types=add_bond_types_from_struct_conn,
        distance_policy=struct_conn_distance_policy,
    )
    atom_array.bonds = atom_array.bonds.merge(new_bonds)

    return atom_array


def get_inter_pn_unit_bond_mask(atom_array: AtomArray) -> np.ndarray:
    """Return a mask indicating which bonds are between two distinct PN units."""
    bond_arr = atom_array.bonds.as_array()
    bond_pn_unit_a = atom_array.pn_unit_iid[bond_arr[:, 0]]
    bond_pn_unit_b = atom_array.pn_unit_iid[bond_arr[:, 1]]
    return bond_pn_unit_a != bond_pn_unit_b
