import functools
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import networkx as nx
import numpy as np
import toolz

from atomworks.common import exists, immutable_lru_cache
from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    CCD_MIRROR_PATH,
    DNA_LIKE_CHEM_TYPES,
    DO_NOT_MATCH_CCD,
    NA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_LIGAND,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType, ChainTypeInfo

logger = logging.getLogger(__name__)


@functools.cache
def aa_chem_comps() -> frozenset[str]:
    """Set of amino acid chemical components.

    Returns:
        Set of amino acid chemical components (e.g., {'ALA', 'ARG', ...}).
    """
    return frozenset(struc.info.groups._get_group_members(list(AA_LIKE_CHEM_TYPES)))


@functools.cache
def na_chem_comps() -> frozenset[str]:
    """Set of nucleic acid chemical components.

    Returns:
        Set of nucleic acid chemical components (e.g., {'DA', 'DC', ...}).
    """
    return frozenset(struc.info.groups._get_group_members(list(NA_LIKE_CHEM_TYPES)))


@functools.cache
def rna_chem_comps() -> frozenset[str]:
    """Set of RNA chemical components.

    Returns:
        Set of RNA chemical components (e.g., {'A', 'C', ...}).
    """
    return frozenset(struc.info.groups._get_group_members(list(RNA_LIKE_CHEM_TYPES)))


@functools.cache
def dna_chem_comps() -> frozenset[str]:
    """Set of DNA chemical components.

    Returns:
        Set of DNA chemical components (e.g., {'DA', 'DC', ...}).
    """
    return frozenset(struc.info.groups._get_group_members(list(DNA_LIKE_CHEM_TYPES)))


@functools.cache
def chem_comp_to_one_letter() -> dict[str, str]:
    """Dictionary mapping the chemical components to their 1-letter code.

    Note:
        Chemical components historically used to be 3-letter codes,
        but nowadays longer codes exist.

    Returns:
        Dictionary mapping chemical component names to their 1-letter codes.

    References:
        `RCSB Chemical Component Dictionary <https://www.rcsb.org/ligand>`_
        `Biotite CCD Module <https://www.biotite-python.org/apidoc/biotite.structure.info.ccd.html>`_
    """
    ccd = struc.info.ccd.get_ccd()
    three_letter_code = ccd["chem_comp"]["three_letter_code"].as_array()
    one_letter_code = ccd["chem_comp"]["one_letter_code"].as_array()

    three_to_one = {}
    for full, one in zip(three_letter_code, one_letter_code, strict=False):
        if (len(one) > 1) or (one == "?"):
            continue
        if full == "?":
            continue
        three_to_one[full] = one

    return three_to_one


@functools.cache
def get_available_ccd_codes_in_mirror(ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> frozenset[str]:
    """Set of all CCD codes available in the local mirror.

    Only counts codes when they adhere to the CCD mirror layout (e.g. .../H/HEM/HEM.cif)

    Args:
        ccd_mirror_path: Path to the CCD mirror directory.

    Returns:
        Set of all available CCD codes in the mirror.

    References:
        `RCSB Chemical Component Dictionary <https://www.rcsb.org/ligand>`_
        `CCD Mirror Layout <https://www.rcsb.org/ligand>`_
    """
    root = os.fspath(ccd_mirror_path)

    # Check if we have a pre-computed cache file
    cache_file = os.path.join(root, ".ccd_codes_cache")
    if os.path.exists(cache_file):
        try:
            # Check if cache is newer than the directory
            cache_mtime = os.path.getmtime(cache_file)
            dir_mtime = os.path.getmtime(root)
            if cache_mtime > dir_mtime:
                with open(cache_file) as f:
                    codes = {line.strip() for line in f if line.strip()}
                    return frozenset(codes)
        except OSError:
            # If cache is corrupted, fall back to scanning
            pass

    # Fall back to filesystem scan
    codes: set[str] = set()

    root_path = Path(root)

    for level1_dir in root_path.iterdir():
        if not level1_dir.is_dir():
            continue
        first_letter = level1_dir.name
        if len(first_letter) != 1:
            continue

        for level2_dir in level1_dir.iterdir():
            if not level2_dir.is_dir():
                continue
            code = level2_dir.name
            if not code or code[0] != first_letter:
                continue

            expected_file = level2_dir / f"{code}.cif"
            if expected_file.is_file():
                codes.add(code)

    # Cache the results for next time
    try:
        with open(cache_file, "w") as f:
            for code in sorted(codes):
                f.write(f"{code}\n")
    except OSError:
        # If we can't write cache, that's okay
        pass

    return frozenset(codes)


@functools.cache
def get_available_ccd_codes_in_biotite() -> frozenset[str]:
    """Set of all CCD codes available in Biotite's built-in Chemical Component Dictionary."""
    return frozenset(struc.info.ccd.get_ccd()["chem_comp"]["id"].as_array())


@functools.cache
def get_available_ccd_codes(ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH) -> frozenset[str]:
    """Returns a frozenset of all CCD codes available.

    If a mirror path is provided, it will be used to check the local mirror first.
    Otherwise, Biotite's built-in CCD will be used.
    """
    mirror_codes = get_available_ccd_codes_in_mirror(ccd_mirror_path) if ccd_mirror_path else frozenset()
    biotite_codes = get_available_ccd_codes_in_biotite()
    return mirror_codes | biotite_codes


def get_ccd_component_from_biotite(ccd_code: str, **parse_ccd_cif_kwargs) -> struc.AtomArray:
    """
    Retrieves a component from the Chemical Component Dictionary using Biotite's built-in functionality.

    Args:
        - ccd_code (str): The three-letter code of the chemical component to retrieve.

    Returns:
        - AtomArray: The atomic structure of the requested component.
    """
    try:
        block = _filter_biotite_ccd_for_ccd_code(ccd_code)
        atom_array = parse_ccd_cif(block, **parse_ccd_cif_kwargs)
        return atom_array
    except KeyError:
        raise ValueError(f"No atom information found for residue '{ccd_code}' in Biotite's CCD") from None


def check_ccd_codes_are_available(
    ccd_codes: Iterable[str], ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> bool:
    """Checks if the provided CCD codes are available in the local mirror."""
    available_ccds = get_available_ccd_codes(ccd_mirror_path)
    invalid_ccds = set(ccd_codes) - available_ccds
    if invalid_ccds:
        which_mirror = "Biotite's built-in CCD" if ccd_mirror_path is None else f"the local mirror at {ccd_mirror_path}"
        if mode == "warn":
            logger.warning(f"The following CCD codes were not found in {which_mirror}: {invalid_ccds}")
        elif mode == "raise":
            raise ValueError(f"The following CCD codes were not found in {which_mirror}: {invalid_ccds}")
    return not bool(invalid_ccds)


def _get_ccd_path(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> os.PathLike:
    """
    Constructs the file path for a Chemical Component Dictionary entry in the local mirror.

    Args:
        - ccd_code (str): The three-letter code of the chemical component.
        - ccd_mirror_path (os.PathLike): Path to the root of the CCD mirror directory.

    Returns:
        - os.PathLike: Full path to the component's CIF file.
    """
    return os.path.join(ccd_mirror_path, ccd_code[0], ccd_code, ccd_code + ".cif")


def _filter_biotite_ccd_for_ccd_code(ccd_code: str) -> pdbx.CIFBlock:
    """Filter the Biotite CCD for a given CCD code."""
    if ccd_code not in get_available_ccd_codes_in_biotite():
        raise KeyError(f"CCD code `{ccd_code}` not found in Biotite's CCD")

    ccd = struc.info.get_ccd()

    # Chem comp
    chem_comp = ccd.get("chem_comp")
    chem_comp = pdbx.convert._filter(chem_comp, chem_comp["id"].as_array() == ccd_code)

    # Chem comp atom
    chem_comp_atom = ccd.get("chem_comp_atom")
    chem_comp_atom = pdbx.convert._filter(chem_comp_atom, chem_comp_atom["comp_id"].as_array() == ccd_code)

    # Chem comp bond
    chem_comp_bond = ccd.get("chem_comp_bond")
    chem_comp_bond = pdbx.convert._filter(chem_comp_bond, chem_comp_bond["comp_id"].as_array() == ccd_code)

    return pdbx.CIFBlock(
        {
            "chem_comp": chem_comp,
            "chem_comp_atom": chem_comp_atom,
            "chem_comp_bond": chem_comp_bond,
        }
    )


def parse_ccd_cif(
    cif: pdbx.CIFFile,
    coords: Literal["model", "ideal_pdbx", "ideal_rdkit"] | None | tuple[str, ...] = (
        "ideal_pdbx",
        "model",
        "ideal_rdkit",
    ),
    add_properties: bool = False,
    add_mapping: bool = False,
) -> struc.AtomArray:
    """Parses a Chemical Component Dictionary CIF file into a Biotite AtomArray structure.

    Args:
        cif: The CIF file containing the component data.
        coords: Type of coordinates to use. Defaults to ("ideal_pdbx", "model", "ideal_rdkit").
            Can be a single coordinate type or a tuple of fallback preferences (e.g., ("ideal_pdbx", "model", "ideal_rdkit")).
            - "model": Use the coordinates that are found in a random (but fixed) pdb file.
            - "ideal_pdbx": Use the idealized coordinates computed by the RCSB PDB (sometimes not available).
            - "ideal_rdkit": Use the idealized coordinates computed by RDKit (sometimes unrealistic).
        add_properties: Whether to include RDKit-computed properties. Defaults to False.
            Properties are available under the ``properties`` attribute of the returned ``AtomArray``.
        add_mapping: Whether to include external resource mappings, such as e.g. the ChEMBL ID.
            Defaults to False.
            Mappings are available under the ``mapping`` attribute of the returned ``AtomArray``.

    Returns:
        AtomArray: The parsed atomic structure with requested annotations and properties.

    Example:
        >>> cif = pdbx.CIFFile.read("path/to/ALA.cif")
        >>> atom_array = parse_ccd_cif(cif, coords="ideal_pdbx")
        >>> # With fallback preferences:
        >>> atom_array = parse_ccd_cif(cif, coords=["ideal_pdbx", "model", "ideal_rdkit"])
    """
    # Convert single value or list to tuple for uniform processing and hashability
    if isinstance(coords, str):
        coord_types = (coords,)
    elif isinstance(coords, list | tuple):
        coord_types = tuple(coords)
    else:
        coord_types = (coords,) if coords is not None else (None,)

    valid_types = ("model", "ideal_pdbx", "ideal_rdkit", None)

    # Validate all coord types
    for coord_type in coord_types:
        if coord_type not in valid_types:
            raise ValueError(
                f"Invalid coordinate type: {coord_type}. Must be one of 'model', 'ideal_pdbx', 'ideal_rdkit' or `None`."
            )

    block = pdbx.convert._get_block(cif, None)

    # Extract metadata
    metadata = block.get("chem_comp")
    ccd_code = metadata["id"].as_item()

    # Extract atom specific information
    atom_data = block.get("chem_comp_atom")

    # Initialize the empty array:
    atoms = struc.AtomArray(atom_data.row_count)

    # Fill standard annotations
    as_bool = lambda x: np.where(x.as_array(str) == "Y", True, False)  # noqa: E731
    atoms.set_annotation("res_name", atom_data.get("comp_id").as_array(str))
    atoms.set_annotation("atom_name", atom_data.get("atom_id").as_array(str))
    atoms.set_annotation("alt_atom_id", atom_data.get("alt_atom_id").as_array(str))
    atoms.set_annotation("element", atom_data.get("type_symbol").as_array(str))
    atoms.set_annotation("charge", atom_data.get("charge").as_array(np.int8))
    atoms.set_annotation("stereo", atom_data.get("pdbx_stereo_config").as_array(str))
    atoms.set_annotation("is_aromatic", as_bool(atom_data.get("pdbx_aromatic_flag")))
    atoms.set_annotation("is_leaving_atom", as_bool(atom_data.get("pdbx_leaving_atom_flag")))
    atoms.set_annotation("is_backbone_atom", as_bool(atom_data.get("pdbx_backbone_atom_flag")))
    atoms.set_annotation("is_n_terminal_atom", as_bool(atom_data.get("pdbx_n_terminal_atom_flag")))
    atoms.set_annotation("is_c_terminal_atom", as_bool(atom_data.get("pdbx_c_terminal_atom_flag")))
    atoms.set_annotation("res_id", np.full(len(atoms), 1))  # We 1-index residue IDs to be consistent with RCSB

    # Try setting hetero flag
    hetero = ccd_code not in struc.info.atoms.NON_HETERO_RESIDUES
    atoms.set_annotation("hetero", [hetero] * len(atoms))

    # Define coordinate columns for each type
    coordinate_columns = {
        "model": ["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"],
        "ideal_pdbx": ["pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"],
        "ideal_rdkit": ["Cartn_x_rdkit", "Cartn_y_rdkit", "Cartn_z_rdkit"],
    }

    # Try each coordinate type until one works
    coords_set = False

    for coord_type in coord_types:
        if coord_type is None:
            # Skip if None is explicitly requested in the preference list
            continue

        try:
            if (coord_type == "ideal_rdkit") and (rdkit_data := block.get("pdbe_chem_comp_rdkit_conformer")):
                # Special case for rdkit as it uses a different dataset
                rdkit_data = block.get("pdbe_chem_comp_rdkit_conformer")
                assert np.all(rdkit_data["atom_id"].as_array(str) == atoms.get_annotation("atom_name"))
                for i, col in enumerate(coordinate_columns[coord_type]):
                    atoms.coord[:, i] = rdkit_data[col].as_array(np.float32)
            else:
                # Standard case for model and ideal_pdbx
                for i, col in enumerate(coordinate_columns[coord_type]):
                    atoms.coord[:, i] = atom_data[col].as_array(np.float32)

            # Check if the coordinates are valid (not all zeros/NaN)
            if np.all(atoms.coord == 0) or np.all(np.isnan(atoms.coord)):
                logger.debug(
                    f"Coordinate type '{coord_type}' for '{ccd_code}' contains only zeros/NaN, trying next option"
                )
                continue

            coords_set = True
            # If we're not using the first preference, log a warning
            if coord_type != coord_types[0]:
                logger.warning(
                    f"Using fallback coordinate type '{coord_type}' for '{ccd_code}' instead of '{coord_types[0]}'"
                )
            break

        except (KeyError, AssertionError):
            # Continue to next coordinate type if this one fails
            logger.debug(f"Coordinate type '{coord_type}' not available for '{ccd_code}', trying next option")
            continue

    # Log warning if no coordinates were set
    if not coords_set and coord_types and coord_types[0] is not None:
        logger.warning(
            f"No suitable coordinates found for '{ccd_code}' among preferences {coord_types}. Coordinates will be 'nan'."
        )
        atoms.coord = np.full((len(atoms), 3), np.nan)

    # Extract bond data
    try:
        bond_data = block.get("chem_comp_bond")
        if bond_data is not None:
            bond_dict = pdbx.convert._parse_intra_residue_bonds(bond_data)
            atoms.bonds = struc.connect_via_residue_names(atoms, custom_bond_dict=bond_dict)
    except KeyError:
        atoms.bonds = None
        logger.warning(f"No bond data found for `{ccd_code}`. Bonds will be `None`.")

    # Set general annotations:
    if add_properties:
        try:
            atoms.properties = toolz.valmap(lambda x: x.as_item(), dict(block["pdbe_chem_comp_rdkit_properties"]))
        except KeyError:
            logger.warning(f"No properties data found for `{ccd_code}`. Properties will be `None`.")
            atoms.properties = None

    if add_mapping:
        try:
            mapping = block.get("pdbe_chem_comp_external_mappings")
            atoms.mapping = dict(zip(mapping["resource"], mapping["resource_id"], strict=True))
        except KeyError:
            atoms.mapping = None
            logger.warning(f"No mapping data found for `{ccd_code}`. Mapping will be `None`.")

    return atoms


@immutable_lru_cache(maxsize=20000, deepcopy=True)
def get_ccd_component_from_mirror(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieves and parses a component from a local mirror of the Chemical Component Dictionary.

    Args:
        ccd_code: The three-letter code of the chemical component.
        ccd_mirror_path: Path to the root of the CCD mirror directory.
        **parse_ccd_cif_kwargs: Additional keyword arguments passed to parse_ccd_cif():
            coords: Type of coordinates to use ("model", "ideal_pdbx", "ideal_rdkit", or None).
                Defaults to "ideal_pdbx".
            add_properties: Whether to include RDKit-computed properties. Defaults to True.
            add_mapping: Whether to include external resource mappings, such as e.g. the ChEMBL ID.
                Defaults to False.

    Returns:
        AtomArray: The parsed atomic structure of the requested component.

    Example:
        >>> atom_array = get_ccd_component_from_mirror("ALA", coords="ideal_pdbx")
    """
    cif = pdbx.CIFFile.read(_get_ccd_path(ccd_code, ccd_mirror_path))
    atom_array = parse_ccd_cif(cif, **parse_ccd_cif_kwargs)
    return atom_array


@immutable_lru_cache(maxsize=200, deepcopy=True)
def atom_array_from_ccd_code(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieves and parses a component from the Chemical Component Dictionary.

    First attempts to retrieve the component from a local mirror if provided and the code exists there.
    Falls back to Biotite's built-in CCD if the code is not found in the local mirror or no mirror path is provided.

    Args:
        ccd_code: The three-letter code of the chemical component.
        ccd_mirror_path: Path to the root of the CCD mirror directory.
        **parse_ccd_cif_kwargs: Additional keyword arguments passed to parse_ccd_cif():
            coords: Type of coordinates to use ("model", "ideal_pdbx", "ideal_rdkit", or None).
                Defaults to "ideal_pdbx".
            add_properties: Whether to include RDKit-computed properties. Defaults to True.
            add_mapping: Whether to include external resource mappings. Defaults to False.

    Returns:
        struc.AtomArray: The parsed atomic structure of the requested component.

    Raises:
        ValueError: If the CCD code is not found in the local mirror or Biotite's built-in CCD.

    Example:
        >>> atom_array = atom_array_from_ccd_code("ALA")
    """
    if ccd_mirror_path and ccd_code in get_available_ccd_codes_in_mirror(ccd_mirror_path):
        return get_ccd_component_from_mirror(ccd_code, ccd_mirror_path, **parse_ccd_cif_kwargs)
    else:
        return get_ccd_component_from_biotite(ccd_code, **parse_ccd_cif_kwargs)


def _find_connected_components_after_removal(graph: nx.Graph, node_to_remove: int) -> list[list[int]]:
    """
    Identifies connected components that would form after removing a node from a graph.

    Args:
        - graph (nx.Graph): The input graph.
        - node_to_remove (int): The node to hypothetically remove.

    Returns:
        - list[list[int]]: List of lists containing node indices in each new component.
    """
    # Get the neighbors before removal
    neighbors = set(graph.neighbors(node_to_remove))
    if not neighbors:
        return []

    # Create subgraph without the node
    subgraph = graph.subgraph(set(graph.nodes) - {node_to_remove})

    # Use BFS from any neighbor to find components
    components = []
    unvisited = neighbors.copy()

    while unvisited:
        start = unvisited.pop()
        if start not in subgraph:
            continue

        # Find all nodes reachable from this neighbor
        component = list(nx.bfs_tree(subgraph, start))
        components.append(component)
        unvisited -= set(component)

    return components


@functools.cache
def get_chem_comp_leaving_atom_names(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[str, ...]]:
    """
    Computes the canonical leaving groups for a given CCD entry based on the PDBs annotation
    of leaving atoms.

    The returned dictionary maps the name of the atom to the names of the atoms that would
    become disconnected if the atom were removed.

    Example:
        >>> get_chem_comp_leaving_atom_names("ALA")
        {'N': ('H2',), 'C': ('OXT', 'HXT'), 'OXT': ('HXT',)}
    """
    # Skip CCD lookup for codes that shouldn't be matched against CCD (e.g., UNL, water-like codes)
    if ccd_code in DO_NOT_MATCH_CCD:
        if mode == "warn":
            logger.debug(f"Skipping CCD lookup for `{ccd_code}` as it's in DO_NOT_MATCH_CCD")
        return {}

    try:
        chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path)
    except (ValueError, AttributeError) as e:
        if mode == "warn":
            logger.warning(f"Failed to compute leaving groups for `{ccd_code}`: {e}")
        elif mode == "raise":
            raise ValueError(f"Failed to compute leaving groups for `{ccd_code}`: {e}") from e
        return {}

    if "is_leaving_atom" not in chem_comp.get_annotation_categories():
        if mode == "warn":
            logger.warning(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups, returning empty dictionary. "
                "Check if your CCD mirror is up to date."
            )
        elif mode == "raise":
            raise ValueError(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups. Check if your CCD mirror is up to date."
            )
        return {}

    # ... initialize output
    leaving_atom_names = defaultdict(list)

    # ... get relevant annotations
    is_leaving_atom = chem_comp.get_annotation("is_leaving_atom")
    atom_name = chem_comp.get_annotation("atom_name")
    element = chem_comp.get_annotation("element")

    # ... skip if no atoms are annotated as leaving atoms (majority of CCD entries)
    if not any(is_leaving_atom):
        return {}

    # ... compute the leaving groups based on the bond graph and annotation
    bond_graph = chem_comp.bonds.as_graph()
    for atom_idx in range(chem_comp.array_length()):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(bond_graph, atom_idx)

        # ... check if all atoms in the connected group are flagged as leaving atoms
        #     by the CCD entry
        for connected_group in connected_groups:
            heavy_atoms: list[int] = list(filter(lambda x: element[x] != "H", connected_group))
            is_leaving_group = (
                all(is_leaving_atom[heavy_atoms]) if len(heavy_atoms) > 0 else all(is_leaving_atom[connected_group])
            )

            if is_leaving_group:
                leaving_atom_names[atom_name[atom_idx]] += [atom_name[idx] for idx in connected_group]

    # ... turn leaving_atom_names into a dictionary of tuples
    leaving_atom_names = {k: tuple(v) for k, v in leaving_atom_names.items()}

    return leaving_atom_names


@functools.cache
def _chem_comp_type_dict() -> dict[str, str]:
    """Get a dictionary of all residue names and their corresponding chemical component types.

    Example:
        >>> _chem_comp_type_dict()["ALA"]
        'L-PEPTIDE LINKING'
    """
    ccd = struc.info.ccd.get_ccd()  # NOTE: biotite caches this internally
    chem_comp_ids = np.char.upper(ccd["chem_comp"]["id"].as_array())
    chem_comp_types = np.char.upper(ccd["chem_comp"]["type"].as_array())
    return dict(zip(chem_comp_ids, chem_comp_types, strict=True))


def get_chem_comp_type(ccd_code: str, mode: Literal["warn", "raise"] = "warn") -> str:
    """Get the chemical component type for a CCD code from the Chemical Component Dictionary (CCD).

    Can be combined with CHEM_TYPES from `atomworks.io_biotite.constants` to determine if a component is a
    protein, nucleic acid, or carbohydrate.

    Args:
        ccd_code (str): The CCD code for the component. E.g. `ALA` for alanine, `NAP` for N-acetyl-D-glucosamine.
        mode (Literal["warn", "raise"]): How to handle unknown chemical component types.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    chem_comp_type = _chem_comp_type_dict().get(ccd_code, None)

    # ... handle unknown chemical component types
    if not exists(chem_comp_type):
        if mode == "raise":
            # ... raise an error if we want to fail loudly
            raise ValueError(f"Chemical component type for `{ccd_code=}` not found in CCD.")
        elif mode == "warn":
            # ... otherwise set chemical component type to "other" - the equivalent of unknown.
            logger.info(f"Chemical component type for `{ccd_code=}` not found in CCD. Using 'other'.")
            chem_comp_type = "OTHER"

    return chem_comp_type


def get_chain_type_from_chem_comp_type(chem_comp_type: str) -> ChainType:
    """Get the ChainType enum corresponding to a chemical component type."""
    return ChainTypeInfo.CHEM_COMP_TYPE_TO_ENUM.get(chem_comp_type, ChainType.OTHER_POLYMER)


def get_chain_type_from_ccd_code(ccd_code: str) -> ChainType:
    """Get the ChainType enum corresponding to a CCD code."""
    return get_chain_type_from_chem_comp_type(get_chem_comp_type(ccd_code))


def get_unknown_ccd_code_for_chem_comp_type(chem_comp_type: str) -> str:
    """Get the CCD code for an unknown chemical component type."""
    if chem_comp_type in AA_LIKE_CHEM_TYPES:
        return UNKNOWN_AA
    elif chem_comp_type in DNA_LIKE_CHEM_TYPES:
        return UNKNOWN_DNA
    elif chem_comp_type in RNA_LIKE_CHEM_TYPES:
        return UNKNOWN_RNA
    else:
        return UNKNOWN_LIGAND


def get_std_to_alt_atom_name_map(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> dict[str, str]:
    """Get a map from standard atom names to alternative atom names."""
    chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path)
    return dict(zip(chem_comp.atom_name, chem_comp.alt_atom_id, strict=True))
