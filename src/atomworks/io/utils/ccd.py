import functools
import logging
import os
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import toolz

from atomworks.common import exists, immutable_lru_cache
from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    ALLOW_BIOTITE_CCD,
    CCD_MIRROR_PATH,
    CCD_POLYMERIZATION_ATOM_OVERRIDES,
    CHEM_TYPE_POLYMERIZATION_ATOMS,
    DEFAULT_CCD_ANNOTATIONS,
    DNA_LIKE_CHEM_TYPES,
    HYDROGEN_LIKE_SYMBOLS,
    MAX_CHEM_TYPE_LENGTH,
    NA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_LIGAND,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.utils.atom_array import _bonds_to_dict, annotate_hydrogens, remove_hydrogens

logger = logging.getLogger(__name__)


# ===== CUSTOM CCD REGISTRY =====
# Global thread-safe registry for custom CCD entries
_ccd_registry_lock = threading.RLock()
_ccd_registry: dict[str, struc.AtomArray] = {}


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
def _get_available_ccd_codes_in_mirror_cached(ccd_mirror_path_str: str) -> frozenset[str]:
    """Internal cached implementation. Do not call directly."""
    root = ccd_mirror_path_str  # Already normalized

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
                    return frozenset(codes) - {UNKNOWN_LIGAND}
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

    # Exclude UNL — it's a generic placeholder with no reliable CCD definition.
    # Unknown polymer residues (UNK, N, DN) are kept: they have stable CCD templates.
    return frozenset(codes) - {UNKNOWN_LIGAND}


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
    # Normalize path and delegate to cached implementation
    return _get_available_ccd_codes_in_mirror_cached(os.fspath(ccd_mirror_path))


@functools.cache
def get_available_ccd_codes_in_biotite() -> frozenset[str]:
    """Set of all CCD codes available in Biotite's built-in Chemical Component Dictionary."""

    # Exclude UNL — it's a generic placeholder with no reliable CCD definition.
    # Unknown polymer residues (UNK, N, DN) are kept: they have stable CCD templates.
    return frozenset(struc.info.ccd.get_ccd()["chem_comp"]["id"].as_array()) - {UNKNOWN_LIGAND}


@functools.cache
def _get_standard_ccd_codes_cached(ccd_mirror_path_str: str) -> frozenset[str]:
    """Internal cached implementation. Do not call directly."""
    if ccd_mirror_path_str:
        # Local mirror configured — use ONLY mirror codes
        codes = get_available_ccd_codes_in_mirror(ccd_mirror_path_str)
    elif ALLOW_BIOTITE_CCD:
        # No local mirror — fall back to biotite if allowed
        codes = get_available_ccd_codes_in_biotite()
    else:
        # No mirror, biotite not allowed
        return frozenset()

    return codes


def get_standard_ccd_codes(ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH) -> frozenset[str]:
    """Returns standard CCD codes (from mirror + Biotite). Cached."""
    # Normalize path (use empty string for None)
    cache_key = os.fspath(ccd_mirror_path) if ccd_mirror_path else ""
    return _get_standard_ccd_codes_cached(cache_key)


def get_available_ccd_codes(ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH) -> frozenset[str]:
    """Returns all available CCD codes including registry entries."""
    # Get (cached) standard codes from mirror (+ Biotite if allowed)
    standard_codes = get_standard_ccd_codes(ccd_mirror_path)

    # ... add custom registry codes
    with _ccd_registry_lock:
        registry_codes = frozenset(_ccd_registry.keys())

    return standard_codes if registry_codes.issubset(standard_codes) else standard_codes | registry_codes


def _standard_ccd_only_cache(cache_decorator: Callable) -> Callable:
    """Wrap a cache decorator to skip caching for non-standard or registry-overridden codes.

    Registry codes are checked first and never cached, so an override of a real code is
    never shadowed by a bundled template that was resolved (and cached) earlier.
    """

    def decorator(func: Callable) -> Callable:
        cached = cache_decorator(func)

        @functools.wraps(func)
        def wrapper(ccd_code: str, *args: Any, **kwargs: Any) -> Any:
            code_upper = ccd_code.upper()
            if code_upper in _ccd_registry or code_upper not in get_standard_ccd_codes():
                return func(ccd_code, *args, **kwargs)
            return cached(ccd_code, *args, **kwargs)

        wrapper.cache_clear = getattr(cached, "cache_clear", None)
        wrapper.cache_info = getattr(cached, "cache_info", None)
        return wrapper

    return decorator


def get_ccd_component_from_biotite(ccd_code: str, **parse_ccd_cif_kwargs) -> struc.AtomArray:
    """Retrieve an isolated copy of a cached component from Biotite's built-in CCD.

    Args:
        - ccd_code (str): The three-letter code of the chemical component to retrieve.

    Returns:
        - AtomArray: The atomic structure of the requested component.
    """
    if isinstance(parse_ccd_cif_kwargs.get("coords"), list):
        parse_ccd_cif_kwargs["coords"] = tuple(parse_ccd_cif_kwargs["coords"])
    return _get_ccd_component_from_biotite_cached(ccd_code, **parse_ccd_cif_kwargs)


@immutable_lru_cache(maxsize=1000)
def _get_ccd_component_from_biotite_cached(ccd_code: str, **parse_ccd_cif_kwargs) -> struc.AtomArray:
    """Parse bundled components once per option set; return a deep copy per lookup."""
    try:
        block = _filter_biotite_ccd_for_ccd_code(ccd_code)
        atom_array = parse_ccd_cif(block, **parse_ccd_cif_kwargs)
        return atom_array
    except KeyError:
        raise ValueError(f"No atom information found for residue '{ccd_code}' in Biotite's CCD") from None


def check_ccd_codes_are_available(
    ccd_codes: Iterable[str], ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> bool:
    """Checks if the provided CCD codes are available in the current environment"""
    available_ccds = get_available_ccd_codes(ccd_mirror_path)
    invalid_ccds = set(ccd_codes) - available_ccds
    if invalid_ccds:
        if mode == "warn":
            logger.warning(f"The following CCD codes were not found in the current environment: {invalid_ccds}")
        elif mode == "raise":
            raise ValueError(f"The following CCD codes were not found in the current environment: {invalid_ccds}")
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


@functools.cache
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


def _build_ccd_template_atom_array(
    comp_id: str,
    chem_comp_type: str,
    atom_names: np.ndarray,
    elements: np.ndarray,
    *,
    charges: np.ndarray | None = None,
    is_aromatic: np.ndarray | None = None,
    is_leaving_atom: np.ndarray | None = None,
    coords: np.ndarray | None = None,
) -> struc.AtomArray:
    """Build an AtomArray with the core annotations shared across CCD template construction sites.

    Sets ``res_name``, ``atom_name``, ``element``, ``charge``, ``res_id``,
    ``hetero``, and ``chem_comp_type``. Callers are responsible for setting
    ``bonds`` and any additional CCD-specific annotations (e.g. ``stereo``).
    """
    n = len(atom_names)
    atoms = struc.AtomArray(n)
    atoms.set_annotation("res_name", np.full(n, comp_id))
    atoms.set_annotation("atom_name", atom_names)
    atoms.set_annotation("element", elements)
    atoms.set_annotation("charge", charges if charges is not None else np.zeros(n, dtype=np.int8))
    atoms.set_annotation("res_id", np.full(n, 1))
    atoms.set_annotation("hetero", np.full(n, comp_id not in struc.info.atoms.NON_HETERO_RESIDUES))
    atoms.set_annotation("chem_comp_type", np.full(n, chem_comp_type, dtype=f"<U{MAX_CHEM_TYPE_LENGTH}"))

    if is_aromatic is not None:
        atoms.set_annotation("is_aromatic", is_aromatic)
    if is_leaving_atom is not None:
        atoms.set_annotation("is_leaving_atom", is_leaving_atom)

    atoms.coord[:] = coords if coords is not None else np.nan
    return atoms


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
    # Structure CIFs use `type`; CCD CIFs use `pdbx_type`. Tolerate either (or neither).
    type_field = next((c for c in ("type", "pdbx_type") if c in metadata), None)
    chem_comp_type = metadata[type_field].as_item().upper() if type_field is not None else "OTHER"

    # Extract atom specific information
    atom_data = block.get("chem_comp_atom")

    # Some CCD entries have no atom data. Return an empty AtomArray carrying
    # the chem_comp_type annotation; `get_chem_comp_type` falls through on
    # len==0, and anything else that expected atoms will fail loudly.
    if atom_data is None:
        atoms = _build_ccd_template_atom_array(
            comp_id=ccd_code,
            chem_comp_type=chem_comp_type,
            atom_names=np.array([], dtype="<U6"),
            elements=np.array([], dtype="<U2"),
        )
        atoms.bonds = struc.BondList(0)
        return atoms

    n_atoms = atom_data.row_count

    def _get_str(field_name: str, default: str = "") -> np.ndarray:
        """Get string field or return default."""
        field = atom_data.get(field_name)
        return field.as_array(str).copy() if field is not None else np.full(n_atoms, default, dtype=str)

    def _get_bool(field_name: str) -> np.ndarray:
        """Get boolean field (Y/N) or return False."""
        field = atom_data.get(field_name)
        return np.where(field.as_array(str) == "Y", True, False) if field is not None else np.full(n_atoms, False)

    # NOTE: We must ``.copy()`` the arrays to ensure they are mutable and not views into the cached CIFBlock data
    atoms = _build_ccd_template_atom_array(
        comp_id=ccd_code,
        chem_comp_type=chem_comp_type,
        atom_names=atom_data.get("atom_id").as_array(str).copy(),
        elements=atom_data.get("type_symbol").as_array(str).copy(),
        charges=atom_data.get("charge").as_array(np.int8).copy(),
        is_aromatic=_get_bool("pdbx_aromatic_flag"),
        is_leaving_atom=_get_bool("pdbx_leaving_atom_flag"),
    )

    # Misc. additional annotations that are not essential but can be convenient
    atoms.set_annotation("alt_atom_id", _get_str("alt_atom_id"))
    atoms.set_annotation("stereo", _get_str("pdbx_stereo_config"))
    atoms.set_annotation("is_backbone_atom", _get_bool("pdbx_backbone_atom_flag"))
    atoms.set_annotation("is_n_terminal_atom", _get_bool("pdbx_n_terminal_atom_flag"))
    atoms.set_annotation("is_c_terminal_atom", _get_bool("pdbx_c_terminal_atom_flag"))

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
        if len(atoms) != 1:
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
    except KeyError as e:
        raise KeyError(
            f"Failed to extract bond data for `{ccd_code}`: missing key {e}. "
            f"Required fields are: comp_id, atom_id_1, atom_id_2, value_order, pdbx_aromatic_flag"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error parsing bond data for `{ccd_code}`: {e!s}") from e

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


@immutable_lru_cache(maxsize=20000, copy_func=lambda x: x.copy())
def get_ccd_component_from_mirror(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieves and parses a component from a local mirror of the Chemical Component Dictionary.

    Args:
        ccd_code: The three-letter code of the chemical component.
        ccd_mirror_path: Path to the root of the CCD mirror directory.
        **parse_ccd_cif_kwargs: Additional keyword arguments passed to ``parse_ccd_cif``.

            * ``coords``: Type of coordinates to use ("model", "ideal_pdbx", "ideal_rdkit", or None).
                Defaults to "ideal_pdbx".
            * ``add_properties``: Whether to include RDKit-computed properties. Defaults to True.
            * ``add_mapping``: Whether to include external resource mappings, such as e.g. the ChEMBL ID.
                Defaults to False.

    Returns:
        AtomArray: The parsed atomic structure of the requested component.

    Example:
        >>> atom_array = get_ccd_component_from_mirror("ALA", coords="ideal_pdbx")
    """
    cif = pdbx.CIFFile.read(_get_ccd_path(ccd_code, ccd_mirror_path))
    atom_array = parse_ccd_cif(cif, **parse_ccd_cif_kwargs)
    return atom_array


def atom_array_from_ccd_code(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieves a component from the Chemical Component Dictionary.

    Checks custom registry first (overriding standard CCD), then uses the local mirror
    if configured (never falling back to biotite), or biotite if no mirror and allowed.

    Args:
        ccd_code: The chemical component code
        ccd_mirror_path: Path to local CCD mirror
        **parse_ccd_cif_kwargs: Additional args passed to parse_ccd_cif()

    Returns:
        AtomArray of the requested component

    Note:
        Registry entries override standard CCD definitions. Use :py:func:`register_custom_ccd_entry`
        to provide custom templates that will be used instead of standard CCD data.
    """
    # Check registry first
    code_upper = ccd_code.upper()
    if code_upper in _ccd_registry:
        entry = _ccd_registry[code_upper].copy()
        # Ensure we honor an explicit ``coords=None``
        if "coords" in parse_ccd_cif_kwargs and parse_ccd_cif_kwargs["coords"] is None:
            entry.coord[:] = np.nan
        return entry

    return atom_array_from_bundled_ccd_code(ccd_code, ccd_mirror_path, **parse_ccd_cif_kwargs)


def atom_array_from_bundled_ccd_code(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieve a component from the bundled CCD, bypassing the custom registry.

    Args:
        ccd_code: The chemical component code.
        ccd_mirror_path: Path to local CCD mirror.
        **parse_ccd_cif_kwargs: Additional args passed to :py:func:`parse_ccd_cif`.
    """
    # Normalize path: None → "" for consistent caching downstream
    ccd_mirror_path = str(ccd_mirror_path or "")

    if ccd_mirror_path:
        # Local mirror configured — use ONLY mirror, no biotite fallback
        if ccd_code in get_available_ccd_codes_in_mirror(ccd_mirror_path):
            return get_ccd_component_from_mirror(ccd_code, ccd_mirror_path, **parse_ccd_cif_kwargs)
        else:
            raise ValueError(
                f"Cannot load CCD component '{ccd_code}': not found in local mirror at {ccd_mirror_path!r}. "
                f"Component may not exist in CCD, or mirror may be outdated."
            )
    elif ALLOW_BIOTITE_CCD:
        # No mirror — use biotite
        return get_ccd_component_from_biotite(ccd_code, **parse_ccd_cif_kwargs)
    else:
        raise ValueError(
            f"Cannot load CCD component '{ccd_code}': no local CCD mirror configured. "
            f"Set CCD_MIRROR_PATH environment variable or set ALLOW_BIOTITE_CCD=1."
        )


def _get_ccd_block(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> pdbx.CIFBlock | None:
    """Resolve a CCD code to a CIFBlock, trying mirror first then biotite.

    Args:
        ccd_code: The CCD code.
        ccd_mirror_path: Path to mirror, or empty string for biotite fallback.
    """
    if ccd_mirror_path:
        try:
            cif = pdbx.CIFFile.read(_get_ccd_path(ccd_code, ccd_mirror_path))
            return pdbx.convert._get_block(cif, None)
        except FileNotFoundError:
            return None
    if ALLOW_BIOTITE_CCD:
        try:
            return _filter_biotite_ccd_for_ccd_code(ccd_code)
        except KeyError:
            return None
    return None


@_standard_ccd_only_cache(functools.lru_cache(maxsize=20_000))
def _read_chem_comp_field(ccd_code: str, field: str, ccd_mirror_path: str) -> str | None:
    """Read a single field from the ``chem_comp`` category of a CCD entry."""
    block = _get_ccd_block(ccd_code, ccd_mirror_path)
    if block is None:
        return None
    try:
        value = block["chem_comp"][field].as_item()
        return value.upper() if value not in ("?", ".", "") else None
    except KeyError:
        return None


def _infer_chem_comp_type_from_atom_names(atom_names: set[str], *, warn: bool = False) -> str:
    """Infer chem_comp_type from atom names.

    Args:
        atom_names: Atom names to use for inference.
        warn: If ``True``, emit a warning when a polymer type is inferred.
    """
    # Default is non-polymer
    chem_comp_type = "NON-POLYMER"

    # Peptide backbone: N, CA, C
    if {"N", "CA", "C"}.issubset(atom_names):
        chem_comp_type = "L-PEPTIDE LINKING"

    # Nucleotide backbone: O3', P
    elif {"O3'", "P"}.issubset(atom_names):
        if {"O2'"}.issubset(atom_names):
            chem_comp_type = "RNA LINKING"
        else:
            chem_comp_type = "DNA LINKING"

    if warn and chem_comp_type != "NON-POLYMER":
        preview = sorted(str(a) for a in atom_names)
        shown = preview[:8] + [f"…(+{len(preview) - 8} more)"] if len(preview) > 8 else preview
        logger.warning(
            f"Inferred polymer type {chem_comp_type} from atom names {shown}. "
            "This is a fallible process! To avoid this warning, set the "
            "`chem_comp_type` annotation explicitly on your custom CCD entry."
        )

    return chem_comp_type


def get_chem_comp_type(
    ccd_code: str,
    atom_names: set[str] | None = None,
    mode: Literal["warn", "raise"] = "warn",
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> str:
    """Get the chemical component type for a CCD code.

    Routes through :func:`atom_array_from_ccd_code` so the custom CCD registry
    is the single interception point.  Falls back to inference from ``atom_names``
    when the code is unknown.

    Args:
        ccd_code: The CCD code for the component. E.g. ``"ALA"`` for alanine.
        atom_names: Atom names used to infer the type when the code is not in the CCD.
        mode: How to handle unknown chemical component types.
        ccd_mirror_path: Path to the local CCD mirror directory.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    try:
        arr = atom_array_from_ccd_code(ccd_code, ccd_mirror_path, coords=None)
        if len(arr) > 0 and "chem_comp_type" in arr.get_annotation_categories():
            return arr.chem_comp_type[0]
    except (ValueError, KeyError):
        pass

    # Fallback: read just the type field without loading the full atom array
    chem_comp_type = _read_chem_comp_field(ccd_code, "type", str(ccd_mirror_path or ""))
    if exists(chem_comp_type):
        return chem_comp_type

    if exists(atom_names):
        return _infer_chem_comp_type_from_atom_names(atom_names, warn=True)
    if mode == "raise":
        raise ValueError(f"Chemical component type for `{ccd_code=}` not found in CCD or custom CCD.")
    logger.info(f"Chemical component type for `{ccd_code=}` not found in CCD or custom CCD. Using 'OTHER'.")
    return "OTHER"


def get_parent_comp_id(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> str | None:
    """Return the canonical parent residue for a modified CCD code, or ``None``.

    Uses the local CCD mirror when configured, falling back to biotite's built-in CCD.
    """
    return _read_chem_comp_field(ccd_code.upper(), "mon_nstd_parent_comp_id", str(ccd_mirror_path or ""))


def is_parent_strict_atom_subset(
    child_code: str,
    parent_code: str,
    ccd_mirror_path: str = "",
    hydrogen_policy: Literal["keep", "remove"] = "remove",
) -> bool:
    """Check if the parent template's heavy-atom names are a strict subset of the child's.

    Args:
        child_code: Modified residue CCD code (e.g. ``"ASB"``).
        parent_code: Parent residue CCD code (e.g. ``"ASP"``).
        ccd_mirror_path: Path to CCD mirror, or empty string for Biotite's built-in.
        hydrogen_policy: Hydrogen handling for templates.
    """
    try:
        child_template = _get_base_ccd_template(child_code, ccd_mirror_path, hydrogen_policy)
        parent_template = _get_base_ccd_template(parent_code, ccd_mirror_path, hydrogen_policy)
    except Exception:
        return False

    child_atoms = set(child_template.atom_name)
    parent_atoms = set(parent_template.atom_name)
    return parent_atoms < child_atoms  # strict subset


def get_parent_template_atom_names(
    parent_code: str,
    ccd_mirror_path: str = "",
    hydrogen_policy: Literal["keep", "remove"] = "remove",
) -> set[str]:
    """Return the set of atom names from a parent CCD template."""
    template = _get_base_ccd_template(parent_code, ccd_mirror_path, hydrogen_policy)
    return set(template.atom_name)


def get_chain_type_from_chem_comp_type(chem_comp_type: str) -> ChainType:
    """Get the ChainType enum corresponding to a chemical component type."""
    return ChainTypeInfo.CHEM_COMP_TYPE_TO_ENUM.get(chem_comp_type, ChainType.OTHER_POLYMER)


def get_chain_type_from_ccd_code(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> ChainType:
    """Get the ChainType enum corresponding to a CCD code."""
    return get_chain_type_from_chem_comp_type(get_chem_comp_type(ccd_code, ccd_mirror_path=ccd_mirror_path))


def _derive_polymerization_atoms_from_template(
    template: struc.AtomArray, expected_elements: tuple[str, str]
) -> tuple[str | None, str | None]:
    """Recover a component's polymerization atoms as those bonded to the leaving atoms."""
    if template.bonds is None or "is_leaving_atom" not in template.get_annotation_categories():
        return None, None
    is_leaving = template.is_leaving_atom
    if not is_leaving.any():
        return None, None

    is_heavy = ~np.isin(template.element, list(HYDROGEN_LIKE_SYMBOLS))

    i, j = template.bonds.as_array()[:, 0], template.bonds.as_array()[:, 1]
    neighbours = np.concatenate([j[is_leaving[i] & ~is_leaving[j]], i[is_leaving[j] & ~is_leaving[i]]])
    candidates = np.unique(neighbours[is_heavy[neighbours]])

    sides = (candidates[template.element[candidates] == element] for element in expected_elements)
    exiting, entering = (str(template.atom_name[s[0]]) if len(s) == 1 else None for s in sides)
    return exiting, entering


@functools.lru_cache(maxsize=1024)
def _get_polymerization_atoms_cached(res_name: str, chain_type: ChainType | None) -> tuple[str | None, str | None]:
    """Cached lookup for canonical (non-registry) residues only."""
    cct = get_chem_comp_type(res_name)
    canonical = CHEM_TYPE_POLYMERIZATION_ATOMS.get(cct)
    if canonical is None:
        canonical = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(get_chain_type_from_chem_comp_type(cct))
    if canonical is None:
        # Use the chain's convention for non-polymer caps, e.g. the terminal NH2 in PDB 3N95.
        canonical = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(chain_type)
    if canonical is None:
        return None, None

    try:
        template = atom_array_from_ccd_code(res_name, coords=None)
    except (ValueError, KeyError):
        # Type came from a component we cannot load -- nothing to cross-check against.
        return canonical

    present = set(template.atom_name)
    if set(canonical).issubset(present):
        return canonical

    # ~15% of polymeric CCD entries do not carry the canonical atom names for their type: the GFP
    # chromophore ``CRO`` is ``L-PEPTIDE LINKING`` but calls its backbone ``N1``/``C3``. Only such
    # components consult the overrides, so a custom component registered under an overridden code
    # still resolves by its own atom names.
    override = CCD_POLYMERIZATION_ATOM_OVERRIDES.get(res_name.upper())
    if override is not None:
        return override

    derived = _derive_polymerization_atoms_from_template(template, struc.infer_elements(canonical))
    return tuple(
        found if found is not None else (canon if canon in present else None)
        for canon, found in zip(canonical, derived, strict=True)
    )


def get_polymerization_atoms(res_name: str, chain_type: ChainType | None = None) -> tuple[str | None, str | None]:
    """Return candidate ``(leaving_atom, entering_atom)`` sites for a polymer bond.

    Looks up the canonical atoms for the chemical component type via :py:func:`get_chem_comp_type`
    against ``CHEM_TYPE_POLYMERIZATION_ATOMS`` (falling back to
    ``ChainTypeInfo.ATOMS_AT_POLYMER_BOND``), and returns them when the component actually carries
    those atom names. A component that does not carry them falls back to
    ``CCD_POLYMERIZATION_ATOM_OVERRIDES``, and then to
    :py:func:`_derive_polymerization_atoms_from_template`. Each side is resolved independently and
    may be ``None``. For caps, ``chain_type`` supplies candidate sites (``NH2`` gives ``(None, "N")``),
    but does not establish that a bond exists.
    """
    # Skip cache for custom registry entries (mutable state)
    if res_name.upper() in _ccd_registry:
        return _get_polymerization_atoms_cached.__wrapped__(res_name, chain_type)

    # Use cached lookup for canonical CCD entries (immutable state)
    return _get_polymerization_atoms_cached(res_name, chain_type)


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
    chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path, coords=None)
    return dict(zip(chem_comp.atom_name, chem_comp.alt_atom_id, strict=True))


def get_ccd_descriptors(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> dict[str, str]:
    """Extract SMILES and InChI descriptors from a CCD entry.

    Parses the ``_pdbx_chem_comp_descriptor`` category to retrieve molecular
    descriptors. Available descriptor types depend on the CCD entry.

    Args:
        ccd_code: The CCD code for the component.
        ccd_mirror_path: Path to the local CCD mirror.

    Returns:
        Dictionary with descriptor types as keys and descriptor values as values.
        Common keys include: "SMILES", "SMILES_CANONICAL", "InChI", "InChIKey".
        Returns empty dict if the CCD code is not found or has no descriptors.

    Example:
        >>> get_ccd_descriptors("ATP")
        {'SMILES_CANONICAL': 'Nc1ncnc2c1ncn2...', 'InChI': 'InChI=1S/...', ...}
    """
    block = _get_ccd_block(ccd_code, str(ccd_mirror_path or ""))
    if block is None:
        return {}

    descriptor_data = block.get("pdbx_chem_comp_descriptor")
    if descriptor_data is None:
        return {}

    # Extract type and descriptor columns
    try:
        desc_types = descriptor_data["type"].as_array(str)
        descriptors = descriptor_data["descriptor"].as_array(str)
    except (KeyError, AttributeError):
        return {}

    # Build the result dictionary
    result = {}
    for desc_type, descriptor in zip(desc_types, descriptors, strict=True):
        # Clean up the descriptor type to create a simple key
        desc_type_upper = desc_type.upper()
        if "SMILES" in desc_type_upper:
            if "CANONICAL" in desc_type_upper:
                result["SMILES_CANONICAL"] = descriptor
            else:
                result["SMILES"] = descriptor
        elif "INCHI" in desc_type_upper:
            if "KEY" in desc_type_upper:
                result["InChIKey"] = descriptor
            else:
                result["InChI"] = descriptor

    return result


# ===== CCD ANNOTATION TRANSFER =====


def add_annotations_from_ccd(
    atom_array: struc.AtomArray,
    annotations: list[str] | None = None,
    overwrite: dict[str, bool] | bool = False,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    hydrogen_policy: Literal["keep", "remove"] = "keep",
    on_mismatch: Literal["ignore", "error_heavy", "error"] = "ignore",
    match_alt_atom_ids: bool = False,
) -> struc.AtomArray:
    """Add CCD structural annotations to atoms in an existing AtomArray.

    Matches atoms to CCD templates by residue and copies specified annotations.

    Args:
        atom_array: Structure to annotate.
        annotations: List of annotation names to add. If ``None``, adds all core
            structural annotations: ``["charge", "stereo", "is_aromatic",
            "is_leaving_atom", "is_backbone_atom", "is_n_terminal_atom",
            "is_c_terminal_atom", "nhyd"]``.
        overwrite: Control overwriting of existing annotations. If bool, applies
            to all. If dict, maps annotation name to overwrite policy.
            Example: ``{"charge": False, "stereo": True}`` keeps existing charges
            but overwrites stereochemistry. Defaults to ``False``.
        ccd_mirror_path: Path to local CCD mirror.
        hydrogen_policy: Whether to use the ``"keep"`` or ``"remove"`` CCD template.
            When ``"remove"``, the template has ``nhyd`` set on heavy atoms so that
            implicit H counts are copied to ``atom_array`` before bond inference.
        on_mismatch: How to handle residue atoms whose names are not in the CCD
            template. One of:

            - ``"ignore"`` (default): silently skip unmatched atoms (debug log only).
            - ``"error_heavy"``: raise ``ValueError`` if any unmatched atom is a heavy atom
            - ``"error"``: raise ``ValueError`` on any unmatched atom.
        match_alt_atom_ids: If True, use the template's ``alt_atom_id`` column for
            the entire residue when it matches more heavy atoms than ``atom_name``.
            Ties use standard names. Defaults to ``False``.

    Returns:
        AtomArray with CCD annotations added (modified in-place).

    Raises:
        ValueError: If ``on_mismatch`` is ``"error"`` or ``"error_heavy"`` and
            unmatched atoms are found.
    """
    # Default annotations
    if annotations is None:
        annotations = list(DEFAULT_CCD_ANNOTATIONS)

    # Normalize overwrite to dict
    if isinstance(overwrite, bool):
        overwrite_dict = {annot: overwrite for annot in annotations}
    else:
        overwrite_dict = overwrite

    # Get available CCD codes for proactive checking
    available_ccds = get_available_ccd_codes(str(ccd_mirror_path or ""))

    # Use start/stop indices instead of boolean masks for O(1) slicing
    _res_start_stops = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_stops = _res_start_stops[:-1], _res_start_stops[1:]

    # Missing value checkers for each annotation
    def _make_checker(annot: str) -> Callable:
        """Create a missing value checker function for the given annotation."""
        if annot == "charge":
            return lambda v: v == 0
        elif annot in ["stereo", "alt_atom_id"]:
            return lambda v: (v == "") | (v == ".") | (v == "?")
        elif annot.startswith("is_"):
            return lambda v: v == False  # noqa: E712
        elif annot == "nhyd":
            return lambda v: v == 0
        else:
            # Handle floats with NaN
            return lambda v: (np.isnan(v) if np.issubdtype(v.dtype, np.floating) else np.zeros(len(v), dtype=bool))

    missing_value_checkers = {annot: _make_checker(annot) for annot in annotations}
    present_annotations = atom_array.get_annotation_categories()
    annotation_arrays = {
        annot: atom_array.get_annotation(annot) for annot in annotations if annot in present_annotations
    }
    # Registry overrides remain reusable within this call without surviving registry changes.
    get_template = functools.cache(_get_base_ccd_template)
    matching_templates = {}

    # Process each unique residue
    for res_start, res_stop in zip(_res_starts, _res_stops, strict=False):
        # Get residue metadata
        chain_id = atom_array.chain_id[res_start]
        res_name = atom_array.res_name[res_start]
        res_id = atom_array.res_id[res_start]

        # Check CCD availability
        if res_name not in available_ccds:
            logger.debug(f"CCD {res_name} not available, skipping annotation for chain {chain_id}, res_id {res_id}")
            continue

        # CCD is available - try to get template
        # (Some CCD files may be malformed, so catch parsing errors)
        try:
            template = get_template(res_name, str(ccd_mirror_path or ""), hydrogen_policy=hydrogen_policy)
        except (AttributeError, ValueError, KeyError) as e:
            logger.debug(f"CCD parsing failed for {res_name} (chain {chain_id}, res_id {res_id}): {e}")
            continue

        # Get atom names for this residue (direct slicing instead of boolean mask)
        residue_atom_names = atom_array.atom_name[res_start:res_stop]

        # Repeated residues share a name layout even when partial or reordered.
        heavy_mask = (
            ~np.isin(atom_array.element[res_start:res_stop], HYDROGEN_LIKE_SYMBOLS) if match_alt_atom_ids else None
        )
        layout = res_name, tuple(residue_atom_names), tuple(heavy_mask) if heavy_mask is not None else ()
        if layout not in matching_templates:
            # Select one naming convention for the entire residue using heavy atoms.
            template_names = template.atom_name
            if match_alt_atom_ids and "alt_atom_id" in template.get_annotation_categories():
                heavy_names = residue_atom_names[heavy_mask]
                n_std = np.isin(heavy_names, template.atom_name).sum()
                n_alt = np.isin(heavy_names, template.alt_atom_id).sum()
                if n_alt > n_std:
                    template_names = template.alt_atom_id

            if len(template) == len(residue_atom_names) and np.array_equal(template_names, residue_atom_names):
                template_idxs = residue_idxs = np.arange(len(template), dtype=np.intp)
            else:
                _, template_idxs, residue_idxs = np.intersect1d(
                    template_names, residue_atom_names, assume_unique=True, return_indices=True
                )
            template_annotations = template.get_annotation_categories()
            matching_templates[layout] = (
                residue_idxs,
                [
                    (annot, template.get_annotation(annot)[template_idxs])
                    for annot in annotations
                    if annot in template_annotations
                ],
            )
        residue_idxs, matched_annotations = matching_templates[layout]

        # Handle atoms that don't match the template according to ``on_mismatch``.
        if len(residue_idxs) != len(residue_atom_names):
            unmatched_local_mask = np.ones(len(residue_atom_names), dtype=bool)
            unmatched_local_mask[residue_idxs] = False
            unmatched_names = set(residue_atom_names[unmatched_local_mask])

            if on_mismatch == "error":
                raise ValueError(
                    f"Residue {res_name} (chain {chain_id}, res_id {res_id}) has atoms "
                    f"not in CCD template: {unmatched_names}."
                )
            if on_mismatch == "error_heavy":
                residue_elements = atom_array.element[res_start:res_stop]
                heavy_unmatched_mask = unmatched_local_mask & ~np.isin(residue_elements, HYDROGEN_LIKE_SYMBOLS)
                if heavy_unmatched_mask.any():
                    heavy_unmatched = set(residue_atom_names[heavy_unmatched_mask])
                    raise ValueError(
                        f"Residue {res_name} (chain {chain_id}, res_id {res_id}) has "
                        f"heavy atoms not in CCD template: {heavy_unmatched}."
                    )

            logger.debug(
                f"Residue {res_name} (chain {chain_id}, res_id {res_id}) has atoms not in CCD template: {unmatched_names}. "
                f"These atoms will not receive CCD annotations. This may indicate non-standard atom naming."
            )

        # Get global indices of matched atoms (direct offset instead of mask indexing)
        residue_global_idxs = res_start + residue_idxs

        # Copy annotations for matched atoms only
        for annot, template_values in matched_annotations:
            # Ensure annotation exists in atom_array
            if annot not in annotation_arrays:
                # Create annotation with default values
                if annot in ["stereo", "alt_atom_id"]:
                    default = np.full(len(atom_array), "", dtype="<U1")
                elif annot.startswith("is_"):
                    default = np.zeros(len(atom_array), dtype=bool)
                elif annot in ("nhyd", "charge"):
                    default = np.zeros(len(atom_array), dtype=np.int8)
                else:
                    default = np.zeros(len(atom_array), dtype=np.float32)
                atom_array.set_annotation(annot, default)
                annotation_arrays[annot] = atom_array.get_annotation(annot)

            # Determine if we should overwrite
            should_overwrite = overwrite_dict.get(annot, False)
            values = annotation_arrays[annot]

            if should_overwrite:
                # Overwrite matched atoms in this residue
                values[residue_global_idxs] = template_values
            else:
                # Supplement only where missing (type-aware check)
                current_values = values[residue_global_idxs]
                missing_mask = missing_value_checkers[annot](current_values)
                values[residue_global_idxs[missing_mask]] = template_values[missing_mask]

    return atom_array


# ===== CUSTOM CCD REGISTRY API =====


def _atoms_and_bonds_match(a: struc.AtomArray, b: struc.AtomArray) -> bool:
    """Return True if ``a`` and ``b`` have the same atom names and bonds."""
    if sorted(a.atom_name.tolist()) != sorted(b.atom_name.tolist()):
        return False
    return _bonds_to_dict(a) == _bonds_to_dict(b)


def _fill_ideal_coords_from_bundled_ccd(code: str, entry: struc.AtomArray) -> None:
    """Fill NaN coordinates on a registry ``entry`` in place from the bundled CCD, matched by atom name.

    Genuine custom ligands (no bundled definition, or atoms not a subset) are left untouched.
    """
    missing = np.isnan(entry.coord).any(axis=-1)
    if not missing.any():
        return
    try:
        bundled = atom_array_from_bundled_ccd_code(code)
    except (ValueError, KeyError):
        return
    name_to_coord = dict(zip(bundled.atom_name, bundled.coord, strict=False))
    if not set(map(str, entry.atom_name)).issubset(name_to_coord):
        return
    for i in np.where(missing)[0]:
        entry.coord[i] = name_to_coord[entry.atom_name[i]]


def register_custom_ccd_entry(code: str, atom_array: struc.AtomArray) -> None:
    """Register a custom CCD entry to override/supplement standard CCD lookups."""
    code = code.upper()

    # Copy first so any annotation we add below never leaks back to the caller's array.
    entry = atom_array.copy()

    # Backfill missing ideal coords from the CCD so we ensure all entries can reference geometry
    _fill_ideal_coords_from_bundled_ccd(code, entry)

    # Ensure that the AtomArray contains the chem_comp_type annotation
    if "chem_comp_type" not in entry.get_annotation_categories():
        logger.warning(
            f"Custom CCD entry '{code}' is missing 'chem_comp_type' annotation. "
            "Inferring from atom names, but this may be inaccurate. "
            "To avoid this warning, set the 'chem_comp_type' annotation explicitly on your custom CCD entry."
        )
        inferred_type = _infer_chem_comp_type_from_atom_names(set(entry.atom_name))
        entry.set_annotation(
            "chem_comp_type", np.full(entry.array_length(), inferred_type, dtype=f"<U{MAX_CHEM_TYPE_LENGTH}")
        )

    # Warn when the custom entry is overwriting AND differs from an existing standard CCD entry.
    if code in get_standard_ccd_codes():
        try:
            differs = not _atoms_and_bonds_match(entry, atom_array_from_ccd_code(code))
        except Exception:
            differs = True
        if differs:
            logger.debug(
                f"Custom CCD entry '{code}' overrides standard CCD definition. "
                "Lookups will use custom entry instead of standard CCD."
            )

    # Check if overwriting existing registry entry
    with _ccd_registry_lock:
        if code in _ccd_registry:
            logger.warning(
                f"Overwriting existing custom CCD entry for '{code}' in registry. " "Previous entry will be replaced."
            )
        _ccd_registry[code] = entry


def register_custom_ccd_entries(entries: dict[str, struc.AtomArray]) -> None:
    """Register multiple custom CCD entries at once."""
    for code, atom_array in entries.items():
        register_custom_ccd_entry(code, atom_array)


def unregister_custom_ccd_entry(code: str) -> bool:
    """Unregister a custom CCD entry. Returns True if entry existed."""
    code = code.upper()

    with _ccd_registry_lock:
        was_present = code in _ccd_registry
        _ccd_registry.pop(code, None)

    if was_present:
        logger.info(f"Unregistered custom CCD entry: {code}")

    return was_present


def get_custom_ccd_entries() -> frozenset[str]:
    """Get set of currently registered custom CCD codes."""
    with _ccd_registry_lock:
        return frozenset(_ccd_registry.keys())


def snapshot_custom_ccd_registry() -> dict[str, struc.AtomArray]:
    """Return a deep copy of the current custom CCD registry."""
    with _ccd_registry_lock:
        return {code: arr.copy() for code, arr in _ccd_registry.items()}


def clear_custom_ccd_registry() -> int:
    """Clear all custom CCD entries. Returns number of entries cleared."""
    with _ccd_registry_lock:
        count = len(_ccd_registry)
        _ccd_registry.clear()

    if count > 0:
        logger.info(f"Cleared {count} custom CCD entries")

    return count


@contextmanager
def custom_ccd_residues(entries: dict[str, struc.AtomArray]) -> Generator[None, None, None]:
    """Temporarily register custom CCD entries. Restores registry state on exit.

    Args:
      entries: Dict mapping CCD codes to AtomArrays.

    Example:
        >>> with custom_ccd_residues({"ALA": custom_ala, "GLY": custom_gly}):
        ...     ala = atom_array_from_ccd_code("ALA")
    """
    # Save current registry state
    with _ccd_registry_lock:
        saved_registry = {code: arr.copy() for code, arr in _ccd_registry.items()}

    try:
        register_custom_ccd_entries(entries)
        yield
    finally:
        # Restore original registry state
        with _ccd_registry_lock:
            _ccd_registry.clear()
            _ccd_registry.update(saved_registry)


def register_custom_residues_from_atom_array(
    atom_array: struc.AtomArray,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> list[str]:
    """Extract and register non-standard residues from AtomArray.

    Compares residue names against standard CCD codes only (not registry).
    This allows re-registration/overwriting of custom residues.

    Args:
      atom_array: Structure containing residues to extract.
      ccd_mirror_path: Path to CCD mirror for standard codes check.

    Returns:
      List of CCD codes that were successfully registered.
    """
    # Get unique residue names
    unique_res_names = set(np.unique(atom_array.res_name))

    # Compare against STANDARD codes only (excludes registry)
    standard_ccd_codes = get_standard_ccd_codes(ccd_mirror_path)
    non_standard_residues = unique_res_names - standard_ccd_codes

    if not non_standard_residues:
        return []

    # Extract only non-standard residues
    non_standard_mask = np.isin(atom_array.res_name, list(non_standard_residues))
    non_standard_atoms = atom_array[non_standard_mask]

    # Track what we've registered to avoid duplicates in single call
    registered_codes = []
    seen = set()

    for residue in struc.residue_iter(non_standard_atoms):
        res_name = residue.res_name[0]

        # Skip if already processed in this call
        if res_name in seen:
            continue
        seen.add(res_name)

        # Skip if contains NaN coordinates
        if np.isnan(residue.coord).any():
            logger.info(f"Skipping {res_name}: contains NaN coordinates")
            continue

        try:
            register_custom_ccd_entry(res_name, residue)
            registered_codes.append(res_name)
        except Exception as e:
            logger.warning(f"Failed to register {res_name}: {e}")

    if registered_codes:
        logger.info(f"Registered {len(registered_codes)} custom CCD entries: {registered_codes}")

    return registered_codes


def _extract_chem_comp_type_from_cif_block(cif_block: pdbx.CIFBlock) -> dict[str, str] | None:
    """Extract chem comp type information from the CIF block, if it exists.

    Structure CIFs use ``chem_comp.type``; CCD CIFs use ``chem_comp.pdbx_type``.

    Returns:
        Mapping from ``chem_comp.id`` to the uppercase chem_comp type, or
        ``None`` if the relevant category/fields are not found.
    """
    if "chem_comp" not in cif_block or "id" not in cif_block["chem_comp"]:
        return None

    chem_comp = cif_block["chem_comp"]
    type_field = next((c for c in ("type", "pdbx_type") if c in chem_comp), None)
    if type_field is None:
        return None

    ids = chem_comp["id"].as_array()
    types = chem_comp[type_field].as_array()
    return {cid: ctype.upper() for cid, ctype in zip(ids, types, strict=False)}


def build_ccd_entries_from_cif_block(
    cif_block: pdbx.CIFBlock,
    on_mismatch: Literal["ignore", "error_heavy", "error"] = "error_heavy",
) -> dict[str, struc.AtomArray]:
    """Build CIF component templates, supplementing only name-compatible CCD entries.

    Authored fields win. CCD fallback requires heavy-atom names from one CCD naming convention;
    otherwise missing types are inferred from names. Multi-atom entries require bonds.
    ``on_mismatch`` rejects unmatched heavy atoms by default (``"error_heavy"``), or any
    unmatched atom with ``"error"``. ``"ignore"`` keeps incompatible components without supplementation.
    """
    if "chem_comp_atom" not in cif_block:
        return {}

    atom_cat = cif_block["chem_comp_atom"]
    all_comp_ids = atom_cat["comp_id"].as_array(str)
    all_atom_ids = atom_cat["atom_id"].as_array(str)

    def _col(name: str, dtype: type = str) -> np.ndarray | None:
        return atom_cat[name].as_array(dtype) if name in atom_cat else None

    def _bool_col(name: str) -> np.ndarray | None:
        return np.where(atom_cat[name].as_array(str) == "Y", True, False) if name in atom_cat else None

    all_elements = _col("type_symbol", str)
    all_charges = _col("charge", np.int8)
    all_aromatic = _bool_col("pdbx_aromatic_flag")
    all_leaving_atom = _bool_col("pdbx_leaving_atom_flag")
    all_x_coords = _col("pdbx_model_Cartn_x_ideal", np.float64)
    all_y_coords = _col("pdbx_model_Cartn_y_ideal", np.float64)
    all_z_coords = _col("pdbx_model_Cartn_z_ideal", np.float64)
    all_coords = (
        np.stack([all_x_coords, all_y_coords, all_z_coords], axis=-1)
        if all_x_coords is not None and all_y_coords is not None and all_z_coords is not None
        else None
    )

    bond_dict = (
        pdbx.convert._parse_intra_residue_bonds(cif_block["chem_comp_bond"]) if "chem_comp_bond" in cif_block else {}
    )

    id_to_type = _extract_chem_comp_type_from_cif_block(cif_block) or {}

    entries: dict[str, struc.AtomArray] = {}
    for comp_id_np in np.unique(all_comp_ids):
        comp_id = str(comp_id_np)
        mask = all_comp_ids == comp_id_np

        atom_names = all_atom_ids[mask].copy()
        if len(atom_names) > 1 and comp_id not in bond_dict:
            # No way to know intra-residue connectivity without authored bonds;
            # skip registration and let downstream lookups fall through to the
            # bundled CCD.
            logger.warning(
                f"Skipping custom CCD registration for '{comp_id}': "
                f"{len(atom_names)} atoms in 'chem_comp_atom' but no "
                f"'chem_comp_bond' rows. Falling through to bundled CCD."
            )
            continue

        elements = all_elements[mask].copy() if all_elements is not None else np.full(len(atom_names), "", dtype=str)
        ccd_atom_names, ccd_alt_names, _ = get_atom_names_for_residue(comp_id, str(CCD_MIRROR_PATH or ""))
        heavy_atom_names = atom_names[~np.isin(elements, HYDROGEN_LIKE_SYMBOLS)]
        matches_ccd = set(heavy_atom_names).issubset(ccd_atom_names) or set(heavy_atom_names).issubset(ccd_alt_names)
        chem_comp_type = id_to_type.get(comp_id) or (
            get_chem_comp_type(comp_id)
            if matches_ccd
            else _infer_chem_comp_type_from_atom_names(set(atom_names), warn=True)
        )

        atoms = _build_ccd_template_atom_array(
            comp_id=comp_id,
            chem_comp_type=chem_comp_type,
            atom_names=atom_names,
            elements=elements,
            charges=all_charges[mask] if all_charges is not None else None,
            is_aromatic=all_aromatic[mask] if all_aromatic is not None else None,
            is_leaving_atom=all_leaving_atom[mask] if all_leaving_atom is not None else None,
            coords=all_coords[mask] if all_coords is not None else None,
        )
        atoms.bonds = (
            struc.connect_via_residue_names(atoms, custom_bond_dict={comp_id: bond_dict[comp_id]})
            if comp_id in bond_dict
            else struc.BondList(len(atom_names))
        )
        # Compatibility is component-wide: never supplement just the overlapping atoms.
        missing_annotations = [
            name
            for name in DEFAULT_CCD_ANNOTATIONS
            if name not in atoms.get_annotation_categories() or (name == "charge" and all_charges is None)
        ]
        if matches_ccd or on_mismatch != "ignore":
            atoms = add_annotations_from_ccd(
                atoms,
                annotations=missing_annotations if matches_ccd else [],
                on_mismatch=on_mismatch,
                match_alt_atom_ids=True,
            )
        entries[comp_id] = atoms

    return entries


@_standard_ccd_only_cache(functools.cache)
def _get_base_ccd_template(
    ccd_code: str,
    ccd_mirror_path: str,  # Must be hashable for lru_cache
    hydrogen_policy: Literal["keep", "remove"],
) -> struc.AtomArray:
    """Get base CCD template (cached implementation).

    This is cached to avoid redundant disk I/O for the same CCD codes.
    Returns a template that will be copied and annotated by get_empty_ccd_template().

    Args:
        ccd_code: CCD code to fetch
        ccd_mirror_path: Path to CCD mirror (as string), or empty string to use Biotite's built-in CCD
        hydrogen_policy: Whether to keep or remove hydrogens from template.

    Returns:
        CCD template AtomArray
    """
    template = atom_array_from_ccd_code(ccd_code, ccd_mirror_path, coords=None)

    if hydrogen_policy == "remove":
        template = annotate_hydrogens(template)
        template = remove_hydrogens(template)

    return template


@_standard_ccd_only_cache(functools.cache)
def get_atom_names_for_residue(
    res_name: str,
    ccd_mirror_path: str,
) -> tuple[frozenset[str], frozenset[str], dict[str, str]]:
    """Get atom name sets and alt→standard mapping from a CCD template.

    Args:
        res_name: Residue name (3-letter code).
        ccd_mirror_path: Path to CCD mirror (as string), or empty string to use Biotite's built-in CCD.

    Returns:
        Tuple of (std_names_set, alt_names_set, alt_to_std_mapping).
        Returns (frozenset(), frozenset(), {}) if template unavailable.
    """
    try:
        template = _get_base_ccd_template(res_name, ccd_mirror_path, hydrogen_policy="keep")
        std_names_set = frozenset(template.atom_name)

        if "alt_atom_id" in template.get_annotation_categories():
            alt_names_set = frozenset(template.alt_atom_id)
            alt_to_std = {
                alt: std
                for std, alt in zip(template.atom_name, template.alt_atom_id, strict=False)
                if alt and alt != std
            }
            return std_names_set, alt_names_set, alt_to_std

        return std_names_set, frozenset(), {}

    except (AttributeError, ValueError, KeyError):
        return frozenset(), frozenset(), {}
