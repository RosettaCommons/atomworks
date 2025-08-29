"""Utility functions to visualize atom arrays with py3Dmol in Jupyter notebooks."""

__all__ = ["view"]

import gzip
import io
import logging
import os
import uuid
from itertools import cycle
from pathlib import Path

import biotite.structure as struc
import numpy as np
import py3Dmol
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import mol, pdb, pdbx

from atomworks.constants import ATOMIC_NUMBER_TO_ELEMENT, METAL_ELEMENTS
from atomworks.io.utils.io_utils import read_any, to_cif_string

logger = logging.getLogger("atomworks.io")

try:
    import pymol_remote.client

    _is_pymol_remote_installed = True
except ImportError:
    _is_pymol_remote_installed = False
    logger.warning("PymolSession not installed, visualization will not work")

IPD_PYMOL_COLORS = [
    "#888888",  # pymol_gray
    "#FAC72C",  # good_yellow
    "#29B0C1",  # good_teal
    "#AAC32F",  # good_green
    "#EC72A4",  # good_pink
    "#4499E7",  # good_blue
    "#DCDCDC",  # good_gray
    "#E44A3E",  # good_red
    "#65B37C",  # good_light_green
    "#4FB9AF",  # paper_teal
    "#FFE0AC",  # paper_navaho
    "#FFC6B2",  # paper_melon
    "#FFACB7",  # paper_pink
    "#D59AB5",  # paper_purple
    "#9596C6",  # paper_lightblue
    "#6686C5",  # paper_blue
    "#4B5FAA",  # paper_darkblue
    "#222222",  # pymol_black
]

_is_metal = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x.capitalize()) in METAL_ELEMENTS)


def view(
    structure: AtomArray | AtomArrayStack,
    *,
    zoom_to_selection: dict[str, int | str] | None = None,
    show_hover: bool = True,
    show_unoccupied: bool = False,
    show_cartoon: bool = True,
    show_surface: bool = True,
    width: int = 600,
    height: int = 400,
    ligand_linewidth: float = 0.2,
    polymer_sidechain_linewidth: float = 0.05,
    min_polymer_size: int = 1,
    colors: list[str] = IPD_PYMOL_COLORS,
) -> py3Dmol.view:
    """Visualize an AtomArray structure using py3Dmol for display in jupyter notebooks.

    Args:
        - structure (AtomArray): The atomic structure to be visualized.
        - zoom_to_selection (dict[str, int | str] | None, optional): A dictionary specifying the
            selection to zoom into. Defaults to None. Here are some examples:
                - `{'serial': 35}` - will zoom to the atom with index 35 in the atom array
                - `{'chain': 'A', 'resi': 35}` - will zoom to the residue id 35 in chain A
                - `{'chain': 'C'} - will zoom to the entire chain C
            !WARNING! If the selection is wrong, the visualization will be empty.
        - show_hover (bool, optional): Whether to enable hover functionality to display atom details.
            Defaults to True.
        - show_unoccupied (bool, optional): Whether to show unoccupied atoms. Defaults to False.
        - show_cartoon (bool, optional): Whether to show the cartoon. Defaults to True.
        - show_surface (bool, optional): Whether to show the surface. Defaults to False.
        - width (int, optional): The width of the visualization window. Defaults to 400.
        - height (int, optional): The height of the visualization window. Defaults to 300.
        - ligand_linewidth (float, optional): The linewidth for ligand representation. Defaults to 0.2.
        - polymer_sidechain_linewidth (float, optional): The linewidth for polymer sidechain representation. Defaults to 0.05.
        - min_polymer_size (int, optional): The minimum size for a chain to be displayed as a polymer. Defaults to 1.
        - colors (list[str], optional): A list of colors to cycle through for different chains. Defaults to IPD_PYMOL_COLORS.

    Returns:
        py3Dmol.view: The py3Dmol view object for the structure visualization.
    """
    if isinstance(structure, AtomArrayStack):
        logger.warning("AtomArrayStack is not supported; using the first model.")
        structure = structure[0]

    # Initialize the py3Dmol view with specified width and height
    view = py3Dmol.view(width=width, height=height)

    # Handle unoccupied atoms
    if not show_unoccupied and ("occupancy" in structure.get_annotation_categories()):
        structure = structure[structure.occupancy > 0]

    # Convert the structure to a temporary CIF string for interacting with py3Dmol
    _tmp_cif_str = to_cif_string(
        structure,
        _allow_ambiguous_bond_annotations=True,
        include_entity_poly=False,
    )
    # ... add the structure model to the view in mmCIF format
    view.addModel(_tmp_cif_str, "structure", format="mmcif")
    # Get the chain IDs from the structure
    chain_ids = struc.get_chains(structure)

    # Iterate over each chain and assign styles based on the type of polymer
    for chain_id, color in zip(chain_ids, cycle(colors)):
        is_protein = np.all(
            struc.filter_polymer(
                structure[structure.chain_id == chain_id], pol_type="peptide", min_size=min_polymer_size
            )
            & struc.filter_amino_acids(structure[structure.chain_id == chain_id])
        )
        is_nucleic = np.any(
            struc.filter_polymer(
                structure[structure.chain_id == chain_id], pol_type="nucleotide", min_size=min_polymer_size
            )
            & struc.filter_nucleotides(structure[structure.chain_id == chain_id])
        )
        is_ion = np.all(_is_metal(structure[structure.chain_id == chain_id].element))

        if is_protein or is_nucleic:
            # Apply protein or nucleic acid style
            style = {"stick": {"radius": polymer_sidechain_linewidth, "style": "outline"}}
            if show_cartoon:
                style["cartoon"] = {"color": color, "arrows": True}
            view.setStyle({"chain": chain_id}, style)

        elif is_ion:
            view.setStyle(
                {"chain": chain_id},
                {"stick": {"radius": polymer_sidechain_linewidth, "style": "outline"}},
            )
        elif is_ion:
            # Apply ion style
            view.setStyle(
                {"chain": chain_id},
                {"sphere": {"scale": 0.8}},
            )
        else:
            # Apply ligand style
            # ... first, set the style for carbon atoms colored by chain
            view.setStyle(
                {"chain": chain_id, "elem": "C"},
                {"stick": {"color": color, "radius": ligand_linewidth}},
            )
            # ... then, set the style for all other atoms based on the element
            view.setStyle(
                {"chain": chain_id, "not": {"elem": "C"}},
                {"stick": {"colorscheme": "element", "radius": ligand_linewidth}},
            )

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "gray"})

    # Add hover functionality to display atom details on hover
    if show_hover:
        js_script = """function(atom,viewer) {
                    if(!atom.label) {
                        atom.label = viewer.addLabel(
                            atom.chain + ':' +
                            atom.resn + '(' + atom.resi + '):' +
                            atom.atom + '(idx' + atom.serial + ')',
                            {position: atom, backgroundColor:"white", fontColor:"black"}
                        );
                    }
                }"""
        view.setHoverable(
            {},
            True,
            js_script,
            """function(atom,viewer) {
                    if(atom.label) {
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }
                    }""",
        )

    # Zoom to the entire structure or to a specific selection if provided
    view.zoomTo()
    if zoom_to_selection is not None:
        view.zoomTo(zoom_to_selection)

    return view


def get_pymol_session(hostname: str | None = None, port: int | None = None) -> "pymol_remote.client.PymolSession":
    """
    Establishes a connection to a PyMOL server and returns a `pymol_remote.client.PymolSession` object.
    First attempts to reuse an existing global session if no hostname/port is specified.
    Otherwise tries to establish a new connection, attempting up to 5 consecutive ports.

    If you want to use `pymol_remote`, make sure to follow the usage instructions at
        https://github.com/Croydon-Brixton/pymol-remote

    Args:
        - hostname (str | None, optional): The hostname of the PyMOL server. Defaults to 'localhost' if None.
        - port (int | None, optional): The starting port number to attempt connection. Defaults to 9123 if None.

    Returns:
        pymol_remote.client.PymolSession: An active connection to the PyMOL server.

    Raises:
        - ImportError: If `pymol_remote` package is not installed.
        - RuntimeError: If unable to establish connection after trying 5 consecutive ports.
    """
    if not _is_pymol_remote_installed:
        raise ImportError("`pymol_remote` is not installed or in the pythonpath, visualization will not work.")

    # ... get existing session if available
    if (hostname is None) and (port is None):
        session = pymol_remote.client._GLOBAL_SERVER_PROXY
        if session:
            return pymol_remote.client.PymolSession(hostname=session.hostname, port=session.port, force_new=False)

    # ... otherwise, try to connect to a new session
    hostname = hostname or "localhost"
    port = port or 9123
    for i in range(5):
        try:
            session = pymol_remote.client.PymolSession(hostname=hostname, port=port + i)
            break
        except Exception:
            session = None
            pass

    if session is None:
        raise RuntimeError(
            f"Failed to connect to Pymol on {hostname}:{port}."
            "Ensure you are using SSH forwarding and `pymol_remote` correctly."
        )

    return session


def view_pymol(
    structure: AtomArray
    | AtomArrayStack
    | pdbx.CIFFile
    | pdbx.BinaryCIFFile
    | pdb.PDBFile
    | pdbx.CIFBlock
    | pdbx.BinaryCIFBlock
    | os.PathLike,
    id: str | None = None,
    hostname: str | None = None,
    port: int | None = None,
    as_bcif: bool = False,
    overwrite: bool = False,
    grid_slot: int | None = None,
) -> str:
    """
    Visualizes an AtomArray structure in PyMOL by connecting to a PyMOL server and loading the structure. If no ID is
    provided, generates a unique identifier for the structure.

    Args:
        - structure (AtomArray | AtomArrayStack | CIFFile | BinaryCIFFile | PDBFile | CIFBlock | BinaryCIFBlock | PathLike):
            The atomic structure to be visualized in PyMOL. For `PathLike`, the file extension is used to determine the format
            of the structure when no `id` is provided.
        - id (str | None, optional): Unique identifier for the structure in PyMOL. If None, generates a random 9-character
            string in XXX-XXX-XXX format. Defaults to None.
        - hostname (str | None, optional): The hostname of the PyMOL server. If None, uses 'localhost' or attempts to reuse
            an existing connection. Defaults to None.
        - port (int | None, optional): The port number for the PyMOL server connection. If None, uses default port 9123 or
            attempts to reuse existing connection. Defaults to None.
        - as_bcif (bool, optional): Whether to transport the structure as BCIF instead of CIF. This speeds up the
            network transfer of the structure but reading bcif files is not supported by all pymol versions.
            (pymol 2.6 (LTS) and 3.1+ support bcif, older versions do not).
            This only takes effect if `structure` is an `AtomArray` or `AtomArrayStack`.
            Defaults to False.
        - overwrite (bool, optional): Whether to overwrite an existing object with the same ID. Defaults to False.
        - grid_slot (int | None, optional): The grid slot to use for the structure. If None, a random slot is chosen.
            Defaults to None.

    Returns:
        str: The identifier used for the structure in PyMOL.

    Raises:
        ImportError: If `pymol_remote` package is not installed.
        RuntimeError: If unable to establish connection to PyMOL server.
    """
    # Establish a connection to the pymol server
    session = get_pymol_session(hostname, port)

    if isinstance(structure, str | Path):
        id = id or os.path.basename(structure).split(".")[0]
        structure = read_any(structure)

    # Generate a unique ID for the structure if not provided
    if id is None:
        # Generate random 9-character string in 3-3-3 format
        random_str = str(uuid.uuid4()).replace("-", "")[:9]
        id = f"{random_str[:3]}-{random_str[3:6]}-{random_str[6:]}"

    if id in session.get_names():
        if overwrite:
            logger.warning(f"Object {id=} already exists in PyMOL, overwriting.")
            session.delete(id)
        else:
            raise ValueError(f"Object {id=} already exists in PyMOL, set overwrite=True to overwrite.")

    # Send to pymol
    if isinstance(structure, AtomArray | AtomArrayStack):
        format = "bcif" if as_bcif else "cif"
        buffer = to_cif_string(
            structure,
            id=id,
            _allow_ambiguous_bond_annotations=True,
            include_entity_poly=True,
            include_nan_coords=False,
            include_bonds=True,
            extra_fields=[],
            as_bcif=as_bcif,
        )
    elif isinstance(structure, pdbx.CIFFile | pdb.PDBFile | mol.SDFile | pdbx.CIFBlock):
        format = {
            pdbx.CIFFile: "cif",
            pdb.PDBFile: "pdb",
            mol.SDFile: "sdf",
            pdbx.CIFBlock: "cif",
        }[type(structure)]
        buffer = io.StringIO()
        if isinstance(structure, pdbx.CIFBlock):
            _tmp = pdbx.CIFFile()
            _tmp[id] = structure
            structure = _tmp
        structure.write(buffer)
        buffer = buffer.getvalue()
    elif isinstance(structure, pdbx.BinaryCIFFile | pdbx.BinaryCIFBlock):
        format = "bcif"
        buffer = io.BytesIO()
        if isinstance(structure, pdbx.BinaryCIFBlock):
            _tmp = pdbx.BinaryCIFFile()
            _tmp[id] = structure
            structure = _tmp
        structure.write(buffer)
        buffer = buffer.getvalue()
    else:
        raise ValueError(
            f"Unsupported structure type: {type(structure)}. Only AtomArray, AtomArrayStack, CIFFile, and BCIFFile are supported."
        )

    # turn str into bytes if it is not already
    if not isinstance(buffer, bytes):
        buffer = buffer.encode("utf-8")
    # compress for faster network transfer
    buffer = gzip.compress(buffer)

    session.set_state(buffer, object=id, format=format)
    grid_slot = np.random.randint(0, 10_000) if grid_slot is None else grid_slot
    session.set("grid_slot", grid_slot, id)

    return id
