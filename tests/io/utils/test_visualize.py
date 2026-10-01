import numpy as np
import py3Dmol
import pytest
from biotite.structure import AtomArray
from biotite.structure.bonds import BondList

from atomworks.io.transforms.atom_array import is_any_coord_nan
from atomworks.io.utils.visualize import view


def _has_pymol_remote() -> bool:
    try:
        from atomworks.io.utils.visualize import get_pymol_session

        get_pymol_session()
        return True
    except Exception:
        return False


skip_if_no_pymol_remote = pytest.mark.skipif(
    not _has_pymol_remote(),
    reason="Could not find a running `pymol-remote` session.",
)


@pytest.fixture
def sample_atom_array():
    """Create a sample AtomArray for testing."""
    atoms = AtomArray(10)
    atoms.bonds = BondList(10)
    atoms.set_annotation("chain_id", ["A"] * 5 + ["B"] * 5)
    atoms.set_annotation("element", [6] * 5 + [7] * 5)  # Carbon and Nitrogen
    atoms.set_annotation("res_id", list(range(1, 11)))
    atoms.set_annotation("res_name", ["ALA"] * 10)
    atoms.set_annotation("atom_name", ["CA"] * 10)
    atoms.coord = np.random.rand(10, 3)
    return atoms


def test_view_basic(sample_atom_array):
    """Test basic functionality of the view function."""
    result = view(sample_atom_array)
    assert isinstance(result, py3Dmol.view)


def test_view_custom_dimensions(sample_atom_array):
    """Test view function with custom width and height."""
    result = view(sample_atom_array, width=800, height=600)
    assert isinstance(result, py3Dmol.view)
    # Note: We can't directly check the dimensions of the view object


def test_view_zoom_to_selection(sample_atom_array):
    """Test zooming to a specific selection."""
    result = view(sample_atom_array, zoom_to_selection={"chain": "A"})
    assert isinstance(result, py3Dmol.view)


def test_view_show_unoccupied(sample_atom_array):
    """Test showing unoccupied atoms."""
    sample_atom_array.set_annotation("occupancy", [1.0] * 8 + [0.0] * 2)
    result_hidden = view(sample_atom_array, show_unoccupied=False)
    result_shown = view(sample_atom_array, show_unoccupied=True)
    assert isinstance(result_hidden, py3Dmol.view)
    assert isinstance(result_shown, py3Dmol.view)


def test_view_custom_colors(sample_atom_array):
    """Test using custom colors for visualization."""
    custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
    result = view(sample_atom_array, colors=custom_colors)
    assert isinstance(result, py3Dmol.view)


def test_view_polymer_types(sample_atom_array):
    """Test visualization of different polymer types."""
    # Modify sample_atom_array to include protein, nucleic acid, and ion
    sample_atom_array.res_name[:3] = ["ALA", "GLY", "SER"]
    sample_atom_array.res_name[3:6] = ["A", "C", "G"]
    sample_atom_array.element[6:] = [11, 12, 13, 14]  # Some metal ions
    result = view(sample_atom_array, min_polymer_size=1)
    assert isinstance(result, py3Dmol.view)


@pytest.mark.parametrize("show_surface", [True, False])
def test_view_surface_option(sample_atom_array, show_surface):
    """Test the show_surface option."""
    result = view(sample_atom_array, show_surface=show_surface)
    assert isinstance(result, py3Dmol.view)


@pytest.mark.parametrize("show_cartoon", [True, False])
def test_view_cartoon_option(sample_atom_array, show_cartoon):
    """Test the show_cartoon option."""
    result = view(sample_atom_array, show_cartoon=show_cartoon)
    assert isinstance(result, py3Dmol.view)


@pytest.mark.parametrize("show_hover", [True, False])
def test_view_hover_option(sample_atom_array, show_hover):
    """Test the show_hover option."""
    result = view(sample_atom_array, show_hover=show_hover)
    assert isinstance(result, py3Dmol.view)


def test_view_invalid_input():
    """Test view function with invalid input."""
    with pytest.raises(AttributeError):
        view("not an AtomArray")


@pytest.mark.requires_pymol_remote
@skip_if_no_pymol_remote
def test_view_pymol_remote(sample_atom_array):
    """Test view function with PyMOL remote."""
    from io import StringIO

    import biotite.structure as struc

    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import load_any
    from atomworks.io.utils.visualize import get_pymol_session, view_pymol
    from tests.io.conftest import get_pdb_path

    result = parse(get_pdb_path("5ocm"))

    array = result["assemblies"]["1"][0]
    obj_name = view_pymol(array)

    session = get_pymol_session()
    assert obj_name in session.get_object_list(), "Object not found in PyMOL session"
    session.get_state(obj_name, format="cif")

    _is_nan_coords = is_any_coord_nan(array)

    array_not_nan = array[~_is_nan_coords]
    loaded_array = load_any(StringIO(session.get_state(obj_name, format="cif")), model=1)
    assert loaded_array.coord.shape == array_not_nan.coord.shape

    # NOTE: To compare actual coordinates we need to sort both arrays by
    #  (chain_id, res_id, res_name, atom_name)
    #  ...(pymol somehow messes up the order)
    # array_not_nan = array_not_nan[struc.info.standardize_order(array_not_nan)]
    loaded_array = loaded_array[struc.info.standardize_order(loaded_array)]
    assert np.allclose(array_not_nan.coord, loaded_array.coord, atol=1e-2, rtol=1e-2)
