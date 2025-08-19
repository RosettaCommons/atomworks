# %% [markdown]
"""
CIFUtils: User Example
======================

This example demonstrates how to use the `atomworks.io` package to load, inspect, and manipulate mmCIF structure files. You'll see how to parse a structure, visualize it, and perform basic analysis.
"""

# %% [markdown]
"""
## 1.1 Loading from mmCIF

We start by loading a structure from an mmCIF file using `atomworks.io.parse`. This function supports various file formats and allows you to specify options such as which assembly to build, whether to add missing atoms, and more.
"""

import io

from biotite.database import rcsb

from atomworks.io import parse
from atomworks.io.utils.testing import get_pdb_path
from atomworks.io.utils.visualize import view


def get_example_path_or_buffer(pdb_id: str) -> str | io.StringIO:
    try:
        # ... if file is locally available
        return get_pdb_path(pdb_id)
    except FileNotFoundError:
        # ... otherwise, fetch the file from RCSB
        return rcsb.fetch(pdb_id, format="cif")


result_dict = parse(
    filename=get_example_path_or_buffer("6lyz"),
    build_assembly=["1"],
    add_missing_atoms=True,
    remove_waters=True,
    hydrogen_policy="remove",
    model=1,
)

print("Keys in parsed result:", list(result_dict.keys()))

# %% [markdown]
"""
## 1.2 Visualizing the asymmetric unit and assembly

You can visualize the asymmetric unit or any assembly using the built-in viewer from `atomworks.io`. This is helpful for quickly inspecting the structure and its components.
"""


asym_unit = result_dict["asym_unit"][0]
asym_unit = asym_unit[asym_unit.occupancy > 0]
view(asym_unit)

assembly = result_dict["assemblies"]["1"][0]
assembly = assembly[assembly.occupancy > 0]
view(assembly)

# %% [markdown]
"""
## 1.3 Inspecting structure metadata

The parsed result contains rich metadata, including chain and ligand information, as well as annotation categories. This information is useful for downstream analysis and filtering.
"""

print("Chain info:", result_dict["chain_info"])
print("Ligand info:", result_dict["ligand_info"])
print("Metadata:", result_dict["metadata"])
print("Annotation categories:", result_dict["asym_unit"][0].get_annotation_categories())

# %% [markdown]
"""
## 1.4 Manipulating AtomArray

You can easily extract coordinates for specific atoms or chains, and inspect bond information. This is useful for custom analysis or feature extraction.
"""

ca = assembly[(assembly.atom_name == "CA") & (assembly.occupancy > 0)]
print("Coordinates of all resolved CA atoms:", ca.coord.shape)

chain = assembly[assembly.chain_id == "A"]
print("Coordinates of chain A (all heavy atoms):", chain.coord.shape)

print("Bond array:", assembly.bonds.as_array())

# %% [markdown]
"""
## 1.5 Distance computations

`biotite.structure` provides convenient functions for distance calculations between atoms or sets of atoms. Here we compute distances between C-alpha atoms.
"""

import biotite.structure as struc

distance = struc.distance(ca.coord[0], ca.coord[1])
print(f"Distance between first two C-alpha atoms: {distance:.2f} Å")

distance = struc.distance(ca[0], ca)
print(f"Distances between first C-alpha atom and all other C-alpha atoms: {distance}")

# %% [markdown]
"""
## 1.6 Efficient neighbor search with CellList

For efficient spatial queries, use `CellList` to find atoms within a certain radius. This is useful for contact analysis and neighborhood queries.
"""

resolved_atom_array = assembly[assembly.occupancy > 0]
cell_list = struc.CellList(resolved_atom_array, cell_size=5.0)

near_atoms = cell_list.get_atoms(resolved_atom_array[0].coord, radius=4)
print(f"Number of atoms within 7 Å of the first atom: {near_atoms.shape[0]}")
print(f"Atom indices: {near_atoms}")
print(f"Chain IDs: {resolved_atom_array.chain_id[near_atoms]}")
print(f"Residue IDs: {resolved_atom_array.res_id[near_atoms]}")
print(f"Residue names: {resolved_atom_array.res_name[near_atoms]}")
