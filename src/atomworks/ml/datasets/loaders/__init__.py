"""Functional loader implementations for AtomWorks datasets.

Loaders are functions that process raw dataset output (e.g., pandas Series) into a Transform-ready format.
E.g., converts what may be dataset-specific metadata into a standard format for use in AtomWorks Transform pipelines.
"""

from .ase import (
    ase_atoms_to_atom_array,
    ase_atoms_to_material_dict,
    create_ase_atoms_loader,
    create_ase_materials_loader,
)
from .cif import (
    create_base_loader,
    create_loader_with_interfaces_and_pn_units_to_score,
    create_loader_with_query_pn_units,
)

__all__ = [
    "ase_atoms_to_atom_array",
    "ase_atoms_to_material_dict",
    "create_ase_atoms_loader",
    "create_ase_materials_loader",
    "create_base_loader",
    "create_loader_with_interfaces_and_pn_units_to_score",
    "create_loader_with_query_pn_units",
]
