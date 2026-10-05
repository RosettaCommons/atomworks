"""Vendorized code from xyz2mol_tm for converting XYZ files to SMILES for transition metal complexes.

Source: https://github.com/jensengroup/xyz2mol_tm
License: MIT (Copyright 2024 Jensen Group)
"""

from atomworks.external.xyz2mol_tm.xyz2mol_tmc import get_tmc_mol

__all__ = ["get_tmc_mol"]
