"""Vendorized Dimorphite-DL: enumerate ionization states of drug-like small molecules.

Source: https://durrantlab.pitt.edu/dimorphite-dl/ (Dimorphite-DL 1.2.4)
License: Apache 2.0 (Copyright 2020 Jacob D. Durrant)

Reference:
    Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
    open-source program for enumerating the ionization states of drug-like
    small molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.
"""

from atomworks.external.dimorphite_dl.dimorphite_dl import protonate_mol_variants

__all__ = ["protonate_mol_variants"]
