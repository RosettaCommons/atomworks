"""Dimorphite-DL's protonation state of a small molecule at one pH, and the site rules that decide it."""

from __future__ import annotations

from rdkit import Chem, rdBase

from atomworks.experimental.protonation.external.dimorphite_dl.dimorphite_dl import (
    ProtSubstructFuncs,
    protonate_mol_variants,
)

__all__ = ["protonate_at_ph", "site_rules"]


def protonate_at_ph(mol: Chem.Mol, ph: float, *, max_variants: int = 128) -> Chem.Mol | None:
    """The first Dimorphite-DL state of *mol* at *ph*, its atoms in input order.

    A site whose pKa is within 0.1 standard deviations of *ph* takes both states, and
    *max_variants* bounds the states kept after each site; *mol* is not modified.

    Returns:
        The protonated molecule, with the input's atom maps, atom properties and conformers (not its
        molecule-level properties); ``None`` if *mol* cannot be neutralised.
    """
    with rdBase.BlockLogs():
        variants = protonate_mol_variants(mol, min_ph=ph, max_ph=ph, pka_precision=0.1, max_variants=max_variants)
    return variants[0] if variants else None


def site_rules() -> tuple[tuple[str, str, tuple[tuple[str, float, float], ...]], ...]:
    """The site rules in priority order: name, SMARTS, and per site its match index, pKa mean and standard deviation."""
    return tuple((name, smarts, sites) for name, smarts, _, sites in ProtSubstructFuncs._compiled_substructures())
