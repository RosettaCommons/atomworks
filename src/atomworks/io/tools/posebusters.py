"""Tools for running PoseBusters validation on RDKit molecules or AtomArrays."""

import contextlib
import logging
from functools import lru_cache
from typing import Any

from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import Mol

from atomworks.constants import METAL_ELEMENTS
from atomworks.io.tools.rdkit import atom_array_to_rdkit

try:
    from posebusters import PoseBusters
except ImportError:
    PoseBusters = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_CHECKS: frozenset[str] = frozenset({"pb_all_atoms_connected", "pb_inchi_convertible"})


def _ensure_mol(mol_or_atom_array: Mol | AtomArray) -> Mol:
    """Convert an AtomArray to an RDKit Mol if needed."""
    if isinstance(mol_or_atom_array, AtomArray):
        return atom_array_to_rdkit(
            mol_or_atom_array,
            set_coord=True,
            hydrogen_policy="keep",
            infer_bonds=False,
            sanitize=True,
            assume_metal_bonds_are_dative=True,
        )
    return mol_or_atom_array


def _check_posebusters_available() -> None:
    if PoseBusters is None:
        raise ImportError(
            "posebusters is required for PoseBusters validation. " "Install it with: pip install posebusters"
        )


@lru_cache(maxsize=1)
def _get_default_posebusters() -> Any:
    """Return a lazily-cached ``PoseBusters(config="mol")`` instance."""
    _check_posebusters_available()
    return PoseBusters(config="mol")


def _merge_check_results(results: list[dict[str, bool]]) -> dict[str, bool]:
    """Merge per-fragment check results via AND across all fragments."""
    all_keys = {k for d in results for k in d}
    return {k: all(d[k] for d in results if k in d) for k in all_keys}


def run_posebusters(
    mol: Mol | AtomArray,
    pb: Any | None = None,
    excluded_checks: frozenset[str] = DEFAULT_EXCLUDED_CHECKS,
) -> tuple[bool, dict[str, bool]]:
    """Run PoseBusters on a single molecule.

    Args:
        mol: RDKit molecule or AtomArray to validate.
        pb: Optional pre-built PoseBusters instance for reuse across calls.
        excluded_checks: Check keys to exclude from pass/fail determination.

    Returns:
        Tuple of ``(failed, check_results)`` where *failed* is ``True`` if PB
        raised an exception or any non-excluded check failed, and *check_results*
        maps ``pb_<check>`` to ``bool`` (empty dict on crash).
    """
    mol = _ensure_mol(mol)
    if pb is None:
        pb = _get_default_posebusters()

    try:
        results_df = pb.bust(mol_pred=mol, full_report=False)
        check_results = {
            f"pb_{k}": bool(v)
            for k, v in results_df.iloc[0].to_dict().items()
            if isinstance(v, bool | int) and k != "file_path"
        }
        relevant = {k: v for k, v in check_results.items() if k not in excluded_checks}
        failed = not (all(relevant.values()) if relevant else True)
        return failed, check_results
    except Exception:
        logger.debug("PoseBusters raised an exception", exc_info=True)
        return True, {}


def validate_organic_fragments(
    mol: Mol | AtomArray,
    pb: Any | None = None,
    excluded_checks: frozenset[str] = DEFAULT_EXCLUDED_CHECKS,
) -> tuple[bool, dict[str, bool]]:
    """Run PoseBusters on each organic fragment of a molecule separately.

    Dative bonds are broken so that metal centres and ligands become separate
    fragments.  Only organic (non-metal) fragments are validated.  Results are
    merged across fragments; the function exits early on the first
    failure, returning only the fragments checked so far.

    Returns:
        Tuple of ``(failed, check_results)``.  For metal-only molecules the
        result is ``(False, {})``.
    """
    mol = _ensure_mol(mol)
    mol_copy = Chem.RWMol(Chem.Mol(mol))
    metal_indices = {a.GetIdx() for a in mol_copy.GetAtoms() if a.GetSymbol().upper() in METAL_ELEMENTS}
    metal_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol_copy.GetBonds()
        if b.GetBeginAtomIdx() in metal_indices or b.GetEndAtomIdx() in metal_indices
    ]

    for i, j in sorted(metal_bonds, reverse=True):
        mol_copy.RemoveBond(i, j)

    with contextlib.suppress(Exception):  # Best-effort sanitization after bond removal
        Chem.SanitizeMol(mol_copy)

    frags = Chem.GetMolFrags(Chem.Mol(mol_copy), asMols=True, sanitizeFrags=False)
    organic_frags = [f for f in frags if not any(a.GetSymbol().upper() in METAL_ELEMENTS for a in f.GetAtoms())]

    if not organic_frags:
        return False, {}

    all_check_results: list[dict[str, bool]] = []
    for frag in organic_frags:
        failed, frag_results = run_posebusters(frag, pb=pb, excluded_checks=excluded_checks)
        all_check_results.append(frag_results)
        if failed:
            return True, _merge_check_results(all_check_results)

    return False, _merge_check_results(all_check_results)


def validate_molecule(
    mol: Mol | AtomArray,
    pb: Any | None = None,
    excluded_checks: frozenset[str] = DEFAULT_EXCLUDED_CHECKS,
) -> tuple[bool, dict[str, bool]]:
    """Validate a molecule with PoseBusters, falling back to per-fragment validation.

    Tries whole-molecule validation first.  If PB crashes or fails on a
    metal-containing molecule, falls back to validating each organic fragment
    independently.

    Returns:
        Tuple of ``(failed, check_results)``.
    """
    mol = _ensure_mol(mol)
    has_metal = any(a.GetSymbol().upper() in METAL_ELEMENTS for a in mol.GetAtoms())
    failed, check_results = run_posebusters(mol, pb=pb, excluded_checks=excluded_checks)
    if not check_results or (failed and has_metal):
        failed, check_results = validate_organic_fragments(mol, pb=pb, excluded_checks=excluded_checks)
    return failed, check_results
