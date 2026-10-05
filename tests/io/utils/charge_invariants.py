"""Shared charge-validation helpers for IO tests."""

import functools

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER
from atomworks.io.utils.atom_array import count_bonded_hydrogens
from atomworks.io.utils.ccd import atom_array_from_ccd_code

_BOND_ORDER_SYMBOL = {1: "-", 2: "=", 3: "≡"}


def _format_bad_atoms(atom_array: AtomArray, mask: np.ndarray) -> str:
    """Format chain/res_id/atom_name for atoms where mask is True, including bond context."""
    idxs = np.where(mask)[0]
    bonds_array = atom_array.bonds.as_array() if atom_array.bonds is not None else None
    lines = []
    for i in idxs:
        bond_info = ""
        if bonds_array is not None:
            m0 = bonds_array[:, 0] == i
            m1 = bonds_array[:, 1] == i
            nb_indices = np.concatenate([bonds_array[m0, 1], bonds_array[m1, 0]])
            bt_values = np.concatenate([bonds_array[m0, 2], bonds_array[m1, 2]])
            bond_strs = []
            for nb_idx, bt in zip(nb_indices, bt_values, strict=False):
                order = BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(bt), 1)
                sym = _BOND_ORDER_SYMBOL.get(order, f"~{order}")
                same_res = (
                    atom_array.chain_id[nb_idx] == atom_array.chain_id[i]
                    and atom_array.res_id[nb_idx] == atom_array.res_id[i]
                    and atom_array.res_name[nb_idx] == atom_array.res_name[i]
                )
                label = atom_array.atom_name[nb_idx]
                if not same_res:
                    label = (
                        f"{atom_array.res_name[nb_idx]}/{label}"
                        f"@{atom_array.chain_id[nb_idx]}{atom_array.res_id[nb_idx]}"
                    )
                bond_strs.append(f"{sym}{label}")
            if bond_strs:
                bond_info = f" bonds=[{', '.join(bond_strs)}]"
        lines.append(
            f"  chain={atom_array.chain_id[i]} res_id={atom_array.res_id[i]} "
            f"res_name={atom_array.res_name[i]} atom_name={atom_array.atom_name[i]} "
            f"element={atom_array.element[i]} charge={atom_array.charge[i]}{bond_info}"
        )
    return "\n".join(lines)


@functools.lru_cache(maxsize=512)
def _get_ccd_charge(res_name: str, atom_name: str) -> int | None:
    """Look up the charge for an atom in the CCD template. Returns None if not found."""
    try:
        ccd = atom_array_from_ccd_code(res_name)
    except (ValueError, KeyError):
        return None
    match = ccd.atom_name == atom_name
    if not np.any(match):
        return None
    return int(ccd.charge[match][0])


def _filter_ccd_justified(atom_array: AtomArray, bad_mask: np.ndarray) -> np.ndarray:
    """Remove atoms from bad_mask whose charge matches the CCD template."""
    if not np.any(bad_mask):
        return bad_mask
    still_bad = bad_mask.copy()
    for idx in np.where(bad_mask)[0]:
        ccd_charge = _get_ccd_charge(atom_array.res_name[idx], atom_array.atom_name[idx])
        if ccd_charge is not None and atom_array.charge[idx] == ccd_charge:
            still_bad[idx] = False
    return still_bad


def check_charge_invariants(pdb_id: str, atom_array: AtomArray) -> None:
    """Run all charge invariant checks on a parsed atom array.

    Collects all failures and reports them in a single assertion at the end.
    """
    failures: list[str] = []
    # (a) No extreme organic charges: |charge| < 2
    organic_elements = {"C", "H", "N", "O", "S", "P", "Se"}
    organic_mask = np.isin(atom_array.element, list(organic_elements))
    bad = organic_mask & (np.abs(atom_array.charge) >= 2)
    if np.any(bad):
        failures.append(f"organic element has extreme charge (|charge| >= 2):\n{_format_bad_atoms(atom_array, bad)}")

    # (b) Backbone neutrality (exempting CCD-justified charges)
    backbone_mask = atom_array.is_backbone_atom
    bad = backbone_mask & (atom_array.charge != 0)
    bad = _filter_ccd_justified(atom_array, bad)
    if np.any(bad):
        failures.append(f"backbone atoms have non-zero charge:\n{_format_bad_atoms(atom_array, bad)}")

    # (c) N+ validation: N with charge +1 must have >= 1 bonded hydrogen
    n_plus_mask = (atom_array.element == "N") & (atom_array.charge == 1)
    for idx in np.where(n_plus_mask)[0]:
        if not count_bonded_hydrogens(atom_array, idx, include_implicit=True) > 0:
            failures.append(
                f"N+ without bonded hydrogen:\n{_format_bad_atoms(atom_array, n_plus_mask & (np.arange(len(atom_array)) == idx))}"
            )
            break  # one example is enough

    # (d) Carbon neutrality
    bad = (atom_array.element == "C") & (atom_array.charge != 0)
    if np.any(bad):
        failures.append(f"carbon atoms have non-zero charge:\n{_format_bad_atoms(atom_array, bad)}")

    # (e) Hydrogen neutrality
    bad = (atom_array.element == "H") & (atom_array.charge != 0)
    if np.any(bad):
        failures.append(f"hydrogen atoms have non-zero charge:\n{_format_bad_atoms(atom_array, bad)}")

    # (f) Oxygen charge range: 0 or -1 (exempting CCD-justified charges)
    o_mask = atom_array.element == "O"
    bad = o_mask & ~np.isin(atom_array.charge, [0, -1])
    bad = _filter_ccd_justified(atom_array, bad)
    if np.any(bad):
        failures.append(
            f"oxygen atoms have unexpected charge (not 0/-1 and differs from CCD):\n{_format_bad_atoms(atom_array, bad)}"
        )

    # (g) Nitrogen charge range: 0 or +1 (exempting CCD-justified charges)
    n_mask = atom_array.element == "N"
    bad = n_mask & ~np.isin(atom_array.charge, [0, 1])
    bad = _filter_ccd_justified(atom_array, bad)
    if np.any(bad):
        failures.append(
            f"nitrogen atoms have unexpected charge (not 0/+1 and differs from CCD):\n{_format_bad_atoms(atom_array, bad)}"
        )

    # (h) Sulfur charge range: 0 or -1 (exempting CCD-justified charges)
    s_mask = atom_array.element == "S"
    bad = s_mask & ~np.isin(atom_array.charge, [0, -1])
    bad = _filter_ccd_justified(atom_array, bad)
    if np.any(bad):
        failures.append(
            f"sulfur atoms have unexpected charge (not 0/-1 and differs from CCD):\n{_format_bad_atoms(atom_array, bad)}"
        )

    short = "; ".join(f.split("\n")[0] for f in failures)
    assert not failures, f"{pdb_id}: charge invariant violations: {short}\n" + "\n".join(failures)
