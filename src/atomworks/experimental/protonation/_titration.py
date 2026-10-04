"""Titrating components with Dimorphite-DL: each atom's charge and hydrogen count at a pH."""

from __future__ import annotations

import logging
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from atomworks.constants import HYDROGEN_LIKE_SYMBOLS, METAL_ELEMENTS
from atomworks.experimental.protonation.dimorphite import protonate_at_ph
from atomworks.io.tools.rdkit import RDKIT_BOND_TYPE_TO_BIOTITE, atom_array_to_rdkit

logger = logging.getLogger(__name__)
_METAL_NUMBERS = frozenset(Chem.GetPeriodicTable().GetAtomicNumber(e.capitalize()) for e in METAL_ELEMENTS)
_PHOSPHORYL_AT_VALENCE_3 = Chem.MolFromSmarts("[P+0;v3;$(P=[O,S])]")


_MULTIPLE = [int(struc.BondType[t]) for t in ("DOUBLE", "AROMATIC_SINGLE", "AROMATIC_DOUBLE", "AROMATIC")]
# N-C-N-C-C ring, the central C bonded to no third N; matched as (N1, C, N3, C4, C5).
_IMIDAZOLE = Chem.MolFromSmarts("[#7]1~[#6;!$([#6](~[#7])(~[#7])~[#7])]~[#7]~[#6]~[#6]~1")


class Imidazoles(NamedTuple):
    """The imidazole rings of a structure: ``n`` ``[k, 2]`` ring nitrogens in atom order; ``tele`` ``[k, 2]``
    whether an N's other ring C bears no heavy substituent while its partner's does (histidine's NE2);
    ``open`` ``[k]`` whether neither N has a third heavy neighbour other than a metal; ``atoms`` ``[k, 5]``."""

    n: np.ndarray
    tele: np.ndarray
    open: np.ndarray
    atoms: np.ndarray


def imidazole_rings(atom_array: AtomArray) -> Imidazoles:
    """The imidazoles of one residue each (HIS, HIC, IMD): N-C-N in a 5-ring whose Kekule form has C=N at that
    carbon and C=C across from it, and no ring atom a multiple bond out of the ring (no imidazolone or -thione).

    One SMARTS match on the residues' heavy-atom bond graph, aromatic bonds read as multiple ones.
    """
    element = np.char.upper(atom_array.element.astype(str))
    heavy = ~np.isin(element, [*HYDROGEN_LIKE_SYMBOLS, *METAL_ELEMENTS])
    residue = struc.get_all_residue_positions(atom_array)
    bonds = atom_array.bonds.as_array() if atom_array.bonds is not None else np.empty((0, 3), dtype=np.int64)
    bonds = bonds[heavy[bonds[:, 0]] & heavy[bonds[:, 1]]]
    degree = np.bincount(bonds[:, :2].ravel(), minlength=len(atom_array))
    graph = Chem.RWMol()
    for symbol in element:
        graph.AddAtom(Chem.Atom({"C": 6, "N": 7}.get(symbol, 0)))
    for left, right, bond_type in bonds[residue[bonds[:, 0]] == residue[bonds[:, 1]]].tolist():
        graph.AddBond(left, right, Chem.BondType.DOUBLE if bond_type in _MULTIPLE else Chem.BondType.SINGLE)
    graph.UpdatePropertyCache(strict=False)
    rings, seen = [], set()
    for n1, c, n2, b, a in graph.GetSubstructMatches(_IMIDAZOLE, uniquify=False):
        cycle = (n1, c, n2, b, a)
        multiple = {
            (x, y.GetIdx())
            for x in cycle
            for y in graph.GetAtomWithIdx(x).GetNeighbors()
            if graph.GetBondBetweenAtoms(x, y.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        }
        if n1 > n2 or c in seen or any(y not in cycle for _, y in multiple) or (a, b) not in multiple:
            continue
        if (n1, c) in multiple or (c, n2) in multiple:
            seen.add(c)
            rings.append((c, cycle))
    rings = [cycle for _, cycle in sorted(rings)]
    n = np.array([(r[0], r[2]) for r in rings], dtype=np.intp).reshape(-1, 2)
    bare = np.array([(degree[r[4]] == 2, degree[r[3]] == 2) for r in rings], dtype=bool).reshape(-1, 2)
    return Imidazoles(
        n, bare & ~bare[:, ::-1], (degree[n] == 2).all(axis=1), np.array(rings, dtype=np.intp).reshape(-1, 5)
    )


def partition_atomized_subgraphs(heavy: AtomArray, atomize_mask: np.ndarray) -> list[np.ndarray]:
    """The connected pieces the bonds among atomized atoms make, as indices into *heavy*."""
    atomized = np.flatnonzero(atomize_mask)
    if not len(atomized):
        return []
    bonds = heavy.bonds.as_array()[:, :2].astype(np.intp)
    local = np.searchsorted(atomized, bonds[atomize_mask[bonds].all(axis=1)])
    graph = csr_matrix((np.ones(len(local), dtype=np.int8), (local[:, 0], local[:, 1])), shape=(len(atomized),) * 2)
    n_components, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(atomized[order], np.cumsum(np.bincount(labels, minlength=n_components))[:-1])


def bonds_within_components(
    heavy: AtomArray, components: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Each component's own bonds, the atoms bonded outside their component, and those links.

    Returns:
        ``(local_bonds, on_boundary, links)``: per component its bonds in bond-list order,
        indexed as slicing the component out would index them; the mask of atoms with a bond to
        another component or to an atom in none; per component ``[k, 3]`` rows of (local atom,
        atomic number of the atom it bonds outside, bond type).
    """
    n = heavy.array_length()
    component_of = np.full(n, -1, dtype=np.intp)
    local_of = np.zeros(n, dtype=np.intp)
    for number, indices in enumerate(components):
        component_of[indices] = number
        local_of[indices] = np.arange(len(indices))

    bond_arr = heavy.bonds.as_array().astype(np.intp)
    left, right = component_of[bond_arr[:, 0]], component_of[bond_arr[:, 1]]
    crossing = left != right
    on_boundary = np.zeros(n, dtype=bool)
    on_boundary[bond_arr[crossing, :2].ravel()] = True
    on_boundary &= component_of >= 0

    inner = np.flatnonzero(~crossing & (left >= 0))
    inner = inner[np.argsort(left[inner], kind="stable")]
    local = bond_arr[inner]
    local[:, :2] = local_of[local[:, :2]]
    counts = np.bincount(left[inner], minlength=len(components))
    table = Chem.GetPeriodicTable()
    numbers = {}
    for element in np.unique(heavy.element.astype(str)):
        try:
            numbers[element] = table.GetAtomicNumber(element.capitalize())
        except RuntimeError:  # 1JQK UNX: element X
            numbers[element] = 0
    links: list[list[tuple[int, int, int]]] = [[] for _ in components]
    for a, b, bond_type in bond_arr[crossing].tolist():
        for inside, outside in ((a, b), (b, a)):
            if component_of[inside] >= 0:
                links[component_of[inside]].append((local_of[inside], numbers[str(heavy.element[outside])], bond_type))
    links = [np.array(sorted(rows), dtype=np.int64).reshape(-1, 3) for rows in links]
    return np.split(local, np.cumsum(counts)[:-1]), on_boundary, links


def _titrate(mol: Chem.Mol, ph: float) -> Chem.Mol | None:
    """Dimorphite-DL's single most likely protonation state of *mol* at *ph*, drawn with *mol*'s bonds, or None.

    Dimorphite-DL's neutralisation can redraw a group (A1L's azide as ``N=[N+]=[N-]``) whose bonds the state
    keeps, so only its protons carry over: an atom that gains or loses one gains or loses a unit of charge.
    """
    # 313 S: Dimorphite-DL's SMILES would bring a non-tetrahedral centre's hydrogens back as atoms.
    stripped = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stripped)
    titrated = protonate_at_ph(stripped, ph, max_variants=1)
    if titrated is None or titrated.GetNumAtoms() != mol.GetNumAtoms():
        return titrated
    drawn = Chem.RWMol(mol)
    for atom, state in zip(drawn.GetAtoms(), titrated.GetAtoms(), strict=True):
        gained = state.GetTotalNumHs() - atom.GetTotalNumHs()
        if gained:
            _set_hydrogens(drawn, atom.GetIdx(), state.GetTotalNumHs(), atom.GetFormalCharge() + gained)
    drawn.UpdatePropertyCache(strict=False)
    return drawn.GetMol()


def _set_hydrogens(mol: Chem.Mol, index: int, count: int, charge: int) -> None:
    """Fix atom *index* of *mol* at exactly *count* hydrogens and formal *charge*."""
    atom = mol.GetAtomWithIdx(int(index))
    atom.SetNumExplicitHs(int(count))
    atom.SetNoImplicit(True)
    atom.SetFormalCharge(int(charge))
    atom.SetNumRadicalElectrons(0)


def _release_ring_bond_orders(mol: Chem.Mol, rings: np.ndarray) -> list[tuple[int, int]]:
    """Mark the bonds of each ring (atom indices in ring order), and of each unsaturated ring fused to it (GVI's
    pyrimidine), aromatic, so sanitizing re-kekulizes them to follow the hydrogens stated on them rather than
    the Kekule form they arrived in; returns their bonds."""
    unsaturated = {a.GetIdx() for a in mol.GetAtoms() if any(b.GetBondTypeAsDouble() > 1 for b in a.GetBonds())}
    rings = [list(ring) for ring in rings.tolist()]
    for other in Chem.GetSymmSSSR(mol):
        other = list(other)
        fused = any(len(set(other) & set(ring)) >= 2 for ring in rings) and other not in rings
        if fused and set(other) <= unsaturated and not any(set(other) == set(ring) for ring in rings):
            rings.append(other)
    released = []
    for ring in rings:
        for left, right in zip(ring, ring[1:] + ring[:1], strict=True):
            mol.GetBondBetweenAtoms(left, right).SetBondType(Chem.BondType.AROMATIC)
            mol.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
            released.append((left, right))
        for index in ring:
            mol.GetAtomWithIdx(index).SetIsAromatic(True)
    return released


def _kept_as_given(sub: AtomArray, exc: Exception) -> None:
    logger.warning(
        "RDKit cannot read %s:%s %s, which keeps the charges and hydrogens it is given: %s",
        *(sub.chain_id[0], sub.res_id[0], sub.res_name[0]),
        " ".join(str(exc).split()[:6]),
    )
    return None


def _restate_coordinating_donors(mol: Chem.Mol, titrated: Chem.Mol, donor_ph: np.ndarray) -> Chem.Mol:
    """Give each metal-coordinating donor of *titrated* its state at its *donor_ph* (NaN: none).

    A metal cation stabilises its donor's conjugate base, so a donor titrates at the higher
    pH :func:`~atomworks.experimental.protonation._assign.effective_donor_ph` gives it (1fdn CYS SG).
    """
    restated = Chem.RWMol(titrated)
    for value in np.unique(donor_ph[np.isfinite(donor_ph)]):
        on_metal = _titrate(mol, float(value))
        if on_metal is None or on_metal.GetNumAtoms() != titrated.GetNumAtoms():
            logger.warning(
                "Could not titrate a metal-coordinating donor in its donating state; it keeps "
                "what the working pH gave it, which may leave a hydrogen pointing at the metal."
            )
            return titrated
        for index in np.flatnonzero(donor_ph == value):
            donor = on_metal.GetAtomWithIdx(int(index))
            _set_hydrogens(restated, index, donor.GetTotalNumHs(), donor.GetFormalCharge())
    try:
        Chem.SanitizeMol(restated)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        logger.warning("Could not put a metal-coordinating donor in its donating state: %s", exc)
        return titrated
    return restated.GetMol()


def titrate_component(
    sub: AtomArray,
    ph: float,
    imidazole_targets: np.ndarray,
    coordinating: np.ndarray,
    stated_hydrogens: np.ndarray,
    donor_ph: np.ndarray,
    links: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Titrate one component (a residue or an atomized subgraph).

    Args:
        sub: The component's heavy atoms and bonds.
        ph: Target pH.
        imidazole_targets: Imidazole ring-N hydrogen counts, -1 where unset.
        coordinating: Most hydrogens each metal donor may keep, -1 for other atoms; a donor held
            to fewer than its state gives keeps its input charge.
        stated_hydrogens: Declared counts, -1 where unset; a declared atom keeps its input charge.
        donor_ph: The pH at which each metal donor takes its donating state, NaN for other atoms.
        links: ``[k, 3]`` (atom, atomic number, bond type) of the component's bonds to other components;
            each is titrated as that atom bearing a methyl, so a linked group keeps its substitution.

    Returns:
        ``(charge, hydrogens, ring_bonds)``: per atom, and the ``[k, 3]`` imidazole ring bonds
        re-kekulized to their targets; ``None`` where RDKit cannot read the component or no Kekule
        form gives its rings their targets.
    """
    try:
        mol = atom_array_to_rdkit(
            sub, set_coord=False, hydrogen_policy="remove", sanitize=True, attempt_fixing_corrupted_molecules=False
        )
    except RuntimeError as exc:  # ASX, GLX: element X
        return _kept_as_given(sub, exc)
    n = sub.array_length()
    # 1TTD DC OP2: a residue titrates with its links capped, so an internal phosphodiester is no monoester.
    if links is not None and len(links):
        capped = Chem.RWMol(mol)
        for index, number, bond_type in links.tolist():
            if not number or number in _METAL_NUMBERS:
                continue
            partner = capped.AddAtom(Chem.Atom(int(number)))
            order = {2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}.get(int(bond_type), Chem.BondType.SINGLE)
            capped.AddBond(int(index), partner, order)
            capped.AddBond(partner, capped.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
        capped = capped.GetMol()
        if Chem.SanitizeMol(Chem.Mol(capped), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
            mol = capped
    # HP4 P1: RDKit reads a phosphoryl P at its lowest valence, 3; it is P(V), as hypophosphite's PH2.
    mol.UpdatePropertyCache(strict=False)
    for (index,) in mol.GetSubstructMatches(_PHOSPHORYL_AT_VALENCE_3):
        _set_hydrogens(mol, index, mol.GetAtomWithIdx(index).GetTotalNumHs() + 2, 0)
    mol.UpdatePropertyCache(strict=False)
    # LCP, 1L5: sanitizing draws a nitro or halogen oxoacid charge-separated; charges return as *sub*'s bonds draw.
    redrawn = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()][:n]) - sub.charge
    titrated = _titrate(mol, ph)
    titrated = mol if titrated is None else titrated
    if np.isfinite(donor_ph).any():
        titrated = _restate_coordinating_donors(mol, titrated, donor_ph)
    titrated.UpdatePropertyCache(strict=False)
    stated_hydrogens = stated_hydrogens.copy()
    for index in np.flatnonzero(coordinating >= 0):
        if titrated.GetAtomWithIdx(int(index)).GetTotalNumHs() > coordinating[index]:
            stated_hydrogens[index] = coordinating[index]
    # 6DMZ HIS 8 NE2: a neutral imidazole nothing else decides, neither N substituted, takes its H on the tele N (the
    # one whose ring neighbour is unsubstituted, histidine's NE2), whichever Kekule form its bonds draw.
    imidazole_targets = imidazole_targets.copy()
    rings = imidazole_rings(sub)
    for pair, tele_n in zip(rings.n[rings.open], rings.tele[rings.open], strict=True):
        free = (imidazole_targets[pair] < 0).all() and (stated_hydrogens[pair] < 0).all() and tele_n.any()
        if free and sum(titrated.GetAtomWithIdx(int(n)).GetTotalNumHs() for n in pair) == 1:
            imidazole_targets[pair] = tele_n

    ring_bonds = np.zeros((0, 3), dtype=np.int64)
    chosen = (imidazole_targets[rings.n] >= 0).all(axis=1)
    if chosen.any():
        released = _release_ring_bond_orders(titrated, rings.atoms[chosen])
        for pair in rings.n[chosen]:
            # 1QS0 HIS NE2: a nitrogen with a proton or a third heavy bond is substituted; a ring substituted on
            # both nitrogens, or on neither, carries its charge on its first one.
            substituted = imidazole_targets[pair] + [titrated.GetAtomWithIdx(int(n)).GetDegree() > 2 for n in pair]
            for first, index in zip((True, False), pair, strict=True):
                _set_hydrogens(titrated, index, imidazole_targets[index], int(substituted.sum()) - 1 if first else 0)
        # Other atoms keep the valences the input drew (a borate at charge 0); only the rings are re-derived.
        titrated.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(titrated, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            kekule = Chem.Mol(titrated)
            Chem.Kekulize(kekule)
        except Chem.MolSanitizeException as exc:  # 1QS0 HIS NE2 read back: no Kekule form fits
            return _kept_as_given(sub, exc)
        bonds = [kekule.GetBondBetweenAtoms(left, right) for left, right in released]
        types = [int(RDKIT_BOND_TYPE_TO_BIOTITE[(bond.GetBondType(), bond.GetIsAromatic())]) for bond in bonds]
        ring_bonds = np.column_stack([np.array(released, dtype=np.int64).reshape(-1, 2), types])

    for index in np.flatnonzero(stated_hydrogens >= 0):
        _set_hydrogens(titrated, index, stated_hydrogens[index], sub.charge[index] + redrawn[index])
    titrated.UpdatePropertyCache(strict=False)

    if titrated.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError("Dimorphite-DL returned a molecule with a different number of atoms.")
    atoms = list(titrated.GetAtoms())[:n]
    charge = (np.array([atom.GetFormalCharge() for atom in atoms]) - redrawn).astype(sub.charge.dtype)
    hydrogens = np.array([atom.GetTotalNumHs() for atom in atoms], dtype=np.int64)
    return charge, hydrogens, ring_bonds
