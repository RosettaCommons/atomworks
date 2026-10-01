"Module for the xyz2mol functionality for TMCs"

import argparse
import logging
import signal
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdchem, rdEHTTools, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

from atomworks.external.xyz2mol_tm.xyz2mol_local import (
    AC2mol,
    chiral_stereo_check,
    read_xyz_file,
    xyz2AC_obabel,
)

# fmt: off
TRANSITION_METALS = ["Sc","Ti","V","Cr","Mn","Fe","Co","La","Ni","Cu","Zn",
                     "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Lu",
                     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
]

TRANSITION_METALS_NUM = [21,22,23,24,25,26,27,57,28,29,30,39,40,41,
                         42,43,44,45,46,47,48,71,72,73,74,75,76,77,78,79,80,
]


ALLOWED_OXIDATION_STATES = {
    "Sc": [3],
    "Ti": [3, 4],
    "V": [2, 3, 4, 5],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 6, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2],
    "Cu": [1, 2],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [3, 4, 5],
    "Mo": [2, 3, 4, 5, 6],
    "Tc": [2, 3, 4, 5, 6, 7],
    "Ru": [2, 3, 4, 5, 6, 7, 8],
    "Rh": [1, 3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [3, 4, 5],
    "W": [2, 3, 4, 5, 6],
    "Re": [2, 3, 4, 5, 6, 7],
    "Os": [3, 4, 5, 6, 7, 8],
    "Ir": [1, 3],
    "Pt": [2, 4],
    "Au": [1, 3],
    "Hg": [1, 2],
}
# fmt: on

logger = logging.getLogger(__name__)

params = Chem.MolStandardize.rdMolStandardize.MetalDisconnectorOptions()
params.splitAromaticC = True
params.splitGrignards = True
params.adjustCharges = False

MetalNon_Hg = "[#3,#11,#12,#19,#13,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]~[B,#6,#14,#15,#33,#51,#16,#34,#52,Cl,Br,I,#85]"

pt = GetPeriodicTable

global atomic_valence_electrons

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[13] = 3
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[18] = 8
atomic_valence_electrons[32] = 4
atomic_valence_electrons[33] = 5  # As
atomic_valence_electrons[35] = 7
atomic_valence_electrons[34] = 6
atomic_valence_electrons[53] = 7

# TMs
atomic_valence_electrons[21] = 3  # Sc
atomic_valence_electrons[22] = 4  # Ti
atomic_valence_electrons[23] = 5  # V
atomic_valence_electrons[24] = 6  # Cr
atomic_valence_electrons[25] = 7  # Mn
atomic_valence_electrons[26] = 8  # Fe
atomic_valence_electrons[27] = 9  # Co
atomic_valence_electrons[28] = 10  # Ni
atomic_valence_electrons[29] = 11  # Cu
atomic_valence_electrons[30] = 12  # Zn

atomic_valence_electrons[39] = 3  # Y
atomic_valence_electrons[40] = 4  # Zr
atomic_valence_electrons[41] = 5  # Nb
atomic_valence_electrons[42] = 6  # Mo
atomic_valence_electrons[43] = 7  # Tc
atomic_valence_electrons[44] = 8  # Ru
atomic_valence_electrons[45] = 9  # Rh
atomic_valence_electrons[46] = 10  # Pd
atomic_valence_electrons[47] = 11  # Ag
atomic_valence_electrons[48] = 12  # Cd

atomic_valence_electrons[57] = 3  # La
atomic_valence_electrons[72] = 4  # Hf
atomic_valence_electrons[73] = 5  # Ta
atomic_valence_electrons[74] = 6  # W
atomic_valence_electrons[75] = 7  # Re
atomic_valence_electrons[76] = 8  # Os
atomic_valence_electrons[77] = 9  # Ir
atomic_valence_electrons[78] = 10  # Pt
atomic_valence_electrons[79] = 11  # Au
atomic_valence_electrons[80] = 12  # Hg


def shell(cmd, shell=False):
    if shell:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        cmd = cmd.split()
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    output, err = p.communicate()
    return output


# NOTE: Changing from source code to take as input and return mol object (vs. smiles)
def fix_NO2(m):
    """Localizes nitro groups that have been assigned a charge of -2 (neutral
    Nitrogen bound to two negatively charged Oxygen atoms).

    These groups are changed to reflect the correct neutral
    configuration of a nitro group. The oxidation state on the
    transition metal is changed accordingly.
    """
    emol = Chem.RWMol(m)
    patt = Chem.MolFromSmarts(
        "[#8-]-[#7+0]-[#8-].[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]"
    )
    matches = emol.GetSubstructMatches(patt)
    for a1, a2, a3, a4 in matches:
        if not emol.GetBondBetweenAtoms(a1, a4) and not emol.GetBondBetweenAtoms(
            a3, a4
        ):
            tm = emol.GetAtomWithIdx(a4)
            o1 = emol.GetAtomWithIdx(a1)
            n = emol.GetAtomWithIdx(a2)
            tm_charge = tm.GetFormalCharge()
            new_charge = tm_charge - 2
            tm.SetFormalCharge(new_charge)
            n.SetFormalCharge(+1)
            o1.SetFormalCharge(0)
            emol.RemoveBond(a1, a2)
            emol.AddBond(a1, a2, rdchem.BondType.DOUBLE)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


# NOTE: Changing from source code to take as input and return mol object (vs. smiles)
def fix_equivalent_Os(m):
    """Localizes and fixes where a neutral atom is coordinating to the metal
    but connected ro a negatively charged atom through resonane.

    The charge is moved to the coordinating atom and charges fixed
    accordingly.
    """
    emol = Chem.RWMol(m)

    patt = Chem.MolFromSmarts("[#6-,#7-,#8-,#15-,#16-]-[*]=[#6,#7,#8,#15,#16]")

    matches = emol.GetSubstructMatches(patt)
    used_atom_ids_1 = []
    used_atom_ids_3 = []
    for atom in emol.GetAtoms():
        if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
            neighbor_idxs = [a.GetIdx() for a in atom.GetNeighbors()]
            for a1, a2, a3 in matches:
                if (
                    a3 in neighbor_idxs
                    and a1 not in neighbor_idxs
                    and a1 not in used_atom_ids_1
                    and a3 not in used_atom_ids_3
                ):
                    used_atom_ids_1.append(a1)
                    used_atom_ids_3.append(a3)

                    emol.RemoveBond(a1, a2)
                    emol.AddBond(a1, a2, Chem.rdchem.BondType.DOUBLE)
                    emol.RemoveBond(a2, a3)
                    emol.AddBond(a2, a3, Chem.rdchem.BondType.SINGLE)
                    emol.GetAtomWithIdx(a1).SetFormalCharge(0)
                    emol.GetAtomWithIdx(a3).SetFormalCharge(-1)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def get_proposed_ligand_charge(ligand_mol, cutoff=-10):
    """Runs an extended Hückel calculation for the ligand defined in
    ligand_mol.

    A suggested charge is found by filling electrons in orbitals <-10eV
    and omparing with total number of valence electrons. If charge is >=
    1 (<-1) and the LUMO (HOMO) is low (high) in energy, two additional
    electrons are added (removed). The suggested charge is returned.
    """
    valence_electrons = 0
    passed, result = rdEHTTools.RunMol(ligand_mol)
    for a in ligand_mol.GetAtoms():
        valence_electrons += atomic_valence_electrons[a.GetAtomicNum()]

    passed, result = rdEHTTools.RunMol(ligand_mol)
    N_occ_orbs = sum(1 for i in result.GetOrbitalEnergies() if i < cutoff)
    charge = valence_electrons - 2 * N_occ_orbs
    percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]
    if N_occ_orbs == len(result.GetOrbitalEnergies()):
        percieved_lumo = np.nan
    else:
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge >= 1 and percieved_lumo < -9:
        N_occ_orbs += 1
        charge += -2
        logger.debug("added two more electrons:", charge, percieved_lumo)
        percieved_lumo = result.GetOrbitalEnergies()[N_occ_orbs]
    while charge < -1 and percieved_homo > -10.2:
        N_occ_orbs -= 1
        charge += 2
        logger.debug("removed two electrons:", charge, percieved_homo)
        percieved_homo = result.GetOrbitalEnergies()[N_occ_orbs - 1]

    return charge


def get_basic_mol(xyz_file, overall_charge):
    """A basic mol-object (that can be usedto do an extended Hückel calculation
    is constructed based on the adjacency matrix evaluated from the xyz-
    coordinates.

    All bonds are single bonds, and charges are only asigned if
    necessary to work with it, i.e. a Nitrogen with four neihbors gets a
    +1 charge, Boron with 4 neighbors gets a -1 charge and oxygen with
    three neighbors gets a +1 charge.
    """
    atoms, _, xyz_coords = read_xyz_file(xyz_file)

    # AC, mol = xyz2AC_huckel(atoms, xyz_coords, overall_charge)
    AC, mol = xyz2AC_obabel(atoms, xyz_coords)
    tm_indxs = [atoms.index(tm) for tm in TRANSITION_METALS_NUM if tm in atoms]

    rwMol = Chem.RWMol(mol)
    length_ac = len(AC)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for i in range(length_ac):
        for j in range(i + 1, length_ac):
            bo = int(round(AC[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 7:
            # explicit_valence = np.sum(AC[i])
            explicit_valence = sum(
                [ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs]
            )
            if explicit_valence == 4:
                a.SetFormalCharge(1)
        if a.GetAtomicNum() == 5:
            # Boron with 4 explicit bonds should be negative
            explicit_valence = sum(
                [ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs]
            )
            if explicit_valence == 4:
                a.SetFormalCharge(-1)
        if a.GetAtomicNum() == 8:
            explicit_valence = sum(
                [ele for idx, ele in enumerate(AC[i]) if idx not in tm_indxs]
            )
            if explicit_valence == 3:
                a.SetFormalCharge(1)

    return mol


def lig_checks(lig_mol, coordinating_atoms):
    """Sending proposed ligand mol object through series of checks.

    - neighbouring coordinating atoms must be connected by pi-bond, aromatic bond (haptic), conjugated system
    - If I have two neighbouring identical charges -> fail, I would rather change the charge and make a bond
     -> suggest new charge adding/subtracting electrons based on these neighbouring charges
    - count partial charges: partial charges that are not negative on ligand coordinating atoms count against this ligand
      -> loop through resonance forms to see if any live up to this, then choose that one.
      -> partial positive charge on coordinating atom is big red flag
      -> If "bad" partial charges still exists suggest a new charge: add/subtract electrons based on the values of the partial charges
    """
    res_mols = rdchem.ResonanceMolSupplier(lig_mol)
    if len(res_mols) == 0:
        res_mols = rdchem.ResonanceMolSupplier(
            lig_mol, flags=Chem.ALLOW_INCOMPLETE_OCTETS
        )
    # Check for neighbouring coordinating atoms:
    possible_lig_mols = []

    for res_mol in res_mols:
        positive_atoms = []
        negative_atoms = []
        N_aromatic = 0
        for a in res_mol.GetAtoms():
            if a.GetIsAromatic():
                N_aromatic += 1
            if a.GetFormalCharge() > 0:
                positive_atoms.append(a.GetIdx())
            if a.GetFormalCharge() < 0 and a.GetIdx() not in coordinating_atoms:
                negative_atoms.append(a.GetIdx())

        possible_lig_mols.append(
            (res_mol, len(positive_atoms), len(negative_atoms), N_aromatic)
        )
    return possible_lig_mols


def get_lig_mol(mol, charge, coordinating_atoms):
    """A sanitizable mol object is created for the ligand, taking into account
    the checks defined in lig_checks.

    We try different charge settings and settings where carbenes are
    allowed/not allowed in case no perfect solution (no partial charges
    on other than the coordinating atoms) can be found. Finally best
    found solution based on criteria in lig_checks is returned.
    """

    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    AC = Chem.rdmolops.GetAdjacencyMatrix(mol)
    lig_mol = AC2mol(
        mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False
    )
    if not lig_mol and charge >= 0:
        charge += -2
        lig_mol = AC2mol(
            mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False
        )
        if not lig_mol:
            return None, charge
    if not lig_mol and charge < 0:
        charge += 2
        lig_mol = AC2mol(
            mol, AC, atoms, charge, allow_charged_fragments=True, use_atom_maps=False
        )
        if not lig_mol:
            charge += -4
            lig_mol = AC2mol(
                mol,
                AC,
                atoms,
                charge,
                allow_charged_fragments=True,
                use_atom_maps=False,
            )
            if not lig_mol:
                return None, charge

    possible_res_mols = lig_checks(lig_mol, coordinating_atoms)
    best_res_mol, lowest_pos, lowest_neg, highest_aromatic = possible_res_mols[0]
    for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in possible_res_mols:
        if N_aromatic > highest_aromatic:
            best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                res_mol,
                N_pos_atoms,
                N_neg_atoms,
                N_aromatic,
            )
        if (
            N_aromatic == highest_aromatic
            and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
        ):
            best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
    if lowest_pos + lowest_neg == 0:
        return best_res_mol, charge

    lig_mol_no_carbene = AC2mol(
        mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=True,
        use_atom_maps=False,
        allow_carbenes=False,
    )
    allow_carbenes = True

    if lig_mol_no_carbene:
        res_mols_no_carbenes = lig_checks(lig_mol_no_carbene, coordinating_atoms)
        for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in res_mols_no_carbenes:
            if (
                N_aromatic > highest_aromatic
                and N_pos_atoms + N_neg_atoms <= lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                    res_mol,
                    N_pos_atoms,
                    N_neg_atoms,
                    N_aromatic,
                )
            if (
                N_aromatic == highest_aromatic
                and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
            ):
                best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
                allow_carbenes = False

    if lowest_pos + lowest_neg == 0:
        logger.debug("found opt solution without carbenes")
        return best_res_mol, charge

    if lowest_pos - lowest_neg + charge < 0:
        new_charge = charge + 2
    else:
        new_charge = charge - 2  # if 0 maybe I should try both

    new_lig_mol = AC2mol(
        mol,
        AC,
        atoms,
        new_charge,
        allow_charged_fragments=True,
        use_atom_maps=False,
        allow_carbenes=allow_carbenes,
    )
    if not new_lig_mol:
        return best_res_mol, charge
    new_possible_res_mols = lig_checks(new_lig_mol, coordinating_atoms)
    for res_mol, N_pos_atoms, N_neg_atoms, N_aromatic in new_possible_res_mols:
        if N_aromatic > highest_aromatic:
            best_res_mol, lowest_pos, lowest_neg, highest_aromatic = (
                res_mol,
                N_pos_atoms,
                N_neg_atoms,
                N_aromatic,
            )
            charge = new_charge
        if (
            N_aromatic == highest_aromatic
            and N_pos_atoms + N_neg_atoms < lowest_pos + lowest_neg
        ):
            best_res_mol, lowest_pos, lowest_neg = res_mol, N_pos_atoms, N_neg_atoms
            charge = new_charge

    return best_res_mol, charge


# NOTE: Modified to directly take mol object as input (vs. xyz file path)
def get_tmc_mol(mol, overall_charge, with_stereo=False):
    """Get TMC mol object from given xyz file.

    Args:
        xyz_file (str) : Path to TMC xyz file
        overall_charge (int): Overall charge of TMC
        with_stereo (bool): Whether to percieve stereochemistry from the 3D data

    Returns:
        tmc_mol (rdkit.Chem.rdchem.Mol): TMC mol object
    """
    tmc_idx = None
    for a in mol.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            tmc_idx = a.GetIdx()

    if tmc_idx is None:
        raise Exception(
            "Found no TM in the input file. Please supply an xyz file with a TM"
        )

    coordinating_atoms = np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol)[tmc_idx, :])[
        0
    ]

    mdis = rdMolStandardize.MetalDisconnector(params)
    mdis.SetMetalNon(Chem.MolFromSmarts(MetalNon_Hg))
    frags = mdis.Disconnect(mol)
    frag_mols = rdmolops.GetMolFrags(frags, asMols=True)

    total_lig_charge = 0
    tm_idx = None
    lig_list = []
    for i, f in enumerate(frag_mols):
        m = Chem.Mol(f)
        atoms = m.GetAtoms()
        for atom in atoms:
            if atom.GetAtomicNum() in TRANSITION_METALS_NUM:
                tm_idx = i
                break
        else:
            lig_charge = get_proposed_ligand_charge(f)

            lig_coordinating_atoms = [
                a.GetIdx()
                for a in m.GetAtoms()
                if a.GetIntProp("__origIdx") in coordinating_atoms
            ]
            lig_mol, lig_charge = get_lig_mol(m, lig_charge, lig_coordinating_atoms)
            if not lig_mol:
                return None

            total_lig_charge += lig_charge
            lig_list.append(lig_mol)

    if tm_idx is None:
        raise Exception(
            "Found no TM in the input file. Please supply an xyz file with a TM"
        )

    tm = Chem.RWMol(frag_mols[tm_idx])
    tm_ox = overall_charge - total_lig_charge

    len(tm.GetAtoms())

    for a in tm.GetAtoms():
        if a.GetAtomicNum() in TRANSITION_METALS_NUM:
            a.SetFormalCharge(tm_ox)

    for lmol in lig_list:
        tm = Chem.CombineMols(tm, lmol)

    emol = Chem.RWMol(tm)
    coordinating_atoms_idx = [
        a.GetIdx()
        for a in emol.GetAtoms()
        if a.GetIntProp("__origIdx") in coordinating_atoms
    ]
    tm_idx = [
        a.GetIdx() for a in emol.GetAtoms() if a.GetIntProp("__origIdx") == tmc_idx
    ][0]
    dMat = Chem.Get3DDistanceMatrix(emol)
    cut_atoms = []
    for i, j in combinations(coordinating_atoms_idx, 2):
        bond = emol.GetBondBetweenAtoms(int(i), int(j))
        if bond and abs(dMat[i, tm_idx] - dMat[j, tm_idx]) >= 0.4:
            logger.debug(
                "Haptic bond pattern with too great distance:",
                dMat[i, tm_idx],
                dMat[j, tm_idx],
            )
            if dMat[i, tm_idx] > dMat[j, tm_idx] and i in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(i)
                cut_atoms.append(i)
            if dMat[j, tm_idx] > dMat[i, tm_idx] and j in coordinating_atoms_idx:
                coordinating_atoms_idx.remove(j)
                cut_atoms.append(j)
    for j in cut_atoms:
        for i in coordinating_atoms_idx:
            bond = emol.GetBondBetweenAtoms(int(i), int(j))
            if (
                bond
                and dMat[i, tm_idx] - dMat[j, tm_idx] >= -0.1
                and i in coordinating_atoms_idx
            ):
                coordinating_atoms_idx.remove(i)

    for i in coordinating_atoms_idx:
        if emol.GetBondBetweenAtoms(i, tm_idx):
            continue
        emol.AddBond(i, tm_idx, Chem.BondType.DATIVE)

    # Fix specific cases
    # NOTE: Changing from source code to operate on mol object directly
    mol = fix_equivalent_Os(emol)
    mol = fix_NO2(mol)

    Chem.SanitizeMol(mol)
    if with_stereo:
        chiral_stereo_check(mol)

    return mol
