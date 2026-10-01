"""Module for generating rdkit molobj/smiles/molecular graph from free atoms.

Main implementation by Jan H. Jensen, based on the paper

    Yeonjoon Kim and Woo Youn Kim
    "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
    to Three-Dimensional Geometry"
    Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777
    DOI: 10.1002/bkcs.10334

Modified by Maria Harris Rasmussen 2024
"""

import copy
import itertools

try:
    from rdkit.Chem import rdEHTTools
except ImportError:
    rdEHTTools = None

import sys
from collections import defaultdict

import networkx as nx
import numpy as np
from rdkit import Chem

global __ATOM_LIST__
__ATOM_LIST__ = [
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
    "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca",
    "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
    "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y", "zr",
    "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn",
    "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd",
    "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb",
    "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg",
    "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th",
    "pa", "u", "np", "pu",
]

global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4, 2]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]
atomic_valence[9] = [1]
atomic_valence[13] = [3, 4]
atomic_valence[14] = [4]
atomic_valence[15] = [3, 5]
atomic_valence[16] = [2, 4, 6]
atomic_valence[17] = [1]
atomic_valence[18] = [0]
atomic_valence[32] = [4]
atomic_valence[33] = [5, 3]
atomic_valence[35] = [1]
atomic_valence[34] = [2]
atomic_valence[52] = [2]
atomic_valence[53] = [1]
atomic_valence[21] = [20]
atomic_valence[22] = [20]
atomic_valence[23] = [20]
atomic_valence[24] = [20]
atomic_valence[25] = [20]
atomic_valence[26] = [20]
atomic_valence[27] = [20]
atomic_valence[28] = [20]
atomic_valence[29] = [20]
atomic_valence[30] = [20]
atomic_valence[39] = [20]
atomic_valence[40] = [20]
atomic_valence[41] = [20]
atomic_valence[42] = [20]
atomic_valence[43] = [20]
atomic_valence[44] = [20]
atomic_valence[45] = [20]
atomic_valence[46] = [20]
atomic_valence[47] = [20]
atomic_valence[48] = [20]
atomic_valence[57] = [20]
atomic_valence[72] = [20]
atomic_valence[73] = [20]
atomic_valence[74] = [20]
atomic_valence[75] = [20]
atomic_valence[76] = [20]
atomic_valence[77] = [20]
atomic_valence[78] = [20]
atomic_valence[79] = [20]
atomic_valence[80] = [20]

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
atomic_valence_electrons[33] = 5
atomic_valence_electrons[35] = 7
atomic_valence_electrons[34] = 6
atomic_valence_electrons[52] = 6
atomic_valence_electrons[53] = 7


def str_atom(atom):
    """Convert integer atom to string atom."""
    global __ATOM_LIST__
    atom = __ATOM_LIST__[atom - 1]
    return atom


def int_atom(atom):
    """Convert str atom to integer atom."""
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_UA(maxValence_list, valence_list):
    """"""
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """"""
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, DU, use_graph=use_graph)[0]
    return BO


def valences_not_too_large(BO, valences):
    """"""
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False
    return True


def charge_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    Q = 0
    q_list = []

    if allow_charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if (
                    not allow_carbenes
                    and number_of_single_bonds_to_C == 2
                    and BO_valences[i] == 2
                ):
                    print("found illegal carbene")
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1
            if q != 0:
                q_list.append(q)
    return charge == Q


def BO_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    """Sanity of bond-orders.

    args:
        BO -
        AC -
        charge -
        DU -

    optional
        allow_charges_fragments -

    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments,
        allow_carbenes=True,
    )

    if check_charge and check_sum:
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """"""
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 6 and BO_valence == 2:
        charge = 0
    elif atom == 13:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    elif atom == 16 and BO_valence == 4:
        charge = 0
    elif atom == 16 and BO_valence == 5:
        charge = 1
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
    use_atom_maps=True,
):
    """Based on code written by Paolo Toscani.

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity
    """

    length_bo = len(BO_matrix)
    length_atoms = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if length_bo != length_atoms:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(
                length_bo, length_atoms
            )
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(length_bo):
        for j in range(i + 1, length_bo):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.RemoveBond(i, j)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge,
            use_atom_maps=use_atom_maps,
        )
    else:
        mol = set_atomic_radicals(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            use_atom_maps=use_atom_maps,
        )

    Chem.SanitizeMol(mol)

    return mol


def set_atomic_charges(
    mol,
    atoms,
    atomic_valence_electrons,
    BO_valences,
    BO_matrix,
    mol_charge,
    use_atom_maps=True,
):
    """"""
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if BO_valences[i] == 2:
                a.SetNumRadicalElectrons(2)
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    return mol


def set_atomic_radicals(
    mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps=True
):
    """The number of radical electrons = absolute atomic charge."""
    atomic_valence[8] = [2, 1]
    atomic_valence[7] = [3, 2]
    atomic_valence[6] = [4, 2]

    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """"""
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, DU, use_graph=True):
    """"""
    N_UA = 10000
    matching_ids = dict()
    matching_ids2 = dict()
    for i, du in zip(UA, DU):
        if du > 1:
            matching_ids[i] = N_UA
            matching_ids2[N_UA] = i
            N_UA += 1

    bonds = get_bonds(UA, AC)
    for i, j in bonds:
        if i in matching_ids:
            bonds.append(tuple(sorted([matching_ids[i], j])))

        elif j in matching_ids:
            bonds.append(tuple(sorted([i, matching_ids[j]])))

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        UA_pair = UA_pairs[0]

        remove_pairs = []
        add_pairs = []
        for i, j in UA_pair:
            if i in matching_ids2 and j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], matching_ids2[j]]))
            elif i in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], j]))
            elif j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([i, matching_ids2[j]]))
        for p1, p2 in zip(remove_pairs, add_pairs):
            UA_pair.remove(p1)
            UA_pair.append(p2)
        return [UA_pair]

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(
    AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True
):
    """Implemenation of algorithm shown in Figure 2.

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure
    """

    global atomic_valence
    global atomic_valence_electrons

    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if atomicNum == 6 and valence == 1:
            possible_valence.remove(2)
        if atomicNum == 6 and not allow_carbenes and valence == 2:
            possible_valence.remove(2)
        if atomicNum == 6 and valence == 2:
            possible_valence.append(3)
        if atomicNum == 16 and valence == 1:
            possible_valence = [1, 2]

        if not possible_valence:
            print(
                "Valence of atom",
                i,
                "is",
                valence,
                "which bigger than allowed max",
                max(atomic_valence[atomicNum]),
                ". Stopping",
            )
            sys.exit()
        valences_list_of_lists.append(possible_valence)

    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    O_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 8
    ]
    N_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 7
    ]
    C_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 6
    ]
    P_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 15
    ]
    S_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 16
    ]

    O_sums = []
    for v_list in itertools.product(*O_valences):
        O_sums.append(v_list)

    N_sums = []
    for v_list in itertools.product(*N_valences):
        N_sums.append(v_list)

    C_sums = []
    for v_list in itertools.product(*C_valences):
        C_sums.append(v_list)

    P_sums = []
    for v_list in itertools.product(*P_valences):
        P_sums.append(v_list)

    S_sums = []
    for v_list in itertools.product(*S_valences):
        S_sums.append(v_list)

    order_dict = dict()
    for i, v_list in enumerate(
        itertools.product(*[O_sums, N_sums, C_sums, P_sums, S_sums])
    ):
        order_dict[v_list] = i

    valence_order_list = []
    for valence_list in valences_list:
        C_sum = []
        N_sum = []
        O_sum = []
        P_sum = []
        S_sum = []
        for v, atomicNum in zip(valence_list, atoms):
            if atomicNum == 6:
                C_sum.append(v)
            if atomicNum == 7:
                N_sum.append(v)
            if atomicNum == 8:
                O_sum.append(v)
            if atomicNum == 15:
                P_sum.append(v)
            if atomicNum == 16:
                S_sum.append(v)

        order_idx = order_dict[
            (tuple(O_sum), tuple(N_sum), tuple(C_sum), tuple(P_sum), tuple(S_sum))
        ]
        valence_order_list.append(order_idx)

    sorted_valences_list = [
        y
        for x, y in sorted(
            zip(valence_order_list, list(itertools.product(*valences_list_of_lists)))
        )
    ]

    for valences in sorted_valences_list:
        UA, DU_from_AC = get_UA(valences, AC_valence)
        check_len = len(UA) == 0
        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, DU_from_AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
            charge_OK = charge_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )

            if status:
                return BO, atomic_valence_electrons
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(
    mol,
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    use_atom_maps=True,
    allow_carbenes=True,
):
    """"""

    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=allow_carbenes,
    )

    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_atom_maps=use_atom_maps,
    )

    if Chem.GetFormalCharge(mol) != charge:
        return None

    return mol


def get_proto_mol(atoms):
    """"""
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """"""
    atomic_symbols = []
    xyz_coordinates = []
    charge = 0

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def xyz2AC(atoms, xyz, charge, use_huckel=False, use_obabel=False):
    """Atoms and coordinates to atom connectivity (AC)

    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge

    optional:
        use_huckel - Use Huckel method for atom connecitivty
        use_obabel - Use Opne Babel method for atom connectivity

    returns
        ac - atom connectivity matrix
        mol - rdkit molecule
    """
    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    elif use_obabel:
        return xyz2AC_obabel(atoms, xyz)
    else:
        return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):
    mol = get_proto_mol(atoms)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol


def get_AC(mol, covalent_factor=1.3):
    """Generate adjacent matrix from atoms and coordinates.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0
    is not

    covalent_factor - 1.3 is an arbitrary factor

    args:
        mol - rdkit molobj with 3D conformer

    optional
        covalent_factor - increase covalent bond length threshold with factor

    returns:
        AC - adjacent matrix
    """
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def xyz2AC_huckel(atomicNumList, xyz, charge):
    """Args.

        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge

    returns
        ac - atom connectivity
        mol - rdkit molecule
    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms, num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge)

    passed, result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair_pop = abs(tri[j, i])
            if pair_pop >= 0.2:
                AC[i, j] = 1
                AC[j, i] = 1

    dMat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        N_con = np.sum(AC[i, :])
        while N_con > max(atomic_valence[a_i.GetAtomicNum()]):
            AC = remove_weakest_bond(mol, i, AC, dMat, pt)
            N_con = np.sum(AC[i, :])

    return AC, mol


def remove_weakest_bond(mol, atom_idx, AC, dMat, pt):
    extra_bond_lengths = []
    bond_atoms = np.nonzero(AC[atom_idx, :])[0]
    a_i = mol.GetAtomWithIdx(atom_idx)
    rcovi = pt.GetRcovalent(a_i.GetAtomicNum())
    for j in bond_atoms:
        a_j = mol.GetAtomWithIdx(int(j))
        rcovj = pt.GetRcovalent(a_j.GetAtomicNum())
        extra_bond_length = dMat[atom_idx, j] - rcovj - rcovi
        extra_bond_lengths.append(extra_bond_length)

    longest_bond_index = bond_atoms[np.argmax(extra_bond_lengths)]
    AC[atom_idx, longest_bond_index] = 0
    AC[longest_bond_index, atom_idx] = 0

    return AC


def xyz2AC_obabel(atoms, xyz, tolerance=0.45):
    """Generate adjacent matrix from atoms and coordinates in a way similar to
    open babels.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0
    is not

    tolerance - 0.45Å is from the open babel paper

    args:
        mol - rdkit molobj with 3D conformer

    optional
        tolerance - atoms connected if distance is shorter than sum of atomic
        radii + tolerance. If too many bonds to an atom; break longest bond

    returns:
        AC - adjacency matrix
    """
    global atomic_valence
    atomic_valence[6] = [4, 2]

    mol = get_proto_mol(atoms)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum())
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum())
            if dMat[i, j] <= Rcov_i + Rcov_j + tolerance:
                AC[i, j] = 1
                AC[j, i] = 1

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        N_con = np.sum(AC[i, :])
        while N_con > max(atomic_valence[a_i.GetAtomicNum()]):
            AC = remove_weakest_bond(mol, i, AC, dMat, pt)
            N_con = np.sum(AC[i, :])

    return AC, mol


def chiral_stereo_check(mol):
    """Find and embed chiral information into the model based on the
    coordinates.

    args:
        mol - rdkit molecule, with embeded conformer
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return


def xyz2mol(
    atoms,
    coordinates,
    charge=0,
    allow_charged_fragments=True,
    use_graph=True,
    use_huckel=False,
    use_obabel=False,
    embed_chiral=True,
    use_atom_maps=True,
):
    """Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mols - list of rdkit molobjects
    """

    AC, mol = xyz2AC(
        atoms, coordinates, charge, use_huckel=use_huckel, use_obabel=use_obabel
    )

    new_mol = AC2mol(
        mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        use_atom_maps=use_atom_maps,
    )

    if embed_chiral:
        chiral_stereo_check(new_mol)

    return new_mol


def canonicalize_smiles(structure_smiles):
    """Remove all structural info an atom mapping information."""
    mol = Chem.MolFromSmiles(structure_smiles, sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(mol)

    return canonical_smiles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage="%(prog)s [options] molecule.xyz")
    parser.add_argument("structure", metavar="structure", type=str)
    parser.add_argument("-s", "--sdf", action="store_true", help="Dump sdf file")
    parser.add_argument(
        "--ignore-chiral", action="store_true", help="Ignore chiral centers"
    )
    parser.add_argument(
        "--no-charged-fragments", action="store_true", help="Allow radicals to be made"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Run xyz2mol without networkx dependencies",
    )
    parser.add_argument(
        "--use-huckel",
        action="store_true",
        help="Use Huckel method for atom connectivity",
    )
    parser.add_argument(
        "--use-obabel",
        action="store_true",
        help="Use Open Babel way of obtaining atom connectivity; recommended for radicals",
    )
    parser.add_argument(
        "-o",
        "--output-format",
        action="store",
        type=str,
        help="Output format [smiles,sdf] (default=sdf)",
    )
    parser.add_argument(
        "-c",
        "--charge",
        action="store",
        metavar="int",
        type=int,
        help="Total charge of the system",
    )
    parser.add_argument(
        "--use-atom-maps",
        action="store_true",
        help="Label atoms with map numbers according to their order in the .xyz file",
    )

    args = parser.parse_args()

    filename = args.structure

    charged_fragments = not args.no_charged_fragments

    quick = not args.no_graph

    embed_chiral = not args.ignore_chiral

    atoms, charge, xyz_coordinates = read_xyz_file(filename)

    use_huckel = args.use_huckel

    use_obabel = args.use_obabel

    if args.charge is not None:
        charge = int(args.charge)

    use_atom_maps = args.use_atom_maps
    if not charged_fragments:
        atomic_valence[8] = [2, 1]
        atomic_valence[7] = [3, 2]
        atomic_valence[6] = [4, 2]

    mols = xyz2mol(
        atoms,
        xyz_coordinates,
        charge=charge,
        use_graph=quick,
        allow_charged_fragments=charged_fragments,
        embed_chiral=embed_chiral,
        use_huckel=use_huckel,
        use_obabel=use_obabel,
        use_atom_maps=use_atom_maps,
    )

    for mol in [mols]:
        if args.output_format == "sdf":
            txt = Chem.MolToMolBlock(mol)
            print(txt)

        else:
            isomeric_smiles = not args.ignore_chiral
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)

            smiles = canonicalize_smiles(smiles)
