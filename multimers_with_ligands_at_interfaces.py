"""
Identifies all multimers with a ligand at the interface between two identical chains.
Uses the atomworks package from the Institute for Protein Design at the University of Washington to parse the CIF files.

Written by Frank DiMaio and Nate Corley, 2025
"""

from atomworks.io import parse
import numpy as np
from atomworks.enums import ChainType
from os import PathLike
import os
import signal
import argparse


def get_peptides_and_ligands(result_dict):
    """Identify from the parsed CIF file:
        - Equivalent peptide chains
        - All ligand chains

    Returns:
        tuple: A tuple containing a dictionary of equivalent peptide chains and a list of ligand chains.
    """
    equiv_peptide_chains = {}
    ligand_chains = []

    # Loop over all chains in the structure
    allchains = result_dict["chain_info"].keys()
    for chain in allchains:
        # Add ligands with more than 5 atoms to the running list of ligands
        if result_dict["chain_info"][chain]["chain_type"] == ChainType.NON_POLYMER:
            atomarray = next(iter(result_dict["assemblies"].values()))[0]
            natoms = sum(atomarray.chain_id == chain)
            ligid = result_dict["chain_info"][chain]["res_name"][0]
            if natoms > 5:
                ligand_chains.append([chain, ligid, natoms])
        if result_dict["chain_info"][chain]["chain_type"] != ChainType.POLYPEPTIDE_L:
            continue
        matched = False

        # For polypeptide chains, check if the sequence is the same as any other chain
        for chain_ref in equiv_peptide_chains.keys():
            if (
                result_dict["chain_info"][chain]["processed_entity_non_canonical_sequence"]
                == result_dict["chain_info"][chain_ref]["processed_entity_non_canonical_sequence"]
            ):
                matched = True
                equiv_peptide_chains[chain_ref].append(chain)
                break
        if not matched:
            equiv_peptide_chains[chain] = [chain]

    # Remove chains with no equivalent chains
    allchains = list(equiv_peptide_chains.keys())
    for chain in allchains:
        if len(equiv_peptide_chains[chain]) == 1:
            del equiv_peptide_chains[chain]

    return equiv_peptide_chains, ligand_chains


def get_contacts(X: np.ndarray, Y: np.ndarray, cdist: float) -> int:
    """Calculate the number of contacts between two sets of points X and Y"""
    Ds = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    return np.sum(Ds < cdist)


def process_triplets(
    result_dict: dict, equiv_peptide_chains: list, ligand_chains: list, cdist: float = 3.6, ncontact: float = 0.5
):
    """Process triplets of chains to find those with interface ligands.

    We consider a ligand to be at the interface between two chains if more than 50% of the atoms in each chain are within 3.6 Angstroms of the ligand.

    Args:
        result_dict: The parsed CIF file (output of cifutils parse)
        equiv_peptide_chains: Dictionary mapping chain IDs to equivalent peptide chains
        ligand_chains: A list of ligand chains, where each entry is a list [chain_id, ligand_id, n_atoms]
        cdist: The distance cutoff for a contact
        ncontact: The fraction of atoms in a chain that must be in contact with the ligand

    Returns:
        list: A list of triples [chain1, chain2, ligand_chain, ligand_id, n_contacts_chain1, n_contacts_chain2]
    """
    # (Extract the first assembly from the parsed CIF file)
    atomarray = next(iter(result_dict["assemblies"].values()))[0]
    triplets = []

    # Loop over all pairs of equivalent peptide chains
    for i, js in equiv_peptide_chains.items():
        for j in js:
            if i == j:
                continue
            # Check if a ligand is in contact with both chains
            for k in ligand_chains:
                xyz_i = atomarray[atomarray.chain_id == i].coord
                xyz_j = atomarray[atomarray.chain_id == j].coord
                xyz_k = atomarray[atomarray.chain_id == k[0]].coord
                ncontacts_ik = get_contacts(xyz_i, xyz_k, cdist)  # /xyz_k.shape[0]
                ncontacts_jk = get_contacts(xyz_j, xyz_k, cdist)  # /xyz_k.shape[0]

                if ncontacts_ik > ncontact and ncontacts_jk > ncontact:
                    triplets.append([i, j, k[0], k[1], ncontacts_ik, ncontacts_jk])

    return triplets


def handler(signum, frame):
    raise Exception("Runtime exceeded")


def find_multimers_with_interface_ligands(file_paths: list[PathLike] | PathLike) -> None:
    """Find all homodimers with a ligand at the interface between the two identical chains."""
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    signal.signal(signal.SIGALRM, handler)

    for file_path in file_paths:
        signal.alarm(10)

        try:
            # ... parse the CIF file with cifutils and extract the AtomArray of the first assembly
            results_dict = parse(
                filename=file_path,
                build_assembly="first",
                add_missing_atoms=False,
                hydrogen_policy="remove",
            )

            equiv_peptide_chains, ligand_chains = get_peptides_and_ligands(results_dict)

            if len(equiv_peptide_chains) == 0 or len(ligand_chains) == 0:
                continue

            interface_ligand_triples = process_triplets(results_dict, equiv_peptide_chains, ligand_chains)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        # Print the results
        for ilt_i in interface_ligand_triples:
            print(
                f"{os.path.basename(file_path)},{ilt_i[0]},{ilt_i[1]},{ilt_i[2]},{ilt_i[3]},{ilt_i[4]:.4f},{ilt_i[5]:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symmetric alignment from pointgroups.")
    parser.add_argument("file", help="list of cif files")
    args = parser.parse_args()

    with open(args.file) as file:
        cifs = [line.rstrip("\n") for line in file]

    find_multimers_with_interface_ligands(cifs)
