# Copyright 2020 Jacob D. Durrant
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dimorphite-DL: enumerate ionization states of drug-like small molecules.

Limitations:
    - Only N, O, S, Se and halogen atoms are ionized.
    - pKa predictions are based on substructure SMARTS patterns calibrated
      for drug-like molecules; accuracy may degrade for unusual chemistries.

Identifies and enumerates the possible protonation sites of an RDKit molecule
at a user-specified pH range using pre-calculated pKa distributions.

Originally authored by Jacob D. Durrant (Dimorphite-DL 1.2.4). Vendored into
atomworks as its molecule API (:func:`protonate_mol_variants`), without the
command-line interface or SMILES-file readers.

Reference:
    Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
    open-source program for enumerating the ionization states of drug-like
    small molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.
"""

from __future__ import annotations

import copy
import functools
import os
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

_SOURCE_INDEX = "_atomworks_protonation_source_index"

#: An aromatic nitrogen of formal charge -1.
_AROMATIC_NITRANION = Chem.MolFromSmarts("[n-]")

#: Reactant queries and reactions that neutralise a molecule, applied in order.
_NEUTRALIZING_REACTIONS = tuple(
    (Chem.MolFromSmarts(reactant), AllChem.ReactionFromSmarts(f"{reactant}>>{product}"))
    for reactant, product in (
        ("[Ov1-1:1]", "[Ov2+0:1]-[H]"),
        ("[#7v4+1:1]-[H]", "[#7v3+0:1]"),
        ("[Ov2-:1]", "[Ov2+0:1]"),
        ("[#7v3+1:1]", "[#7v3+0:1]"),
        ("[#7v2-1;!$([#7]=[#7+]):1]", "[#7+0:1]-[H]"),
        ("[H]-[N:1]-[N:2]#[N:3]", "[N:1]=[N+1:2]=[N:3]-[H]"),
    )
)


def _neutralize(mol: Chem.Mol) -> Chem.Mol | None:
    """Neutralise a molecule (e.g. O-, N+, azides) by applying the SMARTS reactions in turn.

    Args:
        mol: The molecule to neutralise; each fragment is neutralised separately.

    Returns:
        The neutralised molecule with explicit hydrogens, or None if it does not sanitise.
    """
    if len(Chem.GetMolFrags(mol)) > 1:
        result = None
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            neutral = _neutralize(fragment)
            if neutral is None:
                return None
            result = neutral if result is None else Chem.CombineMols(result, neutral)
        return result

    mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)
    while True:
        reaction = next((rxn for query, rxn in _NEUTRALIZING_REACTIONS if mol.HasSubstructMatch(query)), None)
        if reaction is None:
            break
        reactant = mol
        # 1QD7: one product per step; building every product of a reaction to keep the first took over 30 min.
        mol = reaction.RunReactants((mol,), maxProducts=1)[0][0]
        # Reaction SMARTS use their own map labels. Restore caller
        # identity using RDKit's reactant-index provenance instead.
        for atom in mol.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                source = reactant.GetAtomWithIdx(atom.GetIntProp("react_atom_idx"))
                atom.SetAtomMapNum(source.GetAtomMapNum())
                for name, value in source.GetPropsAsDict(includePrivate=True, includeComputed=False).items():
                    if isinstance(value, bool):
                        atom.SetBoolProp(name, value)
                    elif isinstance(value, int):
                        atom.SetIntProp(name, value)
                    elif isinstance(value, float):
                        atom.SetDoubleProp(name, value)
                    elif isinstance(value, str):
                        atom.SetProp(name, value)
        mol.UpdatePropertyCache(strict=False)

    sanitized = Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
    return mol if sanitized == Chem.SanitizeFlags.SANITIZE_NONE else None


def _changes_only_its_valence(mols: list[Chem.Mol], index: int, ring_atoms: set[int]) -> bool:
    """Whether a new charge and hydrogen count on atom *index* can change no atom's valence but its own.

    Outside a ring and without an aromatic bond, the atom takes no part in aromaticity or a Kekule form,
    so sanitizing the molecule again after the change could fail, or change a state, only there.
    """
    if index in ring_atoms:
        return False
    return not any(bond.GetIsAromatic() for mol in mols for bond in mol.GetAtomWithIdx(index).GetBonds())


def _valence_holds(mol: Chem.Mol, index: int) -> bool:
    """Whether RDKit accepts atom *index*'s valence, as sanitizing checks every atom's."""
    try:
        mol.GetAtomWithIdx(index).UpdatePropertyCache(strict=True)
    except Chem.MolSanitizeException:
        return False
    return True


def _parses(smiles: str) -> bool:
    """Whether RDKit reads *smiles* back, after spelling legacy azides charge-separated."""
    smiles = smiles.replace("N=N=N", "N=[N+]=N").replace("NN#N", "N=[N+]=N")
    return Chem.MolFromSmiles(smiles) is not None


class ProtSubstructFuncs:
    """Namespace for protonation-substructure matching and site modification."""

    @staticmethod
    @functools.cache
    def _compiled_substructures() -> tuple[tuple[str, str, Chem.Mol, tuple[tuple[str, float, float], ...]], ...]:
        """The rules of ``site_substructures.smarts`` in priority order, compiled once for every pH.

        The queries are only matched, never modified; no input molecule or pH is retained.
        """
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "site_substructures.smarts")) as f:
            lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
        rules = []
        for fields in lines:
            name, smart = fields[:2]
            sites = tuple((fields[i], float(fields[i + 1]), float(fields[i + 2])) for i in range(2, len(fields) - 1, 3))
            rules.append((name, smart, Chem.MolFromSmarts(smart), sites))
        return tuple(rules)

    @staticmethod
    def _substructures_for_ph(min_ph: float, max_ph: float, pka_std_range: float) -> list[dict[str, Any]]:
        """Fresh state containers borrowing the compiled, read-only query molecules."""
        return [
            {
                "name": name,
                "smart": smart,
                "mol": query,
                "prot_states_for_pH": [
                    [site, ProtSubstructFuncs.define_protonation_state(mean, std * pka_std_range, min_ph, max_ph)]
                    for site, mean, std in sites
                ],
            }
            for name, smart, query, sites in ProtSubstructFuncs._compiled_substructures()
        ]

    @staticmethod
    def define_protonation_state(mean: float, std: float, min_ph: float, max_ph: float) -> str:
        """Determine the protonation state for a site at a given pH range.

        Args:
            mean: The mean pKa value.
            std: The standard deviation (precision).
            min_ph: Minimum pH of the range.
            max_ph: Maximum pH of the range.

        Returns:
            One of ``"PROTONATED"``, ``"DEPROTONATED"``, or ``"BOTH"``.
        """
        if mean - std <= max_ph and min_ph <= mean + std:
            return "BOTH"
        return "PROTONATED" if mean > max_ph else "DEPROTONATED"

    @staticmethod
    def get_prot_sites_and_target_states_from_mol(
        mol: Chem.Mol, subs: list[dict[str, Any]]
    ) -> tuple[list[tuple[int, str, str]], Chem.Mol | None]:
        """Find protonation sites and their target states for a molecule.

        Sites higher in *subs* take priority: once a rule matches, the atoms of
        the match cannot be ionized by a later rule, though they may still form
        its context.

        Args:
            mol: Input molecule.
            subs: Substructure definitions in priority order, as :meth:`_substructures_for_ph` gives them.

        Returns:
            A tuple of (sites, mol) where sites is a list of
            ``(atom_index, target_state, site_name)`` tuples and mol is
            the hydrogenated molecule used for indexing, or ``([], None)``
            if hydrogens cannot be added.
        """
        try:
            mol_used_to_idx_sites = Chem.AddHs(mol)
        except Exception:
            return [], None

        protonation_sites: list[tuple[int, str, str]] = []
        protected: set[int] = set()
        for item in subs:
            prot = item["prot_states_for_pH"]
            site_indices = [int(site[0]) for site in prot]
            matches = [
                match
                for match in mol_used_to_idx_sites.GetSubstructMatches(item["mol"])
                if protected.isdisjoint(match[i] for i in site_indices if i < len(match))
            ]
            for match in matches:
                for site in prot:
                    new_site = (match[int(site[0])], site[1], item["name"])
                    if new_site not in protonation_sites:
                        protonation_sites.append(new_site)
                protected.update(match)

        return protonation_sites, mol_used_to_idx_sites

    @staticmethod
    def protonate_site(mols: list[Chem.Mol], site: tuple[int, str, str], *, remove_hs: bool = True) -> list[Chem.Mol]:
        """Protonate or deprotonate a single site across a list of molecules.

        Nitrogen, oxygen, sulfur and selenium take the hydrogen count their charge
        and bond order give; a site whose name carries ``*`` titrates a
        nitrogen between charges -1 and 0 rather than 0 and +1.

        Args:
            mols: Input molecule objects.
            site: A ``(atom_index, target_state, site_name)`` tuple.
            remove_hs: Remove (and so sanitize) each molecule's hydrogens first; without it
                each molecule, already hydrogen-free, is copied as it is.

        Returns:
            One copy of each molecule per charge the target state allows, with
            hydrogens removed; a molecule whose hydrogens cannot be removed is skipped.
        """
        idx, target_prot_state, prot_site_name = site
        charges = {"DEPROTONATED": [-1], "PROTONATED": [0], "BOTH": [-1, 0]}[target_prot_state]
        output: list[Chem.Mol] = []
        for charge in charges:
            nitrogen_charge = charge + int("*" not in prot_site_name)
            for mol in mols:
                try:
                    mol_copy = Chem.RemoveHs(mol) if remove_hs else Chem.Mol(mol)
                except Exception:
                    continue
                atom = mol_copy.GetAtomWithIdx(idx)
                bond_order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                element = atom.GetAtomicNum()
                if element == 7:
                    atom.SetFormalCharge(nitrogen_charge)
                    if bond_order in (0, 1, 2) or (nitrogen_charge == 1 and bond_order == 3):
                        atom.SetNumExplicitHs(int(3 + nitrogen_charge - bond_order))
                else:
                    atom.SetFormalCharge(charge)
                    if element in (8, 16, 34) and bond_order == 1:
                        atom.SetNumExplicitHs(1 + charge)

                # SMILES spells [nH-] only for an aromatic nitrogen of charge -1, so
                # the SMILES is written only where one exists.
                if mol_copy.HasSubstructMatch(_AROMATIC_NITRANION) and "[nH-]" in Chem.MolToSmiles(mol_copy):
                    atom.SetNumExplicitHs(0)

                mol_copy.UpdatePropertyCache(strict=False)
                output.append(mol_copy)
        return output


def protonate_mol_variants(
    mol: Chem.Mol,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
    pka_precision: float = 1.0,
    max_variants: int = 128,
) -> list[Chem.Mol]:
    """Enumerate protonation states while preserving atom identity and order.

    Args:
        mol: Input molecule, copied before modification.
        min_ph: Lower pH bound.
        max_ph: Upper pH bound.
        pka_precision: Number of pKa standard deviations in the range.
        max_variants: Maximum intermediate variants retained after each site.

    Returns:
        Valid molecular states in first-occurrence order, retaining atom maps,
        atom properties and the input's conformers; empty if the molecule
        cannot be neutralised.
    """
    prepared = copy.deepcopy(mol)
    for atom in prepared.GetAtoms():
        atom.SetIntProp(_SOURCE_INDEX, atom.GetIdx())
    prepared = _neutralize(prepared)
    if prepared is None:
        return []
    try:
        prepared = Chem.RemoveHs(prepared)
    except Exception:
        return []

    subs = ProtSubstructFuncs._substructures_for_ph(min_ph, max_ph, pka_precision)
    sites, mol_used_to_idx_sites = ProtSubstructFuncs.get_prot_sites_and_target_states_from_mol(prepared, subs)
    if mol_used_to_idx_sites is None:
        return []

    new_mols = [mol_used_to_idx_sites]
    properly_formed_mols = [prepared]
    if sites:
        ring_atoms = {index for ring in Chem.GetSymmSSSR(Chem.Mol(prepared)) for index in ring}
        for site in sites:
            # 1QD7: once a site has given products the molecules are hydrogen-free and sanitize, so a change
            # that can break only its own atom's valence is checked there alone.
            local = new_mols[0] is not mol_used_to_idx_sites and _changes_only_its_valence(
                new_mols, site[0], ring_atoms
            )
            candidates = (
                ProtSubstructFuncs.protonate_site(new_mols, site, remove_hs=False)
                if local
                else ProtSubstructFuncs.protonate_site(new_mols, site)
            )
            products = [
                product
                for product in candidates
                if (
                    _valence_holds(product, site[0])
                    if local
                    else Chem.SanitizeMol(Chem.Mol(product), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
                )
            ]
            if not products:
                break
            new_mols = products[:max_variants]
            properly_formed_mols.extend(new_mols)
        if len(properly_formed_mols) == 1:
            new_mols = properly_formed_mols
    else:
        mol_used_to_idx_sites = Chem.RemoveHs(mol_used_to_idx_sites)
        new_mols = [mol_used_to_idx_sites]
        properly_formed_mols.append(mol_used_to_idx_sites)

    seen: set[str] = set()
    output_mols: list[Chem.Mol] = []
    for product in new_mols:
        smi = Chem.MolToSmiles(product, isomericSmiles=True, canonical=True)
        if smi in seen or not _parses(smi):
            continue
        seen.add(smi)
        output_mols.append(product)
    if not output_mols:
        output_mols = next(
            ([m] for m in reversed(properly_formed_mols) if _parses(Chem.MolToSmiles(m, isomericSmiles=True))), []
        )

    ordered_products = []
    for product in output_mols:
        order = sorted(
            range(product.GetNumAtoms()),
            key=lambda i: (
                product.GetAtomWithIdx(i).GetIntProp(_SOURCE_INDEX)
                if product.GetAtomWithIdx(i).HasProp(_SOURCE_INDEX)
                else mol.GetNumAtoms() + i
            ),
        )
        product = Chem.RenumberAtoms(product, order)
        Chem.SanitizeMol(product)
        product.RemoveAllConformers()
        for source_conf in mol.GetConformers():
            conf = Chem.Conformer(product.GetNumAtoms())
            conf.Set3D(source_conf.Is3D())
            for atom in product.GetAtoms():
                if atom.HasProp(_SOURCE_INDEX):
                    conf.SetAtomPosition(atom.GetIdx(), source_conf.GetAtomPosition(atom.GetIntProp(_SOURCE_INDEX)))
            product.AddConformer(conf, assignId=True)
        for atom in product.GetAtoms():
            if atom.HasProp(_SOURCE_INDEX):
                atom.ClearProp(_SOURCE_INDEX)
        ordered_products.append(product)
    return ordered_products
