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
    - Only protonates N, O, and S atoms. Other elements (e.g., P, metals)
      are not considered for ionization.
    - Relies on SMILES string representation and repeated mol<->SMILES
      interconversion, which may lose stereochemistry or produce unexpected
      results for exotic non-drug-like molecules.
    - pKa predictions are based on substructure SMARTS patterns calibrated
      for drug-like molecules; accuracy may degrade for unusual chemistries.

Identifies and enumerates the possible protonation sites of SMILES strings
at a user-specified pH range using pre-calculated pKa distributions.

Originally authored by Jacob D. Durrant (Dimorphite-DL 1.2.4). Vendored
into atomworks and cleaned up for lint compliance, type annotations, and
Google-style docstrings. Rule providers select the scientific rule inventory.

Reference:
    Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
    open-source program for enumerating the ionization states of drug-like
    small molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.
"""

from __future__ import annotations

import argparse
import copy
import functools
import logging
import os
import sys
from io import StringIO
from typing import IO, Any, ClassVar

logger = logging.getLogger(__name__)


def _log_info(*args: Any) -> None:
    """Log message at INFO level using print-like argument joining."""
    logger.info(" ".join(str(a) for a in args))


def _log_error(*args: Any) -> None:
    """Log message at ERROR level using print-like argument joining."""
    logger.error(" ".join(str(a) for a in args))


try:
    import rdkit
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
except ImportError as err:
    msg = "Dimorphite-DL requires RDKit. See https://www.rdkit.org/"
    _log_error(msg)
    raise ImportError(msg) from err


def print_header() -> None:
    """Log citation and help information."""
    _log_info("\nFor help, use: python dimorphite_dl.py --help")
    _log_info("\nIf you use Dimorphite-DL in your research, please cite:")
    _log_info("Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An")
    _log_info("open-source program for enumerating the ionization states of drug-like small")
    _log_info("molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.\n")


def main(params: dict[str, Any] | None = None) -> list[str] | None:
    """Entry point when the script is called from the command line.

    Args:
        params: Optional parameter dictionary. If absent, arguments are
            parsed from the command line.

    Returns:
        A list of protonated SMILES strings when the ``return_as_list``
        parameter is True, otherwise None.
    """
    parser = ArgParseFuncs.get_args()
    args = vars(parser.parse_args())

    if not args["silent"]:
        print_header()

    if params is not None:
        for k, v in params.items():
            args[k] = v

    if __name__ == "__main__":
        if not args["silent"]:
            _log_info("\nPARAMETERS:\n")
            for k in sorted(args.keys()):
                _log_info(k.rjust(13) + ": " + str(args[k]))
            _log_info("")

    if args["test"]:
        TestFuncs.test()
    else:
        if "output_file" in args and args["output_file"] is not None:
            with open(args["output_file"], "w") as file:
                for protonated_smi in Protonate(args):
                    file.write(protonated_smi + "\n")
        elif "return_as_list" in args and args["return_as_list"] is True:
            return list(Protonate(args))
        else:
            for protonated_smi in Protonate(args):
                _log_info(protonated_smi)

    return None


class MyParser(argparse.ArgumentParser):
    """ArgumentParser subclass that prints help on error."""

    def error(self, message: str) -> None:
        """Print help and raise on parse error.

        Args:
            message: The error message from argparse.

        Raises:
            Exception: Always raised with the error message.
        """
        self.print_help()
        msg = f"ERROR: {message}\n\n"
        _log_error(msg)
        raise Exception(msg)

    def print_help(self, file: IO[str] | None = None) -> None:
        """Print help text with usage examples.

        Args:
            file: Output stream. Defaults to stdout.
        """
        _log_info("")

        if file is None:
            file = sys.stdout
        self._print_message(self.format_help(), file)
        _log_info(
            """
examples:
  python dimorphite_dl.py --smiles_file sample_molecules.smi
  python dimorphite_dl.py --smiles "CCC(=O)O" --min_ph -3.0 --max_ph -2.0
  python dimorphite_dl.py --smiles "CCCN" --min_ph -3.0 --max_ph -2.0 --output_file output.smi
  python dimorphite_dl.py --smiles_file sample_molecules.smi --pka_precision 2.0 --label_states
  python dimorphite_dl.py --test"""
        )
        _log_info("")


class ArgParseFuncs:
    """Namespace for command-line argument processing functions."""

    @staticmethod
    def get_args() -> MyParser:
        """Build and return the argument parser.

        Returns:
            A configured argument parser instance.
        """
        parser = MyParser(
            description="Dimorphite 1.2.4: Creates models of "
            + "appropriately protonated small moleucles. "
            + "Apache 2.0 License. Copyright 2020 Jacob D. "
            + "Durrant."
        )
        parser.add_argument(
            "--min_ph",
            metavar="MIN",
            type=float,
            default=6.4,
            help="minimum pH to consider (default: 6.4)",
        )
        parser.add_argument(
            "--max_ph",
            metavar="MAX",
            type=float,
            default=8.4,
            help="maximum pH to consider (default: 8.4)",
        )
        parser.add_argument(
            "--pka_precision",
            metavar="PRE",
            type=float,
            default=1.0,
            help="pKa precision factor (number of standard devations, default: 1.0)",
        )
        parser.add_argument("--smiles", metavar="SMI", type=str, help="SMILES string to protonate")
        parser.add_argument(
            "--smiles_file",
            metavar="FILE",
            type=str,
            help="file that contains SMILES strings to protonate",
        )
        parser.add_argument(
            "--output_file",
            metavar="FILE",
            type=str,
            help="output file to write protonated SMILES (optional)",
        )
        parser.add_argument(
            "--max_variants",
            metavar="MXV",
            type=int,
            default=128,
            help="limit number of variants per input compound (default: 128)",
        )
        parser.add_argument(
            "--label_states",
            action="store_true",
            help="label protonated SMILES with target state " + '(i.e., "DEPROTONATED", "PROTONATED", or "BOTH").',
        )
        parser.add_argument(
            "--silent",
            action="store_true",
            help="do not print any messages to the screen",
        )
        parser.add_argument("--test", action="store_true", help="run unit tests (for debugging)")

        return parser

    @staticmethod
    def clean_args(args: dict[str, Any]) -> dict[str, Any]:
        """Clean and normalise input parameters.

        Fills in defaults for missing keys, removes ``None`` values, and
        converts a bare ``smiles`` string into a file-like object.

        Args:
            args: Mutable dictionary of arguments.

        Returns:
            The cleaned argument dictionary (same object, mutated).

        Raises:
            Exception: If neither ``smiles`` nor ``smiles_file`` is provided.
        """
        defaults: dict[str, Any] = {
            "min_ph": 6.4,
            "max_ph": 8.4,
            "pka_precision": 1.0,
            "label_states": False,
            "test": False,
            "max_variants": 128,
        }

        for key in defaults:
            if key not in args:
                args[key] = defaults[key]

        keys = list(args.keys())
        for key in keys:
            if args[key] is None:
                del args[key]

        if "smiles" not in args and "smiles_file" not in args:
            msg = "Error: No SMILES in params. Use the -h parameter for help."
            _log_error(msg)
            raise Exception(msg)

        if "smiles" in args and isinstance(args["smiles"], str):
            args["smiles_file"] = StringIO(args["smiles"])

        args["smiles_and_data"] = LoadSMIFile(args["smiles_file"], args)

        return args


class UtilFuncs:
    """Namespace for molecular utility functions."""

    @staticmethod
    def neutralize_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol | None:
        """Neutralise a molecule by iteratively applying SMARTS reactions.

        Removes inappropriate charges (e.g. O-, N+) and fixes azide
        representations. The user should not be allowed to specify atom
        valences in most cases.

        Args:
            mol: The RDKit Mol object to neutralise.

        Returns:
            The neutralised Mol object, or None if sanitisation fails.
        """
        if len(Chem.GetMolFrags(mol)) > 1:
            result = None
            for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
                neutral = UtilFuncs.neutralize_mol(fragment)
                if neutral is None:
                    return None
                result = neutral if result is None else Chem.CombineMols(result, neutral)
            return result

        rxn_data: list[list[Any]] = [
            ["[Ov1-1:1]", "[Ov2+0:1]-[H]", None, None],
            ["[#7v4+1:1]-[H]", "[#7v3+0:1]", None, None],
            ["[Ov2-:1]", "[Ov2+0:1]", None, None],
            ["[#7v3+1:1]", "[#7v3+0:1]", None, None],
            ["[#7v2-1;!$([#7]=[#7+]):1]", "[#7+0:1]-[H]", None, None],
            ["[H]-[N:1]-[N:2]#[N:3]", "[N:1]=[N+1:2]=[N:3]-[H]", None, None],
        ]

        for i, rxn_datum in enumerate(rxn_data):
            rxn_data[i][2] = Chem.MolFromSmarts(rxn_datum[0])

        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)

        while True:
            current_rxn = None

            for i, rxn_datum in enumerate(rxn_data):
                (
                    reactant_smarts,
                    product_smarts,
                    substruct_match_mol,
                    rxn_placeholder,
                ) = rxn_datum
                if mol.HasSubstructMatch(substruct_match_mol):
                    if rxn_placeholder is None:
                        current_rxn_str = reactant_smarts + ">>" + product_smarts
                        current_rxn = AllChem.ReactionFromSmarts(current_rxn_str)
                        rxn_data[i][3] = current_rxn
                    else:
                        current_rxn = rxn_data[i][3]
                    break

            if current_rxn is None:
                break
            else:
                reactant = mol
                mol = current_rxn.RunReactants((mol,))[0][0]
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

        sanitize_string = Chem.SanitizeMol(
            mol,
            sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL,
            catchErrors=True,
        )

        return mol if sanitize_string.name == "SANITIZE_NONE" else None

    @staticmethod
    def convert_smiles_str_to_mol(smiles_str: str | None) -> Chem.rdchem.Mol | None:
        """Convert a SMILES string to an RDKit Mol object.

        Performs type checking, fixes common azide issues, and suppresses
        RDKit stderr output during conversion.

        Args:
            smiles_str: The SMILES string to convert.

        Returns:
            An RDKit Mol object, or None on failure.
        """
        if smiles_str is None or type(smiles_str) is not str:
            return None

        smiles_str = smiles_str.replace("N=N=N", "N=[N+]=N")
        smiles_str = smiles_str.replace("NN#N", "N=[N+]=N")

        mol = Chem.MolFromSmiles(smiles_str)
        return None if mol is None else mol

    @staticmethod
    def eprint(*args: Any, **kwargs: Any) -> None:
        """Log to stderr-equivalent channel.

        Args:
            *args: Positional arguments forwarded to logger.
            **kwargs: Unused keyword arguments for compatibility.
        """
        del kwargs
        _log_error(*args)


class LoadSMIFile:
    """Generator that loads and pre-processes SMILES strings from a file."""

    def __init__(self, filename: str | IO[str], args: dict[str, Any]) -> None:
        """Initialise the SMILES file loader.

        Args:
            filename: Path string or file-like object (e.g. StringIO).
            args: The global argument dictionary.
        """
        self.args = args

        if type(filename) is str:
            self.f: IO[str] = open(filename)  # noqa: SIM115
        else:
            self.f = filename

    def __iter__(self) -> LoadSMIFile:
        """Return this generator object."""
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next SMILES record (Python 3 iterator protocol)."""
        return self.next()

    def next(self) -> dict[str, Any]:
        """Read and process the next line from the SMILES file.

        Converts the raw SMILES to a canonical, neutralised form with
        hydrogens removed.

        Returns:
            A dict with ``"smiles"`` (canonical SMILES) and ``"data"``
            (remaining tab-separated fields).

        Raises:
            StopIteration: When the file is exhausted.
        """
        line = self.f.readline()

        if line == "":
            self.f.close()
            raise StopIteration()

        splits = line.split()
        if len(splits) != 0:
            smiles_str = splits[0]

            mol = UtilFuncs.convert_smiles_str_to_mol(smiles_str)
            if mol is None:
                if "silent" in self.args and not self.args["silent"]:
                    UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + line)
                return self.next()

            mol = UtilFuncs.neutralize_mol(mol)
            if mol is None:
                if "silent" in self.args and not self.args["silent"]:
                    UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + line)
                return self.next()

            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                if "silent" in self.args and not self.args["silent"]:
                    UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + line)
                return self.next()

            if mol is None:
                if "silent" in self.args and not self.args["silent"]:
                    UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + line)
                return self.next()

            new_mol_string = Chem.MolToSmiles(mol, isomericSmiles=True)

            return {"smiles": new_mol_string, "data": splits[1:]}
        else:
            return self.next()


class Protonate:
    """Generator that yields protonated SMILES strings one at a time."""

    def __init__(self, args: dict[str, Any]) -> None:
        """Initialise the protonation generator.

        Args:
            args: The argument dictionary (will be cleaned/normalised).
        """
        self.args = args
        self.cur_prot_SMI: list[str] = []

        self.args = ArgParseFuncs.clean_args(args)

        ProtSubstructFuncs.args = args

        rule_provider = self.args.get("rule_provider", ProtSubstructFuncs)
        self.subs = rule_provider.load_protonation_substructs_calc_state_for_ph(
            self.args["min_ph"], self.args["max_ph"], self.args["pka_precision"]
        )

    def __iter__(self) -> Protonate:
        """Return this generator object."""
        return self

    def __next__(self) -> str:
        """Return the next protonated SMILES (Python 3 iterator protocol)."""
        return self.next()

    def next(self) -> str:
        """Return the next protonated SMILES string.

        Handles multi-site protonation by expanding combinations and
        caching results in ``self.cur_prot_SMI``.

        Returns:
            A protonated SMILES string with optional label and tag.

        Raises:
            StopIteration: When all input SMILES have been processed.
        """
        if len(self.cur_prot_SMI) > 0:
            first, self.cur_prot_SMI = self.cur_prot_SMI[0], self.cur_prot_SMI[1:]
            return first

        try:
            smile_and_datum = self.args["smiles_and_data"].next()
        except StopIteration:
            raise StopIteration() from None

        orig_smi = smile_and_datum["smiles"]
        properly_formed_smi_found = [orig_smi]
        data = smile_and_datum["data"]
        tag = " ".join(data)

        (
            sites,
            mol_used_to_idx_sites,
        ) = ProtSubstructFuncs.get_prot_sites_and_target_states(orig_smi, self.subs)
        if mol_used_to_idx_sites is None:
            return self.next()

        new_mols = [mol_used_to_idx_sites]
        if len(sites) > 0:
            for site in sites:
                new_mols = ProtSubstructFuncs.protonate_site(new_mols, site)
                if len(new_mols) > self.args["max_variants"]:
                    new_mols = new_mols[: self.args["max_variants"]]
                    if "silent" in self.args and not self.args["silent"]:
                        UtilFuncs.eprint(
                            "WARNING: Limited number of variants to " + str(self.args["max_variants"]) + ": " + orig_smi
                        )

                properly_formed_smi_found += [Chem.MolToSmiles(m) for m in new_mols]
        else:
            mol_used_to_idx_sites = Chem.RemoveHs(mol_used_to_idx_sites)
            new_mols = [mol_used_to_idx_sites]
            properly_formed_smi_found.append(Chem.MolToSmiles(mol_used_to_idx_sites))

        new_smis = list({Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) for m in new_mols})

        new_smis = [s for s in new_smis if UtilFuncs.convert_smiles_str_to_mol(s) is not None]

        if len(new_smis) == 0:
            properly_formed_smi_found.reverse()
            for smi in properly_formed_smi_found:
                if UtilFuncs.convert_smiles_str_to_mol(smi) is not None:
                    new_smis = [smi]
                    break

        if self.args["label_states"]:
            states = "\t".join([x[1] for x in sites])
            new_lines = [x + "\t" + tag + "\t" + states for x in new_smis]
        else:
            new_lines = [x + "\t" + tag for x in new_smis]

        self.cur_prot_SMI = new_lines

        return self.next()


class ProtSubstructFuncs:
    """Namespace for protonation-substructure matching and site modification."""

    args: ClassVar[dict[str, Any]] = {}

    @staticmethod
    def load_substructre_smarts_file() -> list[str]:
        """Load the substructure SMARTS file, filtering out comments.

        Returns:
            Non-blank, non-comment lines from ``site_substructures.smarts``.
        """
        pwd = os.path.dirname(os.path.realpath(__file__))
        site_structures_file = f"{pwd}/site_substructures.smarts"
        with open(site_structures_file) as f:
            lines = [line for line in f if line.strip() != "" and not line.startswith("#")]
        return lines

    @classmethod
    def load_protonation_substructs_calc_state_for_ph(
        cls, min_ph: float = 6.4, max_ph: float = 8.4, pka_std_range: float = 1
    ) -> list[dict[str, Any]]:
        """Load protonation substructures and calculate states for a pH range.

        Reads the SMARTS definitions file and, for each protonation site,
        determines whether it should be protonated, deprotonated, or both
        at the given pH range.

        Args:
            min_ph: Lower bound of the pH range.
            max_ph: Upper bound of the pH range.
            pka_std_range: Number of standard deviations from the mean pKa
                to consider.

        Returns:
            A list of substructure dicts, each containing ``"name"``,
            ``"smart"``, ``"mol"``, and ``"prot_states_for_pH"``.
        """
        # Public callers own both the nested state containers and RDKit queries.
        subs = cls._substructures_for_ph(min_ph, max_ph, pka_std_range)
        for sub in subs:
            sub["mol"] = Chem.Mol(sub["mol"])
        return subs

    @classmethod
    @functools.lru_cache(maxsize=8)
    def _compiled_substructures(cls) -> tuple[tuple[str, str, Chem.Mol, tuple[tuple[str, float, float], ...]], ...]:
        """Compile the fixed rule file once, independent of requested pH.

        Queries stay private and are only used for read-only substructure
        matching. Never retain input molecules or a growing set of pH values.
        """
        rules = []
        for line in cls.load_substructre_smarts_file():
            fields = line.split()
            if not fields:
                continue
            name, smart = fields[:2]
            sites = tuple((fields[i], float(fields[i + 1]), float(fields[i + 2])) for i in range(2, len(fields) - 1, 3))
            rules.append((name, smart, Chem.MolFromSmarts(smart), sites))
        return tuple(rules)

    @classmethod
    def _substructures_for_ph(cls, min_ph: float, max_ph: float, pka_std_range: float) -> list[dict[str, Any]]:
        """Fresh state containers borrowing private, read-only query molecules."""
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
            for name, smart, query, sites in cls._compiled_substructures()
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
        min_pka = mean - std
        max_pka = mean + std

        if min_pka <= max_ph and min_ph <= max_pka:
            protonation_state = "BOTH"
        elif mean > max_ph:
            protonation_state = "PROTONATED"
        else:
            protonation_state = "DEPROTONATED"

        return protonation_state

    @staticmethod
    def get_prot_sites_and_target_states(
        smi: str, subs: list[dict[str, Any]]
    ) -> tuple[list[tuple[int, str, str]], Chem.rdchem.Mol | None]:
        """Find protonation sites and their target states for a molecule.

        Matches the molecule against the substructure list. Sites higher
        in the list take priority and protect matched atoms from later
        matches.

        Args:
            smi: A SMILES string.
            subs: Substructure definitions from
                :func:`load_protonation_substructs_calc_state_for_ph`.

        Returns:
            A tuple of (sites, mol) where sites is a list of
            ``(atom_index, target_state, site_name)`` tuples and mol is
            the hydrogenated Mol object used for indexing.
        """
        mol_used_to_idx_sites = UtilFuncs.convert_smiles_str_to_mol(smi)
        if mol_used_to_idx_sites is None:
            UtilFuncs.eprint("ERROR:   ", smi)
            return [], None

        return ProtSubstructFuncs.get_prot_sites_and_target_states_from_mol(mol_used_to_idx_sites, subs)

    @staticmethod
    def get_prot_sites_and_target_states_from_mol(
        mol: Chem.rdchem.Mol, subs: list[dict[str, Any]]
    ) -> tuple[list[tuple[int, str, str]], Chem.rdchem.Mol | None]:
        """Find protonation sites and target states for an RDKit Mol.

        Args:
            mol: Input molecule.
            subs: Substructure definitions.

        Returns:
            A tuple of (sites, hydrogenated_mol). If processing fails,
            returns ([], None).
        """
        mol_used_to_idx_sites = copy.deepcopy(mol)
        try:
            mol_used_to_idx_sites = Chem.AddHs(mol_used_to_idx_sites)
        except Exception:
            return [], None

        if mol_used_to_idx_sites is None:
            return [], None

        ProtectUnprotectFuncs.unprotect_molecule(mol_used_to_idx_sites)
        protonation_sites: list[tuple[int, str, str]] = []

        for item in subs:
            smart = item["mol"]
            if mol_used_to_idx_sites.HasSubstructMatch(smart):
                prot = item["prot_states_for_pH"]
                # Only this rule's own protonation sites can be blocked by a
                # higher-priority rule; shared context atoms must not block.
                site_indices = [int(site[0]) for site in prot]
                matches = ProtectUnprotectFuncs.get_unprotected_matches(mol_used_to_idx_sites, smart, site_indices)
                for match in matches:
                    for site in prot:
                        proton = int(site[0])
                        category = site[1]
                        new_site = (match[proton], category, item["name"])

                        if new_site not in protonation_sites:
                            protonation_sites.append(new_site)

                    ProtectUnprotectFuncs.protect_molecule(mol_used_to_idx_sites, match)

        return protonation_sites, mol_used_to_idx_sites

    @staticmethod
    def protonate_site(mols: list[Chem.rdchem.Mol], site: tuple[int, str, str]) -> list[Chem.rdchem.Mol]:
        """Protonate or deprotonate a single site across a list of molecules.

        Args:
            mols: Input molecule objects.
            site: A ``(atom_index, target_state, site_name)`` tuple.

        Returns:
            A list of molecule objects with the site adjusted.
        """
        idx, target_prot_state, prot_site_name = site

        state_to_charge = {"DEPROTONATED": [-1], "PROTONATED": [0], "BOTH": [-1, 0]}

        charges = state_to_charge[target_prot_state]

        output_mols = ProtSubstructFuncs.set_protonation_charge(mols, idx, charges, prot_site_name)

        return output_mols

    @staticmethod
    def set_protonation_charge(
        mols: list[Chem.rdchem.Mol],
        idx: int,
        charges: list[int],
        prot_site_name: str,
    ) -> list[Chem.rdchem.Mol]:
        """Set the formal charge at a protonation site for each molecule.

        Handles nitrogen, oxygen, and sulfur atoms with appropriate
        hydrogen counts based on the charge and bond order.

        Args:
            mols: Input molecule objects.
            idx: Atom index of the protonation site.
            charges: List of charges to assign (one mol copy per charge).
            prot_site_name: Name of the protonation site definition.

        Returns:
            A list of molecule objects with charges assigned.
        """
        output: list[Chem.rdchem.Mol] = []

        for charge in charges:
            nitrogen_charge = charge + 1

            if "*" in prot_site_name:
                nitrogen_charge = nitrogen_charge - 1

            for mol in mols:
                mol_copy = copy.deepcopy(mol)

                try:
                    mol_copy = Chem.RemoveHs(mol_copy)
                except Exception:
                    if "silent" in ProtSubstructFuncs.args and not ProtSubstructFuncs.args["silent"]:
                        UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + Chem.MolToSmiles(mol_copy))
                    continue

                atom = mol_copy.GetAtomWithIdx(idx)

                explicit_bond_order_total = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])

                element = atom.GetAtomicNum()
                if element == 7:
                    atom.SetFormalCharge(nitrogen_charge)

                    if nitrogen_charge == 1 and explicit_bond_order_total == 1:
                        atom.SetNumExplicitHs(3)
                    elif nitrogen_charge == 1 and explicit_bond_order_total == 2:
                        atom.SetNumExplicitHs(2)
                    elif nitrogen_charge == 1 and explicit_bond_order_total == 3:
                        atom.SetNumExplicitHs(1)
                    elif nitrogen_charge == 0 and explicit_bond_order_total == 1:
                        atom.SetNumExplicitHs(2)
                    elif nitrogen_charge == 0 and explicit_bond_order_total == 2:
                        atom.SetNumExplicitHs(1)
                    elif nitrogen_charge == -1 and explicit_bond_order_total == 2:
                        atom.SetNumExplicitHs(0)
                    elif nitrogen_charge == -1 and explicit_bond_order_total == 1:
                        atom.SetNumExplicitHs(1)
                else:
                    atom.SetFormalCharge(charge)
                    if element == 8 or element == 16:
                        if charge == 0 and explicit_bond_order_total == 1:
                            atom.SetNumExplicitHs(1)
                        elif charge == -1 and explicit_bond_order_total == 1:
                            atom.SetNumExplicitHs(0)

                if "[nH-]" in Chem.MolToSmiles(mol_copy):
                    atom.SetNumExplicitHs(0)

                mol_copy.UpdatePropertyCache(strict=False)

                output.append(mol_copy)

        return output


class ProtectUnprotectFuncs:
    """Namespace for atom protection/unprotection during substructure matching."""

    @staticmethod
    def unprotect_molecule(mol: Chem.rdchem.Mol) -> None:
        """Mark all atoms in the molecule as unprotected.

        Args:
            mol: The RDKit Mol object whose atoms to unprotect.
        """
        for atom in mol.GetAtoms():
            atom.SetProp("_protected", "0")

    @staticmethod
    def protect_molecule(mol: Chem.rdchem.Mol, match: tuple[int, ...]) -> None:
        """Mark matched atoms as protected to prevent re-matching.

        Args:
            mol: The RDKit Mol object.
            match: Tuple of atom indices to protect.
        """
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            atom.SetProp("_protected", "1")

    @staticmethod
    def get_unprotected_matches(
        mol: Chem.rdchem.Mol,
        substruct: Chem.rdchem.Mol,
        site_indices: list[int] | None = None,
    ) -> list[tuple[int, ...]]:
        """Find matches this rule is still allowed to ionize.

        Args:
            mol: The molecule to search.
            substruct: The SMARTS substructure pattern.
            site_indices: Positions within the match that this rule would
                actually ionize. When given, only those atoms are checked for
                protection (see :func:`is_match_unprotected`).

        Returns:
            A list of matches (each a tuple of atom indices).
        """
        matches = mol.GetSubstructMatches(substruct)
        unprotected_matches = []
        for match in matches:
            atom_indices = match if site_indices is None else tuple(match[i] for i in site_indices if i < len(match))
            if ProtectUnprotectFuncs.is_match_unprotected(mol, atom_indices):
                unprotected_matches.append(match)
        return unprotected_matches

    @staticmethod
    def is_match_unprotected(
        mol: Chem.rdchem.Mol,
        atom_indices: tuple[int, ...],
    ) -> bool:
        """Check whether the atoms this rule would ionize are unprotected.

        Args:
            mol: The RDKit Mol object.
            atom_indices: Atom indices to check.

        Returns:
            True if none of the checked atoms are protected.
        """
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            protected = atom.GetProp("_protected")
            if protected == "1":
                return False
        return True


class TestFuncs:
    """Built-in self-tests for all 38 protonation groups."""

    @staticmethod
    def test() -> None:
        """Run the full test suite for all ionisable groups."""
        # fmt: off
        smis = [
            # [input, protonated, deprotonated, category]
            ["C#CCO", "C#CCO", "C#CC[O-]", "Alcohol"],
            ["C(=O)N", "NC=O", "[NH-]C=O", "Amide"],
            ["CC(=O)NOC(C)=O", "CC(=O)NOC(C)=O", "CC(=O)[N-]OC(C)=O", "Amide_electronegative"],
            ["COC(=N)N", "COC(N)=[NH2+]", "COC(=N)N", "AmidineGuanidine2"],
            ["Brc1ccc(C2NCCS2)cc1", "Brc1ccc(C2[NH2+]CCS2)cc1", "Brc1ccc(C2NCCS2)cc1", "Amines_primary_secondary_tertiary"],
            ["CC(=O)[n+]1ccc(N)cc1", "CC(=O)[n+]1ccc([NH3+])cc1", "CC(=O)[n+]1ccc(N)cc1", "Anilines_primary"],
            ["CCNc1ccccc1", "CC[NH2+]c1ccccc1", "CCNc1ccccc1", "Anilines_secondary"],
            ["Cc1ccccc1N(C)C", "Cc1ccccc1[NH+](C)C", "Cc1ccccc1N(C)C", "Anilines_tertiary"],
            ["BrC1=CC2=C(C=C1)NC=C2", "Brc1ccc2[nH]ccc2c1", "Brc1ccc2[n-]ccc2c1", "Indole_pyrrole"],
            ["O=c1cc[nH]cc1", "O=c1cc[nH]cc1", "O=c1cc[n-]cc1", "Aromatic_nitrogen_protonated"],
            ["C-N=[N+]=[N@H]", "CN=[N+]=N", "CN=[N+]=[N-]", "Azide"],
            ["BrC(C(O)=O)CBr", "O=C(O)C(Br)CBr", "O=C([O-])C(Br)CBr", "Carboxyl"],
            ["NC(NN=O)=N", "NC(=[NH2+])NN=O", "N=C(N)NN=O", "AmidineGuanidine1"],
            ["C(F)(F)(F)C(=O)NC(=O)C", "CC(=O)NC(=O)C(F)(F)F", "CC(=O)[N-]C(=O)C(F)(F)F", "Imide"],
            ["O=C(C)NC(C)=O", "CC(=O)NC(C)=O", "CC(=O)[N-]C(C)=O", "Imide2"],
            ["CC(C)(C)C(N(C)O)=O", "CN(O)C(=O)C(C)(C)C", "CN([O-])C(=O)C(C)(C)C", "N-hydroxyamide"],
            ["C[N+](O)=O", "C[N+](=O)O", "C[N+](=O)[O-]", "Nitro"],
            ["O=C1C=C(O)CC1", "O=C1C=C(O)CC1", "O=C1C=C([O-])CC1", "O=C-C=C-OH"],
            ["C1CC1OO", "OOC1CC1", "[O-]OC1CC1", "Peroxide2"],
            ["C(=O)OO", "O=COO", "O=CO[O-]", "Peroxide1"],
            ["Brc1cc(O)cc(Br)c1", "Oc1cc(Br)cc(Br)c1", "[O-]c1cc(Br)cc(Br)c1", "Phenol"],
            ["CC(=O)c1ccc(S)cc1", "CC(=O)c1ccc(S)cc1", "CC(=O)c1ccc([S-])cc1", "Phenyl_Thiol"],
            ["C=CCOc1ccc(C(=O)O)cc1", "C=CCOc1ccc(C(=O)O)cc1", "C=CCOc1ccc(C(=O)[O-])cc1", "Phenyl_carboxyl"],
            ["COP(=O)(O)OC", "COP(=O)(O)OC", "COP(=O)([O-])OC", "Phosphate_diester"],
            ["CP(C)(=O)O", "CP(C)(=O)O", "CP(C)(=O)[O-]", "Phosphinic_acid"],
            ["CC(C)OP(C)(=O)O", "CC(C)OP(C)(=O)O", "CC(C)OP(C)(=O)[O-]", "Phosphonate_ester"],
            ["CC1(C)OC(=O)NC1=O", "CC1(C)OC(=O)NC1=O", "CC1(C)OC(=O)[N-]C1=O", "Ringed_imide1"],
            ["O=C(N1)C=CC1=O", "O=C1C=CC(=O)N1", "O=C1C=CC(=O)[N-]1", "Ringed_imide2"],
            ["O=S(OC)(O)=O", "COS(=O)(=O)O", "COS(=O)(=O)[O-]", "Sulfate"],
            ["COc1ccc(S(=O)O)cc1", "COc1ccc(S(=O)O)cc1", "COc1ccc(S(=O)[O-])cc1", "Sulfinic_acid"],
            ["CS(N)(=O)=O", "CS(N)(=O)=O", "CS([NH-])(=O)=O", "Sulfonamide"],
            ["CC(=O)CSCCS(O)(=O)=O", "CC(=O)CSCCS(=O)(=O)O", "CC(=O)CSCCS(=O)(=O)[O-]", "Sulfonate"],
            ["CC(=O)S", "CC(=O)S", "CC(=O)[S-]", "Thioic_acid"],
            ["C(C)(C)(C)(S)", "CC(C)(C)S", "CC(C)(C)[S-]", "Thiol"],
            ["Brc1cc[nH+]cc1", "Brc1cc[nH+]cc1", "Brc1ccncc1", "Aromatic_nitrogen_unprotonated"],
            ["C=C(O)c1c(C)cc(C)cc1C", "C=C(O)c1c(C)cc(C)cc1C", "C=C([O-])c1c(C)cc(C)cc1C", "Vinyl_alcohol"],
            ["CC(=O)ON", "CC(=O)O[NH3+]", "CC(=O)ON", "Primary_hydroxyl_amine"],
        ]

        smis_phos = [
            # [input, protonated, deprotonated1, deprotonated2, category]
            ["O=P(O)(O)OCCCC", "CCCCOP(=O)(O)O", "CCCCOP(=O)([O-])O", "CCCCOP(=O)([O-])[O-]", "Phosphate"],
            ["CC(P(O)(O)=O)C", "CC(C)P(=O)(O)O", "CC(C)P(=O)([O-])O", "CC(C)P(=O)([O-])[O-]", "Phosphonate"],
        ]
        # fmt: on

        cats_with_two_prot_sites = [inf[4] for inf in smis_phos]

        average_pkas = {
            line.split()[0].replace("*", ""): float(line.split()[3])
            for line in ProtSubstructFuncs.load_substructre_smarts_file()
            if line.split()[0] not in cats_with_two_prot_sites
        }
        average_pkas_phos = {
            line.split()[0].replace("*", ""): [
                float(line.split()[3]),
                float(line.split()[6]),
            ]
            for line in ProtSubstructFuncs.load_substructre_smarts_file()
            if line.split()[0] in cats_with_two_prot_sites
        }

        _log_info("Running Tests")
        _log_info("=============")
        _log_info("")

        _log_info("Very Acidic (pH -10000000)")
        _log_info("--------------------------")
        _log_info("")

        args: dict[str, Any] = {
            "min_ph": -10000000,
            "max_ph": -10000000,
            "pka_precision": 0.5,
            "smiles": "",
            "label_states": True,
            "silent": True,
        }

        for smi, protonated, _deprotonated, _category in smis:
            args["smiles"] = smi
            TestFuncs.test_check(args, [protonated], ["PROTONATED"])

        for smi, protonated, _mix, _deprotonated, _category in smis_phos:
            args["smiles"] = smi
            TestFuncs.test_check(args, [protonated], ["PROTONATED"])

        args["min_ph"] = 10000000
        args["max_ph"] = 10000000

        _log_info("")
        _log_info("Very Basic (pH 10000000)")
        _log_info("------------------------")
        _log_info("")

        for smi, _protonated, deprotonated, _category in smis:
            args["smiles"] = smi
            TestFuncs.test_check(args, [deprotonated], ["DEPROTONATED"])

        for smi, _protonated, _mix, deprotonated, _category in smis_phos:
            args["smiles"] = smi
            TestFuncs.test_check(args, [deprotonated], ["DEPROTONATED"])

        _log_info("")
        _log_info("pH is Category pKa")
        _log_info("------------------")
        _log_info("")

        for smi, protonated, deprotonated, category in smis:
            avg_pka = average_pkas[category]

            args["smiles"] = smi
            args["min_ph"] = avg_pka
            args["max_ph"] = avg_pka

            TestFuncs.test_check(args, [protonated, deprotonated], ["BOTH"])

        for smi, protonated, mix, deprotonated, category in smis_phos:
            args["smiles"] = smi

            avg_pka = average_pkas_phos[category][0]
            args["min_ph"] = avg_pka
            args["max_ph"] = avg_pka

            TestFuncs.test_check(args, [mix, protonated], ["BOTH"])

            avg_pka = average_pkas_phos[category][1]
            args["min_ph"] = avg_pka
            args["max_ph"] = avg_pka

            TestFuncs.test_check(args, [mix, deprotonated], ["DEPROTONATED", "DEPROTONATED"])

            avg_pka = 0.5 * (average_pkas_phos[category][0] + average_pkas_phos[category][1])
            args["min_ph"] = avg_pka
            args["max_ph"] = avg_pka
            args["pka_precision"] = 5

            TestFuncs.test_check(args, [mix, deprotonated, protonated], ["BOTH", "BOTH"])

        _log_info("")
        _log_info("Other Tests")
        _log_info("-----------")
        _log_info("")

        smi = "Cc1nc2cc(-c3[nH]c4cc5ccccc5c5c4c3CCN(C(=O)O)[C@@H]5O)cc3c(=O)[nH][nH]c(n1)c23"
        output = list(Protonate({"smiles": smi, "test": False, "silent": True}))

        if "[C-]" in "".join(output).upper():
            msg = "Processing " + smi + " produced a molecule with a carbanion!"
            raise Exception(msg)
        else:
            _log_info("(CORRECT) No carbanion: " + smi)

        smi = "CCCC[C@@H](C(=O)N)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](C(C)C)NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]c2c1cccc2)NC(=O)[C@@H](NC(=O)[C@@H](Cc1ccc(cc1)O)N)CCC(=O)N)C)C)Cc1nc[nH]c1)Cc1ccccc1"
        output = list(Protonate({"smiles": smi, "test": False, "silent": True}))
        if len(output) != 128:
            msg = "Processing " + smi + " produced more than 128 variants!"
            raise Exception(msg)
        else:
            _log_info("(CORRECT) Produced 128 variants: " + smi)

        specific_examples = [
            [
                "O=P(O)(OP(O)(OP(O)(OCC1OC(C(C1O)O)N2C=NC3=C2N=CN=C3N)=O)=O)O",
                (
                    0.5,
                    "[NH3+]c1[nH+]c[nH+]c2c1[nH+]cn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O",
                ),
                (
                    1.0,
                    "[NH3+]c1[nH+]c[nH+]c2c1[nH+]cn2C1OC(COP(=O)(O)OP(=O)([O-])OP(=O)(O)O)C(O)C1O",
                ),
                (
                    2.6,
                    "[NH3+]c1[nH+]c[nH+]c2c1[nH+]cn2C1OC(COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)C(O)C1O",
                ),
                (
                    7.0,
                    "Nc1ncnc2c1ncn2C1OC(COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])C(O)C1O",
                ),
            ],
            [
                "O=P(O)(OP(O)(OCC1C(O)C(O)C(N2C=NC3=C(N)N=CN=C32)O1)=O)OCC(O4)C(O)C(O)C4[N+]5=CC=CC(C(N)=O)=C5",
                (
                    0.5,
                    "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c([NH3+])ncnc54)C(O)C3O)C(O)C2O)c1",
                ),
                (
                    2.5,
                    "NC(=O)c1ccc[n+](C2OC(COP(=O)([O-])OP(=O)([O-])OCC3OC(n4cnc5c([NH3+])ncnc54)C(O)C3O)C(O)C2O)c1",
                ),
                (
                    7.4,
                    "NC(=O)c1ccc[n+](C2OC(COP(=O)([O-])OP(=O)([O-])OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1",
                ),
            ],
        ]
        for example in specific_examples:
            smi = example[0]
            for ph, expected_output in example[1:]:
                output = list(
                    Protonate(
                        {
                            "smiles": smi,
                            "test": False,
                            "min_ph": ph,
                            "max_ph": ph,
                            "pka_precision": 0,
                            "silent": True,
                        }
                    )
                )
                if output[0].strip() == expected_output:
                    _log_info("(CORRECT) " + smi + " at pH " + str(ph) + " is " + output[0].strip())
                else:
                    msg = (
                        smi + " at pH " + str(ph) + " should be " + expected_output + ", but it is " + output[0].strip()
                    )
                    raise Exception(msg)

    @staticmethod
    def test_check(
        args: dict[str, Any],
        expected_output: list[str],
        labels: list[str],
    ) -> None:
        """Verify protonation output against expected values.

        Args:
            args: Arguments to pass to :class:`Protonate`.
            expected_output: Expected SMILES strings.
            labels: Expected state labels (``BOTH``, ``PROTONATED``,
                ``DEPROTONATED``).

        Raises:
            Exception: If the output doesn't match expectations.
        """
        output = list(Protonate(args))
        output = [o.split() for o in output]

        num_states = len(expected_output)

        if len(output) != num_states:
            msg = (
                args["smiles"]
                + " should have "
                + str(num_states)
                + " states at at pH "
                + str(args["min_ph"])
                + ": "
                + str(output)
            )
            UtilFuncs.eprint(msg)
            raise Exception(msg)

        if len({entry[0] for entry in output} - set(expected_output)) != 0:
            msg = (
                args["smiles"]
                + " is not "
                + " AND ".join(expected_output)
                + " at pH "
                + str(args["min_ph"])
                + " - "
                + str(args["max_ph"])
                + "; it is "
                + " AND ".join([entry[0] for entry in output])
            )
            UtilFuncs.eprint(msg)
            raise Exception(msg)

        if len({entry[1] for entry in output} - set(labels)) != 0:
            msg = (
                args["smiles"]
                + " not labeled as "
                + " AND ".join(labels)
                + "; it is "
                + " AND ".join([entry[1] for entry in output])
            )
            UtilFuncs.eprint(msg)
            raise Exception(msg)

        ph_range = sorted({args["min_ph"], args["max_ph"]})
        ph_range_str = "(" + " - ".join(f"{n:.2f}" for n in ph_range) + ")"
        _log_info(
            "(CORRECT) "
            + ph_range_str.ljust(10)
            + " "
            + args["smiles"]
            + " => "
            + " AND ".join([entry[0] for entry in output])
        )


def run(**kwargs: Any) -> None:
    """Run Dimorphite-DL from another Python script.

    Accepts keyword arguments matching the command-line parameters.
    For passing/returning RDKit Mol objects, use :func:`run_with_mol_list`
    instead.

    Args:
        **kwargs: Command-line parameters (see ``--help``).
    """
    main(kwargs)


def protonate_mol_variants(
    mol: Chem.rdchem.Mol,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
    pka_precision: float = 1.0,
    max_variants: int = 128,
    silent: bool = True,
    *,
    rule_provider: type[ProtSubstructFuncs] = ProtSubstructFuncs,
) -> list[Chem.rdchem.Mol]:
    """Enumerate protonation states while preserving atom identity and order.

    Args:
        mol: Input molecule, copied before modification.
        min_ph: Lower pH bound.
        max_ph: Upper pH bound.
        pka_precision: Number of pKa standard deviations in the range.
        max_variants: Maximum intermediate variants retained after each site.
        silent: Suppress warnings for invalid products.
        rule_provider: Provider of the ordered scientific rule inventory.
            Subclass :class:`ProtSubstructFuncs` and override its SMARTS loader
            to select a different fixed inventory. At most eight providers are
            cached; pH requests do not create additional compiled inventories.

    Returns:
        Valid molecular states in first-occurrence order, retaining atom maps.
    """
    args: dict[str, Any] = {
        "min_ph": min_ph,
        "max_ph": max_ph,
        "pka_precision": pka_precision,
        "max_variants": max_variants,
        "silent": silent,
    }
    ProtSubstructFuncs.args = args

    prepared = copy.deepcopy(mol)
    source_index_property = "_atomworks_protonation_source_index"
    for atom in prepared.GetAtoms():
        atom.SetIntProp(source_index_property, atom.GetIdx())
    prepared = UtilFuncs.neutralize_mol(prepared)
    if prepared is None:
        return []
    try:
        prepared = Chem.RemoveHs(prepared)
    except Exception:
        return []
    if prepared is None:
        return []

    subs = rule_provider._substructures_for_ph(min_ph, max_ph, pka_precision)
    sites, mol_used_to_idx_sites = ProtSubstructFuncs.get_prot_sites_and_target_states_from_mol(prepared, subs)
    if mol_used_to_idx_sites is None:
        return []

    new_mols = [mol_used_to_idx_sites]
    properly_formed_mols = [prepared]
    if len(sites) > 0:
        for site in sites:
            new_mols = ProtSubstructFuncs.protonate_site(new_mols, site)
            if len(new_mols) > max_variants:
                new_mols = new_mols[:max_variants]
            properly_formed_mols.extend(new_mols)
    else:
        mol_used_to_idx_sites = Chem.RemoveHs(mol_used_to_idx_sites)
        new_mols = [mol_used_to_idx_sites]
        properly_formed_mols.append(mol_used_to_idx_sites)

    seen: set[str] = set()
    output_mols: list[Chem.rdchem.Mol] = []
    for product in new_mols:
        smi = Chem.MolToSmiles(product, isomericSmiles=True, canonical=True)
        if smi in seen or UtilFuncs.convert_smiles_str_to_mol(smi) is None:
            continue
        seen.add(smi)
        output_mols.append(product)
    if not output_mols:
        for previous in reversed(properly_formed_mols):
            smi = Chem.MolToSmiles(previous, isomericSmiles=True)
            if UtilFuncs.convert_smiles_str_to_mol(smi) is not None:
                output_mols = [previous]
                break
    ordered_products = []
    for product in output_mols:
        order = sorted(
            range(product.GetNumAtoms()),
            key=lambda i: (
                product.GetAtomWithIdx(i).GetIntProp(source_index_property)
                if product.GetAtomWithIdx(i).HasProp(source_index_property)
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
                if atom.HasProp(source_index_property):
                    conf.SetAtomPosition(
                        atom.GetIdx(), source_conf.GetAtomPosition(atom.GetIntProp(source_index_property))
                    )
            product.AddConformer(conf, assignId=True)
        for atom in product.GetAtoms():
            if atom.HasProp(source_index_property):
                atom.ClearProp(source_index_property)
        ordered_products.append(product)
    return ordered_products


def run_with_mol_list(mol_lst: list[Chem.rdchem.Mol], **kwargs: Any) -> list[Chem.rdchem.Mol]:
    """Run Dimorphite-DL on a list of RDKit Mol objects.

    Protonates Mol objects directly while preserving their properties and
    atom identity.

    Args:
        mol_lst: Input RDKit Mol objects.
        **kwargs: Additional command-line parameters (must not include
            ``smiles``, ``smiles_file``, ``output_file``, or ``test``).

    Returns:
        A list of protonated RDKit Mol objects with properties preserved.

    Raises:
        Exception: If forbidden keyword arguments are provided.
    """
    for bad_arg in ["smiles", "smiles_file", "output_file", "test"]:
        if bad_arg in kwargs:
            msg = (
                "You're using Dimorphite-DL's run_with_mol_list(mol_lst, "
                + '**kwargs) function, but you also passed the "'
                + bad_arg
                + '" argument. Did you mean to use the '
                + "run(**kwargs) function instead?"
            )
            UtilFuncs.eprint(msg)
            raise Exception(msg)

    mols: list[Chem.rdchem.Mol] = []
    for m in mol_lst:
        props = m.GetPropsAsDict()
        variants = protonate_mol_variants(
            m,
            min_ph=float(kwargs.get("min_ph", 6.4)),
            max_ph=float(kwargs.get("max_ph", 8.4)),
            pka_precision=float(kwargs.get("pka_precision", 1.0)),
            max_variants=int(kwargs.get("max_variants", 128)),
            silent=bool(kwargs.get("silent", True)),
            rule_provider=kwargs.get("rule_provider", ProtSubstructFuncs),
        )
        for v in variants:
            for prop, val in props.items():
                if type(val) is int:
                    v.SetIntProp(prop, val)
                elif type(val) is float:
                    v.SetDoubleProp(prop, val)
                elif type(val) is bool:
                    v.SetBoolProp(prop, val)
                else:
                    v.SetProp(prop, str(val))
            mols.append(v)

    return mols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
