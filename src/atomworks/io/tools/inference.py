import io
import logging
import os
from abc import ABC
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from rdkit import Chem
from rdkit.Chem import AllChem

import atomworks.io.transforms.atom_array as ta
from atomworks.common import KeyToIntMapper, exists
from atomworks.constants import (
    CCD_MIRROR_PATH,
    STANDARD_AA_ONE_LETTER,
    STANDARD_DNA_ONE_LETTER,
    STANDARD_RNA,
    UNKNOWN_LIGAND,
)
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.template import add_missing_atoms_for_chain, infer_bonds_from_residue_names
from atomworks.io.tools.fasta import (
    infer_chain_type_from_one_letter,
    one_letter_to_ccd_code,
    split_generalized_fasta_sequence,
)
from atomworks.io.tools.rdkit import atom_array_from_rdkit, inchi_to_rdkit
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.bonds import (
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
    hash_atom_array,
)
from atomworks.io.utils.ccd import (
    atom_array_from_ccd_code,
    check_ccd_codes_are_available,
    get_chain_type_from_ccd_code,
    get_chem_comp_type,
    parse_ccd_cif,
)
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIF_LIKE_EXTENSIONS, read_any
from atomworks.io.utils.query import AtomSelection

logger = logging.getLogger("atomworks.io")


class ChemicalComponent(ABC):  # noqa: B024
    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(args_dict: dict) -> "ChemicalComponent":
        if "seq" in args_dict:
            args_dict = {k: v for k, v in args_dict.items() if k != "is_polymer"}
            return SequenceComponent(**args_dict)
        elif "smiles" in args_dict:
            return SmilesComponent(**args_dict)
        elif "inchi" in args_dict:
            return InChIComponent(**args_dict)
        elif "path" in args_dict and str(args_dict["path"]).lower().endswith(
            (".sdf", ".sdf.gz", ".sdf.gzip", ".sdf.zst")
        ):
            return SDFComponent(**args_dict)
        elif "path" in args_dict and any(extension in args_dict["path"] for extension in CIF_LIKE_EXTENSIONS):
            return CIFOrPDBFileComponent(**args_dict)
        elif "ccd_code" in args_dict:
            return CCDComponent(**args_dict)
        else:
            raise ValueError(f"Unknown chemical component type: {args_dict=}")


@dataclass
class SequenceComponent(ChemicalComponent):
    seq: str | list[str]
    chain_type: ChainType | None = None
    chain_id: str | None = None
    include_bonds: bool = True
    msa_path: os.PathLike | None = None

    @staticmethod
    def infer_chain_type(seq: str | list[str]) -> ChainType:
        """Infer chain type from sequence notation."""
        return infer_chain_type_from_one_letter(seq)

    @staticmethod
    def assert_valid_chain_type(seq: list[str], chain_type: ChainType, allow_other: bool = False) -> bool:
        """Asserts that all the CCD codes in the sequence are valid for the given chain type.

        Args:
            seq (list[str]): List of three-letter CCD codes.
            chain_type (ChainType): The chain type to check against.
            allow_other (bool): If True, allow non-CCD codes (e.g., custom NCAA) to be valid.

        Ignore non-CCD codes (e.g., custom NCAA) which are presumed to be valid (and are mapped to "other")
        """
        ccd_codes = set(seq)
        chem_comp_types = {get_chem_comp_type(ccd_code) for ccd_code in ccd_codes}
        if allow_other:
            chem_comp_types.discard("OTHER")

        valid_chem_comp_types = ChainTypeInfo.VALID_CHEM_COMP_TYPES.get(chain_type, chem_comp_types)
        if not chem_comp_types.issubset(valid_chem_comp_types):
            raise ValueError(f"Invalid {chain_type=} for {chem_comp_types=}. Valid are {valid_chem_comp_types=}")

    @staticmethod
    def from_seq(
        seq: str | list[str], *, chain_type: ChainType | str = None, is_polymer: bool | None = None
    ) -> "SequenceComponent":
        chain_type = chain_type or SequenceComponent.infer_chain_type(seq)

        if chain_type in ChainTypeInfo.PROTEINS:
            return Protein(seq=seq, chain_type=chain_type)
        elif chain_type == ChainType.RNA:
            return RNA(seq=seq, chain_type=chain_type)
        elif chain_type == ChainType.DNA:
            return DNA(seq=seq, chain_type=chain_type)
        else:
            return SequenceComponent(seq=seq, chain_type=chain_type)

    @property
    def is_polymer(self) -> bool:
        return self.chain_type.is_polymer()

    def __post_init__(self):
        # If the chain type is not provided, infer it from the sequence
        self.chain_type = self.chain_type or SequenceComponent.infer_chain_type(self.seq)
        self.chain_type = ChainType.as_enum(self.chain_type)

        # If the sequence is a string, split it into a list of one-letter codes
        if isinstance(self.seq, str):
            self.seq = split_generalized_fasta_sequence(self.seq)

        # Process sequence into CCD codes
        if isinstance(self.seq, str):
            self.seq = split_generalized_fasta_sequence(self.seq)

        self.seq = one_letter_to_ccd_code(self.seq, self.chain_type, check_ccd_codes=False)

        # Validate chain type
        SequenceComponent.assert_valid_chain_type(self.seq, self.chain_type, allow_other=True)


@dataclass
class LigandComponent(ChemicalComponent):
    def __post_init__(self):
        self.chain_type = ChainType.as_enum(self.chain_type)

        if self.is_polymer:
            raise ValueError(f"{self.__class__.__name__} must have 'is_polymer=False'")

        if self.chain_type != ChainType.NON_POLYMER:
            raise ValueError(f"{self.__class__.__name__} must have 'chain_type=ChainType.NON_POLYMER'")


@dataclass
class CCDComponent(LigandComponent):
    ccd_code: str
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None


@dataclass
class SmilesComponent(LigandComponent):
    smiles: str
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None
    res_name: str = UNKNOWN_LIGAND


@dataclass
class InChIComponent(LigandComponent):
    inchi: str
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None
    res_name: str = UNKNOWN_LIGAND


@dataclass
class SDFComponent(LigandComponent):
    path: os.PathLike | io.StringIO
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None
    res_name: str = UNKNOWN_LIGAND


@dataclass
class CIFOrPDBFileComponent(ChemicalComponent):
    path: os.PathLike | io.StringIO
    msa_paths: dict[str, os.PathLike] | None = None
    custom_parse_kwargs: dict[str, Any] | None = None
    chain_id: str | None = None

    def __post_init__(self):
        """Initialize the component by parsing the structure file."""
        if self._is_ccd_cif_file():
            self._parse_ccd_style_cif()
        else:
            self._parse_standard_pdb_or_cif()

        # Validate and apply chain_id if provided
        self._validate_and_apply_chain_id()

    def _is_ccd_cif_file(self) -> bool:
        """Check if we are given a CCD CIF file, which by convention includes the _chem_comp_atom field but not the atom_site field"""
        # If not a CIF file, return False
        cif = read_any(self.path)

        if not isinstance(cif, pdbx.CIFFile | pdbx.BinaryCIFFile):
            return False

        keys = list(cif.block.keys())

        has_atom_site = "atom_site" in keys
        has_chem_comp_atom = "chem_comp_atom" in keys

        return has_chem_comp_atom and not has_atom_site

    def _parse_ccd_style_cif(self) -> None:
        """Parse a CCD-style CIF file."""

        if self.custom_parse_kwargs is not None:
            raise ValueError("Custom parse kwargs are not supported for CCD CIF files.")

        logger.warning(
            f"CCD CIF file detected: {self.path}. "
            "This file will be parsed as a CCD CIF file rather than a regular CIF file "
            "(e.g., with an `atom_site` category)."
        )

        self.atom_array = parse_ccd_cif(read_any(self.path))
        self.atom_array.set_annotation("is_polymer", np.full(len(self.atom_array), False))
        self.chain_ids = np.unique(self.atom_array.chain_id)

        # Set occupancy to all 1s since we presumably want to predict everything
        self.atom_array.occupancy = np.full(len(self.atom_array), 1.0)

    def _parse_standard_pdb_or_cif(self) -> None:
        """Parse a standard PDB or CIF structure file."""
        if self.custom_parse_kwargs is None:
            self.custom_parse_kwargs = {}

        # We add missing atoms later to the fully-concatenated inference AtomArray.
        parse_kwargs = {
            **STANDARD_PARSER_ARGS,
            "add_missing_atoms": False,
            "return_atom_array_plus": False,
        } | self.custom_parse_kwargs

        if parse_kwargs["add_missing_atoms"]:
            logger.warning(
                "Missing atoms will be added later to the fully-concatenated inference AtomArray. "
                "It is recommended to set this argument to False in initial CIFOrPDBFileComponent parsing."
            )

        parsing_results = parse(self.path, config=ParseConfig(**parse_kwargs))

        if "assemblies" in parsing_results:
            assemblies = parsing_results["assemblies"]
            # We will keep only the first assembly that was parsed
            first_assembly_id = next(iter(assemblies.keys()))

            if len(assemblies) > 1:
                logger.warning(
                    f"Multiple biological assemblies found in {self.path} and none were specified. "
                    f"Only the first assembly (assembly_id={first_assembly_id}) will be used for inference. "
                    "If you would like to use a different assembly, please specify this in the `parse_kwargs`."
                )

            atom_array_stack = assemblies[first_assembly_id]
        else:
            atom_array_stack = parsing_results["asym_unit"]

        if atom_array_stack.stack_depth() > 1:
            logger.warning(
                f"Multiple models found in {self.path}. Only the first model will be used for inference. "
                "If you would like to use a different model, please specify this in the `parse_kwargs`."
            )

        structure_file_atom_array = atom_array_stack[0]
        self.chain_ids = np.unique(structure_file_atom_array.chain_id)
        self.atom_array = structure_file_atom_array

    def _validate_and_apply_chain_id(self) -> None:
        """Validate chain_id can be applied and assign it if provided.

        Raises:
            ValueError: If chain_id is provided but structure has multiple molecules.
        """
        if self.chain_id is None:
            return

        # Check if structure is a single connected molecule using bond connectivity
        # Use chain_id level for the coarse graph
        nodes, edges = get_coarse_graph_as_nodes_and_edges(self.atom_array, "chain_id")
        connected_components = get_connected_nodes(nodes, edges)

        if len(connected_components) > 1:
            raise ValueError(
                f"Cannot assign chain_id '{self.chain_id}' to CIF file '{self.path}': "
                f"structure contains {len(connected_components)} disconnected molecules. "
                f"chain_id can only be assigned when all atoms form a single connected molecule."
            )

        # Assign the chain_id to all atoms
        self.atom_array.chain_id = np.full(len(self.atom_array), self.chain_id)
        self.chain_ids = np.array([self.chain_id])


@dataclass
class Polymer(SequenceComponent):
    is_polymer: bool = True


@dataclass
class Protein(SequenceComponent):
    chain_type: ChainType = ChainType.POLYPEPTIDE_L

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_AA_ONE_LETTER)


@dataclass
class RNA(SequenceComponent):
    chain_type: ChainType = ChainType.RNA

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_RNA)


@dataclass
class DNA(SequenceComponent):
    chain_type: ChainType = ChainType.DNA

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_DNA_ONE_LETTER)


@dataclass
class Peptide(SequenceComponent):
    chain_type: ChainType = ChainType.POLYPEPTIDE_L
    is_polymer: bool = False


def read_chai_fasta(fasta_path: Path) -> list[ChemicalComponent]:
    from biotite.sequence.io.fasta import FastaFile

    fasta = FastaFile.read(fasta_path)

    components = []
    for metadata, content in fasta.items():
        metadata = metadata.lower()
        if metadata.startswith("ligand"):
            components.append(SmilesComponent(smiles=content))
        elif metadata.endswith(".sdf"):
            components.append(sdf_to_annotated_atom_array(path=content))
        else:
            if "protein" in metadata:
                components.append(Protein(seq=content))
            elif "rna" in metadata:
                components.append(RNA(seq=content))
            elif "dna" in metadata:
                components.append(DNA(seq=content))
            elif "peptide" in metadata:
                components.append(Peptide(seq=content))
            else:
                components.append(SequenceComponent.from_seq(content))
    return components


def build_chain_atom_array(
    seq: list[str],
    chain_id: str,
    chain_type: ChainType,
    *,
    include_bonds: bool = False,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    """Build an AtomArray from a sequence of CCD codes for a single chain.

    Calls :py:func:`add_missing_atoms_for_chain` directly to build complete chains
    from CCD templates, then patches ``occupancy`` to ``1.0`` since every atom in
    the result is conceptually present (not unresolved).

    Args:
        seq: Sequence of three-letter CCD codes.
        chain_id: Chain ID for all residues.
        chain_type: Chain type (determines ``is_polymer`` and other annotations).
        include_bonds: If ``True``, infer bonds, remove leaving atoms, and fix charges/bond types
            via :py:func:`infer_bonds_from_residue_names`. Defaults to ``False``.
        ccd_mirror_path: Path to local CCD mirror.

    Returns:
        AtomArray with residue IDs ``1..len(seq)``, ``occupancy=1.0``, and ``b_factor=nan``.
    """
    chain_type = ChainType.as_enum(chain_type)
    templates, _ = add_missing_atoms_for_chain(
        sequence=list(seq),
        res_ids=list(range(1, len(seq) + 1)),
        chain_id=chain_id,
        chain_type=chain_type,
        ccd_mirror_path=ccd_mirror_path,
    )

    atom_array = struc.concatenate(templates)
    atom_array.set_annotation("occupancy", np.ones(len(atom_array)))

    if include_bonds:
        atom_array = infer_bonds_from_residue_names(
            atom_array,
            sanitize=True,
            ccd_mirror_path=ccd_mirror_path,
        )

    return atom_array


def sequence_to_annotated_atom_array(
    seq: list[str],
    chain_id: str,
    *,
    chain_type: ChainType | str = None,
    include_bonds: bool = True,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    **kwargs,
) -> AtomArray:
    if isinstance(seq, str):
        seq = one_letter_to_ccd_code(
            split_generalized_fasta_sequence(seq), chain_type=chain_type, check_ccd_codes=False
        )

    seq = np.asarray(seq)
    chain_type = ChainType.as_enum(chain_type or SequenceComponent.infer_chain_type(seq))

    # Ensure that the sequence is a valid combination of existing 3-letter CCD codes
    ccd_codes_in_seq = set(seq)
    if UNKNOWN_LIGAND in ccd_codes_in_seq:
        raise ValueError(
            f"Unknown ligand `{UNKNOWN_LIGAND}` found in sequence. If you want to pass a ligand, that "
            f"is not in the CCD, use a SMILES string or SDF file instead."
        )

    check_ccd_codes_are_available(ccd_codes_in_seq, ccd_mirror_path=ccd_mirror_path, mode="raise")

    return build_chain_atom_array(
        seq,
        chain_id,
        chain_type,
        include_bonds=include_bonds,
        ccd_mirror_path=ccd_mirror_path,
    )


def smiles_to_annotated_atom_array(
    smiles: str,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
    res_name: str = UNKNOWN_LIGAND,
) -> AtomArray:
    if backend == "rdkit":
        from atomworks.io.tools.rdkit import atom_array_from_rdkit, smiles_to_rdkit

        mol = smiles_to_rdkit(smiles)
        try:
            # ... generate a conformer to keep the stereochemistry encoded in the SMILES
            #   NOTE: This may stall for 40ish seconds for some difficult molecules like HEM
            #   TODO: Migrate the timeout utils to atomworks.io so we can timeout here.
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.maxAttempts = 1
            AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
        except Exception:
            pass

        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), res_name))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type), dtype=np.int8))
    array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
    array.set_annotation("stereo", np.full(array.array_length(), "N"))
    array.set_annotation("is_backbone_atom", np.full(array.array_length(), False))

    return array


def inchi_to_annotated_atom_array(
    inchi: str,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    res_name: str = UNKNOWN_LIGAND,
) -> AtomArray:
    """Convert an InChI string to an annotated AtomArray.

    Args:
        inchi: The InChI string representing the molecule.
        chain_id: The chain ID to assign.
        chain_type: The chain type. Defaults to "non-polymer".
        is_polymer: Whether the component is a polymer. Defaults to False.
        res_name: The residue name. Defaults to UNKNOWN_LIGAND.

    Returns:
        AtomArray with standard ligand annotations.
    """
    mol = inchi_to_rdkit(inchi)
    try:
        # ... generate a conformer
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.maxAttempts = 1
        AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    except Exception:
        pass

    array = atom_array_from_rdkit(mol)

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), res_name))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type), dtype=np.int8))
    array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
    array.set_annotation("stereo", np.full(array.array_length(), "N"))
    array.set_annotation("is_backbone_atom", np.full(array.array_length(), False))

    return array


def sdf_to_annotated_atom_array(
    path: io.StringIO | os.PathLike,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    res_name: str = UNKNOWN_LIGAND,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
) -> AtomArray:
    if backend == "rdkit":
        from atomworks.io.tools.rdkit import atom_array_from_rdkit, sdf_to_rdkit

        mol = sdf_to_rdkit(path)
        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), res_name))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type), dtype=np.int8))
    array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
    array.set_annotation("stereo", np.full(array.array_length(), "N"))
    array.set_annotation("is_backbone_atom", np.full(array.array_length(), False))
    return array


def ccd_code_to_annotated_atom_array(
    ccd_code: list[str],
    chain_id: str,
    *,
    chain_type: ChainType | str = None,
    is_polymer: bool | None = None,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    check_ccd_codes_are_available([ccd_code], ccd_mirror_path=ccd_mirror_path, mode="raise")

    # ... build the atom array
    array = atom_array_from_ccd_code(ccd_code)

    # ... set or infer chain type
    chain_type = chain_type or get_chain_type_from_ccd_code(ccd_code)
    is_polymer = is_polymer or chain_type.is_polymer()

    # ... update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), ccd_code))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type), dtype=np.int8))

    return array


def assign_res_name_from_atom_array_hash(atom_array: AtomArray, hash_to_id: KeyToIntMapper) -> AtomArray:
    """Assigns a residue name to an array based on its hash.

    The residue names will be assigned as `L:{id}` where `id` is a unique integer assigned to each hash.

    Args:
        ligand_array (AtomArray): The ligand array to assign a residue name to.
        ligand_hash_to_id (KeyToIntMapper): A mapper from ligand hash to ligand ID.
    """
    ligand_hash = hash_atom_array(atom_array, annotations=["element", "atom_name"], bond_order=True)
    ligand_id = hash_to_id(ligand_hash)
    atom_array.res_name = np.full(atom_array.array_length(), f"L:{ligand_id}")
    return atom_array


def standardize_component_keys(component_dict: dict) -> dict:
    """Standardize component dictionary keys for compatibility with AF3's inference API.

    Maps:
        - "sequence" -> "seq"
        - "id" -> "chain_id"
    """
    # Create a copy to avoid modifying the original
    standardized = component_dict.copy()

    # Handle sequence/seq mapping
    if "sequence" in standardized and "seq" not in standardized:
        standardized["seq"] = standardized.pop("sequence")
    elif "sequence" in standardized and "seq" in standardized:
        raise ValueError(f"Both 'sequence' and 'seq' are present in {standardized=}")

    # Handle id/chain_id mapping
    if "id" in standardized and "chain_id" not in standardized:
        standardized["chain_id"] = standardized.pop("id")

    return standardized


def build_msa_paths_by_chain_id_from_component_list(components: list[ChemicalComponent]) -> dict[str, os.PathLike]:
    """Build a dictionary of MSA paths by chain ID from a list of ChemicalComponent objects.

    The composed dictionary may be encoded as extra metadata in the CIF file, and ultimately loaded
    into `chain_info` through `parse`.
    """
    msa_paths_by_chain_id = {}
    for component in components:
        if hasattr(component, "msa_path") and component.msa_path is not None:
            msa_paths_by_chain_id[component.chain_id] = component.msa_path
        elif hasattr(component, "msa_paths") and component.msa_paths is not None:
            for chain_id, msa_path in component.msa_paths.items():
                msa_paths_by_chain_id[chain_id] = msa_path

    return msa_paths_by_chain_id


def _finalize_inference_atom_array(atom_array: AtomArray) -> AtomArray:
    """Add standard post-concat annotations to a concatenated inference atom array."""
    ensure_annotations(atom_array, "chem_comp_type", "atomic_number")

    if "transformation_id" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("transformation_id", np.full(len(atom_array), "1"))

    # Entity + ID annotations must come before IIDs (iid = id + "_" + transformation_id)
    atom_array = ta.add_id_and_entity_annotations(atom_array)
    atom_array = ta.add_iid_annotations(atom_array)

    return atom_array


def _add_bonds_from_strings(
    atom_array: AtomArray,
    bond_strings: list[tuple[str, str]],
) -> AtomArray:
    """Add explicit covalent bonds from CHAIN/RESNAME/RESID/ATOMNAME string pairs.

    Bonds are added as SINGLE bonds. Post-processing (leaving-atom removal, charge and
    bond-type correction) is handled by a subsequent call to
    :py:func:`~atomworks.io.template.infer_bonds_from_residue_names`.

    Uses :py:class:`~atomworks.io.utils.query.AtomSelection` to parse bond strings,
    supporting wildcards (``"*"``) and the same syntax as the rest of the selection API.
    """
    if not bond_strings:
        return atom_array

    raw_bonds = []
    for atom1_str, atom2_str in bond_strings:
        idx1 = AtomSelection.from_selection_str(atom1_str).get_idxs(atom_array)
        idx2 = AtomSelection.from_selection_str(atom2_str).get_idxs(atom_array)
        if len(idx1) != 1 or len(idx2) != 1:
            raise ValueError(
                f"Bond specification must resolve to exactly one atom each: "
                f"{atom1_str!r} → {len(idx1)} atoms, {atom2_str!r} → {len(idx2)} atoms"
            )
        raw_bonds.append((int(idx1[0]), int(idx2[0]), struc.BondType.SINGLE))

    new_bonds = struc.BondList(atom_array.array_length(), np.array(raw_bonds, dtype=np.uint32))
    atom_array.bonds = atom_array.bonds.merge(new_bonds) if atom_array.bonds is not None else new_bonds
    return atom_array


def components_to_atom_array(
    components: list[ChemicalComponent | dict],
    bonds: list[str] | None = None,
    return_components: bool = False,
) -> AtomArray | list[ChemicalComponent]:
    """Build an AtomArray from a list of ChemicalComponent objects and supporting details (bonds).

    Args:
        components (list[ChemicalComponent | dict]): List of ChemicalComponent objects or dictionaries that can be
            converted to ChemicalComponent objects using ChemicalComponent.from_dict().
        bonds (list[str]): List of tuples of atom ids to be bonded. We will add them like spoof `struct_conn` entries,
            ensuring that we remove leaving groups as appropriate. Bonds tuples must be in the format (1-indexed!):
            ```
            (CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME, CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME)
            ```
            e.g., [("A/THR/4/CG", "D/L:1/0/O13"), ("A/CYS/5/SG",  "A/CYS/137/SG")]
        return_components (bool): If True, return the components list as well as the AtomArray. Useful for e.g., mapping
            components to generated chain IDs or inferred chain types.

    Note:
        If manually specifying bonds, we recommend visualizing the bond graph with `matplotlib`
        to ensure that the bonds are correctly specified.
        The res_id numbering follows the RCSB convention (1-indexed).
        Custom CCD entries can be registered using :py:func:`~atomworks.io.utils.ccd.register_custom_ccd_entry`
        to override standard CCD definitions before calling this function.

    Returns:
        AtomArray: The assembled AtomArray, used for visualization or inference.

    Raises:
        ValueError: If there are duplicate chain_ids across input Components
        ValueError: If there are duplicate chain_ids that correspond to non-identical molecular entities.
    """
    standardized_components = []
    for component in components:
        if isinstance(component, dict):
            # Standardize the keys
            component = standardize_component_keys(component)

            # If chain_id is a list, create copies for each chain_id
            if "chain_id" in component and isinstance(component["chain_id"], list):
                for single_chain_id in component["chain_id"]:
                    component_copy = component.copy()
                    component_copy["chain_id"] = single_chain_id
                    standardized_components.append(component_copy)
            else:
                standardized_components.append(component)
        elif isinstance(component, ChemicalComponent):
            standardized_components.append(component)
        else:
            raise ValueError(f"Unknown component type: {type(component)}")

    # Ensure that all components are ChemicalComponent objects
    components = [
        ChemicalComponent.from_dict(component) if isinstance(component, dict) else component
        for component in standardized_components
    ]

    chain_ids = []

    # Get existing chain ids
    for component in components:
        if hasattr(component, "chain_id") and exists(component.chain_id):
            chain_ids.append(component.chain_id)
        elif hasattr(component, "chain_ids") and exists(component.chain_ids):
            chain_ids.extend(component.chain_ids)

    # Raise an exception if there are duplicate chain_ids across input components
    # Note that intra-component duplicates may still be present due to multiple transformations of the same asym_unit
    if len(chain_ids) > len(set(chain_ids)):
        duplicated_chain_ids = set()
        for chain_id in chain_ids:
            if chain_ids.count(chain_id) > 1:
                duplicated_chain_ids.add(chain_id)
        chain_counter = Counter(chain_ids)
        duplicated_chain_ids = {chain_id for chain_id, count in chain_counter.items() if count > 1}
        raise ValueError(
            f"The following chain_ids were present in multiple input components: {duplicated_chain_ids}. "
            f"Please rename chains to avoid this issue."
        )

    # Instantiate a chain id generator
    chain_id_generator = create_chain_id_generator(chain_ids)

    atom_arrays = []
    ligand_hash_to_id = KeyToIntMapper()  # ... to keep track of identical ligands
    for component in components:
        # CIFOrPDBFileComponents already have parsed AtomArrays
        if isinstance(component, CIFOrPDBFileComponent):
            atom_array = component.atom_array
            if np.any(atom_array.chain_id == ""):
                atom_array.chain_id = np.full(atom_array.array_length(), next(chain_id_generator))
                logger.warning(
                    f"Chain ID was not set for {component.path}. "
                    f"The next available chain ID was assigned, assuming that this is a single-chain structure: {atom_array.chain_id[0]}"
                )
            atom_arrays.append(component.atom_array)
            continue

        component.chain_id = component.chain_id or next(chain_id_generator)

        if isinstance(component, SequenceComponent):
            # include_bonds=False: global bond inference runs after all chains are concatenated
            atom_arrays.append(sequence_to_annotated_atom_array(**{**component.as_dict(), "include_bonds": False}))
        elif isinstance(component, SmilesComponent):
            ligand_array = smiles_to_annotated_atom_array(**component.as_dict())
            if component.res_name == UNKNOWN_LIGAND:
                ligand_array = assign_res_name_from_atom_array_hash(ligand_array, ligand_hash_to_id)
            atom_arrays.append(ligand_array)
        elif isinstance(component, InChIComponent):
            ligand_array = inchi_to_annotated_atom_array(**component.as_dict())
            if component.res_name == UNKNOWN_LIGAND:
                ligand_array = assign_res_name_from_atom_array_hash(ligand_array, ligand_hash_to_id)
            atom_arrays.append(ligand_array)
        elif isinstance(component, CCDComponent):
            atom_arrays.append(ccd_code_to_annotated_atom_array(**component.as_dict()))
        elif isinstance(component, SDFComponent):
            ligand_array = sdf_to_annotated_atom_array(**component.as_dict())
            if component.res_name == UNKNOWN_LIGAND:
                ligand_array = assign_res_name_from_atom_array_hash(ligand_array, ligand_hash_to_id)
            atom_arrays.append(ligand_array)
        else:
            raise ValueError(f"Unknown chemical component type: {type(component)}")

    # add required per-array annotations before concatenation so biotite does not fill
    # missing annotations with defaults when arrays have heterogeneous annotation sets
    for arr in atom_arrays:
        if "b_factor" not in arr.get_annotation_categories():
            arr.set_annotation("b_factor", np.full(arr.array_length(), np.nan))
        if "transformation_id" not in arr.get_annotation_categories():
            arr.set_annotation("transformation_id", np.full(arr.array_length(), "1"))

    # ... concatenate all atom arrays into a single AtomArray
    atom_array = struc.concatenate(atom_arrays)

    # ... add explicit struct-conn bonds before infer_bonds_from_residue_names so they
    #     participate in leaving-atom removal and charge/bond-type correction
    if bonds:
        atom_array = _add_bonds_from_strings(atom_array, bonds)

    # ... infer CCD bonds, remove leaving atoms, fix charges and bond types.
    #     Merges with existing bonds (including struct-conn bonds added above).
    atom_array = infer_bonds_from_residue_names(
        atom_array,
        sanitize=True,
    )

    # ... remove hydrogens
    atom_array = ta.remove_hydrogens(atom_array)

    # ... add atomic_number, transformation_id, and all IID/entity/ID annotations.
    #     Must come AFTER bond inference so molecule_id reflects full bond connectivity.
    atom_array = _finalize_inference_atom_array(atom_array)

    # Raise an error if chain_ids with the same name correspond to different entities
    for chain_id in np.unique(atom_array.chain_id):
        subsetted_atom_array = atom_array[atom_array.chain_id == chain_id]
        if len(np.unique(subsetted_atom_array.chain_entity)) > 1:
            raise ValueError(
                f"Chain ID {chain_id} corresponds to multiple non-identical molecular entities. "
                f"Please ensure that each chain_id corresponds to only a single entity."
            )

    if return_components:
        return atom_array, components

    return atom_array
