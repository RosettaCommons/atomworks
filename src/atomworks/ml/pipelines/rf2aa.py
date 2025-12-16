from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from biotite.structure import AtomArray

from atomworks.common import exists
from atomworks.constants import AF3_EXCLUDED_LIGANDS
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinPolyResIdxAnnotation,
    SortLikeRF2AA,
)
from atomworks.ml.transforms.atom_frames import AddAtomFrames
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import (
    AddData,
    ApplyFunction,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
)
from atomworks.ml.transforms.bonds import (
    AddRF2AABondFeaturesMatrix,
    AddRF2AATraversalDistanceMatrix,
    AddTokenBondAdjacency,
)
from atomworks.ml.transforms.chirals import AddRF2AAChiralFeatures
from atomworks.ml.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.encoding import EncodeAtomArray, atom_array_from_encoding
from atomworks.ml.transforms.feature_aggregation.rf2aa import AggregateFeaturesLikeRF2AA
from atomworks.ml.transforms.featurize_unresolved_residues import MaskPolymerResiduesWithUnresolvedFrameAtoms
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RemoveUnsupportedChainTypes,
)
from atomworks.ml.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeRF2AA,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from atomworks.ml.transforms.openbabel_utils import (
    AddOpenBabelMoleculesForAtomizedMolecules,
    GetChiralCentersFromOpenBabel,
)
from atomworks.ml.transforms.rf2aa_assumptions import AssertRF2AAAssumptions, _is_atom
from atomworks.ml.transforms.symmetry import (
    AddPostCropMoleculeEntityToFreeFloatingLigands,
    CreateSymmetryCopyAxisLikeRF2AA,
)
from atomworks.ml.transforms.template import AddRFTemplates, FeaturizeTemplatesLikeRF2AA, RF2AATemplate
from atomworks.ml.utils.numpy import get_connected_components_from_adjacency


class RF2AAInputs(NamedTuple):
    """A named tuple containing the inputs to the RF2AA model."""

    seq: np.ndarray
    msa: np.ndarray
    msa_masked: np.ndarray
    msa_full: np.ndarray
    mask_msa: np.ndarray
    xyz: np.ndarray  # `true_crds` in rf2aa code
    mask: np.ndarray  # `mask_crds` in rf2aa code
    idx_pdb: np.ndarray
    xyz_t: np.ndarray
    t1d: np.ndarray
    mask_t: np.ndarray
    xyz_prev: np.ndarray
    mask_prev: np.ndarray
    same_chain: np.ndarray
    unclamp: np.ndarray
    negative: np.ndarray
    atom_frames: np.ndarray
    bond_feats: np.ndarray
    dist_matrix: np.ndarray
    chirals: np.ndarray
    ch_label: np.ndarray
    symmgp: str
    task: str
    example_id: str  # `item` in rf2aa code

    @classmethod
    def from_dict(cls, data: dict) -> "RF2AAInputs":
        return cls(**{key: data[key] for key in cls._fields})

    def to_atom_array(self, symm_copy: int = 0) -> AtomArray:
        """Decode the inputs into an AtomArray for the given `symm_copy`."""
        is_batched = self.xyz.ndim == 5

        seq = self.msa[0, 0, 0] if is_batched else self.msa[0, 0]
        token_is_atom = _is_atom(seq).unsqueeze(1).expand((len(seq), 36))

        # Get the symmetric copy (i) for the polymer, but the first automorph for the ligand
        atomized = token_is_atom[:, 0]
        xyz = self.xyz[0, symm_copy] if is_batched else self.xyz[symm_copy]
        mask = self.mask[0, symm_copy] if is_batched else self.mask[symm_copy]
        if atomized.any():
            xyz[atomized] = self.xyz[0, symm_copy, atomized] if is_batched else self.xyz[symm_copy, atomized]
            mask[atomized] = self.mask[0, 0, atomized] if is_batched else self.mask[0, atomized]

        molecule_entity = self.ch_label[0] if is_batched else self.ch_label

        chain_id = np.empty(len(seq))
        same_chain = self.same_chain[0] if is_batched else self.same_chain
        for i, idxs in enumerate(get_connected_components_from_adjacency(same_chain.numpy())):
            chain_id[idxs] = i

        return atom_array_from_encoding(
            encoded_coord=xyz,
            encoded_mask=mask,
            encoded_seq=seq,
            chain_id=chain_id,
            chain_entity=molecule_entity,
            encoding=RF2AA_ATOM36_ENCODING,
            token_is_atom=token_is_atom,
        )

    def num_res(self) -> int:
        is_batched = self.xyz.ndim == 5
        msa = self.msa[0] if is_batched else self.msa
        return (~_is_atom(msa[0, 0])).sum().item()

    def num_atoms(self) -> int:
        is_batched = self.xyz.ndim == 5
        msa = self.msa[0] if is_batched else self.msa
        return (_is_atom(msa[0, 0])).sum().item()

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self._fields}


def _is_inference(data: dict) -> bool:
    return data.get("is_inference", False)


def _is_training(data: dict) -> bool:
    return not data.get("is_inference", False)


def _convert_feats_to_rf2aa_input_tuple(data: dict) -> RF2AAInputs:
    data["feats"] = RF2AAInputs.from_dict(data["feats"])
    return data


def build_rf2aa_transform_pipeline(
    *,
    protein_msa_dirs: list[dict],
    rna_msa_dirs: list[dict],
    # Recycles parameters
    n_recycles: int = 5,  # Paper: 5
    # Cropping parameters
    crop_size: int = 256,  # Paper: 256
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_probability: float = 0.5,
    crop_contiguous_probability: float = 0.5,
    # Filtering parameters
    unresolved_ligand_atom_limit: int | float | None = 0.1,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Atomization parameters
    res_names_to_atomize: list[str] | None = None,
    # MSA parameters
    max_msa_sequences: int = 10_000,  # Paper: 10_000
    dense_msa: bool = True,
    n_msa_cluster_representatives: int = 256,  # Paper model: 256
    msa_n_extra_rows: int = 1024,  # Paper mode: 1024
    msa_mask_probability: float = 0.15,
    msa_mask_behavior_probs: dict[str, float] = {
        "replace_with_random_aa": 0.1,
        "replace_with_msa_profile": 0.1,
        "do_not_replace": 0.1,
    },
    order_independent_atom_frame_prioritization: bool = True,
    polymer_token_indices: torch.Tensor = torch.arange(32),  # noqa: B008
    # Template parameters
    n_template: int = 5,
    pick_top_templates: bool = False,
    template_max_seq_similarity: float = 60.0,
    template_min_seq_similarity: float = 10.0,
    template_min_length: int = 10,
    template_lookup_path: PathLike | None = None,
    template_base_dir: PathLike | None = None,
    # Symmetry resolution parameters
    max_automorphs: int = 1_000,
    max_isomorphs: int = 1_000,
    # Miscellaneous parameters
    use_negative_interface_examples: bool = False,
    unclamp_loss_probability: float = 0.1,
    black_hole_init: bool = True,
    black_hole_init_noise_scale: float = 5.0,  # Angstroms (Paper: 5.0)
    # Cache params:
    msa_cache_dir: PathLike | str | None = None,
    assert_rf2aa_assumptions: bool = True,
    convert_feats_to_rf2aa_input_tuple: bool = True,
    # Inference parameters
    is_inference: bool = False,
) -> Compose:
    """
    Creates a transformation pipeline for the RF2AA model, applying a series of transformations to the input data.

    Args:
        - protein_msa_dirs (list[dict]): The directories containing the protein MSAs and their associated file types,
            as a list of dictionaries. If multiple directories are provided, we will search all of them. Note that:
            (a) the directory structure must be flat (i.e., no subdirectories), (b) the files must be named using the
            SHA-256 hash of the sequence (see `hash_sequence` in `utils/misc`), and (c) order matters - we will search the
            directories in the order they are provided, and return the first match
        - rna_msa_dirs (list[dict]): The directories containing the RNA MSAs and their associated file types, as a list
            of dictionaries. See `protein_msa_dirs` for directory structure details.
        - n_recycles (int, optional): Number of recycles for the MSA featurization. Defaults to 5.
        - crop_size (int, optional): Size of the crop for spatial and contiguous cropping (in number of tokens).
            Defaults to 384.
        - crop_center_cutoff_distance (float, optional): Cutoff distance for the center of the crop (in Angstroms).
            Defaults to 15.0.
        - crop_spatial_probability (float, optional): Probability of performing spatial cropping. Defaults to 0.5.
        - crop_contiguous_probability (float, optional): Probability of performing contiguous cropping. Defaults to 0.5.
        - unresolved_ligand_atom_limit (int | float, optional): Limit for above which a ligand is considered unresolved.
            many unresolved atoms has its atoms removed. If None, all atoms are kept, if between 0 and 1, the number of
            atoms is capped at that percentage of the crop size. If an integer >= 1, the number of unresolved atoms is
            capped at that number. Defaults to 0.1.
        - res_names_to_atomize (list[str], optional): List of residue names to *always* atomize. Note that RF2AA already
            atomizes all residues that are not in the encoding (i.e. that are not standard AA, RNA, DNA or special masks).
            Therefore only specify this if you want to deterministically atomize certain standard tokens. Defaults to None.
        - max_msa_sequences (int, optional): Maximum number of MSA sequences to load. Defaults to 10,000.
        - dense_msa (bool, optional): Whether to use dense MSA pairing. Defaults to True.
        - n_msa_cluster_representatives (int, optional): Number of MSA cluster representatives to select. Defaults to 100.
        - msa_n_extra_rows (int, optional): Number of extra rows for MSA. Defaults to 100.
        - msa_mask_probability (float, optional): Probability of masking MSA sequences according to `msa_mask_behavior_probs`.
            Defaults to 0.15.
        - msa_mask_behavior_probs (dict[str, float], optional): Probabilities for different MSA mask behaviors.
            Defaults to {"replace_with_random_aa": 0.1, "replace_with_msa_profile": 0.1, "do_not_replace": 0.1},
            which is the BERT style masking.
        - order_independent_atom_frame_prioritization (bool, optional): Whether to prioritize order-independent atom frames.
            Defaults to True.
        - n_template (int, optional): Number of templates to use. Defaults to 5.
        - pick_top_templates (bool, optional): Whether to pick the top templates if there are more than `n_template`. If
            False, the templates are selected randomly among all templates. Defaults to False.
        - template_max_seq_similarity (float, optional): Maximum sequence similarity cutoff for templates.
            Defaults to 60.0.
        - template_min_seq_similarity (float, optional): Minimum sequence similarity cutoff for templates.
            Defaults to 10.0.
        - template_min_length (int, optional): Minimum length cutoff for templates. Defaults to 10.
        - max_automorphs (int, optional): Maximum number of automorphs after which to cap small molecule ligand
            symmetry resolution. Defaults to 1,000.
        - max_isomorphs (int, optional): Maximum number of polymer isomorphs after which to cap symmetry resolution.
            Defaults to 1,000.
        - use_negative_interface_examples (bool, optional): Whether to use negative interface examples. Defaults to False.
        - unclamp_loss_probability (float, optional): Probability of unclamping the loss during training. Defaults to 0.1.
        - black_hole_init (bool, optional): Whether to use black hole initialization. Defaults to True.
        - black_hole_init_noise_scale (float, optional): Noise scale for black hole initialization. Defaults to 5.0.
        - msa_cache_dir (PathLike | str | None, optional): Directory to cache the MSAs. Defaults to None.
        - assert_rf2aa_assumptions (bool, optional): Whether to assert the RF2AA assumptions that need to be true
            to guarantee a successful forward & backward pass. Defaults to True.
        - convert_feats_to_rf2aa_input_tuple (bool, optional): Whether to convert the features to the RF2AAInputs format.
            Defaults to True.


    For more details on the parameters, see the RF2AA paper and the documentation for the respective Transforms.

    Returns:
        Compose: A composed transformation pipeline.
    """
    if crop_contiguous_probability > 0 or crop_spatial_probability > 0:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    if unresolved_ligand_atom_limit is None:
        unresolved_ligand_atom_limit = 1_000_000
    elif unresolved_ligand_atom_limit < 1:
        unresolved_ligand_atom_limit = np.ceil(crop_size * unresolved_ligand_atom_limit)

    encoding = RF2AA_ATOM36_ENCODING

    transforms = [
        # ============================================
        # 1. Prepare the structure
        # ============================================
        AddData({"is_inference": is_inference}),
        # ...remove hydrogens for efficiency
        RemoveHydrogens(),  # * (already cached from the parser)
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),  # Filter to non-clashing PN units
        RemoveTerminalOxygen(),  # RF2AA does not encode terminal oxygen for AA residues.
        RemoveUnresolvedPNUnits(),  # Remove PN units that are unresolved early (and also after cropping)
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),  # Remove polymers with too few resolved residues
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # ...remove unsupported chain types
        RemoveUnsupportedChainTypes(),  # e.g., DNA_RNA_HYBRID, POLYPEPTIDE_D, etc.
        # RaiseIfTooManyAtoms(max_atoms=max_allowed_num_atoms),
        HandleUndesiredResTokens(undesired_res_names),  # e.g., non-standard residues
        # ...filtering
        # RemoveUnresolvedLigandAtomsIfTooMany(
        #     unresolved_ligand_atom_limit=unresolved_ligand_atom_limit
        # ),  # Crop size * 10%
        # ...add an annotation that is a unique atom ID across the entire structure, and won't change as we crop or reorder the AtomArray
        AddGlobalAtomIdAnnotation(),
        # ...add additional annotations that we'll use later
        AddProteinTerminiAnnotation(),  # e.g., N-terminus, C-terminus
        AddWithinPolyResIdxAnnotation(),  # add annotation relevant for matching MSA and template info
        # ============================================
        # 2. Perform relevant atomizations to arrive at final tokens
        # ============================================
        # ...sample residues to atomize (in RF2AA, with some probability, we atomize protein residues randomly)
        # TODO: SampleResiduesToAtomize
        # ...handle covalent modifications by atomizing and attaching the bonded residue to the non-polymer
        FlagAndReassignCovalentModifications(),
        # ...flag non-polymers for atomization (in case there are polymer tokens outside of a polymer)
        FlagNonPolymersForAtomization(),
        # ...atomize
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_atomize=res_names_to_atomize,
            res_names_to_ignore=encoding.tokens,
            move_atomized_part_to_end=True,
        ),
        # ... sort poly then non-poly
        SortLikeRF2AA(),
        # ... add global and token IDs
        AddGlobalTokenIdAnnotation(),
        # ============================================
        # 3. Extract openbabel molecules for atomized residues and ligands
        # ============================================
        AddOpenBabelMoleculesForAtomizedMolecules(),
        # ... get chiral centers from openbabel molecules
        GetChiralCentersFromOpenBabel(),
    ]

    # Crop
    # ...crop around our query pn_unit(s) early, since we don't need the full structure moving forward
    cropping_transform = RandomRoute(
        transforms=[
            CropContiguousLikeAF3(
                crop_size=crop_size,
                keep_uncropped_atom_array=True,
            ),
            CropSpatialLikeAF3(
                crop_size=crop_size,
                crop_center_cutoff_distance=crop_center_cutoff_distance,
                keep_uncropped_atom_array=True,
            ),
        ],
        probs=[crop_contiguous_probability, crop_spatial_probability],
    )
    transforms.append(
        ConditionalRoute(
            condition_func=_is_inference,
            transform_map={
                True: Identity(),
                False: cropping_transform,
                # Default to Identity during inference (`is_inference == True`)
            },
        )
    )

    transforms += [
        AddPostCropMoleculeEntityToFreeFloatingLigands(),
        # ============================================
        # 4. Encode the structure
        # ============================================
        # ...encode the AtomArray (note that we've already atomized)
        EncodeAtomArray(encoding),
        # ============================================
        # 5. Load and pair MSAs
        # ============================================
        LoadPolymerMSAs(
            protein_msa_dirs=protein_msa_dirs,
            rna_msa_dirs=rna_msa_dirs,
            max_msa_sequences=max_msa_sequences,  # maximum number of sequences to load (we later subsample further)
            msa_cache_dir=Path(msa_cache_dir) if exists(msa_cache_dir) else None,
        ),
        PairAndMergePolymerMSAs(dense=dense_msa),
        EncodeMSA(encoding=encoding, token_to_use_for_gap=encoding.token_to_idx["UNK"]),
        FillFullMSAFromEncoded(pad_token=encoding.token_to_idx["UNK"]),
        # ============================================
        # 5. Load and featurize templates (proteins only)
        # ============================================
        AddRFTemplates(
            max_n_template=n_template,
            pick_top=pick_top_templates,
            max_seq_similarity=template_max_seq_similarity,
            min_seq_similarity=template_min_seq_similarity,
            min_template_length=template_min_length,
            template_lookup_path=template_lookup_path,
            template_base_dir=template_base_dir,
        ),
        # ============================================
        # 6. Add misc. features (chirals, bond features, etc.)
        # ============================================
        # ...chirals
        AddRF2AAChiralFeatures(),
        # ...bonds
        AddTokenBondAdjacency(),
        AddRF2AABondFeaturesMatrix(),
        AddRF2AATraversalDistanceMatrix(),
        # ...atom frames
        AddAtomFrames(order_independent_atom_frame_prioritization=order_independent_atom_frame_prioritization),
        # ============================================
        # 7. Convert to torch and featurize
        # ============================================
        ConvertToTorch(
            keys=[
                "polymer_msas_by_chain_id",
                "encoded",
                "full_msa_details",
                "rf2aa_bond_features_matrix",
                "rf2aa_traversal_distance_matrix",
                "rf2aa_atom_frames",
            ]
        ),
        FeaturizeMSALikeRF2AA(
            n_recycles=n_recycles,
            n_msa_cluster_representatives=n_msa_cluster_representatives,  # Paper model: 256
            n_extra_rows=msa_n_extra_rows,  # Paper mode: 1024
            mask_behavior_probs=msa_mask_behavior_probs,
            mask_probability=msa_mask_probability,
            encoding=encoding,
            polymer_token_indices=polymer_token_indices,
        ),
        FeaturizeTemplatesLikeRF2AA(
            n_template=n_template,
            mask_token_idx=encoding.token_to_idx["<M>"],
            encoding=encoding,
            init_coords=RF2AATemplate.RF2AA_INIT_TEMPLATE_COORDINATES,
        ),
    ]

    transforms += [
        # ============================================
        # 8. Create symmetry copies (isomorphic chain swaps for polys, automorphisms for small molecules)
        # ============================================
        ConditionalRoute(
            condition_func=_is_training,
            transform_map={
                True: CreateSymmetryCopyAxisLikeRF2AA(encoding=encoding, max_automorphs=1, max_isomorphisms=1),
                False: CreateSymmetryCopyAxisLikeRF2AA(
                    encoding=encoding, max_automorphs=max_automorphs, max_isomorphisms=max_isomorphs
                ),
            },
        )
    ]

    transforms += [
        # ============================================
        # 9. Aggregate features into final format for RF2AA and remove unused features
        # ============================================
        AggregateFeaturesLikeRF2AA(
            encoding=encoding,
            use_negative_interface_examples=use_negative_interface_examples,
            unclamp_loss_probability=unclamp_loss_probability,
            black_hole_init=black_hole_init,
            black_hole_init_noise_scale=black_hole_init_noise_scale,
        ),
    ]

    if assert_rf2aa_assumptions:
        transforms.append(AssertRF2AAAssumptions())

    # Convert the features to the RF2AAInputs format
    if convert_feats_to_rf2aa_input_tuple:
        transforms.append(ApplyFunction(_convert_feats_to_rf2aa_input_tuple))

    transforms.append(SubsetToKeys(["example_id", "feats", "ground_truth"]))

    return Compose(transforms, track_rng_state=True)
