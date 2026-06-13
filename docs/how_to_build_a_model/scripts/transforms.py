from atomworks.ml.transforms.base import Transform
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER

class CropToPocket(Transform):
    requires_previous_transforms = ["RemoveHydrogens", "RemoveUnresolvedAtoms"]
    def __init__(self, radius: float = 10.0) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, data: dict) -> dict:
        data["atom_array"] = crop_to_pocket(
            data["atom_array"],
            query_pn_unit_iids=data["query_pn_unit_iids"],
            chain_info=data["chain_info"],
            radius=self.radius,
        )
        return data
    
    def check_input(self, data: dict) -> None:
        assert "atom_array" in data, "Missing atom_array"
        assert "query_pn_unit_iids" in data, "Missing query_pn_unit_iids"
        assert "chain_info" in data, "Missing chain_info"

class FeaturizeForDocking(Transform):
    requires_previous_transforms = ["CropToPocket"]

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["is_ligand"])

    def forward(self, data: dict) -> dict:
        features = featurize_for_docking(data["atom_array"])
        data.update(features)
        return data
    
def crop_to_pocket(
    atom_array: AtomArray,
    query_pn_unit_iids: list,
    chain_info: dict,
    radius: float = 10.0,
) -> AtomArray:
    atom_array = atom_array.copy()

    iid_a, iid_b = query_pn_unit_iids
    chain_a = iid_a.split("_")[0]
    chain_b = iid_b.split("_")[0]

    a_is_polymer = chain_info.get(chain_a, {}).get("is_polymer", True)
    b_is_polymer = chain_info.get(chain_b, {}).get("is_polymer", True)

    if not a_is_polymer:
        ligand_iid, protein_iid = iid_a, iid_b
    elif not b_is_polymer:
        ligand_iid, protein_iid = iid_b, iid_a
    else:
        mask_a = atom_array.pn_unit_iid == iid_a
        mask_b = atom_array.pn_unit_iid == iid_b
        if mask_a.sum() <= mask_b.sum():
            ligand_iid, protein_iid = iid_a, iid_b
        else:
            ligand_iid, protein_iid = iid_b, iid_a

    ligand_mask = atom_array.pn_unit_iid == ligand_iid
    protein_mask = atom_array.pn_unit_iid == protein_iid

    ligand_coords = atom_array.coord[ligand_mask]
    protein_coords = atom_array.coord[protein_mask]

    if len(ligand_coords) == 0:
        raise ValueError(f"Ligand {ligand_iid} has no atoms")
    if len(protein_coords) == 0:
        raise ValueError(f"Protein {protein_iid} has no atoms")

    tree = cKDTree(protein_coords)
    neighbor_indices = tree.query_ball_point(ligand_coords, r=radius)

    total_neighbors = sum(len(n) for n in neighbor_indices)
    if total_neighbors == 0:
        raise ValueError(f"No protein atoms found within {radius}A of ligand {ligand_iid}")

    pocket_local_indices = np.unique(np.concatenate(neighbor_indices).astype(int))
    protein_global_indices = np.where(protein_mask)[0]
    pocket_global_indices = protein_global_indices[pocket_local_indices]
    ligand_global_indices = np.where(ligand_mask)[0]

    keep = np.sort(np.concatenate([pocket_global_indices, ligand_global_indices]))
    is_ligand = np.isin(keep, ligand_global_indices)
    cropped = atom_array[keep]

    return cropped

def featurize_for_docking(atom_array: AtomArray) -> dict:
    is_ligand = atom_array.is_ligand.astype(bool)
    target_coords = atom_array.coord.astype(np.float32)

    input_coords = target_coords.copy()
    input_coords[is_ligand] = 0.0

    atomic_numbers = np.array(
        [ELEMENT_NAME_TO_ATOMIC_NUMBER.get(e.upper(), 0) for e in atom_array.element],
        dtype=np.int64,
    )

    bonds = atom_array.bonds.as_array()
    edge_index = bonds[:, :2].T.astype(np.int64)

    return {
        "atomic_numbers": atomic_numbers,
        "is_ligand": is_ligand,
        "target_coords": target_coords,
        "edge_index": edge_index,
        "input_coords": input_coords,
    }

