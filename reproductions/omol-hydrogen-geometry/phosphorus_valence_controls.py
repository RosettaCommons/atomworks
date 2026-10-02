"""Show why inferred P-H counts cannot select a placement geometry by themselves.

Run with the public AtomWorks release e0f2e4da and RDKit installed:
    python phosphorus_valence_controls.py

These synthetic molecular graphs are deliberately different chemical states.
RDKit and Dimorphite are not expected to guess which graph the user intended.
"""

import json

from rdkit import Chem, rdBase

from atomworks.experimental.protonation.dimorphite import protonate_at_ph


def main():
    cases = [
        ("four C bonds, neutral P", "CP(C)(C)C", 1),
        ("four C bonds, positive P", "C[P+](C)(C)C", 0),
        ("three C bonds and P=O", "CP(=O)(C)C", 0),
        ("three C bonds and P-OH", "CP(O)(C)C", 1),
    ]
    results = []
    for label, smiles, expected_h in cases:
        molecule = Chem.MolFromSmiles(smiles)
        phosphorus = next(a for a in molecule.GetAtoms() if a.GetAtomicNum() == 15)
        state = protonate_at_ph(molecule, 7.4)
        assert state is not None
        titrated = state.GetAtomWithIdx(phosphorus.GetIdx())
        assert phosphorus.GetTotalNumHs() == titrated.GetTotalNumHs() == expected_h
        assert phosphorus.GetFormalCharge() == titrated.GetFormalCharge()
        results.append(
            {
                "case": label,
                "input_smiles": smiles,
                "heavy_neighbors": phosphorus.GetDegree(),
                "formal_charge": phosphorus.GetFormalCharge(),
                "rdkit_h_count": phosphorus.GetTotalNumHs(),
                "dimorphite_h_count": titrated.GetTotalNumHs(),
                "output_smiles": Chem.MolToSmiles(state),
            }
        )
    print(json.dumps({"rdkit_version": rdBase.rdkitVersion, "controls": results}, indent=2))


if __name__ == "__main__":
    main()
