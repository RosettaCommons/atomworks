# OMol phosphorus hydrogen-placement reproduction

Tested with public AtomWorks commit `e0f2e4da0aef2cf6c81a2c506eece3bc4204b75d`.

```bash
python -m pip install 'atomworks @ git+https://github.com/RosettaCommons/atomworks.git@e0f2e4da0aef2cf6c81a2c506eece3bc4204b75d'
python reproduce.py
python phosphorus_valence_controls.py
```

`reproduce.py` intentionally exits with the observed `ValueError`. The two preceding assertions pass: all 38 supplied atoms and their coordinates survive when the P-H is present, and removing only that H while explicitly declaring its required count correctly requests one neutral P-H. The failure therefore isolates placement rather than asking the software to guess protonation.

The fixture is the complete 38-atom phosphorane component A:1 from locally prepared OMol CIF `0594f821.cif`. Four disconnected solvent components are omitted. Every retained coordinate, atom name, and bond is unchanged. Original OMol source identifier:

`omol/electrolytes/solvated_090624/phosphorane_mol1884_solv5_0_1/step4/orca.tar.zst`

Original source LMDB identifier: `27347822`. This is distinct from local prepared-CIF shard/key indexing. Full original prepared CIF SHA256: `2f4197bb1bef01b3f717aafe32bc67855555c6885d5b0417077cefa0d90e1d51`.

The original prepared CIF and later interaction-annotated CIF have identical atom names, coordinates, and bond graphs. The formal charges in this fixture come from the latter component annotation. The original OMol quantum-chemistry coordinates do **not** constitute an authoritative formal-charge or bond-order assignment: the CIF graph is a derived preprocessing representation. In this example the original explicit H, five single bonds, source phosphorane identity and near trigonal-bipyramidal geometry all support the declared P-H state.

Data attribution: [Open Molecules 2025](https://huggingface.co/facebook/OMol25), [dataset documentation and CC BY 4.0 license](https://facebookresearch.github.io/fairchem/omol25/), Levine et al., [The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models](https://arxiv.org/abs/2505.08762). This fixture is a reduced, locally prepared representation of that dataset, as described above.

`phosphorus_valence_controls.py` contains synthetic negative controls. It demonstrates why four heavy neighbors alone cannot establish whether P needs H: a positive formal charge or a P=O bond changes the hydrogen count. The four inputs deliberately represent different chemical states and are not asserted to be RDKit or Dimorphite bugs.

## Separate ambiguity control

Run `python check_ambiguous_phosphorus.py` to inspect a different real prepared OMol complex (original source index `10552741`). Its 100-atom fixture is complete. P1 has three N neighbors and one O neighbor, all represented as single bonds, formal charge zero, and no supplied P-H. The observed P-O distance is 1.53031 Å and all neighbor angles are 103.9–113.1°, compatible with a tetrahedral phosphoryl environment. The current assignment requests P-H for this graph and placement rejects it. Changing only P-O to double, or declaring zero P-H without changing the graph, suppresses that request. Neither is an automatic repair policy: the derived graph and original chemistry must be resolved before adding a fifth ligand. This is a data-annotation diagnostic and a safeguard for extending geometry support, not a claim that all four-coordinate P must lack H.
