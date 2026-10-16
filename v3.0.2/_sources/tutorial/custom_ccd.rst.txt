Custom CCDs and CIF round trips
========================================

Register your ligand before writing. Here ``ligand`` is one complete LG1 residue
with correct atom names, elements, charges, bonds, and a nonempty chain ID;
``atoms`` is the structure containing it (or just ``ligand``).

.. code-block:: python

   from atomworks.io import parse
   from atomworks.io.config import ParseConfig
   from atomworks.io.utils.ccd import custom_ccd_residues
   from atomworks.io.utils.io_utils import to_cif_file

   with custom_ccd_residues({"LG1": ligand}):
       to_cif_file(atoms, "structure.cif")

   config = ParseConfig.from_preset(
       "minimal", build_assembly=None, cif_ccd_on_mismatch="ignore",
       return_atom_array_plus=True,
   )
   restored = parse("structure.cif", config=config)["asym_unit"][0]

``register_custom_ccd_entry("LG1", ligand)`` also works, but persists globally.
The writer generates all three ``chem_comp*`` tables. Missing
``chem_comp_type`` is inferred from atom names; override it if the heuristic is
wrong. An existing annotation is used directly as ``chem_comp.type``.

``"minimal"`` avoids atom rebuilding/filtering; ``"ignore"`` permits intentional
CCD-code collisions. The file is self-contained; no registration is needed to
read it. ``return_atom_array_plus=True`` retains its definitions for rewriting.
Coordinates are rounded to 0.001 Å; this is not exact object serialization.
