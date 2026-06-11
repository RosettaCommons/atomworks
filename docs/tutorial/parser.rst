Parser
======

The parser is the core entry point for converting structural and sequence files (mmCIF, PDB, FASTA, SMILES, etc.) into Biotite's AtomArray API. It supports extensive options for annotation, filtering, and caching.

Example Usage
-------------

.. code-block:: python

   from atomworks.io.parser import parse
   result = parse(filename="/databases/rcsb/cif/ne/3nez.cif.gz")
   print(result["chain_info"])

What's in the Dictionary Returned by the Parser?
------------------------------------------------
- **chain_info**: Mapping of chain IDs to sequence, type, and metadata
- **ligand_info**: Information about ligands in the structure, if there are any
- **asym_unit**: AtomArrayStack of the asymmetric unit, contains similar information to what is contained in a PDB file
- **assemblies**: Mapping of assembly IDs to AtomArrayStacks
- **metadata**: Structure-level metadata, such as crystallization details, the method used to determine the structure, resolution, etc.
- **extra_info**: Internal-use information for caching and compatibility

Parsing Arguments
-----------------

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - filename
     - PathLike / io.StringIO / io.BytesIO
     - —
     - Path to the structural file. Supports .cif, .cif.gz, .pdb, etc.
   * - add_missing_atoms
     - bool
     - True
     - Add missing atoms to the structure. Useful for unresolved residues. Also adds intra- and inter-residue bonds.
   * - add_id_and_entity_annotations
     - bool
     - True
     - Add id and entity annotations at chain, pn-unit, and molecule level to the AtomArray.
   * - add_bond_types_from_struct_conn
     - list[str]
     - ["covale"]
     - List of bond types to add from struct_conn. Default is only covalent bonds.
   * - remove_ccds
     - list[str] or None
     - crystallization aids
     - CCD codes to remove from the structure.
   * - remove_waters
     - bool
     - True
     - Remove water molecules from the structure.
   * - fix_ligands_at_symmetry_centers
     - bool
     - True
     - Patch non-polymer residues at symmetry centers that clash with themselves.
   * - fix_arginines
     - bool
     - True
     - Fix arginine naming ambiguity.
   * - fix_formal_charges
     - bool
     - True
     - Fix formal charges on atoms involved in inter-residue bonds.
   * - fix_bond_types
     - bool
     - True
     - Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.
   * - convert_mse_to_met
     - bool
     - False
     - Convert selenomethionine (MSE) to methionine (MET).
   * - hydrogen_policy
     - "keep" / "remove" / "infer"
     - "keep"
     - Whether to keep, remove, or infer hydrogens.
   * - model
     - int or None
     - None
     - Model number for NMR entries.
   * - build_assembly
     - "first" / "all" / list[str] / tuple[str] / None
     - "all"
     - Which assembly to build: None, "first", "all", or specific IDs.
   * - extra_fields
     - list[str] / "all" / None
     - None
     - Extra fields to include in the AtomArrayStack.


Wrapper Arguments
-----------------

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - file_type
     - "cif" / "pdb" / "mmjson" / None
     - None
     - File type to parse. If None, inferred from filename extension.
   * - ccd_mirror_path
     - PathLike / None
     - None
     - Path to local mirror of the Chemical Component Dictionary. (Recommended)
   * - load_from_cache
     - bool
     - False
     - Load pre-compiled results from cache.
   * - save_to_cache
     - bool
     - False
     - Save parsed results to cache for faster future retrieval.
   * - cache_dir
     - PathLike / None
     - None
     - Directory for cached results. Required if caching is enabled.