Cookbook
========

The atomworks cookbook is intended to be a quick reference for various common tasks. Note the code snippets may not be complete as-written, and may require imports or other prepratory steps.

As the AtomArray object comes from biotite, the `biotite documentation <https://www.biotite-python.org/latest/tutorial/structure/index.html>`__ and `api reference <https://www.biotite-python.org/latest/apidoc/biotite.structure.html>`__ contains additional resources for working with it.

Input
-----

Standard Local File Input
~~~~~~~~~~~~~~~~~~~~~~~~~

Accepts common structure input formats (mmCIF, PDB, mmjson and BCIF/BinaryCIF), both with filenames and file-like objects. CIF files can be gzip compressed. 

See :doc:`the Parser tutorial page <tutorial/parser>` for details on parameters and output. You can also find the API docs for the ``parse`` function :func:`here <atomworks.io.parser.parse>`. (Defaults for selected keyword arguments shown.)::

    result_dict = atomworks.io.parse(filename,
            add_missing_atoms=True,
            remove_waters=True,
            hydrogen_policy="keep",
            )

    # Extract relevant AtomArray:
    asym_unit = result_dict["asym_unit"][0]       # The full asymmetric unit
    assembly = result_dict["assemblies"]["1"][0]  # The Biological Assembly/biounit (possibly comprising multiple asym_units)

Load from Clean Files
~~~~~~~~~~~~~~~~~~~~

The :func:`~atomworks.io.utils.io_utils.load_any` function is suitable for previously processed structures (e.g. those which have already passed through an atomworks preparation pipeline)::

    atom_array = atomworks.io.utils.io_utils.load_any( filename )

Database Input
~~~~~~~~~~~~~~

The :func:`~atomworks.io.parser.parse` function can be used to download the given PDB id from the RCSB website::

    result_dict = atomworks.io.parse( atomworks.io.utils.testing.get_pdb_path_or_buffer( PDBID ) )

CCD/SDF/SMILES Input
~~~~~~~~~~~~~~~~~~~~

:func:`~atomworks.io.tools.inference.ccd_code_to_annotated_atom_array`, :func:`~atomworks.io.tools.inference.sdf_to_annotated_atom_array`, :func:`~atomworks.io.tools.inference.smiles_to_annotated_atom_array` can be used to convert CCD codes, SDF files, or SMILES strings into annotated AtomArrays, respectively.::

   atom_array = atomworks.io.tools.inference.ccd_code_to_annotated_atom_array( ccd_code, chain_id )
   atom_array = atomworks.io.tools.inference.sdf_to_annotated_atom_array( sdf_filename, chain_id )
   atom_array = atomworks.io.tools.inference.smiles_to_annotated_atom_array( smiles_string, chain_id )

Sequence Input
~~~~~~~~~~~~~~

The following is a simple procedure for how to create an annotated AtomArray from a sequence string::

   chain_type = atomworks.enums.ChainType.POLYPEPTIDE_L

   # If you have RF3-style sequence string with parenthesized CCD codes:
   ccd_list = atomworks.io.tools.fasta.one_letter_to_ccd_code( atomworks.io.tools.fasta.split_generalized_fasta_sequence( sequence_string ), chain_type=chain_type )

   # ccd_list is just a list with one CCD code per entry.

   atom_array = atomworks.io.tools.inference.sequence_to_annotated_atom_array( ccd_list, chain_id, chain_type=chain_type )
   # Caution! The atom array will not contain coordinates. 

API docs for these functions: 

- :attr:`~atomworks.enums.ChainType.POLYPEPTIDE_L`
- :func:`~atomworks.io.tools.fasta.one_letter_to_ccd_code`
- :func:`~atomworks.io.tools.fasta.split_generalized_fasta_sequence`
- :func:`~atomworks.io.tools.inference.ccd_code_to_annotated_atom_array`

Output
------

mmCIF output
~~~~~~~~~~~~

:func:`~atomworks.io.utils.io_utils.to_cif_file` supports `.cif`, `.cif.gz` and `.bcif` outputs::

    atomworks.io.utils.io_utils.to_cif_file( atom_array, filename )

Legacy PDB output
~~~~~~~~~~~~~~~~~

While the use of :func:`~atomworks.io.utils.io_utils.to_pdb_string` is possible for creating legacy PDB files, use of mmCIF instead is recommended::

    with open( filename ) as f:
        f.write( atomworks.io.utils.io_utils.to_pdb_string( atom_array ) )

SDF/SMILES output
~~~~~~~~~~~~~~~~~

This procedure may not work reliably for multi-residue AtomArrays (or too-small fractions of a single residue)::

    rdmol = atomworks.io.tools.rdkit.atom_array_to_rdkit( atom_array )

    smiles = rdkit.Chem.MolToSmiles( rdmol )
    
    with rdkit.Chem.SDWriter( sdf_filename ) as w:
        w.write( rdmol )

API docs for these functions:

- :func:`~atomworks.io.tools.rdkit.atom_array_to_rdkit`
- `MolToSmiles <https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles>`_
- `SDWriter <https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SDWriter>`_

FASTA output
~~~~~~~~~~~~

`biotite.to_sequence <https://www.biotite-python.org/latest/apidoc/biotite.structure.to_sequence.html#biotite.structure.to_sequence>`_ will raise an error if the atom_array contains non-polymeric residues::

    results = biotite.to_sequence( atom_array )

    seq_list = results[0] # The second entry in the returned tuple is chain_start_indices


Structure Manipulation
----------------------

Combining AtomArrays
~~~~~~~~~~~~~~~~~~~~

For example, to combine a receptor with a ligand::

    combined_atom_array = atom_array_1 + atom_array_2

Subsetting AtomArray
~~~~~~~~~~~~~~~~~~~~

AtomArray can be subsetted by indexing with a Boolean array, resulting in another AtomArray::

    present = atom_array[ atom_array.occupancy > 0 ]
    calpha  = atom_array[ atom_array.atom_name == "CA" ]
    chainA  = atom_array[ atom_array.chain_id == "A" ]
    gly     = atom_array[ atom_array.res_name == "GLY" ]
    polymer = atom_array[ atom_array.is_polymer ]
    not_bb  = atom_array[ ~atom_array.is_backbone_atom ]
    res_B34 = atom_array[ (atom_array.chain_id == "B") & (atom_array.res_id == 34) ] # Parenthesis are needed.

:mod:`~atomworks.io.transforms.atom_array` contains a number of helpful utility functions (primarily used in structure loading)::

    no_waters = atomworks.io.transforms.atom_array.remove_waters( atom_array )
    no_ccd    = atomworks.io.transforms.atom_array.remove_ccd_components( atom_array, ccd_code_list )
    no_hydro  = atomworks.io.transforms.atom_array.remove_hydrogens( atom_array )
    with_hydrogens = atomworks.io.transforms.atom_array.add_hydrogen_atom_positions( atom_array )


The ``biotite.structure`` module contains a number of `filter <https://www.biotite-python.org/latest/apidoc/biotite.structure.html#filters>`_ functions to help subset the AtomArray::

    sugars = atom_array[ biotite.structure.filter_carbohydrates(atom_array) ]

Changing annotations
~~~~~~~~~~~~~~~~~~~~

Note this naive manipulation doesn't update the ``_id/_iid/pn_unit/molecule/etc.`` identity correspondences::

    atom_array.chain_id[ atom_array.is_polymer ] = "A"
    atom_array.res_id[ atom_array.resid == 1004 ] = 4

Coordinate Manipulation
-----------------------

Simple Translation/Rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    translated = biotite.structure.translate( atom_array, (x,y,z) ) # Displacement can be any array-shaped
    rotated    = biotite.structure.rotate_about_axis( atom_array, axis, angle, center ) # Center can be omitted to rotate around origin

API docs for these functions:

- `biotite.structure.translate <https://www.biotite-python.org/latest/apidoc/biotite.structure.html#biotite.structure.translate>`_
- `biotite.structure.rotate_about_axis <https://www.biotite-python.org/latest/apidoc/biotite.structure.rotate_about_axis.html>`_

Alignment
~~~~~~~~~

If the AtomArrays have identical atom layouts::

    superimposed, transformation = biotite.structure.superimpose( fixed, mobile )
    
    superimposed2 = transformation.apply( mobile2 ) # Apply same transformation to different AtomArray 

If the structures aren't identical, the following uses sequence alignments to find pairings::
    
    superimposed, transformation, fixed_indices, mobile_indices = biotite.structure.superimpose_homologs( fixed, mobile )

If there's low sequence similarity, the following uses structural similarity to find pairings::

    superimposed, transformation, fixed_indices, mobile_indices = biotite.structure.superimpose_structural_homologs( fixed, mobile )

API docs for these functions:

- `biotite.structure.superimpose <https://www.biotite-python.org/latest/apidoc/biotite.structure.superimpose.html#biotite.structure.superimpose>`_
- `biotite.structure.superimpose_homologs <https://www.biotite-python.org/latest/apidoc/biotite.structure.superimpose_homologs.html#biotite.structure.superimpose_homologs>`_
- `biotite.structure.superimpose_structural_homologs <https://www.biotite-python.org/latest/apidoc/biotite.structure.superimpose_structural_homologs.html#biotite.structure.superimpose_structural_homologs>`_


Calculations
------------

Distances/Angles
~~~~~~~~~~~~~~~~

Geometry calculations take AtomArrays all of N entries (or one entry to be broadcasted), resulting in a numpy array of N distances::

    distances = biotite.structure.distance( atom_array_1, atom_array_2 ) # In Angstroms
    angles    = biotite.structure.angle( atom_array_1, atom_array_2, atom_array_3 ) # In radians
    dihedrals = biotite.structure.dihedral( atom_array_1, atom_array_2, atom_array_3, atom_array_4 ) # In radians

There's convenience functions for standard protein dihedrals::

    bb_dihedral = biotite.structure.dihedral_backbone( atom_array ) # tuple of (phi, psi, omega) array

API docs for these functions:

- `biotite.structure.distance <https://www.biotite-python.org/latest/apidoc/biotite.structure.distance.html#biotite.structure.distance>`_
- `biotite.structure.angle <https://www.biotite-python.org/latest/apidoc/biotite.structure.angle.html#biotite.structure.angle>`_
- `biotite.structure.dihedral <https://www.biotite-python.org/latest/apidoc/biotite.structure.dihedral.html#biotite.structure.dihedral>`_
- `biotite.structure.dihedral_backbone <https://www.biotite-python.org/latest/apidoc/biotite.structure.dihedral_backbone.html#biotite.structure.dihedral_backbone>`_

Structural Comparison
~~~~~~~~~~~~~~~~~~~~~

::

    rmsd = biotite.structure.rmsd( reference, subject ) # RMSD without superposition
    lddt = biotite.structure.lddt( reference, subject )
    # fixed_indices & mobile_indices can come from superimpose_structural_homologs()
    tm_score = biotite.structure.tm_score( reference, subject, fixed_indices, mobile_indices )

API docs for these functions:

- `biotite.structure.rmsd <https://www.biotite-python.org/latest/apidoc/biotite.structure.rmsd.html#biotite.structure.rmsd>`_
- `biotite.structure.lddt <https://www.biotite-python.org/latest/apidoc/biotite.structure.lddt.html#biotite.structure.lddt>`_
- `biotite.structure.tm_score <https://www.biotite-python.org/latest/apidoc/biotite.structure.tm_score.html#biotite.structure.tm_score>`_

Visualization
-------------

Viewer
~~~~~~

Interactive viewer (:func:`~atomworks.io.utils.visualize.view`) (for Notebooks)::

    atomworks.io.utils.visualize.view(atom_array)

Other
-----

If there are other common tasks you think are worth including here, please `open an issue <https://github.com/RosettaCommons/atomworks/issues>`_ on Github.

