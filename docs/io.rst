IO
=====

.. csv-table::
   :header: "Function", "Responsibility"

   ``atom_array_from_file``, Return atoms with bonds and annotations using minimal parsing
   ``prepare_atom_array``, Apply configured atom cleanup/completion and bond/annotation assignment to an existing array
   ``parse``, Read and prepare a file with metadata and assemblies
   ``atom_array_to_file``, Write CIF/BCIF to a path or buffer or return content

Use ``parse_atom_array`` when existing atoms need a result dictionary instead of an array.
Use ``atom_array_to_file`` to save atoms with bonds and annotations.
``atom_array_from_file`` wraps ``parse(config="minimal")`` and returns just the atoms.
It selects the first conformer and uses the parser's identifier conventions.
Use CIF/BCIF for StandardAnnotations.

Pass ``CIFWriteConfig`` only when you need custom export settings.

.. autofunction:: atomworks.io.atom_array_from_file

.. autofunction:: atomworks.io.atom_array_to_file

.. toctree::
   :maxdepth: 2

   io/parser
   io/tools
   io/transforms
   io/utils
