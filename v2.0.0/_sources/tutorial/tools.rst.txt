Tools
=====

The tools module provides utilities for working with sequence and structure data, such as FASTA conversion, inference, and RDKit-based operations.

Example: FASTA Conversion
-------------------------
.. code-block:: python

   from atomworks.io.tools.fasta import structure_to_fasta
   fasta = structure_to_fasta("/path/to/structure.cif")
   print(fasta)