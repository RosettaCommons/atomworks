Transforms
==========

The transforms module provides functions for manipulating and transforming AtomArrays and related data structures, such as coordinate transformations, category operations, and more.

Example Usage
-------------
.. code-block:: python

   from atomworks.io.transforms.atom_array import rotate_structure
   rotated = rotate_structure(atom_array, angle=90, axis="z")
