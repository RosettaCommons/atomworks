Atom Selection Syntax
=====================

atomworks provides two complementary string languages for selecting atoms, both living in
:py:mod:`atomworks.io.utils.query` and both usable directly on any ``AtomArray`` /
``AtomArrayStack`` via the monkey-patched ``.query()``, ``.mask()`` and ``.idxs()`` methods:

- **The query language** — pandas-like boolean expressions over annotations and coordinates,
  e.g. ``"(chain_id == 'A') & (res_id < 50)"``, ``"res_name in ['ALA','GLY']"``, ``"z > 0"``,
  ``"has_bonds()"``. Powerful and composable.
- **The path-selection DSL** — a terse, path-like shorthand over five ordered fields,
  e.g. ``"A/ALA/1/CA"``. Compact for the common "chain / residue / atom" case.

The two are bridged by the ``sel('...')`` function, so a path selection can appear **inside** a
query expression and combine with any predicate — see :ref:`sel-in-query` below.

.. contents::
   :local:
   :depth: 1


The path grammar
----------------

A path selection has five ``/``-separated fields, always in this order (trailing fields may be
omitted and default to ``*``)::

    CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME / TRANSFORMATION_ID

Each field is one of:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Form
     - Example
     - Meaning
   * - scalar
     - ``A``, ``ALA``, ``CA``
     - Exactly that value.
   * - wildcard
     - ``*``
     - Any value (also the default for omitted trailing fields).
   * - list ``[...]``
     - ``[A,B]``, ``[CA,CB]``
     - Any of the listed values (a within-field OR).
   * - range (``res_id`` only)
     - ``[5-10]``, ``[1-5, 9]``
     - Inclusive integer range(s); must be inside brackets.

Multiple whole selections separated by top-level commas (or given as a ``list[str]``) are
**unioned**. Commas *inside* ``[...]`` belong to the list, not the union::

    "A/*/[5-10], B/*/[3-8]"      # residues 5-10 of chain A OR residues 3-8 of chain B

Lists across fields expand as a Cartesian product, then union. For example
``"[A,B]/ALA/1/[CA,CB]"`` is the union of the four selections
``A/ALA/1/CA``, ``A/ALA/1/CB``, ``B/ALA/1/CA``, ``B/ALA/1/CB``.

.. note::
   **Ranges must be bracketed.** ``A/*/[5-10]`` is a range; a bare ``A/*/5-10`` is an error.
   This keeps a single, uniform multi-value mechanism (everything multi-valued lives in ``[...]``)
   and lets awkward ids be listed explicitly, e.g. negative residue ids ``A/*/[-5, -3, 0]``.

.. note::
   ``[...]`` lists are multi-valued and therefore only valid through the multi-selection entry
   points — :py:meth:`~atomworks.io.utils.query.AtomSelectionStack.from_query` and the query
   ``sel('...')`` function. The single-valued :py:meth:`~atomworks.io.utils.query.AtomSelection.from_selection_str`
   rejects brackets.


Interactive gallery
-------------------

Pick a selection syntax with the buttons below to see what it matches on **1YCR** — the MDM2
domain (chain A) bound to the p53 transactivation peptide (chain B). The viewer highlights exactly
the atoms that ``atom_array.mask(<string>)`` returns (the mask is baked into the model as atom
serials), so what you see is what the selection selects — including the coordinate-predicate option
that a viewer's own selection language could not express. **Drag to rotate, scroll to zoom, and
click any highlighted atom to label it.**

.. raw:: html
   :file: ../../_static/selection/interactive.html

.. note::
   The viewers load `3Dmol.js <https://github.com/3dmol/3Dmol.js>`_ from a CDN, so an internet connection is
   needed to see them; without JavaScript the page still renders the surrounding text.


.. _sel-in-query:

Combining path selection with the query language
------------------------------------------------

Inside a query expression, ``sel('<path>')`` evaluates a path selection to a boolean mask, so it
composes with every query operator (``&``, ``|``, ``~``, comparisons, ``in``, ``has_bonds()`` …):

.. code-block:: python

    # Cα atoms of chain A that are also above a plane
    atom_array.query("sel('A/*/*/CA') & (z > 0)")

    # the p53 anchor residues, excluding anything with NaN coordinates
    atom_array.mask("sel('B/[PHE,TRP,LEU]') & ~has_nan_coord()")

.. note::
   ``sel``'s argument must be a **quoted string literal**: the query is ordinary Python-expression
   syntax parsed by ``ast``, so an unquoted ``A/*/*/CA`` would parse as arithmetic over undefined
   names. Only one nesting level ever arises — the outer query string and the inner ``sel`` argument
   — so ``"..."`` outside and ``'...'`` inside is enough (standard Python quote alternation; ``ast``
   handles it). For a pure path selection with no predicate, skip the query layer and call
   :py:meth:`~atomworks.io.utils.query.AtomSelectionStack.from_query` directly — no nested quotes.

The query side is not limited to the five path fields — it exposes **every annotation on the
array** as a variable. ``ensure_annotations`` (from ``atomworks.io.utils.annotator``) computes a
large registry of derived annotations on demand, so rich selections become one-liners that still
compose with ``sel('...')``:

.. code-block:: python

    from atomworks.io.utils.annotator import ensure_annotations

    # protein backbone atoms
    ensure_annotations(atom_array, "is_protein_backbone")
    atom_array.query("is_protein_backbone")

    # non-canonical amino acids: protein atoms that are not a standard (or unknown) residue
    ensure_annotations(atom_array, "is_protein", "is_standard_or_unknown_aa")
    atom_array.query("is_protein & ~is_standard_or_unknown_aa")

Any boolean or scalar annotation works — e.g. ``is_ligand``, ``is_metal``, ``is_nucleic_acid``,
``is_protein_sidechain``, ``atomic_number``, ``chem_comp_type``, or a custom annotation you set
yourself.

Unlike a standalone selection, ``sel('...')`` is **non-raising**: if it matches nothing it
contributes an all-``False`` mask instead of raising, so it degrades gracefully inside a larger
boolean expression.


Programmatic API
----------------

.. code-block:: python

    from atomworks.io.utils.query import AtomSelection, AtomSelectionStack

    # Single, one-valued selection
    AtomSelection.from_selection_str("A/ALA/1/CA").get_mask(atom_array)

    # Multi-valued selection (lists, ranges, unions) -> a union of selections
    AtomSelectionStack.from_query("[A,B]/*/[25-35]/CA").get_mask(atom_array)

See :doc:`query` for the full API reference.
