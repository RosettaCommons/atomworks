Installation
============

Use Python 3.11 or newer.

.. code-block:: bash

   python -m pip install atomworks       # Structure IO without PyTorch
   python -m pip install "atomworks[ml]" # IO and ML pipelines

Extras can be combined. Use ``openbabel`` for Open Babel, ``ase`` for ASE databases,
``s3`` for S3, ``catcif`` for CIF archives, ``posebusters`` for structure validation,
``dev`` for development tools, and ``docs`` for documentation builds.

The upcoming 3.0 release is available from the existing release branch:

.. code-block:: bash

   python -m pip install "atomworks[ml] @ git+https://github.com/RosettaCommons/atomworks.git@release/atomworks-3-0"

Development
-----------

.. code-block:: bash

   git clone --branch release/atomworks-3-0 https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[ml,dev,openbabel]"
   python .github/ci/setup_test_data.py
   pytest tests -m "not benchmark and not slow and not requires_digs"

The setup script downloads the same checksum-verified public fixtures used by CI,
including pinned wwPDB revisions for stored-result tests. Tests use Biotite's
built-in CCD by default; see :doc:`mirrors` for optional full PDB/CCD mirrors.
Missing required fixtures are errors. Tests needing an unavailable GPU or external
tool are marked accordingly.

For a complete CPU run, install the ``ase`` extra as well and omit ``not slow``.
Use ``-n 2`` for parallel testing, or ``-n 1`` for memory-intensive tests.

See :doc:`contributor_guide` for documentation builds and release instructions.
