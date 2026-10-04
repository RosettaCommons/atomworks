Installation
============

Use Python 3.11 or newer. Pip installs the core dependencies, including
``python-dotenv`` and the supported Biotite version; Node.js is not required.

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

   git clone https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[ml,dev,openbabel]"
   atomworks setup tests
   pytest tests -m "not benchmark and not slow and not requires_digs"
   pytest tests/experimental/protonation

``atomworks setup tests`` downloads public fixtures and their PDB subset into
``tests/data`` (the structure download requires ``rsync``). Tests use this local
subset and Biotite's built-in CCD by default. A full CCD mirror is optional;
tests requiring an unavailable GPU, mirror or external tool are marked accordingly.
CI uses a checksum-pinned fixture archive and downloads the PDB subset over HTTPS.
Missing required fixtures are errors, not skipped tests.

For a complete CPU run, install the ``ase`` extra as well and omit ``not slow``.
Use ``-n 2`` for parallel testing, or ``-n 1`` for memory-intensive tests.
See :doc:`mirrors` for full PDB/CCD mirrors and environment configuration.

Documentation
-------------

.. code-block:: bash

   python -m pip install -e ".[ml,docs,ase,openbabel]"
   make -C docs html

The build executes the offline parser/protonation/CIF round-trip example.
PR builds upload the rendered site as a ``documentation`` Actions artifact.
Version tags publish only after the scientific CPU tests, installed-wheel checks
and documentation build succeed; publication also requires the protected
``pypi`` environment and its Trusted Publisher configuration.
