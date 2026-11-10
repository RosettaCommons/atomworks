Installation
============

Requires Python 3.12 or newer. Choose the installation that fits your workflow:

.. code-block:: bash

   pip install atomworks               # IO library
   pip install "atomworks[ml]"          # IO + ML library
   pip install "atomworks[ml,scripts]"  # IO + ML + command-line tools

.. _optional-integrations:

.. dropdown:: Optional integrations and developer tools

   Add integrations only when needed: ``s3`` (S3 access), ``hf`` (Hugging Face),
   ``ase`` (ASE/LMDB datasets), ``openbabel`` (additional chemistry backend),
   ``posebusters`` (structure validation), or ``catcif`` (CIF archives).
   For example: ``pip install "atomworks[ml,s3]"``.

   Use ``atomworks[scripts]`` for CLI tools without ML, ``atomworks[dev]`` for
   ML, CLI, visualization, notebooks, and test/lint tools, or ``atomworks[docs]``
   to build the documentation. Specialized integrations remain opt-in.

Development installation
------------------------
For development, use a virtual environment:

.. code-block:: bash

   git clone --branch production https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   python -m venv .venv
   source .venv/bin/activate
   make install  # installs dev and openbabel extras


Running the test suite
----------------------

Public CI runs on GitHub-hosted machines using a versioned bundle containing the
shared fixtures and the PDB subset. To use the same inputs locally:

.. code-block:: bash

   mkdir -p tests/data
   curl --fail --location --retry 3 https://github.com/RosettaCommons/atomworks/releases/download/test-data-3.0.0/atomworks-test-data-3.0.0.tar.gz | tar --skip-old-files -xz -C tests/data
   pip install -e ".[dev,ase,openbabel]"
   pytest tests -n 2 --dist=worksteal -m "not benchmark and not slow and not requires_digs and not requires_pymol_remote and not requires_x3dna"

The bundle includes its public-source provenance. Tests use Biotite's bundled CCD;
a full PDB/CCD mirror is not required. Infrastructure-dependent tests remain excluded.
Run the slow tests separately with one worker when needed to limit memory use.

Setting up PDB/CCD mirrors
--------------------------

For production use or training on the full PDB, you'll want complete mirrors rather than the test subset. See :doc:`mirrors` for detailed instructions on:

* Setting up a full PDB mirror (~100 GB)
* Setting up a CCD mirror (~2 GB)
* Configuring environment variables for production use

Migrating to AtomWorks 3.0
--------------------------

For changes and compatibility guidance, see the `short migration guide <migration/migration-short.html>`_ or the `detailed migration guide <migration/migration-long.html>`_.
