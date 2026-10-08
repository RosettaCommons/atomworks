Installation
============

AtomWorks can be installed in several ways, depending on your workflow and environment. Below are the recommended methods:

0. Prerequisites
-----------------

Before installing AtomWorks, ensure you have the following prerequisites:

* Python 3.11 or higher
* Pip installs ``python-dotenv`` automatically; Node.js is not required.

1. Installing via pip (recommended)
-----------------------------------
This is the easiest way to get started with AtomWorks.

.. code-block:: bash

   pip install atomworks # base installation version without torch (for only atomworks.io)
   pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
   pip install "atomworks[dev]" # with development dependencies
   pip install "atomworks[ml,dev]" # ML and development tools

You can also install AtomWorks with `Open Babel <https://openbabel.org/>`_, an alternative to RDKit:

.. code-block:: bash

   pip install "atomworks[openbabel]"

Combine extras as needed:

.. code-block:: bash

   pip install "atomworks[ml,openbabel,dev]"

Open Babel is not automatically installed with AtomWorks due to its larger size and additional dependencies, only install it if you plan to use it.

2. Development Installation
---------------------------
For development, use a virtual environment:

.. code-block:: bash

   git clone --branch production https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   python -m venv .venv
   source .venv/bin/activate
   make install  # or pip install -e ".[dev]"


3. Running the Test Suite
-------------------------

Public CI runs on GitHub-hosted machines using a versioned bundle containing the
shared fixtures and the PDB subset. To use the same inputs locally:

.. code-block:: bash

   mkdir -p tests/data
   curl --fail --location --retry 3 https://github.com/RosettaCommons/atomworks/releases/download/test-data-3.0.0/atomworks-test-data-3.0.0.tar.gz | tar -xz -C tests/data
   pip install -e ".[ml,dev,ase,openbabel]"
   pytest tests -n 2 --dist=worksteal -m "not benchmark and not slow and not requires_digs and not requires_pymol_remote and not requires_x3dna"

The bundle includes its public-source provenance. Tests use Biotite's bundled CCD;
a full PDB/CCD mirror is not required. Infrastructure-dependent tests remain excluded.
Run the slow tests separately with one worker when needed to limit memory use.

4. Setting Up Full PDB/CCD Mirrors
----------------------------------

For production use or training on the full PDB, you'll want complete mirrors rather than the test subset. See :doc:`mirrors` for detailed instructions on:

* Setting up a full PDB mirror (~100 GB)
* Setting up a CCD mirror (~2 GB)
* Configuring environment variables for production use


.. toctree::
   :maxdepth: 1

   migration/index
