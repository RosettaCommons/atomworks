Installation
============

AtomWorks can be installed in several ways, depending on your workflow and environment. Below are the recommended methods:

0. Prerequisites
-----------------

Before installing AtomWorks, ensure you have the following prerequisites:

* Python 3.11 or higher
* `dotenv <https://www.npmjs.com/package/dotenv>`_

1. Installing via pip (recommended)
-----------------------------------
This is the easiest way to get started with AtomWorks.

.. code-block:: bash

   pip install atomworks # base installation version without torch (for only atomworks.io)
   pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
   pip install "atomworks[dev]" # with development dependencies
   pip install "atomworks[ml,dev]" # with all dependencies"

You can also install AtomWorks with `Open Babel <https://openbabel.org/>`_, an alternative to RDKit:

.. code-block:: bash

   pip install "atomworks[openbabel]"

or for all possible dependencies: 

.. code-block:: bash

   pip install "atomworks[ml,openbabel,dev]"

Open Babel is not automatically installed with AtomWorks due to its larger size and additional dependencies, only install it if you plan to use it.

2. Development Installation
---------------------------
For development:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   make install  # or pip install -e ".[dev]"

To install in a fresh environment:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/atomworks.git
   cd atomworks
   make env


Security constraints for new environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a checkout, you can apply optional security constraints when installing::

   uv pip install -c constraints/security.txt ".[ml,ase,dev,docs]"

Select only the extras you need. Constraints restrict versions without installing
unrequested extras. They select a newer Torch for new ML environments; the
library's broader Torch range is unchanged. Check model and CUDA compatibility
before upgrading an existing environment. See ``constraints/README.md`` for the
audit scope and commands.

3. Running the Test Suite
-------------------------

To run the AtomWorks test suite, you need to download the test data and configure environment variables.

**Step 1: Download test data and the CCD mirror**

From the repository root, run:

.. code-block:: bash

   atomworks setup tests
   atomworks ccd sync tests/data/ccd

The first command extracts the test pack into ``tests/data/`` and downloads missing
PDB structures listed in the pack into ``tests/data/pdb/``. Shared MSA, template,
and metadata fixtures are provided under ``tests/data/shared/``.

The second command separately downloads a full CCD mirror into ``tests/data/ccd/``.
This mirror is not supplied by the test pack. Both commands use explicit default
or destination paths, so configure the environment after the downloads finish.

**Step 2: Create a .env file**

Create a ``.env`` file in the repository root with the paths to the test data:

.. code-block:: bash

   # For running tests with the test pack:
   PDB_MIRROR_PATH=tests/data/pdb
   CCD_MIRROR_PATH=tests/data/ccd

You can copy ``.env.sample`` as a starting point:

.. code-block:: bash

   cp .env.sample .env
   # Then edit .env to set the paths above

**Step 3: Run the tests**

.. code-block:: bash

   # Run all tests (excluding very slow ones)
   pytest tests -m "not very_slow"

   # Run tests in parallel for faster execution
   pytest tests -m "not very_slow" -n auto

   # Run a specific test file
   pytest tests/io/components/test_parser.py


4. Setting Up Full PDB/CCD Mirrors
----------------------------------

For production use or training on the full PDB, you'll want complete mirrors rather than the test subset. See :doc:`mirrors` for detailed instructions on:

* Setting up a full PDB mirror (~100 GB)
* Setting up a CCD mirror (~2 GB)
* Configuring environment variables for production use
