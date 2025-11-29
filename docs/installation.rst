Installation
============

AtomWorks can be installed in several ways, depending on your workflow and environment. Below are the recommended methods:

0. Prerequisites
-----------------

Before installing AtomWorks, ensure you have the following prerequisites:

* Python 3.12 or higher
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

2. Local Conda Environment
--------------------------
For development and testing:

.. code-block:: bash

   git clone git@git.ipd.uw.edu:ai/atomworks.io.git
   cd atomworks
   make install  # or pip install -e "."

   # Create a .env file (see .env.sample) with CCD and PDB paths as needed

To install in a fresh environment:

.. code-block:: bash

   git clone git@git.ipd.uw.edu:ai/atomworks.io.git
   cd atomworks
   # Set up Gitlab credentials in your shell
   echo 'export GITLAB_USER=<Gitlab_Username>' >> .bashrc
   echo 'export GITLAB_TOKEN=<Gitlab_PAT_Token>' >> .bashrc
   source .bashrc
   make env
   pytest tests

#3. As a Dependency in Your Apptainer
#------------------------------------
#Add `atomworks.io/src` to your apptainer's PYTHONPATH:
#
#.. code-block:: bash
#
#   export PYTHONPATH=$PWD/src:$PYTHONPATH
