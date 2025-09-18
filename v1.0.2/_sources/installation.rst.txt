Installation
============

atomworks can be installed in several ways, depending on your workflow and environment. Below are the recommended methods:

1. Installing via pip (recommended)
----------------------
This is the easiest way to get started with atomworks.

.. code-block:: bash

   pip install atomworks # base installation version without torch (for only atomworks.io)
   pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
   pip install "atomworks[dev]" # with development dependencies
   pip install "atomworks[ml,dev]" # with all dependencies


2. Using the Standalone Apptainer
-----------------------------------------------
This is ideal for dataset parsing and generation in a controlled environment.

.. code-block:: bash

   # Set up IPD-specific environment variables
   source ./.ipd/setup.sh
   # Use the provided apptainer image
   ./.ipd/atomworks.sif

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

3. As a Dependency in Your Apptainer
------------------------------------
Add `atomworks.io/src` to your apptainer's PYTHONPATH:

.. code-block:: bash

   export PYTHONPATH=$PWD/src:$PYTHONPATH

Or, if at IPD:

.. code-block:: bash

   source ./.ipd/setup.sh

For new apptainers, see the apptainer.spec file for integration details. 