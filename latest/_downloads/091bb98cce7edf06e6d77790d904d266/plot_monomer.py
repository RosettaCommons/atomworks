"""
Plot a Monomer Structure
========================

This example demonstrates how to parse a monomer PDB file and plot the backbone trace using atomworks.io.

.. figure:: /_static/best_practices_mental_models.png
   :alt: Custom thumbnail for this example
   :width: 300px

   Custom thumbnail image for this example.

**Key steps:**
- Load a PDB file
- Visualize the backbone trace

"""

# %% [markdown]
# ## Import libraries

import io

from biotite.database import rcsb

from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.testing import get_pdb_path
from atomworks.io.utils.visualize import view


def get_example_path_or_buffer(pdb_id: str) -> io.StringIO | str:
    try:
        return get_pdb_path(pdb_id)
    except FileNotFoundError:
        return rcsb.fetch(pdb_id, format="cif")


# %% [markdown]
# ## Load and plot the structure

example = get_example_path_or_buffer("6lyz")  # e.g. '/path/to/6lyz.cif' or io.StringIO(rcsb.fetch("6lyz", "cif"))
atom_array = load_any(example, model=1, extra_fields=["charge", "occupancy"])

# ... inspect the first 15 atoms
print(f"Structure has {atom_array.array_length()} atoms. First 15 atoms:")
atom_array[:15]

# %%

# ... show the structure in a jupyter notebook
view(atom_array[atom_array.chain_id == "A"])
