Remote MSA generation
=====================

Use the ColabFold server without local search binaries or sequence databases.
Install the ``ml`` extra, then call:

.. code-block:: python

    from atomworks.ml.preprocessing.msa.colabfold_server import make_msas_colabfold_server

    make_msas_colabfold_server(
        ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"], "output_msas/", use_templates=True
    )

Sequences are sent to the public ColabFold service by default. Set ``host_url``
to use another compatible server. Results use AtomWorks' hash-sharded MSA
layout. The ``atomworks msa generate`` CLI supports local MMseqs2 and HHblits;
use this Python API for remote generation.

Paired queries and templates
----------------------------

Use ``use_pairing=True`` to submit the sequences of one complex as a paired
query. Templates are available only for unpaired queries. To generate both
unpaired MSAs and paired complex MSAs, use ``make_msas_colabfold_server_batch()`` from
the same module.

Find template hit tables with ``find_template_alignments()`` from
``atomworks.ml.preprocessing.msa.finding``. To retrieve their structures, use
``fetch_template_structures_from_m8_file(m8_path, output_dir)`` from
``atomworks.ml.preprocessing.msa.template_structures``. It uses the PDB mirror
where available and downloads missing entries from RCSB. Returned files are raw
structures: chain mapping, release-date filtering and template featurization
remain the caller's responsibility.
