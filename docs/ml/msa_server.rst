Remote MSA generation
=====================

Use a ColabFold-compatible MMseqs2 server when you do not have local search
binaries or sequence databases. Install the ``ml`` extra, then run::

    atomworks msa generate sequences.csv output_msas/ --backend mmseqs2_server --check-existing

The CSV contains a protein sequence column; use ``--sequence-column`` to select
it when there are multiple columns. The default endpoint is the public ColabFold
server. Submitted sequences are sent to that service. To use your own server::

    atomworks msa generate sequences.csv output_msas/ --backend mmseqs2_server \
        --server-url https://msa.example.org --server-job-timeout 1800

Set ``MSA_SERVER_USERNAME`` and ``MSA_SERVER_PASSWORD`` for HTTP basic
authentication. For API-key authentication, set ``MSA_SERVER_API_KEY`` and pass
``--server-api-key-header`` with the header name expected by your server.

Each batch has a configurable deadline covering submission, polling, retries,
and download. Requests also have individual timeouts. A failed batch raises an
error; rerun with ``--check-existing`` to reuse completed files. This searches
the output directory and any configured ``PROTEIN_MSA_DIRS`` or
``--existing-msa-dirs`` paths.

Output uses the same sequence hashes, sharding, and compressed A3M layout as
local generation. The server filters results by default. Local HHfilter is only
required if you explicitly set ``--max-final-sequences``. The existing
``mmseqs2`` and ``hhblits`` backends retain their local filtering defaults.

ColabFold server headers generally lack ``TaxID=`` annotations, which AtomWorks
uses for multimer pairing. These MSAs therefore do not provide taxonomic pairing.

Python usage
------------

.. code-block:: python

    from atomworks.ml.preprocessing.msa.server import MSAServerConfig, make_msas_mmseqs_server

    make_msas_mmseqs_server(
        ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"],
        "output_msas/",
        config=MSAServerConfig(job_timeout=1800),
    )

For CSV input, select ``MSAGenerationConfig(backend="mmseqs2_server")`` and pass
it to ``make_msas_from_csv``. Its ``server_config`` accepts ``MSAServerConfig``;
``server_max_final_sequences`` optionally enables local HHfilter for remote
results, independently of the local backends' ``max_final_sequences`` setting.
