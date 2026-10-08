Multiple Sequence Alignment in AtomWorks
========================================

AtomWorks provides several command-line tools for Multiple Sequence Alignment (MSA) operations.

Install AtomWorks with the ML extra. Filtering requires hhfilter from HH-suite
on PATH. Local generation requires the selected MMseqs2 or HHblits backend and
configured sequence databases, plus hhfilter for filtering the results. The
ColabFold Python API supports remote generation; see :doc:`ml/msa_server`.
Finding and organizing existing files do not run a sequence search.

Find
----

Provide ``--existing-msa-dirs`` or set ``PROTEIN_MSA_DIRS`` to the directories
containing your MSA files.

.. typer:: atomworks_cli.find:app
    :prog: atomworks msa find
    :show-nested:

Filter
------
.. typer:: atomworks_cli.filter:app
    :prog: atomworks msa filter
    :show-nested:

Generate
--------
.. typer:: atomworks_cli.generate:app
    :prog: atomworks msa generate
    :show-nested:

Organize
--------
.. typer:: atomworks_cli.organize:app
    :prog: atomworks msa organize
    :show-nested:
