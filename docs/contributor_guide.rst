.. _contributor-best-practices:

Contributing
============

Open focused PRs against ``release/atomworks-3-0``; unfinished work stays in draft.
Preserve supplied chemistry, coordinates and annotations. Use descriptive names,
small functions, Google-style docstrings and conventional commit messages.
Describe the change in one sentence and a few bullets, including validation.
Run ``make format`` and relevant tests; add regressions for bugs.

Development and documentation
-----------------------------

Follow the :doc:`installation` guide to create an environment. Install the
additional documentation dependencies and build with warnings treated as errors:

.. code-block:: bash

   uv pip install -e ".[ml,dev,docs,ase,openbabel]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

The gallery executes an offline parser/protonation/CIF round-trip example.
Public CI checks scientific CPU tests, core-wheel installs, archive contents,
formatting and docs. Stored-result inputs are pinned in ``.github/ci/pdb_versions.tsv``;
update inputs and expected results together when adopting a new deposition.
Infrastructure-dependent tests use the manual lab workflow.

Publishing a release
--------------------

The version in ``pyproject.toml`` is explicit. Branch pushes and PR merges do not
publish packages; matching ``v<version>`` tags in the public repository do.

One-time setup
~~~~~~~~~~~~~~

- Create a protected GitHub environment named ``pypi`` with required reviewers
  and deployment rules allowing version tags (``v*``).
- In the existing PyPI ``atomworks`` project's Publishing settings, add a
  `trusted publisher <https://docs.pypi.org/trusted-publishers/adding-a-publisher/>`_
  for owner ``RosettaCommons``, repository ``atomworks``, workflow
  ``release_and_docs.yaml``, and environment ``pypi``. No API token is needed.
- In GitHub Settings > Pages, select **GitHub Actions** as the publishing source.
  Allow version tags in the ``github-pages`` environment's deployment rules.
  The workflow preserves old versions in ``gh-pages`` and deploys the assembled site.

Release 3.0.0
~~~~~~~~~~~~~

1. Merge the reviewed release PRs and wait for CI on the final combined release
   commit. Confirm ``project.version`` is ``3.0.0`` and that neither the version
   on PyPI nor the Git tag already exists.
2. From a clean checkout whose ``origin`` is the public repository, tag that
   reviewed commit:

   .. code-block:: bash

      git fetch origin
      git tag -a v3.0.0 origin/release/atomworks-3-0 -m "AtomWorks 3.0.0"
      git push origin refs/tags/v3.0.0

3. Wait for the tag's build, wheel, science and docs checks, then approve the
   ``pypi`` deployment. It publishes the checked wheel/source archives; subsequent
   jobs create the GitHub Release and deploy versioned documentation.
4. Confirm a fresh ``pip install atomworks==3.0.0``, ``aw --help``, and the
   ``v3.0.0``/``latest`` documentation pages work.

For local archive checks, install ``build``, ``twine`` and ``packaging``, then run:

.. code-block:: bash

   python -m unittest discover -s .github/tests -v
   python -m build
   python .github/release.py artifacts
   python -m twine check --strict dist/*

The same procedure applies to later versions and prereleases such as
``v3.0.0rc1``. Prereleases do not replace stable ``latest`` documentation. Failed
jobs can be retried using their existing artifacts; never move a published tag
or reuse a published version.
