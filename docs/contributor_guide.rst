.. _contributor-best-practices:

Contributing
============

Create a feature branch, make one focused change, and open a PR against
``release/atomworks-3-0`` for the 3.0 release. Keep unfinished work in draft;
public CI runs on drafts too. Preserve supplied chemistry, coordinates and
annotations unless the change explicitly requires otherwise.

Use descriptive names, small functions and Google-style docstrings. Keep commit
messages conventional (for example, ``fix(io): preserve insertion codes``).
Describe the resulting behavior in one sentence, followed by a few bullets for
the changes and validation. Add regression tests for bugs; avoid unrelated
formatting or refactors. Run ``make format`` and the relevant tests before review.

Development and documentation
-----------------------------

Follow the :doc:`installation` guide to create an environment. Install the
additional documentation dependencies and build with warnings treated as errors:

.. code-block:: bash

   uv pip install -e ".[ml,dev,docs,ase,openbabel]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

The gallery executes the offline ``plot_protonation.py`` example. Public CI also
checks scientific CPU tests, clean core-wheel installs, archive contents and
formatting. Stored-result regressions use checksum-verified wwPDB revisions from
``.github/ci/pdb_versions.tsv``; update inputs and expected results together when
intentionally adopting a new deposition. Other fixture downloads record their
hashes in CI artifacts. Infrastructure-dependent tests remain in the manual lab
workflow.

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
  The workflow retains old versions in ``gh-pages`` and explicitly deploys the
  assembled site; pushing that branch with ``GITHUB_TOKEN`` alone does not
  `trigger Pages <https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>`_.

Release 3.0.0
~~~~~~~~~~~~~

1. Merge the reviewed release PRs and wait for CI on the final combined release
   commit. Confirm ``project.version`` is ``3.0.0`` and that neither the version
   on PyPI nor the Git tag already exists.
2. From a clean checkout whose ``origin`` is the public repository, tag that
   reviewed commit:

   .. code-block:: bash

      git fetch origin
      git switch release/atomworks-3-0
      git pull --ff-only origin release/atomworks-3-0
      git tag -a v3.0.0 -m "AtomWorks 3.0.0"
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
