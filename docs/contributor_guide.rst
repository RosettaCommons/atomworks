.. _contributor-best-practices:

===============================
Contributing
===============================

.. note::
   This is a non-exhaustive list of best practices for contributing code, based on industry standards and our team's experience.

As you code
-------------

1. **Reduce cognitive overhead:**
   
   a. Pick meaningful, descriptive variable names.
   
   b. Write docstrings (leverage AI!) and comments. To be used in the API documentation the docstring should 
      follow the Google style guide: `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

   c. Follow the `Python Zen <https://peps.python.org/pep-0020/>`_ – explicit is better than implicit, etc.

2. **Write tests.**

As you commit
---------------

1. Keep commits as "one logical unit". This means that each commit should be a set of related changes  
   that accomplish one task, fix one bug, or implement one feature. Using an editor like `VS Code <https://code.visualstudio.com/docs/sourcecontrol/overview>`_
   or using `GitHub Desktop <https://docs.github.com/en/desktop>`_ can help you stage related changes together.  

2. Adhere to `semantic commit conventions <https://www.conventionalcommits.org/en/v1.0.0/>`_.  

3. Format & lint your code (``make format``).  

4. Submit a draft PR so people know you are working on this & can provide advice/feedback early on.  

As you finalize a PR
---------------------

1. Target **release/atomworks-3-0** for the AtomWorks 3.0 release.
2. Keep overall PR under <400 LOC (lines of code) (Rule of thumb: 500 LOC takes about 1h to review).
3. Read and fill out the `PR checklist <https://github.com/RosettaCommons/atomworks/blob/production/.github/pull_request_template.md>`_.

As you review
---------------

1. Foster a positive review culture – we want to learn from each other. Be critical but kind.
2. Practice light-weight code reviews. Submit something small to atomworks.io/atomworks.ml that fixes a bug / improves documentation / adds a tiny feature to practice this within the next 24h. (Can be less than 30min)
3. Keep review time <1h and <500 LOC for focus.

Contributing to the documentation
---------------------------------
The external AtomWorks documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/#>`_ and hosted on `GitHub Pages <https://docs.github.com/en/pages>`_.
Aside from having AtomWorks and its dependencies installed, to build the documentation locally, you will need to install the documentation requirements:

.. code-block:: bash

   uv pip install -r docs/docs_requirements.txt

To build the documentation, navigate to the ``docs`` directory and run:
   
   .. code-block:: bash

      make html

If you are new to Sphinx, please refer to the `Sphinx documentation <https://www.sphinx-doc.org/en/master/>`_ for guidance on writing and formatting documentation.
All of the documentation is written in reStructuredText (reST) format. For more information on reST, see the `reStructuredText Primer <https://docutils.sourceforge.io/docs/user/rst/quickstart.html>`_.

Other Resources
---------------

- `Best Practices for Code Review | SmartBear <https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/>`_


.. raw:: html

   <hr>

PR Hygiene
=================

When contributing to this repository, please follow these steps:

1. Clone the repository
2. Create the development environment (see the :doc:`installation` guide).
3. Create a new branch for your changes. 
   - Use the following convention to name your branch: ``<category>/<description>``. Categories: ``feat``, ``fix``, ``hotfix``, ``refactor``, ``docs``, ``perf``.
   - Example: ``feat/support-rdkit-small-molecule``
4. Make and commit your changes on your new branch. 
   - Run autoformatting tools (``make format``) before committing.
   - Use commit messages like ``<type>: <description>``. Types: ``feat``, ``fix``, ``refactor``, ``docs``, ``chore``, ``wip``.
   - Example: ``git commit -m "docs: add contributing guidelines"``
5. Open a pull request to ``release/atomworks-3-0`` and describe your changes.
6. Wait for review and merge your changes.


Publishing a release
--------------------

Releases use the explicit version in ``pyproject.toml`` and a matching Git tag.
Branch pushes and PR merges do not publish packages.

One-time repository setup:

- Configure the PyPI ``atomworks`` Trusted Publisher: owner ``RosettaCommons``,
  repository ``atomworks``, workflow ``release_and_docs.yaml``, environment ``pypi``.
- Protect the ``pypi`` environment with reviewer approval and allow version tags.
- Set GitHub Pages Source to **GitHub Actions** and allow version tags in
  ``github-pages``. The workflow preserves versioned pages in ``gh-pages``.

For 3.0.0, merge the reviewed PRs, wait for CI on the final release commit, and
confirm that ``project.version`` is ``3.0.0`` and the tag/PyPI version are unused.
From a checkout whose ``origin`` is the public repository:

.. code-block:: bash

   git fetch origin
   git tag -a v3.0.0 origin/release/atomworks-3-0 -m "AtomWorks 3.0.0"
   git push origin refs/tags/v3.0.0

The tag runs package checks, installed-wheel checks, scientific CPU tests
on GitHub-hosted runners and the docs build. After they pass, approve ``pypi`` to publish the checked
archives; subsequent jobs create the GitHub Release and deploy documentation.
Verify a fresh ``pip install atomworks==3.0.0``, ``aw --help`` and the versioned docs.

Prereleases follow the same process without replacing stable ``latest`` docs.
Never move a published tag or reuse a published package version.
