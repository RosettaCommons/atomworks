# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import atomworks

project = "atomworks"
copyright = "2025, bakerlab"
author = "bakerlab"

release = os.environ.get("ATOMWORKS_DOCS_VERSION", str(atomworks.__version__).removeprefix("v"))
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx_gallery.gen_gallery",  # Generates auto_examples/ from examples/
    "myst_parser",  # Support Markdown tutorial pages
    "sphinx_design",  # Render collapsible tutorial code examples
]

myst_heading_anchors = 4
napoleon_use_ivar = True
html_favicon = "_static/favicon-32x32.png"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/GALLERY_HEADER.rst", "ml/preprocessing.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["navigation.css"]

# Theme options
html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,  # Unlimited depth
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,  # Unlimited depth
    "header_links_before_dropdown": 5,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "logo": {
        "image_light": "_static/atomworks_logo_light.svg",
        "image_dark": "_static/atomworks_logo_dark.svg",
    },
    "navbar_start": ["navbar-logo"],
    "switcher": {
        "json_url": "https://rosettacommons.github.io/atomworks/_static/switcher.json",
        "version_match": release,
    },
}

sphinx_gallery_conf = {
    "filename_pattern": r"/plot_",
    "abort_on_example_error": True,
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # where to put the generated gallery
    "image_scrapers": ("matplotlib",),
    "thumbnail_size": (350, 350),
    "default_thumb_file": "./_static/atomworks_logo_color.svg",
}

html_js_files = [
    ("https://scripts.simpleanalyticscdn.com/latest.js", {"async": "async", "defer": "defer"}),
]
