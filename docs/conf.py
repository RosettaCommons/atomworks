# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys

sys.path.insert(0, os.path.abspath("../src"))

import atomworks

project = "atomworks"
copyright = "2025, bakerlab"
author = "bakerlab"

# Get the raw version from atomworks
raw_version = atomworks.__version__
print(f"Raw version from atomworks: {raw_version}")

# Extract clean version for documentation
# Handle formats like: v2.29.0, v2.29.0+dev26.ad450d1, v2.29.0-dirty, v2.29.0+dev26.ad450d1-dirty
version_match = re.match(r"^v?(\d+\.\d+\.\d+)", str(raw_version))
if version_match:
    version = version_match.group(1)
else:
    # Fallback if regex doesn't match
    version = str(raw_version).lstrip("v").split("+")[0].split("-")[0]

print(f"Clean version for docs: {version}")

# For version switcher, we want to match against the exact version format in switcher.json
# This should match what your GitHub workflow generates
switcher_version = version  # Use clean version for matching

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx_gallery.gen_gallery",  # Generates auto_examples/ from examples/
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,  # Unlimited depth
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,  # Unlimited depth
    "header_links_before_dropdown": 8,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {
        "image_light": "_static/atomworks_logo_light.svg",
        "image_dark": "_static/atomworks_logo_color.svg",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://baker-laboratory.github.io/atomworks-dev/latest/_static/switcher.json",
        "version_match": switcher_version,
    },
    "favicons": [
        {
            "rel": "icon",
            "sizes": "16x16",
            "href": "favicon-16x16.png",
        },
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "favicon-32x32.png",
        },
    ],
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # where to put the generated gallery
    "image_scrapers": ("matplotlib",),
    "thumbnail_size": (350, 350),
    "default_thumb_file": "./_static/atomworks_logo_color.svg",
}
