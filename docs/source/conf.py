"""Sphinx configuration for the MATCHA documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable so sphinx-autoapi can introspect it if needed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

project = "MATCHA"
author = "Merck KGaA, Darmstadt, Germany"
copyright = f"2026, {author}"

extensions = [
    "myst_parser",
    "autoapi.extension",
    "nbsphinx",
    "jupyter_sphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3

autoapi_type = "python"
autoapi_dirs = [str(_REPO_ROOT / "src" / "matcha")]
autoapi_root = "api"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/tests/*", "*/__pycache__/*"]

nbsphinx_execute = "never"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path: list[str] = []
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = ["_static"]
