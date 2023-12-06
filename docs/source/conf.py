# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../elevation_mapping_cupy/script'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sensor_processing/semantic_sensor/script'))

autodoc_mock_imports = [
    "cupy",
    "cupyx",
    "rospy",
    "ros_numpy"
    "torchvision",
    "numpy",
    "scipy",
    "sklearn",
    "dataclasses",
    "ruamel.yaml",
    "opencv-python",
    "simple-parsing",
    "scikit-image",
    "matplotlib",
    "catkin-tools",
    "catkin_pkg",
    "detectron2",
    "torch",
    "shapely",
    "simple_parsing",
    ]

on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

project = "elevation_mapping_cupy"
copyright = "2022, Takahiro Miki, Gian Erni"
author = "Takahiro Miki, Gian Erni"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme_options = {
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    # "style_nav_header_background": "#A00000",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
