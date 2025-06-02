# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../training_step_1/mil_classification'))
sys.path.insert(0, os.path.abspath('../training_step_2/self_supervised_training'))
sys.path.insert(0, os.path.abspath('../visualization_step_1/xai_visualization'))
sys.path.insert(0, os.path.abspath('../visualization_step_2/fusion_visualization'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GRAPHITE'
copyright = '2025, Raktim Kumar Mondol et al.'
author = 'Raktim Kumar Mondol, Ewan K. A. Millar, Peter H. Graham, Lois Browne, Arcot Sowmya, Erik Meijering'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinx.ext.autodoc.typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension ---------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': True,
    'show-inheritance': True,
}

# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    'torch', 'torchvision', 'timm', 'numpy', 'pandas', 'matplotlib', 
    'sklearn', 'cv2', 'PIL', 'tqdm', 'seaborn', 'plotly', 'dash',
    'spams', 'openslide', 'histomicstk', 'cucim'
]

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- MyST configuration -----------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "html_admonition", 
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "substitution",
    "tasklist",
]

# -- Options for copybutton extension ----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
