import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'Thesis'
author = 'Joey'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

html_theme = 'sphinx_rtd_theme'
