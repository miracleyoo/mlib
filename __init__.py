""" mlib is a personal library developed by miracleyoo.

This __init__.py file will use lazy import to import a bunch of
modules I use frequently, and these modules will be automatically
loaded when they are called in the code.

Also, some frequently used package abbreviation inside mlib is 
also included and loaded.
"""
from functools import partial
from pathlib2 import Path
from importlib import reload
from mlib.lang import *

from .utils.logger import *
from .utils.basic import *


_import_dict = {
    "os": "os",
    "sys": "sys",
    "time": "time",
    "math": "math",
    "yaml": "yaml",
    "json": "json",
    "random": "random",
    "op": "os.path",
    "np": "numpy",
    "pd": "pandas",
    "pkl": "pickle",
    "glob": "glob",
    "sp": "subprocess",
    "mp": "multiprocessing",

    "mio": "mlib.file.io",
    "pf": "mlib.file.path_func"
}

for key, value in _import_dict.items():
    exec(f"{key}=LazyLoader('{key}', globals(), '{value}')")
    # globals()[key] = ''

if is_notebook():
    tqdm = LazyLoader("tqdm", globals(), "tqdm.notebook")
    plt = LazyLoader("plt", globals(), "matplotlib.pyplot")
else:
    tqdm = LazyLoader("tqdm", globals(), "tqdm")

"""
# os = LazyLoader("os", globals(), "os")
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
"""
