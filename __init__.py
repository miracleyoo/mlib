from functools import partial
from mlib.lang.lazy_loader import LazyLoader

from .utils.logger import *
from .utils.basic import *

_import_dict = {
    "os": "os",
    "sys": "sys",
    "time": "time",
    "math": "math",
    "yaml": "yaml",
    "random": "random",
    "op": "os.path",
    "np": "numpy",
    "pd": "pandas",
    "pkl": "pickle",
    "glob": "glob",

    "pf": "mlib.file.path_func",
    "lang": "mlib.lang"
}

for key, value in _import_dict.items():
    exec(f"{key}=LazyLoader('{key}', globals(), '{value}')")


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
