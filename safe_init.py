""" The safe version of __init__.py. 
    Only recommend when you are writing your code in *.py files.
    It is used to avoid pylint problem and make auto-completion
    works better.

    If you are using ipython, jupyter or already deploying, please
    use `from mlib import *` instead.

Usage:
    from mlib.safe import *
"""

import os
import sys
import time
import math
import json
import yaml
import tqdm
import glob
import random

import numpy as np
import pandas as pd
import pickle as pkl
import subprocess as sp
import multiprocessing as mp

import os.path as op

from functools import partial
from pathlib2 import Path
from importlib import reload
from mlib.lang import *

import mlib.file.path_func as pf
import mlib.file.io as mio

from .utils.logger import *
from .utils.basic import *