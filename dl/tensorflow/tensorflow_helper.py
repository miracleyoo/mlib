#!/usr/bin python3
""" Some functions frequently used when you call tensorflow. """

import os
import warnings

from pathlib import Path


def set_system_verbosity(loglevel):
    """ Set the verbosity level of tensorflow and suppresses
        future and deprecation warnings from any modules
    From:
        https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
    Can be set to:
        0 - all logs shown
        1 - filter out INFO logs
        2 - filter out WARNING logs
        3 - filter out ERROR logs  
    """
    # TODO suppress tensorflow deprecation warnings """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = loglevel
    if loglevel != '0':
        for warncat in (FutureWarning, DeprecationWarning):
            warnings.simplefilter(action='ignore', category=warncat)