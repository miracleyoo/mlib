import os
import os.path as op

import platform
import pandas as pd
import numpy as np
import pickle as pkl
import shutil
from pathlib2 import Path

# Global variables
_image_extensions = ['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff']
_video_extensions = ['.avi', '.flv', '.mkv', '.mov', '.mp4', '.mpeg', '.webm']

__all__ = ["host_values_selector", "make_new", "get_folder", "get_new_folder", "get_image_paths", "backup_file",
           "stem", "suffix", "listdir"]


def host_values_selector(misaka_value=None, gypsum_value=None, win_value=None, macos_value=None):
    """ Return a path which is compatible for multiple machines based on host name and platform.
    Args:
        XXX_value: The value for this certain machine.
    Returns:
        _ : The value spicified for the current machine.
    """
    ostype = platform.system()
    hostname = platform.node()
    if ostype == 'Windows':
        return win_value
    elif hostname == 'misaka':
        return misaka_value
    elif hostname.startswith("node") or hostname == "gypsum":
        return gypsum_value
    else:
        print("OS type not supported!")
        return None


def get_new_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist, creating new one if existing.

    If the folder already exists, this function will try to make `path`_`idx` folder,
    idx will starts from 1 and keeps +1 until find a valid name without occupation.

    If the folder and its parent folders don't exist, keeps making these series of folders.

    Args:
        path: The path of the new folder.
    Returns:
        _ : The guaranteed new path of the folder/file.
    """
    path = Path(path)
    name = path.name
    root = Path(*path.parts[:-1])

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    else:
        idx = 1
        while True:
            new_name = name+"_"+str(idx)
            new_path = root / new_name
            if not new_path.exists():
                if new_path.suffix == '':
                    os.makedirs(str(new_path))
                break
            idx += 1
        return str(new_path)


def make_new(path):
    return get_new_folder(path)


def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist 
    Args:
        path: The path of the new folder.
    Returns:
        _ : The guaranteed path of the folder/file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_paths(directory, exclude=list(), debug=False):
    """ Return a list of images that reside in a folder """
    image_extensions = _image_extensions
    exclude_names = [op.basename(Path(x).stem[:Path(x).stem.rfind('_')] +
                                 Path(x).suffix) for x in exclude]
    dir_contents = list()

    if not op.exists(directory):
        directory = get_folder(directory)

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    for chkfile in dir_scanned:
        if any([chkfile.name.lower().endswith(ext) for ext in image_extensions]):
            if chkfile.name in exclude_names:
                if debug:
                    print("Already processed %s" % chkfile.name)
                continue
            else:
                dir_contents.append(chkfile.path)

    return dir_contents


def backup_file(root, filename=None):
    """ Backup a given file by appending .bk to the end """
    if filename is not None:
        ori_file = op.join(root, filename)
    backup_file = ori_file + '.bk'
    if op.exists(backup_file):
        os.remove(backup_file)
    if op.exists(ori_file):
        os.rename(ori_file, backup_file)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def suffix(path):
    return os.path.splitext(os.path.basename(path))[1]


def listdir(path,
            is_file=None,
            is_hide=None,
            suffix_type=None,
            start=None,
            end=None,
            contain=None):
    """ Return a file list of a certain folder. All these files meets some certain requirement.

    """
    path = str(path)

    assert is_file is None or type(is_file) == bool
    assert is_hide is None or type(is_hide) == bool
    assert suffix_type is None or type(suffix_type) == str
    assert start is None or type(start) == str
    assert end is None or type(end) == str
    assert contain is None or type(contain) == str

    files = []
    for file in os.listdir(path):
        file = op.join(path, file)

        if is_file is not None and is_file != op.isfile(file):
            continue

        if is_hide is not None and is_hide != stem(file).startswith("."):
            continue

        if suffix_type is not None and suffix(file).lstrip(".") != suffix_type.lstrip("."):
            continue

        if start is not None and not stem(file).startswith(start):
            continue

        if end is not None and not stem(file).endswith(end):
            continue

        if contain is not None and not contain in stem(file):
            continue

        files.append(file)

    files.sort()
    return files
