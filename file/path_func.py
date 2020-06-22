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


def make_new(root, name=None):
    """ Check the ability of making a new folder or create a new file. 
    Args:
        root: The root path where the new file or folder is going to be created.
            If name is not provided, root is the full path of the target object.
        name: The folder/file name.
    Returns:
        _ : The guaranteed path of the folder/file.
    """
    root = Path(root)

    if name is None:
        name = root.name
        root = Path(*root.parts[:-1])

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if not (root/name).exists():
        if (root/name).suffix == '':
            (root/name).mkdir(parents=True, exist_ok=True)
        return str(root/name)
    else:
        idx = 1
        while True:
            new_name = name+"_"+str(idx)
            if not (root/new_name).exists():
                if (root/new_name).suffix == '':
                    os.makedirs(str(root/new_name))
                break
            idx += 1
        return str(root/new_name)


def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def listdir(path, is_file=None, is_hide=None, suffix_type=None, start=None, end=None):
    path = str(path)
    files = [op.join(path, f) for f in os.listdir(path)]
    if is_file:
        files = [i for i in files if op.isfile(i)]
    elif is_file == False:
        files = [i for i in files if not op.isfile(i)]

    if is_hide:
        files = [i for i in files if stem(i).startswith(".")]
    elif is_hide == False:
        files = [i for i in files if not stem(i).startswith(".")]

    if suffix_type is not None:
        files = [i for i in files if suffix(
            i).lstrip(".") == suffix_type.lstrip(".")]

    if start is not None:
        files = [i for i in files if stem(i).startswith(str(start))]

    if end is not None:
        files = [i for i in files if stem(i).endswith(str(end))]
    return files
