#!/usr/bin python3
""" Obtain information about the running system, environment and GPU. 
Url:
    Repo: https://github.com/deepfakes/faceswap
    File: https://github.com/deepfakes/faceswap/blob/master/lib/sysinfo.py

"""
import locale
import os
import platform
import re
import sys
from subprocess import PIPE, Popen
from pathlib2 import Path

import psutil


class SysInfo():
    """ Obtain information about the System, Python and GPU """

    def __init__(self):
        self._system = dict(platform=platform.platform(),
                            system=platform.system(),
                            machine=platform.machine(),
                            release=platform.release(),
                            processor=platform.processor(),
                            cpu_count=os.cpu_count())
        self._python = dict(implementation=platform.python_implementation(),
                            version=platform.python_version())
        self._cuda_path = self.get_cuda_path()

    @property
    def encoding(self):
        """ str: The system preferred encoding """
        return locale.getpreferredencoding()

    @property
    def is_conda(self):
        """ bool: `True` if running in a Conda environment otherwise ``False``. """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def is_linux(self):
        """ bool: `True` if running on a Linux system otherwise ``False``. """
        return self._system["system"].lower() == "linux"

    @property
    def is_macos(self):
        """ bool: `True` if running on a macOS system otherwise ``False``. """
        return self._system["system"].lower() == "darwin"

    @property
    def is_windows(self):
        """ bool: `True` if running on a Windows system otherwise ``False``. """
        return self._system["system"].lower() == "windows"

    @property
    def is_virtual_env(self):
        """ bool: `True` if running inside a virtual environment otherwise ``False``. """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
        return retval

    @property
    def home(self):
        """ str: The home directory of this machine."""
        return str(Path.home())

    @property
    def ram_free(self):
        """ int: The amount of free RAM in bytes. """
        return psutil.virtual_memory().free

    @property
    def ram_total(self):
        """ int: The amount of total RAM in bytes. """
        return psutil.virtual_memory().total

    @property
    def ram_available(self):
        """ int: The amount of available RAM in bytes. """
        return psutil.virtual_memory().available

    @property
    def ram_used(self):
        """ int: The amount of used RAM in bytes. """
        return psutil.virtual_memory().used

    @property
    def fs_command(self):
        """ str: The command line command used to execute the file. """
        return " ".join(sys.argv)

    @property
    def installed_pip(self):
        """ str: The list of installed pip packages within this file's scope. """
        pip = Popen("{} -m pip freeze".format(sys.executable),
                    shell=True, stdout=PIPE)
        installed = pip.communicate()[0].decode().splitlines()
        return "\n".join(installed)

    @property
    def installed_conda(self):
        """ str: The list of installed Conda packages within this file's scope. """
        if not self.is_conda:
            return None
        conda = Popen("conda list", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = conda.communicate()
        if stderr:
            return "Could not get package list"
        installed = stdout.decode().splitlines()
        return "\n".join(installed)

    @property
    def conda_version(self):
        """ str: The installed version of Conda, or `N/A` if Conda is not installed. """
        if not self.is_conda:
            return "N/A"
        conda = Popen("conda --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = conda.communicate()
        if stderr:
            return "Conda is used, but version not found"
        version = stdout.decode().splitlines()
        return "\n".join(version)

    @property
    def git_branch(self):
        """ str: The git branch that is currently being used to execute this file. """
        git = Popen("git status", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = git.communicate()
        if stderr:
            return "Not Found"
        branch = stdout.decode().splitlines()[0].replace("On branch ", "")
        return branch

    @property
    def git_commits(self):
        """ str: The last 5 git commits for the currently running this file. """
        git = Popen("git log --pretty=oneline --abbrev-commit -n 5",
                    shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = git.communicate()
        if stderr:
            return "Not Found"
        commits = stdout.decode().splitlines()
        return ". ".join(commits)

    @property
    def cuda_keys_windows(self):
        """ list: The CUDA Path environment variables stored for Windows users. """
        return [key for key in os.environ.keys() if key.lower().startswith("cuda_path_v")]

    @property
    def cuda_version(self):
        """ str: The installed CUDA version. """
        chk = Popen("nvcc -V", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if not stderr:
            version = re.search(r".*release (?P<cuda>\d+\.\d+)",
                                stdout.decode(self.encoding))
            version = version.groupdict().get("cuda", None)
            if version:
                return version
        # Failed to load nvcc
        if self.is_linux:
            version = self.cuda_version_linux()
        elif self.is_windows:
            version = self.cuda_version_windows()
        else:
            version = "Unsupported OS"
            if self.is_conda:
                version += ". Check Conda packages for Conda Cuda"
        return version

    @property
    def cudnn_version(self):
        """ str: The installed cuDNN version. """
        if self.is_linux:
            cudnn_checkfiles = self.cudnn_checkfiles_linux()
        elif self.is_windows:
            cudnn_checkfiles = self.cudnn_checkfiles_windows()
        else:
            retval = "Unsupported OS"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval

        cudnn_checkfile = None
        for checkfile in cudnn_checkfiles:
            if os.path.isfile(checkfile):
                cudnn_checkfile = checkfile
                break

        if not cudnn_checkfile:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval

        found = 0
        with open(cudnn_checkfile, "r") as ofile:
            for line in ofile:
                if line.lower().startswith("#define cudnn_major"):
                    major = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_minor"):
                    minor = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_patchlevel"):
                    patchlevel = line[line.rfind(" ") + 1:].strip()
                    found += 1
                if found == 3:
                    break
        if found != 3:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval
        return "{}.{}.{}".format(major, minor, patchlevel)

    @staticmethod
    def cudnn_checkfiles_linux():
        """ Obtain the location of the files to check for cuDNN location in Linux.

        Returns
        str:
            The location of the header files for cuDNN
        """
        chk = os.popen(
            "ldconfig -p | grep -P \"libcudnn.so.\\d+\" | head -n 1").read()
        if "libcudnn.so." not in chk:
            return list()
        chk = chk.strip().replace("libcudnn.so.", "")
        cudnn_vers = chk[0]
        cudnn_path = chk[chk.find("=>") + 3:chk.find("libcudnn") - 1]
        cudnn_path = cudnn_path.replace("lib", "include")
        cudnn_checkfiles = [os.path.join(cudnn_path, "cudnn_v{}.h".format(cudnn_vers)),
                            os.path.join(cudnn_path, "cudnn.h")]
        return cudnn_checkfiles

    def cudnn_checkfiles_windows(self):
        """ Obtain the location of the files to check for cuDNN location in Windows.

        Returns
        str:
            The location of the header files for cuDNN
        """
        # TODO A more reliable way of getting the windows location
        if not self._cuda_path and not self.cuda_keys_windows:
            return list()
        if not self._cuda_path:
            self._cuda_path = os.environ[self.cuda_keys_windows[0]]

        cudnn_checkfile = os.path.join(self._cuda_path, "include", "cudnn.h")
        return [cudnn_checkfile]

    def get_cuda_path(self):
        """ Obtain the path to Cuda install location.

        Returns
        -------
        str
            The path to the install location of Cuda on the system
        """
        if self.is_linux:
            path = self.cuda_path_linux()
        elif self.is_windows:
            path = self.cuda_path_windows()
        else:
            path = None
        return path

    @staticmethod
    def cuda_path_linux():
        """ Obtain the path to Cuda install location on Linux.

        Returns
        -------
        str
            The path to the install location of Cuda on a Linux system
        """
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)
        chk = os.popen(
            "ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if ld_library_path and not chk:
            paths = ld_library_path.split(":")
            for path in paths:
                chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1".format(path)).read()
                if chk:
                    break
        if not chk:
            return None
        return chk[chk.find("=>") + 3:chk.find("targets") - 1]

    @staticmethod
    def cuda_path_windows():
        """ Obtain the path to Cuda install location on Windows.

        Returns
        -------
        str
            The path to the install location of Cuda on a Windows system
        """
        cuda_path = os.environ.get("CUDA_PATH", None)
        return cuda_path

    def cuda_version_linux(self):
        """ Obtain the installed version of Cuda on a Linux system.

        Returns
        -------
        The installed CUDA version on a Linux system
        """
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)
        chk = os.popen(
            "ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if ld_library_path and not chk:
            paths = ld_library_path.split(":")
            for path in paths:
                chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1".format(path)).read()
                if chk:
                    break
        if not chk:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda Cuda"
            return retval
        cudavers = chk.strip().replace("libcudart.so.", "")
        return cudavers[:cudavers.find(" ")]

    def cuda_version_windows(self):
        """ Obtain the installed version of Cuda on a Windows system.

        Returns
        -------
        The installed CUDA version on a Windows system
        """
        cuda_keys = self.cuda_keys_windows
        if not cuda_keys:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda Cuda"
            return retval
        cudavers = [key.lower().replace("cuda_path_v", "").replace("_", ".")
                    for key in cuda_keys]
        return " ".join(cudavers)

    def full_info(self):
        """ Obtain extensive system information stats, formatted into a human readable format.

        Returns
        -------
        str
            The system information for the currently running system, formatted for output to
            console or a log file.
        """
        retval = "\n============ System Information ============\n"
        sys_info = {"os_platform": self._system["platform"],
                    "os_machine": self._system["machine"],
                    "os_release": self._system["release"],
                    "py_conda_version": self.conda_version,
                    "py_implementation": self._python["implementation"],
                    "py_version": self._python["version"],
                    "py_command": self.fs_command,
                    "py_virtual_env": self.is_virtual_env,
                    "sys_cores": self._system["cpu_count"],
                    "sys_processor": self._system["processor"],
                    "sys_ram": self.format_ram(),
                    "encoding": self.encoding,
                    "git_branch": self.git_branch,
                    "git_commits": self.git_commits,
                    "gpu_cuda": self.cuda_version,
                    "gpu_cudnn": self.cudnn_version}
        for key in sorted(sys_info.keys()):
            retval += ("{0: <20} {1}\n".format(key + ":", sys_info[key]))
        retval += "\n=============== Pip Packages ===============\n"
        retval += self.installed_pip
        if self.is_conda:
            retval += "\n\n============== Conda Packages ==============\n"
            retval += self.installed_conda
        return retval

    def format_ram(self):
        """ Format the RAM stats into Megabytes to make it more readable.

        Returns
        -------
        str
            The total, available, used and free RAM displayed in Megabytes
        """
        retval = list()
        for name in ("total", "available", "used", "free"):
            value = getattr(self, "_ram_{}".format(name))
            value = int(value / (1024 * 1024))
            retval.append("{}: {}MB".format(name.capitalize(), value))
        return ", ".join(retval)


def get_sysinfo():
    """ Obtain extensive system information stats, formatted into a human readable format.
    If an error occurs obtaining the system information, then the error message is returned
    instead.

    Returns
    -------
    str
        The system information for the currently running system, formatted for output to
        console or a log file.
    """
    try:
        retval = SysInfo().full_info()
    except Exception as err:  # pylint: disable=broad-except
        retval = "Exception occured trying to retrieve sysinfo: {}".format(err)
    return retval

# sysinfo = get_sysinfo()  # pylint: disable=invalid-name
