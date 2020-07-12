""" Check for models in their cache path
    If available, return the path, if not available, get, unzip and install model
"""
import os
import sys
import urllib
import logging
import zipfile

from socket import timeout as socket_timeout, error as socket_error
from tqdm import tqdm


class GetModel():
    """ Check for models in their cache path
        If available, return the path, if not available, get, unzip and install model

        model_filename: The name of the model to be loaded (see notes below)
        cache_dir:      The model cache folder of the current plugin calling this class
                        IE: The folder that holds the model to be loaded.
        git_model_id:   The second digit in the github tag that identifies this model.
                        See https://github.com/deepfakes-models/faceswap-models for more
                        information

        NB: Models must have a certain naming convention:
            IE: <model_name>_v<version_number>.<extension>
            EG: s3fd_v1.pb

            Multiple models can exist within the model_filename. They should be passed as a list
            and follow the same naming convention as above. Any differences in filename should
            occur AFTER the version number.
            IE: [<model_name>_v<version_number><differentiating_information>.<extension>]
            EG: [mtcnn_det_v1.1.py, mtcnn_det_v1.2.py, mtcnn_det_v1.3.py]
                [resnet_ssd_v1.caffemodel, resnet_ssd_v1.prototext]
        """

    def __init__(self, model_filename, cache_dir, git_model_id, url_base="https://github.com/deepfakes-models/faceswap-models/releases/download"):
        self.logger = logging.getLogger(
            __name__)  # pylint:disable=invalid-name
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.git_model_id = git_model_id
        self.url_base = url_base
        self.chunk_size = 1024  # Chunk size for downloading and unzipping
        self.retries = 6
        self.get()
        self.model_path = self._model_path

    @property
    def _model_full_name(self):
        """ Return the model full name from the filename(s) """
        common_prefix = os.path.commonprefix(self.model_filename)
        retval = os.path.splitext(common_prefix)[0]
        self.logger.trace(retval)
        return retval

    @property
    def _model_name(self):
        """ Return the model name from the model full name """
        retval = self._model_full_name[:self._model_full_name.rfind("_")]
        self.logger.trace(retval)
        return retval

    @property
    def _model_version(self):
        """ Return the model version from the model full name """
        retval = int(
            self._model_full_name[self._model_full_name.rfind("_") + 2:])
        self.logger.trace(retval)
        return retval

    @property
    def _model_path(self):
        """ Return the model path(s) in the cache folder """
        retval = [os.path.join(self.cache_dir, fname)
                  for fname in self.model_filename]
        retval = retval[0] if len(retval) == 1 else retval
        self.logger.trace(retval)
        return retval

    @property
    def _model_zip_path(self):
        """ Full path to downloaded zip file """
        retval = os.path.join(
            self.cache_dir, "{}.zip".format(self._model_full_name))
        self.logger.trace(retval)
        return retval

    @property
    def _model_exists(self):
        """ Check model(s) exist """
        if isinstance(self._model_path, list):
            retval = all(os.path.exists(pth) for pth in self._model_path)
        else:
            retval = os.path.exists(self._model_path)
        self.logger.trace(retval)
        return retval

    @property
    def _plugin_section(self):
        """ Get the plugin section from the config_dir """
        path = os.path.normpath(self.cache_dir)
        split = path.split(os.sep)
        retval = split[split.index("plugins") + 1]
        self.logger.trace(retval)
        return retval

    @property
    def _url_section(self):
        """ Return the section ID in github for this plugin type """
        sections = dict(extract=1, train=2, convert=3)
        retval = sections[self._plugin_section]
        self.logger.trace(retval)
        return retval

    @property
    def _url_download(self):
        """ Base URL for models """
        tag = "v{}.{}.{}".format(
            self._url_section, self.git_model_id, self._model_version)
        retval = "{}/{}/{}.zip".format(self.url_base,
                                       tag, self._model_full_name)
        self.logger.trace("Download url: %s", retval)
        return retval

    @property
    def _url_partial_size(self):
        """ Return how many bytes have already been downloaded """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        self.logger.trace(retval)
        return retval

    def get(self):
        """ Check the model exists, if not, download and unzip into location """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self._model_path)
            return
        self.download_model()
        self.unzip_model()
        os.remove(self._model_zip_path)

    def download_model(self):
        """ Download model zip to cache folder """
        self.logger.info("Downloading model: '%s' from: %s",
                         self._model_name, self._url_download)
        for attempt in range(self.retries):
            try:
                downloaded_size = self._url_partial_size
                req = urllib.request.Request(self._url_download)
                if downloaded_size != 0:
                    req.add_header(
                        "Range", "bytes={}-".format(downloaded_size))
                response = urllib.request.urlopen(req, timeout=10)
                self.logger.debug("header info: {%s}", response.info())
                self.logger.debug("Return Code: %s", response.getcode())
                self.write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urllib.error.HTTPError, urllib.error.URLError) as err:
                if attempt + 1 < self.retries:
                    self.logger.warning("Error downloading model (%s). Retrying %s of %s...",
                                        str(err), attempt + 2, self.retries)
                else:
                    self.logger.error("Failed to download model. Exiting. (Error: '%s', URL: "
                                      "'%s')", str(err), self._url_download)
                    self.logger.info(
                        "You can try running again to resume the download.")
                    self.logger.info("Alternatively, you can manually download the model from: %s "
                                     "and unzip the contents to: %s",
                                     self._url_download, self.cache_dir)
                    sys.exit(1)

    def write_zipfile(self, response, downloaded_size):
        """ Write the model zip file to disk """
        length = int(response.getheader("content-length")) + downloaded_size
        if length == downloaded_size:
            self.logger.info("Zip already exists. Skipping download")
            return
        write_type = "wb" if downloaded_size == 0 else "ab"
        with open(self._model_zip_path, write_type) as out_file:
            pbar = tqdm(desc="Downloading",
                        unit="B",
                        total=length,
                        unit_scale=True,
                        unit_divisor=1024)
            if downloaded_size != 0:
                pbar.update(downloaded_size)
            while True:
                buffer = response.read(self.chunk_size)
                if not buffer:
                    break
                pbar.update(len(buffer))
                out_file.write(buffer)
            pbar.close()

    def unzip_model(self):
        """ Unzip the model file to the cache folder """
        self.logger.info("Extracting: '%s'", self._model_name)
        try:
            zip_file = zipfile.ZipFile(self._model_zip_path, "r")
            self.write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            self.logger.error("Unable to extract model file: %s", str(err))
            sys.exit(1)

    def write_model(self, zip_file):
        """ Extract files from zip file and write, with progress bar """
        length = sum(f.file_size for f in zip_file.infolist())
        fnames = zip_file.namelist()
        self.logger.debug(
            "Zipfile: Filenames: %s, Total Size: %s", fnames, length)
        pbar = tqdm(desc="Decompressing",
                    unit="B",
                    total=length,
                    unit_scale=True,
                    unit_divisor=1024)
        for fname in fnames:
            out_fname = os.path.join(self.cache_dir, fname)
            self.logger.debug("Extracting from: '%s' to '%s'",
                              self._model_zip_path, out_fname)
            zipped = zip_file.open(fname)
            with open(out_fname, "wb") as out_file:
                while True:
                    buffer = zipped.read(self.chunk_size)
                    if not buffer:
                        break
                    pbar.update(len(buffer))
                    out_file.write(buffer)
        zip_file.close()
        pbar.close()
