"""
Library providing convenient classes and methods for saving and loading all types of files.
"""
import os
import sys
import cv2
import os.path as op

from . import path_func as pf
from ..lang.adv_import import install_if_not_exist
from ..cv.video.video_reader import VideoReader
from ..cv.video.video_generator import gen_video
from ..cv.image.image import cv2_read_img

__all__ = ["save", "load", "write", "read", "dump", "suffix_map"]

suffix_map = {
    "pickle": ["pkl", "p", "pickle", "pt"],
    "json":   ["json"],
    "yaml":   ["yaml", "yml"],
    "txt":    ["txt", "log", ""],
    "excel":  ["xlsx", "xls", "csv"],
    "mat":    ["mat"],
    "image":  ["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    "audio":  ["mp3", "wav", "aac", "flac"],
    "video":  ["mp4", "avi", "mov", "mkv", "flv", "webm"]
}


def is_simple_type(obj):
    return type(obj) in [str, int, float, bool]


def is_1dlist(obj):
    return ((isinstance(obj, list) or isinstance(obj, tuple))
            and all([is_simple_type(i) for i in obj]))


class IOCenter(object):
    ext = ""
    cls_type = ""
    woptions = ""
    roptions = ""

    @classmethod
    def process_path(cls, path):
        pf.get_folder(op.split(path)[0])
        if pf.suffix(path).strip(".") not in suffix_map[cls.cls_type]:
            path = f"{op.splitext(path)[0]}.{cls.ext}"
        return path

    @classmethod
    def save(cls, obj, path, woptions=None, **kwds):
        raise NotImplementedError()

    @classmethod
    def load(cls, path, roptions=None, **kwds):
        raise NotImplementedError()


class PickleController(IOCenter):
    ext = "pkl"
    cls_type = "pickle"
    woptions = "wb"
    roptions = "rb"

    @classmethod
    def save(cls, obj, path, woptions=None, **kwds):
        if woptions is None:
            woptions = cls.woptions

        path = cls.process_path(path)
        with open(path, woptions) as f:
            pkl.dump(obj, f, **kwds)

    @classmethod
    def load(cls, path, roptions=None, **kwds):
        if roptions is None:
            roptions = cls.roptions

        if op.exists(path):
            with open(path, roptions) as f:
                obj = pkl.load(f, **kwds)
            return obj
        else:
            raise FileNotFoundError(path)


class YAMLController(IOCenter):
    ext = "yaml"
    cls_type = "yaml"
    woptions = "w"
    roptions = "r"

    @classmethod
    def save(cls, obj, path, woptions=None, **kwds):
        if woptions is None:
            woptions = cls.woptions

        path = cls.process_path(path)
        with open(path, woptions) as f:
            yaml.dump(obj, f, **kwds)

    @classmethod
    def load(cls, path, roptions=None, **kwds):
        if roptions is None:
            roptions = cls.roptions

        if op.exists(path):
            with open(path, roptions) as f:
                obj = yaml.load(f, Loader=yaml.Loader, **kwds)
            return obj
        else:
            raise FileNotFoundError(path)


class JSONController(IOCenter):
    ext = "json"
    cls_type = "json"
    woptions = "w"
    roptions = "r"

    @classmethod
    def save(cls, obj, path, woptions=None, **kwds):
        if woptions is None:
            woptions = cls.woptions

        path = cls.process_path(path)
        with open(path, woptions) as f:
            json.dump(obj, f, **kwds)

    @classmethod
    def load(cls, path, roptions=None, **kwds):
        if roptions is None:
            roptions = cls.roptions

        if op.exists(path):
            with open(path, roptions) as f:
                obj = json.load(f, **kwds)
            return obj
        else:
            raise FileNotFoundError(path)


class TXTController(IOCenter):
    ext = "txt"
    cls_type = "txt"
    woptions = "w"
    roptions = "r"

    @classmethod
    def save(cls, obj, path, woptions=None, **kwds):
        if woptions is None:
            woptions = cls.woptions

        path = cls.process_path(path)
        with open(path, woptions) as f:
            if is_simple_type(obj):
                f.write(str(obj))
            elif is_1dlist(obj):
                f.write("\n".join(str(item) for item in obj))
            else:
                f.write(pprint.pformat(obj, **kwds))

    @classmethod
    def load(cls, path, roptions=None, **kwds):
        if roptions is None:
            roptions = cls.roptions

        if op.exists(path):
            with open(path, roptions) as f:
                obj = f.readlines()
            return obj
        else:
            raise FileNotFoundError(path)


class MATController(IOCenter):
    ext = "mat"
    cls_type = "mat"

    @classmethod
    def save(cls, obj, path, **kwds):
        path = cls.process_path(path)
        sio.savemat(path, obj, **kwds)

    @classmethod
    def load(cls, path, **kwds):
        if op.exists(path):
            return sio.loadmat(path, **kwds)
        else:
            raise FileNotFoundError(path)


class ExcelController(IOCenter):
    ext = "xlsx"
    cls_type = "excel"

    @classmethod
    def save(cls, obj, path, **kwds):
        suffix = pf.suffix(path).strip(".")
        path = cls.process_path(path)
        if suffix == "csv":
            assert hasattr(obj, "to_csv")
            obj.to_csv(path, **kwds)
        else:
            assert hasattr(obj, "to_excel")
            obj.to_excel(path, **kwds)

    @classmethod
    def load(cls, path, **kwds):
        suffix = pf.suffix(path).strip(".")
        if op.exists(path):
            if suffix == "csv":
                obj = pd.read_csv(path, **kwds)
            else:
                obj = pd.read_excel(path, **kwds)
            return obj
        else:
            raise FileNotFoundError(path)


class ImageController(IOCenter):
    """ Controller for image files.
    Args:
        via: The method you choose to load the file. 
            You can specify "cv2" or "pillow".

    """
    ext = "jpg"
    cls_type = "image"

    @classmethod
    def save(cls, obj, path, via="cv2", cvt_rgb=True, **kwds):
        path = cls.process_path(path)
        if via == "cv2":
            if cvt_rgb and len(obj.shape)>=3:
                obj = cv2.cvtColor(obj, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, obj, **kwds)
        elif via == "pillow":
            assert hasattr(obj, "save")
            obj.save(path, **kwds)
        else:
            raise ValueError("Wrong via type! Please choose cv2 or pillow.")

    @classmethod
    def load(cls, path, via="cv2", **kwds):
        if op.exists(path):
            if via == "cv2":
                return cv2_read_img(path, **kwds)
            elif via == "pillow":
                return Image.open(path, **kwds)
            else:
                raise ValueError(
                    "Wrong via type! Please choose cv2 or pillow.")

        else:
            raise FileNotFoundError(path)


class VideoController(IOCenter):
    ext = "mp4"
    cls_type = "video"

    @classmethod
    def save(cls, obj, path, **kwds):
        if "fps" not in kwds.keys():
            kwds["fps"] = 25
        path = cls.process_path(path)
        gen_video(path, obj, **kwds)

    @classmethod
    def load(cls, path, **kwds):
        if op.exists(path):
            return VideoReader(path, **kwds)
        else:
            raise FileNotFoundError(path)


class AudioController(IOCenter):
    ext = "wav"
    cls_type = "audio"

    @classmethod
    def save(cls, obj, path, **kwds):
        """ Write (normalized) audio files.

        Save audio data provided as an array of shape [channels, samples] to a WAV, FLAC, or OGG file. 
        channels can be up to 65535 for WAV, 255 for OGG, and 8 for FLAC. For monaural audio the array
        can be one-dimensional.
        It uses soundfile to write the audio files.
        Args:
            file (str or int or file-like object): file name of output audio file. The format (WAV, FLAC, OGG) will be inferred from the file name
            signal (numpy.ndarray): audio data to write
            sampling_rate (int): sample rate of the audio data
            precision (str, optional): precision of writen file, can be ‘16bit’, ‘24bit’, ‘32bit’. Only available for WAV files. Default: 16bit
            normalize (bool, optional): normalize audio data before writing. Default: False
            kwargs: pass on further arguments to soundfile.write()
        """
        path = cls.process_path(path)
        if 'rate' not in locals():
            rate = 16000
        af.write(path, signal=obj, sampling_rate=rate, **kwds)

    @classmethod
    def load(cls, path, **kwds):
        """ Read audio file.

        It uses soundfile for WAV, FLAC, and OGG files. All other audio files are first converted to WAV by sox or ffmpeg.
        Args:
            file (str or int or file-like object): file name of input audio file
            duration (float, optional): return only a specified duration in seconds. Default: None
            offset (float, optional): start reading at offset in seconds. Default: 0
            always_2d (bool, optional): if True it always returns a two-dimensional signal even for mono sound files. Default: False
            kwargs – pass on further arguments to soundfile.read()
        Returns:
            numpy.ndarray: a two-dimensional array in the form [channels, samples]. If the sound file has only one channel, a one-dimensional array is returned
            int: sample rate of the audio file
        """
        if op.exists(path):
            return af.read(path, **kwds)
        else:
            raise FileNotFoundError(path)


def selector(path, cls_type):
    if cls_type is None:
        suffix = pf.suffix(path).strip(".").lower()
        for suf in suffix_map.keys():
            if suffix in suffix_map[suf]:
                cls_type = suf

    if cls_type == "pickle":
        install_if_not_exist(package_name="pickle",
                             imported_name="pkl", scope=globals())
        cont = PickleController()
    elif cls_type == "json":
        install_if_not_exist(package_name="json", scope=globals())
        cont = JSONController()
    elif cls_type == "yaml":
        install_if_not_exist(package_name="yaml", scope=globals())
        cont = YAMLController()
    elif cls_type == "txt":
        install_if_not_exist(package_name="pprint", scope=globals())
        cont = TXTController()
    elif cls_type == "mat":
        install_if_not_exist(package_name="scipy",
                             import_name="scipy.io", imported_name="sio", scope=globals())
        cont = MATController()
    elif cls_type == "excel":
        install_if_not_exist(package_name="pandas",
                             imported_name="pd", scope=globals())
        cont = ExcelController()
    elif cls_type == "audio":
        install_if_not_exist(package_name="audiofile",
                             imported_name="af", scope=globals())
        cont = AudioController()
    elif cls_type == "image":
        install_if_not_exist(package_name="pillow",
                             import_name="PIL.Image", imported_name="Image", scope=globals())

        cont = ImageController()
    elif cls_type == "video":
        cont = VideoController()
    else:
        raise TypeError(
            "Please input a file with a supported filetype, or specify a valid cls_type")

    return cont


def save(obj, path, woptions=None, cls_type=None, **kwds):
    cont = selector(path, cls_type)
    if woptions is not None:
        cont.save(obj, path, woptions=woptions, **kwds)
    else:
        cont.save(obj, path, **kwds)


def load(path, roptions=None, cls_type=None, **kwds):
    cont = selector(path, cls_type)
    if roptions is not None:
        return cont.load(path, roptions=roptions, **kwds)
    else:
        return cont.load(path, **kwds)


write = save
dump = save
read = load
