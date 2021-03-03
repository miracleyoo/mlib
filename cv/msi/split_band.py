import cv2
import logging
import numpy as np
import os.path as op
from collections import namedtuple
from math import ceil

import mlib.file.path_func as pf
from mlib.file import io as mio
from mlib.cv import VideoReader, gen_video

Size = namedtuple("Size", ["height", "width"])

_wavelength = [[856, 872, 840, 824, 974],
               [744, 728, 712, 696, 600],
               [664, 680, 647, 632, 616],
               [910, 926, 894, 888, 958],
               [792, 808, 776, 760, 942]]

logger = logging.getLogger(__name__)

__all__ = ["MultiSpectralDecoder"]


class MultiSpectralDecoder():
    """ Split Raw HSI/MSI image/video into (C, H, W) shape images.
    Usage:
        Set path variable when init or later, then call split_image(for images)
        or split_frame_at_index, split_video(for vidwo). Lastly, .band() method
        can return a certain band/bands from the splitted image last time. 
    Args:
        path: A path string of an image/video.
        output_root: The output path string.
        row_band_num: The mosaic's row number.
        col_band_num: The mosaic's column number.
        wave_lengthes: Wavelengths matrix of a single mosaic.
    """
    def __init__(self, path=None, output_root=None, row_band_num=5, col_band_num=5, wave_lengthes=_wavelength):
        self.row_band_num = row_band_num
        self.col_band_num = col_band_num
        self.wave_lengthes = wave_lengthes

        self.splited = None
        self.video_handler = None
        self.img = None
        self.frame_size = None
        self._path = None
        self._output_root = None

        if path is not None:
            self.path = path
        self.output_root = output_root

    @property
    def path(self):
        return self._path

    @property
    def output_root(self):
        return self._output_root

    @output_root.setter
    def output_root(self, output_root):
        self._output_root = output_root

    @path.setter
    def path(self, path):
        self._path = path
        if pf.suffix(path) in mio.suffix_map["video"]:
            self.video_handler = mio.load(path)
            self.frame_size = self.video_handler.size
        elif pf.suffix(path) in mio.suffix_map["image"]:
            self.img = mio.load(path)
            self.frame_size = Size(*self.img.shape[:2])
        else:
            raise TypeError("Please set the path to an image or video file!")

    def band(self, index):
        """ Return a certain band's splitted image.

        Args:
            index -> int: The band of channel index. If the pixel is saved in
                (5x5) format in the image, band(11) means the band in (3, 2).
            index -> list/tuple: The coordinate of the target band. Like (3, 2)
        """
        if self.splited is not None:
            if isinstance(index, int):
                return self.splited[index//self.col_band_num][index % self.col_band_num]
            elif isinstance(index, list) or isinstance(index, tuple):
                return self.splited[index[0]][index[1]]
            else:
                raise TypeError("index should be int or tuple or list!")
        else:
            raise ValueError(
                "Please call split_image or split_video at first!")

    def _split_frame(self, frame):
        """ The actual function that performs the image splition.

        Args:
            frame: An input image. It should be loaded using cv2.
        """
        split_frames = []
        dim = len(frame.shape)
        for row in range(self.row_band_num):
            temp = []
            for col in range(self.col_band_num):
                row_idxs = np.arange(
                    row, self.frame_size.height, self.row_band_num)
                col_idxs = np.arange(
                    col, self.frame_size.width, self.col_band_num)
                if dim == 4:
                    band_frame = frame[:, row_idxs, :, :][:, :, col_idxs, :]
                else:
                    band_frame = frame[row_idxs, :][:, col_idxs]
                temp.append(band_frame)
            split_frames.append(temp)
        return split_frames

    def split_video(self, path=None, frames_chunk_size=1800, max_frames=2700):
        """ Split a video into videos with different bands.

        If the video frame number is larger than the max_frames, it will be automatically
        splited into different "parts", and the output name will be something like:
        632nm_band13_part1.avi This is due to the large file size of MSI videos and the
        system memory cannot hold the whole video in many cases.

        Args:
            path: The path of a multispectral video.
            frame_chunk_size: The frame number for each temperally splited video.
            max_frames: The maximum frame number for a video to avoid temperal splition.
        """
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("Please specify a path at first!")

        if self.output_root is None:
            output_root = pf.get_folder(
                op.join(*op.split(self.path)[:-1], pf.stem(self.path)))
        else:
            output_root = pf.get_folder(
                op.join(self.output_root, pf.stem(self.path)))

        frame_number = self.video_handler.frame_num
        if frame_number < max_frames:
            ranges = [range(frame_number)]
        else:
            ranges = [range(i*frames_chunk_size, min((i+1)*frames_chunk_size, frame_number))
                      for i in range(ceil(frame_number/frames_chunk_size))]

        logger.info(f"This video is divided into {len(ranges)} parts.")

        if len(ranges) > 1:
            for row in range(self.row_band_num):
                for col in range(self.col_band_num):
                    band_num = row*self.col_band_num+col
                    pf.get_folder(op.join(output_root,
                                          f"{self.wave_lengthes[row][col]}nm_band{band_num}"))

        split_temp = []
        for idx, range_item in enumerate(ranges):
            for frame_idx in range_item:
                frame = self.video_handler.read_frame_at_index(
                    frame_idx, complete=True)[0]
                split_temp.append(self._split_frame(frame))

            for row in range(self.row_band_num):
                for col in range(self.col_band_num):
                    temp_frames = [split_temp[i][row][col]
                                   for i in range(len(range_item))]
                    temp_frames = np.vstack(temp_frames)
                    band_num = row*self.col_band_num+col
                    if len(ranges) > 1:
                        output_path = op.join(
                            output_root, f"{self.wave_lengthes[row][col]}nm_band{band_num}", f"{idx}.avi")
                    else:
                        output_path = op.join(
                            output_root, f"{self.wave_lengthes[row][col]}nm_band{band_num}_part{idx}.avi")
                    gen_video(output_path, temp_frames, self.video_handler.fps)

    def split_frame_at_index(self, frame_idx=0, path=None, save=False):
        """ Split one frame at a certain index in the video.

        Args:
            frame_idx: The index of the frame you want to split.
            path: The path of a multispectral video.
            save: Whether you want to save the splitted frame to image files.
        """
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("Please specify a path at first!")

        assert isinstance(frame_idx, int)
        frame = self.video_handler.read_frame_at_index(
            frame_idx, complete=True)[0]
        self.splited = self._split_frame(frame)
        if save:
            self._save_splitted_images()

    def split_image(self, path=None, save=False):
        """ Split one msi image.

        Args:
            path: The path of a multispectral video.
            save: Whether you want to save the splitted frame to image files.
        """
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("Please specify a path at first!")

        self.splited = self._split_frame(self.img)

        if save:
            self._save_splitted_images()

    def _save_splitted_images(self):
        """ Save the splitted images to files.
        """
        if self.output_root is None:
            output_root = pf.get_folder(
                op.join(*op.split(self.path)[:-1], pf.stem(self.path)))
        else:
            output_root = pf.get_folder(
                op.join(self.output_root, pf.stem(self.path)))

        for row in range(self.row_band_num):
            for col in range(self.col_band_num):
                band_num = row*self.col_band_num+col
                mio.save(self.splited[row][col], op.join(
                    output_root, f"{self.wave_lengthes[row][col]}nm_band{band_num}.png"))
