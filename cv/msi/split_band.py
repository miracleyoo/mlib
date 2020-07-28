import cv2
import logging
import numpy as np
import os.path as op
from collections import namedtuple
from math import ceil

import mlib.file.io as mio
import mlib.file.path_func as pf
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
    def __init__(self, path=None, output_root=None, row_band_num=5, col_band_num=5):
        self.row_band_num = row_band_num
        self.col_band_num = col_band_num
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

    def split_video(self, path=None, frames_chunk_size=3600, max_frames=4500):
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
                    gen_video(
                        op.join(output_root, f"{_wavelength[row][col]}nm_band{band_num}_part{idx}.avi"), temp_frames, self.video_handler.fps)

    def split_video_frame_at_index(self, frame_idx=0, path="", save=False):
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

    def split_image(self, path="", save=False):
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("Please specify a path at first!")

        self.splited = self._split_frame(self.img)

        if save:
            self._save_splitted_images()

    def _save_splitted_images(self):
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
                    output_root, f"{_wavelength[row][col]}nm_band{band_num}.png"))
