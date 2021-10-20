import numpy as np
import os.path as op
import cv2
import time
import argparse
from pathlib2 import Path

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),

    "d346":(346, 260) # DAVIS DVS Camera
}

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    '.avi': cv2.VideoWriter_fourcc(*'XVID'),
    # '.mp4': cv2.VideoWriter_fourcc(*'h264'),
    '.mp4': cv2.VideoWriter_fourcc(*'mp4v'),
}

class VideoRecorder():
    """ Record a video using certain parameters, and output a timestamps log.
    Args:
        out_path: The path to output the video.
        fps: Frame per second.
        width: The video frame width.
        height: The video frame height.
        res: Standard resolutions.
        timelog: Whether to keep a timestamps log.
    """
    def __init__(self, out_path=None, fps=30, width=1280, height=720, res=None, timelog=True) -> None:
        assert res in ['240p', '360p', '480p', '720p', '1080p', '2k', '4k', 'd346', None]
        self.out_path = out_path if out_path is not None else 'video.avi'
        self.out_path = get_new_path(self.out_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.res = res
        self.timelog = timelog
        self.cap = None
        self.out_handler = None
        self.video_type = self.get_video_type(self.out_path)
        self.get_log_path()
    
    def start(self, cam_idx=0):
        """ Start a video stream using device at `cam_idx`.
        """
        self.cap = cv2.VideoCapture(cam_idx)
        self.set_dims()
        real_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_handler = cv2.VideoWriter(self.out_path, self.video_type, self.fps, real_size)
        if self.timelog:
            logger = open(self.log_path, 'w+')
        while True:
            ret, frame = self.cap.read()
            print(frame.shape, self.width, self.height)
            if self.timelog:
                logger.write(str(time.time())+'\n')
            out_handler.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        out_handler.release()
        cv2.destroyAllWindows()
        if self.timelog:
            logger.close()

    # grab resolution dimensions and set video capture to it.
    def set_dims(self):
        if self.res is not None:
            self.width, self.height = STD_DIMENSIONS[self.res]
        # change the current caputre device to the resulting resolution
        if self.cap is not None:
            self.change_res(self.cap, self.width, self.height)

    def get_log_path(self):
        root, name = op.split(self.out_path)
        name = op.splitext(name)[0]
        self.log_path = op.join(root, name+'_timestamps.txt')

    # Set resolution for the video capture
    # Function adapted from https://kirr.co/0l6qmh
    @staticmethod
    def change_res(cap, width, height):
        cap.set(3, width)
        cap.set(4, height)
    
    @staticmethod
    def get_video_type(path):
        _, ext = op.splitext(path)
        if ext in VIDEO_TYPE:
            return VIDEO_TYPE[ext]
        return VIDEO_TYPE['.avi']

    @staticmethod
    def returnCameraIndexes():
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        print(f"Cameras at the following indexes are valid: {arr}")
        return arr

def get_new_path(path):
    """ Return a path to a file, creat its parent folder if it doesn't exist, creat new one if existing.

    If the folder/file already exists, this function will use `path`_`idx` as new name, and make
    corresponding folder if `path` is a folder.
    idx will starts from 1 and keeps +1 until find a valid name without occupation.

    If the folder and its parent folders don't exist, keeps making these series of folders.

    Args:
        path: The path of a file/folder.
    Returns:
        _ : The guaranteed new path of the folder/file.
    """
    path = Path(path)
    root = Path(*path.parts[:-1])

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        new_path = path
        if new_path.suffix == '':
            new_path.mkdir()
    else:
        idx = 1
        while True:
            stem = path.stem+"_"+str(idx)
            new_path = root / (stem+path.suffix)
            if not new_path.exists():
                if new_path.suffix == '':
                    new_path.mkdir()
                break
            idx += 1
    return str(new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', type=str, help='Standard resolution string.')
    parser.add_argument('-o', '--out_path', type=str, default='records/video.avi', help='The output video path.')
    
    parser.add_argument('-c', '--cam', type=int, help='Camera index')
    parser.add_argument('-f', '--fps', type=int, default=30, help='FPS.')
    parser.add_argument('-n', '--no_timelog', action='store_true', help='Do not keep time log.')
    
    args = parser.parse_args()
    rec = VideoRecorder(out_path=args.out_path, res=args.resolution, fps=args.fps, timelog=(not args.no_timelog))
    
    if args.cam is None:
        arr = rec.returnCameraIndexes()
        rec.start(arr[0])
    else:
        rec.start(args.cam)