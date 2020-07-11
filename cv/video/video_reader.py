import os
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, path=None, insets=(0, 0)):
        """Creates a new VideoReader.

        Arguments:
            insets: amount to inset the image by, as a percentage of 
                (width, height). This lets you "zoom in" to an image 
                to remove unimportant content around the borders. 
                Useful for face detection, which may not work if the 
                faces are too small.
        """
        self.insets = insets
        self.path = path

    def set_path(self, path):
        self.path = path

    def read_metadata(self):
        """Extract the necessary information from a video file.

        Returns:
            frame_num: The total frame number of the video.
            fps: The `frame per second` value of the video.
            width: The frame width of the video.
            height: The frame height of the video.
        """
        vidcap = cv2.VideoCapture(self.path)
        frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        vidcap.release()
        return fps, frame_num, width, height

    def read_frames(self, num_frames, jitter=0, seed=None):
        """Reads frames that are always evenly spaced throughout the video.

        Arguments:
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
            jitter: if not 0, adds small random offsets to the frame indices;
                this is useful so we don't always land on even or odd frames
            seed: random seed for jittering; if you set this to a fixed value,
                you probably want to set it only on the first video 
        """
        assert num_frames > 0

        capture = cv2.VideoCapture(self.path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = np.linspace(
            0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        logger.debug(f"Frame indexes: {frame_idxs}")
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter,
                                               jitter, len(frame_idxs))
            frame_idxs = np.clip(
                frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(self.path, capture, frame_idxs)
        capture.release()
        return result

    def read_all_frames(self):
        capture = cv2.VideoCapture(self.path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        frame_idxs = np.arange(frame_count)
        result = self._read_frames_at_indices(self.path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, num_frames, seed=None):
        """Picks the frame indices at random.

        Arguments:
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
        """
        assert num_frames > 0
        np.random.seed(seed)

        capture = cv2.VideoCapture(self.path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = sorted(np.random.choice(
            np.arange(0, frame_count), num_frames))
        result = self._read_frames_at_indices(self.path, capture, frame_idxs)

        capture.release()
        return result

    def read_frames_at_indices(self, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(self.path)
        result = self._read_frames_at_indices(self.path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    logger.error(
                        "Error grabbing frame %d from movie %s" % (frame_idx, path))
                    break

                # Need to look at this frame?
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    logger.debug(f"Retriving frame: {frame_idx}")
                    if not ret or frame is None:
                        logger.error(
                            "Error retrieving frame %d from movie %s" % (frame_idx, path))
                        break

                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read

            logger.info("No frames read from movie %s" % path)
            return None
        except:
            logger.exception("Exception while reading movie %s" % path)
            return None

    def read_middle_frame(self):
        """Reads the frame from the middle of the video."""
        capture = cv2.VideoCapture(self.path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        result = self._read_frame_at_index(
            self.path, capture, frame_count // 2)
        capture.release()
        return result

    def read_frame_at_index(self, frame_idx):
        """Reads a single frame from a video.

        If you just want to read a single frame from the video, this is more
        efficient than scanning through the video to find the frame. However,
        for reading multiple frames it's not efficient.

        My guess is that a "streaming" approach is more efficient than a 
        "random access" approach because, unless you happen to grab a keyframe, 
        the decoder still needs to read all the previous frames in order to 
        reconstruct the one you're asking for.

        Returns a NumPy array of shape (1, H, W, 3) and the index of the frame,
        or None if reading failed.
        """
        capture = cv2.VideoCapture(self.path)
        result = self._read_frame_at_index(self.path, capture, frame_idx)
        capture.release()
        return result

    def _read_frame_at_index(self, path, capture, frame_idx):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            logger.error("Error retrieving frame %d from movie %s" %
                         (frame_idx, path))
            return None
        else:
            frame = self._postprocess_frame(frame)
            return np.expand_dims(frame, axis=0), [frame_idx]

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]

        if self.insets[1] > 0:
            H = frame.shape[1]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]

        return frame
