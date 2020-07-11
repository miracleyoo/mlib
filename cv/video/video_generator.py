import cv2
import numpy as np

def gen_video(video_path, imgs, fps, width=None, height=None):
    """Generate a video from an image serie.
    Args:
        video_path: The output video path.
        imgs: The image serie used for generation.
        fps: Frame per second value of the output video.
        width: The frame width of the video.
        height: The frame height of the video.
    Returns:
        None
    """
    ext = video_path.split('.')[-1]
    if ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    elif ext == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #*'XVID')
    else:
        # if not .mp4 or avi, we force it to mp4
        video_path = video_path.replace(ext, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    if width is None or height is None:
        height, width= imgs[0].shape[:2]
    else:
        imgs_ = [cv2.resize(img, (width, height)) for img in imgs]
        imgs = imgs_

    out = cv2.VideoWriter(video_path, fourcc, fps, (np.int32(width), np.int32(height)))

    for image in imgs:
        out.write(np.uint8(image))  # Write out frame to video

    # Release everything if job is finished
    out.release()
    print('The output video is ' + video_path)