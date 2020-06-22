import cv2
import numpy as np
from PIL import Image


def load_cv_img(img):
    """Convert the input to an OpenCV image.
    Args:
        img: A string, a PIL.Image object, or an OpenCV image
    Returns:
        An OpenCV format, RGB channel ordered image
    """
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, np.ndarray):
        pass
    else:
        try:
            img = np.asarray(img)
        except:
            raise TypeError(
                "Input type is not supported! Only string, numpy array and PIL Image is supported here.")
    return img


def crop_by_box(img, face):
    """Crop a face from the whole image
    Args:
        img: The input image.
        face: Face bounding box[x1, y1, x2, y2].
    Returns:
        The cropped face image.
    """
    return img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]


def constrain_box(box, w, h):
    """Check and make sure the box is not exceeding the boundary.
    Args:
        box: Box (Left, Top, Right, Bottom) location list.
        w: The width of the image which the box lives on.
        h: The height of the image which the box lives on.
    Returns:
        The constrained box (Left, Top, Right, Bottom) location list.
    """
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(w, box[2])
    box[3] = min(h, box[3])
    return box


def bounding_box_rec2square(img, box, scale=1):
    """ Turn a rectangle bounding box into square shape.
        The center of the two bounding box is the same.
    Args:
        img: A ndarray format of image. loadeed using cv2.
        scale: The ratio of the new box to the (width+height)/2 of the original box.
    Returns:
        big_box: The generated square bounding box.
    """
    # In case the box is out of image
    box = constrain_box(box, img.shape[1], img.shape[0])
    big_box = []

    width = abs(box[0]-box[2])
    height = abs(box[1]-box[3])
    center = (int(box[0]+width/2), int(box[1]+height/2))
    square_len = int((width+height)/2)

    big_box.append(center[0] - scale*square_len//2)
    big_box.append(center[1] - scale*square_len//2)
    big_box.append(center[0] + scale*square_len//2)
    big_box.append(center[1] + scale*square_len//2)

    big_box = constrain_box(big_box, img.shape[1], img.shape[0])
    return big_box


def resize(img, dim):
    """Resize a image.
    Args:
        img: The input image
        dim: (width, height)
    """
    return cv2.resize(load_cv_img(img), dim, interpolation=cv2.INTER_AREA)
