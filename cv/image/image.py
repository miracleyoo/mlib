import cv2
import logging
import numpy as np
from hashlib import sha1

__all__ = ["get_cv_img", "cv2_read_img", "crop_by_box", "constrain_box",
           "bounding_box_rec2square", "resize", "hash_encode_image", "hash_image_file"]


def get_cv_img(img, cvt_rgb=True):
    """Convert the input to an OpenCV image.
    Args:
        img: A string, a PIL.Image object, or an OpenCV image
    Returns:
        An OpenCV format, RGB channel ordered image
    """
    if isinstance(img, str):
        img = cv2_read_img(img, cvt_rgb=True)
    elif isinstance(img, np.ndarray):
        pass
    else:
        try:
            img = np.asarray(img)
        except:
            raise TypeError(
                "Input type is not supported! Only string, numpy array and PIL Image is supported here.")
    return img


def cv2_read_img(filename, raise_error=False, cvt_rgb=True):
    """ Read an image with cv2 and check that an image was actually loaded.
        Logs an error if the image returned is None. or an error has occured.
        Pass raise_error=True if error should be raised """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("Requested image: '{filename}'")
    success = True
    image = None
    try:
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        if cvt_rgb and len(image.shape) >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError
    except TypeError:
        success = False
        msg = "Error while reading image (TypeError): '{}'".format(filename)
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except ValueError:
        success = False
        msg = ("Error while reading image. This is most likely caused by special characters in "
               "the filename: '{}'".format(filename))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except Exception as err:  # pylint:disable=broad-except
        success = False
        msg = "Failed to load image '{}'. Original Error: {}".format(
            filename, str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    logger.debug("Loaded image: '%s'. Success: %s", filename, success)
    return image


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
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def hash_image_file(filename):
    """ Return an image file's sha1 hash """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    img = cv2_read_img(filename, raise_error=True)
    img_hash = sha1(img).hexdigest()
    logger.debug("filename: '%s', hash: %s", filename, img_hash)
    return img_hash


def hash_encode_image(image, extension):
    """ Encode the image, get the hash and return the hash with
        encoded image """
    img = cv2.imencode(extension, image)[
        1]  # pylint:disable=no-member,c-extension-no-member
    f_hash = sha1(
        cv2.imdecode(  # pylint:disable=no-member,c-extension-no-member
            img,
            cv2.IMREAD_UNCHANGED)).hexdigest()  # pylint:disable=no-member,c-extension-no-member
    return f_hash, img
