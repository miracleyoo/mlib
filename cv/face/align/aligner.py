import numpy

from .umeyama import umeyama
from .align_eyes import align_eyes
from numpy.linalg import inv
import cv2

from .landmark import FACIAL_LANDMARKS_2D

def get_align_mat(lmark, size, should_align_eyes):
    if type(lmark) != numpy.ndarray:
        lmark = numpy.array(lmark)
    mat_umeyama = umeyama(lmark[17:], FACIAL_LANDMARKS_2D, True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = numpy.matrix(lmark)

    # cv2 expects points to be in the form np.array([ [[x1, y1]], [[x2, y2]], ... ]), we'll expand the dim
    landmarks = numpy.expand_dims(landmarks, axis=1)

    # Align the landmarks using umeyama
    umeyama_landmarks = cv2.transform(landmarks, mat_umeyama, landmarks.shape)

    # Determine a rotation matrix to align eyes horizontally
    mat_align_eyes = align_eyes(umeyama_landmarks, size)

    # Extend the 2x3 transform matrices to 3x3 so we can multiply them
    # and combine them as one
    mat_umeyama = numpy.matrix(mat_umeyama)
    mat_umeyama.resize((3, 3))
    mat_align_eyes = numpy.matrix(mat_align_eyes)
    mat_align_eyes.resize((3, 3))
    mat_umeyama[2] = mat_align_eyes[2] = [0, 0, 1]

    # Combine the umeyama transform with the extra rotation matrix
    transform_mat = mat_align_eyes * mat_umeyama

    # Remove the extra row added, shape needs to be 2x3
    transform_mat = numpy.delete(transform_mat, 2, 0)
    transform_mat = transform_mat / size
    return transform_mat
