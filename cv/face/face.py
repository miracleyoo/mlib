import cv2
import numpy as np
from PIL import Image


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
    

def cut_head(imgs, point):
    h, w = imgs[0].shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 8
    delta_y = (y2 - y1) / 5
    delta_x = np.random.randint(delta_x)
    delta_y = np.random.randint(delta_y)
    x1_ = np.int(np.maximum(0, x1 - delta_x))
    x2_ = np.int(np.minimum(w-1, x2 + delta_x))
    y1_ = np.int(np.maximum(0, y1 - delta_y))
    y2_ = np.int(np.minimum(h-1, y2 + delta_y * 0.5))
    imgs_new = []
    for i, im in enumerate(imgs):
        im = im[y1_:y2_, x1_:x2_, :]
        imgs_new.append(im)
    locs = [x1_, y1_, x2_, y2_]
    return imgs_new, locs


def crop_eye(img, points):
    eyes_list = []

    left_eye = points[36:42, :]
    right_eye = points[42:48, :]

    eyes = [left_eye, right_eye]
    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])

        w = rp - lp
        h = bp - tp

        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(img.shape[1], rp + 0.25 * w))
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(img.shape[0], bp + 1.75 * h))

        eyes_list.append(img[tp_:bp_, lp_:rp_, :])
    return eyes_list
