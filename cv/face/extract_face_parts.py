#!/usr/bin/env python
# coding: utf-8

import cv2
import dlib
import argparse
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from collections import OrderedDict
from data_lib import face
from data_lib import Extract_Align
from pathlib2 import Path

front_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("./data_lib/dlib_model/mmod_human_face_detector.dat")
lmark_predictor = dlib.shape_predictor('./data_lib/dlib_model/shape_predictor_68_face_landmarks.dat')
align_extractor = Extract_Align.Align()

# Facial Landmark Partition
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17)),
    ("chin", (8, 11)),
    ("left_cheek", (12, 17)),
    ("right_cheek", (0, 5))
])

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def get_cv_img(img):
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
            raise TypeError("Input type is not supported! Only string, numpy array and PIL Image is supported here.")
    return img

def dlib_pipeline(img, show=True):
    img = get_cv_img(img)
    rect = front_face_detector(img)[0]
    lmark = face.shape_to_np(lmark_predictor(img, rect))
    box = (rect.left(), rect.top(), rect.right(), rect.bottom())
    return box, lmark

def crop(img, points, parts='left_eye', scale=[0.25,0.25]):
    eyes_list = []
    parts_points = points[FACIAL_LANDMARKS_IDXS[parts][0]:FACIAL_LANDMARKS_IDXS[parts][1],:]
    lp = np.min(parts_points[:, 0])
    rp = np.max(parts_points[:, 0])
    tp = np.min(parts_points[:, -1])
    bp = np.max(parts_points[:, -1])

    w = rp - lp
    h = bp - tp

    lp_ = int(np.maximum(0, np.minimum(lp - scale[0] * w, lp-h/4 )))
    rp_ = int(np.minimum(img.shape[1], np.maximum(rp + scale[0] * w, rp+h/4)))
    tp_ = int(np.maximum(0, np.minimum(tp - scale[1] * h, tp-w/4 )))
    bp_ = int(np.minimum(img.shape[0], np.maximum(bp + scale[1] * h, bp+w/4)))

    return img[tp_:bp_, lp_:rp_, :]

def crop_all(img):
    cropped={}
    img = get_cv_img(img)
    rect=dlib.rectangle(0,0,img.shape[1],img.shape[0])
    lmark = face.shape_to_np(lmark_predictor(img, rect))
    if lmark is None or len(lmark)==0:
        return None
    img_align, alignment = align_extractor.extract(img, lmark, 256, align_eyes=True)
    lmark_align = align_extractor.transform_points(lmark, alignment, 256, padding=48)
    cropped["left_eye"] = crop(img_align, lmark_align, "left_eye", [0.25, 1.75])
    cropped["right_eye"] = crop(img_align, lmark_align, "right_eye", [0.25, 1.75])
    cropped["nose"] = crop(img_align, lmark_align, "nose", [0.25, 0])
    cropped["lips"] = crop(img_align, lmark_align, "mouth", [0.1, 0.1])
#     cropped["left_cheek"] = crop(img_align, lmark_align, "left_cheek", [0, 0])
#     cropped["right_cheek"] = crop(img_align, lmark_align, "right_cheek", [0, 0])
    
    cropped["left_eye"] = cv2.resize(cropped["left_eye"], (96, 96), interpolation = cv2.INTER_AREA)
    cropped["right_eye"] = cv2.resize(cropped["right_eye"], (96, 96), interpolation = cv2.INTER_AREA)
    cropped["nose"] = cv2.resize(cropped["nose"], (96, 96), interpolation = cv2.INTER_AREA)
    cropped["lips"] = cv2.resize(cropped["lips"], (96, 96), interpolation = cv2.INTER_AREA)
#     cropped["left_cheek"] = cv2.resize(cropped["left_cheek"], (48, 96), interpolation = cv2.INTER_AREA)
#     cropped["right_cheek"] = cv2.resize(cropped["right_cheek"], (48, 96), interpolation = cv2.INTER_AREA)
#     cropped["cheek"] = np.hstack([cropped["left_cheek"], cropped["right_cheek"]])
    return lmark, cropped

face_root="/mnt/nfs/work1/trahman/aowal/deepfake-dataset/cropped_face_database"
parts_root = "/mnt/nfs/work1/trahman/zhongyangzha/deepfake-dataset/cropped_parts_database"
face_root=Path(face_root)

temp=[list(i.iterdir()) for i in list(face_root.iterdir())]
video_paths=list(flat(temp))

video_names=[]
frame_names=[]
landmarks=[]
for video_path in video_paths:
    print("==> Now processing: ", video_path)
    img_paths = list(video_path.rglob("*.png"))
    img_paths = sorted(img_paths, key=lambda x:int(x.stem.split('_')[-1]))
    parts_dir = Path(parts_root, *img_paths[0].parts[len(face_root.parts):-1])
    if not parts_dir.exists():
        new_paths_left  = {img:(parts_dir / (img.stem+"_left"+img.suffix)) for img in img_paths}
        new_paths_right = {img:(parts_dir / (img.stem+"_right"+img.suffix)) for img in img_paths}
        new_paths_nose  = {img:(parts_dir / (img.stem+"_nose"+img.suffix)) for img in img_paths}
        new_paths_lips  = {img:(parts_dir / (img.stem+"_lips"+img.suffix)) for img in img_paths}
    #     new_paths_cheek={img:Path(parts_root, *img.parts[len(face_root.parts):-1], img.stem+"_cheek"+img.suffix) for img in img_paths}
        for img_path in img_paths:
            try:
                lmark, cropped = crop_all(str(img_path))
                if lmark is not None:
                    parts_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(new_paths_left[img_path]), cv2.cvtColor(cropped["left_eye"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(new_paths_right[img_path]), cv2.cvtColor(cropped["right_eye"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(new_paths_nose[img_path]), cv2.cvtColor(cropped["nose"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(new_paths_lips[img_path]), cv2.cvtColor(cropped["lips"], cv2.COLOR_RGB2BGR))
        #             cv2.imwrite(str(new_paths_cheek[img_path]), cv2.cvtColor(cropped["cheek"], cv2.COLOR_RGB2BGR))
                    video_names.append(video_path.stem)
                    frame_names.append(img_path.name)
                    landmarks.append(str(lmark.tolist()))
            except:
                print("==> Video: ", str(video_path), "can not be cropped.")
                break

data = {'Video_Name':video_names, 'Frame_Name':frame_names, 'landmark':landmarks}
df = pd.DataFrame(data) 
df.to_csv(str(Path(parts_root)/'landmarks.csv'), index=False)
