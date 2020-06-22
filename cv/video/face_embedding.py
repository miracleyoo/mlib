import sys, os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from face.det_retinaface import api as Retina
from mlib import read_video
from mlib.basic import *
from mlib.bounding_box import *
from mlib import image as image_handler


class Video_Processing():
    def __init__(self):
        self.retina = Retina.RetinaFaceDetector()
        self.video_handler = read_video.VideoReader()
        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def gen_video_embedding(self, video_path, threshold=0.85):
        faces = []
        frames=self.video_handler.read_all_frames(video_path)[0][:150]
        for frame in frames:
            res = self.retina.detect(frame, no_lmark=False)
            if len(res) >0 and res[0][4]>threshold:
                box = res[0][:4]
                box = bounding_box_rec2square(frame, box)
                ret = image_handler.crop_box(frame, box)
                faces.append(np.rollaxis(image_handler.resize(ret, (160, 160)), 2, 0))
            else:
                faces.append(np.zeros[3, 160, 160])
        embedding = self.resnet(torch.Tensor(faces))
        return embedding