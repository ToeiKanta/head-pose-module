import numpy as np
import cv2
import dlib
import os
import pickle

def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor(relative('../facialLandmarks/shape_predictor_68_face_landmarks.dat'))
model = dlib.face_recognition_model_v1(relative('./dlib_face_recognition_resnet_model_v1.dat'))

class Recog_feature_extract_module:
    def __init__(self):
        self.FACE_DESC = []
        self.FACE_NAME = []

    def train(self,name,cropped,shape):
        dets = detector(cropped, 1)
        for k, d in enumerate(dets):
            # shape = sp(cropped, d)
            face_desc = model.compute_face_descriptor(cropped, shape, 200)
            self.FACE_DESC.append(face_desc)
            self.FACE_NAME.append(name)
        print("ADD " + name + "'s face done..")
        pickle.dump((self.FACE_DESC, self.FACE_NAME), open(relative('./tempmodel/trainset.pk'), 'ab'))
