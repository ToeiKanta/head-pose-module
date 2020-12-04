#
#   Headpose Detection
#   Modified by Qhan
#   Last Update: 2019.1.9
#

import argparse
import cv2
import dlib
import numpy as np
import os
import os.path as osp

from timer import Timer
from utils import Annotator
from face_detection import RetinaFace


t = Timer()

class HeadposeDetection():

    # 3D facial model coordinates
    landmarks_3d_list = [
        np.array([
            [ 0.000,  0.000,   0.000],    # Nose tip
            [ 0.000, -8.250,  -1.625],    # Chin
            [-5.625,  4.250,  -3.375],    # Left eye left corner
            [ 5.625,  4.250,  -3.375],    # Right eye right corner
            [-3.750, -3.750,  -3.125],    # Left Mouth corner
            [ 3.750, -3.750,  -3.125]     # Right mouth corner 
        ], dtype=np.double),
        np.array([
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
            [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
            [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
            [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
            [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
            [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
            [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
            [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
            [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
            [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
        ], dtype=np.double),
        np.array([
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
        ], dtype=np.double),
        np.array([
            [ 0.000,  0.000,   0.000],    # Nose tip
            [-4.800,  4.250,  -3.375],    # Left eye left corner [-5.625,  4.250,  -3.375]
            [ 4.800,  4.250,  -3.375],    # Right eye right corner [ 5.625,  4.250,  -3.375]
            [-3.750, -3.750,  -3.125],    # Left Mouth corner
            [ 3.750, -3.750,  -3.125]     # Right mouth corner 
        ], dtype=np.double)
    ]

    # 2d facial landmark list
    lm_2d_index_list = [
        [30, 8, 36, 45, 48, 54],
        [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
        [33, 36, 39, 42, 45], # 5 points
        [30, 36, 45, 48, 54], # 5 point retina
    ]

    def __init__(self, lm_type=1, predictor="model/shape_predictor_68_face_landmarks.dat", verbose=False):
        # self.bbox_detector = RetinaFace(gpu_id=0)
        self.landmark_predictor = dlib.shape_predictor(predictor)

        self.lm_2d_index = self.lm_2d_index_list[lm_type]
        self.landmarks_3d = self.landmarks_3d_list[lm_type]

        self.v = verbose


    def to_numpy(self, landmarks):
        coords = []
        for i in self.lm_2d_index:
            coords += [[landmarks.part(i).x, landmarks.part(i).y]]
        return np.array(coords).astype(np.int)

    def get_landmarks(self, im, box):
        # bbox_detector = RetinaFace(gpu_id=0)

        # Detect bounding boxes of faces
        t.tic('bb')
        
        # faces = bbox_detector(im) if im is not None else []
        if self.v: 
            print(', bb: %.2f' % t.toc('bb'), end='ms')

        # Detect landmark of first face
        t.tic('lm')
        face_box = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
        face_box = dlib.rectangle(left=box[0], top=box[1], right=box[2], bottom=box[3])
        # face_box = np.ar///ray(face_box).astype(int)
        # print(f'type(face_box)//// {type(face_box)}')
        landmarks_2d = self.landmark_predictor(im, face_box)

        # Choose specific landmarks corresponding to 3D facial model
        landmarks_2d = self.to_numpy(landmarks_2d)
        if self.v: 
            print(', lm: %.2f' % t.toc('lm'), end='ms')
            
        rect = [box[0], box[1], box[2], box[3]]

        return landmarks_2d.astype(np.double), rect



    def get_headpose(self, im, landmarks_2d, verbose=False):
        h, w, c = im.shape
        f = w # column size = x axis length (focal length)
        u0, v0 = w / 2, h / 2 # center of image plane
        camera_matrix = np.array(
            [[f, 0, u0],
             [0, f, v0],
             [0, 0, 1]], dtype = np.double
         )
         
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1)) 

        # Find rotation, translation
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)
        
        if verbose:
            print("Camera Matrix:\n {0}".format(camera_matrix))
            print("Distortion Coefficients:\n {0}".format(dist_coeffs))
            print("Rotation Vector:\n {0}".format(rotation_vector))
            print("Translation Vector:\n {0}".format(translation_vector))

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs


    # rotation vector to euler angles
    def get_angles(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec)) # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return [rx, ry, rz]

    # moving average history
    history = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}
    
    def add_history(self, values):
        for (key, value) in zip(self.history, values):
            self.history[key] += [value]
            
    def pop_history(self):
        for key in self.history:
            self.history[key].pop(0)
            
    def get_history_len(self):
        return len(self.history['lm'])
            
    def get_ma(self):
        res = []
        for key in self.history:
            res += [np.mean(self.history[key], axis=0)]
        return res

    # return image and angles
    def process_image(self, im, box, draw=True, ma=3, history = None, landmarks = None):
        if history == None:
            self.history = history
        # landmark Detection
        # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if landmarks is None:
            landmarks_2d, bbox = self.get_landmarks(im,box)
        else:
            self.lm_2d_index = self.lm_2d_index_list[3]
            self.landmarks_3d = self.landmarks_3d_list[3]
            landmarks_2d, bbox_ = self.get_landmarks(im[:2,:2],box)
            # landmarks_2d = [[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]]
            
            # print("=======\n")
            # print(f'{landmarks_2d} {landmarks_2d_2}')
            # print("\n=\n")
            # print(f'{bbox_2} {box}')
            # print("=======\n")
            landmarks_2d[0][0] = landmarks[2][0]
            landmarks_2d[0][1] = landmarks[2][1]
            landmarks_2d[1][0] = landmarks[0][0]
            landmarks_2d[1][1] = landmarks[0][1]
            landmarks_2d[2][0] = landmarks[1][0]
            landmarks_2d[2][1] = landmarks[1][1]
            landmarks_2d[3][0] = landmarks[3][0]
            landmarks_2d[3][1] = landmarks[3][1]
            landmarks_2d[4][0] = landmarks[4][0]
            landmarks_2d[4][1] = landmarks[4][1]
            bbox = box
            # input()
        
        # if no face deteced, return original image
        if landmarks_2d is None:
            return im, None

        # Headpose Detection
        t.tic('hp')
        rvec, tvec, cm, dc = self.get_headpose(im, landmarks_2d)
        if self.v: 
            print(', hp: %.2f' % t.toc('hp'), end='ms')
            
        if ma > 1:
            self.add_history([landmarks_2d, bbox, rvec, tvec, cm, dc])
            if self.get_history_len() > ma:
                self.pop_history()
            landmarks_2d, bbox, rvec, tvec, cm, dc = self.get_ma()

        t.tic('ga')
        angles = self.get_angles(rvec, tvec)
        if self.v: 
            print(', ga: %.2f' % t.toc('ga'), end='ms')
        if draw:
            t.tic('draw')
            annotator = Annotator(im, angles, bbox, landmarks_2d, rvec, tvec, cm, dc, b=10.0)
            im = annotator.draw_all()
            if self.v: 
                print(', draw: %.2f' % t.toc('draw'), end='ms' + ' ' * 10)
        return im, angles, self.history


def main(args):
    in_dir = args["input_dir"]
    out_dir = args["output_dir"]

    # Initialize head pose detection
    hpd = HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

    for filename in os.listdir(in_dir):
        name, ext = osp.splitext(filename)
        if ext in ['.jpg', '.png', '.gif']: 
            print("> image:", filename, end='')
            image = cv2.imread(in_dir + filename)
            res, angles = hpd.process_image(image)
            cv2.imwrite(out_dir + name + '_out.png', res)
        else:
            print("> skip:", filename, end='')
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='DIR', dest='input_dir', default='images/')
    parser.add_argument('-o', metavar='DIR', dest='output_dir', default='res/')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())

    if not osp.exists(args["output_dir"]): os.mkdir(args["output_dir"])
    if args["output_dir"][-1] != '/': args["output_dir"] += '/'
    if args["input_dir"][-1] != '/': args["input_dir"] += '/'
    main(args)
